# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict
from concurrent import futures as futures
from os import path as osp
from pathlib import Path

import mmcv
import numpy as np
from PIL import Image
from skimage import io


def get_image_index_str(img_idx, use_prefix_id=False):
    if use_prefix_id:
        return '{}'.format(img_idx)
    else:
        return '{}'.format(img_idx)


def get_spa_info_path(idx,
                        prefix,
                        info_type='image_2',
                        file_tail='.png',
                        training=True,
                        relative_path=True,
                        exist_check=True,
                        use_prefix_id=False):
    img_idx_str = get_image_index_str(idx, use_prefix_id)
    img_idx_str += file_tail
    if info_type.split("/")[0] in ["cam_img", "calib", "label"]:
        prefix = Path(prefix, info_type)
    else:
        prefix = Path(prefix)
    if training:
        file_path = prefix / img_idx_str
    else:
        file_path = prefix / img_idx_str
    if exist_check and not (file_path).exists():
        raise ValueError('file not exist: {}'.format(file_path))
    if relative_path:
        return str(file_path)
    else:
        return str(file_path)


def get_image_path(idx,
                   prefix,
                   info_type=None,
                   training=True,
                   relative_path=True,
                   exist_check=True,
                   file_tail='.png',
                   use_prefix_id=False):
    return get_spa_info_path(idx, prefix, info_type, file_tail, training,
                               relative_path, exist_check, use_prefix_id)


def get_label_path(idx,
                   prefix,
                   info_type='label',
                   training=True,
                   relative_path=True,
                   exist_check=True,
                   use_prefix_id=False):
    return get_spa_info_path(idx, prefix, info_type, '.txt', training,
                               relative_path, exist_check, use_prefix_id)


def get_plane_path(idx,
                   prefix,
                   training=True,
                   relative_path=True,
                   exist_check=True,
                   info_type='planes',
                   use_prefix_id=False):
    return get_spa_info_path(idx, prefix, info_type, '.txt', training,
                               relative_path, exist_check, use_prefix_id)


def get_velodyne_path(idx,
                      prefix,
                      training=True,
                      relative_path=True,
                      exist_check=True,
                      use_prefix_id=False):
    return get_spa_info_path(idx, prefix, 'velo/data', '.bin', training,
                               relative_path, exist_check, use_prefix_id)


def get_calib_path(idx,
                   prefix,
                   info_type='calib/',
                   training=True,
                   relative_path=True,
                   exist_check=True,
                   use_prefix_id=False):
    return get_spa_info_path(idx, prefix, 'calib/', '.txt', training,
                               relative_path, exist_check, use_prefix_id)


def get_pose_path(idx,
                  prefix,
                  training=True,
                  relative_path=True,
                  exist_check=True,
                  use_prefix_id=False):
    return get_spa_info_path(idx, prefix, 'pose', '.txt', training,
                               relative_path, exist_check, use_prefix_id)


def get_timestamp_path(idx,
                       prefix,
                       training=True,
                       relative_path=True,
                       exist_check=True,
                       use_prefix_id=False):
    return get_spa_info_path(idx, prefix, 'timestamp', '.txt', training,
                               relative_path, exist_check, use_prefix_id)


def get_label_anno(label_path):
    annotations = {}
    annotations.update({
        'name': [],
        'track_id': [],
        'truncated': [],
        'occluded': [],
        'alpha': [],
        'bbox': [],
        'dimensions': [],
        'location': [],
        'rotation_y': []
    })
    with open(label_path, 'r') as f:
        lines = f.readlines()
    # if len(lines) == 0 or len(lines[0]) < 15:
    #     content = []
    # else:
    content = [line.strip().split(' ') for line in lines]
    num_objects = len([x[0] for x in content if x[0] != 'DontCare'])
    annotations['name'] = np.array([x[0] for x in content])
    num_gt = len(annotations['name'])
    annotations['track_id'] = np.array([x[1] for x in content])
    annotations['truncated'] = np.array([float(x[2]) for x in content])
    annotations['occluded'] = np.array([int(x[3]) for x in content])
    annotations['alpha'] = np.array([float(x[4]) for x in content])
    annotations['bbox'] = np.array([[float(info) for info in x[5:9]]
                                    for x in content]).reshape(-1, 4)
    # dimensions will convert hwl format to standard lhw(camera) format.
    annotations['dimensions'] = np.array([[float(info) for info in x[9:12]]
                                          for x in content
                                          ]).reshape(-1, 3)[:, [2, 0, 1]]
    annotations['location'] = np.array([[float(info) for info in x[12:15]]
                                        for x in content]).reshape(-1, 3)
    annotations['rotation_y'] = np.array([float(x[15])
                                          for x in content]).reshape(-1)
    if len(content) != 0 and len(content[0]) == 17:  # have score
        annotations['score'] = np.array([float(x[16]) for x in content])
    else:
        annotations['score'] = np.zeros((annotations['bbox'].shape[0], ))
    index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
    annotations['index'] = np.array(index, dtype=np.int32)
    annotations['group_ids'] = np.arange(num_gt, dtype=np.int32)
    return annotations


def _extend_matrix(mat):
    mat = np.concatenate([mat, np.array([[0., 0., 0., 1.]])], axis=0)
    return mat


def get_spa_image_info(path,
                         training=True,
                         label_info=True,
                         velodyne=False,
                         calib=False,
                         with_plane=False,
                         image_ids=1384,
                         extend_matrix=True,
                         num_worker=8,
                         relative_path=True,
                         with_imageshape=True):
    """
    spa annotation format version 2:
    {
        [optional]points: [N, 3+] point cloud
        [optional, for spa]image: {
            image_idx: ...
            image_path: ...
            image_shape: ...
        }
        point_cloud: {
            num_features: 4
            velodyne_path: ...
        }
        [optional, for spa]calib: {
            R0_rect: ...
            Tr_velo_to_cam: ...
            P2: ...
        }
        annos: {
            location: [num_gt, 3] array
            dimensions: [num_gt, 3] array
            rotation_y: [num_gt] angle array
            name: [num_gt] ground truth name array
            [optional]difficulty: spa difficulty
            [optional]group_ids: used for multi-part object
        }
    }
    """
    root_path = Path(path)
    if not isinstance(image_ids, list):
        image_ids = list(range(image_ids))

    def map_func(idx):
        info = {}
        pc_info = {'num_features': 4}
        calib_info = {}

        image_info = {'image_idx': [idx.split("*")[2] for i in range(5)]}
        annotations = None
        if velodyne:
            pc_info['velodyne_path'] = get_velodyne_path(
                idx.split("*")[-1], Path(path,idx.split("*")[0],idx.split("*")[1], "velo/bin/data/"), training, relative_path)

        cam_front_left = get_image_path(idx.split("*")[2], Path(path,idx.split("*")[0],idx.split("*")[1]), "cam_img/1/data_rgb/", ".png", training,
                                                  relative_path)
        cam_front = get_image_path(idx.split("*")[2], Path(path,idx.split("*")[0],idx.split("*")[1]), "cam_img/4/data_rgb/", ".png", training,
                                                  relative_path)
        cam_front_right = get_image_path(idx.split("*")[2], Path(path,idx.split("*")[0],idx.split("*")[1]), "cam_img/5/data_rgb/", ".png", training,
                                                  relative_path)                                                                                    
        cam_back_left = get_image_path(idx.split("*")[2], Path(path,idx.split("*")[0],idx.split("*")[1]), "cam_img/2/data_rgb/", ".png", training,
                                                  relative_path)
        cam_back_right = get_image_path(idx.split("*")[2], Path(path,idx.split("*")[0],idx.split("*")[1]), "cam_img/3/data_rgb/", ".png", training,
                                                  relative_path)

        cam_list = [cam_front_left, cam_back_left, cam_back_right, cam_front, cam_front_right] 
        image_info['image_path'] = cam_list


        if with_imageshape:
            img_shape = []
            for _, cam in enumerate(cam_list):
                img_path = image_info['image_path'][_]
            # img_path = image_info['image_path']
                if relative_path: 
                    img_path = str(img_path)
                
                img_shape.append(np.array(
                    io.imread(img_path).shape[:2], dtype=np.int32))
            image_info['image_shape'] = img_shape
            
        if label_info:
            label_paths = []
            annotations = []
            for ii in [1,2,3,4,5]:
                label_path = get_label_path(idx.split("*")[2], Path(path,idx.split("*")[0],idx.split("*")[1]), "label/label_{}".format(ii), training, relative_path)
                if relative_path:
                    # label_path = str(root_path / label_path)
                    label_path = str(label_path)
                label_paths.append(label_path)
                annotation = get_label_anno(label_path)
                annotations.append(annotation)
        info['image'] = image_info
        info['point_cloud'] = pc_info
        if calib:
            calib_path = get_calib_path(
                idx.split("*")[2], Path(path,idx.split("*")[0],idx.split("*")[1]), "calib/", training, relative_path=True)
            with open(calib_path, 'r') as f:
                lines = f.read().splitlines()
            P0 = np.array([float(info) for info in lines[0].split(' ')[1:13]
                           ]).reshape([3, 4])
            P1 = np.array([float(info) for info in lines[1].split(' ')[1:13]
                           ]).reshape([3, 4])
            P2 = np.array([float(info) for info in lines[2].split(' ')[1:13]
                           ]).reshape([3, 4])
            P3 = np.array([float(info) for info in lines[3].split(' ')[1:13]
                           ]).reshape([3, 4])
            P4 = np.array([float(info) for info in lines[3].split(' ')[1:13]
                           ]).reshape([3, 4])
            
            if extend_matrix:
                P0 = _extend_matrix(P0)
                P1 = _extend_matrix(P1)
                P2 = _extend_matrix(P2)
                P3 = _extend_matrix(P3)
                P4 = _extend_matrix(P4)
            R0_rect = np.array([
                float(info) for info in lines[5].split(' ')[1:10]
            ]).reshape([3, 3])
            if extend_matrix:
                rect_4x4 = np.zeros([4, 4], dtype=R0_rect.dtype)
                rect_4x4[3, 3] = 1.
                rect_4x4[:3, :3] = R0_rect
            else:
                rect_4x4 = R0_rect

            Tr_velo_to_cam = np.array([
                float(info) for info in lines[6].split(' ')[1:13]
            ]).reshape([3, 4])
            Tr_imu_to_velo = np.array([
                float(info) for info in lines[7].split(' ')[1:13]
            ]).reshape([3, 4])
            if extend_matrix:
                Tr_velo_to_cam = _extend_matrix(Tr_velo_to_cam)
                Tr_imu_to_velo = _extend_matrix(Tr_imu_to_velo)
            
            Tr_0 = np.array([float(info) for info in lines[8].split(' ')[1:]])
            Tr_1 = np.array([float(info) for info in lines[9].split(' ')[1:]])
            Tr_2 = np.array([float(info) for info in lines[10].split(' ')[1:]])
            Tr_3 = np.array([float(info) for info in lines[11].split(' ')[1:]])
            Tr_4 = np.array([float(info) for info in lines[12].split(' ')[1:]])

            calib_info['P0'] = P0
            calib_info['P1'] = P1
            calib_info['P2'] = P2
            calib_info['P3'] = P3
            calib_info['P4'] = P4
            calib_info['R0_rect'] = rect_4x4
            calib_info['Tr_velo_to_cam'] = Tr_velo_to_cam
            calib_info['Tr_imu_to_velo'] = Tr_imu_to_velo
            calib_info['Tr_0'] = P0[:, 3][:3]
            calib_info['Tr_1'] = P1[:, 3][:3]
            calib_info['Tr_2'] = P2[:, 3][:3]
            calib_info['Tr_3'] = P3[:, 3][:3]
            calib_info['Tr_4'] = P4[:, 3][:3]
            info['calib'] = calib_info

        if with_plane:
            pass
            # plane_path = get_plane_path(idx, path, training, relative_path)
            # if relative_path:
            #     plane_path = str(root_path / plane_path)
            # lines = mmcv.list_from_file(plane_path)
            # info['plane'] = np.array([float(i) for i in lines[3].split()])

        if annotations is not None:
            data_anno = {}
            for ii, anno in enumerate(annotations):
                if ii == 0:
                    for key in anno.keys():
                        data_anno[key] = anno[key]
                    data_anno['mask'] = np.full(len(data_anno['name']), ii)
                else:
                    for key in anno.keys():
                        if key in ['bbox', 'dimensions', 'location']:
                            data_anno[key] = np.vstack((data_anno[key], anno[key]))
                        else:
                            data_anno[key] = np.hstack((data_anno[key], anno[key]))
                    mask = np.full(len(anno['name']), ii)
                    data_anno['mask'] = np.hstack((data_anno['mask'], mask))

            info['annos'] = data_anno
            add_difficulty_to_annos(info)
        return info

    with futures.ThreadPoolExecutor(num_worker) as executor:
        image_infos = executor.map(map_func, image_ids)

    return list(image_infos)


class WaymoInfoGatherer:
    """
    Parallel version of waymo dataset information gathering.
    Waymo annotation format version like spa:
    {
        [optional]points: [N, 3+] point cloud
        [optional, for spa]image: {
            image_idx: ...
            image_path: ...
            image_shape: ...
        }
        point_cloud: {
            num_features: 6
            velodyne_path: ...
        }
        [optional, for spa]calib: {
            R0_rect: ...
            Tr_velo_to_cam0: ...
            P0: ...
        }
        annos: {
            location: [num_gt, 3] array
            dimensions: [num_gt, 3] array
            rotation_y: [num_gt] angle array
            name: [num_gt] ground truth name array
            [optional]difficulty: spa difficulty
            [optional]group_ids: used for multi-part object
        }
    }
    """

    def __init__(self,
                 path,
                 training=True,
                 label_info=True,
                 velodyne=False,
                 calib=False,
                 pose=False,
                 extend_matrix=True,
                 num_worker=8,
                 relative_path=True,
                 with_imageshape=True,
                 max_sweeps=5) -> None:
        self.path = path
        self.training = training
        self.label_info = label_info
        self.velodyne = velodyne
        self.calib = calib
        self.pose = pose
        self.extend_matrix = extend_matrix
        self.num_worker = num_worker
        self.relative_path = relative_path
        self.with_imageshape = with_imageshape
        self.max_sweeps = max_sweeps

    def gather_single(self, idx):
        root_path = Path(self.path)
        info = {}
        pc_info = {'num_features': 6}
        calib_info = {}

        image_info = {'image_idx': idx}
        annotations = None
        if self.velodyne:
            pc_info['velodyne_path'] = get_velodyne_path(
                idx,
                self.path,
                self.training,
                self.relative_path,
                use_prefix_id=True)
            with open(
                    get_timestamp_path(
                        idx,
                        self.path,
                        self.training,
                        relative_path=False,
                        use_prefix_id=True)) as f:
                info['timestamp'] = np.int64(f.read())
        image_info['image_path'] = get_image_path(
            idx,
            self.path,
            self.training,
            self.relative_path,
            info_type='image_0',
            file_tail='.jpg',
            use_prefix_id=True)
        if self.with_imageshape:
            img_path = image_info['image_path']
            if self.relative_path:
                img_path = str(root_path / img_path)
            # io using PIL is significantly faster than skimage
            w, h = Image.open(img_path).size
            image_info['image_shape'] = np.array((h, w), dtype=np.int32)
        if self.label_info:
            label_path = get_label_path(
                idx,
                self.path,
                self.training,
                self.relative_path,
                info_type='label_all',
                use_prefix_id=True)
            if self.relative_path:
                label_path = str(root_path / label_path)
            annotations = get_label_anno(label_path)
        info['image'] = image_info
        info['point_cloud'] = pc_info
        if self.calib:
            calib_path = get_calib_path(
                idx,
                self.path,
                self.training,
                relative_path=False,
                use_prefix_id=True)
            with open(calib_path, 'r') as f:
                lines = f.readlines()
            P0 = np.array([float(info) for info in lines[0].split(' ')[1:13]
                           ]).reshape([3, 4])
            P1 = np.array([float(info) for info in lines[1].split(' ')[1:13]
                           ]).reshape([3, 4])
            P2 = np.array([float(info) for info in lines[2].split(' ')[1:13]
                           ]).reshape([3, 4])
            P3 = np.array([float(info) for info in lines[3].split(' ')[1:13]
                           ]).reshape([3, 4])
            P4 = np.array([float(info) for info in lines[4].split(' ')[1:13]
                           ]).reshape([3, 4])
            if self.extend_matrix:
                P0 = _extend_matrix(P0)
                P1 = _extend_matrix(P1)
                P2 = _extend_matrix(P2)
                P3 = _extend_matrix(P3)
                P4 = _extend_matrix(P4)
            R0_rect = np.array([
                float(info) for info in lines[5].split(' ')[1:10]
            ]).reshape([3, 3])
            if self.extend_matrix:
                rect_4x4 = np.zeros([4, 4], dtype=R0_rect.dtype)
                rect_4x4[3, 3] = 1.
                rect_4x4[:3, :3] = R0_rect
            else:
                rect_4x4 = R0_rect

            Tr_velo_to_cam = np.array([
                float(info) for info in lines[6].split(' ')[1:13]
            ]).reshape([3, 4])
            if self.extend_matrix:
                Tr_velo_to_cam = _extend_matrix(Tr_velo_to_cam)
            calib_info['P0'] = P0
            calib_info['P1'] = P1
            calib_info['P2'] = P2
            calib_info['P3'] = P3
            calib_info['P4'] = P4
            calib_info['R0_rect'] = rect_4x4
            calib_info['Tr_velo_to_cam'] = Tr_velo_to_cam
            info['calib'] = calib_info
        if self.pose:
            pose_path = get_pose_path(
                idx,
                self.path,
                self.training,
                relative_path=False,
                use_prefix_id=True)
            info['pose'] = np.loadtxt(pose_path)

        if annotations is not None:
            info['annos'] = annotations
            info['annos']['camera_id'] = info['annos'].pop('score')
            add_difficulty_to_annos(info)

        sweeps = []
        prev_idx = idx
        while len(sweeps) < self.max_sweeps:
            prev_info = {}
            prev_idx -= 1
            prev_info['velodyne_path'] = get_velodyne_path(
                prev_idx,
                self.path,
                self.training,
                self.relative_path,
                exist_check=False,
                use_prefix_id=True)
            if_prev_exists = osp.exists(
                Path(self.path) / prev_info['velodyne_path'])
            if if_prev_exists:
                with open(
                        get_timestamp_path(
                            prev_idx,
                            self.path,
                            self.training,
                            relative_path=False,
                            use_prefix_id=True)) as f:
                    prev_info['timestamp'] = np.int64(f.read())
                prev_pose_path = get_pose_path(
                    prev_idx,
                    self.path,
                    self.training,
                    relative_path=False,
                    use_prefix_id=True)
                prev_info['pose'] = np.loadtxt(prev_pose_path)
                sweeps.append(prev_info)
            else:
                break
        info['sweeps'] = sweeps

        return info

    def gather(self, image_ids):
        if not isinstance(image_ids, list):
            image_ids = list(range(image_ids))
        image_infos = mmcv.track_parallel_progress(self.gather_single,
                                                   image_ids, self.num_worker)
        return list(image_infos)


def spa_anno_to_label_file(annos, folder):
    folder = Path(folder)
    for anno in annos:
        image_idx = anno['metadata']['image_idx']
        label_lines = []
        for j in range(anno['bbox'].shape[0]):
            label_dict = {
                'name': anno['name'][j],
                'alpha': anno['alpha'][j],
                'bbox': anno['bbox'][j],
                'location': anno['location'][j],
                'dimensions': anno['dimensions'][j],
                'rotation_y': anno['rotation_y'][j],
                'score': anno['score'][j],
            }
            label_line = spa_result_line(label_dict)
            label_lines.append(label_line)
        label_file = folder / f'{get_image_index_str(image_idx)}.txt'
        label_str = '\n'.join(label_lines)
        with open(label_file, 'w') as f:
            f.write(label_str)


def add_difficulty_to_annos(info):
    min_height = [40, 25,
                  25]  # minimum height for evaluated groundtruth/detections
    max_occlusion = [
        0, 1, 2
    ]  # maximum occlusion level of the groundtruth used for evaluation
    max_trunc = [
        0.15, 0.3, 0.5
    ]  # maximum truncation level of the groundtruth used for evaluation
    annos = info['annos']
    dims = annos['dimensions']  # lhw format
    bbox = annos['bbox']
    height = bbox[:, 3] - bbox[:, 1]
    occlusion = annos['occluded']
    truncation = annos['truncated']
    diff = []
    easy_mask = np.ones((len(dims), ), dtype=np.bool)
    moderate_mask = np.ones((len(dims), ), dtype=np.bool)
    hard_mask = np.ones((len(dims), ), dtype=np.bool)
    i = 0
    for h, o, t in zip(height, occlusion, truncation):
        if o > max_occlusion[0] or h <= min_height[0] or t > max_trunc[0]:
            easy_mask[i] = False
        if o > max_occlusion[1] or h <= min_height[1] or t > max_trunc[1]:
            moderate_mask[i] = False
        if o > max_occlusion[2] or h <= min_height[2] or t > max_trunc[2]:
            hard_mask[i] = False
        i += 1
    is_easy = easy_mask
    is_moderate = np.logical_xor(easy_mask, moderate_mask)
    is_hard = np.logical_xor(hard_mask, moderate_mask)

    for i in range(len(dims)):
        if is_easy[i]:
            diff.append(0)
        elif is_moderate[i]:
            diff.append(1)
        elif is_hard[i]:
            diff.append(2)
        else:
            diff.append(-1)
    annos['difficulty'] = np.array(diff, np.int32)
    return diff


def spa_result_line(result_dict, precision=4):
    prec_float = '{' + ':.{}f'.format(precision) + '}'
    res_line = []
    all_field_default = OrderedDict([
        ('name', None),
        ('truncated', -1),
        ('occluded', -1),
        ('alpha', -10),
        ('bbox', None),
        ('dimensions', [-1, -1, -1]),
        ('location', [-1000, -1000, -1000]),
        ('rotation_y', -10),
        ('score', 0.0),
    ])
    res_dict = [(key, None) for key, val in all_field_default.items()]
    res_dict = OrderedDict(res_dict)
    for key, val in result_dict.items():
        if all_field_default[key] is None and val is None:
            raise ValueError('you must specify a value for {}'.format(key))
        res_dict[key] = val

    for key, val in res_dict.items():
        if key == 'name':
            res_line.append(val)
        elif key in ['truncated', 'alpha', 'rotation_y', 'score']:
            if val is None:
                res_line.append(str(all_field_default[key]))
            else:
                res_line.append(prec_float.format(val))
        elif key == 'occluded':
            if val is None:
                res_line.append(str(all_field_default[key]))
            else:
                res_line.append('{}'.format(val))
        elif key in ['bbox', 'dimensions', 'location']:
            if val is None:
                res_line += [str(v) for v in all_field_default[key]]
            else:
                res_line += [prec_float.format(v) for v in val]
        else:
            raise ValueError('unknown key. supported key:{}'.format(
                res_dict.keys()))
    return ' '.join(res_line)
