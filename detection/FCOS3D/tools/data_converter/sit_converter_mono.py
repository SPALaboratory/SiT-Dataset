from itertools import count
import os
from collections import OrderedDict
from os import path as osp
from typing import List, Tuple, Union

import mmcv
import json
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion
from shapely.geometry import MultiPoint, box
import copy
from mmdet3d.core.bbox import points_cam2img
from mmdet3d.datasets import NuScenesDataset
import cv2
from pathlib import Path
import matplotlib.pyplot as plt

nus_categories = ('car', 'bicycle', 'motorcycle', 'pedestrian', 'truck', 'bus', 'kickboard')

nus_attributes = ('cycle.with_rider', 'cycle.without_rider',
                  'pedestrian.moving', 'pedestrian.standing',
                  'pedestrian.sitting_lying_down', 'vehicle.moving',
                  'vehicle.parked', 'vehicle.stopped', 'None',
                  'vehicle.bicycle', 'vehicle.truck','vehicle.bus.bendy',
                   )

DefaultAttribute = {
        'car': 'vehicle.parked',
        'pedestrian': 'pedestrian.moving',
        'motorcycle': 'cycle.without_rider',
        'bicycle': 'vehicle.moving',
        'bus': 'vehicle.bus.bendy',
        'truck': 'vehicle.truck',
        'kickboard': 'vehicle.moving',
    }


def euler_to_rotmat2(euler_angles):
    roll, pitch, yaw = euler_angles
    rot_x = np.array([[1, 0, 0],
                      [0, np.cos(roll), -np.sin(roll)],
                      [0, np.sin(roll), np.cos(roll)]])
    rot_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                      [0, 1, 0],
                      [-np.sin(pitch), 0, np.cos(pitch)]])
    rot_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                      [np.sin(yaw), np.cos(yaw), 0],
                      [0, 0, 1]])
    rot_mat = np.dot(rot_z, np.dot(rot_y, rot_x))
    return rot_mat

def make_3x3_mat(mat):
    mat_3x3 = np.eye(3)
    mat_3x3[:2, :2] = mat[:2,:2]
    mat_3x3[:2, 2] = mat[:2, 3]
    return mat_3x3
    
def xyzrpy_to_rot_mat_string(string, break_flag=False):
    xyzrpy = []
    for s in string.split(','):
        if s != "":
            xyzrpy.append(float(s))
    
    xyz = xyzrpy[:3]
    rpy = xyzrpy[3:6]
    R = euler_to_rotmat2([rpy[0], rpy[1], rpy[2]])
    mat = np.concatenate([R, np.array([xyz]).T], axis=1)
    mat = np.concatenate([mat, np.array([[0,0,0,1]])])
    if break_flag:
        breakpoint()
    return make_3x3_mat(mat)

def xyzrpy_to_rot_mat(xyzrpy, make_3x3 = True):    
    xyz = xyzrpy[:3]
    rpy = xyzrpy[3:6]
    R = euler_to_rotmat2([rpy[0], rpy[1], rpy[2]])
    mat = np.concatenate([R, np.array([xyz]).T], axis=1)
    mat = np.concatenate([mat, np.array([[0,0,0,1]])])
    if make_3x3:
        return make_3x3_mat(mat)
    else:
        return mat

def xyzypr_to_rot_mat(string):
    xyzypr = []
    for s in string.split(' '):
        if s != "":
            xyzypr.append(float(s))
    xyz = xyzypr[:3]
    ypr = xyzypr[3:6]
    R = euler_to_rotmat2([ypr[2], ypr[1], ypr[0]])
    mat = np.concatenate([R, np.array([xyz]).T], axis=1)
    mat = np.concatenate([mat, np.array([[0,0,0,1]])])
    return mat


def make_4x4_mat(mat):
    mat_3x3 = np.eye(3)
    mat_3x3[:2, :2] = mat[:2,:2]
    mat_3x3[:2, 2] = mat[:2, 3]
    return mat_3x3


def draw_projected_box_2d(image, points_2d, color=(0, 255, 255), thickness=2):
    for i in range(4):
        point_1 = tuple(points_2d[i].astype(int))
        point_2 = tuple(points_2d[(i + 1) % 4].astype(int))
        point_3 = tuple(points_2d[i + 4].astype(int))
        point_4 = tuple(points_2d[((i + 1) % 4) + 4].astype(int))
        
        cv2.line(image, point_1, point_2, color, thickness)
        cv2.line(image, point_1, point_3, color, thickness)
        cv2.line(image, point_2, point_4, color, thickness)
        cv2.line(image, point_3, point_4, color, thickness)
        
def draw_projected_box3d(image, qs, color=(0,255,255), thickness=1):
    ''' Draw 3d bounding box in image
        qs: (8,3) array of vertices for the 3d box in following order:
          1 -------- 0
         /|         /|
        2 -------- 3 .
        | |        | |
        . 5 -------- 4
        |/         |/
        6 -------- 7
    '''
    qs = qs.astype(np.int32)
    for k in range(0,4):
        i,j=k,(k+1)%4
        cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.LINE_AA)
        i,j=k+4,(k+1)%4 + 4
        cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.LINE_AA)
        i,j=k,k+4
        cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.LINE_AA)
    cv2.line(image, ((qs[0,0]+qs[1,0])//2, (qs[0,1]+qs[1,1])//2), ((qs[0,0]+qs[1,0]+qs[2,0]+qs[3,0])//4, (qs[0,1]+qs[1,1]+qs[2,1]+qs[3,1])//4), (0,255,0), thickness, cv2.LINE_AA)
    cv2.line(image, ((qs[4,0]+qs[5,0])//2, (qs[4,1]+qs[5,1])//2), ((qs[4,0]+qs[5,0]+qs[6,0]+qs[7,0])//4, (qs[4,1]+qs[5,1]+qs[6,1]+qs[7,1])//4), (0,255,0), thickness, cv2.LINE_AA)
    return image
    
def draw_projected_box3d_on_bev(ax, corners_3d_, color=(0,255,255), thickness=1):
    point_index = [[0,1], [1,2], [2,3],[3,0]]
    for idcs in point_index:
        ax.plot(corners_3d_[idcs][:,0], corners_3d_[idcs][:,1], 'r')
    center_front = (corners_3d_[0] + corners_3d_[1])/2
    center = corners_3d_[[0,1,2,3]].sum(0)/4
    ax.plot([center_front[0], center[0]], [center_front[1], center[1]], 'y')
    return ax

def draw_projected_center(image, qs, color=(0,255,255), thickness=1):
    qs = qs.astype(np.int32)
    cv2.circle(image, (qs[0],qs[1]), 5, (0, 0,255), -1)
    return image

    
def euler_to_rotmat(euler_angles):
    """
    Convert Euler angles to rotation matrix using X-Y-Z convention
    :param euler_angles: array-like object of Euler angles (in radians)
    :return: 3x3 rotation matrix
    """
    roll, pitch, yaw = euler_angles
    rot_x = np.array([[1, 0, 0],
                      [0, np.cos(roll), -np.sin(roll)],
                      [0, np.sin(roll), np.cos(roll)]])
    rot_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                      [0, 1, 0],
                      [-np.sin(pitch), 0, np.cos(pitch)]])
    rot_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                      [np.sin(yaw), np.cos(yaw), 0],
                      [0, 0, 1]])
    rot_mat = np.dot(rot_z, np.dot(rot_y, rot_x))
    return rot_mat

def rotmat_to_euler(rot_mat):
    """
    Convert rotation matrix to Euler angles using X-Y-Z convention
    :param rot_mat: 3x3 rotation matrix
    :return: array-like object of Euler angles (in radians)
    """
    sy = np.sqrt(rot_mat[0, 0] * rot_mat[0, 0] + rot_mat[1, 0] * rot_mat[1, 0])
    singular = sy < 1e-6
    if not singular:
        roll = np.arctan2(rot_mat[2, 1], rot_mat[2, 2])
        pitch = np.arctan2(-rot_mat[2, 0], sy)
        yaw = np.arctan2(rot_mat[1, 0], rot_mat[0, 0])
    else:
        roll = np.arctan2(-rot_mat[1, 2], rot_mat[1, 1])
        pitch = np.arctan2(-rot_mat[2, 0], sy)
        yaw = 0
    return np.array([roll, pitch, yaw])

def rot_mat_to_yaw(rot_mat):
    yaw = np.arctan2(rot_mat[1, 0], rot_mat[0, 0])
    return yaw

def _read_imageset_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return [line.splitlines()[0] for line in lines]

def create_sit_infos_mono(root_path,
                          info_prefix,
                          version='v1.0-sit-trainval',
                          max_sweeps=10, overfit=False):
    """Create info file of nuscene dataset.

    Given the raw data, generate its related info file in pkl format.

    Args:
        root_path (str): Path of the data root.
        info_prefix (str): Prefix of the info file to be generated.
        version (str, optional): Version of the data.
            Default: 'v1.0-trainval'.
        max_sweeps (int, optional): Max number of sweeps.
            Default: 10.
    """
    from nuscenes.utils import splits
    available_vers = ['sit-trainval', 'sit-test', 'v1.0-test', 'v1.0-mini']
    assert version in available_vers
    if version == 'v1.0-trainval':
        train_scenes = splits.train
        val_scenes = splits.val
    elif version == 'v1.0-test':
        test_scenes = splits.test
    elif version == 'v1.0-mini':
        train_scenes = splits.mini_train
        val_scenes = splits.mini_val
    elif version == "sit-trainval":
        imageset_folder = Path(root_path) / 'ImageSets'
        # train_scenes = _read_imageset_file(str(imageset_folder / 'train.txt'))
        # val_scenes = _read_imageset_file(str(imageset_folder / 'val.txt'))
        train_scenes = _read_imageset_file(str(imageset_folder / 'train_temp.txt'))
        val_scenes = _read_imageset_file(str(imageset_folder / 'val_temp.txt'))
        
        
        
    
    elif version == "sit-test":
        imageset_folder = Path(root_path) / 'ImageSets'
        test_scenes = _read_imageset_file(str(imageset_folder / 'test.txt'))
    else:
        raise ValueError('unknown')

    test = 'test' in version
    if test:
        print('test scene: {}'.format(len(test_scenes)))
        test_nusc_infos = _fill_test_infos_mono(
            None, test_scenes, test = test, max_sweeps = max_sweeps)
    else:
        print('train scene: {}, val scene: {}'.format(
            len(train_scenes), len(val_scenes)))
        train_nusc_infos, val_nusc_infos = _fill_trainval_infos_mono(
            None, train_scenes, val_scenes, test, max_sweeps=max_sweeps)

    metadata = dict(version=version)
    if test:
        print('test sample: {}'.format(len(test_nusc_infos)))
        data = dict(infos=test_nusc_infos, metadata=metadata)
        info_path = osp.join(root_path,
                             '{}_infos_test.pkl'.format(info_prefix))
        mmcv.dump(data, info_path)
    else:
        print('train sample: {}, val sample: {}'.format(
            len(train_nusc_infos), len(val_nusc_infos)))
        if overfit:
            data = dict(infos=train_nusc_infos, metadata=metadata)
            info_path = osp.join(root_path,
                                '{}_infos_train.pkl'.format(info_prefix))
            mmcv.dump(data, info_path)
        else:
            data = dict(infos=train_nusc_infos, metadata=metadata)
            info_path = osp.join(root_path,
                                '{}_infos_train.pkl'.format(info_prefix))
            mmcv.dump(data, info_path)
            data['infos'] = val_nusc_infos
            info_val_path = osp.join(root_path,
                                    '{}_infos_val.pkl'.format(info_prefix))
            mmcv.dump(data, info_val_path)



def get_label_anno(label_path):
    annotations = {}
    annotations.update({
        'name': [],
        'track_id': [],
        'cam_id': [],
        'bbox': [],
        'dimensions': [],
        'location': [],
        'rotation_y': []
    })

    name_map={'Car':'car', 'Truck':'truck', 'Bus':'bus', 'Pedestrian':'pedestrian','Bicyclist':'bicycle', 'Motorcycle':'motorcycle', 
              'Kickboard':'kickboard', 'Vehicle':'car', 'Pedestrian_sitting':'pedestrian', 'Pedestrain_sitting':'pedestrian',
              'Cyclist' : 'bicycle', 'Motorcyclist':'motorcycle'
              }
    # append : only read when 3d labels are exists
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            lines = f.readlines()
        content = [line.strip().split(' ') for line in lines]

        annotations['name'] = np.array([ name_map[x[0]] for x in content])
        # sitting_mask = np.array([True if i != 'pedestrian_sitting' else False  for i in annotations['name']])
        sitting_mask = np.array([True for i in annotations['name']])
        annotations['name'] = annotations['name'][sitting_mask]
        annotations['track_id'] = np.array([x[1] for x in content])[sitting_mask]
        annotations['cam_id'] = np.array([float(x[2]) for x in content])[sitting_mask]
        annotations['bbox'] = np.array([[float(info) for info in x[3:7]]
                                        for x in content]).reshape(-1, 4)[sitting_mask] 
        
        annotations['dimensions'] = np.abs(np.array([[float(info) for info in x[2:5]]
                                            for x in content
                                            ]).reshape(-1, 3)[:, [1, 2, 0]])[sitting_mask] #h, l, w -> l ,w, h

        annotations['location'] = np.array([[float(info) for info in x[5:8]]
                                            for x in content]).reshape(-1, 3)[sitting_mask]

        annotations['rotation_y'] = np.array([float(x[8])
                                            for x in content]).reshape(-1)[sitting_mask]
                                            
        if len(content) != 0 and len(content[0]) == 10:  # have score
            annotations['score'] = np.array([float(x[9]) for x in content])[sitting_mask]
        else:
            annotations['score'] = np.ones((annotations['bbox'].shape[0], ))
    
  
    return annotations   

def get_ego_matrix(label_path):
    with open(label_path, 'r') as f:
        lines = f.readlines()
    return np.array([float(i) for i in lines[0].split(",")]).reshape(4, 4)

def box_center_to_corner_3d(centers, dims, angles):

    translation = centers[:, 0:3]
    # h, w, l = dims[:, 0], dims[:, 1], dims[:, 2]
    l, w, h = dims[:, 0], dims[:, 1], dims[:, 2]
    rotation = angles

    # Create a bounding box outline
    x_corners = np.array([[l_ / 2, l_ / 2, -l_ / 2, -l_ / 2, l_ / 2, l_ / 2, -l_ / 2, -l_ / 2] for l_ in l])
    #[[l_ / 2, l_ / 2, -l_ / 2, -l_ / 2, l_ / 2, l_ / 2, -l_ / 2, -l_ / 2] for l_ in l]
    y_corners = np.array([[w_ / 2, -w_ / 2, -w_ / 2, w_ / 2, w_ / 2, -w_ / 2, -w_ / 2, w_ / 2] for w_ in w])
    #[w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
    z_corners = np.array([[-h_ / 2, -h_ / 2, -h_ / 2, -h_ / 2, h_ / 2, h_ / 2, h_ / 2, h_ / 2] for h_ in h])
    #[-h/2, -h/2, -h/2, -h/2, h/2, h/2, h/2, h/2] #[0, 0, 0, 0, -h, -h, -h, -h]
    bounding_box = np.array([np.vstack([x_corners[i], y_corners[i], z_corners[i]]) for i in range(x_corners.shape[0])])


    rotation_matrix = np.array([np.array([[np.cos(rotation_),  -np.sin(rotation_), 0],
                                            [np.sin(rotation_), np.cos(rotation_), 0],
                                            [0,  0,  1]]) for rotation_ in rotation])


    corner_box = np.array([np.dot(rotation_matrix[i], bounding_box[i]).T + translation[i] for i in range(x_corners.shape[0])])

    return corner_box

def get_pts_in_3dbox(pc, corners):
    num_pts_in_gt = []
    for num, corner in enumerate(corners):
        x_max, x_min = corner[:, 0].max(), corner[:, 0].min()
        y_max, y_min = corner[:, 1].max(), corner[:, 1].min()
        z_max, z_min = corner[:, 2].max(), corner[:, 2].min()
        
        count = 0
        for point in pc:
            if (x_min<= point[0] <= x_max and \
                 y_min <= point[1] <= y_max and \
                 z_min <= point[2] <= z_max):
                count += 1
        
        num_pts_in_gt.append(count)
    return num_pts_in_gt

def get_pts_in_3dbox_(pc, corners):
    num_pts_in_gt = []
    for num, corner in enumerate(corners):
        x_max, x_min = corner[:, 0].max(), corner[:, 0].min()
        y_max, y_min = corner[:, 1].max(), corner[:, 1].min()
        z_max, z_min = corner[:, 2].max(), corner[:, 2].min()

        mask_x = np.logical_and(pc[:,0] >= x_min, pc[:, 0] <= x_max)
        mask_y = np.logical_and(pc[:,1] >= y_min, pc[:, 1] <= y_max)
        mask_z = np.logical_and(pc[:,2] >= z_min, pc[:, 2] <= z_max)
        mask = mask_x * mask_y * mask_z
        num_pts_in_gt.append(mask.sum())
    
    return num_pts_in_gt

def get_pts_index_in_3dbox_(pc, corners):
    num_pts_in_gt = []
    for num, corner in enumerate(corners):
        x_max, x_min = corner[:, 0].max(), corner[:, 0].min()
        y_max, y_min = corner[:, 1].max(), corner[:, 1].min()
        z_max, z_min = corner[:, 2].max(), corner[:, 2].min()

        mask_x = np.logical_and(pc[:,0] >= x_min, pc[:, 0] <= x_max)
        mask_y = np.logical_and(pc[:,1] >= y_min, pc[:, 1] <= y_max)
        mask_z = np.logical_and(pc[:,2] >= z_min, pc[:, 2] <= z_max)
        mask = mask_x * mask_y * mask_z
        num_pts_in_gt.append(mask)
    
    return num_pts_in_gt

def _fill_test_infos_mono(nusc,
                         test_scenes,
                         test=True,
                         max_sweeps=10):
    # single input image -> info related annotation
    """Generate the test infos from the raw data.

    Args:
        nusc (:obj:`NuScenes`): Dataset class in the nuScenes dataset.
        test_scenes (list[str]): Basic information of test scenes.
        test (bool, optional): Whether use the test mode. In test mode, no
            annotations can be accessed. Default: False.
        max_sweeps (int, optional): Max number of sweeps. Default: 10.

    Returns:
        tuple[list[dict]]: Information of training set and validation set
            that will be saved to the info file.
    """
    test_nusc_infos = []
    root_path = Path("./data/sit_full/")
    iter_count = 0
    for path in mmcv.track_iter_progress(test_scenes):
        place = path.split("*")[0]
        scene = path.split("*")[1]
        frame = path.split("*")[2]
        token = path
        
        calib_path = root_path / place / scene / "calib" / "{}.txt".format(frame)
        if calib_path is not None:
            with open(calib_path, 'r') as f:
                lines = f.read().splitlines()
                P0_intrinsic = np.array([float(info) for info in lines[0].split(' ')[1:10]
                            ]).reshape([3, 3])
                P1_intrinsic = np.array([float(info) for info in lines[1].split(' ')[1:10]
                            ]).reshape([3, 3])
                P2_intrinsic = np.array([float(info) for info in lines[2].split(' ')[1:10]
                            ]).reshape([3, 3])
                P3_intrinsic = np.array([float(info) for info in lines[3].split(' ')[1:10]
                            ]).reshape([3, 3])
                P4_intrinsic = np.array([float(info) for info in lines[4].split(' ')[1:10]
                            ]).reshape([3, 3])
                P0_extrinsic = np.array([float(info) for info in lines[5].split(' ')[1:13]
                            ]).reshape([3, 4])
                P1_extrinsic = np.array([float(info) for info in lines[6].split(' ')[1:13]
                            ]).reshape([3, 4])
                P2_extrinsic = np.array([float(info) for info in lines[7].split(' ')[1:13]
                            ]).reshape([3, 4])
                P3_extrinsic = np.array([float(info) for info in lines[8].split(' ')[1:13]
                            ]).reshape([3, 4])
                P4_extrinsic = np.array([float(info) for info in lines[9].split(' ')[1:13]
                            ]).reshape([3, 4])
                P0_distortion = np.array([float(info) for info in lines[11].split(' ')[1:6]
                            ]).reshape([1, 5])
                P1_distortion = np.array([float(info) for info in lines[12].split(' ')[1:6]
                            ]).reshape([1, 5])
                P2_distortion = np.array([float(info) for info in lines[13].split(' ')[1:6]
                            ]).reshape([1, 5])
                P3_distortion = np.array([float(info) for info in lines[14].split(' ')[1:6]
                            ]).reshape([1, 5])
                P4_distortion = np.array([float(info) for info in lines[15].split(' ')[1:6]
                            ]).reshape([1, 5])
        # projection_matrix = [P0, P1, P2, P3, P4]
        projection_matrix = [ [P0_intrinsic, P0_extrinsic, P0_distortion], \
                              [P1_intrinsic, P1_extrinsic, P1_distortion], \
                              [P2_intrinsic, P2_extrinsic, P2_distortion], \
                              [P3_intrinsic, P3_extrinsic, P3_distortion], \
                              [P4_intrinsic, P4_extrinsic, P4_distortion], 
            ]
        velo_path = root_path / place / scene / "velo/concat/bin_data" / "{}.bin".format(frame)


        mmcv.check_file_exist(velo_path)

        

        # obtain 5 image's information per frame
        camera_types = [
            'CAM_FRONT',
            'CAM_FRONT_RIGHT',
            'CAM_FRONT_LEFT',
            # 'CAM_BACK',
            'CAM_BACK_LEFT',
            'CAM_BACK_RIGHT',
        ] # 5: front, 4: front_left, 3: front_right, 2: back_right, 1:back_right

        camera_mapping = {4:'CAM_FRONT', 1:'CAM_FRONT_LEFT', 5:'CAM_FRONT_RIGHT', 3:'CAM_BACK_RIGHT',  2:'CAM_BACK_LEFT'}

        cam_num_list = [1, 2, 3, 4, 5]
        anno_path_ = []
        odom_path_ = []
        cam_path_ = []
        for num in cam_num_list:
            cam_path_.append(root_path / place / scene / "cam_img/{}".format(num) / "data_undist" / "{}.png".format(frame))
        
        anno_path_.append(root_path / place / scene / "label_3d/{}.txt".format(frame))
        odom_path_.append(root_path / place / scene / "ego_trajectory/{}.txt".format(frame))

        '''
        info = {
            'lidar_path': velo_path,
                'token': token+"*"+str(cam),
                'sweeps': [],
                'cams': None,
        }
        '''
        
        # token을 이미지 한장이 아니라 전체 frame으로 통합   
        info = {
            'lidar_path': velo_path,
                'token': token,
                'sweeps': [],
                'cams': None,
        }
        

        # token 1개에 5개에 대응되는 image, annotation 정보 list 형태로 수집
        cam_info_list = []
        gt_boxes_list = []
        names_list = []
        data_anno_list = []
        valid_flag_list = []

        if not test:
                data_anno = {}
                for id, a_path in enumerate(anno_path_):
                    anno_ = get_label_anno(a_path)
                    if id == 0:
                        for key in anno_.keys():
                            data_anno[key] = anno_[key]
                    else:
                        for key in anno_.keys():
                            if key in ['bbox', 'dimensions', 'location']:
                                data_anno[key] = np.vstack((data_anno[key], anno_[key]))
                            else:
                                data_anno[key] = np.hstack((data_anno[key], anno_[key]))


        for num, cam in enumerate(cam_num_list):
            # chgd token
            '''
            info = {
                'lidar_path': velo_path,
                'token': token+"*"+str(cam),
                'sweeps': [],
                'cams': None,
            }
            
            cam_info = {'data_path':cam_path_[cam-1], 'type': camera_mapping[cam]}
            cam_info.update(cam_intrinsic=projection_matrix[cam-1])
            info['cams'] = cam_info
            '''
            
            cam_info_list.append({'data_path':cam_path_[cam-1], 'type': camera_mapping[cam], 'cam_intrinsic':projection_matrix[cam-1]})

            sweeps = []
            info['sweeps'] = sweeps

            if not test:
                locs = np.array(data_anno['location']).reshape(-1, 3)
                dims = np.array(data_anno['dimensions']).reshape(-1, 3) # order : l,w,h 
                rots = np.array(data_anno['rotation_y']).reshape(-1, 1)
                names = np.array(data_anno['name'])
                tokens = np.array(token)
                track_ids = np.array(data_anno['track_id'])

                gt_boxes = np.concatenate([locs, dims, rots], axis=1)

                ego_motion = get_ego_matrix(odom_path_[0])
                ego_yaw = rotmat_to_euler(ego_motion[:3, :3])[2]
                
                        
                gt_boxes[:, 6] -= ego_yaw
                comp_obj_center = np.matmul(np.linalg.inv(ego_motion), np.concatenate([gt_boxes[:,:3], np.ones(gt_boxes[:,:3].shape[0]).reshape(1, -1).T], axis=1).T).T
                gt_boxes[:, :3] = comp_obj_center[:, :3]

                gt_boxes_corners = box_center_to_corner_3d(gt_boxes[:, :3], gt_boxes[:, 3:6], gt_boxes[:, 6])
                points = np.fromfile(velo_path, dtype=np.float32, count=-1).reshape(-1, 4) # 4 : x, y, z, intensity
                num_pts_list = get_pts_in_3dbox_(points, gt_boxes_corners)
                projection_m = np.eye(4)
                projection_m[:3, :] = np.matmul(projection_matrix[cam-1][0], projection_matrix[cam-1][1])

                center3d = np.array(gt_boxes[:, :3])
                center3d = np.concatenate([center3d,  np.ones_like(center3d[:,:1])], axis=1)
                center2d = np.matmul(projection_m, center3d.T).T
                center2d[:,0] /= center2d[:,2]
                center2d[:,1] /= center2d[:,2] 
                
                depth_limit = center2d[:,2] > 0
                left_limit = center2d[:,0] >= 0
                right_limit = center2d[:,0] < 1920
                total_limit = depth_limit * left_limit * right_limit

                data_anno['num_lidar_pts'] = np.array(num_pts_list)[total_limit]
                valid_flag = (data_anno['num_lidar_pts'] > 0)


        # cams : multi-view camera images with camera parameter
        info['cams'] = cam_info_list
        if not test:
            info['gt_boxes'] = gt_boxes
            info['gt_names'] = names

            info['num_lidar_pts'] = data_anno['num_lidar_pts']
            info['valid_flag'] = valid_flag

        test_nusc_infos.append(info)

        iter_count += 1

    return test_nusc_infos



def _fill_trainval_infos_mono(nusc,
                         train_scenes,
                         val_scenes,
                         test=False,
                         max_sweeps=10):
    # single input image -> info related annotation
    """Generate the train/val infos from the raw data.

    Args:
        nusc (:obj:`NuScenes`): Dataset class in the nuScenes dataset.
        train_scenes (list[str]): Basic information of training scenes.
        val_scenes (list[str]): Basic information of validation scenes.
        test (bool, optional): Whether use the test mode. In test mode, no
            annotations can be accessed. Default: False.
        max_sweeps (int, optional): Max number of sweeps. Default: 10.

    Returns:
        tuple[list[dict]]: Information of training set and validation set
            that will be saved to the info file.
    """
    train_nusc_infos = []
    val_nusc_infos = []
    root_path = Path("./data/sit_full/")

    iter_count = 0

    for path in mmcv.track_iter_progress(train_scenes):
        place = path.split("*")[0]
        scene = path.split("*")[1]
        frame = path.split("*")[2]
        token = path
        
        calib_path = root_path / place / scene / "calib" / "{}.txt".format(frame)
        if calib_path is not None:
            with open(calib_path, 'r') as f:
                lines = f.read().splitlines()
                P0_intrinsic = np.array([float(info) for info in lines[0].split(' ')[1:10]
                            ]).reshape([3, 3])
                P1_intrinsic = np.array([float(info) for info in lines[1].split(' ')[1:10]
                            ]).reshape([3, 3])
                P2_intrinsic = np.array([float(info) for info in lines[2].split(' ')[1:10]
                            ]).reshape([3, 3])
                P3_intrinsic = np.array([float(info) for info in lines[3].split(' ')[1:10]
                            ]).reshape([3, 3])
                P4_intrinsic = np.array([float(info) for info in lines[4].split(' ')[1:10]
                            ]).reshape([3, 3])
                P0_extrinsic = np.array([float(info) for info in lines[5].split(' ')[1:13]
                            ]).reshape([3, 4])
                P1_extrinsic = np.array([float(info) for info in lines[6].split(' ')[1:13]
                            ]).reshape([3, 4])
                P2_extrinsic = np.array([float(info) for info in lines[7].split(' ')[1:13]
                            ]).reshape([3, 4])
                P3_extrinsic = np.array([float(info) for info in lines[8].split(' ')[1:13]
                            ]).reshape([3, 4])
                P4_extrinsic = np.array([float(info) for info in lines[9].split(' ')[1:13]
                            ]).reshape([3, 4])
                P0_distortion = np.array([float(info) for info in lines[11].split(' ')[1:6]
                            ]).reshape([1, 5])
                P1_distortion = np.array([float(info) for info in lines[12].split(' ')[1:6]
                            ]).reshape([1, 5])
                P2_distortion = np.array([float(info) for info in lines[13].split(' ')[1:6]
                            ]).reshape([1, 5])
                P3_distortion = np.array([float(info) for info in lines[14].split(' ')[1:6]
                            ]).reshape([1, 5])
                P4_distortion = np.array([float(info) for info in lines[15].split(' ')[1:6]
                            ]).reshape([1, 5])
        # projection_matrix = [P0, P1, P2, P3, P4]
        projection_matrix = [ [P0_intrinsic, P0_extrinsic, P0_distortion], \
                              [P1_intrinsic, P1_extrinsic, P1_distortion], \
                              [P2_intrinsic, P2_extrinsic, P2_distortion], \
                              [P3_intrinsic, P3_extrinsic, P3_distortion], \
                              [P4_intrinsic, P4_extrinsic, P4_distortion], 
            ]
        velo_path = root_path / place / scene / "velo/concat/bin_data" / "{}.bin".format(frame)

        mmcv.check_file_exist(velo_path)

        

        # obtain 5 image's information per frame
        camera_types = [
            'CAM_FRONT',
            'CAM_FRONT_RIGHT',
            'CAM_FRONT_LEFT',
            # 'CAM_BACK',
            'CAM_BACK_LEFT',
            'CAM_BACK_RIGHT',
        ] # 5: front, 4: front_left, 3: front_right, 2: back_right, 1:back_right

        camera_mapping = {4:'CAM_FRONT', 1:'CAM_FRONT_LEFT', 5:'CAM_FRONT_RIGHT', 3:'CAM_BACK_RIGHT',  2:'CAM_BACK_LEFT'}

        cam_num_list = [1, 2, 3, 4, 5]
        anno_path_ = []
        odom_path_ = []
        cam_path_ = []
        for num in cam_num_list:
            cam_path_.append(root_path / place / scene / "cam_img/{}".format(num) / "data_undist" / "{}.png".format(frame))
        
        anno_path_.append(root_path / place / scene / "label_3d/{}.txt".format(frame))
        odom_path_.append(root_path / place / scene / "ego_trajectory/{}.txt".format(frame))

        '''
        info = {
            'lidar_path': velo_path,
                'token': token+"*"+str(cam),
                'sweeps': [],
                'cams': None,
        }
        '''
        
        # token을 이미지 한장이 아니라 전체 frame으로 통합   
        info = {
            'lidar_path': velo_path,
                'token': token,
                'sweeps': [],
                'cams': None,
        }
        

        # token 1개에 5개에 대응되는 image, annotation 정보 list 형태로 수집
        cam_info_list = []
        gt_boxes_list = []
        names_list = []
        data_anno_list = []
        valid_flag_list = []

        if not test:
                data_anno = {}
                for id, a_path in enumerate(anno_path_):
                    anno_ = get_label_anno(a_path)
                    if id == 0:
                        for key in anno_.keys():
                            data_anno[key] = anno_[key]
                    else:
                        for key in anno_.keys():
                            if key in ['bbox', 'dimensions', 'location']:
                                data_anno[key] = np.vstack((data_anno[key], anno_[key]))
                            else:
                                data_anno[key] = np.hstack((data_anno[key], anno_[key]))


        for num, cam in enumerate(cam_num_list):
            # chgd token
            '''
            info = {
                'lidar_path': velo_path,
                'token': token+"*"+str(cam),
                'sweeps': [],
                'cams': None,
            }
            
            cam_info = {'data_path':cam_path_[cam-1], 'type': camera_mapping[cam]}
            cam_info.update(cam_intrinsic=projection_matrix[cam-1])
            info['cams'] = cam_info
            '''
            
            cam_info_list.append({'data_path':cam_path_[cam-1], 'type': camera_mapping[cam], 'cam_intrinsic':projection_matrix[cam-1]})

            sweeps = []
            info['sweeps'] = sweeps


            locs = np.array(data_anno['location']).reshape(-1, 3)
            dims = np.array(data_anno['dimensions']).reshape(-1, 3) # order : l,w,h 
            rots = np.array(data_anno['rotation_y']).reshape(-1, 1)
            names = np.array(data_anno['name'])
            tokens = np.array(token)
            track_ids = np.array(data_anno['track_id'])

            gt_boxes = np.concatenate([locs, dims, rots], axis=1)

            ego_motion = get_ego_matrix(odom_path_[0])
            ego_yaw = rotmat_to_euler(ego_motion[:3, :3])[2]
            
                    
            gt_boxes[:, 6] -= ego_yaw
            comp_obj_center = np.matmul(np.linalg.inv(ego_motion), np.concatenate([gt_boxes[:,:3], np.ones(gt_boxes[:,:3].shape[0]).reshape(1, -1).T], axis=1).T).T
            gt_boxes[:, :3] = comp_obj_center[:, :3]

            gt_boxes_corners = box_center_to_corner_3d(gt_boxes[:, :3], gt_boxes[:, 3:6], gt_boxes[:, 6])
            points = np.fromfile(velo_path, dtype=np.float32, count=-1).reshape(-1, 4) # 4 : x, y, z, intensity
            num_pts_list = get_pts_in_3dbox_(points, gt_boxes_corners)
            projection_m = np.eye(4)
            projection_m[:3, :] = np.matmul(projection_matrix[cam-1][0], projection_matrix[cam-1][1])

            center3d = np.array(gt_boxes[:, :3])
            center3d = np.concatenate([center3d,  np.ones_like(center3d[:,:1])], axis=1)
            center2d = np.matmul(projection_m, center3d.T).T
            center2d[:,0] /= center2d[:,2]
            center2d[:,1] /= center2d[:,2] 
            
            depth_limit = center2d[:,2] > 0
            left_limit = center2d[:,0] >= 0
            right_limit = center2d[:,0] < 1920
            total_limit = depth_limit * left_limit * right_limit

            data_anno['num_lidar_pts'] = np.array(num_pts_list)[total_limit]
            valid_flag = (data_anno['num_lidar_pts'] > 0)



        # cams : multi-view camera images with camera parameter
        info['cams'] = cam_info_list
        info['gt_boxes'] = gt_boxes
        info['gt_names'] = names

        info['num_lidar_pts'] = data_anno['num_lidar_pts']
        info['valid_flag'] = valid_flag


        train_nusc_infos.append(info)

        iter_count += 1

    iter_count = 0
    for path in mmcv.track_iter_progress(val_scenes):
        place = path.split("*")[0]
        scene = path.split("*")[1]
        frame = path.split("*")[2]
        token = path
        
        calib_path = root_path / place / scene / "calib" / "{}.txt".format(frame)
        if calib_path is not None:
            with open(calib_path, 'r') as f:
                lines = f.read().splitlines()
                P0_intrinsic = np.array([float(info) for info in lines[0].split(' ')[1:10]
                            ]).reshape([3, 3])
                P1_intrinsic = np.array([float(info) for info in lines[1].split(' ')[1:10]
                            ]).reshape([3, 3])
                P2_intrinsic = np.array([float(info) for info in lines[2].split(' ')[1:10]
                            ]).reshape([3, 3])
                P3_intrinsic = np.array([float(info) for info in lines[3].split(' ')[1:10]
                            ]).reshape([3, 3])
                P4_intrinsic = np.array([float(info) for info in lines[4].split(' ')[1:10]
                            ]).reshape([3, 3])
                P0_extrinsic = np.array([float(info) for info in lines[5].split(' ')[1:13]
                            ]).reshape([3, 4])
                P1_extrinsic = np.array([float(info) for info in lines[6].split(' ')[1:13]
                            ]).reshape([3, 4])
                P2_extrinsic = np.array([float(info) for info in lines[7].split(' ')[1:13]
                            ]).reshape([3, 4])
                P3_extrinsic = np.array([float(info) for info in lines[8].split(' ')[1:13]
                            ]).reshape([3, 4])
                P4_extrinsic = np.array([float(info) for info in lines[9].split(' ')[1:13]
                            ]).reshape([3, 4])
                P0_distortion = np.array([float(info) for info in lines[11].split(' ')[1:6]
                            ]).reshape([1, 5])
                P1_distortion = np.array([float(info) for info in lines[12].split(' ')[1:6]
                            ]).reshape([1, 5])
                P2_distortion = np.array([float(info) for info in lines[13].split(' ')[1:6]
                            ]).reshape([1, 5])
                P3_distortion = np.array([float(info) for info in lines[14].split(' ')[1:6]
                            ]).reshape([1, 5])
                P4_distortion = np.array([float(info) for info in lines[15].split(' ')[1:6]
                            ]).reshape([1, 5])
        # projection_matrix = [P0, P1, P2, P3, P4]
        projection_matrix = [ [P0_intrinsic, P0_extrinsic, P0_distortion], \
                              [P1_intrinsic, P1_extrinsic, P1_distortion], \
                              [P2_intrinsic, P2_extrinsic, P2_distortion], \
                              [P3_intrinsic, P3_extrinsic, P3_distortion], \
                              [P4_intrinsic, P4_extrinsic, P4_distortion], 
            ]
        velo_path = root_path / place / scene / "velo/concat/bin_data" / "{}.bin".format(frame)

        mmcv.check_file_exist(velo_path)


        # obtain 5 image's information per frame
        camera_types = [
            'CAM_FRONT',
            'CAM_FRONT_RIGHT',
            'CAM_FRONT_LEFT',
            # 'CAM_BACK',
            'CAM_BACK_LEFT',
            'CAM_BACK_RIGHT',
        ]
        camera_mapping = {4:'CAM_FRONT', 1:'CAM_FRONT_LEFT', 5:'CAM_FRONT_RIGHT', 3:'CAM_BACK_RIGHT',  2:'CAM_BACK_LEFT'}

        cam_num_list = [1, 2, 3, 4, 5]
        anno_path_ = []
        odom_path_ = []
        cam_path_ = []
        for num in cam_num_list:

            cam_path_.append(root_path / place / scene / "cam_img/{}".format(num) / "data_undist" / "{}.png".format(frame))
        
        anno_path_.append(root_path / place / scene / "label_3d/{}.txt".format(frame))
        odom_path_.append(root_path / place / scene / "ego_trajectory/{}.txt".format(frame))

        # token 1개에 5개에 대응되는 image, annotation 정보 list 형태로 수집
        cam_info_list = []
        gt_boxes_list = []
        names_list = []
        data_anno_list = []
        valid_flag_list = []

        info = {
                'lidar_path': velo_path,
                'token': token,
                'sweeps': [],
                'cams': None,
            }

        if not test:
                data_anno = {}
                for id, a_path in enumerate(anno_path_):
                    anno_ = get_label_anno(a_path)
                    if id == 0:
                        for key in anno_.keys():
                            data_anno[key] = anno_[key]
                    else:
                        for key in anno_.keys():
                            if key in ['bbox', 'dimensions', 'location']:
                                data_anno[key] = np.vstack((data_anno[key], anno_[key]))
                            else:
                                data_anno[key] = np.hstack((data_anno[key], anno_[key]))

        for num, cam in enumerate(cam_num_list):
            '''
            info = {
                'lidar_path': velo_path,
                'token': token+"*"+str(cam),
                'sweeps': [],
                'cams': None,
            }
            '''
            
            cam_info_list.append({'data_path':cam_path_[cam-1], 'type': camera_mapping[cam], 'cam_intrinsic':projection_matrix[cam-1]})

            sweeps = []
            info['sweeps'] = sweeps

            locs = np.array(data_anno['location']).reshape(-1, 3)
            dims = np.array(data_anno['dimensions']).reshape(-1, 3) # order : l,w,h 
            rots = np.array(data_anno['rotation_y']).reshape(-1, 1)
            names = np.array(data_anno['name'])
            tokens = np.array(token)
            track_ids = np.array(data_anno['track_id'])

            gt_boxes = np.concatenate([locs, dims, rots], axis=1)

            ego_motion = get_ego_matrix(odom_path_[0])
            ego_yaw = rotmat_to_euler(ego_motion[:3, :3])[2]
            
                    
            gt_boxes[:, 6] -= ego_yaw
            comp_obj_center = np.matmul(np.linalg.inv(ego_motion), np.concatenate([gt_boxes[:,:3], np.ones(gt_boxes[:,:3].shape[0]).reshape(1, -1).T], axis=1).T).T
            gt_boxes[:, :3] = comp_obj_center[:, :3]

            gt_boxes_corners = box_center_to_corner_3d(gt_boxes[:, :3], gt_boxes[:, 3:6], gt_boxes[:, 6])
            points = np.fromfile(velo_path, dtype=np.float32, count=-1).reshape([-1, 4]) # 4 : x, y, z, intensity
            num_pts_list = get_pts_in_3dbox_(points, gt_boxes_corners)
            
            projection_m = np.eye(4)
            projection_m[:3, :] = np.matmul(projection_matrix[cam-1][0], projection_matrix[cam-1][1])
            intrins_m = np.eye(4) # chgd
            intrins_m[:3, :3] = projection_matrix[cam-1][0]
            extrins_m = np.linalg.inv(np.concatenate((projection_matrix[cam-1][1], np.concatenate((np.zeros(3), np.ones(1)))[None])))
            projection_m = np.matmul(intrins_m, extrins_m)
            center3d = np.array(gt_boxes[:, :3])
            center3d = np.concatenate([center3d,  np.ones_like(center3d[:,:1])], axis=1)
            center2d = np.matmul(projection_m, center3d.T).T
            center2d[:,0] /= center2d[:,2]
            center2d[:,1] /= center2d[:,2] 
            
            depth_limit = center2d[:,2] > 0
            left_limit = center2d[:,0] >= 0
            right_limit = center2d[:,0] < 1920
            total_limit = depth_limit * left_limit * right_limit

            data_anno['num_lidar_pts'] = np.array(num_pts_list)[total_limit]

            valid_flag = (data_anno['num_lidar_pts'] > 0)

            
        # cams : multi-view camera images with camera parameter
        info['cams'] = cam_info_list
        info['gt_boxes'] = gt_boxes
        info['gt_names'] = names

        info['num_lidar_pts'] = data_anno['num_lidar_pts']
        info['valid_flag'] = valid_flag

        val_nusc_infos.append(info)
        

        iter_count += 1

    return train_nusc_infos, val_nusc_infos


def box_center_to_corner_3d_(box_center):
    # To return
    translation = box_center[0:3]
    l, w, h = box_center[3], box_center[4], box_center[5]
    # rotation = -box_center[6] + np.pi
    rotation = box_center[6]
    # rotation = -box_center[6] + np.pi/2

    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
    z_corners = [h/2, h/2, h/2, h/2, -h/2, -h/2, -h/2, -h/2] #[0, 0, 0, 0, -h, -h, -h, -h]
    # z_corners = [-h/2, -h/2, -h/2, -h/2, h/2, h/2, h/2, h/2]
    bounding_box = np.vstack([x_corners, y_corners, z_corners])

    rotation_matrix = np.array([[np.cos(rotation),  -np.sin(rotation), 0],
                                [np.sin(rotation), np.cos(rotation), 0],
                                [0,  0,  1]])


    corner_box = (np.dot(rotation_matrix, bounding_box).T + translation).T

    return corner_box

def generate_record_(ann_rec: dict, x1: float, y1: float, x2: float, y2: float,
                    sample_data_token: str, filename: str) -> OrderedDict:
    """Generate one 2D annotation record given various information on top of
    the 2D bounding box coordinates.

    Args:
        ann_rec (dict): Original 3d annotation record.
        x1 (float): Minimum value of the x coordinate.
        y1 (float): Minimum value of the y coordinate.
        x2 (float): Maximum value of the x coordinate.
        y2 (float): Maximum value of the y coordinate.
        sample_data_token (str): Sample data token.
        filename (str):The corresponding image file where the annotation
            is present.

    Returns:
        dict: A sample 2D annotation record.
            - file_name (str): file name
            - image_id (str): sample data token
            - area (float): 2d box area
            - category_name (str): category name
            - category_id (int): category id
            - bbox (list[float]): left x, top y, dx, dy of 2d box
            - iscrowd (int): whether the area is crowd
    """
    
    repro_rec = OrderedDict()
    repro_rec['sample_data_token'] = sample_data_token
    coco_rec = dict()

    relevant_keys = [
        'attribute_tokens',
        'category_name',
        'instance_token',
        'next',
        'num_lidar_pts',
        'num_radar_pts',
        'prev',
        'sample_annotation_token',
        'sample_data_token',
        'visibility_token',
    ]

    for key, value in ann_rec.items():
        if key in relevant_keys:
            repro_rec[key] = value

    repro_rec['bbox_corners'] = [x1, y1, x2, y2]
    repro_rec['filename'] = filename[14:] #filename

    coco_rec['file_name'] = filename[14:] #filename
    coco_rec['image_id'] = sample_data_token
    coco_rec['area'] = (y2 - y1) * (x2 - x1)

    cat_name = ann_rec['attribute_name']
    coco_rec['category_name'] = cat_name
    coco_rec['category_id'] = nus_categories.index(cat_name)
    coco_rec['bbox'] = [x1, y1, x2 - x1, y2 - y1]
    coco_rec['iscrowd'] = 0

    return coco_rec


def export_2d_annotation(root_path, info_path, version, mono3d=True, test = False):
    """Export 2d annotation from the info file and raw data.

    Args:
        root_path (str): Root path of the raw data.
        info_path (str): Path of the info file.
        version (str): Dataset version.
        mono3d (bool, optional): Whether to export mono3d annotation.
            Default: True.
    """
    # get bbox annotations for camera
    camera_types = [
        'CAM_FRONT',
        'CAM_FRONT_RIGHT',
        'CAM_FRONT_LEFT',
        # 'CAM_BACK',
        'CAM_BACK_LEFT',
        'CAM_BACK_RIGHT',
    ]
    nusc_infos = mmcv.load(info_path)['infos']
    cat2Ids = [
        dict(id=nus_categories.index(cat_name), name=cat_name)
        for cat_name in nus_categories
    ]

    coco_ann_id = 0
    coco_2d_dict = dict(annotations=[], images=[], categories=cat2Ids)
    cur_frame = None

    for info in mmcv.track_iter_progress(nusc_infos):

        cam_info_list = info['cams']
        
        root_path = './data/sit_full'
        
        file_name_list = []
        cam_intrinsic_list = []
        coco_info_list = []
            
        height, width, _ = 1200, 1920, 3
        for num,cam_info in enumerate(cam_info_list):
            place = str(cam_info['data_path']).split("/")[-6]
            scene = str(cam_info['data_path']).split("/")[-5]
            frame = str(cam_info['data_path']).split("/")[-1].split(".")[0]

            velo_path = root_path +'/'+ place +'/'+ scene +'/'+ "velo/concat/bin_data/{}.bin".format(frame)
            
            points = np.fromfile(velo_path, dtype=np.float32, count=-1).reshape([-1, 4]) 
            
            if os.path.exists(root_path + "/" + place + "/" + scene + "/label_3d/{}.txt".format(frame)):
                coco_infos, label_, ego_motion, ego_yaw = get_2d_boxes_(
                    None,
                    cam_info,
                    info,
                    visibilities=['', '1', '2', '3', '4'],
                    mono3d=mono3d,
                    )
                
                coco_info_list.append(coco_infos)
  
                height, width, _ = 1200, 1920, 3
                locs = np.array(label_['location']).reshape(-1, 3)
                dims = np.array(label_['dimensions']).reshape(-1, 3) # order : l,w,h 
                rots = np.array(label_['rotation_y']).reshape(-1, 1)
                
                gt_boxes = np.concatenate([locs, dims, rots], axis=1)
                
                gt_names = label_['name']
            
                
                gt_boxes[:, 6] -= ego_yaw
                comp_obj_center = np.matmul(np.linalg.inv(ego_motion), np.concatenate([gt_boxes[:,:3], np.ones(gt_boxes[:,:3].shape[0]).reshape(1, -1).T], axis=1).T).T
                gt_boxes[:, :3] = comp_obj_center[:, :3]

                # coordinate is differenct between lidar and cam
                gt_boxes_corners = box_center_to_corner_3d(gt_boxes[:, :3], gt_boxes[:, 3:6], gt_boxes[:, 6])
                gt_center_3d_homo = np.concatenate((gt_boxes[:, :3], np.ones_like(gt_boxes[:,:1])),axis=1)
                gt_boxes[:, [3,4,5]] = gt_boxes[:, [3,5,4]] # lwh -> lhw
                
                        
                # chgd 5 별개로 전처리
                projection_m = np.eye(4)
                projection_m[:3, :3] = cam_info['cam_intrinsic'][0]
                intrins_m = np.eye(4) # chgd
                intrins_m[:3, :3] = cam_info['cam_intrinsic'][0]

                
                gt_boxes[:,:3] = np.matmul(cam_info['cam_intrinsic'][1], gt_center_3d_homo.T).T
                center3d = np.array(gt_boxes[:, :3])
                center3d = np.concatenate([center3d,  np.ones_like(center3d[:,:1])], axis=1)
                center2d = np.matmul(intrins_m, center3d.T).T
                center2d[:,0] /= center2d[:,2]
                center2d[:,1] /= center2d[:,2] 
                num_pts_list = get_pts_in_3dbox_(points, gt_boxes_corners)
                num_pts = np.array(num_pts_list)
                val_flag = num_pts >= 5
                center2d = center2d[val_flag]
                gt_boxes = gt_boxes[val_flag]
            
                depth_limit = center2d[:,2] > 0
                left_limit = center2d[:,0] >= 0
                right_limit = center2d[:,0] < 1920
                total_limit = depth_limit * left_limit * right_limit
                gt_boxes = gt_boxes[total_limit]
                
                # append
                gt_names = gt_names[val_flag]
                gt_names = gt_names[total_limit]


                 
            file_name_list.append(str(cam_info['data_path']))
            cam_intrinsic_list.append(cam_info['cam_intrinsic'])
            
        # all camera views to annotation
        if not test:
            coco_2d_dict['images'].append(
                dict(
                    file_name=file_name_list,
                    id=info['token'],
                    token=info['token'],
                    cam_intrinsic=cam_intrinsic_list,
                    width=width,
                    height=height,
                    gt_boxes=info['gt_boxes'],
                    gt_names=info['gt_names']
                    ))

        else:
            coco_2d_dict['images'].append(
                dict(
                    file_name=file_name_list,
                    id=info['token'],
                    token=info['token'],
                    cam_intrinsic=cam_intrinsic_list,
                    width=width,
                    height=height,
                    ))


        # append
        if len(coco_info_list) == 0:
            continue
        
        for coco_infos in coco_info_list:
            for coco_info in coco_infos:
                if coco_info is None:
                    continue
                # add an empty key for coco format
                coco_info['segmentation'] = []
                coco_info['id'] = coco_ann_id
                coco_2d_dict['annotations'].append(coco_info)
                coco_ann_id += 1

    if mono3d:
        json_prefix = f'{info_path[:-4]}_mono3d'
    else:
        json_prefix = f'{info_path[:-4]}'
    '''
    coco_2d_dict['images'] :5개의 매트릭스의 리스트
    file_name : camera image path list
    id = info token : place / scene / frame
    token = info token
    cam_intrinsic : camera intrinsic list
    width : width
    height : height
    gt_boxes : gt_boxes
    gt_names : gt_names
    annotations : coco_info list 
    [{'filename' : ,
          'image_id' : , # same as token
          'area' : ,
          'category_name' :,
          'category_id' : ,
          'bbox : ,
          'is_crowd' : ,
          'bbox_cam_3d' : ,
          'velo_cam_3d' : ,
          'centers2d' : ,
          'attribute_name' : ,
          'attribute_id' : ,
        }]
    '''
    
    mmcv.dump(coco_2d_dict, f'{json_prefix}.coco.json')


def get_2d_boxes_(nusc,
                 cam_info,
                 info,
                 visibilities: List[str],
                 mono3d=True,
                 vis_flag = False,
                 pointcloud = False,
                 ):
    """Get the 2D annotation records for a given `sample_data_token`.

    Args:
        sample_data_token (str): Sample data token belonging to a camera
            keyframe.
        visibilities (list[str]): Visibility filter.
        mono3d (bool): Whether to get boxes with mono3d annotation.

    Return:
        list[dict]: List of 2D annotation record that belongs to the input
            `sample_data_token`.
    """
    cam_num = str(cam_info['data_path']).split("/")[5]
    root_path = './data/sit_full'
    place = str(cam_info['data_path']).split("/")[-6]
    scene = str(cam_info['data_path']).split("/")[-5]
    frame = str(cam_info['data_path']).split("/")[-1].split(".")[0]
    
    
    label_path = root_path + "/" + place + "/" + scene + "/label_3d/{}.txt".format(frame)
    
    label_ = get_label_anno(label_path)
    velo_path = root_path +'/'+ place +'/'+ scene +'/'+ "velo/concat/bin_data/{}.bin".format(frame)
    points = np.fromfile(velo_path, dtype=np.float32, count=-1).reshape([-1, 4]) 
    odom_path = root_path +'/'+ place +'/'+ scene +'/'+ "ego_trajectory/{}.txt".format(frame)
    repro_recs = []
    
    ego_motion = get_ego_matrix(odom_path)
    ego_yaw = rotmat_to_euler(ego_motion[:3, :3])[2]

    for i in range(label_['name'].shape[0]):
        ann_rec = {}
        ann_rec['token'] = '{}*{}*{}*{}'.format(place, scene, frame, cam_num)
        ann_rec['sample_token'] = ann_rec['token']
        ann_rec['instance_token'] = ann_rec['token']
        ann_rec['visibility_token'] = 1 # label_['occluded'][num]
        ann_rec['attribute_tokens'] = ann_rec['token']
        ann_rec['translation'] = label_['location'][i].tolist()
        ann_rec['size'] = label_['dimensions'][i].tolist()
        ann_rec['rotation'] = list(Quaternion(axis=[0, 0, 1], radians=label_['rotation_y'][i]))
        ann_rec['prev'] = ""
        ann_rec['next'] = ""
        
        # for pts in 3dbox
        gt_boxes = np.array(ann_rec['translation'] + ann_rec['size'] + [label_['rotation_y'][i]])
        
            
        comp_obj_center = np.matmul(np.linalg.inv(ego_motion), np.concatenate([gt_boxes[:3], np.array([1])]).T).T
        gt_boxes[6] -= ego_yaw
        gt_boxes[:3] = comp_obj_center[:3]

        corners_3d = box_center_to_corner_3d_(gt_boxes) # lwh
        num_pts_list = get_pts_in_3dbox_(points, np.expand_dims(corners_3d.T, axis=0))
        ann_rec['num_lidar_pts'] = num_pts_list[0]

        
        ann_rec['attribute_name'] = label_['name'][i]
        ann_rec['attribute_id'] = label_['name'][i]

        #3d box projection
        intrinsic = cam_info['cam_intrinsic'][0]
        projection_m = np.eye(4)
        projection_m[:3, :] = np.matmul(intrinsic, cam_info['cam_intrinsic'][1])
        
        
        corners_3d_ = np.concatenate([corners_3d.T, np.ones(corners_3d.T[:,:3].shape[0]).reshape(1, -1).T], axis=1).T
        corners_2d = np.matmul(projection_m, corners_3d_).T
        corners_2d[:,0] /= corners_2d[:,2]
        corners_2d[:,1] /= corners_2d[:,2]
        
        intensity = points[:,3].copy()
        points[:,3] = 1
        p_corners_3d_ = points
        p_corners_2d=np.matmul(projection_m[None], p_corners_3d_[:,:,None]).squeeze()
        p_corners_2d[:,0] /= p_corners_2d[:,2]
        p_corners_2d[:,1] /= p_corners_2d[:,2]

        min_x, max_x = int(corners_2d[:, 0].min()), int(corners_2d[:, 0].max())
        min_y, max_y = int(corners_2d[:, 1].min()), int(corners_2d[:, 1].max())

        repro_rec = generate_record_(ann_rec, min_x, min_y, max_x, max_y,
                                    ann_rec['token'], str(cam_info['data_path']))

        # If mono3d=True, add 3D annotations in camera coordinates
        
        if mono3d and (repro_rec is not None):
            # 3dbox center projection
            center3d = np.array(gt_boxes[:3])
            center3d = np.concatenate([center3d,  np.array([1])])
            center2d = np.matmul(projection_m, center3d.T).T
            center2d[0] /= center2d[2]
            center2d[1] /= center2d[2]
            
            # chgd coordinate is differenct between lidar and cam
            gt_center_3d_homo = np.concatenate((gt_boxes[:3], np.ones_like([gt_boxes[0]])))
            gt_boxes[:3] = np.matmul(cam_info['cam_intrinsic'][1], gt_center_3d_homo)
            
            # yaw 값 변경
            roll, pitch, yaw = rotmat_to_euler(np.linalg.inv(cam_info['cam_intrinsic'][1][:3,:3]))
            gt_boxes[6] -= yaw
            gt_boxes[[3,4,5]] = gt_boxes[[3,5,4]] # lwh -> lhw
            
            repro_rec['bbox_cam3d'] = gt_boxes
            repro_rec['velo_cam3d'] = np.array([0, 0]).tolist()

            repro_rec['center2d'] = center2d.squeeze().tolist()
            # normalized center2D + depth
            # if samples with depth < 0 will be removed
            if repro_rec['center2d'][2] <= 0 or\
                repro_rec['center2d'][0] < 0 or\
                repro_rec['center2d'][0] > 1920:
                continue
            if vis_flag:
                img = draw_projected_box3d(img, corners_2d[:, :2], thickness=1)
                img = draw_projected_center(img, center2d)
                
            attr_id = nus_attributes.index(DefaultAttribute[ann_rec['attribute_name']])
            repro_rec['attribute_name'] = DefaultAttribute[ann_rec['attribute_name']]
            repro_rec['attribute_id'] = attr_id
            repro_recs.append(repro_rec)

    return repro_recs, label_, ego_motion, ego_yaw


def post_process_coords(
    corner_coords: List, imsize: Tuple[int, int] = (1600, 900)
) -> Union[Tuple[float, float, float, float], None]:
    """Get the intersection of the convex hull of the reprojected bbox corners
    and the image canvas, return None if no intersection.

    Args:
        corner_coords (list[int]): Corner coordinates of reprojected
            bounding box.
        imsize (tuple[int]): Size of the image canvas.

    Return:
        tuple [float]: Intersection of the convex hull of the 2D box
            corners and the image canvas.
    """
    polygon_from_2d_box = MultiPoint(corner_coords).convex_hull
    img_canvas = box(0, 0, imsize[0], imsize[1])

    if polygon_from_2d_box.intersects(img_canvas):
        img_intersection = polygon_from_2d_box.intersection(img_canvas)
        intersection_coords = np.array(
            [coord for coord in img_intersection.exterior.coords])

        min_x = min(intersection_coords[:, 0])
        min_y = min(intersection_coords[:, 1])
        max_x = max(intersection_coords[:, 0])
        max_y = max(intersection_coords[:, 1])

        return min_x, min_y, max_x, max_y
    else:
        return None


def generate_record(ann_rec: dict, x1: float, y1: float, x2: float, y2: float,
                    sample_data_token: str, filename: str) -> OrderedDict:
    """Generate one 2D annotation record given various information on top of
    the 2D bounding box coordinates.

    Args:
        ann_rec (dict): Original 3d annotation record.
        x1 (float): Minimum value of the x coordinate.
        y1 (float): Minimum value of the y coordinate.
        x2 (float): Maximum value of the x coordinate.
        y2 (float): Maximum value of the y coordinate.
        sample_data_token (str): Sample data token.
        filename (str):The corresponding image file where the annotation
            is present.

    Returns:
        dict: A sample 2D annotation record.
            - file_name (str): file name
            - image_id (str): sample data token
            - area (float): 2d box area
            - category_name (str): category name
            - category_id (int): category id
            - bbox (list[float]): left x, top y, dx, dy of 2d box
            - iscrowd (int): whether the area is crowd
    """
    repro_rec = OrderedDict()
    repro_rec['sample_data_token'] = sample_data_token
    coco_rec = dict()

    relevant_keys = [
        'attribute_tokens',
        'category_name',
        'instance_token',
        'next',
        'num_lidar_pts',
        'num_radar_pts',
        'prev',
        'sample_annotation_token',
        'sample_data_token',
        'visibility_token',
    ]

    for key, value in ann_rec.items():
        if key in relevant_keys:
            repro_rec[key] = value

    repro_rec['bbox_corners'] = [x1, y1, x2, y2]
    repro_rec['filename'] = filename

    coco_rec['file_name'] = filename
    coco_rec['image_id'] = sample_data_token
    coco_rec['area'] = (y2 - y1) * (x2 - x1)

    if repro_rec['category_name'] not in NuScenesDataset.NameMapping:
        return None
    cat_name = NuScenesDataset.NameMapping[repro_rec['category_name']]
    coco_rec['category_name'] = cat_name
    coco_rec['category_id'] = nus_categories.index(cat_name)
    coco_rec['bbox'] = [x1, y1, x2 - x1, y2 - y1]
    coco_rec['iscrowd'] = 0

    return coco_rec
