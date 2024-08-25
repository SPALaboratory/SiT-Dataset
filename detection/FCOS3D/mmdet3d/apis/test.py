# Copyright (c) OpenMMLab. All rights reserved.
from os import path as osp

import mmcv
import torch
from mmcv.image import tensor2imgs

from mmdet3d.models import (Base3DDetector, Base3DSegmentor,
                            SingleStageMono3DDetector)
from ..core import show_multi_modality_result

def single_gpu_test(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    show_score_thr=0.3,
                    show_bev=False,
                    **kwargs):
    """Test model with single gpu.

    This method tests model with single gpu and gives the 'show' option.
    By setting ``show=True``, it saves the visualization results under
    ``out_dir``.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        show (bool, optional): Whether to save viualization results.
            Default: True.
        out_dir (str, optional): The path to save visualization results.
            Default: None.

    Returns:
        list[dict]: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset)) # len(dataset) == 1799
    for i, data in enumerate(data_loader):
        # if i < 2000:
        #     try:
        #         batch_size = len(result)
        #     except:
        #         batch_size = 1
        #     for _ in range(batch_size):
        #         prog_bar.update()
        #     continue
        with torch.no_grad():
            # rescale = True 설정 시, camera projection 결과가 이상할 수 있음 -> 결과적으로 lidar에서 잘 되는지?
            result = model(return_loss=False, rescale=False, **data)
            #result = model(return_loss=False, rescale=False, **data)
        if show:
            # Visualize the results of MMDetection3D model
            # 'show_results' is MMdetection3D visualization API
            # models_3d = (Base3DDetector, Base3DSegmentor,
            #              SingleStageMono3DDetector)
            # if isinstance(model.module, models_3d):
            #     model.module.show_results(
            #         data,
            #         result,
            #         out_dir=out_dir,
            #         show=show,
            #         score_thr=show_score_thr,
            #         )
            # # Visualize the results of MMDetection model
            # # 'show_result' is MMdetection visualization API
            # else:
            #     batch_size = len(result)
            #     if batch_size == 1 and isinstance(data['img'][0],
            #                                       torch.Tensor):
            #         img_tensor = data['img'][0]
            #     else:
            #         img_tensor = data['img'][0].data[0]
            #     img_metas = data['img_metas'][0].data[0]
            #     imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            #     assert len(imgs) == 
            # (img_metas)

            #     for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
            #         h, w, _ = img_meta['img_shape']
            #         img_show = img[:h, :w, :]

            #         ori_h, ori_w = img_meta['ori_shape'][:-1]
            #         img_show = mmcv.imresize(img_show, (ori_w, ori_h))

            #         if out_dir:
            #             out_file = osp.join(out_dir, img_meta['ori_filename'])
            #         else:
            #             out_file = None

            #         model.module.show_result(
            #             img_show,
            #             result[i],
            #             show=show,
            #             out_file=out_file,
            #             score_thr=show_score_thr)
            pipeline = None
            pipeline = dataset._get_pipeline(pipeline)
            if 'img_bbox' in result[0].keys():
                result_bbox = result[0]['img_bbox']
            data_info = dataset.data_infos[i]
            img_path = data_info['file_name']
            file_name = osp.split(img_path)[-1].split('.')[0]
            cam_num = osp.basename(osp.dirname(osp.dirname(img_path)))
            img, img_metas = dataset._extract_data(i, pipeline,
                                                ['img', 'img_info'])
            # need to transpose channel to first dim
            try:
                img = img.numpy().transpose(1, 2, 0)
            except:
                pass
            pred_bboxes = result_bbox['boxes_3d']
            threshold = 0.2
            pred_bboxes = result_bbox['boxes_3d'][result_bbox['scores_3d'] > threshold]
            gt_bboxes = dataset.get_ann_info(i)['gt_bboxes_3d']
            show_multi_modality_result(
                data_info,
                img,
                gt_bboxes,
                pred_bboxes,
                torch.tensor(img_metas['cam_intrinsic'][0]),
                torch.tensor(img_metas['cam_intrinsic'][1]),
                out_dir,
                file_name,
                box_mode='camera',
                show=show,
                cam_num=cam_num,
                show_bev=show_bev)
            pass
        
        # if len(results)!=1:
        # breakpoint()
        results.extend(result)    
        # result = [result] 
        # results.append(result)

        batch_size = len(result) 

        # for multi-view, we should divide batch_size by number of camera
        for _ in range(batch_size):
            prog_bar.update()
            
    return results


def single_gpu_test_mv(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    show_score_thr=0.3,
                    show_bev=False,
                    **kwargs):
    """Test model with single gpu.

    This method tests model with single gpu and gives the 'show' option.
    By setting ``show=True``, it saves the visualization results under
    ``out_dir``.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        show (bool, optional): Whether to save viualization results.
            Default: True.
        out_dir (str, optional): The path to save visualization results.
            Default: None.

    Returns:
        list[dict]: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset)) # len(dataset) == 1799
    for i, data in enumerate(data_loader):
        # if i < 2000:
        #     try:
        #         batch_size = len(result)
        #     except:
        #         batch_size = 1
        #     for _ in range(batch_size):
        #         prog_bar.update()
        #     continue
        with torch.no_grad():
            result = model(return_loss=False, rescale=False, **data)
            #result = model(return_loss=False, rescale=False, **data)
            #breakpoint()
            
            # import copy
            # import cv2
            # import numpy as np

            
            # visualize pred box in bev
            '''
            image = np.zeros((80, 80, 3))
            image[39, 39, :] = [0, 0, 255]
            img_path = data['img_metas'][0].data[0][0]['filename'][0]

            for j in range(5):
                extrinsic_list = copy.deepcopy(data['img_metas'][0].data[0][0]['cam2img'][j][1])
                extrinsic_list.append([0.0, 0.0, 0.0, 1.0])
                extrinsic_matrix = np.array(extrinsic_list)
                rotation_matrix = extrinsic_matrix[:3, :3]
                translation_matrix = extrinsic_matrix[:3, 3]
                extrinsic_inv = np.eye(4)

                rotation_matrix_inv = copy.deepcopy(rotation_matrix).T
                translation_matrix_inv = -rotation_matrix_inv.dot(translation_matrix)

                extrinsic_inv[:3, :3] = rotation_matrix_inv
                extrinsic_inv[:3, 3] = translation_matrix_inv

                pred_box_center3d = copy.deepcopy(result[j]['img_bbox']['boxes_3d']).tensor.cpu().numpy()[:, :3]

                pred_box_center3d_homo = np.concatenate((pred_box_center3d, np.ones_like(pred_box_center3d[:,:1])), axis = -1)

                pred_global_box_xy = np.matmul(extrinsic_inv, pred_box_center3d_homo.T).T[:,:2]


                for k in range(len(pred_global_box_xy)):
                    x = int(pred_global_box_xy[k, 0]) + 39  
                    y = int(pred_global_box_xy[k, 1]) + 39
                    image[x, y, :] = [255, 0, 0]

            cv2.imwrite('./output_img/after_convert/{}_{}'.format(img_path.split('/')[3], img_path.split('/')[-1]), image)
            '''




            # import copy
            # import cv2
            # import numpy as np
            # from mmdet3d.core.bbox import points_cam2img, LiDARInstance3DBoxes, Box3DMode
            
            
            # for i in range(5):

            #     img_path = data['img_metas'][0].data[0][0]['filename'][i]

            #     img = cv2.imread(str('./' + img_path))
            #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            #     #img = cv2.resize(img, (960, 600))

            #     bbox_3d_list = copy.deepcopy(result[i]['img_bbox']['boxes_3d']) # CameraInstance3DBoxes


            #     cam2img = np.array(copy.deepcopy(data['img_metas'][0].data[0][0]['cam2img'][i][0]))

            #     cam2img[0][0] /= 0.5
            #     cam2img[0][2] /= 0.5
            #     cam2img[1][1] /= 0.5
            #     cam2img[1][2] /= 0.5

            #     for k in range(len(bbox_3d_list)):
            #         # 1. rescale projection check
            #         corners_3d = bbox_3d_list[k].corners  
            #         points_3d = corners_3d.reshape(-1, 3)
            #         if not isinstance(cam2img, torch.Tensor):
            #             cam2img = torch.from_numpy(np.array(cam2img))
            #         assert (cam2img.shape == torch.Size([3, 3]) or cam2img.shape == torch.Size([4, 4]))
            #         cam2img = cam2img.float().cpu()
            #         # project to 2d to get image coords (uv)
            #         uv_origin = points_cam2img(points_3d, cam2img)
            #         uv_origin = (uv_origin - 1).round()
            #         imgfov_pts_2d = uv_origin[..., :2].reshape(1, 8, 2).numpy()
            
            #         line_indices = ((0, 1), (0, 3), (0, 4), (1, 2), (1, 5), (3, 2), (3, 7),
            #             (4, 5), (4, 7), (2, 6), (5, 6), (6, 7))
            #         corners = imgfov_pts_2d[0].astype(np.int)
            #         for start, end in line_indices:
            #             cv2.line(img, (corners[start, 0], corners[start, 1]),
            #                     (corners[end, 0], corners[end, 1]), (255, 0, 0), 1,
            #                     cv2.LINE_AA)


            #         cv2.imwrite('./output_img/prediction/{}_{}_{}'.format(img_path.split('/')[3], img_path.split('/')[5], img_path.split('/')[-1]), img)





        # if show:
            # Visualize the results of MMDetection3D model
            # 'show_results' is MMdetection3D visualization API
            # models_3d = (Base3DDetector, Base3DSegmentor,
            #              SingleStageMono3DDetector)
            # if isinstance(model.module, models_3d):
            #     model.module.show_results(
            #         data,
            #         result,
            #         out_dir=out_dir,
            #         show=show,
            #         score_thr=show_score_thr,
            #         )
            # # Visualize the results of MMDetection model
            # # 'show_result' is MMdetection visualization API
            # else:
            #     batch_size = len(result)
            #     if batch_size == 1 and isinstance(data['img'][0],
            #                                       torch.Tensor):
            #         img_tensor = data['img'][0]
            #     else:
            #         img_tensor = data['img'][0].data[0]
            #     img_metas = data['img_metas'][0].data[0]
            #     imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            #     assert len(imgs) == 
            # (img_metas)

            #     for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
            #         h, w, _ = img_meta['img_shape']
            #         img_show = img[:h, :w, :]

            #         ori_h, ori_w = img_meta['ori_shape'][:-1]
            #         img_show = mmcv.imresize(img_show, (ori_w, ori_h))

            #         if out_dir:
            #             out_file = osp.join(out_dir, img_meta['ori_filename'])
            #         else:
            #             out_file = None

            #         model.module.show_result(
            #             img_show,
            #             result[i],
            #             show=show,
            #             out_file=out_file,
            #             score_thr=show_score_thr)
            # pipeline = None
            # pipeline = dataset._get_pipeline(pipeline)
            # if 'img_bbox' in result[0].keys():
            #     result_bbox = result[0]['img_bbox']
            # data_info = dataset.data_infos[i]
            # img_path = data_info['file_name']
            # file_name = osp.split(img_path)[-1].split('.')[0]
            # cam_num = osp.basename(osp.dirname(osp.dirname(img_path)))
            # img, img_metas = dataset._extract_data(i, pipeline,
            #                                     ['img', 'img_info'])
            # # need to transpose channel to first dim
            # try:
            #     img = img.numpy().transpose(1, 2, 0)
            # except:
            #     pass
            # pred_bboxes = result_bbox['boxes_3d']
            # threshold = 0.2
            # pred_bboxes = result_bbox['boxes_3d'][result_bbox['scores_3d'] > threshold]
            # gt_bboxes = dataset.get_ann_info(i)['gt_bboxes_3d']
            # show_multi_modality_result(
            #     data_info,
            #     img,
            #     gt_bboxes,
            #     pred_bboxes,
            #     torch.tensor(img_metas['cam_intrinsic'][0]),
            #     torch.tensor(img_metas['cam_intrinsic'][1]),
            #     out_dir,
            #     file_name,
            #     box_mode='camera',
            #     show=show,
            #     cam_num=cam_num,
            #     show_bev=show_bev)
            # pass
        
        # result = model output = image 별 대응되는 예측값들
        # if len(results)!=1:
        # breakpoint()
        #results.extend(result)
        #breakpoint()    
        result = [result] 
        results.append(result)

        batch_size = len(result) 

        # for multi-view, we should divide batch_size by number of camera
        for _ in range(batch_size):
            prog_bar.update()
            
    return results