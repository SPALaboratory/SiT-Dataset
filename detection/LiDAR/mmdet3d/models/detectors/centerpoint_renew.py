# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmdet3d.core import bbox3d2result, merge_aug_bboxes_3d
from ..builder import DETECTORS
from .mvx_two_stage import MVXTwoStageDetector
from mmdet3d.utils.simplevis import *
import cv2
import numpy as np
import os

@DETECTORS.register_module()
class CenterPoint(MVXTwoStageDetector):
    """Base class of Multi-modality VoxelNet."""

    def __init__(self,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(CenterPoint,
              self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained, init_cfg)

    @property
    def with_velocity(self):
        """bool: Whether the head predicts velocity"""
        return self.pts_bbox_head is not None and \
            self.pts_bbox_head.with_velocity

    def extract_pts_feat(self, pts, img_feats, img_metas):
        """Extract features of points."""
        if not self.with_pts_bbox:
            return None
        voxels, num_points, coors = self.voxelize(pts)

        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0] + 1
        x = self.pts_middle_encoder(voxel_features, coors, batch_size)
        x = self.pts_backbone(x)
        if self.with_pts_neck:
            x = self.pts_neck(x)
        return x

    def forward_pts_train(self,
                          pts_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore=None):
        """Forward function for point cloud branch.

        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.

        Returns:
            dict: Losses of each branch.
        """
        outs = self.pts_bbox_head(pts_feats)
        
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs)
        
        vis_flag = False
        scene_cnt = 0

        if vis_flag:
            # breakpoint()
            bbox_list = self.pts_bbox_head.get_bboxes(
                outs, img_metas, rescale=False)
            # bbox_results = [
            #     bbox3d2result(bboxes, scores, labels)
            #     for bboxes, scores, labels in bbox_list
            # ]

            save_path = "./visualize_output_bev_gt_pred/range_512/"
            os.makedirs(save_path, exist_ok=True)

            for batch in range(len(bbox_list)):
                rng = save_path.split("_")[5].split("/")[0]
                velo_path = str(img_metas[batch]['pts_filename'])
                idx = img_metas[batch]['sample_idx']
                place = idx.split("*")[0]
                scene = idx.split("*")[1]
                frame = idx.split("*")[-1]
                

                points = np.fromfile(velo_path, dtype=np.float32, count=-1).reshape([-1, 4])


                pred_boxes = bbox_list[batch][0].tensor.detach().clone().cpu().numpy()
                point = points
                pred_boxes[:, 6] *= -1
                bev_pred = nuscene_vis_pred(point, pred_boxes)
                # cv2.imwrite(save_path+"{}*{}*{}_pred.png".format(place, scene, frame), bev_pred)
                # breakpoint()
                bev_gt = nuscene_vis_gt(point, gt_bboxes_3d[batch].tensor.detach().clone().cpu().numpy()) # GT bbox의 BEV map
                # cv2.imwrite(save_path+"{}*{}*{}_gt.png".format(place, scene, frame), bev_gt)

                
                # bev_gt의 값이 [0,255,0]이고 bev_pred의 값이 [255,0,0]이면 해당 값을 가져오도록 조건 설정
                bev_integrated = np.where(
                    np.all(bev_gt == [0, 255, 0], axis=2)[:, :, None], [0, 255, 0],  # bev_gt의 조건
                    np.where(
                        np.all(bev_pred == [255, 0, 0], axis=2)[:, :, None], [255, 0, 0],  # bev_pred의 조건
                        bev_pred  # bev_pred의 조건에 해당하지 않는 경우 bev_pred의 값 선택
                    )
                )
                
                cv2.imwrite(save_path+"{}_{}_{}_{}_intrgrated.png".format(rng, place, scene, frame), bev_integrated)
            

        return losses
    
    # def forward_test(self, points, img_metas, gt_bboxes_3d, gt_labels_3d,img=None, **kwargs):
    #     """
    #     Args:
    #         points (list[torch.Tensor]): the outer list indicates test-time
    #             augmentations and inner torch.Tensor should have a shape NxC,
    #             which contains all points in the batch.
    #         img_metas (list[list[dict]]): the outer list indicates test-time
    #             augs (multiscale, flip, etc.) and the inner list indicates
    #             images in a batch
    #         img (list[torch.Tensor], optional): the outer
    #             list indicates test-time augmentations and inner
    #             torch.Tensor should have a shape NxCxHxW, which contains
    #             all images in the batch. Defaults to None.
    #     """
    #     img_feats, pts_feats = self.extract_feats(points, img_metas, img)
    #     pts_feats = pts_feats[0]
    #     # pts_feats = torch.stack(pts_feats, dim=0)
    #     outs = self.pts_bbox_head(pts_feats)
       
    #     vis_flag = True

    #     if vis_flag:
    #         # breakpoint()
    #         bbox_list = self.pts_bbox_head.get_bboxes(
    #             outs, img_metas, rescale=False)
    #         # bbox_results = [
    #         #     bbox3d2result(bboxes, scores, labels)
    #         #     for bboxes, scores, labels in bbox_list
    #         # ]
    #         # breakpoint()
    #         save_path = "./visualize_output_bev_gt_pred/range_512/"
    #         os.makedirs(save_path, exist_ok=True)

    #         for batch in range(len(bbox_list)):
    #             rng = save_path.split("_")[5].split("/")[0]
    #             velo_path = str(img_metas[batch][0]['pts_filename'])
    #             idx = img_metas[batch][0]['sample_idx']
    #             place = idx.split("*")[0]
    #             scene = idx.split("*")[1]
    #             frame = idx.split("*")[-1]
                

    #             # points = np.fromfile(velo_path, dtype=np.float32, count=-1).reshape([-1, 4])


    #             pred_boxes = bbox_list[batch][0].tensor.detach().clone().cpu().numpy()
    #             point = points[0][0]
    #             point = point.detach().clone().cpu().numpy()
    #             pred_boxes[:, 6] *= -1
    #             # breakpoint()
    #             bev_pred = nuscene_vis_pred(point, pred_boxes)
    #             # cv2.imwrite(save_path+"{}*{}*{}_pred.png".format(place, scene, frame), bev_pred)
    #             # breakpoint()
    #             bev_gt = nuscene_vis_gt(point, gt_bboxes_3d[0][0][batch].tensor.detach().clone().cpu().numpy()) # GT bbox의 BEV map
    #             # cv2.imwrite(save_path+"{}*{}*{}_gt.png".format(place, scene, frame), bev_gt)

                
    #             # bev_gt의 값이 [0,255,0]이고 bev_pred의 값이 [255,0,0]이면 해당 값을 가져오도록 조건 설정
    #             bev_integrated = np.where(
    #                 np.all(bev_gt == [0, 255, 0], axis=2)[:, :, None], [0, 255, 0],  # bev_gt의 조건
    #                 np.where(
    #                     np.all(bev_pred == [255, 0, 0], axis=2)[:, :, None], [255, 0, 0],  # bev_pred의 조건
    #                     bev_pred  # bev_pred의 조건에 해당하지 않는 경우 bev_pred의 값 선택
    #                 )
    #             )
                
    #             cv2.imwrite(save_path+"{}_{}_{}_{}_intrgrated.png".format(rng, place, scene, frame), bev_integrated)

        # for var, name in [(points, 'points'), (img_metas, 'img_metas')]:
        #     if not isinstance(var, list):
        #         raise TypeError('{} must be a list, but got {}'.format(
        #             name, type(var)))

        # num_augs = len(points)
        # if num_augs != len(img_metas):
        #     raise ValueError(
        #         'num of augmentations ({}) != num of image meta ({})'.format(
        #             len(points), len(img_metas)))

        # if num_augs == 1:
        #     img = [img] if img is None else img
        #     return self.simple_test(points[0], img_metas[0], img[0], **kwargs)
        # else:
        #     return self.aug_test(points, img_metas, img, **kwargs)




    def simple_test_pts(self, x, img_metas, rescale=False):
        """Test function of point cloud branch."""
        # outs = self.pts_bbox_head(x[0])
        outs = self.pts_bbox_head(x) # base
        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]

        vis_flag = False
        # import pdb; pdb.set_trace()
        if vis_flag:
            # save_path = "/home/changwon/detection_task/mmdetection3d/viz_in_model/pred/"
            save_path = '/home/cwkang/data/ROS/for_rebuttle_viz/pred_1101/'
            import os
            os.makedirs(save_path, exist_ok=True)
            for batch in range(len(bbox_list)):
                idx = img_metas[batch]['sample_idx']

                pred_boxes = bbox_list[batch][0].tensor.detach().clone().cpu().numpy()
                score = bbox_list[batch][1].detach().clone().cpu().numpy()
                cls = bbox_list[batch][2].detach().clone().cpu().numpy()

                txt_list = np.hstack([pred_boxes, score.reshape(pred_boxes.shape[0],1)])

                with open(save_path + "{}.txt".format(idx), "w") as file:
                    for c in txt_list:
                        file.write("{} {} {} {} {} {} {} {}".format(c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]) + "\n")


        return bbox_results

    def aug_test_pts(self, feats, img_metas, rescale=False):
        """Test function of point cloud branch with augmentaiton.

        The function implementation process is as follows:

            - step 1: map features back for double-flip augmentation.
            - step 2: merge all features and generate boxes.
            - step 3: map boxes back for scale augmentation.
            - step 4: merge results.

        Args:
            feats (list[torch.Tensor]): Feature of point cloud.
            img_metas (list[dict]): Meta information of samples.
            rescale (bool, optional): Whether to rescale bboxes.
                Default: False.

        Returns:
            dict: Returned bboxes consists of the following keys:

                - boxes_3d (:obj:`LiDARInstance3DBoxes`): Predicted bboxes.
                - scores_3d (torch.Tensor): Scores of predicted boxes.
                - labels_3d (torch.Tensor): Labels of predicted boxes.
        """
        # only support aug_test for one sample
        outs_list = []
        for x, img_meta in zip(feats, img_metas):
            outs = self.pts_bbox_head(x)
            # outs = self.pts_bbox_head(x[0])
            # merge augmented outputs before decoding bboxes
            for task_id, out in enumerate(outs):
                for key in out[0].keys():
                    if img_meta[0]['pcd_horizontal_flip']:
                        outs[task_id][0][key] = torch.flip(
                            outs[task_id][0][key], dims=[2])
                        if key == 'reg':
                            outs[task_id][0][key][:, 1, ...] = 1 - outs[
                                task_id][0][key][:, 1, ...]
                        elif key == 'rot':
                            outs[task_id][0][
                                key][:, 0,
                                     ...] = -outs[task_id][0][key][:, 0, ...]
                        elif key == 'vel':
                            outs[task_id][0][
                                key][:, 1,
                                     ...] = -outs[task_id][0][key][:, 1, ...]
                    if img_meta[0]['pcd_vertical_flip']:
                        outs[task_id][0][key] = torch.flip(
                            outs[task_id][0][key], dims=[3])
                        if key == 'reg':
                            outs[task_id][0][key][:, 0, ...] = 1 - outs[
                                task_id][0][key][:, 0, ...]
                        elif key == 'rot':
                            outs[task_id][0][
                                key][:, 1,
                                     ...] = -outs[task_id][0][key][:, 1, ...]
                        elif key == 'vel':
                            outs[task_id][0][
                                key][:, 0,
                                     ...] = -outs[task_id][0][key][:, 0, ...]

            outs_list.append(outs)

        preds_dicts = dict()
        scale_img_metas = []

        # concat outputs sharing the same pcd_scale_factor
        for i, (img_meta, outs) in enumerate(zip(img_metas, outs_list)):
            pcd_scale_factor = img_meta[0]['pcd_scale_factor']
            if pcd_scale_factor not in preds_dicts.keys():
                preds_dicts[pcd_scale_factor] = outs
                scale_img_metas.append(img_meta)
            else:
                for task_id, out in enumerate(outs):
                    for key in out[0].keys():
                        preds_dicts[pcd_scale_factor][task_id][0][key] += out[
                            0][key]

        aug_bboxes = []

        for pcd_scale_factor, preds_dict in preds_dicts.items():
            for task_id, pred_dict in enumerate(preds_dict):
                # merge outputs with different flips before decoding bboxes
                for key in pred_dict[0].keys():
                    preds_dict[task_id][0][key] /= len(outs_list) / len(
                        preds_dicts.keys())
            bbox_list = self.pts_bbox_head.get_bboxes(
                preds_dict, img_metas[0], rescale=rescale)
            bbox_list = [
                dict(boxes_3d=bboxes, scores_3d=scores, labels_3d=labels)
                for bboxes, scores, labels in bbox_list
            ]
            aug_bboxes.append(bbox_list[0])

        if len(preds_dicts.keys()) > 1:
            # merge outputs with different scales after decoding bboxes
            merged_bboxes = merge_aug_bboxes_3d(aug_bboxes, scale_img_metas,
                                                self.pts_bbox_head.test_cfg)
            return merged_bboxes
        else:
            for key in bbox_list[0].keys():
                bbox_list[0][key] = bbox_list[0][key].to('cpu')
            return bbox_list[0]

    def aug_test(self, points, img_metas, imgs=None, rescale=False):
        """Test function with augmentaiton."""
        img_feats, pts_feats = self.extract_feats(points, img_metas, imgs)
        bbox_list = dict()
        if pts_feats and self.with_pts_bbox:
            pts_bbox = self.aug_test_pts(pts_feats, img_metas, rescale)
            bbox_list.update(pts_bbox=pts_bbox)
        return [bbox_list]
