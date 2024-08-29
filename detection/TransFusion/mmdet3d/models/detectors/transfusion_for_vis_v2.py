import mmcv
import torch
from mmcv.parallel import DataContainer as DC
from mmcv.runner import force_fp32
from os import path as osp
from torch import nn as nn
from torch.nn import functional as F
from mmdet3d.utils.simplevis import *

from mmdet3d.core import (Box3DMode, Coord3DMode, bbox3d2result,
                          merge_aug_bboxes_3d, show_result)
from mmdet3d.ops import Voxelization
from mmdet.core import multi_apply
from mmdet.models import DETECTORS
from .. import builder
from .mvx_two_stage import MVXTwoStageDetector
import cv2
from itertools import chain
from tracemalloc import get_object_traceback
import numpy as np
from numpy.lib.histograms import histogram_bin_edges
from pyrsistent import v
from skimage.util.dtype import img_as_int
import seaborn as sns
import matplotlib.pyplot as plt
import os
from skimage import io
from matplotlib.lines import Line2D


@DETECTORS.register_module()
class TransFusionDetector(MVXTwoStageDetector):
    """Base class of Multi-modality VoxelNet."""

    def __init__(self, **kwargs):
        super(TransFusionDetector, self).__init__(**kwargs)

        self.freeze_img = kwargs.get('freeze_img', True)
        self.init_weights(pretrained=kwargs.get('pretrained', None))

    def init_weights(self, pretrained=None):
        """Initialize model weights."""
        super(TransFusionDetector, self).init_weights(pretrained)

        if self.freeze_img:
            if self.with_img_backbone:
                for param in self.img_backbone.parameters():
                    param.requires_grad = False
            if self.with_img_neck:
                for param in self.img_neck.parameters():
                    param.requires_grad = False

    def extract_img_feat(self, img, img_metas):
        """Extract features of images."""
        if self.with_img_backbone and img is not None:
            input_shape = img.shape[-2:]
            # update real input shape of each single img
            for img_meta in img_metas:
                img_meta.update(input_shape=input_shape)

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_(0)
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.view(B * N, C, H, W)
            img_feats = self.img_backbone(img.float())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
        return img_feats

    def extract_pts_feat(self, pts, img_feats, img_metas):
        """Extract features of points."""
        if not self.with_pts_bbox:
            return None
        voxels, num_points, coors = self.voxelize(pts)
        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors,
                                                )
        batch_size = coors[-1, 0] + 1
        x = self.pts_middle_encoder(voxel_features, coors, batch_size)
        x = self.pts_backbone(x)
        if self.with_pts_neck:
            x = self.pts_neck(x)
        return x

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        """Apply dynamic voxelization to points.

        Args:
            points (list[torch.Tensor]): Points of each sample.

        Returns:
            tuple[torch.Tensor]: Concatenated points, number of points
                per voxel, and coordinates.
        """
        voxels, coors, num_points = [], [], []
        for res in points:
            res_voxels, res_coors, res_num_points = self.pts_voxel_layer(res)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return voxels, num_points, coors_batch

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      proposals=None,
                      gt_bboxes_ignore=None):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        img_feats, pts_feats = self.extract_feat(
            points, img=img, img_metas=img_metas)
        losses = dict()
        if pts_feats:
            losses_pts = self.forward_pts_train(pts_feats, img_feats, gt_bboxes_3d,
                                                gt_labels_3d, img_metas,
                                                gt_bboxes_ignore)
            losses.update(losses_pts)
        if img_feats:
            losses_img = self.forward_img_train(
                img_feats,
                img_metas=img_metas,
                gt_bboxes=gt_bboxes,
                gt_labels=gt_labels,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposals=proposals)
            losses.update(losses_img)
        return losses

    def forward_pts_train(self,
                          pts_feats,
                          img_feats,
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
        outs = self.pts_bbox_head(pts_feats, img_feats, img_metas)
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs)

        # def get_ego_matrix(label_path):
        #     with open(label_path, 'r') as f:
        #         lines = f.readlines()
        #     return np.array([float(i) for i in lines[0].split(",")]).reshape(4, 4)

        # def rotmat_to_euler(rot_mat):
        #     """
        #     Convert rotation matrix to Euler angles using X-Y-Z convention
        #     :param rot_mat: 3x3 rotation matrix
        #     :return: array-like object of Euler angles (in radians)
        #     """
        #     sy = np.sqrt(rot_mat[0, 0] * rot_mat[0, 0] + rot_mat[1, 0] * rot_mat[1, 0])
        #     singular = sy < 1e-6
        #     if not singular:
        #         roll = np.arctan2(rot_mat[2, 1], rot_mat[2, 2])
        #         pitch = np.arctan2(-rot_mat[2, 0], sy)
        #         yaw = np.arctan2(rot_mat[1, 0], rot_mat[0, 0])
        #     else:
        #         roll = np.arctan2(-rot_mat[1, 2], rot_mat[1, 1])
        #         pitch = np.arctan2(-rot_mat[2, 0], sy)
        #         yaw = 0
        #     return np.array([roll, pitch, yaw])

        def box_center_to_corner_3d_(box_center):
            # To return
            translation = box_center[:,0:3]
            l, w, h = box_center[:,3], box_center[:,4], box_center[:,5]
            
            # rotation = -box_center[6] + np.pi
            rotation = box_center[:,6]
            # rotation = -box_center[6] + np.pi/2
            # breakpoint()
            x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
            y_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
            z_corners = [-h/2, -h/2, -h/2, -h/2, h/2, h/2, h/2, h/2] #[0, 0, 0, 0, -h, -h, -h, -h]
            # z_corners = [0, 0, 0, 0, h, h, h, h]
            bounding_box = np.vstack([x_corners, y_corners, z_corners])

            rotation_matrix = np.array([[np.cos(rotation),  -np.sin(rotation), 0],
                                        [np.sin(rotation), np.cos(rotation), 0],
                                        [0,  0,  1]])


            corner_box = (np.dot(rotation_matrix, bounding_box).T + translation).T

            return corner_box
        
        def interpolate_line(p1, p2, min_val, max_val):
            """
            두 점 p1, p2 사이의 직선 상에서 min_val과 max_val 사이의 점을 찾는 함수.
            """
            if p2 == p1:
                return p1
            ratio = (min_val - p1) / (p2 - p1)
            ratio = np.clip(ratio, 0, 1)
            return p1 + ratio * (p2 - p1)

        def clip_and_interpolate(points_2d, img_width, img_height):
            """
            2D 좌표를 이미지 경계 내로 클리핑 및 보간하는 함수.
            이미지 경계를 넘어가는 경우 보간된 점들과 원래 점들을 모두 포함하여 반환한다.
            """
            new_points = points_2d.copy().T.tolist()  # 원래 점들을 리스트로 초기화

            for i in range(points_2d.shape[1]):
                p1 = points_2d[:, i]
                for j in range(i + 1, points_2d.shape[1]):
                    p2 = points_2d[:, j]

                    # x축 방향으로 보간
                    if (p1[0] < 0 and p2[0] >= 0) or (p1[0] > img_width and p2[0] <= img_width):
                        new_x = 0 if p1[0] < 0 else img_width
                        new_y = interpolate_line(p1[1], p2[1], p1[0], new_x)
                        new_points.append([new_x, new_y])

                    # y축 방향으로 보간
                    if (p1[1] < 0 and p2[1] >= 0) or (p1[1] > img_height and p2[1] <= img_height):
                        new_y = 0 if p1[1] < 0 else img_height
                        new_x = interpolate_line(p1[0], p2[0], p1[1], new_y)
                        new_points.append([new_x, new_y])

            return np.array(new_points).T
        
        def draw_projected_box3d(image, qs, color=(255,255,255), thickness=2):
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
            # color = (255, 158, 0)
            color = (0, 158, 255)
            qs = qs.astype(np.int32)
            for k in range(0,4):
                # Ref: http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
                i,j=k,(k+1)%4
                # use LINE_AA for opencv3
                cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.LINE_AA)

                i,j=k+4,(k+1)%4 + 4
                cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.LINE_AA)

                i,j=k,k+4
                cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.LINE_AA)
            return image

        def draw_center(image, qs):
            # breakpoint()
            qs = qs.astype(np.int32)
            for point in qs:
                x, y = point
                # breakpoint()
                cv2.circle(image, (x,y), 5, (0,255,0),-1)

            return image
        
        def draw_center_near(image, qs):
            # breakpoint()
            qs = qs.astype(np.int32)
            for point in qs:
                x, y = point
                # breakpoint()
                cv2.circle(image, (x,y), 5, (0,0,255),-1)

            return image

        def get_box_corners(self, center, size, rotation):
            l, w, h = size / 2.0
            x_corners = [l, l, -l, -l, l, l, -l, -l]
            y_corners = [w, -w, -w, w, w, -w, -w, w]
            z_corners = [h, h, h, h, -h, -h, -h, -h]
            corners = np.vstack([x_corners, y_corners, z_corners])
            
            # Apply rotation
            R = np.array([[np.cos(rotation), -np.sin(rotation), 0],
                        [np.sin(rotation), np.cos(rotation), 0],
                        [0, 0, 1]])
            corners = np.dot(R, corners)
            
            # Apply translation
            corners = corners + np.array(center).reshape(3, 1)
            
            return corners.T

        vis_flag = False # only visualization in batch size = 1

        if vis_flag:
            while isinstance(gt_bboxes_3d, list):
                gt_bboxes_3d = gt_bboxes_3d[0]
            # breakpoint()
            gt_bboxes_3d = gt_bboxes_3d.tensor.detach().clone().cpu().numpy() # convert to array
            # breakpoint()

            bbox_list = self.pts_bbox_head.get_bboxes(
                outs, img_metas, rescale=False)
            # breakpoint()
            save_path = "./data/spa/gt_projection_augX/"
            data_path = "./data/spa/"
            for batch in range(len(bbox_list)):
                velo_path = str(img_metas[batch]['pts_filename'])
                idx = img_metas[batch]['sample_idx']
                place = idx.split("*")[0]
                scene = idx.split("*")[1]
                frame = idx.split("*")[-1]
                img_list = [] # 해당 BEV map에 이용된 카메라뷰들(1~5)
                for img in img_metas[batch]['filename']:
                    img_list.append(str(img))
                # breakpoint()
                
                
                gt_bboxes_3d_center = gt_bboxes_3d[:,:3]
                gt_bboxes_3d_center = torch.tensor(gt_bboxes_3d_center)
                # # beside_gt_bboxes_3d_center = gt_bboxes_3d_center.astype(np.int64)
                # # corners_3d = box_center_to_corner_3d_(gt_boxes.squeeze())
                # points = np.fromfile(velo_path, dtype=np.float32, count=-1).reshape([-1, 4])
                # point = points
                # num_points = len(gt_bboxes_3d)
                
                # bev = nuscene_vis(point, gt_bboxes_3d) # GT bbox의 BEV map

                img_scale_factor = (
                    img_metas[batch]['scale_factor'][:2]
                                        if 'scale_factor' in img_metas[batch].keys() else [1.0, 1.0])
                                        
                img_flip = img_metas[batch]['flip'] if 'flip' in img_metas[batch].keys() else False
                img_crop_offset = (
                    img_metas[batch]['img_crop_offset']
                    if 'img_crop_offset' in img_metas[batch].keys() else 0)
                img_shape = img_metas[batch]['img_shape'][:2]
                img_pad_shape = img_metas[batch]['input_shape'][:2]
                
                for view_idx in range(5):
                    # img = np.array(io.imread(img_list[view_idx]), dtype=np.int32)
                    # breakpoint()
                    img = cv2.imread(img_list[view_idx])
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    # # Show image before drawing boxes
                    # plt.imshow(img)
                    # plt.title(f"Original Image")
                    # plt.axis("off")
                    # plt.show()

                    # gt_bboxes = []

                    # for box in gt_bboxes_3d:
                    #     center = box[:3]
                    #     size = box[3:6]
                    #     rotation = box[6]
                    #     corners = self.get_box_corners(center, size, rotation)
                    #     gt_bboxes.append(corners)

                    # for corners in gt_bboxes:
                    #     pass

                    num_points = gt_bboxes_3d.shape[0]
                    lidar2img_rt = torch.tensor(img_metas[batch]['lidar2img'][view_idx])
                    gt_bboxes_3d_center_4d = torch.cat([gt_bboxes_3d_center, gt_bboxes_3d_center.new_ones(size=(num_points, 1))], dim=-1)
                    # gt_bboxes_3d_center_4d = gt_bboxes_3d_center_4d.float()
                    gt_bboxes_3d_center_2d = gt_bboxes_3d_center_4d @ lidar2img_rt.float().t()
                    # breakpoint()
                    gt_bboxes_3d_center_2d[:, 2] = torch.clamp(gt_bboxes_3d_center_2d[:, 2], min=1e-5)
                    gt_bboxes_3d_center_2d[:, 0] /= gt_bboxes_3d_center_2d[:, 2]
                    gt_bboxes_3d_center_2d[:, 1] /= gt_bboxes_3d_center_2d[:, 2]
                    '''
                    # gt박스를 위한 코너 점들
                    corners_3d = box_center_to_corner_3d_(gt_bboxes_3d.squeeze())
                    corners_2d = corners_3d @ lidar2img_rt.float().t()

                    corners_2d[:, 2] = torch.clamp(corners_2d[:, 2], min=1e-5)
                    corners_2d[:, 0] /= corners_2d[:, 2]
                    corners_2d[:, 1] /= corners_2d[:, 2]
                    '''
                    # 중심점에서 약간 떨어진 좌표
                    gt_bboxes_3d_center_4d_near = gt_bboxes_3d_center_4d.float()
                    gt_bboxes_3d_center_2d_near = gt_bboxes_3d_center_4d_near @ lidar2img_rt.float().t()
                    # breakpoint()
                    gt_bboxes_3d_center_2d_near[:, 2] = torch.clamp(gt_bboxes_3d_center_2d_near[:, 2], min=1e-5)
                    gt_bboxes_3d_center_2d_near[:, 0] /= gt_bboxes_3d_center_2d_near[:, 2]
                    gt_bboxes_3d_center_2d_near[:, 1] /= gt_bboxes_3d_center_2d_near[:, 2]
                    
                    # img transformation: scale -> crop -> flip
                    # the image is resized by img_scale_factor
                    # breakpoint()
                    img_scale_factor = torch.tensor(img_scale_factor)
                    img_coors = gt_bboxes_3d_center_2d[:, 0:2] * img_scale_factor  # Nx2
                    img_coors -= img_crop_offset

                    img_coors_near = gt_bboxes_3d_center_2d_near[:, 0:2] * img_scale_factor  # Nx2
                    img_coors_near -= img_crop_offset

                    # grid sample, the valid grid range should be in [-1,1]
                    coor_x, coor_y = torch.split(img_coors, 1, dim=1)  # each is Nx1

                    coor_x_near, coor_y_near = torch.split(img_coors_near, 1, dim=1)  # each is Nx1

                
                    # breakpoint()
                    if img_flip:
                        # by default we take it as horizontal flip
                        # use img_shape before padding for flip
                        orig_h, orig_w = img_shape
                        coor_x = orig_w - coor_x

                        coor_x_near = orig_w - coor_x_near

                    coor_x, coor_corner_x = coor_x[0:len(gt_bboxes_3d_center_2d), :], coor_x[len(gt_bboxes_3d_center_2d):, :]
                    coor_y, coor_corner_y = coor_y[0:len(gt_bboxes_3d_center_2d), :], coor_y[len(gt_bboxes_3d_center_2d):, :]
                    # coor_corner_x = coor_corner_x.reshape(len(gt_bboxes_3d_center_2d), 8, 1)
                    # coor_corner_y = coor_corner_y.reshape(len(gt_bboxes_3d_center_2d), 8, 1)
                    # coor_corner_xy = torch.cat([coor_corner_x, coor_corner_y], dim=-1)
                    coor_x_near, coor_corner_x_near = coor_x_near[0:len(gt_bboxes_3d_center_2d_near), :], coor_x_near[len(gt_bboxes_3d_center_2d_near):, :]
                    coor_y_near, coor_corner_y_near = coor_y_near[0:len(gt_bboxes_3d_center_2d_near), :], coor_y_near[len(gt_bboxes_3d_center_2d_near):, :]
                    # breakpoint()
                    h, w = img_pad_shape
                    # h, w = 448,717
                    re_img = cv2.resize(img, (w, h))
                    on_the_image = (coor_x > 0) * (coor_x < w) * (coor_y > 0) * (coor_y < h)
                    on_the_image = on_the_image.squeeze()

                    on_the_image_near = (coor_x_near > 0) * (coor_x_near < w) * (coor_y_near > 0) * (coor_y_near < h)
                    on_the_image_near = on_the_image_near.squeeze()
                    # skip the following computation if no object query fall on current image
                    # if on_the_image.sum() <= 1:
                    #     continue
                    # on_the_image_mask[sample_idx, on_the_image] = view_idx
                    # if (corners_2d[:, 2] < 0).sum() >0:
                    #     continue

                    gt_bboxes_3d_center_2d = gt_bboxes_3d_center_2d.numpy()
                    img_coors = img_coors.numpy()

                    gt_bboxes_3d_center_2d_near = gt_bboxes_3d_center_2d_near.numpy()
                    img_coors_near = img_coors_near.numpy()
                    # breakpoint()

                    # img_ = draw_projected_box3d(re_img, img_coors)
                    img_ = draw_center(re_img, img_coors) # gt center는 초록색
                    # img_ = draw_projected_box3d(img_, corners_2d[:, :2])
                    
                    _img_ = draw_center_near(img_, img_coors_near) 
                    
                    # if on_the_image.sum() > 1:
                    #     img_ = draw_center(re_img, img_coors) 
                    # else:
                    #     img_ = re_img
                    a = cv2.imwrite(save_path + "{}_{}_cam_img_{}".format(place, scene, view_idx+1) + f'{frame}.png', _img_)
                    # a = cv2.imwrite(save_path + "{}_{}_cam_img_{}".format(place, scene, view_idx+1) + f'{frame}.png', img_)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

                    print(" {} - {} - {} - {}".format(place, scene, view_idx+1, frame))

                # exit()        

                # breakpoint()

        return losses

    def simple_test_pts(self, x, x_img, img_metas, rescale=False):
        """Test function of point cloud branch."""
        outs = self.pts_bbox_head(x, x_img, img_metas)
        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results

    def simple_test(self, points, img_metas, img=None, rescale=False):
        """Test function without augmentaiton."""
        img_feats, pts_feats = self.extract_feat(
            points, img=img, img_metas=img_metas)

        bbox_list = [dict() for i in range(len(img_metas))]
        if pts_feats and self.with_pts_bbox:
            bbox_pts = self.simple_test_pts(
                pts_feats, img_feats, img_metas, rescale=rescale)
            for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
                result_dict['pts_bbox'] = pts_bbox
        if img_feats and self.with_img_bbox:
            bbox_img = self.simple_test_img(
                img_feats, img_metas, rescale=rescale)
            for result_dict, img_bbox in zip(bbox_list, bbox_img):
                result_dict['img_bbox'] = img_bbox
        return bbox_list
