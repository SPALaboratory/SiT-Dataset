from ..registry import DETECTORS
from .single_stage import SingleStageDetector
from copy import deepcopy 
import os
import cv2
from det3d.utils.simplevis import *

@DETECTORS.register_module
class PointPillars(SingleStageDetector):
    def __init__(
        self,
        reader,
        backbone,
        neck,
        bbox_head,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
    ):
        super(PointPillars, self).__init__(
            reader, backbone, neck, bbox_head, train_cfg, test_cfg, pretrained
        )

    def extract_feat(self, data):
        input_features = self.reader(
            data["features"], data["num_voxels"], data["coors"]
        )
        x = self.backbone(
            input_features, data["coors"], data["batch_size"], data["input_shape"]
        )
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward(self, example, return_loss=True, **kwargs):
        voxels = example["voxels"]
        coordinates = example["coordinates"]
        num_points_in_voxel = example["num_points"]
        num_voxels = example["num_voxels"]

        batch_size = len(num_voxels)

        data = dict(
            features=voxels,
            num_voxels=num_points_in_voxel,
            coors=coordinates,
            batch_size=batch_size,
            input_shape=example["shape"][0],
        )

        x = self.extract_feat(data)
        preds = self.bbox_head(x)

        if return_loss:
            loss = self.bbox_head.loss(example, preds)
            vis_flag = False
            #import pdb; pdb.set_trace()
            if vis_flag:
                temp = self.bbox_head.predict(example.copy(), preds.copy(), self.test_cfg)
                
                save_path = "/home/changwon/detection_task/Det3D/viz_in_model/pred/"
                os.makedirs(save_path, exist_ok=True)
                bbox_list = temp[0]['box3d_lidar'].detach().clone().cpu().numpy()
                scores = temp[0]['scores'].detach().clone().cpu().numpy()
                mask = (scores[np.argsort(scores)[::-1]] >0.5)
                bbox_list = bbox_list[np.argsort(scores)[mask]]
                place = temp[0]['metadata']['token'].split("*")[0]
                scene = temp[0]['metadata']['token'].split("*")[1]
                frame = temp[0]['metadata']['token'].split("*")[2]
                velo_path = "./data/spa/{}/{}/velo/concat/bin_data/{}.bin".format(place, scene, frame)
                points = np.fromfile(velo_path, dtype=np.float32, count=-1).reshape([-1, 4])

                pred_boxes = bbox_list #torch.cat([xs, ys, batch_hei, batch_dim, batch_vel, batch_rot], dim=2)
                point = points
                #pred_boxes[:, 8] *= -1
                pred_boxes[:, [0,1,2,3,4,5,8,6,7]] #locs, dims, rots, velos
                #pred_boxes[:, 6] = -pred_boxes[:, 6] * np.pi/2
                bev = nuscene_vis(point, pred_boxes)
                cv2.imwrite(save_path+"pred_{}*{}*{}.png".format(place, scene, frame), bev)

                save_path = "/home/changwon/detection_task/Det3D/viz_in_model/gt/"
                os.makedirs(save_path, exist_ok=True)
                for num in range(len(example['gt_boxes_and_cls'])):
                    if num ==  0:
                        bbox_list = example['gt_boxes_and_cls'][num][0][:, [0,1,2,3,4,5,6]].detach().clone().cpu().numpy()
                        velo_path = "./data/spa/{}/{}/velo/concat/bin_data/{}.bin".format(place, scene, frame)
                        points = np.fromfile(velo_path, dtype=np.float32, count=-1).reshape([-1, 4])
                        gt_boxes = bbox_list
                        point = points
                        #gt_boxes[:, 6] *= -1
                        #gt_boxes[:, 6] = -gt_boxes[:, 6] * np.pi/2
                        bev = nuscene_vis(point, gt_boxes)
                        cv2.imwrite(save_path+"gt_{}*{}*{}-{}.png".format(place, scene, frame, num), bev)
            # return self.bbox_head.loss(example, preds)
            return loss
        else:
            return self.bbox_head.predict(example, preds, self.test_cfg)

    def forward_two_stage(self, example, return_loss=True, **kwargs):
        voxels = example["voxels"]
        coordinates = example["coordinates"]
        num_points_in_voxel = example["num_points"]
        num_voxels = example["num_voxels"]

        batch_size = len(num_voxels)

        data = dict(
            features=voxels,
            num_voxels=num_points_in_voxel,
            coors=coordinates,
            batch_size=batch_size,
            input_shape=example["shape"][0],
        )

        x = self.extract_feat(data)
        bev_feature = x 
        preds = self.bbox_head(x)

        # manual deepcopy ...
        new_preds = []
        for pred in preds:
            new_pred = {} 
            for k, v in pred.items():
                new_pred[k] = v.detach()

            new_preds.append(new_pred)

        boxes = self.bbox_head.predict(example, new_preds, self.test_cfg)

        if return_loss:
            return boxes, bev_feature, self.bbox_head.loss(example, preds)
        else:
            return boxes, bev_feature, None 