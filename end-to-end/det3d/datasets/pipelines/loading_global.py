import os.path as osp
import warnings
import numpy as np
from functools import reduce

import pycocotools.mask as maskUtils

from pathlib import Path
from copy import deepcopy
from det3d import torchie
from det3d.core import box_np_ops
import pickle 
import os 
from ..registry import PIPELINES
import pdb
from det3d.utils.simplevis import *

def _dict_select(dict_, inds):
    for k, v in dict_.items():
        if isinstance(v, dict):
            _dict_select(v, inds)
        else:
            dict_[k] = v[inds]

def read_file(path, tries=2, num_point_feature=4, painted=False):
    if painted:
        dir_path = os.path.join(*path.split('/')[:-2], 'painted_'+path.split('/')[-2])
        painted_path = os.path.join(dir_path, path.split('/')[-1]+'.npy')
        points =  np.load(painted_path)
        points = points[:, [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]] # remove ring_index from features 
    else:
        try:
            points = np.fromfile(path, dtype=np.float32).reshape(-1, 5)[:, :num_point_feature]
        except:
            points = np.fromfile(path, dtype=np.float32).reshape(-1, 4)[:, :num_point_feature]

    return points


def remove_close(points, radius: float) -> None:
    """
    Removes point too close within a certain radius from origin.
    :param radius: Radius below which points are removed.
    """
    x_filt = np.abs(points[0, :]) < radius
    y_filt = np.abs(points[1, :]) < radius
    not_close = np.logical_not(np.logical_and(x_filt, y_filt))
    points = points[:, not_close]
    return points


def read_sweep(sweep, painted=False):
    min_distance = 1.0
    points_sweep = read_file(str(sweep["lidar_path"]), painted=painted).T
    points_sweep = remove_close(points_sweep, min_distance)

    nbr_points = points_sweep.shape[1]
    if sweep["transform_matrix"] is not None:
        points_sweep[:3, :] = sweep["transform_matrix"].dot(
            np.vstack((points_sweep[:3, :], np.ones(nbr_points)))
        )[:3, :]
    curr_times = sweep["time_lag"] * np.ones((1, points_sweep.shape[1]))

    return points_sweep.T, curr_times.T

def read_single_waymo(obj):
    points_xyz = obj["lidars"]["points_xyz"]
    points_feature = obj["lidars"]["points_feature"]

    # normalize intensity 
    points_feature[:, 0] = np.tanh(points_feature[:, 0])

    points = np.concatenate([points_xyz, points_feature], axis=-1)
    
    return points 

def read_single_waymo_sweep(sweep):
    obj = get_obj(sweep['path'])

    points_xyz = obj["lidars"]["points_xyz"]
    points_feature = obj["lidars"]["points_feature"]

    # normalize intensity 
    points_feature[:, 0] = np.tanh(points_feature[:, 0])
    points_sweep = np.concatenate([points_xyz, points_feature], axis=-1).T # 5 x N

    nbr_points = points_sweep.shape[1]

    if sweep["transform_matrix"] is not None:
        points_sweep[:3, :] = sweep["transform_matrix"].dot( 
            np.vstack((points_sweep[:3, :], np.ones(nbr_points)))
        )[:3, :]

    curr_times = sweep["time_lag"] * np.ones((1, points_sweep.shape[1]))
    
    return points_sweep.T, curr_times.T


def get_obj(path):
    with open(path, 'rb') as f:
            obj = pickle.load(f)
    return obj 


@PIPELINES.register_module
class LoadPointCloudFromFile(object):
    def __init__(self, dataset="KittiDataset", **kwargs):
        self.type = dataset
        self.random_select = kwargs.get("random_select", False)
        self.npoints = kwargs.get("npoints", 16834)

    def __call__(self, res, info):

        res["type"] = self.type

        if self.type == "NuScenesDataset":

            nsweeps = res["lidar"]["nsweeps"]

            lidar_path = Path(info["lidar_path"])
            points = read_file(str(lidar_path), painted=res["painted"])

            sweep_points_list = [points]
            sweep_times_list = [np.zeros((points.shape[0], 1))]

            assert (nsweeps - 1) == len(
                info["sweeps"]
            ), "nsweeps {} should equal to list length {}.".format(
                nsweeps, len(info["sweeps"])
            )

            rng = np.random.default_rng(0)
            for i in rng.choice(len(info["sweeps"]), nsweeps - 1, replace=False):
                sweep = info["sweeps"][i]
                points_sweep, times_sweep = read_sweep(sweep, painted=res["painted"])
                sweep_points_list.append(points_sweep)
                sweep_times_list.append(times_sweep)
            
            points = np.concatenate(sweep_points_list, axis=0)
            times = np.concatenate(sweep_times_list, axis=0).astype(points.dtype)

            res["lidar"]["points"] = points
            res["lidar"]["times"] = times
            res["lidar"]["combined"] = np.hstack([points, times])
            
        elif self.type == "WaymoDataset":
            path = info['path']
            nsweeps = res["lidar"]["nsweeps"]
            obj = get_obj(path)
            points = read_single_waymo(obj)
            res["lidar"]["points"] = points

            if nsweeps > 1: 
                sweep_points_list = [points]
                sweep_times_list = [np.zeros((points.shape[0], 1))]

                assert (nsweeps - 1) == len(
                    info["sweeps"]
                ), "nsweeps {} should be equal to the list length {}.".format(
                    nsweeps, len(info["sweeps"])
                )

                for i in range(nsweeps - 1):
                    sweep = info["sweeps"][i]
                    points_sweep, times_sweep = read_single_waymo_sweep(sweep)
                    sweep_points_list.append(points_sweep)
                    sweep_times_list.append(times_sweep)

                points = np.concatenate(sweep_points_list, axis=0)
                times = np.concatenate(sweep_times_list, axis=0).astype(points.dtype)

                res["lidar"]["points"] = points
                res["lidar"]["times"] = times
                res["lidar"]["combined"] = np.hstack([points, times])
        elif self.type == "SPA_Nus_Dataset":

            nsweeps = res["lidar"]["nsweeps"]

            lidar_path = Path(info["lidar_path"])
            points = read_file(str(lidar_path), painted=res["painted"])

            sweep_points_list = [points]
            sweep_times_list = [np.zeros((points.shape[0], 1))]

            # assert (nsweeps - 1) == len(
            #     info["sweeps"]
            # ), "nsweeps {} should equal to list length {}.".format(
            #     nsweeps, len(info["sweeps"])
            # )

            # rng = np.random.default_rng(0)
            # for i in rng.choice(len(info["sweeps"]), nsweeps - 1, replace=False):
            #     sweep = info["sweeps"][i]
            #     points_sweep, times_sweep = read_sweep(sweep, painted=res["painted"])
            #     sweep_points_list.append(points_sweep)
            #     sweep_times_list.append(times_sweep)
            
            points = np.concatenate(sweep_points_list, axis=0)
            times = np.concatenate(sweep_times_list, axis=0).astype(points.dtype)

            res["lidar"]["points"] = points
            res["lidar"]["times"] = times
            res["lidar"]["combined"] = np.hstack([points, times])
        else:
            raise NotImplementedError

        return res, info


@PIPELINES.register_module
class LoadPointCloudAnnotations(object):
    def __init__(self, with_bbox=True, **kwargs):
        pass

    def __call__(self, res, info):

        if res["type"] in ["NuScenesDataset"] and "gt_boxes" in info:
            gt_boxes = info["gt_boxes"].astype(np.float32)
            gt_boxes[np.isnan(gt_boxes)] = 0

            boxes, names, tokens, rtokens, velocity, rvelocity, trajectory = [], [], [], [], [], [], []

            for i in range(res["metadata"]["timesteps"]):
                try:
                    boxes.append(gt_boxes[:,i,:])
                    names.append(info["gt_names"][:,i])
                    tokens.append(info["gt_boxes_token"][:,i])
                    rtokens.append(info["gt_boxes_rtoken"][:,i])
                    velocity.append(info["gt_boxes_velocity"][:,i,:].astype(np.float32))
                    rvelocity.append(info["gt_boxes_rvelocity"][:,i,:].astype(np.float32))
                    trajectory.append(info["gt_trajectory"][:,i])

                except:
                    print("No Annotations in Scene")
                    boxes.append(gt_boxes)
                    names.append(info["gt_names"])
                    tokens.append(info["gt_boxes_token"])
                    rtokens.append(info["gt_boxes_rtoken"])
                    velocity.append(info["gt_boxes_velocity"].astype(np.float32))
                    rvelocity.append(info["gt_boxes_rvelocity"].astype(np.float32))
                    trajectory.append(info["gt_trajectory"])


            res["lidar"]["annotations"] = {
                "boxes": boxes,
                "names": names,
                "tokens": tokens,
                "rtokens": rtokens,
                "velocities": velocity,
                "rvelocities" : rvelocity,
                "trajectory" : trajectory,
                "bev" : info["bev"] if "bev" in info else np.zeros((1, 180,180))
            }

        elif res["type"] == 'WaymoDataset' and "gt_boxes" in info:
            res["lidar"]["annotations"] = {
                "boxes": info["gt_boxes"].astype(np.float32),
                "names": info["gt_names"],
            }
        
        elif res["type"] in ["SPA_Nus_Dataset"] and "gt_boxes" in info:
            gt_boxes = info["gt_boxes"].astype(np.float32)
            gt_tokens = info['gt_boxes_token']


            gt_boxes[np.isnan(gt_boxes)] = 0


            boxes, names, tokens, rtokens, velocity, rvelocity, trajectory, track_id = [], [], [], [], [], [], [], []

            for i in range(res["metadata"]["timesteps"]):
                vis_flag = False
                if vis_flag:
                    token_ = info["gt_boxes_token"][:,i][0]
                    place = token_.split("*")[0]
                    scene = token_.split("*")[1]
                    frame = token_.split("*")[2]
                    save_path = "/home/changwon/detection_task/Det3D/viz_in_model/preprocessing/"
                    os.makedirs(save_path, exist_ok=True)

                    bbox_list = gt_boxes[:,i,[0,1,2,3,4,5,-2]] #x,y,z,l,w,h,rot
                    velo_path = "./data/spa/{}/{}/velo/concat/bin_data/{}.bin".format(place, scene, frame)
                    points = np.fromfile(velo_path, dtype=np.float32, count=-1).reshape([-1, 4])

                    pred_boxes = bbox_list
                    point = points
                    #pred_boxes[:, 6] *= -1
                    bev = nuscene_vis(point, pred_boxes)
                    cv2.imwrite(save_path+"pred_{}*{}*{}-{}.png".format(place, scene, frame, i), bev)
                
                gt_token
                
                try:
                    boxes.append(gt_boxes[:,i,:])
                    names.append(info["gt_names"][:,i])
                    tokens.append(info["gt_boxes_token"][:,i])
                    rtokens.append(info["gt_boxes_rtoken"][:,i])
                    velocity.append(info["gt_boxes_velocity"][:,i,:].astype(np.float32))
                    rvelocity.append(info["gt_boxes_rvelocity"][:,i,:].astype(np.float32))
                    trajectory.append(info["gt_trajectory"][:,i])
                    track_id.append(info['gt_track_id'][:, i])

                except:
                    print("No Annotations in Scene")
                    boxes.append(gt_boxes)
                    names.append(info["gt_names"])
                    tokens.append(info["gt_boxes_token"])
                    rtokens.append(info["gt_boxes_rtoken"])
                    velocity.append(info["gt_boxes_velocity"].astype(np.float32))
                    rvelocity.append(info["gt_boxes_rvelocity"].astype(np.float32))
                    trajectory.append(info["gt_trajectory"])
                    track_id.append(info['gt_track_id'])


            res["lidar"]["annotations"] = {
                "boxes": boxes,
                "names": names,
                "tokens": tokens,
                "rtokens": rtokens,
                "velocities": velocity,
                "rvelocities" : rvelocity,
                "trajectory" : trajectory,
                "track_id" : track_id,
                "bev" : info["bev"] if "bev" in info else np.zeros((1, 180,180))
            }

        else:
            pass 

        return res, info