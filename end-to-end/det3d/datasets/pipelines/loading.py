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
            points = np.fromfile(path, dtype=np.float32, count=-1).reshape(-1, 5)[:, :num_point_feature]
        except:
            points = np.fromfile(path, dtype=np.float32, count=-1).reshape(-1, 4)[:, :num_point_feature]

    return points

def filter_points_in_range_np(points, range_):
    """
    Filters and returns points within a specified range using NumPy.
    
    Parameters:
        points (np.array): A NumPy array of points in [x, y, z, intensity] format.
        range_ (list or np.array): A list or np.array specifying the range in [xmin, ymin, zmin, xmax, ymax, zmax] format.
        
    Returns:
        np.array: A NumPy array containing only the points within the specified range.
    """
    # Ensure the input range_ is a numpy array for efficient calculations.
    range_np = np.asarray(range_)
    xmin, ymin, zmin, xmax, ymax, zmax = range_np
    
    # Create masks for each condition.
    mask_x = np.logical_and(points[:, 0] >= xmin, points[:, 0] <= xmax)
    mask_y = np.logical_and(points[:, 1] >= ymin, points[:, 1] <= ymax)
    mask_z = np.logical_and(points[:, 2] >= zmin, points[:, 2] <= zmax)
    
    # Combine the masks with logical AND, and use it to index the input points array.
    mask = np.logical_and.reduce((mask_x, mask_y, mask_z))
    
    return points[mask]

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

def read_sweep_sit(sweep, painted=False):
    min_distance = 1.0
    points_sweep = read_file(sweep["lidar_path"], painted=painted).T
    points_sweep = remove_close(points_sweep, min_distance)
    N = points_sweep.shape[1]

    nbr_points = points_sweep.shape[1]
    # if sweep["transform_matrix"] is not None:
    #     points_sweep[:3, :] = sweep["transform_matrix"].dot(
    #         np.vstack((points_sweep[:3, :], np.ones(nbr_points)))
    #     )[:3, :]
    curr_times = sweep["time_lag"].split("_")[0]
    if curr_times == '0':
        time_ = np.zeros((N, 1)) 
    else:
        time_ = np.ones((N, 1)) 

    return points_sweep.T, np.ones((N, 1))

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

def get_ego_matrix(label_path):
    with open(label_path, 'r') as f:
        lines = f.readlines()
    return np.array([float(i) for i in lines[0].split(",")]).reshape(4, 4)

@PIPELINES.register_module
class LoadPointCloudFromFile(object):
    def __init__(self, dataset="KittiDataset", **kwargs):
        self.type = dataset
        self.random_select = kwargs.get("random_select", False)
        self.npoints = kwargs.get("npoints", 16834)

    def __call__(self, res, info):

        res["type"] = self.type
        # pdb.set_trace()
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
        elif self.type == "SiT_Dataset":
            pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
            nsweeps = res["lidar"]["nsweeps"]

            lidar_path = Path(info["lidar_path"])

            sweep_points_list = []
            sweep_times_list = []

            present_loc = info["sweeps"][0]['transform_matrix'][:3, -1]
            for i in range(10):
                sweep = deepcopy(info["sweeps"][i])
                points_sweep, times_sweep = read_sweep_sit(sweep, painted=res["painted"])
                points_sweep = np.fromfile(sweep['lidar_path'], dtype=np.float32, count=-1).reshape(-1, 4)
                points_sweep = filter_points_in_range_np(points_sweep, pc_range)
                
                times_sweep = np.ones((points_sweep.shape[0], 1))
                points_sweep_ = np.concatenate([deepcopy(points_sweep[:, :3]), np.ones(points_sweep.shape[0]).reshape(1, -1).T], axis=1).T
                    
                ego_motion = deepcopy(info["sweeps"][i]['transform_matrix'])
                prev_loc = info["sweeps"][i]['transform_matrix'][:3, -1]
                ego_rot = np.eye(4)
                ego_rot[:3, :3] = ego_motion[:3, :3]
                
                # ego_rot = ego_motion
                ego_rot[:3, -1] = -1 * (present_loc - prev_loc)
                points_sweep_ = deepcopy(np.matmul(ego_rot, points_sweep_).T)
                points_sweep[:, :3] = points_sweep_[:, :3]
                # points_sweep[:, :3] = points_sweep_[:, :3] + first_loc.T

                sweep_points_list.append(points_sweep)
                sweep_times_list.append(times_sweep)
            # ============================================================
            
            points = np.concatenate(sweep_points_list, axis=0)
            # points[:, :3] -= first_loc.T
            times = np.concatenate(sweep_times_list, axis=0).astype(points.dtype)

                
            place = info['token'].split("*")[0]
            scene = info['token'].split("*")[1]
            frame = info['token'].split("*")[2]
            root_path = Path("./data/sit/")
            
            odom_path = root_path / place / scene / "ego_trajectory/{}.txt".format(frame)
            ego_motion = get_ego_matrix(odom_path)
            ego_rot = np.eye(4)
            ego_rot[:3, :3] = np.linalg.inv(ego_motion[:3, :3])
            # ego_rot[:3, :3] = ego_motion[:3, :3]

            points_ = np.concatenate([points[:, :3], np.ones(points.shape[0]).reshape(1, -1).T], axis=1).T
            points_ = np.matmul(ego_rot, points_).T
            points[:, :3] = points_[:, :3]
            # points[:, :3] = points_.T[:, :3]
            
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
        
        elif res["type"] in ["SiT_Dataset"] and "gt_boxes" in info:
            try:
                gt_boxes = info["gt_boxes"].astype(np.float32)
            except:
                import pdb; pdb.set_trace()
            root_path = Path("./data/sit/")
            gt_boxes[np.isnan(gt_boxes)] = 0

            boxes, names, tokens, rtokens, velocity, rvelocity, trajectory, track_id = [], [], [], [], [], [], [], []


            # import pdb; pdb.set_trace()
            for i in range(res["metadata"]["timesteps"]):

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

            if len(boxes) == 0:
                print('0')

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