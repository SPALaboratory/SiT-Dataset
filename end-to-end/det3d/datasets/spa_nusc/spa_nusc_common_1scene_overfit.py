from datetime import time
from time import get_clock_info
import numpy as np
import cv2
import pickle
import pdb 
from pathlib import Path
from functools import reduce
from typing import List

from tqdm import tqdm
from pyquaternion import Quaternion

import os
from det3d.utils.simplevis import *

from itertools import tee
from copy import deepcopy

try:
    from nuscenes import NuScenes
    from nuscenes.utils import splits
    from nuscenes.utils.data_classes import Box, LidarPointCloud
    from nuscenes.eval.detection.config import config_factory
    from nuscenes.eval.detection.evaluate import NuScenesEval
    from nuscenes.eval.detection.render import visualize_sample
except:
    print("nuScenes devkit not Found!")

import pdb 

general_to_detection = {
    "human.pedestrian.adult": "pedestrian",
    "human.pedestrian.child": "pedestrian",
    "human.pedestrian.wheelchair": "ignore",
    "human.pedestrian.stroller": "ignore",
    "human.pedestrian.personal_mobility": "ignore",
    "human.pedestrian.police_officer": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian",
    "animal": "ignore",
    "vehicle.car": "car",
    "vehicle.motorcycle": "ignore",
    "vehicle.bicycle": "ignore",
    "vehicle.bus.bendy": "ignore",
    "vehicle.bus.rigid": "ignore",
    "vehicle.truck": "ignore",
    "vehicle.construction": "ignore",
    "vehicle.emergency.ambulance": "ignore",
    "vehicle.emergency.police": "ignore",
    "vehicle.trailer": "ignore",
    "movable_object.barrier": "ignore",
    "movable_object.trafficcone": "ignore",
    "movable_object.pushable_pullable": "ignore",
    "movable_object.debris": "ignore",
    "static_object.bicycle_rack": "ignore",
}

cls_attr_dist = {
    # "barrier": {
    #     "cycle.with_rider": 0,
    #     "cycle.without_rider": 0,
    #     "pedestrian.moving": 0,
    #     "pedestrian.sitting_lying_down": 0,
    #     "pedestrian.standing": 0,
    #     "vehicle.moving": 0,
    #     "vehicle.parked": 0,
    #     "vehicle.stopped": 0,
    # },
    # "bicycle": {
    #     "cycle.with_rider": 2791,
    #     "cycle.without_rider": 8946,
    #     "pedestrian.moving": 0,
    #     "pedestrian.sitting_lying_down": 0,
    #     "pedestrian.standing": 0,
    #     "vehicle.moving": 0,
    #     "vehicle.parked": 0,
    #     "vehicle.stopped": 0,
    # },
    # "bus": {
    #     "cycle.with_rider": 0,
    #     "cycle.without_rider": 0,
    #     "pedestrian.moving": 0,
    #     "pedestrian.sitting_lying_down": 0,
    #     "pedestrian.standing": 0,
    #     "vehicle.moving": 9092,
    #     "vehicle.parked": 3294,
    #     "vehicle.stopped": 3881,
    # },
    "car": {
        "cycle.with_rider": 0,
        "cycle.without_rider": 0,
        "pedestrian.moving": 0,
        "pedestrian.sitting_lying_down": 0,
        "pedestrian.standing": 0,
        "vehicle.moving": 114304,
        "vehicle.parked": 330133,
        "vehicle.stopped": 46898,
    },
    # "construction_vehicle": {
    #     "cycle.with_rider": 0,
    #     "cycle.without_rider": 0,
    #     "pedestrian.moving": 0,
    #     "pedestrian.sitting_lying_down": 0,
    #     "pedestrian.standing": 0,
    #     "vehicle.moving": 882,
    #     "vehicle.parked": 11549,
    #     "vehicle.stopped": 2102,
    # },
    # "ignore": {
    #     "cycle.with_rider": 307,
    #     "cycle.without_rider": 73,
    #     "pedestrian.moving": 0,
    #     "pedestrian.sitting_lying_down": 0,
    #     "pedestrian.standing": 0,
    #     "vehicle.moving": 165,
    #     "vehicle.parked": 400,
    #     "vehicle.stopped": 102,
    # },
    # "motorcycle": {
    #     "cycle.with_rider": 4233,
    #     "cycle.without_rider": 8326,
    #     "pedestrian.moving": 0,
    #     "pedestrian.sitting_lying_down": 0,
    #     "pedestrian.standing": 0,
    #     "vehicle.moving": 0,
    #     "vehicle.parked": 0,
    #     "vehicle.stopped": 0,
    # },
    "pedestrian": {
        "cycle.with_rider": 0,
        "cycle.without_rider": 0,
        "pedestrian.moving": 157444,
        "pedestrian.sitting_lying_down": 13939,
        "pedestrian.standing": 46530,
        "vehicle.moving": 0,
        "vehicle.parked": 0,
        "vehicle.stopped": 0,
    },
    # "traffic_cone": {
    #     "cycle.with_rider": 0,
    #     "cycle.without_rider": 0,
    #     "pedestrian.moving": 0,
    #     "pedestrian.sitting_lying_down": 0,
    #     "pedestrian.standing": 0,
    #     "vehicle.moving": 0,
    #     "vehicle.parked": 0,
    #     "vehicle.stopped": 0,
    # },
    # "trailer": {
    #     "cycle.with_rider": 0,
    #     "cycle.without_rider": 0,
    #     "pedestrian.moving": 0,
    #     "pedestrian.sitting_lying_down": 0,
    #     "pedestrian.standing": 0,
    #     "vehicle.moving": 3421,
    #     "vehicle.parked": 19224,
    #     "vehicle.stopped": 1895,
    # },
    # "truck": {
    #     "cycle.with_rider": 0,
    #     "cycle.without_rider": 0,
    #     "pedestrian.moving": 0,
    #     "pedestrian.sitting_lying_down": 0,
    #     "pedestrian.standing": 0,
    #     "vehicle.moving": 21339,
    #     "vehicle.parked": 55626,
    #     "vehicle.stopped": 11097,
    # },
}

def _second_det_to_nusc_box(detection):
    box3d = detection["box3d_lidar"].detach().cpu().numpy()
    scores = detection["scores"].detach().cpu().numpy()
    labels = detection["label_preds"].detach().cpu().numpy()
    
    box3d[:, -1] = -box3d[:, -1] - np.pi / 2

    box_list = []
    for i in range(box3d.shape[0]):
        quat = Quaternion(axis=[0, 0, 1], radians=box3d[i, -1])
        velocity = (*box3d[i, 6:8], 0.0)
        box = Box(
            center = box3d[i, :3],
            size = box3d[i, 3:6],
            orientation=quat,
            label=labels[i],
            score=scores[i],
            velocity=velocity,
        )
        
        box_list.append(box)

    return box_list

def window(iterable, size):
    iters = tee(iterable, size)
    for i in range(1, size):
        for each in iters[i:]:
            next(each, None)

    return zip(*iters)

def get_time(nusc, src_token, dst_token):
    time_last = 1e-6 * nusc.get('sample', src_token)["timestamp"]
    time_first = 1e-6 * nusc.get('sample', dst_token)["timestamp"]
    time_diff = time_first - time_last

    return time_diff 

def get_time_(nusc, src_token, dst_token):
    # time_last = 1e-6 * int(src_token.split("*")[-1])
    # time_first = 1e-6 * int(dst_token.split("*")[-1])
    # time_diff = time_first - time_last
    time_diff = 0.2

    return time_diff 

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

def quaternion_to_euler(quaternion):
    """
    Convert quaternion to euler angles.

    :param quaternion: numpy array of shape (4,), representing quaternion in wxyz order.
    :return: numpy array of shape (3,), representing euler angles in roll-pitch-yaw order, in radians.
    """
    qw, qx, qy, qz = quaternion
    # roll (x-axis rotation)
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx ** 2 + qy ** 2)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    # pitch (y-axis rotation)
    sinp = 2 * (qw * qy - qz * qx)
    if np.abs(sinp) >= 1:
        pitch = np.sign(sinp) * np.pi / 2
    else:
        pitch = np.arcsin(sinp)
    # yaw (z-axis rotation)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy ** 2 + qz ** 2)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.array([roll, pitch, yaw])

def get_ego_matrix(label_path):
    with open(label_path, 'r') as f:
        lines = f.readlines()
    return np.array([float(i) for i in lines[0].split(",")]).reshape(4, 4)

def euler_to_quaternion(euler):
    """
    Convert euler angles to quaternion.

    :param euler: numpy array of shape (3,), representing euler angles in roll-pitch-yaw order, in radians.
    :return: numpy array of shape (4,), representing quaternion in wxyz order.
    """
    roll, pitch, yaw = euler
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return np.array([qw, qx, qy, qz])

def get_label_path_from_token(token):
    place_ = token.split("*")[0]
    scene_ = token.split("*")[1]
    frame_ = token.split("*")[2]

    root_path = Path("./data/spa")
    anno_path_ = root_path / place_ / scene_ / "label/{}.txt".format(frame_)
    return [anno_path_]

def center_distance(gt_box, pred_box) -> float:
    """
    L2 distance between the box centers (xy only).
    :param gt_box: GT annotation sample.
    :param pred_box: Predicted sample.
    :return: L2 distance.
    """
    return np.linalg.norm(np.array(pred_box.center[:2]) - np.array(gt_box.center[:2]))

def trajectory(nusc, boxes, time, timesteps=7, past=False):
    target = boxes[-1]
    
    static_forecast = deepcopy(boxes[0])

    linear_forecast = deepcopy(boxes[0])
    vel = linear_forecast.velocity[:2]
    disp = np.sum(list(map(lambda x: np.array(list(vel) + [0]) * x, time)), axis=0)

    if past:
        linear_forecast.center = linear_forecast.center - disp

    else:
        linear_forecast.center = linear_forecast.center + disp
    
    if center_distance(target, static_forecast) < max(target.wlh[0], target.wlh[1]):
        return "static"

    elif center_distance(target, linear_forecast) < max(target.wlh[0], target.wlh[1]):
        return "linear"

    else:
        return "nonlinear"
    
    return "nonlinear"

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

def get_annotations_(nusc, train_scenes, idx, timesteps, root_path):
    forecast_annotations = []
    forecast_boxes = []   
    forecast_trajectory = []

    root_path = Path("./data/spa/")
    token = train_scenes[idx]
    anno_path = get_label_path_from_token(token)
    anno = get_anno_from_path(anno_path)
    place = token.split("*")[0]
    scene = token.split("*")[1]
    frame = token.split("*")[2]
    odom_path = root_path / place / scene / "ego_trajectory/{}.txt".format(frame)
    
    # box inverse compensation from ego
    ego_motion = get_ego_matrix(odom_path)
    #ego_yaw = rotmat_to_euler(ego_motion[:3, :3])[2]
    #anno['rotation_y'] += ego_yaw
    #comp_obj_center = np.matmul(np.linalg.inv(ego_motion), np.concatenate([anno['location'][:,:3], np.ones(anno['location'][:,:3].shape[0]).reshape(1, -1).T], axis=1).T).T
    #anno['location'][:, :3] = comp_obj_center[:, :3]

    anno['location'][:, :3] -= ego_motion[:3, 3]

    vis_flag = False
    if vis_flag:
        
        save_path = "/home/changwon/detection_task/Det3D/viz_in_model/preprocessing/"
        os.makedirs(save_path, exist_ok=True)

        locs = np.array(anno['location']).reshape(-1, 3)
        dims = np.array(anno['dimensions']).reshape(-1, 3) # order : l,w,h 
        rots = np.array(anno['rotation_y']).reshape(
                -1, 1
            )
        gt_boxes = np.concatenate(
                [locs, dims, rots], axis=1
            )

        bbox_list = gt_boxes
        velo_path = "./data/spa/{}/{}/velo/concat/bin_data/{}.bin".format(place, scene, frame)
        points_ = np.fromfile(velo_path, dtype=np.float32, count=-1).reshape([-1, 4])
        points = np.concatenate([points_[:, :3], np.ones(points_.shape[0]).reshape(1, -1).T], axis=1).T
        points = np.matmul(ego_motion, points).T

        pred_boxes = bbox_list
        point = points
        #pred_boxes[:, 6] *= -1
        bev = nuscene_vis(point, pred_boxes)
        cv2.imwrite(save_path+"pred_{}*{}*{}.png".format(place, scene, frame), bev)

    for num_ in range(anno['name'].shape[0]): # object별 수행
        tracklet_box = []
        tracklet_annotation = []
        tracklet_trajectory = []
        track_id = anno['track_id'][num_]

        for step in range(timesteps):
            if step == 0:
                box = Box(center = anno["location"][num_].tolist(),
                        size = anno["dimensions"][num_].tolist(), #wlh
                        orientation = Quaternion(axis=[0, 0, 1], radians=anno["rotation_y"][num_].tolist()),
                    #   velocity = nusc.box_velocity(annotation["token"]),
                        name = anno["name"][num_],
                        token = token)
                dumy_anno = {}
                for key in list(anno.keys()):
                    dumy_anno[key] = np.array(anno[key][num_])
                dumy_anno.update(transform=get_ego_matrix(odom_path))
                # dumy_anno.update(transform=np.eye(4))

                try: #train scene의 마지막 idx 이후의 예외처리
                    if len(train_scenes[idx+step-1]) != 0:
                        dumy_anno.update(prev=train_scenes[idx+step-1], next=train_scenes[idx+step+1], token=token)
                    else:
                        dumy_anno.update(prev="", next=train_scenes[idx+step+1], token=token)
                    tracklet_box.append(box)
                    tracklet_annotation.append(dumy_anno)
                except:
                    # print(1)
                    if len(train_scenes[idx+step-1]) != 0:
                        dumy_anno.update(prev=train_scenes[idx+step-1], next=train_scenes[idx], token=token)
                    else:
                        dumy_anno.update(prev="", next=train_scenes[idx], token=token)
                    tracklet_box.append(box)
                    tracklet_annotation.append(dumy_anno)
            else:
                try: #next step이 없을 경우 except로 가서 전스텝을 저장
                    token_ = train_scenes[idx+step]
                    place_ = token_.split("*")[0]
                    scene_ = token_.split("*")[1]
                    frame_ = token_.split("*")[2]
                    odom_path_ = root_path / place_ / scene_ / "ego_trajectory/{}.txt".format(frame_)
                    anno_path_ = get_label_path_from_token(token_)
                    anno_ = get_anno_from_path(anno_path_)
                    ego_motion_ = get_ego_matrix(odom_path_)
                    # ego_yaw_ = rotmat_to_euler(ego_motion_[:3, :3])[2]
                    # anno_['rotation_y'] += ego_yaw_
                    # comp_obj_center_ = np.matmul(np.linalg.inv(ego_motion_), np.concatenate([anno_['location'][:,:3], np.ones(anno_['location'][:,:3].shape[0]).reshape(1, -1).T], axis=1).T).T
                    # anno_['location'][:, :3] = comp_obj_center_[:, :3]
                    anno_['location'][:, :3] -= ego_motion_[:3, 3]

                    if np.where(anno_['track_id']==track_id)[0].shape[0] != 0: #next step에서 tracked_obj이 있을때만 경로 추가
                        track_index = np.where(anno_['track_id']==track_id)[0].item()
                        box_ = Box(center = anno_["location"][track_index].tolist(),
                                size = anno_["dimensions"][track_index].tolist(),
                                orientation = Quaternion(axis=[0,0,1], radians=anno_["rotation_y"][track_index]),
                            #   velocity = nusc.box_velocity(annotation["token"]),
                                name = anno_["name"][track_index],
                                token = token_)
                        dumy_anno_ = {}
                        for key in list(anno_.keys()):
                            dumy_anno_[key] = np.array(anno_[key][track_index])
                        dumy_anno_.update(transform=get_ego_matrix(odom_path_))
                        # dumy_anno_.update(transform=np.eye(4))
                        try: # next step이 없을 경우 현재 스텝을 저장하도록 함
                            dumy_anno_.update(prev=train_scenes[idx+step-1], next=train_scenes[idx+step+1], token=token_)
                        except:
                            dumy_anno_.update(prev=train_scenes[idx+step-1], next='', token=token_)

                        tracklet_box.append(box_)
                        tracklet_annotation.append(dumy_anno_)
                    else:
                        tracklet_box.append(tracklet_box[-1])
                        tracklet_annotation.append(tracklet_annotation[-1])
                except: # 다음 토큰이 없을때 여기로 오게 됨
                    box_ = Box(center = [0, 0, 0],
                            size = [0, 0, 0],
                            orientation = Quaternion(axis=[0,0,1], radians=0),
                        #   velocity = nusc.box_velocity(annotation["token"]),
                            name = 'pedestrian',
                            token = token_)
                    dumy_anno_ = {}
                    for key in list(anno_.keys()):
                        dumy_anno_[key] = np.array(anno_[key][track_index])
                    dumy_anno_.update(transform=get_ego_matrix(odom_path_))
                    # dumy_anno_.update(transform=np.eye(4))
                    try: # next step이 없을 경우 현재 스텝을 저장하도록 함
                        dumy_anno_.update(prev=train_scenes[idx+step-1], next=train_scenes[idx+step+1], token=token_)
                    except:
                        dumy_anno_.update(prev=train_scenes[idx+step-1], next='', token=token_)

                    tracklet_box.append(box_)
                    tracklet_annotation.append(dumy_anno_)
                    continue

        tokens = [b["token"] for b in tracklet_annotation]

        time = [get_time_(nusc, src, dst) for src, dst in window(tokens, 2)]
        
        vis_flag = False
        for ii, _box in enumerate(tracklet_box):
            cur_box_center = _box.center

            # # 다시 ego motion만큼 돌려주는 이유 : global coord에서 velocity를 구하기 위함
            # cur_M = tracklet_annotation[ii]["transform"]
            # cur_box_center = np.matmul(cur_M, np.concatenate([cur_box_center[:3], np.array([1])]).T).T

            try:
                next_box_center = tracklet_box[ii+1].center
                # next_M = tracklet_annotation[ii+1]["transform"]
                # next_box -> global -> cur_local
                # next_box_center = np.matmul(next_M, np.concatenate([next_box_center[:3], np.array([1])]).T).T

                for_velo = (next_box_center[:3] - cur_box_center[:3])/0.2 # 5Hz : 0.2s
                # for_velo = np.matmul(np.linalg.inv(cur_M[:3, :3]), for_velo[:3].T).T

                # for_local_rot = np.eye(4)
                # for_local_rot[:3, :3] = cur_M[:3, :3]
                # for_velo = np.matmul(np.linalg.inv(for_local_rot), np.concatenate([for_velo[:3], np.array([1])]).T).T

                # for_velo = (cur_box_center[:3] - next_box_center[:3])/0.2 # 5Hz : 0.2s
                tracklet_box[ii].velocity = for_velo
                tracklet_annotation[ii]['velocity'] = for_velo
            except:
                if len(tracklet_box) == 1:
                    tracklet_box[ii].velocity = np.array([0,0,0])
                    tracklet_annotation[ii]['velocity'] = np.array([0,0,0])
                else:
                    tracklet_box[ii].velocity = tracklet_box[ii-1].velocity
                    tracklet_annotation[ii]['velocity'] = tracklet_annotation[ii-1]['velocity']

            if vis_flag:
                place = _box.token.split("*")[0]
                scene = _box.token.split("*")[1]
                frame = _box.token.split("*")[2]
                save_path = "/home/changwon/detection_task/Det3D/viz_in_model/preprocessing_traj/"
                os.makedirs(save_path, exist_ok=True)

                locs = np.array(_box.center).reshape(-1, 3)
                dims = np.array(_box.wlh).reshape(-1, 3) #[:, [1,0,2]] # order : wlh -> l,w,h 
                rots = np.array(_box.orientation.yaw_pitch_roll[0]).reshape(
                        -1, 1
                    )
                velos = np.array(_box.velocity).reshape(-1, 3)[:, :2]
                gt_boxes = np.concatenate(
                        [locs, dims, rots, velos], axis=1
                    )

                bbox_list = gt_boxes
                velo_path = "./data/spa/{}/{}/velo/concat/bin_data/{}.bin".format(place, scene, frame)

                odom_path_ = root_path / place / scene / "ego_trajectory/{}.txt".format(frame)
                ego_motion_ = get_ego_matrix(odom_path_)
                ego_rot = np.eye(4)
                ego_rot[:3, :3] = ego_motion_[:3, :3]


                points_ = np.fromfile(velo_path, dtype=np.float32, count=-1).reshape([-1, 4])
                points = np.concatenate([points_[:, :3], np.ones(points_.shape[0]).reshape(1, -1).T], axis=1).T
                points = np.matmul(ego_motion, points).T

                pred_boxes = bbox_list
                point = points
                #pred_boxes[:, 6] *= -1
                bev = nuscene_vis(point, pred_boxes)
                cv2.imwrite(save_path+"traj_{}*{}*{}-timestep-{}-cls-{}.png".format(place, scene, frame, ii, track_id), bev)
                print(ii)

        tracklet_trajectory = trajectory(nusc, tracklet_box, time, timesteps) # 이부분에서는 linear, nonlinear indicator

        forecast_boxes.append(tracklet_box)
        forecast_annotations.append(tracklet_annotation)
        forecast_trajectory.append(timesteps*[tracklet_trajectory])

    return forecast_boxes, forecast_annotations, forecast_trajectory

def get_calib(calib_path):
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
        # projection_matrix = [P0, P1, P2, P3, P4]
        projection_matrix = [ [P0_intrinsic, P0_extrinsic], \
                              [P1_intrinsic, P1_extrinsic], \
                              [P2_intrinsic, P2_extrinsic], \
                              [P3_intrinsic, P3_extrinsic], \
                              [P4_intrinsic, P4_extrinsic], 
            ]
        return projection_matrix

def _fill_trainval_infos(nusc, train_scenes, val_scenes, test=False, nsweeps=1, filter_zero=True, timesteps=7, past=False):    
    train_nusc_infos = []
    val_nusc_infos = []

    root_path = Path("./data/spa/")
    # for path in tqdm(train_scenes):
    for ii, path in enumerate(tqdm(train_scenes[:-6])):
        place = path.split("*")[0]
        scene = path.split("*")[1]
        frame = path.split("*")[2]
        token = path
        calib_path = root_path / place / scene / "calib" / "{}.txt".format(frame)
        projection_matrix = get_calib(calib_path)
        velo_path = root_path / place / scene / "velo/concat/bin_data" / "{}.bin".format(frame)

        camera_mapping = {4:'CAM_FRONT', 1:'CAM_FRONT_LEFT', 5:'CAM_FRONT_RIGHT', 3:'CAM_BACK_RIGHT',  2:'CAM_BACK_LEFT'}
        camera_mapping_ = {'CAM_FRONT':4, 'CAM_FRONT_LEFT':1, 'CAM_FRONT_RIGHT':5, 'CAM_BACK_RIGHT':3,  'CAM_BACK_LEFT':2}

        cam_num_list = [1, 2, 3, 4, 5]
        cam_path_ = []
        for num in cam_num_list:
            cam_path_.append(root_path / place / scene / "cam_img/{}".format(num) / "data_rgb" / "{}.png".format(frame))

        info = {
            "lidar_path": velo_path,
            "cam_front_path": cam_path_[camera_mapping_['CAM_FRONT']],
            "cam_intrinsic": projection_matrix[camera_mapping_['CAM_FRONT']],
            "token": token,
            "timestamp": frame,
        }

        sweeps = []
        count = 0
        while len(sweeps) < nsweeps - 1:
            # if curr_sd_rec["prev"] == "":
                count += 1
                try:
                    next_step_ = train_scenes[ii+1]
                    place_ = next_step_.split("*")[0]
                    scene_ = next_step_.split("*")[1]
                    frame_ = next_step_.split("*")[2]
                    token_ = next_step_
                    velo_path_ = root_path / place_ / scene_ / "velo/concat/bin_data" / "{}.bin".format(frame_)
                    sweep = {
                        "lidar_path": velo_path_,
                        "sample_data_token": token_,
                        "transform_matrix": None,
                        "time_lag": frame_ + "_{}".format(count),
                        # time_lag: 0,
                    }
                    sweeps.append(sweep)
                except:
                    sweeps.append(sweeps[-1])

        info["sweeps"] = sweeps

        if not test:
            place_ = path.split("*")[0]
            scene_ = path.split("*")[1]
            frame_ = path.split("*")[2]
            bev_path = root_path / place_ / scene_ / "bev" / "{}.png".format(frame_)
            ego_map = cv2.imread(str(bev_path), cv2.IMREAD_COLOR)
            # ego_map = nusc.explorer.get_ego_centric_map(sweeps[0]["sample_data_token"])
            bev = cv2.resize(ego_map, dsize=(180, 180), interpolation=cv2.INTER_CUBIC)

            forecast_boxes, forecast_annotations, forecast_trajectory = get_annotations_(nusc, train_scenes, ii, timesteps, root_path) #timesteps = 7, past = flag

            # num_point filtering
            gt_box_ = np.stack([np.array(list(forecast_boxes[i][0].center) + list(forecast_boxes[i][0].wlh) + [forecast_boxes[i][0].orientation.yaw_pitch_roll[0]]) for i in range(len(forecast_boxes))])
            corners_3d = box_center_to_corner_3d_(gt_box_[:, :3], gt_box_[:, 3:6], gt_box_[:, 6])
            velo_path_ = root_path / place_ / scene_ / "velo/concat/bin_data" / "{}.bin".format(frame_)
            points_ = np.fromfile(velo_path_, dtype=np.float32, count=-1).reshape([-1, 4])

            odom_path = root_path / place_ / scene_ / "ego_trajectory/{}.txt".format(frame_)
            ego_motion = get_ego_matrix(odom_path)
            ego_rot = np.eye(4)
            ego_rot[:3, :3] = ego_motion[:3, :3]
            points = np.concatenate([points_[:, :3], np.ones(points_.shape[0]).reshape(1, -1).T], axis=1).T
            points = np.matmul(ego_rot, points).T
            num_pts_list = get_pts_in_3dbox_(points, corners_3d)
            mask = np.array([pts > 0 for pts in num_pts_list], dtype=bool).reshape(-1)

            # mask = np.full(len(forecast_boxes), True)
            locs = [np.array([b.center for b in boxes]).reshape(-1, 3) for boxes in forecast_boxes]
            # locs = [np.array([i.center for i in boxes]) for boxes in forecast_boxes]
            rlocs = [np.array([b.center for b in boxes]).reshape(-1, 3) for boxes in forecast_boxes]

            dims = [np.array([b.wlh for b in boxes]).reshape(-1, 3) for boxes in forecast_boxes]
            # rots = np.array([b.orientation.yaw_pitch_roll[0] for b in ref_boxes]).reshape(-1, 1)
            velocity = [np.array([b.velocity for b in boxes]).reshape(-1, 3) for boxes in forecast_boxes]
            rvelocity = [np.array([b.velocity for b in boxes]).reshape(-1, 3) for boxes in forecast_boxes]
            # velocity = [np.array([[0, 0, 0] for b in boxes]).reshape(-1, 3) for boxes in forecast_boxes]
            # rvelocity = [np.array([[0, 0, 0] for b in boxes]).reshape(-1, 3) for boxes in forecast_boxes]

            # rots = [np.array([quaternion_yaw(b.orientation) for b in boxes]).reshape(-1, 1) for boxes in forecast_boxes]
            rots = [np.array([b.orientation.yaw_pitch_roll[0] for b in boxes]).reshape(-1, 1) for boxes in forecast_boxes]
            rrots = [np.array([b.orientation.yaw_pitch_roll[0] for b in boxes]).reshape(-1, 1) for boxes in forecast_boxes]

            names = [np.array([b.name for b in boxes]) for boxes in forecast_boxes]
            
            tokens = [np.array([b.token for b in boxes]) for boxes in forecast_boxes]
            rtokens = [np.array([b.token for b in boxes]) for boxes in forecast_boxes]

            trajectory = [np.array([b for b in boxes]) for boxes in forecast_trajectory]

            # gt_boxes = [np.concatenate([locs[i], dims[i], velocity[i][:, :2], rvelocity[i][:, :2], -rots[i] - np.pi / 2, -rrots[i] - np.pi / 2], axis=1) for i in range(len(annotations))]
            gt_boxes = np.array([np.concatenate([locs[i], dims[i], velocity[i][:, :2], rvelocity[i][:, :2], rots[i], rrots[i]], axis=1) for i in range(len(forecast_boxes))])
            # gt_boxes = np.concatenate([i[np.newaxis, :]for i in gt_boxes], axis=0)
            vis_flag=False
            if vis_flag:
                for _, bb in enumerate(gt_boxes[mask]):
                    obj = bb[np.newaxis, :]
                    for i in range(timesteps):
                        if vis_flag:
                            save_path = "/home/changwon/detection_task/Det3D/viz_in_model/preprocessing_forcase_boxes/"
                            os.makedirs(save_path, exist_ok=True)

                            token_ = forecast_boxes[_][i].token
                            place_ = token_.split("*")[0]
                            scene_ = token_.split("*")[1]
                            frame_ = token_.split("*")[2]

                            odom_path = root_path / place / scene / "ego_trajectory/{}.txt".format(frame)
                            ego_motion = get_ego_matrix(odom_path)
                            ego_rot = np.eye(4)
                            ego_rot[:3, :3] = ego_motion[:3, :3]

                            velo_path = "./data/spa/{}/{}/velo/concat/bin_data/{}.bin".format(place_, scene_, frame_)
                            points_ = np.fromfile(velo_path, dtype=np.float32, count=-1).reshape([-1, 4])
                            points = np.concatenate([points_[:, :3], np.ones(points_.shape[0]).reshape(1, -1).T], axis=1).T
                            points = np.matmul(ego_rot, points).T

                            gt_boxes_ = obj[:, i][:, [0,1,2,3,4,5,-2]] #wlh

                            # points[:, :3] -= ego_motion[:3, 3]
                            # gt_boxes_[:, :3] = gt_boxes_[:, :3] - ego_motion[:3, 3]

                            #pred_boxes[:, 6] *= -1
                            bev = nuscene_vis(points, gt_boxes_)
                            cv2.imwrite(save_path+"pred_{}*{}*{}-timestap-{}.png".format(place_, scene_, frame_, i), bev)

            assert len(forecast_boxes) == len(gt_boxes) == len(velocity) == len(rvelocity)


            if len(forecast_boxes) > 0:
                if not filter_zero:
                    info["gt_boxes"] = np.array(gt_boxes)
                    info["gt_boxes_velocity"] = np.array(velocity)
                    info["gt_boxes_rvelocity"] = np.array(rvelocity)
                    info["gt_names"] = np.array([[n for n in name] for name in names])
                    info["gt_boxes_token"] = np.array(tokens)
                    info["gt_boxes_rtoken"] = np.array(rtokens)
                    info["gt_trajectory"] = np.array(trajectory)
                    info['gt_track_id'] = np.array([np.array([b['track_id'] for b in boxes]) for boxes in forecast_annotations])
                    info["bev"] = bev
                else:
                    info["gt_boxes"] = np.array(gt_boxes)[mask]
                    info["gt_boxes_velocity"] = np.array(velocity)[mask]
                    info["gt_boxes_rvelocity"] = np.array(rvelocity)[mask]
                    # info["gt_names"] = np.array([[general_to_detection[n] for n in name] for name in names])[mask]
                    info["gt_names"] = np.array([[n for n in name] for name in names])[mask]
                    info["gt_boxes_token"] = np.array(tokens)[mask]
                    info["gt_boxes_rtoken"] = np.array(rtokens)[mask]
                    info["gt_trajectory"] = np.array(trajectory)[mask]
                    info['gt_track_id'] = np.array([np.array([b['track_id'] for b in boxes]) for boxes in forecast_annotations])[mask]
                    info["bev"] = bev
            else:
                # mask = np.array([(anno['num_lidar_pts'] + anno['num_radar_pts']) >0 for anno in forecast_boxes], dtype=bool).reshape(-1)
                mask = np.full(len(forecast_boxes), True)
                locs = np.array([b.center for b in forecast_boxes]).reshape(-1, 3)
                dims = np.array([b.wlh for b in forecast_boxes]).reshape(-1, 3)
                # rots = np.array([b.orientation.yaw_pitch_roll[0] for b in ref_boxes]).reshape(-1, 1)
                velocity = np.array([b.velocity for b in forecast_boxes]).reshape(-1, 3)
                rvelocity = np.array([b.rvelocity for b in forecast_boxes]).reshape(-1, 3)
                # velocity = [np.array([[0, 0, 0] for b in boxes]).reshape(-1, 3) for boxes in forecast_boxes]
                # rvelocity = [np.array([[0, 0, 0] for b in boxes]).reshape(-1, 3) for boxes in forecast_boxes]
                rots = np.array([quaternion_yaw(b.orientation) for b in forecast_boxes]).reshape(-1, 1)
                names = np.array([b.name for b in forecast_boxes])
                tokens = np.array([b.token for b in forecast_boxes])
                if locs.shape[0] != 0:
                    gt_boxes = np.concatenate([locs, dims, velocity[:, :2], rvelocity[:, :2], rots ], axis=1)
                else:
                    gt_boxes = np.array([]).reshape(-1, 11)
                # trajectory = np.array(["static" for b in forecast_boxes])

                info["gt_boxes"] = gt_boxes
                info["gt_boxes_velocity"] = velocity
                info["gt_boxes_rvelocity"] = rvelocity
                info["gt_names"] = np.array([general_to_detection[name] for name in names])
                info["gt_boxes_token"] = tokens
                info["gt_boxes_rtoken"] = tokens
                info["gt_boxes_rtoken"] = tokens
                info["gt_trajectory"] = trajectory
                info['gt_track_id'] = np.array([np.array([b['track_id'] for b in boxes]) for boxes in forecast_annotations])
                info['bev'] = bev

            train_nusc_infos.append(info)
        
    # preprocessing val set
    for ii, path in enumerate(tqdm(val_scenes[:-6])):
        place = path.split("*")[0]
        scene = path.split("*")[1]
        frame = path.split("*")[2]
        token = path
        calib_path = root_path / place / scene / "calib" / "{}.txt".format(frame)
        projection_matrix = get_calib(calib_path)
        velo_path = root_path / place / scene / "velo/concat/bin_data" / "{}.bin".format(frame)

        camera_mapping = {4:'CAM_FRONT', 1:'CAM_FRONT_LEFT', 5:'CAM_FRONT_RIGHT', 3:'CAM_BACK_RIGHT',  2:'CAM_BACK_LEFT'}
        camera_mapping_ = {'CAM_FRONT':4, 'CAM_FRONT_LEFT':1, 'CAM_FRONT_RIGHT':5, 'CAM_BACK_RIGHT':3,  'CAM_BACK_LEFT':2}

        cam_num_list = [1, 2, 3, 4, 5]
        cam_path_ = []
        for num in cam_num_list:
            cam_path_.append(root_path / place / scene / "cam_img/{}".format(num) / "data_rgb" / "{}.png".format(frame))

        info = {
            "lidar_path": velo_path,
            "cam_front_path": cam_path_[camera_mapping_['CAM_FRONT']],
            "cam_intrinsic": projection_matrix[camera_mapping_['CAM_FRONT']],
            "token": token,
            "timestamp": frame,
        }

        sweeps = []
        count = 0
        while len(sweeps) < nsweeps - 1:
            # if curr_sd_rec["prev"] == "":
                count += 1
                try:
                    next_step_ = train_scenes[ii+1]
                    place_ = next_step_.split("*")[0]
                    scene_ = next_step_.split("*")[1]
                    frame_ = next_step_.split("*")[2]
                    token_ = next_step_
                    velo_path_ = root_path / place_ / scene_ / "velo/concat/bin_data" / "{}.bin".format(frame_)
                    sweep = {
                        "lidar_path": velo_path_,
                        "sample_data_token": token_,
                        "transform_matrix": None,
                        "time_lag": frame_ + "_{}".format(count),
                        # time_lag: 0,
                    }
                    sweeps.append(sweep)
                except:
                    sweeps.append(sweeps[-1])

        info["sweeps"] = sweeps

        if not test:
            place_ = path.split("*")[0]
            scene_ = path.split("*")[1]
            frame_ = path.split("*")[2]
            bev_path = root_path / place_ / scene_ / "bev" / "{}.png".format(frame_)
            ego_map = cv2.imread(str(bev_path), cv2.IMREAD_COLOR)
            # ego_map = nusc.explorer.get_ego_centric_map(sweeps[0]["sample_data_token"])
            bev = cv2.resize(ego_map, dsize=(180, 180), interpolation=cv2.INTER_CUBIC)

            forecast_boxes, forecast_annotations, forecast_trajectory = get_annotations_(nusc, val_scenes, ii, timesteps, root_path) #timesteps = 7, past = flag

            # num_point filtering
            gt_box_ = np.stack([np.array(list(forecast_boxes[i][0].center) + list(forecast_boxes[i][0].wlh) + [forecast_boxes[i][0].orientation.yaw_pitch_roll[0]]) for i in range(len(forecast_boxes))])
            corners_3d = box_center_to_corner_3d_(gt_box_[:, :3], gt_box_[:, 3:6], gt_box_[:, 6])
            velo_path_ = root_path / place_ / scene_ / "velo/concat/bin_data" / "{}.bin".format(frame_)
            points_ = np.fromfile(velo_path_, dtype=np.float32, count=-1).reshape([-1, 4])

            odom_path = root_path / place_ / scene_ / "ego_trajectory/{}.txt".format(frame_)
            ego_motion = get_ego_matrix(odom_path)
            ego_rot = np.eye(4)
            ego_rot[:3, :3] = ego_motion[:3, :3]
            points = np.concatenate([points_[:, :3], np.ones(points_.shape[0]).reshape(1, -1).T], axis=1).T
            points = np.matmul(ego_rot, points).T
            num_pts_list = get_pts_in_3dbox_(points, corners_3d)
            mask = np.array([pts > 0 for pts in num_pts_list], dtype=bool).reshape(-1)

            # mask = np.full(len(forecast_boxes), True)

            locs = [np.array([b.center for b in boxes]).reshape(-1, 3) for boxes in forecast_boxes]
            # locs = [np.array([i.center for i in boxes]) for boxes in forecast_boxes]
            rlocs = [np.array([b.center for b in boxes]).reshape(-1, 3) for boxes in forecast_boxes]

            dims = [np.array([b.wlh for b in boxes]).reshape(-1, 3) for boxes in forecast_boxes]
            # rots = np.array([b.orientation.yaw_pitch_roll[0] for b in ref_boxes]).reshape(-1, 1)
            velocity = [np.array([b.velocity for b in boxes]).reshape(-1, 3) for boxes in forecast_boxes]
            rvelocity = [np.array([b.velocity for b in boxes]).reshape(-1, 3) for boxes in forecast_boxes]
            # velocity = [np.array([[0, 0, 0] for b in boxes]).reshape(-1, 3) for boxes in forecast_boxes]
            # rvelocity = [np.array([[0, 0, 0] for b in boxes]).reshape(-1, 3) for boxes in forecast_boxes]

            # rots = [np.array([quaternion_yaw(b.orientation) for b in boxes]).reshape(-1, 1) for boxes in forecast_boxes]
            rots = [np.array([b.orientation.yaw_pitch_roll[0] for b in boxes]).reshape(-1, 1) for boxes in forecast_boxes]
            rrots = [np.array([b.orientation.yaw_pitch_roll[0] for b in boxes]).reshape(-1, 1) for boxes in forecast_boxes]

            names = [np.array([b.name for b in boxes]) for boxes in forecast_boxes]
            
            tokens = [np.array([b.token for b in boxes]) for boxes in forecast_boxes]
            rtokens = [np.array([b.token for b in boxes]) for boxes in forecast_boxes]

            trajectory = [np.array([b for b in boxes]) for boxes in forecast_trajectory]

            # gt_boxes = [np.concatenate([locs[i], dims[i], velocity[i][:, :2], rvelocity[i][:, :2], -rots[i] - np.pi / 2, -rrots[i] - np.pi / 2], axis=1) for i in range(len(annotations))]
            gt_boxes = np.array([np.concatenate([locs[i], dims[i], velocity[i][:, :2], rvelocity[i][:, :2], rots[i], rrots[i]], axis=1) for i in range(len(forecast_boxes))])
            # gt_boxes = np.concatenate([i[np.newaxis, :]for i in gt_boxes], axis=0)
            vis_flag=False
            if vis_flag:
                for _, bb in enumerate(gt_boxes[mask]):
                    obj = bb[np.newaxis, :]
                    for i in range(timesteps):
                        if vis_flag:
                            save_path = "/home/changwon/detection_task/Det3D/viz_in_model/preprocessing_forcase_boxes/"
                            os.makedirs(save_path, exist_ok=True)

                            token_ = forecast_boxes[_][i].token
                            place_ = token_.split("*")[0]
                            scene_ = token_.split("*")[1]
                            frame_ = token_.split("*")[2]

                            odom_path = root_path / place / scene / "ego_trajectory/{}.txt".format(frame)
                            ego_motion = get_ego_matrix(odom_path)
                            ego_rot = np.eye(4)
                            ego_rot[:3, :3] = ego_motion[:3, :3]

                            velo_path = "./data/spa/{}/{}/velo/concat/bin_data/{}.bin".format(place_, scene_, frame_)
                            points_ = np.fromfile(velo_path, dtype=np.float32, count=-1).reshape([-1, 4])
                            points = np.concatenate([points_[:, :3], np.ones(points_.shape[0]).reshape(1, -1).T], axis=1).T
                            points = np.matmul(ego_rot, points).T

                            gt_boxes_ = obj[:, i][:, [0,1,2,3,4,5,-2]] #wlh

                            # points[:, :3] -= ego_motion[:3, 3]
                            # gt_boxes_[:, :3] = gt_boxes_[:, :3] - ego_motion[:3, 3]

                            #pred_boxes[:, 6] *= -1
                            bev = nuscene_vis(points, gt_boxes_)
                            cv2.imwrite(save_path+"pred_{}*{}*{}-timestap-{}.png".format(place_, scene_, frame_, i), bev)

            assert len(forecast_boxes) == len(gt_boxes) == len(velocity) == len(rvelocity)


            if len(forecast_boxes) > 0:
                if not filter_zero:
                    info["gt_boxes"] = np.array(gt_boxes)
                    info["gt_boxes_velocity"] = np.array(velocity)
                    info["gt_boxes_rvelocity"] = np.array(rvelocity)
                    info["gt_names"] = np.array([[n for n in name] for name in names])
                    info["gt_boxes_token"] = np.array(tokens)
                    info["gt_boxes_rtoken"] = np.array(rtokens)
                    info["gt_trajectory"] = np.array(trajectory)
                    info['gt_track_id'] = np.array([np.array([b['track_id'] for b in boxes]) for boxes in forecast_annotations])
                    info["bev"] = bev
                else:
                    info["gt_boxes"] = np.array(gt_boxes)[mask]
                    info["gt_boxes_velocity"] = np.array(velocity)[mask]
                    info["gt_boxes_rvelocity"] = np.array(rvelocity)[mask]
                    # info["gt_names"] = np.array([[general_to_detection[n] for n in name] for name in names])[mask]
                    info["gt_names"] = np.array([[n for n in name] for name in names])[mask]
                    info["gt_boxes_token"] = np.array(tokens)[mask]
                    info["gt_boxes_rtoken"] = np.array(rtokens)[mask]
                    info["gt_trajectory"] = np.array(trajectory)[mask]
                    info['gt_track_id'] = np.array([np.array([b['track_id'] for b in boxes]) for boxes in forecast_annotations])[mask]
                    info["bev"] = bev
            else:
                # mask = np.array([(anno['num_lidar_pts'] + anno['num_radar_pts']) >0 for anno in forecast_boxes], dtype=bool).reshape(-1)
                mask = np.full(len(forecast_boxes), True)
                locs = np.array([b.center for b in forecast_boxes]).reshape(-1, 3)
                dims = np.array([b.wlh for b in forecast_boxes]).reshape(-1, 3)
                # rots = np.array([b.orientation.yaw_pitch_roll[0] for b in ref_boxes]).reshape(-1, 1)
                velocity = np.array([b.velocity for b in forecast_boxes]).reshape(-1, 3)
                rvelocity = np.array([b.rvelocity for b in forecast_boxes]).reshape(-1, 3)
                # velocity = [np.array([[0, 0, 0] for b in boxes]).reshape(-1, 3) for boxes in forecast_boxes]
                # rvelocity = [np.array([[0, 0, 0] for b in boxes]).reshape(-1, 3) for boxes in forecast_boxes]
                rots = np.array([quaternion_yaw(b.orientation) for b in forecast_boxes]).reshape(-1, 1)
                names = np.array([b.name for b in forecast_boxes])
                tokens = np.array([b.token for b in forecast_boxes])
                if locs.shape[0] != 0:
                    gt_boxes = np.concatenate([locs, dims, velocity[:, :2], rvelocity[:, :2], rots ], axis=1)
                else:
                    gt_boxes = np.array([]).reshape(-1, 11)
                # trajectory = np.array(["static" for b in forecast_boxes])

                info["gt_boxes"] = gt_boxes
                info["gt_boxes_velocity"] = velocity
                info["gt_boxes_rvelocity"] = rvelocity
                info["gt_names"] = np.array([general_to_detection[name] for name in names])
                info["gt_boxes_token"] = tokens
                info["gt_boxes_rtoken"] = tokens
                info["gt_boxes_rtoken"] = tokens
                info["gt_trajectory"] = trajectory
                info['gt_track_id'] = np.array([np.array([b['track_id'] for b in boxes]) for boxes in forecast_annotations])
                info['bev'] = bev

            val_nusc_infos.append(info)

    return train_nusc_infos, val_nusc_infos


def quaternion_yaw(q: Quaternion) -> float:
    """
    Calculate the yaw angle from a quaternion.
    Note that this only works for a quaternion that represents a box in lidar or global coordinate frame.
    It does not work for a box in the camera frame.
    :param q: Quaternion of interest.
    :return: Yaw angle in radians.
    """

    # Project into xy plane.
    v = np.dot(q.rotation_matrix, np.array([1, 0, 0]))

    # Measure yaw using arctan.
    yaw = np.arctan2(v[1], v[0])

    return yaw

def _read_imageset_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return [line.splitlines()[0] for line in lines]

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

def box_center_to_corner_3d_(centers, dims, angles):
    translation = centers[:, 0:3]
    w, l, h = dims[:, 0], dims[:, 1], dims[:, 2]
    rotation = angles

    # Create a bounding box outline
    x_corners = np.array([[l_ / 2, l_ / 2, -l_ / 2, -l_ / 2, l_ / 2, l_ / 2, -l_ / 2, -l_ / 2] for l_ in l])
    y_corners = np.array([[w_ / 2, -w_ / 2, -w_ / 2, w_ / 2, w_ / 2, -w_ / 2, -w_ / 2, w_ / 2] for w_ in w])
    z_corners = np.array([[-h_ / 2, -h_ / 2, -h_ / 2, -h_ / 2, h_ / 2, h_ / 2, h_ / 2, h_ / 2] for h_ in h])
    bounding_box = np.array([np.vstack([x_corners[i], y_corners[i], z_corners[i]]) for i in range(x_corners.shape[0])])


    rotation_matrix = np.array([np.array([[np.cos(rotation_),  -np.sin(rotation_), 0],
                                            [np.sin(rotation_), np.cos(rotation_), 0],
                                            [0,  0,  1]]) for rotation_ in rotation])


    corner_box = np.array([np.dot(rotation_matrix[i], bounding_box[i]).T + translation[i] for i in range(x_corners.shape[0])])

    return corner_box

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
    # name_map={'car':'car', 'motocycle':'motorcycle', 'pedestrian':'pedestrian','cyclist':'bicycle', 'motorcycle':'motorcycle'}
    name_map={'Car':'car', 'Truck':'truck', 'Bus':'bus', 'Pedestrian':'pedestrian','Bicyclist':'bicycle', 'Motorcycle':'motorcycle', 
              'Kickboard':'kickboard', 'Vehicle':'car', 'Pedestrian_sitting':'pedestrian_sitting', 'Pedestrain_sitting':'pedestrian_sitting',
              'Cyclist' : 'bicycle', 'Motorcyclist':'motorcycle'
              }
    with open(label_path, 'r') as f:
        lines = f.readlines()

    content = [line.strip().split(' ') for line in lines]

    annotations['name'] = np.array([ name_map[x[0]] for x in content])
    # sitting_mask = np.array([True if i != 'pedestrian_sitting' and i != 'car' else False  for i in annotations['name']])
    sitting_mask = np.array([True if i != 'pedestrian_sitting' and i != 'car' and i != 'bicycle' and i != 'motorcycle' and i != 'truck' else False  for i in annotations['name']])
    annotations['name'] = annotations['name'][sitting_mask]
    annotations['track_id'] = np.array([int(x[1]) for x in content])[sitting_mask]
    annotations['cam_id'] = np.array([float(x[2]) for x in content])[sitting_mask]
    annotations['bbox'] = np.array([[float(info) for info in x[3:7]]
                                    for x in content]).reshape(-1, 4)[sitting_mask] #1102.00 110.00 1596.00 1203.00

    # annotations['dimensions'] = np.abs(np.array([[float(info) for info in x[7:10]]
    #                                       for x in content
    #                                       ]).reshape(-1, 3)[:, [1, 2, 0]])[sitting_mask] #h, l, w -> l, w, h
    annotations['dimensions'] = np.abs(np.array([[float(info) for info in x[7:10]]
                                          for x in content
                                          ]).reshape(-1, 3)[:, [2, 1, 0]])[sitting_mask] #h, l, w -> w, l, h

    annotations['location'] = np.array([[float(info) for info in x[10:13]]
                                        for x in content]).reshape(-1, 3)[sitting_mask]

    annotations['rotation_y'] = np.array([float(x[13])
                                          for x in content]).reshape(-1)[sitting_mask]
    
    annotations['rotation_y'] = -(-annotations['rotation_y'] + np.pi/2)
                                          
    if len(content) != 0 and len(content[0]) == 15:  # have score
        annotations['score'] = np.array([float(x[14]) for x in content])[sitting_mask]
    else:
        annotations['score'] = np.ones((annotations['bbox'].shape[0], ))
    

    # for unique mask
    _, mask_index = np.unique(annotations['track_id'], return_index=True)
    annotations['name'] = annotations['name'][mask_index]
    annotations['track_id'] = annotations['track_id'][mask_index]
    annotations['cam_id'] = annotations['cam_id'][mask_index]
    annotations['bbox'] = annotations['bbox'][mask_index]
    annotations['dimensions'] = annotations['dimensions'][mask_index]
    annotations['location'] = annotations['location'][mask_index]
    annotations['rotation_y'] = annotations['rotation_y'][mask_index]
    annotations['score'] = annotations['score'][mask_index]

    mask_index = np.argsort(annotations['track_id'])
    annotations['name'] = annotations['name'][mask_index]
    annotations['track_id'] = annotations['track_id'][mask_index]
    annotations['cam_id'] = annotations['cam_id'][mask_index]
    annotations['bbox'] = annotations['bbox'][mask_index]
    annotations['dimensions'] = annotations['dimensions'][mask_index]
    annotations['location'] = annotations['location'][mask_index]
    annotations['rotation_y'] = annotations['rotation_y'][mask_index]
    annotations['score'] = annotations['score'][mask_index]
    return annotations 

def get_anno_from_path(path):
    data_anno = {}
    for id, a_path in enumerate(path):
        anno_ = get_label_anno(a_path)
        if id == 0:
            for key in anno_.keys():
                data_anno[key] = anno_[key]
            data_anno['mask'] = np.full(len(data_anno['name']), id)
        else:
            for key in anno_.keys():
                if key in ['bbox', 'dimensions', 'location']:
                    data_anno[key] = np.vstack((data_anno[key], anno_[key]))
                else:
                    data_anno[key] = np.hstack((data_anno[key], anno_[key]))
            mask = np.full(len(anno_['name']), id)
            data_anno['mask'] = np.hstack((data_anno['mask'], mask))

    return data_anno


def create_spa_nusc_infos(root_path, version="v1.0-trainval", experiment="trainval_forecast", nsweeps=20, filter_zero=True, timesteps=7, past=False):
    nusc = NuScenes(version=version, dataroot=root_path, verbose=True)
    available_vers = ["v1.0-trainval", "v1.0-test", "v1.0-mini", "v1.0-spa-trainval"]
    assert version in available_vers
    if version == "v1.0-trainval":
        train_scenes = splits.train
        # random.shuffle(train_scenes)
        # train_scenes = train_scenes[:int(len(train_scenes)*0.2)]
        val_scenes = splits.val
    elif version == "v1.0-test":
        train_scenes = splits.test
        val_scenes = []
    elif version == "v1.0-mini":
        train_scenes = splits.mini_train
        val_scenes = splits.mini_val
    elif version == "v1.0-spa-trainval":
        imageset_folder = Path(root_path) / 'ImageSets'
        train_img_ids = _read_imageset_file(str(imageset_folder / 'train.txt'))
        val_img_ids = _read_imageset_file(str(imageset_folder / 'val.txt'))
        #test_img_ids = _read_imageset_file(str(imageset_folder / 'test.txt'))
        train_scenes = train_img_ids
        val_scenes = val_img_ids
    else:
        raise ValueError("unknown")
    test = "test" in version
    root_path = Path(root_path)

    # # filter exist scenes. you may only download part of dataset.
    # available_scenes = _get_available_scenes(nusc)
    # available_scene_names = [s["name"] for s in available_scenes]
    # train_scenes = list(filter(lambda x: x in available_scene_names, train_scenes))
    # val_scenes = list(filter(lambda x: x in available_scene_names, val_scenes))
    # train_scenes = set(
    #     [
    #         available_scenes[available_scene_names.index(s)]["token"]
    #         for s in train_scenes
    #     ]
    # )
    # val_scenes = set(
    #     [available_scenes[available_scene_names.index(s)]["token"] for s in val_scenes]
    # )

    if test:
        print(f"test scene: {len(train_scenes)}")
    else:
        print(f"train scene: {len(train_scenes)}, val scene: {len(val_scenes)}")

    train_nusc_infos, val_nusc_infos = _fill_trainval_infos(
        nusc, train_scenes, val_scenes, test, nsweeps=nsweeps, filter_zero=filter_zero, timesteps=timesteps
    )

    if test:
        print(f"test sample: {len(train_nusc_infos)}")
        with open(
            root_path / "{}/infos_test_{:02d}sweeps_withvelo.pkl".format(experiment, nsweeps), "wb"
        ) as f:
            pickle.dump(train_nusc_infos, f)
    else:
        print(
            f"train sample: {len(train_nusc_infos)}, val sample: {len(val_nusc_infos)}"
        )
        with open(
            root_path / "{}/infos_train_{:02d}sweeps_withvelo_filter_{}.pkl".format(experiment, nsweeps, filter_zero), "wb"
        ) as f:
            pickle.dump(train_nusc_infos, f)
        with open(
            root_path / "{}/infos_val_{:02d}sweeps_withvelo_filter_{}.pkl".format(experiment, nsweeps, filter_zero), "wb"
        ) as f:
            pickle.dump(val_nusc_infos, f)


def eval_main(nusc, eval_version, res_path, eval_set, output_dir, forecast, tp_pct, static_only,
              cohort_analysis, topK, root, association_oracle, nogroup):
    # nusc = NuScenes(version=version, dataroot=str(root_path), verbose=True)
    cfg = config_factory(eval_version)

    nusc_eval = NuScenesEval(
        nusc,
        config=cfg,
        result_path=res_path,
        eval_set=eval_set,
        output_dir=output_dir,
        verbose=True,
        forecast=forecast,
        tp_pct=tp_pct,
        static_only=static_only,
        cohort_analysis=cohort_analysis,
        topK=topK,
        root=root,
        association_oracle=association_oracle,
        nogroup=nogroup
    )
    metrics_summary = nusc_eval.main(plot_examples=10,cohort_analysis=cohort_analysis)
