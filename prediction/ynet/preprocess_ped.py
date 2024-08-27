import os
from os import path as osp
import json
from shapely.geometry import LineString, Point
from shapely import affinity
import numpy as np
import pandas as pd
    

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
    
def xyzrpy_to_rot_mat_string(string, mat_3x3=True):
    xyzrpy = []
    for s in string.split(','):
        if s != "":
            xyzrpy.append(float(s))
    
    xyz = xyzrpy[:3]
    rpy = xyzrpy[3:6]
    R = euler_to_rotmat2([rpy[0], rpy[1], rpy[2]])
    mat = np.concatenate([R, np.array([xyz]).T], axis=1)
    mat = np.concatenate([mat, np.array([[0,0,0,1]])])
    if mat_3x3:
        return make_3x3_mat(mat)
    else:
        return mat

def get_4x4_matrix(label_path):
    with open(label_path, 'r') as f:
        lines = f.readlines()
    return np.array([float(i) for i in lines[0].split(",")]).reshape(4, 4)


def convert_to_homogeneous_and_back(vec, mat):
    x, y, yaw = vec
    H = np.array([
        [np.cos(yaw), -np.sin(yaw), x],
        [np.sin(yaw), np.cos(yaw), y],
        [0, 0, 1]
    ])

    H = mat @ H
    x_prime = H[0, 2]
    y_prime = H[1, 2]
    yaw_prime = np.arctan2(H[1, 0], H[0, 0])
    
    return np.array([x_prime, y_prime, yaw_prime])



def label_to_number(label):
    if label.lower() == 'husky':
        return np.array([[1,0,0,0,0,0,0,0]])
    elif label.lower() == 'pedestrian':
        return np.array([[0,1,0,0,0,0,0,0]])
    elif label.lower() == 'pedestrain_sitting':
        return np.array([[0,0,1,0,0,0,0,0]])
    elif label.lower() == 'car':
        return np.array([[0,0,0,1,0,0,0,0]])
    elif label.lower() == 'motorcyclist':
        return np.array([[0,0,0,0,1,0,0,0]])
    elif label.lower() == 'cyclist':
        return np.array([[0,0,0,0,0,1,0,0]])
    elif label.lower() == 'truck':
        return np.array([[0,0,0,0,0,0,1,0]])
    elif label.lower() == 'bus':
        return np.array([[0,0,0,0,0,0,0,1]])
    
def makedir_recursive(path, src_dir, dst_dir):
    path = osp.join(dst_dir, osp.basename(path))
    path_base = osp.dirname(path)
    if not osp.exists(path_base):
        os.makedirs(path_base)
    path += '.pkl'
    return path
    
    
    

def make_prediction_data(made_dir, tracking_info, pad, track_idx,
                                     data, frameList, numPedsList, pedsList, target_ids, orig_data, pad_list):
    
    data_temp = []
    frameList_temp = []
    numPedsList_temp = []
    pedsList_temp = []
    orig_data_temp = []
    target_ids_temp = []
    for i in range(len(tracking_info[:,0,0])):
        valid_bool = np.array(pad[i], dtype=np.bool8)
        valid_idcs = np.arange(valid_bool.shape[0])
        valid_idcs = valid_idcs[valid_bool]
        # chgd
        data_temp.append(tracking_info[i,valid_idcs])
        frameList_temp.append(tracking_info[i,0,0])
        numPedsList_temp.append(valid_bool.sum())
        pedsList_temp.append(valid_idcs)
        orig_data_temp.append(tracking_info[i,track_idx])
    data.append(tracking_info)
    frameList.append(tracking_info[:,0,0])
    numPedsList.append(np.array(numPedsList_temp))
    pedsList.append(np.array(pedsList_temp))
    orig_data.append(tracking_info[:,track_idx])
    target_ids.append(track_idx)
    pad_list.append(pad)
    
    return data, frameList, numPedsList, pedsList, target_ids, pad_list

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
    return make_3x3_mat(mat)
# open json

old_semantic_mat = {}
old_semantic_mat['Lobby_1'] = xyzypr_to_rot_mat("-17.2599   23.8105 -0.586412    2.54687 -0.0126974 0.00336044")
old_semantic_mat['Lobby_2'] = xyzypr_to_rot_mat("-17.2599   23.8105 -0.586412    2.54687 -0.0126974 0.00336044")
old_semantic_mat['Lobby_3'] = xyzypr_to_rot_mat("-17.2599   23.8105 -0.586412    2.54687 -0.0126974 0.00336044")
old_semantic_mat['Lobby_4'] = xyzypr_to_rot_mat("-17.2599   23.8105 -0.586412    2.54687 -0.0126974 0.00336044")
old_semantic_mat['Lobby_5'] = xyzypr_to_rot_mat("-17.2599   23.8105 -0.586412    2.54687 -0.0126974 0.00336044")
old_semantic_mat['Lobby_6'] = xyzypr_to_rot_mat("-17.2599   23.8105 -0.586412    2.54687 -0.0126974 0.00336044")
old_semantic_mat['Lobby_7'] = xyzypr_to_rot_mat("-17.2599   23.8105 -0.586412    2.54687 -0.0126974 0.00336044")
old_semantic_mat['Lobby_8'] = xyzypr_to_rot_mat("-17.2599   23.8105 -0.586412    2.54687 -0.0126974 0.00336044")
old_semantic_mat['Lobby_9'] = xyzypr_to_rot_mat("-17.2599   23.8105 -0.586412    2.54687 -0.0126974 0.00336044")

old_semantic_mat['Corridor_1'] = xyzypr_to_rot_mat("-1.82975 -2.26758 0.106983 0.0993932 0.0123184 -0.025678")
old_semantic_mat['Corridor_4'] = xyzypr_to_rot_mat("-1.82975 -2.26758 0.106983 0.0993932 0.0123184 -0.025678")
old_semantic_mat['Corridor_2'] = xyzypr_to_rot_mat("-1.82975 -2.26758 0.106983 0.0993932 0.0123184 -0.025678")
old_semantic_mat['Corridor_3'] = xyzypr_to_rot_mat("-1.82975 -2.26758 0.106983 0.0993932 0.0123184 -0.025678")
old_semantic_mat['Corridor_5'] = xyzypr_to_rot_mat("-1.82975 -2.26758 0.106983 0.0993932 0.0123184 -0.025678")
old_semantic_mat['Corridor_6'] = xyzypr_to_rot_mat("24.1508   4.80143 -0.877616   0.224937  0.0154687 -0.0179819")
old_semantic_mat['Corridor_7'] = xyzypr_to_rot_mat("24.1508   4.80143 -0.877616   0.224937  0.0154687 -0.0179819")
old_semantic_mat['Corridor_8'] = xyzypr_to_rot_mat("24.1508   4.80143 -0.877616   0.224937  0.0154687 -0.0179819")
old_semantic_mat['Corridor_9'] = xyzypr_to_rot_mat("24.1508   4.80143 -0.877616   0.224937  0.0154687 -0.0179819")
old_semantic_mat['Corridor_10'] = xyzypr_to_rot_mat("-1.86791  -2.38461 0.0668298 0.134665 -3.08387  3.12782")
old_semantic_mat['Corridor_11'] = xyzypr_to_rot_mat("-1.86791  -2.38461 0.0668298 0.134665 -3.08387  3.12782")
                                
old_semantic_mat['Hallway_1'] = xyzypr_to_rot_mat("-0.378413  0.501727 0.0871115   0.728172  -0.014466 0.00957549")
old_semantic_mat['Hallway_2'] = xyzypr_to_rot_mat("-0.378413  0.501727 0.0871115   0.728172  -0.014466 0.00957549")
old_semantic_mat['Hallway_3'] = xyzypr_to_rot_mat("-0.378413  0.501727 0.0871115   0.728172  -0.014466 0.00957549")
old_semantic_mat['Hallway_4'] = xyzypr_to_rot_mat("-0.378413  0.501727 0.0871115   0.728172  -0.014466 0.00957549")
old_semantic_mat['Hallway_5'] = xyzypr_to_rot_mat("0.88389 -0.931077 0.0167018 1.43376 3.14147 3.10547")
old_semantic_mat['Hallway_6'] = xyzypr_to_rot_mat("0.88389 -0.931077 0.0167018 1.43376 3.14147 3.10547")
old_semantic_mat['Hallway_7'] = xyzypr_to_rot_mat("0.88389 -0.931077 0.0167018 1.43376 3.14147 3.10547")
old_semantic_mat['Hallway_8'] = xyzypr_to_rot_mat("1.5422  0.451621 0.0189186    2.03641 -0.0480602 0.00366217")
old_semantic_mat['Hallway_9'] = xyzypr_to_rot_mat("1.5422  0.451621 0.0189186    2.03641 -0.0480602 0.00366217")
old_semantic_mat['Hallway_10'] = xyzypr_to_rot_mat("1.5422  0.451621 0.0189186    2.03641 -0.0480602 0.00366217")
old_semantic_mat['Hallway_11'] = xyzypr_to_rot_mat("1.5422  0.451621 0.0189186    2.03641 -0.0480602 0.00366217")
old_semantic_mat['Hallway_12'] = xyzypr_to_rot_mat("1.5422  0.451621 0.0189186    2.03641 -0.0480602 0.00366217")

old_semantic_mat['Cafeteria_1'] = xyzypr_to_rot_mat("-0.792777  -8.63622  0.309088    2.80992 0.00318242  0.0193545")
old_semantic_mat['Cafeteria_2'] = xyzypr_to_rot_mat("-0.792777  -8.63622  0.309088    2.80992 0.00318242  0.0193545")
old_semantic_mat['Cafeteria_1'] = xyzypr_to_rot_mat("-0.800868  -8.62386  0.272736      1.32639     0.017442 -0.000945303")
old_semantic_mat['Cafeteria_2'] = xyzypr_to_rot_mat("-0.800868  -8.62386  0.272736      1.32639     0.017442 -0.000945303")
old_semantic_mat['Cafeteria_3'] = xyzypr_to_rot_mat("-4.42124 -6.64816 0.165212    2.61593 -0.0135371 0.00285866")
old_semantic_mat['Cafeteria_4'] = xyzypr_to_rot_mat("-4.42124 -6.64816 0.165212    2.61593 -0.0135371 0.00285866")
old_semantic_mat['Cafeteria_5'] = xyzypr_to_rot_mat("-4.42124 -6.64816 0.165212    2.61593 -0.0135371 0.00285866")
old_semantic_mat['Cafeteria_6'] = xyzypr_to_rot_mat("-4.42124 -6.64816 0.165212    2.61593 -0.0135371 0.00285866")
                                    
old_semantic_mat['Courtyard_1'] = xyzypr_to_rot_mat("-0.794114  -1.37272 0.0167796 2.58499 3.10378 -3.1307")
old_semantic_mat['Courtyard_2'] = xyzypr_to_rot_mat("-0.794114  -1.37272 0.0167796 2.58499 3.10378 -3.1307")
old_semantic_mat['Courtyard_3'] = xyzypr_to_rot_mat("-0.794114  -1.37272 0.0167796 2.58499 3.10378 -3.1307")
old_semantic_mat['Courtyard_4'] = xyzypr_to_rot_mat("-0.11828 0.260273 0.129865 0.969964  3.13062  3.10571")
old_semantic_mat['Courtyard_5'] = xyzypr_to_rot_mat("9.78504    0.72182 -0.0878919  2.59297 -3.13916  3.13277")
old_semantic_mat['Courtyard_6'] = xyzypr_to_rot_mat("0 0 0 0 0 0")
old_semantic_mat['Courtyard_7'] = xyzypr_to_rot_mat("0 0 0 0 0 0")
old_semantic_mat['Courtyard_8'] = xyzypr_to_rot_mat("0 0 0 0 0 0")
old_semantic_mat['Courtyard_9'] = xyzypr_to_rot_mat("0 0 0 0 0 0")

semantic_mat = {}
semantic_mat['Lobby_1'] = xyzypr_to_rot_mat("-17.2722   23.7915 -0.584731     1.44112 -0.00299132   0.0100814")
semantic_mat['Lobby_2'] = xyzypr_to_rot_mat("-17.2722   23.7915 -0.584731     1.44112 -0.00299132   0.0100814")
semantic_mat['Lobby_3'] = xyzypr_to_rot_mat("-17.2722   23.7915 -0.584731     1.44112 -0.00299132   0.0100814")
semantic_mat['Lobby_4'] = xyzypr_to_rot_mat("-17.2722   23.7915 -0.584731     1.44112 -0.00299132   0.0100814")
semantic_mat['Lobby_5'] = xyzypr_to_rot_mat("-17.2722   23.7915 -0.584731     1.44112 -0.00299132   0.0100814")
semantic_mat['Lobby_6'] = xyzypr_to_rot_mat("-17.2722   23.7915 -0.584731     1.44112 -0.00299132   0.0100814")
semantic_mat['Lobby_7'] = xyzypr_to_rot_mat("-17.2722   23.7915 -0.584731     1.44112 -0.00299132   0.0100814")
semantic_mat['Lobby_8'] = xyzypr_to_rot_mat("-17.2722   23.7915 -0.584731     1.44112 -0.00299132   0.0100814")
semantic_mat['Lobby_9'] = xyzypr_to_rot_mat("-17.2722   23.7915 -0.584731     1.44112 -0.00299132   0.0100814")

semantic_mat['Corridor_1'] = xyzypr_to_rot_mat("-1.82721  -2.31141 0.0712436  1.02064 -3.11275  -3.1333")
semantic_mat['Corridor_4'] = xyzypr_to_rot_mat("-1.82721  -2.31141 0.0712436  1.02064 -3.11275  -3.1333")
semantic_mat['Corridor_2'] = xyzypr_to_rot_mat("-1.82721  -2.31141 0.0712436  1.02064 -3.11275  -3.1333")
semantic_mat['Corridor_3'] = xyzypr_to_rot_mat("-1.82721  -2.31141 0.0712436  1.02064 -3.11275  -3.1333")
semantic_mat['Corridor_5'] = xyzypr_to_rot_mat("-1.82721  -2.31141 0.0712436  1.02064 -3.11275  -3.1333")
semantic_mat['Corridor_6'] = xyzypr_to_rot_mat("24.2136   4.83924 -0.897132  1.74878 -3.13003  3.12608")
semantic_mat['Corridor_7'] = xyzypr_to_rot_mat("24.2136   4.83924 -0.897132  1.74878 -3.13003  3.12608")
semantic_mat['Corridor_8'] = xyzypr_to_rot_mat("24.2136   4.83924 -0.897132  1.74878 -3.13003  3.12608")
semantic_mat['Corridor_9'] = xyzypr_to_rot_mat("24.2136   4.83924 -0.897132  1.74878 -3.13003  3.12608")
semantic_mat['Corridor_10'] = xyzypr_to_rot_mat("-1.86353  -2.37633 0.0698851  1.27143 -3.12997  3.08369")
semantic_mat['Corridor_11'] = xyzypr_to_rot_mat("-1.86353  -2.37633 0.0698851  1.27143 -3.12997  3.08369")

semantic_mat['Hallway_1'] = xyzypr_to_rot_mat("-0.335307  0.490096 0.0644958     1.53061  -0.0189423 -0.00241017")
semantic_mat['Hallway_2'] = xyzypr_to_rot_mat("-0.378413  0.501727 0.0871115   0.728172  -0.014466 0.00957549")
semantic_mat['Hallway_3'] = xyzypr_to_rot_mat("-0.378413  0.501727 0.0871115   0.728172  -0.014466 0.00957549")
semantic_mat['Hallway_4'] = xyzypr_to_rot_mat("-0.335307  0.490096 0.0644958     1.53061  -0.0189423 -0.00241017")
semantic_mat['Hallway_5'] = xyzypr_to_rot_mat("0.89041  -0.893881 -0.0107157  0.122374 0.0318579 0.0123532")
semantic_mat['Hallway_6'] = xyzypr_to_rot_mat("0.89041  -0.893881 -0.0107157  0.122374 0.0318579 0.0123532")
semantic_mat['Hallway_7'] = xyzypr_to_rot_mat("0.89041  -0.893881 -0.0107157  0.122374 0.0318579 0.0123532")
semantic_mat['Hallway_8'] = xyzypr_to_rot_mat("1.5422  0.451621 0.0189186    2.03641 -0.0480602 0.00366217")
semantic_mat['Hallway_9'] = xyzypr_to_rot_mat("0.305223   2.4823 0.192273    1.54488 -0.0184657  0.0828688")
semantic_mat['Hallway_10'] = xyzypr_to_rot_mat("0.305223   2.4823 0.192273    1.54488 -0.0184657  0.0828688")
semantic_mat['Hallway_11'] = xyzypr_to_rot_mat("0.305223   2.4823 0.192273    1.54488 -0.0184657  0.0828688")
semantic_mat['Hallway_12'] = xyzypr_to_rot_mat("0.305223   2.4823 0.192273    1.54488 -0.0184657  0.0828688")

semantic_mat['Cafeteria_1'] = xyzypr_to_rot_mat("-0.800868  -8.62386  0.272736      1.32639     0.017442 -0.000945303")
semantic_mat['Cafeteria_2'] = xyzypr_to_rot_mat("-0.800868  -8.62386  0.272736      1.32639     0.017442 -0.000945303")
semantic_mat['Cafeteria_3'] = xyzypr_to_rot_mat("-4.32775 -6.47274 0.178147   0.998814 -0.0080133 -0.0123")
semantic_mat['Cafeteria_4'] = xyzypr_to_rot_mat("-4.32775 -6.47274 0.178147   0.998814 -0.0080133 -0.0123")
semantic_mat['Cafeteria_5'] = xyzypr_to_rot_mat("-4.32775 -6.47274 0.178147   0.998814 -0.0080133 -0.0123")
semantic_mat['Cafeteria_6'] = xyzypr_to_rot_mat("-4.32775 -6.47274 0.178147   0.998814 -0.0080133 -0.0123")
 
semantic_mat['Courtyard_1'] = xyzypr_to_rot_mat("-0.801373  -1.34415  0.014289  2.58479  3.10192 -3.12801")
semantic_mat['Courtyard_2'] = xyzypr_to_rot_mat("-0.801373  -1.34415  0.014289  2.58479  3.10192 -3.12801")
semantic_mat['Courtyard_3'] = xyzypr_to_rot_mat("-0.801373  -1.34415  0.014289  2.58479  3.10192 -3.12801")
semantic_mat['Courtyard_4'] = xyzypr_to_rot_mat("-0.110392  0.275038  0.145271 0.969741  3.13007  3.10399")
semantic_mat['Courtyard_5'] = xyzypr_to_rot_mat("9.78177   0.749531 -0.0500631     2.35755 0.000599851  0.00960307")
semantic_mat['Courtyard_6'] = xyzypr_to_rot_mat("0.00917176 -0.0179223  0.0380164 0.553627 -3.14082  3.13631")
semantic_mat['Courtyard_7'] = xyzypr_to_rot_mat("0.00917176 -0.0179223  0.0380164 0.553627 -3.14082  3.13631")
semantic_mat['Courtyard_8'] = xyzypr_to_rot_mat("0.00917176 -0.0179223  0.0380164 0.553627 -3.14082  3.13631")
semantic_mat['Courtyard_9'] = xyzypr_to_rot_mat("0.00917176 -0.0179223  0.0380164 0.553627 -3.14082  3.13631")


semantic_mat['Subway_Entrance'] = np.array(
[[-6.48716404e-01,  7.61030241e-01, -2.95228193e+06],
 [-7.61030241e-01, -6.48716404e-01,  2.94661654e+06],
 [ 0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
semantic_mat['Three_way_Intersection'] = np.array(
[[-7.64828355e-01, -6.44234109e-01,  2.92933817e+06],
 [ 6.44234109e-01, -7.64828355e-01,  2.97012495e+06],
 [ 0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
semantic_mat['Cafe_street'] = np.array(
[[ 9.10139659e-01,  4.14301583e-01, -2.02022771e+06],
 [-4.14301583e-01,  9.10139659e-01, -3.64826798e+06],
 [ 0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
semantic_mat['Outdoor_Alley'] = np.array(
 [[ 9.99086221e-01, -4.27401851e-02, -1.49732725e+05],
[ 4.27401851e-02,  9.99086221e-01, -4.16723007e+06],
[ 0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
semantic_mat['Crossroad'] =np.array(
 [[ 9.99086221e-01, -4.27401851e-02, -1.49732725e+05],
[ 4.27401851e-02,  9.99086221e-01, -4.16723007e+06],
[ 0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])


def make_prediction_data(made_dir, tracking_info, pad, track_idx,
                                    for_result_ynet_pd_list, ynet_pd_list, metaId, cur_scene, trackId,):
    
    timestamps, n_agent, _ = tracking_info.shape
    i, j = 0 , 0
    for j in range(n_agent):
        if pad[:, j].sum() == 0 and tracking_info[i,j,5] == 1: #ped만 target
            tar = 1
        else:
            tar = 0
        if tar == 1:
            for i in range(timestamps):
                new_dict ={'frame': int(str(metaId)+str(int(tracking_info[i,j,0]))),
                        'trackID':cur_scene + '_' + str(trackId),
                        'x':tracking_info[i,j,2],
                        'y':tracking_info[i,j,3],
                        'sceneId':cur_scene,
                        'metaId':metaId,
                        'class':tracking_info[i,j,5:9]}
                ynet_pd_list.append(new_dict)
                for_result_ynet_pd_list.append(new_dict)
            trackId += 1
    return for_result_ynet_pd_list, ynet_pd_list, trackId


def int_coords(x):
    return np.array(x)[:2].round().astype(np.int32)

def convert_to_pixel_coordinate( point_list, patchbox, angle_in_degrees, canvas_size):
    canvas_h = canvas_size[0]
    canvas_w = canvas_size[1]

    patch_x, patch_y, patch_h, patch_w = patchbox
    scale_height = canvas_h / patch_h
    scale_width = canvas_w / patch_w
    trans_x = -patch_x + patch_w / 2.0
    trans_y = -patch_y + patch_h / 2.0
    if len(point_list) == 1:
        point_list.append(point_list[0])
    polyline = LineString(point_list)

    polyline = affinity.rotate(polyline, angle_in_degrees,
                                    origin=(patch_x, patch_y), use_radians=False)
    polyline = affinity.affine_transform(polyline,
                                            [0.0, 1.0, 1.0, 0.0, trans_y, trans_x])
    polyline = affinity.scale(polyline, xfact=scale_width, yfact=scale_height, origin=(0, 0))
    polyline = np.array([int_coords(poly) for poly in polyline.coords])
    return polyline

def scene_patch_canvas_size(scene):

    patch_canvas = {
        'Cafeteria'                 :(-20,0,65),
        'Corridor'                  :(15,0,135),
        'Lobby'                     :(-10,12,110),
        'Hallway'                   :(15,0,65,),
        'Courtyard'                 :(10,0,75,),
        'Subway_Entrance'           :(60,-20,250),
        'Three_way_Intersection'    :(30,45,270,),
        'Crossroad'                 :(40,0,300),
        'Outdoor_Alley'             :(40,0,300),
        'Cafe_street'               :(-10,-70,320,),
    }
    canvas_size = int(patch_canvas[scene][2]*50/4)
    
    return (patch_canvas[scene][0], patch_canvas[scene][1], patch_canvas[scene][2], patch_canvas[scene][2]),\
        (canvas_size, canvas_size)




dataset = './SiT_dataset'
trajectory_path = './data/trajectory_txt'
os.makedirs(trajectory_path, exist_ok=True)

first_loc_path = "./SiT_dataset/first_loc"
first_loc_path = './first_loc'

for place in os.listdir(dataset):
    print(place)
    if place not in ['Courtyard', 'Subway_Entrance', 'Cafeteria', 'Three_way_Intersection', 'Cafe_street', 'Corridor', 'Lobby', 'Outdoor_Alley', 'Crossroad', 'Hallway']:
        continue


    for scene in os.listdir(osp.join(dataset, place)):
        
        frame_list = os.listdir(osp.join(dataset, place, scene, 'label_3d'))
        scene_label = {}

        if place in list(semantic_mat.keys()):
            file = open(osp.join(first_loc_path, scene+".txt"), "r")
            loc_string = file.readlines()[0]
            pose_init = xyzrpy_to_rot_mat_string(loc_string)
            to_sem_mat = semantic_mat[place]@pose_init
        else:
            to_sem_mat = semantic_mat[scene]@np.linalg.inv(old_semantic_mat[scene])
    
    
        for frame in frame_list:
            husky_file = open(osp.join(dataset, place, scene, "ego_trajectory", frame), 'r')
            husky_loc = np.array(list(map(float, husky_file.readlines()[0].split(',')))).reshape(4,4)
            husky_loc_3x3 = np.eye(3)
            husky_loc_3x3[:2, :2] = husky_loc[:2,:2]
            husky_loc_3x3[:2, 2] = husky_loc[:2, 3]
            husky_loc_on_sem = to_sem_mat@husky_loc_3x3
            scene_label[int(frame.split('.')[0])] = []
            x = husky_loc_on_sem[0,2]
            y = husky_loc_on_sem[1,2]
            yaw = np.arctan2(husky_loc_on_sem[1, 0], husky_loc_on_sem[0, 0])
            scene_label[int(frame.split('.')[0])].append(f"{frame.split('.')[0]} husky 0 {str(x)} {str(y)} {str(yaw)}\n")

            label_file = open(osp.join(dataset, place, scene, 'label_3d', frame), "r") 
            label_list = label_file.readlines()
            label_file.close()
            for cur_label in label_list:
                label = cur_label.split(' ')
                if label[0] == "Pedestrian":
                    track_id = int(label[1].split(":")[1])
                elif label[0] == "Pedestrain_sitting":
                    track_id = int(label[1].split(":")[1]) + 10000
                elif label[0] == 'Car':
                    track_id = int(label[1].split(":")[1]) + 20000
                elif label[0] == 'Motorcyclist':
                    track_id = int(label[1].split(":")[1]) + 30000
                elif label[0] == 'Cyclist':
                    track_id = int(label[1].split(":")[1]) + 40000
                elif label[0] == 'Truck':
                    track_id = int(label[1].split(":")[1]) + 50000
                elif label[0] == 'Bus':
                    track_id = int(label[1].split(":")[1]) + 60000
                try:
                    xyyaw = convert_to_homogeneous_and_back(np.array([float(label[5]), float(label[6]), float(label[8])]), to_sem_mat)
                    scene_label[int(frame.split('.')[0])].append(f"{frame.split('.')[0]} {label[0]} {track_id} {str(xyyaw[0])} {str(xyyaw[1])} {str(xyyaw[2])}\n")
                except:
                    scene_label[int(frame.split('.')[0])] = []
                    xyyaw = convert_to_homogeneous_and_back(np.array([float(label[5]), float(label[6]), float(label[8])]), to_sem_mat)
                    scene_label[int(frame.split('.')[0])].append(f"{frame.split('.')[0]} {label[0]} {track_id} {str(xyyaw[0])} {str(xyyaw[1])} {str(xyyaw[2])}\n")
        for k, value in scene_label.items():
            scene_label[k] = sorted(value)
        scene_label = dict(sorted(scene_label.items()))
        new_label = []
        for k, value in scene_label.items():
            for v_ in value:
                new_label.append(v_)
        os.makedirs(osp.join(trajectory_path, place, scene), exist_ok=True)
        with open(osp.join(trajectory_path, place, scene, "trajectory.txt"), "w") as file:
            for obj in new_label:
                file.writelines(obj)

n_pred = 91 # hist+cur+pred
# train_val_list = ['train', 'val', 'test']
train_val_list = ['train', 'val']
trajectory_txt = 'trajectory.txt'

columns = ['trackId', 'xmin', 'xmax', 'ymin', 'ymax','frame','lost','occluded','generated','label']
trackId = 0
metaId = 0

col = ['time', 'label', 'trackID', 'x', 'y', 'yaw']

trajectory_path = './data/trajectory_txt'
ynet_dst_dir = './data'

with open('./data/split.json', 'r') as json_file:
    train_val_dict = json.load(json_file)
    
places = sorted(os.listdir(trajectory_path))
total_data = 0

# class_dict = {'husky': 0, 'pedestrian': 1, 'car': 2, 'motorcycle': 3}
class_dict = {
'husky' : 0,
'Pedestrian' : 1,
'Pedestrain_sitting' : 2,
'Car' : 3,
'Motorcyclist' : 4,
'Cyclist' : 5,
'Truck' : 6,
'Bus' : 7,
}

idx_label = {
0:'husky',
1:'Pedestrian',
2:'Pedestrain_sitting',
3:'Car',
4:'Motorcyclist',
5:'Cyclist',
6:'Truck',
7:'Bus',
}

os.makedirs(ynet_dst_dir, exist_ok=True)

scene_list = ['Cafeteria', 'Corridor', 'Lobby', 'Hallway', 'Courtyard', 'Subway_Entrance', 'Three_way_Intersection', 'Crossroad', 'Cafe_street', 'Outdoor_Alley']


for train_flag in train_val_list:
    print(train_flag)
    ynet_pd_list = []
    for_result_ynet_pd_list = []
    for place in places:
        loca_files = os.listdir(osp.join(trajectory_path, place))
        for cur_scene in loca_files:
            if cur_scene not in train_val_dict[train_flag]:
                continue

            print(cur_scene)
            
            for scene in scene_list:
                if scene in cur_scene:
                    patchbox, canvas_size = scene_patch_canvas_size(scene)
                    break

            angle_in_degrees = 90 #param2
            trackId = 0
            metaId = 0
            traj_file = osp.join(trajectory_path, place, cur_scene, trajectory_txt)

            f = open(traj_file, 'r')
            lines = f.readlines()
            full_anno = []
            for line in lines:
                line = line.strip().split(' ')
                line[0] = int(line[0])
                line[1] = class_dict[line[1]]
                line[2] = int(line[2])
                line[3] = float(line[3])
                line[4] = float(line[4])
                line[5] = float(line[5])
                full_anno.append(line)

            df = pd.DataFrame(full_anno)
            df.columns = col

            timestamps = list(np.sort(df['time'].unique()))
            timestamps_np = np.array(list(map(int, timestamps)))
            num_timestep = len(timestamps)
            historical_df = df[df['time'].isin(timestamps)]
            actor_ids = list(historical_df['trackID'].unique())
            
            num_nodes = len(actor_ids)

            padding_mask = np.zeros((num_timestep, num_nodes), dtype=bool)
            tracking_info_np = np.zeros((num_timestep, num_nodes, 6)) # timestamp, tracking ID, x, y, yaw, class
            track_id = 0

            for actor_id, actor_df in df.groupby('trackID'):
                node_steps = [timestamps.index(timestamp) for timestamp in actor_df['time']]
                padding_mask[node_steps, track_id] = True
                x_trackID = np.ones(len(node_steps)) * int(actor_df['trackID'].values[0])
                x_timestamp = timestamps_np[node_steps]
                x = np.array(list(map(float, actor_df['x'].values)))
                y = np.array(list(map(float, actor_df['y'].values)))
                yaw = np.array(list(map(float, actor_df['yaw'].values)))
                class_num = actor_df['label'].values[0] 

                x_class = np.ones(len(node_steps)) * int(class_num)
                
                xy_yaw = np.stack([x_timestamp, x_trackID, x, y, yaw, x_class], axis=1)
                tracking_info_np[node_steps, track_id] = xy_yaw
                tracking_info_np[:, track_id, 0] = timestamps_np
                tracking_info_np[:, track_id, 1] = track_id
                tracking_info_np[:, track_id, 5] = class_num
                
                track_id += 1

            timestamps, num_actors, attr = tracking_info_np.shape
            for actor_id in range(num_actors):
                cur_xy = tracking_info_np[padding_mask[:, actor_id], actor_id, 2:4]

                point_list = []
                for j in range(len(cur_xy)):
                    point_list.append(Point(cur_xy[j]))
                polyline = convert_to_pixel_coordinate(point_list, patchbox, angle_in_degrees, canvas_size)
                polyline = polyline.astype(np.float32)

                if cur_xy.shape == (1,2):
                    tracking_info_np[padding_mask[:, actor_id], actor_id, 2:4] = polyline[0]    
                else:
                    tracking_info_np[padding_mask[:, actor_id], actor_id, 2:4] = polyline

            padding_mask = ~padding_mask
            total_time_stamp = padding_mask.shape[0]
            for seq_idx in range(num_timestep):
                for track_idx in range(num_nodes):
                    if_break = False
                    if padding_mask[seq_idx:seq_idx+n_pred, track_idx].sum() != 0 or tracking_info_np[0, track_idx, 5] != 1\
                        or seq_idx+n_pred > total_time_stamp:
                        continue
                    else:
                        for_result_ynet_pd_list, ynet_pd_list, trackId = \
                            make_prediction_data(None, tracking_info_np[seq_idx:seq_idx+n_pred:,], padding_mask[seq_idx:seq_idx+n_pred],
                                            track_idx, for_result_ynet_pd_list, ynet_pd_list, metaId, cur_scene, trackId)
                        metaId+=1
                        if_break == True
                        break
    ynet_pd = pd.DataFrame(ynet_pd_list)
    ynet_pd.to_pickle(osp.join(ynet_dst_dir, train_flag+'.pkl'))
    
