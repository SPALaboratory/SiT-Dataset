# %%

import os
import csv

import json
from shapely.geometry import Polygon, MultiPolygon, LineString, Point, box
from shapely import affinity

from shapely.geometry import Point, LineString, Polygon

import numpy as np
import pandas as pd

from typing import Dict, List, Tuple, Callable

import pickle as pkl
from tqdm import tqdm
import math
# import cv2

def label_to_number(label, one_hot = False):
    if one_hot:
        if label == 'husky':
            return np.array([[1,0,0,0]])
        elif label == 'Pedestrian':
            return np.array([[0,1,0,0]])
        elif label == 'car':
            return np.array([[0,0,1,0]])
        else:
            return np.array([[0,0,0,1]])
    else:
        if label == 'husky':
            return 1
        elif label == 'Pedestrian':
            return 2
        elif label == 'Car':
            return 3
        else:
            return 4

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
    polyline = affinity.rotate(polyline, 90,
                                    origin=(patch_x, patch_y), use_radians=False)
    polyline = affinity.affine_transform(polyline,
                                            [0.0, 1.0, 1.0, 0.0, trans_y, trans_x])
                                            # [1.0, 0.0, 0.0, 1.0, trans_x, trans_y])
    polyline = affinity.scale(polyline, xfact=scale_width, yfact=scale_height, origin=(0, 0))
    polyline = np.array([int_coords(poly) for poly in polyline.coords])
    return polyline

def scene_patch_canvas_size(scene):
    patch_canvas = {   #+Left  +Down   +Scaledown +Resolution +Clockwise
            'Cafeteria'   :(-20,  0,      65,      65/400*5000,     0),
            'Lobby' :(-10,  12,     110,    110/400*5000,     0),
            'Corridor' :(15,   0,      135,     135/400*5000,     0),
            'Subway_Entrance'         :(60,   -20,    250,    250/400*5000,     0),
            'Crossroad'        :(40,   0,      300,    300/400*5000,     0),
            'Outdoor_Alley'   :(40,   0,      300,    300/400*5000,     0),
            'Hallway'           :(15,    0,      65,    65/400*5000,     0),
            'Courtyard'       :(10,  0,       75,    75/400*5000,     0),
            'Three_way_Intersection'          :(30,    45,      270,    270/400*5000,     0),
            'Cafe_street'      :(-10,    -70,      320,    320/400*5000,     0),
        }
    canvas_size = int(patch_canvas[scene][2]*50/4)
    
    return (patch_canvas[scene][0], patch_canvas[scene][1], patch_canvas[scene][2], patch_canvas[scene][2]),\
        (canvas_size, canvas_size)

n_pred = 91 # hist+cur+pred
train_val_list = ['train', 'val', 'test']
trajectory_txt = 'trajectory.txt'

with open('/media/hdd/jyyun/husky/Human-Trajectory-Prediction-via-Neural-Social-Physics/preprocess/split.json') as f:
    train_val_dict = json.load(f)

patchbox = (0, 0, 400, 400)
angle_in_degrees = 90 
canvas_size = (5000, 5000)
columns = ['trackId', 'xmin', 'xmax', 'ymin', 'ymax','frame','lost','occluded','generated','label']
col = ['time', 'trackID', 'x', 'y', 'label'] 

src_dir = '/media/hdd/jyyun/husky/Human-Trajectory-Prediction-via-Neural-Social-Physics/preprocess/trajectory_txt'
dst_dir = '/media/hdd/jyyun/husky/Human-Trajectory-Prediction-via-Neural-Social-Physics/preprocess/data_SiT'

places = sorted(os.listdir(src_dir))
class_dict = {'husky': 1, 'pedestrian': 2, 'car': 3, 'motorcycle': 4}

os.makedirs(dst_dir, exist_ok=True)
for train_val in tqdm(train_val_list):
    os.makedirs(os.path.join(dst_dir, train_val), exist_ok=True)

for train_flag in train_val_list:
    for place in places:
        loca_files = os.listdir(os.path.join(src_dir, place))
        for cur_scene in loca_files:
            if cur_scene not in train_val_dict[train_flag]:
                continue

            patchbox, canvas_size = scene_patch_canvas_size(place)
            
            traj_abs_file = os.path.join(src_dir, place, cur_scene, trajectory_txt)

            f = open(traj_abs_file, 'r')
            lines = f.readlines()
            full_anno = []
            for line in lines:
                line_orig = line.strip().split(' ')
                line = []
                line.append(int(line_orig[0]))
                line.append(int(line_orig[2]))
                line.append(float(line_orig[3]))
                line.append(float(line_orig[4]))
                line.append(label_to_number(line_orig[1]))
                full_anno.append(line)
            df = pd.DataFrame(full_anno)
            df.columns = col

            df_np = df.to_numpy(dtype='O') #time, trackID, x, y, class
            frames = np.unique(df_np[:, 0]).tolist()
            trackID_uniq = np.unique(df_np[:, 1]).tolist()

            num_nodes = len(trackID_uniq)
            
            frame_data = []
            for frame in frames:
                frame_data.append(df_np[frame == df_np[:, 0], :])
            
            df_np = np.concatenate(frame_data)
            
            import cv2
            from matplotlib.pyplot import plot as plt
            semantic_map = cv2.imread("/media/hdd/jyyun/husky/Human-Trajectory-Prediction-via-Neural-Social-Physics/semantic_map/" + (place+'.png')) #21JY
            semantic_map = np.transpose(semantic_map[:, :, 0])
            for id in trackID_uniq:
                df_np_track = df_np[df_np[:, 1]==id]
                point_list_track = []
                for j in range(df_np_track.shape[0]):
                    point_list_track.append(Point(df_np_track[j, 2:4]))
                polyline_track = convert_to_pixel_coordinate(point_list_track, patchbox, angle_in_degrees, canvas_size)
                polyline_track = polyline_track.astype(np.float32)
                
                
                # cv2.imshow("se", semantic_map)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                # semantic_map = np.transpose(semantic_map[:, :, 0]) #(1331, 1962)
            
                # cv2.polylines(semantic_map, [np.int32(polyline_track)], False, (0,0,255), 4)
                
                if df_np_track[:, 2:4].shape == (1,2):
                    df_np[df_np[:, 1]==id, 2:4] = polyline_track[0]    
                else:
                    df_np[df_np[:, 1]==id, 2:4] = polyline_track
            
            # cv2.imshow("se", semantic_map)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            save_path = os.path.join(os.path.join(dst_dir, train_flag), cur_scene, 'processed.npy')
            if not os.path.exists(os.path.join(os.path.join(dst_dir, train_flag), cur_scene)):
                os.makedirs(os.path.join(os.path.join(dst_dir, train_flag), cur_scene))
            np.save(save_path, df_np)
        

