# %%

import os
import csv
import matplotlib.pyplot as plt
#%matplotlib inline

import json
from shapely.geometry import Polygon, MultiPolygon, LineString, Point, box
from shapely import affinity

from shapely.geometry import Point, LineString, Polygon

import numpy as np
import cv2
import pandas as pd

from typing import Dict, List, Tuple, Callable
Color = Tuple[float, float, float]

from functools import reduce

import pickle as pkl

# %%


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
    
def makedir_recursive(path, src_dir, dst_dir):
    path = path.replace(src_dir, dst_dir)
    path_base = os.path.dirname(path)
    
    if not os.path.exists(path_base):
        os.makedirs(path_base)
    path += '.pkl'
    return path

def get_key(val, my_dict):
    for key, value in my_dict.items():
         if val == value:
             return key

def make_prediction_data(made_dir, tracking_info, pad, track_idx,
                                     data, frameList, numPedsList, pedsList, target_ids, orig_data):
    
    
    for i in range(len(tracking_info[:,0,0])):
        
        valid_bool = np.array(pad[i], dtype=np.bool8)
        valid_idcs = np.arange(valid_bool.shape[0])
        valid_idcs = valid_idcs[valid_bool]
    
        data[dataset_index].append(tracking_info[i,valid_idcs, 1:4])
        frameList[dataset_index].append(tracking_info[i,0,0])
        numPedsList[dataset_index].append(valid_bool.sum())
        pedsList[dataset_index].append(tracking_info[i, valid_idcs, 1].tolist())
        # orig_data.append(tracking_info[i,track_idx])
    target_list = []
    for idx in np.where(pad.sum(axis=0) == 91)[0].tolist():
        if tracking_info[0, idx, 4] != 2.0:
            continue
        else:
            target_list.append(get_key(idx, lookup_table))
    target_ids[dataset_index].append(target_list)
    
    return data, frameList, numPedsList, pedsList, target_ids

def second_interpolate(tracking_info_np, padding_mask, seq_idx, track_id):
    start_point = tracking_info_np[seq_idx-3, lookup_table[track_id]]
    end_point = tracking_info_np[seq_idx, lookup_table[track_id]]
    tracking_info_np[seq_idx-2, lookup_table[track_id], 2:4] = start_point[2:4] + (end_point[2:4] - start_point[2:4]) * 1/3 
    tracking_info_np[seq_idx-1, lookup_table[track_id], 2:4] = start_point[2:4] + (end_point[2:4] - start_point[2:4]) * 2/3
    padding_mask[seq_idx-2, lookup_table[track_id]] = True
    padding_mask[seq_idx-1, lookup_table[track_id]] = True

    all_frame_data[dataset_index][seq_idx-2] = \
        np.append(all_frame_data[dataset_index][seq_idx-2], np.array([track_id, tracking_info_np[seq_idx-2, lookup_table[track_id], 2], tracking_info_np[seq_idx-2, lookup_table[track_id], 3]])).reshape(-1, 3)
    all_frame_data[dataset_index][seq_idx-1] = \
        np.append(all_frame_data[dataset_index][seq_idx-1], np.array([track_id, tracking_info_np[seq_idx-1, lookup_table[track_id], 2], tracking_info_np[seq_idx-1, lookup_table[track_id], 3]])).reshape(-1, 3)
    pedsList_data[dataset_index][seq_idx-2].append(float(track_id))
    pedsList_data[dataset_index][seq_idx-1].append(float(track_id))

    numPeds_data[dataset_index][seq_idx-2] += 1
    numPeds_data[dataset_index][seq_idx-1] += 1

    return tracking_info_np

def first_interpolate(tracking_info_np, padding_mask, seq_idx, track_id):
    start_point = tracking_info_np[seq_idx-2, lookup_table[track_id]]
    end_point = tracking_info_np[seq_idx, lookup_table[track_id]]
    tracking_info_np[seq_idx-1, lookup_table[track_id], 2:4] = start_point[2:4] + (end_point[2:4] - start_point[2:4]) * 1/2
    padding_mask[seq_idx-1, lookup_table[track_id]] = True

    all_frame_data[dataset_index][seq_idx-1] = \
        np.append(all_frame_data[dataset_index][seq_idx-1], np.array([track_id, tracking_info_np[seq_idx-1, lookup_table[track_id], 2], tracking_info_np[seq_idx-1, lookup_table[track_id], 3]])).reshape(-1, 3)
    pedsList_data[dataset_index][seq_idx-1].append(float(track_id))
    numPeds_data[dataset_index][seq_idx-1] += 1

    return tracking_info_np


# %%
color_ego = (255, 0, 0)
color_neighbor = (0, 0, 255)
angle_in_degrees = 90 #param2
n_pred = 91 #46 # hist+cur+pred

trajectory_txt = 'trajectory.txt'
src_dir = 'data'
dst_dir = 'deepenai_0603'
local_path = '/media/hdd/jyyun/husky/social-lstm/preprocess/trajectory_txt'
places = sorted(os.listdir(local_path))
save_path = '/media/hdd/jyyun/husky/social-lstm/data'
col = ['time', 'trackID', 'y', 'x', 'label']
label_dict = {'husky': 1, 'pedestrian': 2, 'car': 3, 'motorcycle': 4}
train_val_list = ['train', 'val', 'test']

csv_file = pd.read_csv('/media/hdd/jyyun/husky/social-lstm/preprocess/csv/csv/Scene_cutting_ICCV.csv')
csv_file = csv_file.fillna('_')
csv_file.iloc[0]

train_val_dict = {}
for train_flag in train_val_list:
    train_val_dict[train_flag] = []
for i in range(len(csv_file)):
    train_val_test = csv_file.iloc[i]['train_val_test']
    date = str(csv_file.loc[i].Date).split(".")[0]
    File_name = csv_file.iloc[i].File_name
    location = csv_file.iloc[i].Location
    cut_id = csv_file.iloc[i].cut_id
    Modified_name = csv_file.iloc[i]['Modified_name']
    offset = csv_file.iloc[i]['Offset Crosscheck']
    paper_name = csv_file.iloc[i]['paper_name']
    paper_location = csv_file.iloc[i]['paper_location']
    selected = csv_file.iloc[i]['selected_camready']
    if selected == 'o':
        train_val_dict[train_val_test].append(paper_name)

total_data = 0
total_num = 0
total_track_num = 0
for train_val in train_val_list:
    train_val_num = 0
    train_val_track_num = 0
    # local_files = sorted(os.listdir(os.path.join(local_path, train_val)))
    
    # containing pedID, x, y
    all_frame_data = []
    # Validation frame data
    valid_frame_data = []
    # frameList_data would be a list of lists corresponding to each dataset
    # Each list would contain the frameIds of all the frames in the dataset
    frameList_data = []
    valid_numPeds_data= []
    # numPeds_data would be a list of lists corresponding to each dataset
    # Ech list would contain the number of pedestrians in each frame in the dataset
    numPeds_data = []
    
    #each list includes ped ids of this frame
    pedsList_data = []
    valid_pedsList_data = []
    # target ped ids for each sequence
    target_ids = []
    orig_data = []
    padding_mask = []

    dataset_index = 0

    preprocessed_data_dict_jy = {}
    all_frame_data_jy = []
    frameList_data_jy = []
    numPeds_data_jy =[]
    pedsList_data_jy = []
    target_ids_jy = []
    orig_data_jy = []

    for place in places:
        loca_files = os.listdir(os.path.join(local_path, place))
            
        for cur_scene in loca_files:
            # if 'sungsu' not in cur_scene:
            #     continue

            if cur_scene not in train_val_dict[train_val]:
                    continue
            
            print(cur_scene)

            cur_abs_scene = os.path.join(local_path, train_val, cur_scene)
            traj_abs_file = os.path.join(local_path, place, cur_scene, trajectory_txt)

            # traj_abs_file = '/media/hdd/jyyun/husky/social-lstm/preprocess/1.make_trjectory_txt/230208_hanyang_plz_1/trajectory.txt'

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
                # line = list(map(float, line))
                full_anno.append(line)
            f = pd.DataFrame(full_anno)
            f.columns = ['time', 'trackID', 'x', 'y', 'label']

            # f = pd.read_csv(traj_abs_file, dtype={'time':'int','trackID':'int' }, delimiter = ' ',  header=None, names=col)
            # target_ids_self = np.array(f.drop_duplicates(subset={'trackID'}, keep='first', inplace=False)['trackID'])
            target_ids_self = np.array(f.drop_duplicates(subset=['trackID'], keep='first', inplace=False)['trackID']) #0603_JY

            data = np.array(f) #(2900, 4)
            orig_data.append(data)

            data = np.swapaxes(data,0,1) #(4, 2900)

            frameList = data[0, :].tolist()
            numFrames = len(frameList)

            frameList_data.append(frameList)
            # Initialize the list of numPeds for the current dataset
            numPeds_data.append([])
            numPeds_data_jy.append([])
            valid_numPeds_data.append([])

            # Initialize the list of numpy arrays for the current dataset
            all_frame_data.append([])
            all_frame_data_jy.append([])
            # Initialize the list of numpy arrays for the current dataset
            valid_frame_data.append([])

            # list of peds for each frame
            pedsList_data.append([])
            pedsList_data_jy.append([])
            valid_pedsList_data.append([])

            target_ids_jy.append([])
            frameList_data_jy.append([])

            target_ids.append(target_ids_self)

            padding_mask = np.zeros((len(np.unique(frameList)), len(target_ids_self)), dtype=bool)
            tracking_info_np = np.zeros((len(np.unique(frameList)), len(target_ids_self), 5)) #timestamp, tracking ID, y, x, label
            lookup_table = dict(zip(target_ids_self, range(0, len(target_ids_self))))

            for ind, frame in enumerate(np.unique(frameList)): #21JY

                pedsInFrame = data[:, data[0, :] == frame]
                pedsList = pedsInFrame[1, :].tolist()

                pedsWithPos = []

                for ped in pedsList:
                    current_x = pedsInFrame[2, pedsInFrame[1, :] == ped][0]
                    current_y = pedsInFrame[3, pedsInFrame[1, :] == ped][0]
                    label = pedsInFrame[4, pedsInFrame[1, :] == ped][0]

                    padding_mask[ind, lookup_table[ped]] = True #21JY
                    tracking_info_np[ind, lookup_table[ped], 0] = frame
                    tracking_info_np[ind, lookup_table[ped], 1] = ped
                    tracking_info_np[ind, lookup_table[ped], 2] = current_x
                    tracking_info_np[ind, lookup_table[ped], 3] = current_y
                    tracking_info_np[ind, lookup_table[ped], 4] = label

                    pedsWithPos.append([ped, current_x, current_y])
                
                all_frame_data[dataset_index].append(np.array(pedsWithPos))
                pedsList_data[dataset_index].append(pedsList)
                numPeds_data[dataset_index].append(len(pedsList))

            for track_idx in target_ids_self:
        
                start_OK = False
                first_none = False
                second_none = False
                for seq_idx in range(len(np.unique(frameList))):
                    # print(seq_idx, padding_mask[seq_idx, track_idx], tracking_info_np[seq_idx, track_idx])
                    if not padding_mask[seq_idx, lookup_table[track_idx]] and first_none == False:
                        first_none = True
                        # print('hello1')
                        continue
                    elif not padding_mask[seq_idx, lookup_table[track_idx]] and second_none == False:
                        second_none = True
                        # print('hello2')
                        continue
                    elif not padding_mask[seq_idx, lookup_table[track_idx]]:
                        first_none = True
                        second_none = True
                        start_OK = False
                        # print('hello3')
                        continue
                    else:
                        # print('hello4')
                        if first_none == True and second_none == True and start_OK == True:
                            tracking_info_np = second_interpolate(tracking_info_np, padding_mask, seq_idx, track_idx)
                        elif first_none == True and second_none == False and start_OK == True:
                            tracking_info_np = first_interpolate(tracking_info_np, padding_mask, seq_idx, track_idx)
                        
                        first_none = False
                        second_none = False
                        start_OK = True
                        continue
            
            made_dir_jy = makedir_recursive(cur_abs_scene, src_dir, dst_dir)
            track_num = 0
            for seq_idx in range(len(np.unique(frameList))):
                for track_idx in target_ids_self:
                    if_break = False
                    if padding_mask[seq_idx:seq_idx+n_pred, lookup_table[track_idx]].sum() != n_pred or tracking_info_np[0, lookup_table[track_idx], 4] != 2.0:
                        continue
                    else:
                        track_num +=1
                        all_frame_data_jy, frameList_data_jy, numPeds_data_jy, pedsList_data_jy, target_ids_jy = \
                            make_prediction_data(made_dir_jy, tracking_info_np[seq_idx:seq_idx+n_pred:,], padding_mask[seq_idx:seq_idx+n_pred],
                                            lookup_table[track_idx], all_frame_data_jy, frameList_data_jy, numPeds_data_jy, pedsList_data_jy, target_ids_jy, orig_data_jy)
                        if_break == True
                        break
            dataset_index += 1
        
    f = open(os.path.join(save_path, train_val, "trajectories_" + train_val + ".cpkl"), "wb")
    pkl.dump((all_frame_data_jy, frameList_data_jy, numPeds_data_jy, valid_numPeds_data, valid_frame_data, pedsList_data_jy, valid_pedsList_data, target_ids_jy, orig_data), f, protocol=2)
    f.close()