import os
import pickle
from tqdm import tqdm
import numpy as np

traj_path = '/media/hdd/jyyun/husky/Human-Trajectory-Prediction-via-Neural-Social-Physics/data/SDD/total_pickle'
# traj_list = os.listdir(traj_path)
goals_path = '/media/hdd/jyyun/husky/Human-Trajectory-Prediction-via-Neural-Social-Physics/result_goal0603/'
folder_list = os.listdir(goals_path)
for folder in tqdm(folder_list):
    goals_list = os.listdir(os.path.join(goals_path, folder))
    for goal_list in goals_list:
        with open(os.path.join(goals_path, folder, goal_list), 'rb') as f:
            goals = pickle.load(f)
        trackID_np = np.stack(goals['trackID'][0::11]).reshape(-1)
        goals_np = np.stack(goals['pred'].values[0::11])
        traj_list = goal_list.split('.')[:-1][0] + '.pickle'
        with open(os.path.join(traj_path, traj_list), 'rb') as ft:
            traj_data = pickle.load(ft)
        goals_info = []
        for traj in traj_data[4]:
            traj_idx = np.where(trackID_np == traj)[0][0]
            goals_info.append(goals_np[traj_idx])
        with open(os.path.join(goals_path, folder, goal_list), 'wb') as fw:
            pickle.dump(goals_info, fw, protocol=pickle.HIGHEST_PROTOCOL)
    