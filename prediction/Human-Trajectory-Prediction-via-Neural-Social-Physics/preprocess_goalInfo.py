import os
import pickle
from tqdm import tqdm
import numpy as np

traj_path = 'preprocess/processed_data_SiT/train_pickle'
traj_path2 = 'preprocess/processed_data_SiT/val_pickle'
# traj_list = os.listdir(traj_path)
goals_path = 'result_0/'
target_path = 'result_SiT/'
folder_list = os.listdir(goals_path)
for folder in tqdm(folder_list):
    goals_list = os.listdir(os.path.join(goals_path, folder))
    for goal_list in tqdm(goals_list):
        try:
            with open(os.path.join(goals_path, folder, goal_list), 'rb') as f:
                goals = pickle.load(f)
            trackID_np = np.stack(goals['trackID'][0::21]).reshape(-1)
            goals_np = np.stack(goals['pred'].values[0::21])
            traj_list = goal_list.split('.')[:-1][0] + '.pickle'
            try:
                with open(os.path.join(traj_path, traj_list), 'rb') as ft:
                    traj_data = pickle.load(ft)
            except:
                with open(os.path.join(traj_path2, traj_list), 'rb') as ft:
                    traj_data = pickle.load(ft)
            goals_info = []
            track_Id_list = []
            for traj in traj_data[4]:
                track_Id_list.append(int(traj.split('_')[-1]))
                traj_idx = np.where(trackID_np == traj)[0][0]
                goals_info.append(goals_np[traj_idx])
            with open(os.path.join(target_path, folder, goal_list), 'wb') as fw:
                pickle.dump(goals_info, fw, protocol=pickle.HIGHEST_PROTOCOL)
            # with open(os.path.join('track_id', goal_list), 'wb') as ft:
            #     pickle.dump(track_Id_list, ft, protocol=pickle.HIGHEST_PROTOCOL)
        except:
            continue