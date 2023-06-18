import pandas as pd
import yaml
import argparse
import torch
from model_spa_v1 import YNet



CONFIG_FILE_PATH = 'config/spa_0307v1_2.yaml'  # yaml config file containing all the hyperparameters
EXPERIMENT_NAME = 'spa_0307v1_2_temp'  # arbitrary name for this experiment
DATASET_NAME = 'spa'

TRAIN_DATA_PATH = 'data/SPA/deepen_v1/train.pkl'
TRAIN_IMAGE_PATH = 'data/SPA/raster/v1_raster_img'
VAL_DATA_PATH = 'data/SPA/deepen_v1/val.pkl'
VAL_IMAGE_PATH = 'data/SPA/raster/v1_raster_img'
OBS_LEN = 11  # in timesteps
OBS_LEN = 11  # in timesteps
PRED_LEN = 35  # in timesteps
NUM_GOALS = 20  # K_e
NUM_TRAJ = 1  # K_a

BATCH_SIZE = 4

with open(CONFIG_FILE_PATH) as file:
    params = yaml.load(file, Loader=yaml.FullLoader)



experiment_name = CONFIG_FILE_PATH.split('.yaml')[0].split('config/')[1]
# import pickle5 as pickle
import pickle5 as pickle
with open(TRAIN_DATA_PATH, 'rb') as f:
    df_train = pickle.load(f)
with open(VAL_DATA_PATH, 'rb') as f:
    df_val = pickle.load(f)
# df_train = pd.read_pickle(TRAIN_DATA_PATH)
# df_val = pd.read_pickle(VAL_DATA_PATH)
model = YNet(obs_len=OBS_LEN, pred_len=PRED_LEN, params=params)

model.train(df_train, df_val, params, train_image_path=TRAIN_IMAGE_PATH, val_image_path=VAL_IMAGE_PATH,
            experiment_name=EXPERIMENT_NAME, batch_size=BATCH_SIZE, num_goals=NUM_GOALS, num_traj=NUM_TRAJ, 
            device=None, dataset_name=DATASET_NAME)


