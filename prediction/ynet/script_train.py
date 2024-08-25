import yaml
from model import YNet
import pickle5 as pickle


CONFIG_FILE_PATH = 'config/sit.yaml'  # yaml config file containing all the hyperparameters
EXPERIMENT_NAME = 'sit_ynet'  # arbitrary name for this experiment
DATASET_NAME = 'sit'

TRAIN_DATA_PATH = 'data/train.pkl'
TRAIN_IMAGE_PATH = 'data/semantic_map/'
VAL_DATA_PATH = 'data/val.pkl'
VAL_IMAGE_PATH = 'data/semantic_map/'
OBS_LEN = 21  # in timesteps
PRED_LEN = 70  # in timesteps
NUM_GOALS = 20  # K_e
NUM_TRAJ = 1  # K_a

BATCH_SIZE = 8

with open(CONFIG_FILE_PATH) as file:
    params = yaml.load(file, Loader=yaml.FullLoader)

experiment_name = CONFIG_FILE_PATH.split('.yaml')[0].split('config/')[1]

with open(TRAIN_DATA_PATH, 'rb') as f:
    df_train = pickle.load(f)
with open(VAL_DATA_PATH, 'rb') as f:
    df_val = pickle.load(f)
    
model = YNet(obs_len=OBS_LEN, pred_len=PRED_LEN, params=params)
model.train(df_train, df_val, params, train_image_path=TRAIN_IMAGE_PATH, val_image_path=VAL_IMAGE_PATH,
            experiment_name=EXPERIMENT_NAME, batch_size=BATCH_SIZE, num_goals=NUM_GOALS, num_traj=NUM_TRAJ, 
            device=None, dataset_name=DATASET_NAME)


