import yaml
from model_for_nsp import YNet
import pickle5 as pickle


CONFIG_FILE_PATH = 'config/sit.yaml'  # yaml config file containing all the hyperparameters
EXPERIMENT_NAME = 'sit_ynet'  # arbitrary name for this experiment
DATASET_NAME = 'sit'

MODEL_WEIGHT = 'pretrained_models/sit_ynet.pt'
MODEL_WEIGHT = 'pretrained_models_lr00001/sit_ynet_weights_bestADE8.22494125366211.pt'

TRAIN_DATA_PATH = 'data/train.pkl'
TRAIN_IMAGE_PATH = 'data/semantic_map/'
VAL_DATA_PATH = 'data/val.pkl'
VAL_IMAGE_PATH = 'data/semantic_map/'
OBS_LEN = 21  # in timesteps
PRED_LEN = 70  # in timesteps
NUM_GOALS = 20  # K_e
NUM_TRAJ = 1  # K_a

BATCH_SIZE = 16

with open(CONFIG_FILE_PATH) as file:
    params = yaml.load(file, Loader=yaml.FullLoader)

experiment_name = CONFIG_FILE_PATH.split('.yaml')[0].split('config/')[1]
with open(TRAIN_DATA_PATH, 'rb') as f:
    df_train = pickle.load(f)
with open(VAL_DATA_PATH, 'rb') as f:
    df_val = pickle.load(f)

model = YNet(obs_len=OBS_LEN, pred_len=PRED_LEN, params=params, model_weight=MODEL_WEIGHT)
model.evaluate(df_train, df_val, params, train_image_path=TRAIN_IMAGE_PATH, val_image_path=VAL_IMAGE_PATH,
            model_weight=MODEL_WEIGHT, batch_size=BATCH_SIZE, num_goals=NUM_GOALS, num_traj=NUM_TRAJ, 
            device=None, dataset_name=DATASET_NAME)


