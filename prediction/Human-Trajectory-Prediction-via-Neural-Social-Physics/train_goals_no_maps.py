import numpy as np
import torch
import yaml
from model_goals import *
from utils import *
import torch.optim as optim
from torch.autograd import Variable
import argparse
import pickle
import os

def train(train_batches):

    model.train()
    total_loss = 0
    criterion = nn.MSELoss()

    shuffle_index = torch.randperm(len(train_batches)) #Q) 216: len(train_batches)
    for k in shuffle_index[:30]: #Q) 다 안함
        traj = train_batches[k] #(409, 20, 4)
        traj = torch.squeeze(torch.DoubleTensor(traj).to(device)) #torch.Size([409, 20, 4])
        y = traj[:, params['past_length']:, :2] #peds*future_length*2

        dest = y[:, -1, :].to(device)
        #dest_state = traj[:, -1, :]
        future = y.contiguous().to(device) #torch.Size([409, 12, 2])

        future_vel = (dest - traj[:, params['past_length'] - 1, :2]) / (torch.tensor(params['future_length']).to(device) * 0.1) #peds*2 #21JY
        future_vel_norm = torch.norm(future_vel, dim=-1) #peds
        initial_speeds = torch.unsqueeze(future_vel_norm, dim=-1) #peds*1

        num_peds = traj.shape[0]
        numNodes = num_peds
        hidden_states = Variable(torch.zeros(numNodes, params['rnn_size']))
        cell_states = Variable(torch.zeros(numNodes, params['rnn_size']))
        hidden_states = hidden_states.to(device)
        cell_states = cell_states.to(device)

        for m in range(1, params['past_length']):  #Q) 왜 1부터? -> 어차피 첫스텝은 0이기 때문일듯
            current_step = traj[:, m, :2]  # peds*2
            current_vel = traj[:, m, 2:]  # peds*2
            input_lstm = torch.cat((current_step, current_vel), dim=1)  # peds*4
            outputs_features, hidden_states, cell_states = model.forward_lstm(input_lstm, hidden_states, cell_states)

        predictions = torch.zeros(num_peds, params['future_length'], 2).to(device)
        prediction, w_v = model.forward_next_step(current_step, current_vel, initial_speeds, dest,
                                                  outputs_features, device=device)
        predictions[:, 0, :] = prediction

        current_step = prediction #peds*2
        current_vel = w_v #peds*2

        for t in range(params['future_length'] - 1):
            input_lstm = torch.cat((current_step, current_vel), dim=1)
            outputs_features, hidden_states, cell_states = model.forward_lstm(input_lstm, hidden_states, cell_states)

            future_vel = (dest - prediction) / ((70-t-1) * 0.1)  # peds*2 #21JY
            future_vel_norm = torch.norm(future_vel, dim=-1)  # peds
            initial_speeds = torch.unsqueeze(future_vel_norm, dim=-1)  # peds*1

            prediction, w_v = model.forward_next_step(current_step, current_vel, initial_speeds, dest,
                                                      outputs_features, device=device)
            predictions[:, t+1, :] = prediction

            current_step = prediction  # peds*2
            current_vel = w_v # peds*2
        optimizer.zero_grad()

        loss = calculate_loss(criterion, future, predictions)
        loss.backward()

        total_loss += loss.item()
        optimizer.step()

    return total_loss / 30 #21JY

def test(traj, generated_goals):
    model.eval()

    with torch.no_grad():
        traj = torch.squeeze(torch.DoubleTensor(traj).to(device))
        generated_goals = torch.DoubleTensor(generated_goals).to(device)

        y = traj[:, params['past_length']:, :2]  # peds*future_length*2
        y = y.cpu().numpy()
        dest = generated_goals

        future_vel = (dest - traj[:, params['past_length'] - 1, :2]) / (torch.tensor(params['future_length']).to(device) * 0.1)  # peds*2 #21JY
        future_vel_norm = torch.norm(future_vel, dim=-1)  # peds
        initial_speeds = torch.unsqueeze(future_vel_norm, dim=-1)  # peds*1

        num_peds = traj.shape[0]
        numNodes = num_peds
        hidden_states = Variable(torch.zeros(numNodes, params['rnn_size']))
        cell_states = Variable(torch.zeros(numNodes, params['rnn_size']))
        hidden_states = hidden_states.to(device)
        cell_states = cell_states.to(device)

        for m in range(1, params['past_length']):  #
            current_step = traj[:, m, :2]  # peds*2
            current_vel = traj[:, m, 2:]  # peds*2
            input_lstm = torch.cat((current_step, current_vel), dim=1)  # peds*4
            outputs_features, hidden_states, cell_states = model.forward_lstm(input_lstm, hidden_states, cell_states)

        predictions = torch.zeros(num_peds, params['future_length'], 2).to(device)

        prediction, w_v = model.forward_next_step(current_step, current_vel, initial_speeds, dest,
                                              outputs_features, device=device)

        predictions[:, 0, :] = prediction

        current_step = prediction #peds*2
        current_vel = w_v #peds*2

        for t in range(params['future_length'] - 1):
            input_lstm = torch.cat((current_step, current_vel), dim=1)
            outputs_features, hidden_states, cell_states = model.forward_lstm(input_lstm, hidden_states, cell_states)

            future_vel = (dest - prediction) / ((70 - t - 1) * 0.1)  # peds*2
            future_vel_norm = torch.norm(future_vel, dim=-1)  # peds
            initial_speeds = torch.unsqueeze(future_vel_norm, dim=-1)  # peds*1

            prediction, w_v = model.forward_next_step(current_step, current_vel, initial_speeds, dest,
                                                  outputs_features, device=device)
            predictions[:, t + 1, :] = prediction

            current_step = prediction  # peds*2
            current_vel = w_v  # peds*2

        predictions = predictions.cpu().numpy()

        test_ade = np.mean(np.linalg.norm(y - predictions, axis = 2), axis=1) # peds
        test_fde = np.linalg.norm((y[:,-1,:] - predictions[:, -1, :]), axis=1) #peds
    return test_ade, test_fde

parser = argparse.ArgumentParser(description='NSP')
parser.add_argument('--gpu_index', '-gi', type=int, default=0)
parser.add_argument('--save_file', '-sf', type=str, default='SDD_goals_no_maps_v1_ICCV.pt')
args = parser.parse_args()

CONFIG_FILE_PATH = '/media/hdd/jyyun/husky/Human-Trajectory-Prediction-via-Neural-Social-Physics/config/sdd_goals.yaml'  # yaml config file containing all the hyperparameters
with open(CONFIG_FILE_PATH) as file:
    params = yaml.load(file, Loader=yaml.FullLoader)
print(params)

dtype = torch.float64
torch.set_default_dtype(dtype)
device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
if torch.cuda.is_available():
    torch.cuda.set_device(args.gpu_index)
print(device)

path_train = '/media/hdd/jyyun/husky/Human-Trajectory-Prediction-via-Neural-Social-Physics/data/SDD/train_pickle/'
scenes_train = os.listdir(path_train)
path_test = '/media/hdd/jyyun/husky/Human-Trajectory-Prediction-via-Neural-Social-Physics/data/SDD/val_pickle/'
scenes_test = os.listdir(path_test)

goals_path = '/media/hdd/jyyun/husky/Human-Trajectory-Prediction-via-Neural-Social-Physics/result_ade1.15422523021698/result_goal20/'
goals_list = os.listdir(goals_path)
# goal_info20 = []
# for goal_list in goals_list:
#     with open(os.path.join(goals_path, goal_list), 'rb') as f:
#         goals = pickle.load(f)
#     goal_info20.append(goals)

train_batches = []
train_traj = []
test_datas = []
test_loca_list = []
for scene_test in scenes_test:
    with open(os.path.join(path_test, scene_test), 'rb') as f:
        test_data = pickle.load(f)
    test_datas.append(test_data[0])
    test_loca_list.append(scene_test)
# for scene_train in scenes_train:
#     with open(os.path.join(path_train, scene_train), 'rb') as f:
#         train_data = pickle.load(f)
#     traj = train_data[0]
#     traj = traj[:, :, :2]
#     train_traj.append(traj)
# for traj in train_traj:
#     traj -= traj[:, :1, :]
# train_traj = augment_data(train_traj) #Q) 할지 말지 결정 필요
# for traj in train_traj:
#     traj_vel = calculate_v(traj)
#     traj_complete_translated = np.concatenate((traj, traj_vel), axis=-1)
#     train_batches.append(traj_complete_translated)

list_all_traj_com_trans_test = []
list_all_goals_trans_test = []
for i, (test_scene, traj) in enumerate(zip(test_loca_list, test_datas)):
    goal_scene = test_scene.split('.')[:-1][0] + '.pkl'
    with open(os.path.join(goals_path, goal_scene), 'rb') as f:
        goals = pickle.load(f)
    predicted_goals = goals
    predicted_goals = np.swapaxes(np.concatenate(predicted_goals).reshape(-1, 20, 70, 2)[:, :, -1, :], 0, 1) #ICCV
# for i, (traj, predicted_goals) in enumerate(zip(test_datas, goal_info20)):
    traj_vel = calculate_v(traj[:, :, :2]) #Q) train은 translation먼저 하고 속도 구함?
    traj_translated = translation(traj[:, :, :2])
    traj_complete_translated = np.concatenate((traj_translated, traj_vel), axis=-1)  # peds*20*4
    predicted_goals_translated = translation_goals(predicted_goals, traj[:, :, :2])  # peds*2
    list_all_traj_com_trans_test.append(traj_complete_translated)
    list_all_goals_trans_test.append(predicted_goals_translated)

traj_complete_translated_test = np.concatenate(list_all_traj_com_trans_test) # peds*20*4
goals_translated_test = np.concatenate(list_all_goals_trans_test, axis=1)  #peds*2

model = NSP(params["input_size"], params["embedding_size"], params["rnn_size"], params["output_size"],  params["enc_dest_state_size"], params["dec_tau_size"])
model = model.double().to(device)

optimizer = optim.Adam(model.parameters(), lr=  params["learning_rate"])

best_ade = 50
best_fde = 50

for e in range(params['num_epochs']):
    total_loss = train(train_batches) #Q) total loss 잘못됨

    ade_20 = np.zeros((20, len(traj_complete_translated_test)))
    fde_20 = np.zeros((20, len(traj_complete_translated_test)))
    for j in range(20):
        test_ade, test_fde= test(traj_complete_translated_test, goals_translated_test[j, :, :])
        ade_20[j, :] = test_ade
        fde_20[j, :] = test_fde
    test_ade = np.mean(np.min(ade_20, axis=0))
    test_fde = np.mean(np.min(fde_20, axis=0))
    print()

    f=open("./txt/train_goals_no_maps_ICCV.txt", 'a') #JY
    # f.write('\n(epoch {}), ade1s = {:.3f}, fde1s = {:.3f}, ade2s = {:.3f}, fde2s = {:.3f}, ade3s = {:.3f}, fde3s = {:.3f}, ade4s = {:.3f}, fde4s = {:.3f}, ade = {:.3f}, fde = {:.3f}\n'\
    #     .format(e, err_epoch_1s, f_err_epoch_1s, err_epoch_2s, f_err_epoch_2s, err_epoch_3s, f_err_epoch_3s, err_epoch_4s, f_err_epoch_4s, err_epoch, f_err_epoch))
    f.write('\n(epoch {}), ade = {:.3f}, fde = {:.3f}\n'\
        .format(e, test_ade, test_fde))
    f.close()

    if best_ade > test_ade:
        print("Epoch: ", e)
        print('################## BEST PERFORMANCE {:0.2f} ########'.format(test_ade))
        best_ade = test_ade
        best_fde = test_fde
        save_path = 'saved_models/' + args.save_file
        torch.save({'hyper_params': params,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                        }, save_path)
        print("Saved model to:\n{}".format(save_path))

    print('epoch:', e)
    print("Train Loss", total_loss)
    print("Test ADE", test_ade)
    print("Test FDE", test_fde)
    print("Test Best ADE Loss So Far", best_ade)
    print("Test Best FDE Loss So Far", best_fde)
