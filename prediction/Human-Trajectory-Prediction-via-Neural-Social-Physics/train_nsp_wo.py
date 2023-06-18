from model_nsp_wo import *
from torch.autograd import Variable
from utils import *
import torch.optim as optim
import argparse
import os
import pickle
import cv2
import yaml
from tqdm import tqdm

def train(path, scenes, isSemanticMap):

    model.train()
    total_loss = 0
    batch_num = 0
    criterion = nn.MSELoss()
    shuffle_index = torch.randperm(len(scenes)) 

    for i in tqdm(shuffle_index):
        scene = scenes[i]
        load_name = path + scene
        with open(load_name, 'rb') as f:
            data = pickle.load(f)
        traj_complete, supplement, first_part = data[0], data[1], data[2]
        traj_complete = np.array(traj_complete) 
        if len(traj_complete.shape) == 1:
            continue
        first_frame_total = traj_complete[:,0,:2]
        traj_translated = translation(traj_complete[:, :, :2])
        traj_complete_translated_total = np.concatenate((traj_translated, traj_complete[:, :, 2:]), axis=-1)
        supplement_translated_total = translation_supp(supplement, traj_complete[:, :, :2])

        batch = 500
        for index in range(0, traj_complete_translated_total.shape[0], batch):
            if index + batch > traj_complete_translated_total.shape[0] and index != 0:
                break

            if index + batch > traj_complete_translated_total.shape[0] and index == 0:
                batch = traj_complete_translated_total.shape[0]

            traj, supplement = torch.DoubleTensor(traj_complete_translated_total[index:index+batch]).to(device), torch.DoubleTensor(supplement_translated_total[index:index+batch]).to(device)
            first_frame = torch.DoubleTensor(first_frame_total[index:index+batch]).to(device)

            if scene.split('_')[1] == 'hanyang':
                scene_name = scene.split('_')[1] + '_' + scene.split('_')[2]
            elif scene.split('_')[1] + '_' + scene.split('_')[2] == 'sungsu_2':
                scene_name = 'sungsu_2'
            else:
                scene_name = scene.split('_')[1]
            semantic_map = cv2.imread(semantic_path_train + (scene_name+'.png')) #21JY
            semantic_map = np.transpose(semantic_map[:, :, 0]) #(1331, 1962)

            y = traj[:, params['past_length']:, :2] #peds*future_length*2
            dest = y[:, -1, :].to(device)
            future = y.contiguous().to(device)

            future_vel = (dest - traj[:, params['past_length'] - 1, :2]) / (torch.tensor(params['future_length']).to(device) * 0.2) #peds*2
            future_vel_norm = torch.norm(future_vel, dim=-1) #peds
            initial_speeds = torch.unsqueeze(future_vel_norm, dim=-1) #peds*1

            num_peds = traj.shape[0]
            numNodes = num_peds

            hidden_states1 = Variable(torch.zeros(numNodes, params['rnn_size']))
            cell_states1 = Variable(torch.zeros(numNodes, params['rnn_size']))
            hidden_states1 = hidden_states1.to(device)
            cell_states1 = cell_states1.to(device)
            hidden_states2 = Variable(torch.zeros(numNodes, params['rnn_size']))
            cell_states2 = Variable(torch.zeros(numNodes, params['rnn_size']))
            hidden_states2 = hidden_states2.to(device)
            cell_states2 = cell_states2.to(device)

            for m in range(1, params['past_length']):  #
                current_step = traj[:, m, :2]  # peds*2
                current_vel = traj[:, m, 2:]  # peds*2
                input_lstm = torch.cat((current_step, current_vel), dim=1)  # peds*4
                outputs_features1, hidden_states1, cell_states1, outputs_features2, hidden_states2, cell_states2  \
                    = model.forward_lstm(input_lstm, hidden_states1, cell_states1, hidden_states2, cell_states2)

            predictions = torch.zeros(num_peds, params['future_length'], 2).to(device)

            coefficients, curr_supp= model.forward_coefficient_people(outputs_features2, supplement[:, 10, :, :], current_step, current_vel, device)  # peds*maxpeds*2, peds*(max_peds + 1)*4
            prediction, w_v = model.forward_next_step(current_step, current_vel, initial_speeds, dest,
                                                    outputs_features1, coefficients, curr_supp, sigma, semantic_map, first_frame, k_env, device=device, isSemanticMap=isSemanticMap)
            predictions[:, 0, :] = prediction

            current_step = prediction #peds*2
            current_vel = w_v #peds*2

            for t in range(params['future_length'] - 1):
                input_lstm = torch.cat((current_step, current_vel), dim=1)  # peds*4
                outputs_features1, hidden_states1, cell_states1, outputs_features2, hidden_states2, cell_states2 \
                    = model.forward_lstm(input_lstm, hidden_states1, cell_states1, hidden_states2, cell_states2)

                future_vel = (dest - prediction) / ((35-t-1) * 0.2) # peds*2
                future_vel_norm = torch.norm(future_vel, dim=-1)  # peds
                initial_speeds = torch.unsqueeze(future_vel_norm, dim=-1)  # peds*1

                coefficients, curr_supp = model.forward_coefficient_people(outputs_features2, supplement[:, 10+t+1, :, :], current_step, current_vel, device)
                prediction, w_v = model.forward_next_step(current_step, current_vel, initial_speeds, dest,
                                                        outputs_features1, coefficients, curr_supp, sigma, semantic_map,
                                                        first_frame, k_env, device=device, isSemanticMap=isSemanticMap)
                predictions[:, t+1, :] = prediction

                current_step = prediction  # peds*2
                current_vel = w_v # peds*2
            optimizer.zero_grad()

            loss = calculate_loss(criterion, future, predictions)
            loss.backward()

            total_loss += loss.item()
            optimizer.step()

            batch_num += 1

    return total_loss / batch_num #len(scenes)

def test(path, scenes, generated_goals, isSemanticMap):
    model.eval()
    all_ade = []
    all_fde = []
    all_ade_1s = []
    all_fde_1s = []
    all_ade_3s = []
    all_fde_3s = []
    all_ade_5s = []
    all_fde_5s = []

    index = 0
    all_traj = []
    all_scenes = []

    final_ade = []
    final_fde = []
    final_ade_1s = []
    final_fde_1s = []
    final_ade_3s = []
    final_fde_3s = []
    final_ade_5s = []
    final_fde_5s = []

    with torch.no_grad():
        for i, scene in tqdm(enumerate(scenes)):
            load_name = path + scene
            with open(load_name, 'rb') as f:
                data = pickle.load(f)
            traj_complete, supplement, first_part = data[0], data[1], data[2]
            traj_complete = np.array(traj_complete)
            if len(traj_complete.shape) == 1:
                index += 1
                continue
            traj_translated = translation(traj_complete[:, :, :2])
            traj_complete_translated_total = np.concatenate((traj_translated, traj_complete[:, :, 2:]), axis=-1)
            supplement_translated_total = translation_supp(supplement, traj_complete[:, :, :2])

            batch = 500
            for index in range(0, traj_complete_translated_total.shape[0], batch):
                if index + batch > traj_complete_translated_total.shape[0] and index != 0:
                    break

                if index + batch > traj_complete_translated_total.shape[0] and index == 0:
                    batch = traj_complete_translated_total.shape[0]

                traj, supplement = torch.DoubleTensor(traj_complete_translated_total[index:index+batch]).to(device), torch.DoubleTensor(
                    supplement_translated_total[index:index+batch]).to(device)
                if scene.split('_')[1] == 'hanyang':
                    scene_name = scene.split('_')[1] + '_' + scene.split('_')[2]
                elif scene.split('_')[1] + '_' + scene.split('_')[2] == 'sungsu_2':
                    scene_name = 'sungsu_2'
                else:
                    scene_name = scene.split('_')[1]
                semantic_map = cv2.imread(semantic_path_test + (scene_name+'.png'))
                semantic_map = np.transpose(semantic_map[:, :, 0])
                y = traj[:, params['past_length']:, :2]  # peds*future_length*2
                y = y.cpu().numpy()
                first_frame = torch.DoubleTensor(traj_complete[index:index+batch][:, 0, :2]).to(device)  # peds*2
                num_peds = traj.shape[0]
                ade_20 = np.zeros((20, len(traj_complete[index:index+batch])))
                fde_20 = np.zeros((20, len(traj_complete[index:index+batch])))
                ade_20_1s = np.zeros((20, len(traj_complete[index:index+batch])))
                fde_20_1s = np.zeros((20, len(traj_complete[index:index+batch])))
                ade_20_3s = np.zeros((20, len(traj_complete[index:index+batch])))
                fde_20_3s = np.zeros((20, len(traj_complete[index:index+batch])))
                ade_20_5s = np.zeros((20, len(traj_complete[index:index+batch])))
                fde_20_5s = np.zeros((20, len(traj_complete[index:index+batch])))
                predictions_20 = np.zeros((20, num_peds, params['future_length'], 2))

                with open(os.path.join(generated_goals, scene.split('.')[0] + '.pkl'), 'rb') as f:
                    goals = pickle.load(f)
                predicted_goals = goals
                predicted_goals = np.swapaxes(np.concatenate(predicted_goals).reshape(-1, 20, 35, 2)[:, :, -1, :], 0, 1)
                for j in range(20):
                    goals_translated = translation_goals(predicted_goals[j][index:index+batch], traj_complete[index:index+batch][:,:,:2]) # 20*peds*2

                    dest = torch.DoubleTensor(goals_translated).to(device)

                    future_vel = (dest - traj[:, params['past_length'] - 1, :2]) / (torch.tensor(params['future_length']).to(device) * 0.2) # peds*2
                    future_vel_norm = torch.norm(future_vel, dim=-1)  # peds
                    initial_speeds = torch.unsqueeze(future_vel_norm, dim=-1)  # peds*1

                    numNodes = num_peds

                    hidden_states1 = Variable(torch.zeros(numNodes, params['rnn_size']))
                    cell_states1 = Variable(torch.zeros(numNodes, params['rnn_size']))
                    hidden_states1 = hidden_states1.to(device)
                    cell_states1 = cell_states1.to(device)
                    hidden_states2 = Variable(torch.zeros(numNodes, params['rnn_size']))
                    cell_states2 = Variable(torch.zeros(numNodes, params['rnn_size']))
                    hidden_states2 = hidden_states2.to(device)
                    cell_states2 = cell_states2.to(device)

                    for m in range(1, params['past_length']): 
                        current_step = traj[:, m, :2]  # peds*2
                        current_vel = traj[:, m, 2:]  # peds*2
                        input_lstm = torch.cat((current_step, current_vel), dim=1)  # peds*4
                        outputs_features1, hidden_states1, cell_states1, outputs_features2, hidden_states2, cell_states2 \
                            = model.forward_lstm(input_lstm, hidden_states1, cell_states1, hidden_states2, cell_states2)

                    predictions = torch.zeros(num_peds, params['future_length'], 2).to(device)

                    coefficients, curr_supp = model.forward_coefficient_people(outputs_features2, supplement[:, 10, :, :], current_step, current_vel, device)  # peds*maxpeds*2, peds*(max_peds + 1)*4

                    prediction, w_v = model.forward_next_step(current_step, current_vel, initial_speeds, dest,
                                                            outputs_features1, coefficients, curr_supp, sigma, semantic_map,
                                                            first_frame, k_env, device=device, isSemanticMap=isSemanticMap)
                    predictions[:, 0, :] = prediction

                    current_step = prediction  # peds*2
                    current_vel = w_v  # peds*2

                    for t in range(params['future_length'] - 1):
                        input_lstm = torch.cat((current_step, current_vel), dim=1)  # peds*4
                        outputs_features1, hidden_states1, cell_states1, outputs_features2, hidden_states2, cell_states2 \
                            = model.forward_lstm(input_lstm, hidden_states1, cell_states1, hidden_states2, cell_states2)

                        future_vel = (dest - prediction) / ((35 - t - 1) * 0.2) # peds*2
                        future_vel_norm = torch.norm(future_vel, dim=-1)  # peds
                        initial_speeds = torch.unsqueeze(future_vel_norm, dim=-1)  # peds*1
                        
                        coefficients, current_supplement = model.forward_coefficient_people(outputs_features2, supplement[:, 10 + t + 1, :, :],
                                                                        current_step, current_vel,
                                                                        device=device)

                        prediction, w_v = model.forward_next_step(current_step, current_vel, initial_speeds, dest,
                                                                outputs_features1, coefficients, curr_supp, sigma,
                                                                semantic_map,
                                                                first_frame, k_env, device=device, isSemanticMap=isSemanticMap)

                        predictions[:, t + 1, :] = prediction

                        current_step = prediction  # peds*2
                        current_vel = w_v  # peds*2

                    predictions = predictions.cpu().numpy()
                    dest = dest.cpu().numpy()

                    # ADE error
                    test_ade = np.mean(np.linalg.norm(y - predictions, axis = 2), axis=1) # peds
                    test_fde = np.linalg.norm((y[:,-1,:] - predictions[:, -1, :]), axis=1) # peds
                    test_ade1s = np.mean(np.linalg.norm(y - predictions, axis = 2)[:, :5], axis=1)
                    test_fde1s = np.linalg.norm((y[:,5,:] - predictions[:, 5, :]), axis=1)
                    test_ade3s = np.mean(np.linalg.norm(y - predictions, axis = 2)[:, :15], axis=1)
                    test_fde3s = np.linalg.norm((y[:,15,:] - predictions[:, 15, :]), axis=1)
                    test_ade5s = np.mean(np.linalg.norm(y - predictions, axis = 2)[:, :25], axis=1)
                    test_fde5s = np.linalg.norm((y[:,25,:] - predictions[:, 25, :]), axis=1)

                    ade_20[j, :] = test_ade
                    fde_20[j, :] = test_fde
                    ade_20_1s[j, :] = test_ade1s
                    fde_20_1s[j, :] = test_fde1s
                    ade_20_3s[j, :] = test_ade3s
                    fde_20_3s[j, :] = test_fde3s
                    ade_20_5s[j, :] = test_ade5s
                    fde_20_5s[j, :] = test_fde5s

                    predictions_20[j] = predictions
                ade_single = np.min(ade_20, axis=0)  # peds
                fde_single = np.min(fde_20, axis=0)  # peds

                ade_single_1s = []
                fde_single_1s = []
                ade_single_3s = []
                fde_single_3s = []
                ade_single_5s = []
                fde_single_5s = []
                for idx, argmin in enumerate(np.argmin(ade_20, axis=0)):
                    ade_single_1s.append(ade_20_1s[argmin, idx])
                    fde_single_1s.append(fde_20_1s[argmin, idx])
                    ade_single_3s.append(ade_20_3s[argmin, idx])
                    fde_single_3s.append(fde_20_3s[argmin, idx])
                    ade_single_5s.append(ade_20_5s[argmin, idx])
                    fde_single_5s.append(fde_20_5s[argmin, idx])

                ade_single_1s = np.stack(ade_single_1s)
                fde_single_1s = np.stack(fde_single_1s)
                ade_single_3s = np.stack(ade_single_3s)
                fde_single_3s = np.stack(fde_single_3s)
                ade_single_5s = np.stack(ade_single_5s)
                fde_single_5s = np.stack(fde_single_5s)

                all_ade.append(ade_single)
                all_fde.append(fde_single)
                all_ade_1s.append(ade_single_1s)
                all_fde_1s.append(fde_single_1s)
                all_ade_3s.append(ade_single_3s)
                all_fde_3s.append(fde_single_3s)
                all_ade_5s.append(ade_single_5s)
                all_fde_5s.append(fde_single_5s)

                all_traj.append(predictions_20)
                all_scenes.append(scene)

            scene_ade = np.mean(np.concatenate(all_ade))
            scene_fde = np.mean(np.concatenate(all_fde))
            scene_ade_1s = np.mean(np.concatenate(all_ade_1s))
            scene_fde_1s = np.mean(np.concatenate(all_fde_1s))
            scene_ade_3s = np.mean(np.concatenate(all_ade_3s))
            scene_fde_3s = np.mean(np.concatenate(all_fde_3s))
            scene_ade_5s = np.mean(np.concatenate(all_ade_5s))
            scene_fde_5s = np.mean(np.concatenate(all_fde_5s))
            final_ade.append(scene_ade)
            final_fde.append(scene_fde)
            final_ade_1s.append(scene_ade_1s)
            final_fde_1s.append(scene_fde_1s)
            final_ade_3s.append(scene_ade_3s)
            final_fde_3s.append(scene_fde_3s)
            final_ade_5s.append(scene_ade_5s)
            final_fde_5s.append(scene_fde_5s)
            
        ade = np.mean(final_ade)
        fde = np.mean(final_fde)
        ade_1s = np.mean(final_ade_1s)
        fde_1s = np.mean(final_fde_1s)
        ade_3s = np.mean(final_ade_3s)
        fde_3s = np.mean(final_fde_3s)
        ade_5s = np.mean(final_ade_5s)
        fde_5s = np.mean(final_fde_5s)
    return ade, fde, ade_1s, fde_1s, ade_3s, fde_3s, ade_5s, fde_5s

parser = argparse.ArgumentParser(description='NSP')

parser.add_argument('--gpu_index', '-gi', type=int, default=0)
parser.add_argument('--save_file', '-sf', type=str, default='SiT_nsp_wo_complete_1_v1')

args = parser.parse_args()

CONFIG_FILE_PATH = 'config/SiT_nsp_wo.yaml'  # yaml config file containing all the hyperparameters
with open(CONFIG_FILE_PATH) as file:
    params = yaml.load(file, Loader=yaml.FullLoader)
print(params)

dtype = torch.float64
torch.set_default_dtype(dtype)
device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
if torch.cuda.is_available():
    torch.cuda.set_device(args.gpu_index)
print(device)


model = NSP(params["input_size"], params["embedding_size"], params["rnn_size"], params["output_size"],  params["enc_size"], params["dec_size"])
model = model.double().to(device)

load_path = 'saved_models/SiT_goals_v1.pt'
checkpoint_trained = torch.load(load_path)
load_path_ini = 'saved_models/SiT_nsp_wo_ini.pt'
checkpoint_ini = torch.load(load_path_ini)
checkpoint_dic = new_point(checkpoint_trained['model_state_dict'], checkpoint_ini['model_state_dict'])
model.load_state_dict(checkpoint_dic, strict=False)

# load_path = 'saved_models/SiT_goals_v1.pt'
# checkpoint_trained = torch.load(load_path)
# model.load_state_dict(checkpoint_trained['model_state_dict'], strict=False)

sigma = torch.tensor(100)

parameter_train = select_para(model)
k_env = torch.tensor(65.0).to(device)
k_env.requires_grad = True

optimizer = optim.Adam([{'params': parameter_train}, {'params': [k_env]}], lr=  params["learning_rate"])

best_ade = 50
best_fde = 50

path_train = 'data/SiT/train_pickle/'
scenes_train = os.listdir(path_train)
path_test = 'data/SiT/val_pickle/'
scenes_test = os.listdir(path_test)
semantic_path_train = 'data/SiT/train_masks/'
semantic_maps_name_train = os.listdir(semantic_path_train)
semantic_path_test = 'data/SiT/val_masks/'
semantic_maps_name_test = os.listdir(semantic_path_test)

goals_path = 'Y-Net_results/result_goal20'
goals_list = os.listdir(goals_path)

for e in tqdm(range(params['num_epochs'])):
    total_loss = train(path_train, scenes_train, isSemanticMap=True)
    #test_ade, test_fde, ade_1s, fde_1s, ade_3s, fde_3s, ade_5s, fde_5s = test(path_test, scenes_test, goals_path, isSemanticMap=True)

    # print()
    # f=open("./txt/train_NSP_wo_v1_0614_norobot.txt", 'a') #JY
    # f.write('\n(epoch {}), ade1s = {:.3f}, fde1s = {:.3f}, ade3s = {:.3f}, fde3s = {:.3f}, ade5s = {:.3f}, fde5s = {:.3f}, ade = {:.3f}, fde = {:.3f}\n'\
    #    .format(e, ade_1s, fde_1s, ade_3s, fde_3s, ade_5s, fde_5s, test_ade, test_fde))
    # f.close()

    if True: #best_ade > test_ade:
        print("Epoch: ", e)
        #print('################## BEST PERFORMANCE {:0.2f} ########'.format(test_ade))
        #best_ade = test_ade
        #best_fde = test_fde
        save_path = 'saved_models/' + args.save_file + '_' + str(e) + '.pt'
        torch.save({'hyper_params': params,
                    'model_state_dict': model.state_dict(),
                    'k_env': k_env
                        }, save_path)
        print("Saved model to:\n{}".format(save_path))


    print('epoch:', e)
    print('k_env:', k_env)
    print("Train Loss", total_loss)
    #print("Test ADE", test_ade)
    #print("Test FDE", test_fde)
    #print("Test Best ADE Loss So Far", best_ade)
    #print("Test Best Min FDE", best_fde)
