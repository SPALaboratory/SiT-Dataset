import os
import numpy as np
import json
import math

def euler_from_quaternion(x, y, z, w):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
     
        return roll_x, pitch_y, yaw_z # in radians

with open('./results_pred_nusc.json') as json_file:
    json_data = json.load(json_file)

jkeys = list(json_data['results'].keys())

jdata = json_data['results']

class_name = ['car', 'bicycle', 'motorcycle', 'pedestrian']

file_dict = {}

for i in range(len(jkeys)):
    for j in range(len(jdata[jkeys[i]])):
        cur_dict = jdata[jkeys[i]][j]
        sample_token = cur_dict['sample_token'].split('*')
        if sample_token[0] not in file_dict.keys():
            file_dict[sample_token[0]] = {}
        if sample_token[1] not in file_dict[sample_token[0]].keys():
            file_dict[sample_token[0]][sample_token[1]] = []
        if sample_token[2] not in file_dict[sample_token[0]][sample_token[1]]:
            file_dict[sample_token[0]][sample_token[1]].append(sample_token[2])

split_dict = {'train': {},
              'val': {},
              'test': {}
             }
for j in split_dict.keys():
    try:
        split_set = open('./ImageSets/'+j+'.txt').read().split('\n')[:-1]
    except:
        continue
    split_list = []
    for i in split_set:
        split_list.append(i.split('*'))
    for i in split_list:
        if i[0] not in split_dict[j].keys():
            split_dict[j][i[0]] = {}
        if i[1] not in split_dict[j][i[0]].keys():
            split_dict[j][i[0]][i[1]] = []
        split_dict[j][i[0]][i[1]].append(i[2])

file_idx = 0
for k, j  in enumerate(split_dict['val']):
    for aa, bb in enumerate(split_dict['val'][j]):
        oxt_save_list = []
        for cc, dd in enumerate(split_dict['val'][j][bb]):
            traj = open('./'+j+'/'+bb+'/ego_trajectory/'+dd+'.txt').read().split(',')
            traj = list(map(float, traj))
            oxt_ele = [cc, dd]
            oxt_ele.extend(traj)
            oxt_save_list.append(list(map(str, oxt_ele)))
        np.savetxt('../tracking/val/oxts/'+"%04d" % int(file_idx)+'.txt', oxt_save_list, fmt='%s')
        file_idx += 1

det_dict = {}

for i in range(len(jkeys)):
    for j in range(len(jdata[jkeys[i]])):
        cur_dict = jdata[jkeys[i]][j]
        sample_token = cur_dict['sample_token'].split('*')
        if sample_token[0] not in det_dict.keys():
            det_dict[sample_token[0]] = {}
        if sample_token[1] not in det_dict[sample_token[0]].keys():
            det_dict[sample_token[0]][sample_token[1]] = {'car': [],
                                                         'bicycle': [],
                                                         'motorcycle': [],
                                                         'pedestrian': []
                                                         }
        frame_id = file_dict[sample_token[0]][sample_token[1]].index(sample_token[2])
        label = class_name.index(cur_dict['detection_name'])
        width, length, height = cur_dict['size'] 
        loc_x, loc_y, loc_z = cur_dict['translation']
        score = cur_dict['detection_score']

        rot_w, rot_x, rot_y, rot_z = cur_dict['rotation']
        roll_x, pitch_y, yaw_z = euler_from_quaternion(rot_x, rot_y, rot_z, rot_w)

        ap_list = [frame_id, ',', label, ',', -1, ',', -1, ',', -1, ',', -1, ',', score, ',', height, ',',\
                                          width, ',', length, ',', loc_x, ',', loc_y, ',', str(-float(loc_z)), ',', yaw_z, ',', -10, '\n']
        det_dict[sample_token[0]][sample_token[1]][cur_dict['detection_name']].append(map(str, ap_list))

for i in split_dict.keys():
    if split_dict[i] == file_dict:
        save_set = i
        break

f_idx = 0
for aa, bb in enumerate(det_dict.keys()):
    for cc, dd in enumerate(det_dict[bb].keys()):
        for ee in det_dict[bb][dd].keys():
            os.makedirs('../detection/centerpoint_'+ee+'_'+save_set, exist_ok=True)
            f = open('../detection/centerpoint_'+ee+'_'+save_set+'/'+"%04d" % int(f_idx)+'.txt', 'w')
            for k in range(len(det_dict[bb][dd][ee])):
                f.writelines(det_dict[bb][dd][ee][k])
            f.close()
        f_idx += 1
