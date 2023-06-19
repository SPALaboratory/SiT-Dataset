import os
import numpy as np
import json
import math
import pickle
import copy

folder_path = './label/'
det_pred = '../../data/spa/convert_det_format/'

with open(det_pred + 'results_pred_nusc.json') as json_file:
    json_data = json.load(json_file)

jkeys = list(json_data['results'].keys())

jdata = json_data['results']

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
    split_set = open(det_pred + 'ImageSets/'+j+'.txt').read().split('\n')[:-1]
    split_list = []
    for i in split_set:
        split_list.append(i.split('*'))
    for i in split_list:
        if i[0] not in split_dict[j].keys():
            split_dict[j][i[0]] = {}
        if i[1] not in split_dict[j][i[0]].keys():
            split_dict[j][i[0]][i[1]] = []
        split_dict[j][i[0]][i[1]].append(i[2])


with open(det_pred + 'results_gt_nusc.json') as json_file:
    gt_data = json.load(json_file)

gkeys = list(gt_data['results'].keys())
gdata = gt_data['results']

gt_dict = {}
for i in range(len(gkeys)):
    cur_dict = gdata[gkeys[i]]
    sample_token = cur_dict[0]['sample_token'].split('*')
    if sample_token[0] not in gt_dict.keys():
        gt_dict[sample_token[0]] = {}
    if sample_token[1] not in gt_dict[sample_token[0]].keys():
        gt_dict[sample_token[0]][sample_token[1]] = {}
    if sample_token[2] not in gt_dict[sample_token[0]][sample_token[1]]:
        gt_dict[sample_token[0]][sample_token[1]][sample_token[2]+'.txt'] = cur_dict

dir_dict = {}
global_id = 0
f_idx = 0
for ij, i in enumerate(gt_dict.keys()):
    for aa, bb in enumerate(gt_dict[i].keys()):
        det_gt_list = os.listdir('./convert_label/'+i+'/'+bb+'/label')
        if i not in dir_dict.keys():
            dir_dict[i] = {}
        if bb not in dir_dict[i].keys():
            dir_dict[i][bb] = []
        frame_idx = 0
        max_id = 0
        for j, k in enumerate(sorted(det_gt_list)):
            if k[:-4] not in split_dict['val'][i][bb]:
                continue
            gt_file = open('./convert_label/'+i+'/'+bb+'/label/'+k)
            gt_data = gt_file.read()
            gt_data = gt_data.split("\n")
            gt_data = gt_data[:-1]
            for l in range(len(gt_data)):
                old_gt = gt_data[l].split(' ')
                new_gt = [str(frame_idx), str(int(old_gt[1])+global_id), old_gt[0], str(0), str(0), str(-1), str(-9999.0), str(-9999.0)]
                new_gt.extend(old_gt[5:])
                for m, n in enumerate(gt_dict[i][bb][k]):
                    size_list = list(map(str, n['size']))
                    old_gt_size = [old_gt[9], old_gt[8], old_gt[7]]
                    if size_list == old_gt_size:
                        trans_list = list(map(str, n['translation']))
                        new_gt[13] = trans_list[0]
                        new_gt[14] = trans_list[1]
                        new_gt[15] = trans_list[2]
                dir_dict[i][bb].append(new_gt)
            frame_idx += 1
            gt_file.close()
        with open(folder_path+"%04d" % int(f_idx)+'.txt', 'w') as f:
            for lines in dir_dict[i][bb]:
                lines.extend('\n')
                line = " ".join(lines) 
                f.writelines(line)
            f.close()
        f_idx += 1
