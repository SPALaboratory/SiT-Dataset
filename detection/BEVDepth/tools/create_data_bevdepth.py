# Copyright (c) OpenMMLab. All rights reserved.
import pickle
from os import path as osp
import argparse
import numpy as np
from nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion
from tools.data_converter import sit_converter
import copy



classes = [
    'pedestrian', 'car', 'truck', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle',  'traffic_cone'
]

def sit_data_prep(root_path,
                       info_prefix,
                       version,
                       dataset,
                       max_sweeps=10,
                       overfit=False):
    """Prepare data related to SiT dataset.
    ###########################################
    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        version (str): Dataset version.
        max_sweeps (int, optional): Number of input consecutive frames. 
            Default: 10
    """
    sit_converter.create_sit_infos(
        root_path, info_prefix, version=version, max_sweeps=max_sweeps,overfit=overfit)

    
def sit_add_ann_adj_info(extra_tag,version,overfit=False):
    sit_version = version
    from mmdet3d.datasets.sit_bev_dataset import SiT_bev_Dataset as SiT_Dataset
    if sit_version == 'sit-test':
        set='test'
        sit_dataset = SiT_Dataset(
            ann_file=f'./data/sit/{extra_tag}_infos_{set}.pkl',
            pipeline=[],
            classes=SiT_Dataset.CLASSES,
            test_mode=False,
            modality=None,
            )
        dataset_path = f'./data/sit/{extra_tag}_infos_{set}.pkl'
        dataset = pickle.load(open(dataset_path, 'rb'))
        class_name_to_label = {name: i for i, name in enumerate(SiT_Dataset.CLASSES)}

        for id in range(len(dataset['infos'])): 
            if id % 10 == 0:
                print('%d/%d' % (id, len(dataset['infos'])))
            info = dataset['infos'][id]
            
            # get sweep adjacent frame info
            sample = sit_dataset.get('sample', info['token'])
            box=sample['gt_boxes']
            velocity=sample['gt_velocity']
            if len(velocity) != len(box):
                velocity = np.zeros((box.shape[0], 2))
            gt_boxes = np.concatenate((box, velocity), axis=1)
            gt_boxes = gt_boxes.tolist()  
            gt_names =sample['gt_names']
            gt_names = gt_names.tolist()  # Convert to list
            gt_labels = [class_name_to_label[name] for name in gt_names]

            ann_infos = [gt_boxes, gt_labels]
            dataset['infos'][id]['ann_infos'] = ann_infos
            dataset['infos'][id]['scene_token'] = sample['token']
            dataset['infos'][id]['occ_path'] = ' '
        with open('./data/sit/%s_infos_%s.pkl' % (extra_tag, set),
                'wb') as fid:
            pickle.dump(dataset, fid)
        
    else:
        if overfit:
            set='train'
            sit_dataset = SiT_Dataset(
                ann_file=f'./data/sit/{extra_tag}_infos_{set}_overfit.pkl',
                pipeline=[],
                classes=SiT_Dataset.CLASSES,
                test_mode=False,
                modality=None,
                )
            dataset_path = f'./data/sit/{extra_tag}_infos_{set}_overfit.pkl'
            dataset = pickle.load(open(dataset_path, 'rb'))
            class_name_to_label = {name: i for i, name in enumerate(SiT_Dataset.CLASSES)}

            for id in range(len(dataset['infos'])): 
                if id % 10 == 0:
                    print('%d/%d' % (id, len(dataset['infos'])))
                info = dataset['infos'][id]
                
                # get sweep adjacent frame info
                sample = sit_dataset.get('sample', info['token'])
                box=sample['gt_boxes']
                velocity=sample['gt_velocity']
                if len(velocity) != len(box):
                    velocity = np.zeros((box.shape[0], 2))
                gt_boxes = np.concatenate((box, velocity), axis=1)
                gt_boxes = gt_boxes.tolist()  
                gt_names =sample['gt_names']
                gt_names = gt_names.tolist()  # Convert to list
                gt_labels = [class_name_to_label[name] for name in gt_names]

                ann_infos = [gt_boxes, gt_labels]
                dataset['infos'][id]['ann_infos'] = ann_infos
                dataset['infos'][id]['scene_token'] = sample['token']
                dataset['infos'][id]['occ_path'] = ' '
            with open('./data/sit/%s_infos_%s.pkl' % (extra_tag, set),
                    'wb') as fid:
                pickle.dump(dataset, fid)
        
        else:
            for set in ['train', 'val']:
                sit_dataset = SiT_Dataset(
                    ann_file=f'./data/sit/{extra_tag}_infos_{set}.pkl',
                    pipeline=[],
                    classes=SiT_Dataset.CLASSES,
                    test_mode=False,
                    modality=None,
                    )
                dataset_path = f'./data/sit/{extra_tag}_infos_{set}.pkl'
                dataset = pickle.load(open(dataset_path, 'rb'))
                class_name_to_label = {name: i for i, name in enumerate(SiT_Dataset.CLASSES)}

                for id in range(len(dataset['infos'])): 
                    if id % 10 == 0:
                        print('%d/%d' % (id, len(dataset['infos'])))
                    info = dataset['infos'][id]
                    
                    # get sweep adjacent frame info
                    sample = sit_dataset.get('sample', info['token'])
                    box=sample['gt_boxes']
                    velocity=sample['gt_velocity']
                    if len(velocity) != len(box):
                        velocity = np.zeros((box.shape[0], 2))
                    gt_boxes = np.concatenate((box, velocity), axis=1)
                    gt_boxes = gt_boxes.tolist()  
                    gt_names =sample['gt_names']
                    gt_names = gt_names.tolist()  # Convert to list
                    gt_labels = [class_name_to_label[name] for name in gt_names]

                    ann_infos = [gt_boxes, gt_labels]
                    dataset['infos'][id]['ann_infos'] = ann_infos
                    dataset['infos'][id]['scene_token'] = sample['token']
                    dataset['infos'][id]['occ_path'] = ' '
                with open('./data/sit/%s_infos_%s.pkl' % (extra_tag, set),
                        'wb') as fid:
                    pickle.dump(dataset, fid)
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SiT Data Preparation')
    parser.add_argument('--version', type=str, required=True, help='Dataset version')
    parser.add_argument('--overfit', action='store_true', help='Overfit mode')
    args = parser.parse_args()
    
    dataset = 'SiT_bev_Dataset'

    version = args.version
    root_path = './data/sit'
    extra_tag = 'sit_bev'
    overfit = args.overfit
    print('sit_data_prepare_start!')
    sit_data_prep(
        root_path=root_path,
        info_prefix=extra_tag,
        version=version,
        dataset=dataset,
        max_sweeps=10,
        overfit=overfit)
    
    print('sit_data_prepare_complete!')

    print('sit_add_ann_adj_info_start!')
    sit_add_ann_adj_info(extra_tag, version=version, overfit=overfit) 
    print('sit_add_ann_infos complete!')