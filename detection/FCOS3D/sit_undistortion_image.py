import os
import yaml
import cv2
import numpy as np
import multiprocessing
from tqdm import tqdm
from pathlib import Path

def process(rgb_img_path, calib_path, save_path, img_list, cam_num):
    
    for idx, frame in enumerate(img_list):
        # print("{} / {}".format(idx, len(img_list)))
        calib_path_ = str(calib_path) + "/{}.txt".format(frame)
        with open(calib_path_, 'r') as f:
            lines = f.read().splitlines()
            P0_intrinsic = np.array([float(info) for info in lines[0].split(' ')[1:10]
                        ]).reshape([3, 3])
            P1_intrinsic = np.array([float(info) for info in lines[1].split(' ')[1:10]
                        ]).reshape([3, 3])
            P2_intrinsic = np.array([float(info) for info in lines[2].split(' ')[1:10]
                        ]).reshape([3, 3])
            P3_intrinsic = np.array([float(info) for info in lines[3].split(' ')[1:10]
                        ]).reshape([3, 3])
            P4_intrinsic = np.array([float(info) for info in lines[4].split(' ')[1:10]
                        ]).reshape([3, 3])
            P0_extrinsic = np.array([float(info) for info in lines[5].split(' ')[1:13]
                        ]).reshape([3, 4])
            P1_extrinsic = np.array([float(info) for info in lines[6].split(' ')[1:13]
                        ]).reshape([3, 4])
            P2_extrinsic = np.array([float(info) for info in lines[7].split(' ')[1:13]
                        ]).reshape([3, 4])
            P3_extrinsic = np.array([float(info) for info in lines[8].split(' ')[1:13]
                        ]).reshape([3, 4])
            P4_extrinsic = np.array([float(info) for info in lines[9].split(' ')[1:13]
                        ]).reshape([3, 4])
            # import pdb;pdb.set_trace()
            P0_distortion = np.array([float(info) for info in lines[11].split(' ')[1:]
                        ])
            P1_distortion = np.array([float(info) for info in lines[12].split(' ')[1:]
                        ])
            P2_distortion = np.array([float(info) for info in lines[13].split(' ')[1:]
                        ])
            P3_distortion = np.array([float(info) for info in lines[14].split(' ')[1:]
                        ])
            P4_distortion = np.array([float(info) for info in lines[15].split(' ')[1:]
                        ])
            intrinsic_list = [P0_intrinsic, P1_intrinsic, P2_intrinsic, P3_intrinsic, P4_intrinsic]
            extrinsic_list = [P0_extrinsic, P1_extrinsic, P2_extrinsic, P3_extrinsic, P4_extrinsic]
            distortion_list = [P0_distortion, P1_distortion, P2_distortion, P3_distortion, P4_distortion]
            
        D = distortion_list[cam_num-1]
        K = intrinsic_list[cam_num-1]
        R = np.eye(3)
        P = K
        
        img = cv2.imread(str(rgb_img_path) + "/{}.png".format(frame))
        
        mapx, mapy = cv2.initUndistortRectifyMap(K, D, R, P, (1920, 1200), cv2.CV_32FC1)
        img = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
        
        cv2.imwrite(str(save_path) +'{}.png'.format(frame), img)
    print("end processing {}".format(save_path))


def main():
    data_path = "./data/sit_full/"

    place_list = os.listdir(data_path)
    place_list.sort()
    n_jobs = 5
    p = multiprocessing.Pool(n_jobs)
    res = []
    for place in place_list:
        if place in ['ImageSets','sit_infos_train.pkl', 'sit_infos_val.pkl', 'sit_infos_train_mono3d.coco.json', 'sit_infos_val_mono3d.coco.json']:
            continue

        place_path = Path(data_path, place)
        scene_list = os.listdir(place_path)
        for scene in scene_list:
            scene_path = Path(place_path, scene)

            cam_path = Path(scene_path, 'cam_img/')
            calib_path = Path(scene_path, 'calib/')
            for cam_num in [1, 2, 3, 4, 5]:
                rgb_img_path = Path(scene_path, "cam_img/{}/data_rgb/".format(cam_num))

                os.makedirs(str(cam_path) + '/{}/data_undist/'.format(cam_num), exist_ok=True)
                save_path = str(cam_path) + '/{}/data_undist/'.format(cam_num)
            
                img_list = os.listdir(str(rgb_img_path))
                img_list = [int(i.split(".png")[0]) for i in img_list]
                img_list.sort()
                res.append(p.apply_async(process, 
                            kwds=dict(
                            rgb_img_path=rgb_img_path,
                            calib_path=calib_path, 
                            save_path=save_path,
                            img_list=img_list,
                            cam_num=cam_num,
                             )))
                
    for r in tqdm(res):
        r.get()




if __name__ == '__main__':
    main()