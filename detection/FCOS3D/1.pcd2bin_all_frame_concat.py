from genericpath import exists
import os
import pypcd.pypcd as pypcd
import numpy as np
from pathlib2 import Path

data_path = "./data/sit_full/"
place_list = os.listdir(data_path)
place_list.sort()
for place in place_list:
    if place in ['samples', 'ImageSets','sweeps','maps','v1.0-sit-trainval']:
        continue

    place_path = data_path + place + "/"
    scene_list = os.listdir(place_path)
    scene_list.sort()
    for scene in scene_list:
        velo_path = place_path + scene + "/velo/concat/data/"
        save_path = place_path + scene + "/velo/concat/bin_data/"
        Path(save_path).mkdir(exist_ok=True, parents=True)
        pcd_list = os.listdir(velo_path)
        pcd_list.sort()
        for _, pcd in enumerate(pcd_list):
            print("===== scene : {} - {} ===== {}/{} ".format(place, scene, _, len(pcd_list)))
            ## Get pcd file
            pc = pypcd.PointCloud.from_path(velo_path + pcd)
            # pc = o3d.io.read_point_cloud(velo_path + pcd)
            # points_32 = np.asarray(pc.points)
            
            

            ## Generate bin file name
            name = pcd.split(".")[0] + ".bin"
            
            ## Get data from pcd (x, y, z, intensity, ring, time)
            np_x = (np.array(pc.pc_data['x'], dtype=np.float32)).astype(np.float32)
            np_y = (np.array(pc.pc_data['y'], dtype=np.float32)).astype(np.float32)
            np_z = (np.array(pc.pc_data['z'], dtype=np.float32)).astype(np.float32)
            np_i = (np.array(pc.pc_data['intensity'], dtype=np.float32)).astype(np.float32)/256

            ## Stack all data    
            points_32 = np.transpose(np.vstack((np_x, np_y, np_z, np_i)))

            ## Save bin file                                    
            points_32.tofile(save_path+name)