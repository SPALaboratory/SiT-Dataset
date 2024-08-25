from itertools import chain
from tracemalloc import get_object_traceback
import numpy as np
from numpy.lib.histograms import histogram_bin_edges
from pyrsistent import v
from skimage.util.dtype import img_as_int
import seaborn as sns
import matplotlib.pyplot as plt
import os
from skimage import io
from matplotlib.lines import Line2D
import cv2

def txt_read(path):
    with open(path) as f: 
        a = f.read().splitlines()
    return a

def get_ego_matrix(label_path):
    with open(label_path, 'r') as f:
        lines = f.readlines()
    return np.array([float(i) for i in lines[0].split(",")]).reshape(4, 4)

def rotmat_to_euler(rot_mat):
    sy = np.sqrt(rot_mat[0, 0] * rot_mat[0, 0] + rot_mat[1, 0] * rot_mat[1, 0])
    singular = sy < 1e-6
    if not singular:
        roll = np.arctan2(rot_mat[2, 1], rot_mat[2, 2])
        pitch = np.arctan2(-rot_mat[2, 0], sy)
        yaw = np.arctan2(rot_mat[1, 0], rot_mat[0, 0])
    else:
        roll = np.arctan2(-rot_mat[1, 2], rot_mat[1, 1])
        pitch = np.arctan2(-rot_mat[2, 0], sy)
        yaw = 0
    return np.array([roll, pitch, yaw])

def box_center_to_corner_3d(centers, dims, angles):
    
    translation = centers[:, 0:3]
    l, w, h = dims[:, 0], dims[:, 1], dims[:, 2]
    rotation = angles

    # Create a bounding box outline
    x_corners = np.array([[l_ / 2, l_ / 2, -l_ / 2, -l_ / 2, l_ / 2, l_ / 2, -l_ / 2, -l_ / 2] for l_ in l])
    y_corners = np.array([[w_ / 2, -w_ / 2, -w_ / 2, w_ / 2, w_ / 2, -w_ / 2, -w_ / 2, w_ / 2] for w_ in w])
    z_corners = np.array([[-h_ / 2, -h_ / 2, -h_ / 2, -h_ / 2, h_ / 2, h_ / 2, h_ / 2, h_ / 2] for h_ in h])
    bounding_box = np.array([np.vstack([x_corners[i], y_corners[i], z_corners[i]]) for i in range(x_corners.shape[0])])

    rotation_matrix = np.array([np.array([[np.cos(rotation_),  -np.sin(rotation_), 0],
                                            [np.sin(rotation_), np.cos(rotation_), 0],
                                            [0,  0,  1]]) for rotation_ in rotation])


    corner_box = np.array([np.dot(rotation_matrix[i], bounding_box[i]).T + translation[i] for i in range(x_corners.shape[0])])
    return corner_box

def box_center_to_corner_3d_(box_center):
    translation = box_center[0:3]
    l, w, h = box_center[3], box_center[4], box_center[5]
    rotation = box_center[6]

    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
    z_corners = [-h/2, -h/2, -h/2, -h/2, h/2, h/2, h/2, h/2] #[0, 0, 0, 0, -h, -h, -h, -h]
    bounding_box = np.vstack([x_corners, y_corners, z_corners])

    rotation_matrix = np.array([[np.cos(rotation),  -np.sin(rotation), 0],
                                [np.sin(rotation), np.cos(rotation), 0],
                                [0,  0,  1]])

    corner_box = (np.dot(rotation_matrix, bounding_box).T + translation).T

    return corner_box

def draw_projected_box3d(image, qs, color=(255,255,255), thickness=2):
    qs = qs.astype(np.int32)
    for k in range(0,4):
       i,j=k,(k+1)%4
       cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.LINE_AA)
       i,j=k+4,(k+1)%4 + 4
       cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.LINE_AA)
       i,j=k,k+4
       cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.LINE_AA)
    return image

def interpolate_line(p1, p2, min_val, max_val):
    if p2 == p1:
        return p1
    ratio = (min_val - p1) / (p2 - p1)
    ratio = np.clip(ratio, 0, 1)
    return p1 + ratio * (p2 - p1)

def clip_and_interpolate(points_2d, img_width, img_height):
    new_points = points_2d.copy().T.tolist() 

    for i in range(points_2d.shape[1]):
        p1 = points_2d[:, i]
        for j in range(i + 1, points_2d.shape[1]):
            p2 = points_2d[:, j]
            if (p1[0] < 0 and p2[0] >= 0) or (p1[0] > img_width and p2[0] <= img_width):
                new_x = 0 if p1[0] < 0 else img_width
                new_y = interpolate_line(p1[1], p2[1], p1[0], new_x)
                new_points.append([new_x, new_y])
            if (p1[1] < 0 and p2[1] >= 0) or (p1[1] > img_height and p2[1] <= img_height):
                new_y = 0 if p1[1] < 0 else img_height
                new_x = interpolate_line(p1[0], p2[0], p1[1], new_y)
                new_points.append([new_x, new_y])

    return np.array(new_points).T

colors = sns.color_palette('Paired', 9 * 2)
names = ['car', 'Van', 'Truck', 'pedestrian', 'Person_sitting', 'cyclist', 'Tram', 'Misc', 'DontCare']

if __name__ == '__main__':
  data_path = "./data//sit_full/"
  save_path = "./vis_3dbox_on_img/"
  os.makedirs(save_path, exist_ok=True)
  width = 1920
  height = 1200

  place_list = os.listdir(data_path)
  place_list.sort()
  for place in place_list:
      print(place)
      if place == "ImageSets" or ".pkl" in place or 'json' in place :
        continue
      place_path = data_path + "{}/".format(place)
      scene_list = os.listdir(place_path)
      scene_list.sort()
      for scene in scene_list:
        scene_path = place_path + "{}/".format(scene)
        
        for cam_num in [1,2,3,4,5]:
          img_path = scene_path + "cam_img/{}/data_rgb/".format(cam_num)
          frame_list = os.listdir(img_path)
          frame_list = [int(i.split(".")[0]) for i in frame_list]
          frame_list.sort()
          frame_list = [str(i) for i in frame_list]
          for count, frame in enumerate(frame_list):
            if count >=1:
              continue
            file_id = frame.split(".png")[0]
            img = np.array(io.imread(img_path + f'{file_id}.png'), dtype=np.int32)
            img_ = cv2.imread(img_path + f'{file_id}.png', cv2.IMREAD_COLOR)
            
            calib_path = data_path + "{}/{}/calib/{}.txt".format(place, scene, file_id)
            with open(calib_path, 'r') as f:
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
            mapx, mapy = cv2.initUndistortRectifyMap(K, D, R, P, (width, height), cv2.CV_32FC1)
            img_ = cv2.remap(img_, mapx, mapy, cv2.INTER_LINEAR)

            label_path = data_path + "{}/{}/label_3d/{}.txt".format(place, scene, file_id)
            try:
              with open(label_path, 'r') as f:
                labels = f.readlines()
            except:
              continue

            for line in labels:
              line = line.split()
              lab, cls, h, l, w, x, y, z, rot = line
              h, l, w, x, y, z, rot = map(float, [h, l, w, x, y, z, rot])
              ego_path = data_path + "{}/{}/ego_trajectory/{}.txt".format(place, scene, file_id)
              gt_boxes = np.array((x, y, z, l, w, h, rot)).reshape(1, -1)
              ego_motion = get_ego_matrix(ego_path)
              ego_yaw = rotmat_to_euler(ego_motion[:3, :3])[2]
              gt_boxes[:, 6] += ego_yaw
              comp_obj_center = np.matmul(np.linalg.inv(ego_motion), np.concatenate([gt_boxes[:,:3], np.ones(gt_boxes[:,:3].shape[0]).reshape(1, -1).T], axis=1).T).T
              gt_boxes[:, :3] = comp_obj_center[:, :3]
              gt_boxes[:, 6] *= -1
              corners_3d = box_center_to_corner_3d_(gt_boxes.squeeze())

              if lab != 'DontCare': 
                intrinsic = intrinsic_list[cam_num-1]
                projection_m = np.eye(4)
                projection_m[:3, :] = np.matmul(intrinsic, extrinsic_list[cam_num-1])
                corners_3d_ = np.concatenate([corners_3d.T, np.ones(corners_3d.T[:,:3].shape[0]).reshape(1, -1).T], axis=1).T
                corners_2d = np.matmul(projection_m, corners_3d_).T
                corners_2d[:,0] /= corners_2d[:,2]
                corners_2d[:,1] /= corners_2d[:,2]

                if (corners_2d[:, 2] < 0).sum() >0:
                  continue
                
                if corners_2d.shape[0] == 0 :
                  continue
                
                if lab == 'Pedestrian':
                  color = (0, 158, 255)
                else:
                  color = (0, 0, 255)
                    
                      
                img_ = draw_projected_box3d(img_, corners_2d[:, :2], color=color)
              
            cv2.imwrite(save_path + "{}_{}_{}_".format(place, scene, cam_num) + f'{file_id}.png', img_)

            cv2.waitKey(0)
            cv2.destroyAllWindows()

            print(" {} - {} - {} - {}".format(place, scene, cam_num, frame))
