@PIPELINES.register_module()
class Visualizer(object):
    def __init__(self, save_dir):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def __call__(self, results):
        sensors = results['cam_names']

        for sensor in sensors:
            cam_info = results['curr']['cams'][sensor]
            img_path = cam_info['data_path']
            cam_intrinsic = np.array(cam_info['cam_intrinsic'])
            # Load image
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Show image before drawing boxes
            plt.imshow(img)
            plt.title(f"Original Image {sensor}")
            plt.axis("off")
            plt.show()

            # Transformations: lidar -> lidar_ego -> global -> cam_ego -> image

            # lidar to lidar_ego
            lidar2ego_translation = np.array(results['curr']['lidar2ego_translation'])
            lidar2ego_rotation = Quaternion(results['curr']['lidar2ego_rotation']).rotation_matrix
            lidar2ego_extrinsic = np.eye(4)
            lidar2ego_extrinsic[:3, :3] = lidar2ego_rotation
            lidar2ego_extrinsic[:3, 3] = lidar2ego_translation
            
            # cam to lidar
            cam2lidar_translation = np.array(cam_info['sensor2lidar_translation'])
            cam2lidar_rotation = np.array(cam_info['sensor2lidar_rotation'])
            cam2lidar_extrinsic = np.eye(4)
            cam2lidar_extrinsic[:3, :3] = cam2lidar_rotation
            cam2lidar_extrinsic[:3, 3] = cam2lidar_translation
            lidar2cam_extrinsic = np.linalg.inv(cam2lidar_extrinsic)

            # lidar_ego to global
            l_ego2global_translation = np.array(results['curr']['ego2global_translation'])
            l_ego2global_rotation = Quaternion(results['curr']['ego2global_rotation']).rotation_matrix
            lidarego2global_extrinsic = np.eye(4)
            lidarego2global_extrinsic[:3, :3] = l_ego2global_rotation
            lidarego2global_extrinsic[:3, 3] = l_ego2global_translation

            # cam_ego to global
            c_ego2global_translation = np.array(cam_info['ego2global_translation'])
            c_ego2global_rotation = Quaternion(cam_info['ego2global_rotation']).rotation_matrix
            cam_ego2global_extrinsic = np.eye(4)
            cam_ego2global_extrinsic[:3, :3] = c_ego2global_rotation
            cam_ego2global_extrinsic[:3, 3] = c_ego2global_translation

            # global to cam_ego (inverse)
            global2cam_ego_extrinsic = np.linalg.inv(cam_ego2global_extrinsic)

            # cam_ego to sensor
            sensor2ego_translation = np.array(cam_info['sensor2ego_translation'])
            sensor2ego_rotation = Quaternion(cam_info['sensor2ego_rotation']).rotation_matrix
            sensor2ego_extrinsic = np.eye(4)
            sensor2ego_extrinsic[:3, :3] = sensor2ego_rotation
            sensor2ego_extrinsic[:3, 3] = sensor2ego_translation

            ego2sensor_extrinsic = np.linalg.inv(sensor2ego_extrinsic)

            # Bounding boxes in lidar coordinates
            bbox = results['curr']['gt_boxes']  # 3D bounding boxes
            bboxes = []
            for box in bbox:
                center = box[:3]
                size = box[3:6]
                rotation = box[6]
                corners = self.get_box_corners(center, size, rotation)
                bboxes.append(corners)

            for corners in bboxes:
                # Homogeneous Coordinates
                corners_hom = np.hstack((corners, np.ones((8, 1))))

                # Transformations
                corners_l2e = lidar2ego_extrinsic @ corners_hom.T
                corners_e2g = lidarego2global_extrinsic @ corners_l2e
                corners_g2e = global2cam_ego_extrinsic @ corners_e2g
                corners_sensor = ego2sensor_extrinsic @ corners_g2e

                # Normalize homogeneous coordinates
                corners_sensor = corners_sensor[:3, :] / corners_sensor[3, :]

                # Filter out points with negative z values
                if np.any(corners_sensor[2, :] < 0):
                    continue

                # Convert to image coordinates
                corners_img = cam_intrinsic @ corners_sensor
                corners_img = corners_img[:2, :] / corners_img[2, :]

                # Draw bounding box on the image
                for i in range(4):
                    pt1 = (int(corners_img[0, i]), int(corners_img[1, i]))
                    pt2 = (int(corners_img[0, (i+1)%4]), int(corners_img[1, (i+1)%4]))
                    pt3 = (int(corners_img[0, i+4]), int(corners_img[1, i+4]))
                    pt4 = (int(corners_img[0, (i+1)%4+4]), int(corners_img[1, (i+1)%4+4]))
                    cv2.line(img, pt1, pt2, (255, 0, 0), 2)
                    cv2.line(img, pt1, pt3, (255, 0, 0), 2)
                    cv2.line(img, pt3, pt4, (255, 0, 0), 2)
                    cv2.line(img, pt2, pt4, (255, 0, 0), 2)

            # Save the result
            save_path = os.path.join(self.save_dir, f'{results["sample_idx"]}_img_with_boxes_{sensor}.png')
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, img_bgr)

        return results

    def get_box_corners(self, center, size, rotation):
        l, w, h = size / 2.0
        x_corners = [l, l, -l, -l, l, l, -l, -l]
        y_corners = [w, -w, -w, w, w, -w, -w, w]
        z_corners = [h, h, h, h, -h, -h, -h, -h]
        corners = np.vstack([x_corners, y_corners, z_corners])
        
        # Apply rotation
        R = np.array([[np.cos(rotation), -np.sin(rotation), 0],
                      [np.sin(rotation), np.cos(rotation), 0],
                      [0, 0, 1]])
        corners = np.dot(R, corners)
        
        # Apply translation
        corners = corners + np.array(center).reshape(3, 1)
        
        return corners.T
