from datetime import time
import numpy as np
import cv2
import pickle
import pdb 
from pathlib import Path
from functools import reduce
from typing import List

from tqdm import tqdm
from pyquaternion import Quaternion

from itertools import tee
from copy import deepcopy

try:
    # import pdb; pdb.set_trace()
    from nuscenes import NuScenes
    from nuscenes.utils import splits
    from nuscenes.utils.data_classes import Box, LidarPointCloud
    from nuscenes.eval.detection.config import config_factory
    # from nuscenes.eval.detection.evaluate import NuScenesEval
    from .evaluate import NuScenesEval
    from nuscenes.eval.detection.render import visualize_sample
except:
    print("nuScenes devkit not Found!")

import pdb 

general_to_detection = {
    "human.pedestrian.adult": "pedestrian",
    "human.pedestrian.child": "pedestrian",
    "human.pedestrian.wheelchair": "ignore",
    "human.pedestrian.stroller": "ignore",
    "human.pedestrian.personal_mobility": "ignore",
    "human.pedestrian.police_officer": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian",
    "animal": "ignore",
    "vehicle.car": "car",
    "vehicle.motorcycle": "ignore",
    "vehicle.bicycle": "ignore",
    "vehicle.bus.bendy": "ignore",
    "vehicle.bus.rigid": "ignore",
    "vehicle.truck": "ignore",
    "vehicle.construction": "ignore",
    "vehicle.emergency.ambulance": "ignore",
    "vehicle.emergency.police": "ignore",
    "vehicle.trailer": "ignore",
    "movable_object.barrier": "ignore",
    "movable_object.trafficcone": "ignore",
    "movable_object.pushable_pullable": "ignore",
    "movable_object.debris": "ignore",
    "static_object.bicycle_rack": "ignore",
}

cls_attr_dist = {
    # "barrier": {
    #     "cycle.with_rider": 0,
    #     "cycle.without_rider": 0,
    #     "pedestrian.moving": 0,
    #     "pedestrian.sitting_lying_down": 0,
    #     "pedestrian.standing": 0,
    #     "vehicle.moving": 0,
    #     "vehicle.parked": 0,
    #     "vehicle.stopped": 0,
    # },
    # "bicycle": {
    #     "cycle.with_rider": 2791,
    #     "cycle.without_rider": 8946,
    #     "pedestrian.moving": 0,
    #     "pedestrian.sitting_lying_down": 0,
    #     "pedestrian.standing": 0,
    #     "vehicle.moving": 0,
    #     "vehicle.parked": 0,
    #     "vehicle.stopped": 0,
    # },
    # "bus": {
    #     "cycle.with_rider": 0,
    #     "cycle.without_rider": 0,
    #     "pedestrian.moving": 0,
    #     "pedestrian.sitting_lying_down": 0,
    #     "pedestrian.standing": 0,
    #     "vehicle.moving": 9092,
    #     "vehicle.parked": 3294,
    #     "vehicle.stopped": 3881,
    # },
    "car": {
        "cycle.with_rider": 0,
        "cycle.without_rider": 0,
        "pedestrian.moving": 0,
        "pedestrian.sitting_lying_down": 0,
        "pedestrian.standing": 0,
        "vehicle.moving": 114304,
        "vehicle.parked": 330133,
        "vehicle.stopped": 46898,
    },
    # "construction_vehicle": {
    #     "cycle.with_rider": 0,
    #     "cycle.without_rider": 0,
    #     "pedestrian.moving": 0,
    #     "pedestrian.sitting_lying_down": 0,
    #     "pedestrian.standing": 0,
    #     "vehicle.moving": 882,
    #     "vehicle.parked": 11549,
    #     "vehicle.stopped": 2102,
    # },
    # "ignore": {
    #     "cycle.with_rider": 307,
    #     "cycle.without_rider": 73,
    #     "pedestrian.moving": 0,
    #     "pedestrian.sitting_lying_down": 0,
    #     "pedestrian.standing": 0,
    #     "vehicle.moving": 165,
    #     "vehicle.parked": 400,
    #     "vehicle.stopped": 102,
    # },
    # "motorcycle": {
    #     "cycle.with_rider": 4233,
    #     "cycle.without_rider": 8326,
    #     "pedestrian.moving": 0,
    #     "pedestrian.sitting_lying_down": 0,
    #     "pedestrian.standing": 0,
    #     "vehicle.moving": 0,
    #     "vehicle.parked": 0,
    #     "vehicle.stopped": 0,
    # },
    "pedestrian": {
        "cycle.with_rider": 0,
        "cycle.without_rider": 0,
        "pedestrian.moving": 157444,
        "pedestrian.sitting_lying_down": 13939,
        "pedestrian.standing": 46530,
        "vehicle.moving": 0,
        "vehicle.parked": 0,
        "vehicle.stopped": 0,
    },
    # "traffic_cone": {
    #     "cycle.with_rider": 0,
    #     "cycle.without_rider": 0,
    #     "pedestrian.moving": 0,
    #     "pedestrian.sitting_lying_down": 0,
    #     "pedestrian.standing": 0,
    #     "vehicle.moving": 0,
    #     "vehicle.parked": 0,
    #     "vehicle.stopped": 0,
    # },
    # "trailer": {
    #     "cycle.with_rider": 0,
    #     "cycle.without_rider": 0,
    #     "pedestrian.moving": 0,
    #     "pedestrian.sitting_lying_down": 0,
    #     "pedestrian.standing": 0,
    #     "vehicle.moving": 3421,
    #     "vehicle.parked": 19224,
    #     "vehicle.stopped": 1895,
    # },
    # "truck": {
    #     "cycle.with_rider": 0,
    #     "cycle.without_rider": 0,
    #     "pedestrian.moving": 0,
    #     "pedestrian.sitting_lying_down": 0,
    #     "pedestrian.standing": 0,
    #     "vehicle.moving": 21339,
    #     "vehicle.parked": 55626,
    #     "vehicle.stopped": 11097,
    # },
}

def _second_det_to_nusc_box(detection):
    box3d = detection["box3d_lidar"].detach().cpu().numpy()
    scores = detection["scores"].detach().cpu().numpy()
    labels = detection["label_preds"].detach().cpu().numpy()
    
    # box3d[:, -1] = -box3d[:, -1] - np.pi / 2 # ori

    box_list = []
    for i in range(box3d.shape[0]):
        quat = Quaternion(axis=[0, 0, 1], radians=box3d[i, -1])
        velocity = (*box3d[i, 6:8], 0.0)
        box = Box(
            center = box3d[i, :3],
            size = box3d[i, 3:6],
            orientation=quat,
            label=labels[i],
            score=scores[i],
            velocity=velocity,
        )
        
        box_list.append(box)

    return box_list


def _lidar_nusc_box_to_global(nusc, boxes, sample_token):
    try:
        s_record = nusc.get("sample", sample_token)
        sample_data_token = s_record["data"]["LIDAR_TOP"]
    except:
        sample_data_token = sample_token

    sd_record = nusc.get("sample_data", sample_data_token)
    cs_record = nusc.get("calibrated_sensor", sd_record["calibrated_sensor_token"])
    pose_record = nusc.get("ego_pose", sd_record["ego_pose_token"])

    box_list = []
    for box in boxes:
        # Move box to ego vehicle coord system
        box.rotate(Quaternion(cs_record["rotation"]))
        box.translate(np.array(cs_record["translation"]))
        # Move box to global coord system
        box.rotate(Quaternion(pose_record["rotation"]))
        box.translate(np.array(pose_record["translation"]))

        box_list.append(box)

    return box_list


def _get_available_scenes(nusc):
    available_scenes = []
    print("total scene num:", len(nusc.scene))
    for scene in nusc.scene:
        scene_token = scene["token"]
        scene_rec = nusc.get("scene", scene_token)
        sample_rec = nusc.get("sample", scene_rec["first_sample_token"])
        sd_rec = nusc.get("sample_data", sample_rec["data"]["LIDAR_TOP"])
        has_more_frames = True
        scene_not_exist = False
        while has_more_frames:
            lidar_path, boxes, _ = nusc.get_sample_data(sd_rec["token"])
            if not Path(lidar_path).exists():
                scene_not_exist = True
                break
            else:
                break
        if scene_not_exist:
            continue
        available_scenes.append(scene)
    print("exist scene num:", len(available_scenes))
    return available_scenes


def get_sample_data(
    nusc, sample_data_token: str, selected_anntokens: List[str] = None
):
    """
    Returns the data path as well as all annotations related to that sample_data.
    Note that the boxes are transformed into the current sensor's coordinate frame.
    :param sample_data_token: Sample_data token.
    :param selected_anntokens: If provided only return the selected annotation.
    :return: (data_path, boxes, camera_intrinsic <np.array: 3, 3>)
    """

    # Retrieve sensor & pose records
    sd_record = nusc.get("sample_data", sample_data_token)
    cs_record = nusc.get("calibrated_sensor", sd_record["calibrated_sensor_token"])
    sensor_record = nusc.get("sensor", cs_record["sensor_token"])
    pose_record = nusc.get("ego_pose", sd_record["ego_pose_token"])

    data_path = nusc.get_sample_data_path(sample_data_token)

    if sensor_record["modality"] == "camera":
        cam_intrinsic = np.array(cs_record["camera_intrinsic"])
    else:
        cam_intrinsic = None

    # Retrieve all sample annotations and map to sensor coordinate system.
    if selected_anntokens is not None:
        boxes = list(map(nusc.get_box, selected_anntokens))
    else:
        boxes = nusc.get_boxes(sample_data_token)

    # Make list of Box objects including coord system transforms.
    box_list = []
    for box in boxes:
        box.velocity = nusc.box_velocity(box.token)
        # Move box to ego vehicle coord system
        box.translate(-np.array(pose_record["translation"]))
        box.rotate(Quaternion(pose_record["rotation"]).inverse)

        #  Move box to sensor coord system
        box.translate(-np.array(cs_record["translation"]))
        box.rotate(Quaternion(cs_record["rotation"]).inverse)

        box_list.append(box)

    return data_path, box_list, cam_intrinsic

def window(iterable, size):
    iters = tee(iterable, size)
    for i in range(1, size):
        for each in iters[i:]:
            next(each, None)

    return zip(*iters)

def get_time(nusc, src_token, dst_token):
    time_last = 1e-6 * nusc.get('sample', src_token)["timestamp"]
    time_first = 1e-6 * nusc.get('sample', dst_token)["timestamp"]
    time_diff = time_first - time_last

    return time_diff 

def center_distance(gt_box, pred_box) -> float:
    """
    L2 distance between the box centers (xy only).
    :param gt_box: GT annotation sample.
    :param pred_box: Predicted sample.
    :return: L2 distance.
    """
    return np.linalg.norm(np.array(pred_box.center[:2]) - np.array(gt_box.center[:2]))

def trajectory(nusc, boxes, time, timesteps=7, past=False):
    target = boxes[-1]
    
    static_forecast = deepcopy(boxes[0])

    linear_forecast = deepcopy(boxes[0])
    vel = linear_forecast.velocity[:2]
    disp = np.sum(list(map(lambda x: np.array(list(vel) + [0]) * x, time)), axis=0)

    if past:
        linear_forecast.center = linear_forecast.center - disp

    else:
        linear_forecast.center = linear_forecast.center + disp
    
    if center_distance(target, static_forecast) < max(target.wlh[0], target.wlh[1]):
        return "static"

    elif center_distance(target, linear_forecast) < max(target.wlh[0], target.wlh[1]):
        return "linear"

    else:
        return "nonlinear"

def get_annotations(nusc, annotations, ref_boxes, timesteps, past):
    forecast_annotations = []
    forecast_boxes = []   
    forecast_trajectory = []
    sample_tokens = [s["token"] for s in nusc.sample]

    for annotation, ref_box in zip(annotations, ref_boxes):
        tracklet_box = []
        tracklet_annotation = []
        tracklet_trajectory = []

        token = nusc.sample[sample_tokens.index(annotation["sample_token"])]["data"]["LIDAR_TOP"]
        sd_record = nusc.get("sample_data", token)
        cs_record = nusc.get("calibrated_sensor", sd_record["calibrated_sensor_token"])
        pose_record = nusc.get("ego_pose", sd_record["ego_pose_token"])

        pannotation = annotation
        for i in range(timesteps):
            box = Box(center = annotation["translation"],
                      size = ref_box.wlh,
                      orientation = Quaternion(annotation["rotation"]),
                      velocity = nusc.box_velocity(annotation["token"]),
                      name = annotation["category_name"],
                      token = annotation["token"])

            box.translate(-np.array(pose_record["translation"]))
            box.rotate(Quaternion(pose_record["rotation"]).inverse)

            #  Move box to sensor coord system
            box.translate(-np.array(cs_record["translation"]))
            box.rotate(Quaternion(cs_record["rotation"]).inverse)

            tracklet_box.append(box)
            tracklet_annotation.append(annotation)
            
            next_token = annotation["next"]
            prev_token = pannotation["prev"]

            if past:
                if next_token != "":
                    pannotation = nusc.get("sample_annotation", next_token)
                
                if prev_token != "":
                    annotation = nusc.get("sample_annotation", prev_token)
            else:
                if next_token != "":
                    annotation = nusc.get("sample_annotation", next_token)
                
                if prev_token != "":
                    pannotation = nusc.get("sample_annotation", prev_token)
        
        tokens = [b["sample_token"] for b in tracklet_annotation]
        time = [get_time(nusc, src, dst) for src, dst in window(tokens, 2)]
        tracklet_trajectory = trajectory(nusc, tracklet_box, time, timesteps)

        forecast_boxes.append(tracklet_box)
        forecast_annotations.append(tracklet_annotation)
        forecast_trajectory.append(timesteps*[tracklet_trajectory])

    return forecast_boxes, forecast_annotations, forecast_trajectory

def _fill_trainval_infos(nusc, train_scenes, val_scenes, test=False, nsweeps=20, filter_zero=True, timesteps=7, past=False):
    from nuscenes.utils.geometry_utils import transform_matrix
    
    train_nusc_infos = []
    val_nusc_infos = []

    ref_chan = "LIDAR_TOP"  # The radar channel from which we track back n sweeps to aggregate the point cloud.
    chan = "LIDAR_TOP"  # The reference channel of the current sample_rec that the point clouds are mapped to.

    for sample in tqdm(nusc.sample):
        """ Manual save info["sweeps"] """
        # Get reference pose and timestamp
        # ref_chan == "LIDAR_TOP"
        ref_sd_token = sample["data"][ref_chan]
        ref_sd_rec = nusc.get("sample_data", ref_sd_token)
        ref_cs_rec = nusc.get(
            "calibrated_sensor", ref_sd_rec["calibrated_sensor_token"]
        )
        ref_pose_rec = nusc.get("ego_pose", ref_sd_rec["ego_pose_token"])
        ref_time = 1e-6 * ref_sd_rec["timestamp"]

        ref_lidar_path, ref_boxes, _ = get_sample_data(nusc, ref_sd_token)

        ref_cam_front_token = sample["data"]["CAM_FRONT"]
        ref_cam_path, _, ref_cam_intrinsic = nusc.get_sample_data(ref_cam_front_token)

        # Homogeneous transform from ego car frame to reference frame
        ref_from_car = transform_matrix(
            ref_cs_rec["translation"], Quaternion(ref_cs_rec["rotation"]), inverse=True
        )

        # Homogeneous transformation matrix from global to _current_ ego car frame
        car_from_global = transform_matrix(
            ref_pose_rec["translation"],
            Quaternion(ref_pose_rec["rotation"]),
            inverse=True,
        )

        info = {
            "lidar_path": ref_lidar_path,
            "cam_front_path": ref_cam_path,
            "cam_intrinsic": ref_cam_intrinsic,
            "token": sample["token"],
            "sweeps": [],
            "ref_from_car": ref_from_car,
            "car_from_global": car_from_global,
            "timestamp": ref_time,
        }


        sample_data_token = sample["data"][chan]
        curr_sd_rec = nusc.get("sample_data", sample_data_token)

        sweeps = []
        while len(sweeps) < nsweeps - 1:
            if curr_sd_rec["prev"] == "":
                if len(sweeps) == 0:
                    sweep = {
                        "lidar_path": ref_lidar_path,
                        "sample_data_token": curr_sd_rec["token"],
                        "transform_matrix": None,
                        "time_lag": curr_sd_rec["timestamp"] * 0,
                        # time_lag: 0,
                    }
                    sweeps.append(sweep)
                else:
                    sweeps.append(sweeps[-1])
            else:
                curr_sd_rec = nusc.get("sample_data", curr_sd_rec["prev"])

                # Get past pose
                current_pose_rec = nusc.get("ego_pose", curr_sd_rec["ego_pose_token"])
                global_from_car = transform_matrix(
                    current_pose_rec["translation"],
                    Quaternion(current_pose_rec["rotation"]),
                    inverse=False,
                )

                # Homogeneous transformation matrix from sensor coordinate frame to ego car frame.
                current_cs_rec = nusc.get(
                    "calibrated_sensor", curr_sd_rec["calibrated_sensor_token"]
                )
                car_from_current = transform_matrix(
                    current_cs_rec["translation"],
                    Quaternion(current_cs_rec["rotation"]),
                    inverse=False,
                )

                tm = reduce(
                    np.dot,
                    [ref_from_car, car_from_global, global_from_car, car_from_current],
                )

                lidar_path = nusc.get_sample_data_path(curr_sd_rec["token"])

                time_lag = ref_time - 1e-6 * curr_sd_rec["timestamp"]

                sweep = {
                    "lidar_path": lidar_path,
                    "sample_data_token": curr_sd_rec["token"],
                    "transform_matrix": tm,
                    "global_from_car": global_from_car,
                    "car_from_current": car_from_current,
                    "time_lag": time_lag,
                }
                sweeps.append(sweep)

        info["sweeps"] = sweeps

        assert (len(info["sweeps"]) == nsweeps - 1), f"sweep {curr_sd_rec['token']} only has {len(info['sweeps'])} sweeps, you should duplicate to sweep num {nsweeps-1}"
        """ read from api """
        if not test:
            ego_map = nusc.explorer.get_ego_centric_map(sweeps[0]["sample_data_token"])
            bev = cv2.resize(ego_map, dsize=(180, 180), interpolation=cv2.INTER_CUBIC)

            annotations = [nusc.get("sample_annotation", token) for token in sample["anns"]]
            forecast_boxes, forecast_annotations, forecast_trajectory = get_annotations(nusc, annotations, ref_boxes, timesteps, past)

            mask = np.array([(anno['num_lidar_pts'] + anno['num_radar_pts']) > 0 for anno in annotations], dtype=bool).reshape(-1)
            locs = [np.array([b.center for b in boxes]).reshape(-1, 3) for boxes in forecast_boxes]
            rlocs = [np.array([b.center for b in boxes]).reshape(-1, 3) for boxes in forecast_boxes]

            dims = [np.array([b.wlh for b in boxes]).reshape(-1, 3) for boxes in forecast_boxes]
            # rots = np.array([b.orientation.yaw_pitch_roll[0] for b in ref_boxes]).reshape(-1, 1)
            velocity = [np.array([b.velocity for b in boxes]).reshape(-1, 3) for boxes in forecast_boxes]
            rvelocity = [np.array([b.velocity for b in boxes]).reshape(-1, 3) for boxes in forecast_boxes]
            rots = [np.array([quaternion_yaw(b.orientation) for b in boxes]).reshape(-1, 1) for boxes in forecast_boxes]
            rrots = [np.array([quaternion_yaw(b.orientation) for b in boxes]).reshape(-1, 1) for boxes in forecast_boxes]

            names = [np.array([b.name for b in boxes]) for boxes in forecast_boxes]
            tokens = [np.array([b.token for b in boxes]) for boxes in forecast_boxes]
            rtokens = [np.array([b.token for b in boxes]) for boxes in forecast_boxes]

            trajectory = [np.array([b for b in boxes]) for boxes in forecast_trajectory]

            gt_boxes = [np.concatenate([locs[i], dims[i], velocity[i][:, :2], rvelocity[i][:, :2], -rots[i] - np.pi / 2, -rrots[i] - np.pi / 2], axis=1) for i in range(len(annotations))]
            # gt_boxes = np.concatenate([locs, dims, rots], axis=1)
            assert len(annotations) == len(gt_boxes) == len(velocity) == len(rvelocity)

            if len(annotations) > 0:
                if not filter_zero:
                    info["gt_boxes"] = np.array(gt_boxes)
                    info["gt_boxes_velocity"] = np.array(velocity)
                    info["gt_boxes_rvelocity"] = np.array(rvelocity)
                    info["gt_names"] = np.array([[general_to_detection[n] for n in name] for name in names])
                    info["gt_boxes_token"] = np.array(tokens)
                    info["gt_boxes_rtoken"] = np.array(rtokens)
                    info["gt_trajectory"] = np.array(trajectory)
                    info["bev"] = bev

                else:
                    info["gt_boxes"] = np.array(gt_boxes)[mask, :]
                    info["gt_boxes_velocity"] = np.array(velocity)[mask, :]
                    info["gt_boxes_rvelocity"] = np.array(rvelocity)[mask, :]
                    info["gt_names"] = np.array([[general_to_detection[n] for n in name] for name in names])[mask]
                    info["gt_boxes_token"] = np.array(tokens)[mask]
                    info["gt_boxes_rtoken"] = np.array(rtokens)[mask]
                    info["gt_trajectory"] = np.array(trajectory)[mask]
                    info["bev"] = bev
            else:
                mask = np.array([(anno['num_lidar_pts'] + anno['num_radar_pts']) >0 for anno in annotations], dtype=bool).reshape(-1)

                locs = np.array([b.center for b in ref_boxes]).reshape(-1, 3)
                dims = np.array([b.wlh for b in ref_boxes]).reshape(-1, 3)
                # rots = np.array([b.orientation.yaw_pitch_roll[0] for b in ref_boxes]).reshape(-1, 1)
                velocity = np.array([b.velocity for b in ref_boxes]).reshape(-1, 3)
                rvelocity = np.array([b.rvelocity for b in ref_boxes]).reshape(-1, 3)
                rots = np.array([quaternion_yaw(b.orientation) for b in ref_boxes]).reshape(-1, 1)
                names = np.array([b.name for b in ref_boxes])
                tokens = np.array([b.token for b in ref_boxes])
                gt_boxes = np.concatenate([locs, dims, velocity[:, :2], rvelocity[:, :2], -rots - np.pi / 2], axis=1)
                trajectory = np.array(["static" for b in ref_boxes])

                info["gt_boxes"] = gt_boxes
                info["gt_boxes_velocity"] = velocity
                info["gt_boxes_rvelocity"] = rvelocity
                info["gt_names"] = np.array([general_to_detection[name] for name in names])
                info["gt_boxes_token"] = tokens
                info["gt_boxes_rtoken"] = tokens
                info["gt_boxes_rtoken"] = tokens
                info["gt_trajectory"] = trajectory
                info['bev'] = bev

        if sample["scene_token"] in train_scenes:
            train_nusc_infos.append(info)
        else:
            val_nusc_infos.append(info)

    return train_nusc_infos, val_nusc_infos


def quaternion_yaw(q: Quaternion) -> float:
    """
    Calculate the yaw angle from a quaternion.
    Note that this only works for a quaternion that represents a box in lidar or global coordinate frame.
    It does not work for a box in the camera frame.
    :param q: Quaternion of interest.
    :return: Yaw angle in radians.
    """

    # Project into xy plane.
    v = np.dot(q.rotation_matrix, np.array([1, 0, 0]))

    # Measure yaw using arctan.
    yaw = np.arctan2(v[1], v[0])

    return yaw


def create_nuscenes_infos(root_path, version="v1.0-trainval", experiment="trainval_forecast", nsweeps=20, filter_zero=True, timesteps=7, past=False):
    nusc = NuScenes(version=version, dataroot=root_path, verbose=True)
    available_vers = ["v1.0-trainval", "v1.0-test", "v1.0-mini"]
    assert version in available_vers
    if version == "v1.0-trainval":
        train_scenes = splits.train
        # random.shuffle(train_scenes)
        # train_scenes = train_scenes[:int(len(train_scenes)*0.2)]
        val_scenes = splits.val
    elif version == "v1.0-test":
        train_scenes = splits.test
        val_scenes = []
    elif version == "v1.0-mini":
        train_scenes = splits.mini_train
        val_scenes = splits.mini_val
    else:
        raise ValueError("unknown")
    test = "test" in version
    root_path = Path(root_path)
    # filter exist scenes. you may only download part of dataset.
    available_scenes = _get_available_scenes(nusc)
    available_scene_names = [s["name"] for s in available_scenes]
    train_scenes = list(filter(lambda x: x in available_scene_names, train_scenes))
    # train_scenes = train_scenes[:int(len(train_scenes)/7)]
    val_scenes = list(filter(lambda x: x in available_scene_names, val_scenes))
    # val_scenes = val_scenes[:int(len(val_scenes)/7)]
    train_scenes = set(
        [
            available_scenes[available_scene_names.index(s)]["token"]
            for s in train_scenes
        ]
    )
    val_scenes = set(
        [available_scenes[available_scene_names.index(s)]["token"] for s in val_scenes]
    )
    if test:
        print(f"test scene: {len(train_scenes)}")
    else:
        print(f"train scene: {len(train_scenes)}, val scene: {len(val_scenes)}")

    train_nusc_infos, val_nusc_infos = _fill_trainval_infos(
        nusc, train_scenes, val_scenes, test, nsweeps=nsweeps, filter_zero=filter_zero, timesteps=timesteps
    )

    if test:
        print(f"test sample: {len(train_nusc_infos)}")
        with open(
            root_path / "{}/infos_test_{:02d}sweeps_withvelo.pkl".format(experiment, nsweeps), "wb"
        ) as f:
            pickle.dump(train_nusc_infos, f)
    else:
        print(
            f"train sample: {len(train_nusc_infos)}, val sample: {len(val_nusc_infos)}"
        )
        with open(
            root_path / "{}/infos_train_{:02d}sweeps_withvelo_filter_{}.pkl".format(experiment, nsweeps, filter_zero), "wb"
        ) as f:
            pickle.dump(train_nusc_infos, f)
        with open(
            root_path / "{}/infos_val_{:02d}sweeps_withvelo_filter_{}.pkl".format(experiment, nsweeps, filter_zero), "wb"
        ) as f:
            pickle.dump(val_nusc_infos, f)


def eval_main(nusc, eval_version, res_path, eval_set, output_dir, forecast, tp_pct, static_only,
              cohort_analysis, topK, root, association_oracle, nogroup):
    # nusc = NuScenes(version=version, dataroot=str(root_path), verbose=True)
    cfg = config_factory(eval_version)

    nusc_eval = NuScenesEval(
        nusc,
        config=cfg,
        result_path=res_path,
        eval_set=eval_set,
        output_dir=output_dir,
        verbose=True,
        forecast=forecast,
        tp_pct=tp_pct,
        static_only=static_only,
        cohort_analysis=cohort_analysis,
        topK=topK,
        root=root,
        association_oracle=association_oracle,
        nogroup=nogroup
    )
    metrics_summary = nusc_eval.main(plot_examples=10,cohort_analysis=cohort_analysis)
