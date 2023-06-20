import copy
from pathlib import Path
import pickle

import fire, os
import sys 

sys.path.append('~/Workspace/FutureDet')
sys.path.append('~/Workspace/Core/nuscenes-forecast/python-sdk')

from det3d.datasets.nuscenes import nusc_common as nu_ds
from det3d.datasets.spa_nusc import spa_nusc_common as spa_nusc_ds
from det3d.datasets.utils.create_gt_database import create_groundtruth_database
from det3d.datasets.waymo import waymo_common as waymo_ds

def nuscenes_data_prep(root_path, version, experiment="trainval_forecast", nsweeps=20, filter_zero=True, timesteps=7):
    past = True if "past" in experiment else False

    if not os.path.isdir(root_path + "/" + experiment):
        os.makedirs(root_path + "/" + experiment)

    nu_ds.create_nuscenes_infos(root_path, version=version, experiment=experiment, nsweeps=nsweeps, filter_zero=filter_zero, timesteps=timesteps, past=past)
    create_groundtruth_database(
        "NUSC",
        root_path + "/{}".format(experiment),
        Path(root_path) / "{}/infos_train_{:02d}sweeps_withvelo_filter_{}.pkl".format(experiment, nsweeps, filter_zero),
        nsweeps=nsweeps,
        timesteps=timesteps
    )

# def spa_nusc_data_prep(root_path, version, experiment="trainval_forecast", nsweeps=20, filter_zero=True, timesteps=7):
def spa_nusc_data_prep(root_path, version, experiment="trainval_forecast", nsweeps=1, filter_zero=True, timesteps=7):
    past = True if "past" in experiment else False

    if not os.path.isdir(root_path + "/" + experiment):
        os.makedirs(root_path + "/" + experiment)

    spa_nusc_ds.create_spa_nusc_infos(root_path, version=version, experiment=experiment, nsweeps=nsweeps, filter_zero=filter_zero, timesteps=timesteps, past=past)
    create_groundtruth_database(
        "SPA_Nus",
        root_path + "/{}".format(experiment),
        Path(root_path) / "{}/infos_train_{:02d}sweeps_withvelo_filter_{}.pkl".format(experiment, nsweeps, filter_zero),
        nsweeps=nsweeps,
        timesteps=timesteps
    )

def waymo_data_prep(root_path, split, nsweeps=1):
    waymo_ds.create_waymo_infos(root_path, split=split, nsweeps=nsweeps)
    if split == 'train': 
        create_groundtruth_database(
            "WAYMO",
            root_path,
            Path(root_path) / "infos_train_{:02d}sweeps_filter_zero_gt.pkl".format(nsweeps),
            used_classes=['VEHICLE', 'CYCLIST', 'PEDESTRIAN'],
            nsweeps=nsweeps
        )
    

if __name__ == "__main__":
    fire.Fire()

#python tools/create_data.py nuscenes_data_prep --root_path ../Data/nuScenes --version v1.0-trainval --timesteps 6
