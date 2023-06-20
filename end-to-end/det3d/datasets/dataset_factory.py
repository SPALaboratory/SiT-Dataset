from .nuscenes import NuScenesDataset
from .waymo import WaymoDataset
from .spa_nusc import SPA_Nus_Dataset

dataset_factory = {
    "NUSC": NuScenesDataset,
    "WAYMO": WaymoDataset,
    "SPA_Nus" : SPA_Nus_Dataset,
}


def get_dataset(dataset_name):
    return dataset_factory[dataset_name]
