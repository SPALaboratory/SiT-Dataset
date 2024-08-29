from .nuscenes import NuScenesDataset
from .waymo import WaymoDataset
from .sit import SiT_Dataset

dataset_factory = {
    "NUSC": NuScenesDataset,
    "WAYMO": WaymoDataset,
    "SiT" : SiT_Dataset,
}


def get_dataset(dataset_name):
    return dataset_factory[dataset_name]
