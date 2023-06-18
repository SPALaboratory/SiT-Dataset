

## Introduction
For 3d object detection on SiT Dataset, we use the MMDetection3D toolbox.

The performance of baselines are as follows.


|**Methods**|**Modality**|**mAP &uarr;**| **AP(0.25) &uarr;** |**AP(0.5) &uarr;** | **AP(1.0) &uarr;** | **AP(2.0) &uarr;** | **Pretrained** |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|**PointPillars**|LiDAR| 0.319 | 0.202 | 0.316 | 0.346 | 0.414 |<a href="">TBD</a>|
|**CenterPoint-P**|LiDAR| 0.382 | 0.233 | 0.388 | 0.424 | 0.482 |<a href="">TBD</a>|
|**CenterPoint-V**|LiDAR| 0.514 | 0.352 | **0.522**| 0.556 | 0.620 |<a href="">TBD</a>|



## Data generation

#### Concat 
```
python tools/create_data.py spa_nus --root-path ./data/spa --version v1.0-spa-trainval --max-sweeps 1 --out-dir ./data/spa --extra-tag spa_nusc
```

#### Top
```
python tools/create_data.py spa_nus_top --root-path ./data/spa --version v1.0-spa-trainval --max-sweeps 1 --out-dir ./data/spa --extra-tag spa_nusc_top
```

#### Bottom
```
python tools/create_data.py spa_nus_bottom --root-path ./data/spa --version v1.0-spa-trainval --max-sweeps 1 --out-dir ./data/spa --extra-tag spa_nusc_bottom
```

## Training
#### PointPillars training
```
python tools/train.py configs/_base_/spa_nusc_pointpillar.py
python tools/train.py configs/_base_/spa_nusc_top_pointpillar.py
python tools/train.py configs/_base_/spa_nusc_bottom_pointpillar.py
```


#### Ceonterpoint(P) training
```
python tools/train.py configs/_base_/spa_nusc_centerpoint_ped.py
python tools/train.py configs/_base_/spa_nusc_top_centerpoint_ped.py
python tools/train.py configs/_base_/spa_nusc_bottom_centerpoint_ped.py
```


#### Centerpoint(V) training
```
python tools/train.py configs/_base_/spa_nusc_centerpoint_voxel.py
python tools/train.py configs/_base_/spa_nusc_top_centerpoint_voxel.py
python tools/train.py configs/_base_/spa_nusc_bottom_centerpoint_voxel.py
```





## License

This project is released under the [Apache 2.0 license](LICENSE).



## Citation

If you find this project useful in your research, please consider cite:

```latex
@misc{mmdet3d2020,
    title={{MMDetection3D: OpenMMLab} next-generation platform for general {3D} object detection},
    author={MMDetection3D Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmdetection3d}},
    year={2020}
}
```


Our code is implemented based on mmdetection3d. We express our sincere gratitude for the advancements made by mmdetection3d.

[![docs](https://img.shields.io/badge/docs-latest-blue)](https://mmdetection3d.readthedocs.io/en/latest/)
[![badge](https://github.com/open-mmlab/mmdetection3d/workflows/build/badge.svg)](https://github.com/open-mmlab/mmdetection3d/actions)
[![codecov](https://codecov.io/gh/open-mmlab/mmdetection3d/branch/master/graph/badge.svg)](https://codecov.io/gh/open-mmlab/mmdetection3d)
[![license](https://img.shields.io/github/license/open-mmlab/mmdetection3d.svg)](https://github.com/open-mmlab/mmdetection3d/blob/master/LICENSE)

