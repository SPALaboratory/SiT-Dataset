<div align="center">
  <img src="resources/mmdet3d-logo.png" width="600"/>
  <div>&nbsp;</div>
  <div align="center">
    <b><font size="5">OpenMMLab website</font></b>
    <sup>
      <a href="https://openmmlab.com">
        <i><font size="4">HOT</font></i>
      </a>
    </sup>
    &nbsp;&nbsp;&nbsp;&nbsp;
    <b><font size="5">OpenMMLab platform</font></b>
    <sup>
      <a href="https://platform.openmmlab.com">
        <i><font size="4">TRY IT OUT</font></i>
      </a>
    </sup>
  </div>
  <div>&nbsp;</div>
</div>

[![docs](https://img.shields.io/badge/docs-latest-blue)](https://mmdetection3d.readthedocs.io/en/latest/)
[![badge](https://github.com/open-mmlab/mmdetection3d/workflows/build/badge.svg)](https://github.com/open-mmlab/mmdetection3d/actions)
[![codecov](https://codecov.io/gh/open-mmlab/mmdetection3d/branch/master/graph/badge.svg)](https://codecov.io/gh/open-mmlab/mmdetection3d)
[![license](https://img.shields.io/github/license/open-mmlab/mmdetection3d.svg)](https://github.com/open-mmlab/mmdetection3d/blob/master/LICENSE)


## Introduction
For 3d object detection on SiT Dataset, we use the MMDetection3D toolbox.

The performance of baselines are as follows.

|      Models     |  mAP  |
|:---------------:|:-----:|
|  PointPillars   | 0.319 |
| CenterPoint - P | 0.382 |
| CenterPoint - V | 0.382 |

## Data generation

#### concat 
python tools/create_data.py spa_nus --root-path ./data/spa --version v1.0-spa-trainval --max-sweeps 1 --out-dir ./data/spa --extra-tag spa_nusc

#### top
python tools/create_data.py spa_nus_top --root-path ./data/spa --version v1.0-spa-trainval --max-sweeps 1 --out-dir ./data/spa --extra-tag spa_nusc_top

#### bottom
python tools/create_data.py spa_nus_bottom --root-path ./data/spa --version v1.0-spa-trainval --max-sweeps 1 --out-dir ./data/spa --extra-tag spa_nusc_bottom

## Training
#### PointPillars training
python tools/train.py configs/_base_/spa_nusc_pointpillar.py
python tools/train.py configs/_base_/spa_nusc_top_pointpillar.py
python tools/train.py configs/_base_/spa_nusc_bottom_pointpillar.py

#### Ceonterpoint training
python tools/train.py configs/_base_/spa_nusc_centerpoint_ped.py
python tools/train.py configs/_base_/spa_nusc_top_centerpoint_ped.py
python tools/train.py configs/_base_/spa_nusc_bottom_centerpoint_ped.py

#### Ceonterpoint training
python tools/train.py configs/_base_/spa_nusc_centerpoint_voxel.py
python tools/train.py configs/_base_/spa_nusc_top_centerpoint_voxel.py
python tools/train.py configs/_base_/spa_nusc_bottom_centerpoint_voxel.py

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

# spa_3d_detection
