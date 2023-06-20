## Installation 

Modified from [det3d](https://github.com/poodarchu/Det3D/tree/56402d4761a5b73acd23080f537599b0888cce07)'s original document.

### Requirements

- OS: Ubuntu 18.04
- PyTorch: 1.7.1
- spconv: 1.0
- [APEX](https://github.com/neeharperi/apex)
- [Sparse Convolutions (spconv)](https://github.com/neeharperi/spconv)

#### Notes
- Installing spconv is the most challenging part of the setup process. We would recommend checking out the issues and documentation from the [original implementation](https://github.com/traveller59/spconv) for common modifications to spconv and PyTorch. 

- As part of this code release we have installed this software and run the training and evaluation scripts on a new AWS instance to verify the installation process described below. 

### Basic Installation 


#### spconv
```bash
git clone git@github.com:neeharperi/spconv.git
```

#### APEX

```bash
git clone git@github.com:neeharperi/apex.git
```

#### nuScenes end-to-end forecasting dev-kit

```bash
git clone git@github.com:neeharperi/nuscenes-forecast.git
```

#### Compiling RotatedNMS, APEX, and spconv

```bash
# Modify path to APEX, spconv, CUDA and CUDNN in FutureDet/setup.sh
bash setup.sh
```

## Use FutureDet
Be sure to change the paths in configs and syspath in the following files:
- train.py
- evaluate.py
- trajectory.py
- visualize.py
- det3d/datasets/nuscenes/nuscenes.py
- tools/create_data.py
- tools/dist_test.py

### Benchmark Evaluation and Training

#python tools/create_data.py spa_nusc_data_prep --root_path SPA_DATASET_ROOT --version v1.0-spa-trainval --timesteps 7

```

In the end, the data and info files should be organized as follows

```
# For nuScenes Dataset 
└── SPA-NUSC_DATASET_ROOT
      ├── samples       <-- key frames
      ├── sweeps        <-- frames without annotation
      ├── maps          <-- unused
      |── v1.0-spa-trainval <-- metadata and annotations
      |__ trainval_forecast
          |── infos_train_1sweeps_withvelo_filter_True.pkl <-- train annotations
          |── infos_val_1sweeps_withvelo_filter_True.pkl <-- val annotations
          |── dbinfos_train_1sweeps_withvelo.pkl <-- GT database info files
          |── gt_database_1sweeps_withvelo <-- GT database 
```


Use the following command to start a distributed training and evaluation. The models and logs will be saved to ```models/CONFIG_NAME```. Results will be save to ```results/CONFIG_NAME``` 

#### FaF*
```bash

# Pedestrians
python ./tools/train.py configs/centerpoint/spa_nus_centerpoint_pedestrian_forecast_n3_detection.py

```

#### FutureDet
```bash

# Pedestrians
python ./tools/train.py configs/centerpoint/spa_nus_centerpoint_pedestrian_forecast_n3dtf_detection.py


```
#### Evaluation Parameters
```
extractBox -> Uses modelCheckPoint to run inference on GPUs and save results to disk
tp_pct -> TP percentage thresholds for ADE@TP % and FDE@TP %. Setting tp_pct to -1 returns AVG ADE/FDE over all TP threholds.
static_only -> Rescores stationary objects to have higher confidence. Result from Table 1.
eval_only -> Uses cached results to run evaluation
forecast_mode -> Detection association method. [Constant Velocity -> velocity_constant, FaF* -> velocity_forward, FutureDet -> velocity_dense]
classname -> Select class to evaluate. car and pedestrian currently supported.
rerank -> Assignment of forecasting score. [last, first, average]
cohort_analysis -> Reports evaluation metrics per motion subclass static/linear/nonlinear.
K -> topK evaluation, only useful for FutureDet
```

## Acknowlegement
This project is not possible without multiple great opensourced codebases. We list some notable examples below.  

* [det3d](https://github.com/poodarchu/det3d)
* [second.pytorch](https://github.com/traveller59/second.pytorch)
* [CenterTrack](https://github.com/xingyizhou/CenterTrack)
* [CenterNet](https://github.com/xingyizhou/CenterNet) 
* [mmcv](https://github.com/open-mmlab/mmcv)
* [mmdetection](https://github.com/open-mmlab/mmdetection)
* [OpenPCDet](https://github.com/open-mmlab/OpenPCDet)
* [CenterPoint](https://github.com/tianweiy/CenterPoint)

If you find this codebase useful, please consider citing:

    @article{peri2022futuredet,
      title={Forecasting from LiDAR via Future Object Detection},
      author={Peri, Neehar and Luiten, Jonathon and Li, Mengtian and Osep, Aljosa and Leal-Taixe, Laura and Ramanan, Deva},
      journal={arXiv:2203.16297},
      year={2022},
    }

