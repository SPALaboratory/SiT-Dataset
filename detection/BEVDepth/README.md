

## BEVDepth

BEVDepth is a 3D object detection model that uses Bird's Eye View (BEV) representation to integrate depth information from multiple camera views. It enhances the accuracy of detecting objects by combining spatial features across different perspectives. BEVDepth is widely used as a foundational model for BEV-based research, particularly in autonomous driving applications. We provide BEVDepth as the baseline for SiT-Dataset research on multi-view cameras.

The performance of the baseline models on the validation set is as follows.

|**Methods**|**Modality**|**mAP &uarr;**| **AP(0.25) &uarr;** |**AP(0.5) &uarr;** | **AP(1.0) &uarr;** | **AP(2.0) &uarr;** | **Pretrained** |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|**BEVDepth**|Camera| 0.270 | 0.0189 | 0.1832 | 0.3610 | 0.5160 |<a href="https://drive.google.com/file/d/1RttgPBNatAI2Nm6po2IzvLQPS2rU3Y3-/view?usp=sharing">Gdrive</a>|


## Data Tree
```
BEVDet_git
├─ configs
│  ├─ bevdepth
│  │   └─ bevdepth.py
├─ data  
│  └─ sit
├─ mmdet3d
│  └─ datasets
│        ├─ pipelines 
│        └─ sit_bev_dataset.py 
├─ tools
│  ├─ data_converter
│  │     └─ sit_converter.py  
│  │     └─ sit_data_utils.py  
│  ├─ create_data_bevdepth.py
│  ├─ train.py
│  └─ test.py
└─ README.md
```

## Environment
```
sudo docker run -it --gpus all --ipc=host -v /mnt:/mnt --name sit_bevdepth jeongseon824/sit_bevdepth:latest /bin/bash
```

## Data generation
```
python tools/create_data_bevdepth.py --version sit-trainval
python tools/create_data_bevdepth.py --version sit-test
```


## Training
```
#Single
python tools/train.py $config
#Multi
./tools/dist_train.sh $config num_gpu

#Examples
python tools/train.py configs/bevdepth/sit_bevdepth_r50_704x256.py --work-dir work_dirs/bevdepth
./tools/dist_train.sh configs/bevdepth/sit_bevdepth_r50_704x256.py 4 --work-dir work_dirs/bevdepth
```

## Evaluation
```
#Single
python tools/test.py $config $checkpoint --eval mAP 
python tools/test.py $config $checkpoint --out $name.pkl 
#Multi
./tools/dist_test.sh $config $checkpoint num_gpu -- out 
./tools/dist_test.sh $config $checkpoint num_gpu --eval mAP //Metric

#Examples
python tools/test.py configs/bevdepth/sit_bevdepth_r50_704x256.py work_dirs/bevdepth_aug/epoch_30.pth --eval mAP
python tools/test.py configs/bevdepth/sit_bevdepth_r50_704x256.py work_dirs/bevdepth_aug/epoch_30.pth --work_dirs/out result_val.pkl
python tools/test.py configs/bevdepth/sit_bevdepth_r50_704x256.py work_dirs/bevdepth_aug/epoch_30.pth --out work_dirs/result_test.pkl

./tools/dist_test.sh configs/bevdepth/sit_bevdepth_r50_704x256.py work_dirs/bevdepth_aug/epoch_30.pth 4 --eval mAP
./tools/dist_test.sh configs/bevdepth/sit_bevdepth_r50_704x256.py work_dirs/bevdepth_aug/epoch_30.pth 4 --out work_dirs/result_val.pkl
./tools/dist_test.sh configs/bevdepth/sit_bevdepth_r50_704x256.py work_dirs/bevdepth_aug/epoch_30.pth 4 --out work_dirs/result_test.pkl
```


## License

This project is released under the [Apache 2.0 license](LICENSE).

## Acknowledgement
- <a href="https://github.com/open-mmlab/mmdetection3d">mmdetection3d</a>
- <a href="https://github.com/HuangJunJie2017/BEVDet">BEVDet</a>


[![docs](https://img.shields.io/badge/docs-latest-blue)](https://mmdetection3d.readthedocs.io/en/latest/)
[![badge](https://github.com/open-mmlab/mmdetection3d/workflows/build/badge.svg)](https://github.com/open-mmlab/mmdetection3d/actions)
[![codecov](https://codecov.io/gh/open-mmlab/mmdetection3d/branch/master/graph/badge.svg)](https://codecov.io/gh/open-mmlab/mmdetection3d)
[![license](https://img.shields.io/github/license/open-mmlab/mmdetection3d.svg)](https://github.com/open-mmlab/mmdetection3d/blob/master/LICENSE)

