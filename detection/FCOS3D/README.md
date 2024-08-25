

## Introduction
Research on 3D object recognition models using cameras in autonomous driving is actively underway. In this context, we provide the baseline for the FCOS3D model.

### FCOS3D
FCOS3D is a model for 3D object detection that leverages a fully convolutional one-stage detector framework. It operates on monocular images to predict 3D bounding boxes, including depth, size, and orientation of objects. The model is designed to improve accuracy and efficiency in 3D object detection for applications like autonomous driving.

The performance of the baseline models on the validation set is as follows.


|**Methods**|**Modality**|**mAP &uarr;**| **AP(0.25) &uarr;** |**AP(0.5) &uarr;** | **AP(1.0) &uarr;** | **AP(2.0) &uarr;** | **Pretrained** |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|**FCOS3D**|Camera| 0.1696 |0.0 | 0.0366 | 0.2186 | 0.4234 |<a href="https://drive.google.com/file/d/1RttgPBNatAI2Nm6po2IzvLQPS2rU3Y3-/view?usp=sharing">Gdrive</a>|


## Environment
```
sudo docker run -it --gpus all --ipc=host -v /mnt:/mnt --name sit_fcos3d geunjubaek/sit_fcos3d /bin/bash
```

## Preprocess
```
python2 pcd2bin_all_frame_concat    # We upload only pcd file for convenience. Therefore, you should convert pcd to bin file with python2 not 3.
python sit_undistortion_image       # You should also undistort the images.
python visualize_3dbox_on_image     # We provide a code of 3d boxes visualization on images.
```


## Data generation
```
# After git clone, just use setup.py 
python setup.py develop

# Make pkl, json file 
# Create train, val pkl 
python tools/create_data_mono.py sit --root-path ./data/sit_full --version sit-trainval --out-dir ./data/sit_full --extra-tag sit --max-sweeps 1
# Create test pkl
python tools/create_data_mono.py sit --root-path ./data/sit_full --version sit-test --out-dir ./data/sit_full --extra-tag sit --max-sweeps 1
```


## Training
```
python tools/train.py $config
python tools/train.py configs/sit_fcos3d/sit_fcos3d_r50_960x600.py
```

## Evaluation
```
# We only provide you evaluation code in multi-view setting
# Please use this code when you finish training your model.
python tools/test_mv.py ./configs/sit_fcos3d/sit_fcos3d_r50_960x600.py ./work_dirs/sit_fcos3d_r50_960x600/epoch_30.pth --eval EVAL --gpu-id 0 
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

