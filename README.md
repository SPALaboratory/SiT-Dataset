

<h1> <a href="https://spalaboratory.github.io/SiT/"> SiT Dataset: Socially Interactive Pedestrian Trajectory Dataset for Social Navigation Robots </a> </h1>

<!-- <p align="center">
  <a href="https://www.spa.hanyang.ac.kr/ " target="_blank">
    <img src="./images/230618_SPA_logo.png" align="center" width="55%">
  </a>
  <a href="https://www.hanyang.ac.kr/web/eng" target="_blank">
    <img src="./images/HYU_initial_eng.png" align="center" width="44%">
  </a>
</p>
 -->
 
<p align="center">
  </br>
  <a href="https://spalaboratory.github.io/SiT">
    <img src="./images/230620_SPA_logo.png" align="center" width="35%" alt="SPA Logo">
  </a>
</br></br>
</p>

<p align="center">
<a href="https://youtu.be/9zc-5Gxjvzk"><img src="https://img.shields.io/badge/Demo_Video-FF0000?style=for-the-badge&logo=youtube&logoColor=white" alt="YouTube Video"/></a>
<a href="https://drive.google.com/drive/folders/1kYGPJPoWn3J8s0mRWXZv9X-MOR5Hcg8u"><img src="https://img.shields.io/badge/Data_set-4285F4?style=for-the-badge&logo=google-drive&logoColor=white" alt="Data set"/></a>
<a href="https://spalaboratory.github.io/SiT/"><img src="https://img.shields.io/badge/home_page-181717?style=for-the-badge&logo=googlehome&logoColor=white" alt="Github.io"/></a>
</p>

<!-- ![Demo_data Illustration](./images/230605_main.png) -->

<p align="center">
  <img src="./images/fig_1_w_logo.png" alt>
   <em align="center">Example of SiT dataset</em>
</p>


## Updates
* [2023-05-29] We opened SiT Dataset <a href="https://github.com/SPALaboratory/SiT-Dataset">Github</a>.
* [2023-05-29] We opened SiT Dataset <a href="https://spalaboratory.github.io/SiT/">Website.</a>
* [2023-06-18] Semantic map data of SiT Dataset released on <a href="https://github.com/SPALaboratory/SiT-Dataset/tree/main/semantic-maps">Github.</a>


## Upcomings
* [2023-07] <a href="https://drive.google.com/drive/folders/1kYGPJPoWn3J8s0mRWXZv9X-MOR5Hcg8u?usp=sharing">SiT Mini-Dataset</a> release on public.
* [2023-07] Dockerfiles for each perception task release on <a href="https://hub.docker.com/u/spalaboratory">Dockerhub</a>.
* [2023-07] SiT benchmark and devkit for each perception task release.
* [2023-07] SiT Mini-Dataset Rosbag files release on public.
* [2023-09] SiT Full dataset with rosbag files release on public.
* [2023-09] Pretrained models for 3D object detection and Trajectory prediction release on public.
* [2023-10] Dockerfiles for each perception task release.
* [2024-01] SiT End-to-End pedestrain trajectory prediction challenge starts on Eval AI.
* [2023-07] Pretrained models for 3D object detection.
<!-- * [2023-09] SiT Full dataset upload. -->
<!-- * [2023-07] ROS bagfile of SiT Full dataset upload. -->
<!-- * [2023-08] SiT benchmark for end-to-end trajectory prediction release. -->

## Overview
Our Social Interactive Trajectory (SiT) dataset is a unique collection of pedestrian trajectories for designing advanced social navigation robots. It includes a range of sensor data, annotations, and offers a unique perspective from a robot navigating crowded environments, capturing dynamic human-robot interactions. It's meticulously organized for training and evaluating models across tasks like 3D detection, 3D multi-object tracking, and trajectory prediction, providing an end-to-end modular approach. It includes a comprehensive benchmark and exhibits the performance of several baseline models. This dataset is a valuable resource for future pedestrian trajectory prediction research, supporting the development of safe and agile social navigation robots.

## Robot Platform & Sensor Setup
![Sensor Setup Illustration](./images/230607_husky.png)
* Ubuntu 18.04 LTS
* ROS Melodic
* <a href="https://clearpathrobotics.com/husky-unmanned-ground-vehicle-robot/"> Clearpath Husky UGV </a>
* Velodyne VLP-16 * 2
* RGB Camera Basler a2A1920-51gc PRO GigE * 5
* MTi-680G IMU & GPS * 1
* VectorNAV VN-100 IMU * 1


## Ground Truth
We provide GT boxes for 2D and 3D data as below.
* 2D: Class name, Class name:Track ID, Camera number, Top left X coordinate, Top left Y coordinate, Width (w), and Height (h)
* 3D: Class name, Class name:Track ID, Height (h), Length (l), Width (w), X, Y, and Z coordinates, and rotation (rot).


<!-- 
### 3D Object Detection
This is the document for how to use our dataset for various perception tasks.
We tested the SiT Dataset for detection frameworks on the following enviroment:
* Python 3.8.13 (3.10+ does not support open3d.)
* Ubuntu 18.04/20.04
* Torch 1.11.0+cu113
* CUDA 11.3
* opencv 4.2.0.32

### 3D Multi-Object Tracking(3DMOT)
This is the document for how to use our dataset for various perception tasks.
We tested the SiT Dataset for Tracking frameworks on the following enviroment:
* Python 3.8.13 (3.10+ does not support open3d.)
* Ubuntu 18.04/20.04
* Torch 1.11.0+cu113
* CUDA 11.3
* opencv 4.2.0.32

### Pedestrian Trajectory Prediction
This is the document for how to use our dataset for various perception tasks.
We tested the SiT Dataset for prediction frameworks on the following enviroment:
* Python 3.8.13 (3.10+ does not support open3d.)
* Ubuntu 18.04/20.04
* Torch 1.11.0+cu113
* CUDA 11.3
* opencv 4.2.0.32 -->



## Benchmarks
We provide benchmarks and pretrained models for 3D pedestrian detection, 3D Multi-Object Tracking, Pedestrian Trajectory Prediction and End-to-End Motion Forecasting.

### <a href="https://github.com/SPALaboratory/SiT-Dataset/tree/main/detection">3D Object Detection</a>

|**Methods**|**Modality**|**mAP &uarr;**| **AP(0.25) &uarr;** |**AP(0.5) &uarr;** | **AP(1.0) &uarr;** | **AP(2.0) &uarr;** | **Pretrained** |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|**<a href="https://www.arxiv-vanity.com/papers/2104.10956/">FCOS3D</a>**|Camera| 0.131 | 0.054 | 0.147 | 0.162 | 0.162 | <a href="">TBD</a>|
|**<a href="https://openaccess.thecvf.com/content_CVPR_2019/html/Lang_PointPillars_Fast_Encoders_for_Object_Detection_From_Point_Clouds_CVPR_2019_paper.html">PointPillars</a>**|LiDAR| 0.319 | 0.202 | 0.316 | 0.346 | 0.414 |<a href="">TBD</a>|
|**<a href="https://openaccess.thecvf.com/content/CVPR2021/html/Yin_Center-Based_3D_Object_Detection_and_Tracking_CVPR_2021_paper.html">CenterPoint-P</a>**|LiDAR| 0.382 | 0.233 | 0.388 | 0.424 | 0.482 |<a href="">TBD</a>|
|**<a href="https://openaccess.thecvf.com/content/CVPR2021/html/Yin_Center-Based_3D_Object_Detection_and_Tracking_CVPR_2021_paper.html">CenterPoint-V</a>**|LiDAR| 0.514 | 0.352 | **0.522**| 0.556 | 0.620 |<a href="">TBD</a>|
|**<a href="https://openaccess.thecvf.com/content/CVPR2022/html/Bai_TransFusion_Robust_LiDAR-Camera_Fusion_for_3D_Object_Detection_With_Transformers_CVPR_2022_paper.html">Transfusion-P</a>**|Fusion| 0.396 | 0.213 | 0.371 | 0.451 | 0.551 |<a href="">TBD</a>|
|**<a href="https://openaccess.thecvf.com/content/CVPR2022/html/Bai_TransFusion_Robust_LiDAR-Camera_Fusion_for_3D_Object_Detection_With_Transformers_CVPR_2022_paper.html">Transfusion-V</a>**|Fusion| **0.533** | **0.360** | 0.512 | **0.587** | **0.672** |<a href="">TBD</a>|

### <a href="https://github.com/SPALaboratory/SiT-Dataset/tree/main/tracking">3D Multi-Object Trajectory Tracking</a>



|**Method**| **sAMOTA&uarr;** | **AMOTA&uarr;** | **AMOTP(m)&darr;** | **MOTA&uarr;**| **MOTP(m)&darr;** | **IDS&darr;** |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|**<a href="https://openaccess.thecvf.com/content_CVPR_2019/html/Lang_PointPillars_Fast_Encoders_for_Object_Detection_From_Point_Clouds_CVPR_2019_paper.html">PointPillars</a> + <a href="https://arxiv.org/abs/2008.08063">AB3DMOT</a>** | 0.3679 | 0.0826 | 0.5125 | 0.2073 | 0.9702 | 1048 |
|**<a href="https://openaccess.thecvf.com/content/CVPR2021/html/Yin_Center-Based_3D_Object_Detection_and_Tracking_CVPR_2021_paper.html">Centerpoint Detector</a> + <a href="https://arxiv.org/abs/2008.08063">AB3DMOT</a>** | 0.4626 | 0.1159 | 0.3757 | 0.3438 | 0.8360 | **554** |
|**<a href="https://openaccess.thecvf.com/content/CVPR2021/html/Yin_Center-Based_3D_Object_Detection_and_Tracking_CVPR_2021_paper.html">Centerpoint Tracker</a>** | **0.7244** | **0.2793** | **0.2611** | **0.5150** | **0.4274** | 1136 |

### <a href="https://github.com/SPALaboratory/SiT-Dataset/tree/main/prediction">Pedestrian Trajectory Prediction</a>

<!-- |**Vanilla LSTM**| X|1.156 | 2.205 | 1.601 | 3.157 | <a href="">TBD</a>| -->
|**Name**|**Map**|**ADE<sub>5<sub>** **&darr;**|**FDE<sub>5<sub>** **&darr;**| **ADE<sub>20<sub>** **&darr;** | **FDE<sub>20<sub>** **&darr;** | **Pretrained** |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|**<a href="https://openaccess.thecvf.com/content_cvpr_2016/html/Alahi_Social_LSTM_Human_CVPR_2016_paper.html">Social-LSTM</a>**| X | 1.336 | 2.554 | 1.319 | 2.519 | <a href="">TBD</a>|
|**<a href="https://openaccess.thecvf.com/content/ICCV2021/html/Mangalam_From_Goals_Waypoints__Paths_to_Long_Term_Human_Trajectory_ICCV_2021_paper.html">Y-NET</a>**| X | 1.188 | 2.427 | 0.640 | 1.547 | <a href="">TBD</a>|
|**<a href="https://openaccess.thecvf.com/content/ICCV2021/html/Mangalam_From_Goals_Waypoints__Paths_to_Long_Term_Human_Trajectory_ICCV_2021_paper.html">Y-NET</a>**| O | 1.036 | 2.306 | 0.596 | 1.370 | <a href="">TBD</a>|
  |**<a href="https://link.springer.com/chapter/10.1007/978-3-031-19830-4_22">NSP-SFM</a>**| X | 1.036 | 1.947 | 0.529 | 0.936 | <a href="">TBD</a>|
|**<a href="https://link.springer.com/chapter/10.1007/978-3-031-19830-4_22">NSP-SFM</a>**| O | **0.808** | **1.549** | **0.443** | **0.807** | <a href="">TBD</a>|

### End-to-End Pedestrian Motion Forecasting
|**Method**| **mAP** **&uarr;** |  **mAP<sub>f<sub>** **&uarr;** | **ADE<sub>5<sub>** **&darr;** | **FDE<sub>5<sub>** **&darr;** | **Pretrained** |
|:---:|:---:|:---:|:---:|:---:|:---:|
|**<a href="https://openaccess.thecvf.com/content_cvpr_2018/html/Luo_Fast_and_Furious_CVPR_2018_paper.html">Fast and Furious</a>**| **0.490** | **0.079** | **1.915** | **3.273** |<a href="">TBD</a>|
|**<a href="https://openaccess.thecvf.com/content/CVPR2022/html/Peri_Forecasting_From_LiDAR_via_Future_Object_Detection_CVPR_2022_paper.html">FutureDet-P</a>**|0.209 | 0.037 | 2.532 | 4.537|<a href="">TBD</a>|
|**<a href="https://openaccess.thecvf.com/content/CVPR2022/html/Peri_Forecasting_From_LiDAR_via_Future_Object_Detection_CVPR_2022_paper.html">FutureDet-V</a>**|0.408 | 0.053 | 2.416 | 4.409|<a href="">TBD</a>|

## Download Dataset
* Download SiT Mini dataset from below Google Drive link. <br>
**<a href="https://drive.google.com/drive/folders/1kYGPJPoWn3J8s0mRWXZv9X-MOR5Hcg8u?usp=sharing"> Click Download link. </a> <br>**

* Full dataset and Rosbag files will be uploaded(TBD).

## ROS Bag Raw Data
ROS bagfiles include below sensor data:
| Topic Name | Message Type | Message Descriptison |
:------------ | :------------- | :---------------------------------
/29_camera/pylon_camera_node/<br>image_raw/compressed  | sensor_msgs/CompressedImage  | Compressed Bayer Image by Basler a2A1920-51gv PRO GigE
/41_camera/pylon_camera_node/<br>image_raw/compressed  | sensor_msgs/CompressedImage  | Compressed Bayer Image by Basler a2A1920-51gv PRO GigE
/46_camera/pylon_camera_node/<br>image_raw/compressed  | sensor_msgs/CompressedImage  | Compressed Bayer Image by Basler a2A1920-51gv PRO GigE
/47_camera/pylon_camera_node/<br>image_raw/compressed  | sensor_msgs/CompressedImage  | Compressed Bayer Image by Basler a2A1920-51gv PRO GigE
/65_camera/pylon_camera_node/<br>image_raw/compressed  | sensor_msgs/CompressedImage  | Compressed Bayer Image by Basler a2A1920-51gv PRO GigE
/bottom/velodyne_points | sensor_msgs/PointCloud2 | Pointcloud by Velodyne VLP-16
/top/velodyne_points | sensor_msgs/PointCloud2 | Pointcloud by Velodyne VLP-16
/vn100/vectornav/IMU | sensor_msgs/Imu | VN-100 IMU
/xsens/filter/position_interpolated  | geometry_msgs/Vector3Stamped | Interpolated GNSS data to the timestep of top velodyne
/xsens/filter/positionlla  | geometry_msgs/Vector3Stamped | GNSS by MTi-680
/xsens/imu/data      | sensor_msgs/Imu | IMU by MTi-680
/xsens/imu_interpolated   | sensor_msgs/Imu | Interpolated IMU data to the timestep of top velodyne
  
  
## License <a rel="license_cc" href="http://creativecommons.org/licenses/by-nc-nd/4.0/"><img alt="by-nc-nd_4.0" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-nd/4.0/88x31.png" /></a> <a rel="license_apache_2"><img alt="apache_2.0" style="border-width:0; width:8%;" src="https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/ASF_Logo.svg/220px-ASF_Logo.svg.png"/></a>

The SiT dataset is published under the CC BY-NC-ND License 4.0, and all codes are published under the Apache License 2.0.
<!-- 

## Citation
```
@misc{sitdataset,
      title={SiT Dataset: Data, Benchmarks and Analysis}, 
      author={Jongwook Bae, Jungho Kim, Junyong Yun, Changwon Kang, Junho Lee, Jeongseon Choi, Chanhyeok Kim, and Jun-Won Choi},
      year={2023},
      eprint={},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
``` -->

## Acknowledgement
The SiT dataset is contributed by [Jongwook Bae](https://github.com/Eddie-JUB), [Jungho Kim](https://github.com/SPA-junghokim), [Junyong Yun](https://github.com/JunyongYun-SPA), [Changwon Kang](https://github.com/rkdckddnjs9), [Junho Lee](https://github.com/jhlee-ai), [Jeongseon Choi](https://github.com/junction824), [Chanhyeok Kim](https://github.com/chkim-spa), [Jungwook Choi](https://jchoi-hyu.github.io/), advised by [Jun-Won Choi](https://www.spa.hanyang.ac.kr/faculty).
<!--다른 모델 저자들 링크  -->
We thank the maintainers of the following projects that enable us to develop SiT Dataset: [`MMDetection`](https://github.com/open-mmlab/mmdetection) by MMLAB
