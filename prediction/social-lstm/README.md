
# Getting Started

## Dataset Download
See the main github page.


## File tree
Please set up the configuration to match the tree structure below.
```
ln -s {SiT Dataset} SiT_dataset
```
```
social-lstm
├── SiT_dataset
├── data
│   ├── trajectory_txt
│   ├── trajectories_train.cpkl
│   ├── trajectories_val.cpkl
│   ├── trajectories_test.cpkl
```


## Docker pull
```shell
# pull and run docker image, immediately. 
sudo docker run -it -e DISPLAY=unix$DISPLAY --gpus all --ipc=host -v {local_social-lstm}:/mnt/social-lstm -e XAUTHORITY=/tmp/.docker.xauth --name sit_social_lstm spalaboratory/social-lstm:torch1.9.1_cu111 /bin/bash

# docker start & docker exec
docker start sit_social_lstm && docker exec -it sit_social_lstm /bin/bash
```


## Preprocess
```
python /preprocess/preprocess_socialLSTM_JY_0603_deepen.py
```

## Training
```
python train.py
```


## Inference for validation set
```
python validation_sample20.py
```

