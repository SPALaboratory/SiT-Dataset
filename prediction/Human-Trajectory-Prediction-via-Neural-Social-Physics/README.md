
# Getting Started

## Dataset Download
See the main github page.


## File tree
Please set up the configuration to match the tree structure below.
```
ln -s {SiT Dataset} SiT_dataset
```
```
Human-Trajectory-Prediction-via-Neural-Social-Physics
├── SiT_dataset
├── data
│   ├── trajectory_txt
│   ├── tran.pkl
│   ├── val.pkl
│   ├── test.pkl
```


## Docker pull
```shell
# pull and run docker image, immediately. 
sudo docker run -it -e DISPLAY=unix$DISPLAY --gpus all --ipc=host -v {local_NSP}:/mnt/Human-Trajectory-Prediction-via-Neural-Social-Physics -e XAUTHORITY=/tmp/.docker.xauth --name sit_NSP spalaboratory/Human-Trajectory-Prediction-via-Neural-Social-Physics:torch1.9.1_cu111 /bin/bash

# docker start & docker exec
docker start sit_NSP && docker exec -it sit_NSP /bin/bash
```


## Preprocess
```
python preprocess_goalInfo.py #Y-net results preprocessing
```

## Training
```
python train_goals.py #train NSP w/o agent & map interaction
python train_nsp_wo.py #train NSP
```


## Inference for training & validation set
```
python valid_nsp_wo.py
```

