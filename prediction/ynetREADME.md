
# Getting Started

## Dataset Download
See the main github page.


## File tree
Please set up the configuration to match the tree structure below.
```
ln -s {SiT Dataset} SiT_dataset
```
```
ynet
├── SiT_dataset
├── data
│   ├── raster/
│   ├── tran/
│   ├── val/
│   ├── test/
```


## Docker pull
```shell
# pull and run docker image, immediately. 
sudo docker run -it -e DISPLAY=unix$DISPLAY --gpus all --ipc=host -v {local_ynet}:/mnt/ynet -e XAUTHORITY=/tmp/.docker.xauth --name sit_ynet spalaboratory/ynet:torch1.9.1_cu111 /bin/bash

# docker start & docker exec
docker start sit_ynet && docker exec -it sit_ynet /bin/bash
```


## Preprocess
```
python preprocess.py
```

## Training
```
python train.py
```


## Inference for validation set
```
python test.py
```


## Save prediction trajectories for NSP model
```
python test_for_nsp.py
```
