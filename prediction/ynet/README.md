
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
├─ SiT_dataset
├─ config
├─ utils
├─ data
│   ├─ semantic_maps_json
│   ├─ semantic_map
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
python preprocess_ped.py # for pedestrian history
python preprocess_map.py # for semantic map
```

## Training
```
python script_train.py
```


## Inference for training & validation set
This code stores the prediction results for NSP. First, it performs inference on the train set, followed by the validation set. It also provides separate errors for indoor and outdoor. Change the variable "MODEL_WEIGHT" in the script_test.py file with the name of the saved pretrained weights in "pretrained". Please move the results saved in the "./result_for_nsp" folder to be used for training/inference in NSP.
```
python script_test.py
```

