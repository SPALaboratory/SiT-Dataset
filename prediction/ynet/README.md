
# Getting Started

## Dataset Download
See the main github page.


## File tree
Please set up the configuration to match the tree structure below.
```
ln -s {$SiT-dataset} ./data/SiT
```
```
ynet
├── data/SiT
│   ├── raster/
│   ├── tran/
│   ├── val/
│   ├── test/
```


## Docker images
```shell
docker pull spalabatory/ynet:torch1.9.1_cu111
docker run spalabatory/ynet:torch1.9.1_cu111
```


## Preprocess
```
python preprocess.py
```

## Training
```
python train.py
```


## Inference for Validation set
```
python test.py
```


## Save prediction trajectories for NSP model
```
python test_for_nsp.py
```
