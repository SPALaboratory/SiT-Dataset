## 3D Pedestrian MOT


# Evaluation

## AB3DMOT

### Convert detection format to tracking format
```shell
# Convert detection results to tracking format
cd AB3DMOT/data/spa/convert_det_format
python convert_det_result_oxt.py

# Convert labels to tracking label format
cd AB3DMOT/scripts/spa
python convert_det_to_track.py
```

### Tracker
```
cd AB3DMOT
python main.py --dataset spa --det_name centerpoint
```

### Evaluation
```
cd AB3DMOT
python scripts/spa/evaluate.py centerpoint_pedestrian_val_H1 1 val
```

## CenterPoint

### Tracker
```
cd CenterPoint/tools/spa_tracking
python pub_test.py
```

### Evaluation
```shell
# Convert CenterPoint tracker results to AB3DMOT evaluation format
cp tracking_result.json AB3DMOT/results/spa/CenterPoint_results
cd AB3DMOT/results/spa/CenterPoint_results
python convert_result.py

# run evaluation
cd AB3DMOT/scripts/spa/evaluate_quick.py CenterPoint_results 1 val
```


