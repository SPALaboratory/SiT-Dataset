# #aug 적용 X
./tools/dist_test.sh configs/bevdepth/bevdepth.py work_dirs/bevdepth/epoch_30.pth 4 --eval mAP
# #iad
./tools/dist_test.sh configs/bevdepth/bevdepth.py work_dirs/bevdepth_ida/epoch_30.pth 4 --eval mAP
#bda
./tools/dist_test.sh configs/bevdepth/bevdepth.py work_dirs/bevdepth_bda/epoch_30.pth 4 --eval mAP
#aug(bda+ida)
./tools/dist_test.sh configs/bevdepth/bevdepth.py work_dirs/bevdepth_aug/epoch_30.pth 4 --eval mAP
