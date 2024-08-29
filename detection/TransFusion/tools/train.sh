# CONFIG=transfusion_spa_nusc_voxel_LC_only_ped_augX
# CONFIG_PATH=DotProd
# CUDA_LAUNCH_BLOCKING=1 PORT=11112 CUDA_VISIBLE_DEVICES=2,3 ./tools/dist_train.sh projects/configs/$CONFIG_PATH/$CONFIG.py 2 —work-dir ./work_dirs/$CONFIG_PATH/$CONFIG
# PORT=11112 CUDA_VISIBLE_DEVICES=2 ./tools/dist_test.sh ./projects/configs/$CONFIG_PATH/$CONFIG.py ./work_dirs/$CONFIG_PATH/$CONFIG/epoch_12_ema.pth 1 —eval mAp  >> ./work_dirs/$CONFIG_PATH/$CONFIG/result12ema.txt
# python send_result_mail.py ./work_dirs/$CONFIG_PATH/$CONFIG/result12ema.txt 13

# sleep 7h
# CONFIG=transfusion_spa_nusc_voxel_LC_only_ped_augX
# PORT=11112 CUDA_VISIBLE_DEVICES=3 ./tools/dist_train.sh configs/$CONFIG.py 


CUDA_VISIBLE_DEVICES=3 python tools/train.py configs/transfusion_spa_pillar_LC_only_ped_v2_000005.py --seed 2024


