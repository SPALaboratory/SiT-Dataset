# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
# point_cloud_range = [-20, -20, -3, 20, 20, 3]

class_names = [
    'pedestrian',
]
dataset_type = 'SPA_Nus_Dataset_Top'
data_root = 'data/spa/'
# Input modality for nuScenes dataset, this is consistent with the submission
# format which requires the information in input_modality.
input_modality = dict(
    use_lidar=True,
    use_camera=False,
    use_radar=False,
    use_map=False,
    use_external=False)
file_client_args = dict(backend='disk')
# Uncomment the following if use ceph or other file clients.
# See https://mmcv.readthedocs.io/en/latest/api.html#mmcv.fileio.FileClient
# for more details.
# file_client_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/nuscenes/': 's3://nuscenes/nuscenes/',
#         'data/nuscenes/': 's3://nuscenes/nuscenes/'
#     }))
train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        file_client_args=file_client_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    # dict(
    #     type='GlobalRotScaleTrans',
    #     rot_range=[-0.3925, 0.3925],
    #     scale_ratio_range=[0.95, 1.05],
    #     translation_std=[0, 0, 0]),
    # dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='PointShuffle'),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        file_client_args=file_client_args),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            # dict(
            #     type='GlobalRotScaleTrans',
            #     rot_range=[0, 0],
            #     scale_ratio_range=[1., 1.],
            #     translation_std=[0, 0, 0]),
            # dict(type='RandomFlip3D'),
            dict(
                type='PointsRangeFilter', point_cloud_range=point_cloud_range),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ])
]
# construct a pipeline for data and gt loading in show function
# please keep its loading function consistent with test_pipeline (e.g. client)
eval_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        file_client_args=file_client_args),
    dict(
        type='DefaultFormatBundle3D',
        class_names=class_names,
        with_label=False),
    dict(type='Collect3D', keys=['points'])
]

train_load_interval = 100
with_velocity = False
data = dict(
    samples_per_gpu=1, 
    workers_per_gpu=1,
    # train=dict(
    #     type=dataset_type,
    #     data_root=data_root,
    #     ann_file=data_root + 'spa_nusc_infos_train.pkl',
    #     pipeline=train_pipeline,
    #     classes=class_names,
    #     modality=input_modality,
    #     test_mode=False,
    #     # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
    #     # and box_type_3d='Depth' in sunrgbd and scannet dataset.
    #     box_type_3d='LiDAR'),
    train=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
                        type=dataset_type,
                        data_root=data_root,
                        ann_file=data_root + 'spa_nusc_top_infos_train.pkl',
                        pipeline=train_pipeline,
                        classes=class_names,
                        modality=input_modality,
                        test_mode=False,
                        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
                        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
                        box_type_3d='LiDAR'
            )
        ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'spa_nusc_top_infos_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR'),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'spa_nusc_top_infos_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR'))

    # train=dict(
    #     type='CBGSDataset',
    #     dataset=dict(
    #         type=dataset_type,
    #         data_root=data_root,
    #         ann_file=data_root + 'spa_nusc_infos_train.pkl',
    #         pipeline=train_pipeline,
    #         load_interval=train_load_interval,
    #         classes=class_names,
    #         with_velocity=with_velocity,
    #         test_mode=False,
    #         use_valid_flag=True,
    #         # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
    #         # and box_type_3d='Depth' in sunrgbd and scannet dataset.
    #         box_type_3d='LiDAR')),
    # val=dict(
    #     pipeline=test_pipeline,
    #     classes=class_names,
    #     ann_file=data_root + 'spa_nusc_infos_val.pkl',
    #     with_velocity=with_velocity),
    # test=dict(
    #     pipeline=test_pipeline,
    #     classes=class_names,
    #     ann_file=data_root + 'spa_nusc_infos_val.pkl',
    #     with_velocity=with_velocity))


# For nuScenes dataset, we usually evaluate the model at the end of training.
# Since the models are trained by 24 epochs by default, we set evaluation
# interval to be 24. Please change the interval accordingly if you do not
# use a default schedule.
evaluation = dict(interval=20, pipeline=eval_pipeline)
