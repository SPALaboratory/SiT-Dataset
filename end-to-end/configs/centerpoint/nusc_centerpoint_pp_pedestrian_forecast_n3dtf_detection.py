import itertools
import logging

from det3d.utils.config_tool import get_downsample_factor

timesteps = 7
DOUBLE_FLIP=False
TWO_STAGE=False
REVERSE=False 
SPARSE=False
DENSE=True
BEV_MAP=False
FORECAST_FEATS=True
CLASSIFY=False
WIDE=False

sampler_type = "trajectory"

tasks = [
    #dict(num_class=1, class_names=["car"]),
    dict(num_class=1, class_names=["pedestrian"]),
]

class_names = list(itertools.chain(*[t["class_names"] for t in tasks]))

# training and testing settings
target_assigner = dict(
    tasks=tasks,
)


# model settings
model = dict(
    type="PointPillars",
    pretrained=None,
    reader=dict(
        type="PillarFeatureNet",
        num_filters=[64, 64],
        num_input_features=5,
        with_distance=False,
        voxel_size=(0.2, 0.2, 8),
        pc_range=(-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
    ),
    backbone=dict(type="PointPillarsScatter", ds_factor=1),
    neck=dict(
        type="RPN",
        layer_nums=[3, 5, 5],
        ds_layer_strides=[2, 2, 2],
        ds_num_filters=[64, 128, 256],
        us_layer_strides=[0.5, 1, 2],
        us_num_filters=[128, 128, 128],
        num_input_features=64,
        logger=logging.getLogger("RPN"),
    ),
    bbox_head=dict(
        # type='RPNHead',
        type="CenterHead",
        in_channels=sum([128, 128, 128]),
        tasks=tasks,
        dataset='nuscenes',
        weight=0.25,
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        common_heads={'reg': (2, 2), 'height': (1, 2), 'dim':(3, 2), 'rot':(2, 2), 'vel': (2, 2)}, # (output_channel, num_conv)
        share_conv_channel=64,
        dcn_head=False,
        timesteps=timesteps,
        two_stage=TWO_STAGE,
        reverse=REVERSE,
        sparse=SPARSE,
        dense=DENSE,
        bev_map=BEV_MAP,
        forecast_feature=FORECAST_FEATS,
        classify=CLASSIFY,
        wide_head=WIDE,
    ),
)

assigner = dict(
    target_assigner=target_assigner,
    out_size_factor=get_downsample_factor(model),
    gaussian_overlap=0.1,
    max_objs=1000,
    min_radius=2,
    radius_mult=True,
    sampler_type=sampler_type,
)


train_cfg = dict(assigner=assigner)

test_cfg = dict(
    post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
    max_per_img=500,
    nms=dict(
        nms_pre_max_size=1000,
        nms_post_max_size=83,
        nms_iou_threshold=0.2,
    ),
    score_threshold=0.1,
    pc_range=[-51.2, -51.2],
    out_size_factor=get_downsample_factor(model),
    voxel_size=[0.2, 0.2]
)

# dataset settings
dataset_type = "NuScenesDataset"
nsweeps = 20
data_root = "/home/ubuntu/Workspace/Data/nuScenes/trainval_forecast"

if sampler_type == "standard":
    sample_group=[
        #dict(car=2),
        dict(pedestrian=2),
    ]
else:
    sample_group=[
        #dict(static_car=2),
        dict(static_pedestrian=2),
        #dict(linear_car=4),
        dict(linear_pedestrian=2),
        #dict(nonlinear_car=6),
        dict(nonlinear_pedestrian=4),
    ]

db_sampler = dict(
    type="GT-AUG",
    enable=False,
    db_info_path= data_root + "/dbinfos_train_20sweeps_withvelo.pkl",
    sample_groups=sample_group,
    db_prep_steps=[
        dict(
            filter_by_min_num_points=dict(
                car=5,
                #pedestrian=5,
            )
        ),
        dict(filter_by_difficulty=[-1],),
    ],
    global_random_rotation_range_per_object=[0, 0],
    rate=1.0,
    sampler_type=sampler_type
    )

train_preprocessor = dict(
    mode="train",
    shuffle_points=True,
    global_rot_noise=[-0.3925, 0.3925],
    global_scale_noise=[0.95, 1.05],
    global_translate_std=0.5,
    db_sampler=db_sampler,
    class_names=class_names,
    sampler_type=sampler_type
)

val_preprocessor = dict(
    mode="val",
    shuffle_points=False,
    sampler_type=sampler_type
)

voxel_generator = dict(
    range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
    voxel_size=[0.2, 0.2, 8],
    max_points_in_voxel=20,
    max_voxel_num=[30000, 60000],
)

train_pipeline = [
    dict(type="LoadPointCloudFromFile", dataset=dataset_type),
    dict(type="LoadPointCloudAnnotations", with_bbox=True),
    dict(type="Preprocess", cfg=train_preprocessor),
    dict(type="Voxelization", cfg=voxel_generator),
    dict(type="AssignLabel", cfg=train_cfg["assigner"]),
    dict(type="Reformat"),
]
test_pipeline = [
    dict(type="LoadPointCloudFromFile", dataset=dataset_type),
    dict(type="LoadPointCloudAnnotations", with_bbox=True),
    dict(type="Preprocess", cfg=val_preprocessor),
    dict(type="Voxelization", cfg=voxel_generator),
    dict(type="AssignLabel", cfg=train_cfg["assigner"]),
    dict(type="Reformat"),
]

train_anno = data_root + "/infos_train_20sweeps_withvelo_filter_True.pkl"
val_anno = data_root + "/infos_val_20sweeps_withvelo_filter_True.pkl"
test_anno = data_root + "/infos_test_20sweeps_withvelo_filter_True.pkl"

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=train_anno,
        ann_file=train_anno,
        nsweeps=nsweeps,
        class_names=class_names,
        pipeline=train_pipeline,
        timesteps=timesteps,
    ),
    val=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=val_anno,
        test_mode=True,
        ann_file=val_anno,
        nsweeps=nsweeps,
        class_names=class_names,
        pipeline=test_pipeline,
        timesteps=timesteps
    ),
    test=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=test_anno,
        ann_file=test_anno,
        nsweeps=nsweeps,
        class_names=class_names,
        pipeline=test_pipeline,
        timesteps=timesteps
    ),
)


optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# optimizer
optimizer = dict(
    type="adam", amsgrad=0.0, wd=0.01, fixed_wd=True, moving_average=False,
)
lr_config = dict(
    type="one_cycle", lr_max=0.001, moms=[0.95, 0.85], div_factor=10.0, pct_start=0.4,
)

checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=25,
    hooks=[
        dict(type="TextLoggerHook"),
        # dict(type='TensorboardLoggerHook')
    ],
)
# yapf:enable
# runtime settings
total_epochs = 20
device_ids = range(8)
dist_params = dict(backend="nccl", init_method="env://")
log_level = "INFO"
work_dir = './models/{}/'.format(__file__[__file__.rfind('/') + 1:-3])
load_from = None
resume_from = None 
workflow = [('train', 1)]

