# Copyright (c) Phigent Robotics. All rights reserved.

dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './result/work_ATT_test'
load_from =  None
resume_from = None
no_validate = True


#for IterBasedRunner
# runner = dict(type='IterBasedRunner', max_epochs=24) #IterBasedRunner
# workflow = [('train', 20), ('val', 5)] #for IterBasedRunner
# max_loops = 50 #for IterBasedRunner

#for EpochBasedRunner
runner = dict(type='EpochBasedRunner', max_epochs=1)
workflow = [('train', 1), ('val', 1)]#, ('test', 1)]
max_loops = 1

checkpoint_config = dict(interval=500)
# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'

# For nuScenes we usually do 10-class detection
class_names = []

data_config = {
    'cams': [],
    'Ncams': 1,
    'input_size': (960, 640),
    'src_size': (1920, 1280),
    'thickness': 2,
    'angle_class': 36,

    # Augmentation
    'resize': (-0.06, 0.11),
    'rot': (-5.4, 5.4),
    'flip': False,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,

}
grid_config = {
    'x': [-19.2, 19.2, 0.2],
    'y': [0.0, 96.0, 0.2],
    'z': [-5, 3, 8],
    'grid_height': 6,
    'depth': [1.0, 60.0, 1.0],
    'grid_res': [0.2, 0.2, 0.6],
    'offset': [-19.2, 0.0, 1.08],
    'grid_size': [38.4, 96],
    'depth_discretization': 'LID',
    'dbound': [4.0, 70.0, 0.6]
}


model = dict(
    type='LaneATT2DDetector',
    img_backbone=dict(
        pretrained='torchvision://resnet34',  #'torchvision://resnet50',
        type='ResNet',
        depth=34,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        with_cp=True,
        style='pytorch'),
    # img_neck=dict(
    #     type='DLANeck',
    #     in_channels=[256, 512],#[1024, 2048],
    #     start_level=0,
    #     end_level=-1),
    img_neck=dict(
        type='CustomFPN',
        in_channels=[128, 256, 512],#[1024, 2048],
        out_channels=256,
        num_outs=1,
        start_level=0,
        out_ids=[0]),
    lane_head=dict(
        type='LaneATT2DHead',
        in_channel=256,
        num_classes=1,
        seg_loss=dict(type='Lane_FocalLoss'),
        haf_loss=dict(type='RegL1Loss'),
        vaf_loss=dict(type='RegL1Loss'),
    ))

# Data
dataset_type = 'OpenLane_Dataset_ATT'#'IterBasedRunner', EpochBasedRunner
images_dir = '/workspace/openlane_all/images'
json_file_dir_train = '/workspace/openlane_all/lane3d_300/training/'
json_file_dir_val = '/workspace/openlane_all/lane3d_300/validation/'
file_client_args = dict(backend='disk')

train_pipeline = [
    dict(
        type='LoadAnnotationsSeg',
        bda_aug_conf=None,
        classes=class_names,
        is_train=False),

    dict(
        type='Collect3D', 
        keys=['img_inputs', 'maps_bev', 'maps_2d'], 
        meta_keys=['gt_lanes_3d', 'gt_lanes_2d', 'file_path'])
]

test_pipeline = [
    dict(
        type='LoadAnnotationsSeg',
        bda_aug_conf=None,
        classes=class_names,
        is_train=False),

    dict(
        type='Collect3D', 
        keys=['img_inputs', 'maps_bev', 'maps_2d'], 
        meta_keys=['gt_lanes_3d', 'gt_lanes_2d', 'file_path'])
]

eval_pipeline = [
    dict(
        type='LoadAnnotationsSeg',
        bda_aug_conf=None,
        classes=class_names,
        is_train=False),

    dict(
        type='Collect3D', 
        keys=['img_inputs', 'maps_bev', 'maps_2d'], 
        meta_keys=['gt_lanes_3d', 'gt_lanes_2d', 'file_path'])
]

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

share_data_config = dict(
    type=dataset_type,
    # modality=input_modality,
)

test_data_config = dict(
    images_dir=images_dir,
    json_file_dir=json_file_dir_val,
    data_config=data_config,
    grid_config=grid_config,
    pipeline=test_pipeline,
    CLASSES=None,
    test_mode=True,
    use_valid_flag=False,
)

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=0,
    shuffle=True,
    train=dict(
        images_dir=images_dir,
        json_file_dir=json_file_dir_train,
        data_config=data_config,
        grid_config=grid_config,
        pipeline=train_pipeline,
        CLASSES=None,
        test_mode=False,
        use_valid_flag=False,
    ),

    val=test_data_config,
    test=test_data_config)

# evaluation = dict(interval=1, pipeline=eval_pipeline)

for key in ['train', 'val', 'test']:
    data[key].update(share_data_config)

# Optimizer
optimizer = dict(type='AdamW', lr=2e-4, weight_decay=1e-07)
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=0.001,
    step=[24,])

log_config = dict(
    interval=4,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

custom_hooks = [
    # dict(
    #     type='MEGVIIEMAHook',
    #     init_updates=10560,
    #     priority='NORMAL',
    # ),
    dict(
        type='BevLaneVisAllHook',
    ),
]


# unstable
# fp16 = dict(loss_scale='dynamic')