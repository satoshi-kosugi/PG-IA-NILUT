exp_name = 'pg_ia_nilut_fivekrgb_third'

custom_imports=dict(
    imports=['adaint'],
    allow_failed_imports=False)

# model settings
model = dict(
    type='PARAM_PREDICTOR',
    n_ranks=5,
    n_vertices=33,
    backbone='tpami', # 'res18'
    pretrained=False,
    n_colors=3,
    recons_loss=dict(type='MSELoss', loss_weight=1.0, reduction='mean'),
    ### Replace pretrained_PG_IA_NILUT_path with the path to the best-performing model.
    pretrained_PG_IA_NILUT_path="work_dirs/pg_ia_nilut_fivekrgb_second/iter_***.pth")
# model training and testing settings
train_cfg = {}
test_cfg = dict(metrics=['PSNR'], crop_border=0)

# dataset settings
train_dataset_type = 'FiveK'
val_dataset_type = 'FiveK'

train_pipeline = [
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='lq',
        backend='pillow',
        channel_order='rgb'),
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='gt',
        backend='pillow',
        channel_order='rgb'),
    dict(type='RandomRatioCrop', keys=['lq', 'gt'], crop_ratio=(0.6, 1.0)),
    dict(type='Flip', keys=['lq', 'gt'], flip_ratio=0.5, direction='horizontal'),
    dict(type='RandomColorJitter', keys=['lq'], brightness=0.2, saturation=0.2),
    dict(type='FlexibleRescaleToZeroOne', keys=['lq', 'gt'], precision=32),
    dict(type='ImageToTensor', keys=['lq', 'gt']),
    dict(type='Collect', keys=['lq', 'gt'], meta_keys=['lq_path', 'gt_path'])
]

test_pipeline = [
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='lq',
        backend='pillow',
        channel_order='rgb'),
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='gt',
        backend='pillow',
        channel_order='rgb'),
    dict(type='FlexibleRescaleToZeroOne', keys=['lq', 'gt'], precision=32),
    dict(type='ImageToTensor', keys=['lq', 'gt']),
    dict(
        type='Collect',
        keys=['lq', 'gt'],
        meta_keys=['lq_path', 'gt_path'])
]

data = dict(
    workers_per_gpu=4,
    train_dataloader=dict(samples_per_gpu=1),
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1, workers_per_gpu=1),

    # train
    train=dict(
        type=train_dataset_type,
        dir_lq='data/FiveK/input/JPG/480p',
        dir_gt='data/FiveK/expertC/JPG/480p',
        ann_file='data/FiveK/train.txt',
        pipeline=train_pipeline,
        test_mode=False,
        filetmpl_lq='{}.jpg',
        filetmpl_gt='{}.jpg'),
    # val
    val=dict(
        type=val_dataset_type,
        dir_lq='data/FiveK/input/JPG/480p',
        dir_gt='data/FiveK/expertC/JPG/480p',
        ann_file='data/FiveK/test.txt',
        pipeline=test_pipeline,
        test_mode=True,
        filetmpl_lq='{}.jpg',
        filetmpl_gt='{}.jpg'),
    # test
    test=dict(
        type=val_dataset_type,
        dir_lq='data/FiveK/input/JPG/480p',
        dir_gt='data/FiveK/expertC/JPG/480p',
        ann_file='data/FiveK/test.txt',
        pipeline=test_pipeline,
        test_mode=True,
        filetmpl_lq='{}.jpg',
        filetmpl_gt='{}.jpg'),
)

# optimizer
optimizers = dict(
    type='Adam',
    lr=1e-4,
    weight_decay=0,
    betas=(0.9, 0.999),
    eps=1e-8,
    paramwise_cfg=dict(custom_keys={'adaint': dict(lr_mult=0.1)}))
lr_config = None

# learning policy
total_iters = 4500*400

checkpoint_config = dict(interval=4500, save_optimizer=True, by_epoch=False)
evaluation = dict(interval=4500, save_image=False)
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
    ])
visual_config = None

# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = f'./work_dirs/{exp_name}'
load_from = None
resume_from = None
workflow = [('train', 1)]
find_unused_parameters = True
