exp_name = 'pg_ia_nilut_fivekrgb_second'

custom_imports=dict(
    imports=['adaint'],
    allow_failed_imports=False)

# model settings
model = dict(
    type='PG_IA_NILUT_with_PARAMS',
    n_vertices=33,
    n_colors=3,
    pg_loss_factor=0.1,
    reverse_loss=True,
    recons_loss=dict(type='MSELoss', loss_weight=1.0, reduction='mean'),
    pretrained_path="work_dirs/pg_ia_nilut_fivekrgb_first/iter_45000.pth",
    pg_loss_freq=10,
    param_scale=10,
    classnames=[
        ['Overexposed photo.', 'Underexposed photo.'],
        ['Clear photo.', 'Unclear photo.'],
        ["Full color photo.", "No color photo."],
        ["Yellow tinted photo.", "Blue tinted photo."],
        ["Magenta tinted photo.", "Green tinted photo."]
    ],
    target_loss_weight = [10, 7, 4, 1, 1],
    )
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
        channel_order='rgb',
        use_cache=True),
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='gt',
        backend='pillow',
        channel_order='rgb',
        use_cache=True),
    dict(type='Resize2', keys=['lq', 'gt'], scale=(448, 448), keep_ratio=True),
    dict(type='RandomCrop', keys=['lq', 'gt'], crop_size=448),
    dict(type='FlexibleRescaleToZeroOne', keys=['lq', 'gt'], precision=32),
    dict(type='ImageToTensor', keys=['lq', 'gt']),
    dict(
        type='Collect',
        keys=['lq', 'gt'],
        meta_keys=['lq_path', 'gt_path'])
]

test_pipeline = [
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='lq',
        backend='pillow',
        channel_order='rgb',
        use_cache=True),
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='gt',
        backend='pillow',
        channel_order='rgb',
        use_cache=True),
    dict(type='FlexibleRescaleToZeroOne', keys=['lq', 'gt'], precision=32),
    dict(type='ImageToTensor', keys=['lq', 'gt']),
    dict(
        type='Collect',
        keys=['lq', 'gt'],
        meta_keys=['lq_path', 'gt_path'])
]

data = dict(
    workers_per_gpu=2,
    train_dataloader=dict(samples_per_gpu=2),
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
        ann_file='data/FiveK/train.txt',
        pipeline=test_pipeline,
        test_mode=True,
        filetmpl_lq='{}.jpg',
        filetmpl_gt='{}.jpg'),
    # test
    test=dict(
        type=val_dataset_type,
        dir_lq='data/FiveK/input/JPG/480p',
        dir_gt='data/FiveK/expertC/JPG/480p',
        ann_file='data/FiveK/train.txt',
        pipeline=test_pipeline,
        test_mode=True,
        filetmpl_lq='{}.jpg',
        filetmpl_gt='{}.jpg'),
)

# optimizer
optimizers = dict(
    type='SGD',
    lr=0.1,
    weight_decay=0,
    # betas=(0.9, 0.999),
    # eps=1e-8,
    paramwise_cfg=dict(custom_keys={'adaint': dict(lr_mult=0.1)}))
lr_config = None

# learning policy
total_iters = 4500*400//data["train_dataloader"]["samples_per_gpu"]

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
