exp_name = 'pg_ia_nilut_ppr10k_a_first'

target = 'a' # change this line (a/b/c) to use other groundtruths

custom_imports=dict(
    imports=['adaint'],
    allow_failed_imports=False)

# model settings
model = dict(
    type='PG_IA_NILUT_without_PARAMS',
    n_vertices=33,
    n_colors=3,
    pg_loss_factor=0.1,
    recons_loss=dict(type='MSELoss', loss_weight=1.0, reduction='mean'),
    classnames=[
        ['Overexposed photo.', 'Underexposed photo.'],
        ['High contrast photo.', 'Low contrast photo.'],
        ["Full color photo.", "No color photo."],
        ["Yellow tinted photo.", "Blue tinted photo."],
        ["Magenta tinted photo.", "Green tinted photo."]
    ],
    target_loss_weight = [10, 3, 4, 1, 1],
    dataset="ppr10K",
    )
# model training and testing settings
train_cfg = dict()
test_cfg = dict(metrics=['PSNR'], crop_border=0)

# dataset settings
train_dataset_type = 'PPR10K'
val_dataset_type = 'PPR10K'

train_pipeline = [
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='lq',
        backend='cv2',
        flag='unchanged'),
    dict(type='FlipChannels', keys=['lq']), # BGR->RGB
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='gt',
        backend='pillow',
        channel_order='rgb'),
    dict(type='ResizePad', keys=['lq', 'gt'], scale=(448, 448), backend='cv2', interpolation='bilinear'),
    dict(type='FlexibleRescaleToZeroOne', keys=['lq', 'gt'], precision=32),
    dict(type='ImageToTensor', keys=['lq', 'gt']),
    dict(type='Collect', keys=['lq', 'gt'], meta_keys=['lq_path', 'gt_path'])
]

test_pipeline = [
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='lq',
        backend='cv2',
        flag='unchanged'),
    dict(type='FlipChannels', keys=['lq']), # BGR->RGB
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
    workers_per_gpu=2,
    train_dataloader=dict(samples_per_gpu=1),
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1, workers_per_gpu=1),

    # train
    train=dict(
        type=train_dataset_type,
        dir_lq='data/PPR10K/train_val_images_tif_360p/source',
        dir_gt=f'data/PPR10K/train_val_images_tif_360p/target_{target}',
        ann_file='data/PPR10K/train.txt',
        pipeline=train_pipeline,
        test_mode=False,
        filetmpl_lq='{}.tif',
        filetmpl_gt='{}.tif'),
    # val
    val=dict(
        type=val_dataset_type,
        dir_lq='data/PPR10K/train_val_images_tif_360p/source',
        dir_gt=f'data/PPR10K/train_val_images_tif_360p/target_{target}',
        ann_file='data/PPR10K/train.txt',
        pipeline=test_pipeline,
        test_mode=True,
        filetmpl_lq='{}.tif',
        filetmpl_gt='{}.tif'),
    # test
    test=dict(
        type=val_dataset_type,
        dir_lq='data/PPR10K/train_val_images_tif_360p/source',
        dir_gt=f'data/PPR10K/train_val_images_tif_360p/target_{target}',
        ann_file='data/PPR10K/train.txt',
        pipeline=test_pipeline,
        test_mode=True,
        filetmpl_lq='{}.tif',
        filetmpl_gt='{}.tif'),
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
total_iters = 8875 * 10 //data["train_dataloader"]["samples_per_gpu"]

checkpoint_config = dict(interval=total_iters, save_optimizer=True, by_epoch=False)
evaluation = dict(interval=10**10, save_image=False)
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
