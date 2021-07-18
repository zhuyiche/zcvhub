_base_ = [
    '../_base_/models/resnet50.py', '../_base_/datasets/imagenet_bs64.py',
    '../_base_/schedules/imagenet_bs256.py'
]

# checkpoint saving
checkpoint_config = dict(interval=10)
# yapf:disable
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = '/cv_data/zhuyichen/zcvhub/work_dirs/cls/resnet50_b32x8_imagenet/epoch_20.pth'
workflow = [('train', 1), ('val', 1)]