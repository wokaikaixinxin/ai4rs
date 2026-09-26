_base_ = [
    'oriented_rpn_r50_fpn.py',
    'vrsbench.py',
    '../../../configs/_base_/schedules/schedule_1x.py',
    '../../../configs/_base_/default_runtime.py'
]


custom_imports = dict(
    imports=['projects.rotated_rpn.rpn'], allow_failed_imports=False)

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001))

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (2 GPUs) x (8 samples per GPU)
auto_scale_lr = dict(enable=False, base_batch_size=16)