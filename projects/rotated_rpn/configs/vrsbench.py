vrsbench_data_root = 'data/VRSBench'

backend_args = None
train_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args=backend_args),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(type='mmdet.Resize', scale=(512, 512), keep_ratio=True),
    dict(
        type='mmdet.RandomFlip',
        prob=0.75,
        direction=['horizontal', 'vertical', 'diagonal']),
    dict(type='mmdet.PackDetInputs')]


train_dataloader = dict(
    batch_size=8,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=None,
    pin_memory=True,
    dataset=dict(
        type='VRSBenchDetDataset',
        data_root=vrsbench_data_root,
        ann_file='Annotations_train',
        data_prefix=dict(img_path='Images_train'),
        pipeline=train_pipeline))

val_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args=backend_args),
    dict(type='mmdet.Resize', scale=(512, 512), keep_ratio=True),
    # avoid bboxes being resized
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                    'scale_factor'))
]

val_dataloader = dict(
    batch_size=8,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='VRSBenchDetDataset',
        data_root=vrsbench_data_root,
        ann_file='Annotations_val',
        data_prefix=dict(img_path='Images_val'),
        test_mode=True,
        pipeline=val_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(type='OBBRecallMetric', metric='proposal_fast',
                     iou_thrs=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])
test_evaluator = val_evaluator