# dataset settings
dataset_type = 'ICDAR15Dataset'
data_root = 'data/icdar2015/'
filter_difficulty = True

backend_args = None

train_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args=backend_args),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(type='mmdet.Resize', scale=(800, 800), keep_ratio=True),
    dict(
        type='mmdet.RandomFlip',
        prob=0.75,
        direction=['horizontal', 'vertical', 'diagonal']),
    dict(type='mmdet.PackDetInputs')
]
val_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args=backend_args),
    dict(type='mmdet.Resize', scale=(800, 800), keep_ratio=True),
    # avoid bboxes being resized
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
test_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args=backend_args),
    dict(type='mmdet.Resize', scale=(1100, 1100), keep_ratio=True),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=None,
    dataset=dict(
        type=dataset_type,
        filter_difficulty=filter_difficulty,
        data_root=data_root,
        ann_file='ic15_textdet_train_gt/',
        # Note: the '/' connot be at the begin of annfile! It is relative path, not absolute path!
        data_prefix=dict(img_path='ic15_textdet_train_img/'),
        # Note: the '/' connot be at the begin of img_path!  It is relative path, not absolute path!
        filter_cfg=dict(filter_empty_gt=True),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        filter_difficulty=filter_difficulty,
        data_root=data_root,
        ann_file='ic15_textdet_test_gt/',
        # Note: the '/' connot be at the begin of annfile! It is relative path, not absolute path!
        data_prefix=dict(img_path='ic15_textdet_test_img/'),
        # Note: the '/' connot be at the begin of img_path!  It is relative path, not absolute path!
        test_mode=True,
        pipeline=val_pipeline))
# test_dataloader = val_dataloader

val_evaluator = dict(type='ICDAR2015Metric', metric='mAP')
# test_evaluator = val_evaluator

# inference on test dataset and format the output results
# for submission. Note: the test set has no annotation.
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='ic15_textdet_test_img/'),
        test_mode=True,
        pipeline=test_pipeline))
test_evaluator = dict(
    type='ICDAR2015Metric',
    format_only=True,
    merge_patches=True,
    outfile_prefix='./work_dirs/icdar2015/textdet/')
