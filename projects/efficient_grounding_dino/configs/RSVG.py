# dataset settings
dataset_type = 'projects.efficient_grounding_dino.efficient_grounding_dino.DIORRSVGDataset'
data_root = 'data/dior_rsvg'

backend_args = None


train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, with_label=True),
    dict(type='mmdet.Resize', scale=(640, 640), keep_ratio=True),
    dict(type='mmdet.RandomFlip', prob=0.),
    dict(type='mmdet.PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction', 'text',
                   'custom_entities'))]


test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='mmdet.Resize', scale=(640, 640), keep_ratio=True),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, with_label=True),
    dict(type='mmdet.PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                    'scale_factor', 'text', 'gt_bboxes'))]


train_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='mmdet.AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='Annotations',
        split_file=['train.txt', 'val.txt'],
        data_prefix=dict(img_path='JPEGImages/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        ))

val_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='JPEGImages/'),
        ann_file='Annotations',
        split_file='test.txt',
        pipeline=test_pipeline))

test_dataloader = val_dataloader

val_evaluator = [dict(type='projects.efficient_grounding_dino.efficient_grounding_dino.RSVGMetric_top1',
                      metric=['Pr@0.5', 'Pr@0.6', 'Pr@0.7', 'Pr@0.8', 'Pr@0.9', 'meanIoU', 'cumIoU'])]
test_evaluator = val_evaluator
