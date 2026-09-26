# dataset settings
data_root = 'data/VRSBench'
lang_model_name = '/root/bert-base-uncased'

dataset_type = 'projects.rotated_grounding_dino.rotated_grounding_dino.VRSBenchVGDataset'
dataset_type_2 = 'projects.rotated_grounding_dino.rotated_grounding_dino.VRSBenchVGValDataset'

backend_args = None

# Note: Don't use RandomFlip in train_pipeline !!! Because there are directions in referring expression.
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(type='mmdet.Resize', scale=(512, 512), keep_ratio=True),
    dict(
        type='projects.rotated_grounding_dino.rotated_grounding_dino.RandomSamplingNegPos',
        tokenizer_name=lang_model_name,
        num_sample_negative=85,
        max_tokens=256),
    dict(type='mmdet.PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction', 'text',
                   'custom_entities', 'tokens_positive', 'dataset_mode'))]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='mmdet.Resize', scale=(512, 512), keep_ratio=True),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, with_label=False, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(type='mmdet.PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                    'scale_factor', 'text', 'custom_entities',
                   'tokens_positive'))]

train_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='mmdet.AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='Annotations_train',
        data_prefix=dict(img_path='Images_train'),
        pipeline=train_pipeline,
        ))

val_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type_2,
        data_root=data_root,
        data_prefix=dict(img_path='Images_val'),
        ann_file='VRSBench_EVAL_referring.json',
        pipeline=test_pipeline))

test_dataloader = val_dataloader

val_evaluator = [dict(type='projects.rotated_grounding_dino.rotated_grounding_dino.RSVGMetric_top1',
                      metric=['Pr@0.5', 'Pr@0.6', 'Pr@0.7', 'Pr@0.8', 'Pr@0.9', 'meanIoU', 'cumIoU'])]

test_evaluator = val_evaluator