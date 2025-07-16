_base_ = 'point2rbox_v2-1x-dota.py'

# This allows the model to use gt points during inference
model = dict(bbox_head=dict(pseudo_generator=True))

test_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(type='ConvertWeakSupervision', point_proportion=1., hbox_proportion=0),
    dict(type='mmdet.Resize', scale=(1024, 1024), keep_ratio=True),
    dict(type='mmdet.PackDetInputs')
]

# Use train set for inference and turn off the augmentation
test_dataloader = _base_.train_dataloader
test_dataloader['_delete_'] = True
test_dataloader['dataset']['pipeline'] = test_pipeline
test_evaluator = dict(_delete_=True,
                    type='DOTAMetric',
                    metric='mAP',
                    format_only=True,
                    outfile_prefix='data/split_ss_dota/point2rbox_v2_pseudo_labels')