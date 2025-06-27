_base_ = [
    '../../../configs/_base_/datasets/dota.py',
    '../../../configs/_base_/schedules/schedule_2x.py',
    '../../../configs/_base_/default_runtime.py'
]

custom_imports = dict(
    imports=['projects.rotated_DiffusionDet.rotateddiffusiondet'], allow_failed_imports=False)

num_proposals = 900
classes = 15
# model settings
model = dict(
    type='DiffusionDet',
    data_preprocessor=dict(
        type='mmdet.DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    backbone=dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='mmdet.FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    bbox_head=dict(
        type='AngleBranchDyDiffDetHead',
        num_classes=classes,
        feat_channels=256,
        num_proposals=num_proposals,
        num_heads=6,
        deep_supervision=True,
        prior_prob=0.01,
        snr_scale=2.0,
        sampling_timesteps=1,
        ddim_sampling_eta=1.0,
        single_head=dict(
            type='AngleBranchSingleDiffDetHead',
            num_cls_convs=1,
            num_reg_convs=3,
            dim_feedforward=2048,
            num_heads=8,
            dropout=0.0,
            act_cfg=dict(type='ReLU', inplace=True),
            dynamic_conv=dict(dynamic_dim=64, dynamic_num=2)),
        angle_coder=dict(
            type='CSLCoder',
            angle_version='le90',
            omega=1,
            window='gaussian',
            radius=6),
        roi_extractor=dict(
            type='mmdet.SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=2),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32, 64]),
        # criterion
        criterion=dict(
            type='AngleBranchDiffusionDetCriterion',
            num_classes=classes,
            assigner=dict(
                type='AngleBranchDiffusionDetMatcher',
                match_costs=[
                    dict(
                        type='mmdet.FocalLossCost',
                        alpha=0.25,
                        gamma=2.0,
                        weight=2.0,
                        eps=1e-8),
                    dict(type='mmdet.BBoxL1Cost', weight=2.0, box_format='xyxy'),
                    dict(type='mmdet.IoUCost', iou_mode='giou', weight=5.0),
                    dict(type='mmdet.CrossEntropyLossCost', weight=2.0)
                ],
                center_radius=2.5,
                candidate_topk=5),
            loss_cls=dict(
                type='mmdet.FocalLoss',
                use_sigmoid=True,
                alpha=0.25,
                gamma=2.0,
                reduction='sum',
                loss_weight=2.0),
            loss_bbox=dict(type='mmdet.L1Loss', reduction='sum', loss_weight=2.0),
            loss_giou=dict(type='mmdet.GIoULoss', reduction='sum',
                           loss_weight=5.0),
            loss_angle=dict(
                type='mmdet.CrossEntropyLoss', use_sigmoid=True, reduction='sum',
                loss_weight=2.0)
        )),
    test_cfg=dict(
        use_nms=True,
        score_thr=0.5,
        min_bbox_size=0,
        nms=dict(type='nms_rotated', iou_threshold=0.6),
    ))

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        _delete_=True, type='AdamW', lr=2.5e-5, weight_decay=1e-4),
    clip_grad=dict(max_norm=1.0, norm_type=2))

train_cfg=dict(val_interval=6)
default_hooks = dict(checkpoint=dict(interval=1))

# base_batch_size = (2 GPUs) x (2 samples per GPU)
# auto_scale_lr = dict(base_batch_size=2 * 2, enable=True)