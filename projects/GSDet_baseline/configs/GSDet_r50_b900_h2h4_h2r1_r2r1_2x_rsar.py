_base_ = [
    '../../../configs/_base_/datasets/rsar.py',
    '../../../configs/_base_/schedules/schedule_2x.py',
    '../../../configs/_base_/default_runtime.py'
]

custom_imports = dict(
    imports=['projects.GSDet_baseline.gsdet'], allow_failed_imports=False)

pretrain = 'https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/GSDet_baseline/' \
            'pretrain/diffusiondet_r50_fpn_500-proposals_1-step_crop-ms-480-800-450k_coco_new2.pth'

num_classes = 6
num_proposals = 900
hbox2hbox = 4
hbox2rbox = 1
rbox2rbox = 1
angle_version = 'le90'
model = dict(
    type='GSDet',
    init_cfg=dict(type='Pretrained', checkpoint=pretrain),
    data_preprocessor=dict(
        type='mmdet.DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32,
        boxtype2tensor=False),
    backbone=dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    neck=dict(
        type='mmdet.FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='Hbox2HboxLayer',
        num_classes=num_classes,
        feat_channels=256,
        num_proposals=num_proposals,
        num_heads=hbox2hbox,
        deep_supervision=True,
        prior_prob=0.01,
        single_head=dict(
            type='SingleHbox2HboxHead',
            num_cls_convs=1,
            num_reg_convs=3,
            dim_feedforward=2048,
            num_heads=8,
            dropout=0.0,
            act_cfg=dict(type='ReLU', inplace=True),
            dynamic_conv=dict(dynamic_dim=64, dynamic_num=2)),
        roi_extractor=dict(
            type='mmdet.SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=2),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32, 64]),
        # criterion
        criterion=dict(
            type='Hbox2HboxLayerCriterion',
            num_classes=num_classes,
            assigner=dict(
                type='Hbox2HboxLayerMatcher',
                match_costs=[
                    dict(
                        type='mmdet.FocalLossCost',
                        alpha=0.25,
                        gamma=2.0,
                        weight=2.0,
                        eps=1e-8),
                    dict(type='mmdet.BBoxL1Cost', weight=5.0, box_format='xyxy'),
                    dict(type='mmdet.IoUCost', iou_mode='giou', weight=2.0)
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
            loss_bbox=dict(type='mmdet.L1Loss', reduction='sum', loss_weight=5.0),
            loss_giou=dict(type='mmdet.GIoULoss', reduction='sum',
                           loss_weight=2.0))
    ),
    roi_head=dict(
        type='Decoder',
        num_stages=hbox2rbox+rbox2rbox,
        num_proposals = num_proposals,
        angle_version = angle_version,
        stage_loss_weights=[1] * (hbox2rbox+rbox2rbox),
        bbox_roi_extractor=
            [
            dict(
                type='mmdet.SingleRoIExtractor',
                roi_layer=dict(
                    type='RoIAlign', output_size=7, sampling_ratio=0),
                out_channels=256,
                featmap_strides=[4, 8, 16, 32])] * hbox2rbox + \
            [
            dict(
                type='RotatedSingleRoIExtractor',
                roi_layer=dict(
                    type='RoIAlignRotated',
                    out_size=7,
                    sample_num=2,
                    clockwise=True),
                out_channels=256,
                featmap_strides=[4, 8, 16, 32])
            ] * rbox2rbox,
        bbox_head=
            [
            dict(
                type='Hbox2RboxLayer',
                num_classes=num_classes,
                angle_version=angle_version,
                feat_channels=256,
                dim_feedforward=2048,
                num_cls_convs=1,
                num_reg_convs=3,
                num_heads=8,
                dropout=0.0,
                pooler_resolution=7,
                act_cfg=dict(type='ReLU', inplace=True),
                dynamic_conv=dict(dynamic_dim=64, dynamic_num=2),
                reg_predictor_cfg=dict(type='mmdet.Linear'),
                cls_predictor_cfg=dict(type='mmdet.Linear'),
                loss_bbox=dict(type='mmdet.L1Loss', loss_weight=2.0),
                loss_iou=dict(type='RotatedIoULoss', mode='linear', loss_weight=5.0),
                loss_cls=dict(
                    type='mmdet.FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=2.0),
                bbox_coder=dict(
                    type='MidpointOffsetCoderv2',
                    angle_version=angle_version,
                    target_means=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    # note : set target_stds to a vary small value !!!!!
                    target_stds=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                    use_box_type=False),
            )] * hbox2rbox + \
            [
            dict(
                type='Rbox2RboxLayer',
                num_classes=num_classes,
                angle_version=angle_version,
                feat_channels=256,
                dim_feedforward=2048,
                num_cls_convs=1,
                num_reg_convs=3,
                num_heads=8,
                dropout=0.0,
                pooler_resolution=7,
                act_cfg=dict(type='ReLU', inplace=True),
                dynamic_conv=dict(dynamic_dim=64, dynamic_num=2),
                reg_predictor_cfg=dict(type='mmdet.Linear'),
                cls_predictor_cfg=dict(type='mmdet.Linear'),
                loss_bbox=dict(type='mmdet.L1Loss', loss_weight=2.0),
                loss_iou=dict(type='RotatedIoULoss', mode='linear', loss_weight=5.0),
                loss_cls=dict(
                    type='mmdet.FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=2.0),
                bbox_coder=dict(
                    type='DeltaXYWHTRBBoxCoder',
                    angle_version=angle_version,
                    norm_factor=None,
                    edge_swap=True,
                    proj_xy=True,
                    target_means=(.0, .0, .0, .0, .0),
                    target_stds=(1., 1., 1., 1., 1.),
                    use_box_type=False)
              )] * rbox2rbox
        ),
    # model training and testing settings
    train_cfg=dict(
        rpn = None,
        rcnn=
        [
            dict(
                assigner=dict(
                type='TopkHungarianAssigner',
                topk=2,
                cls_cost=dict(type='mmdet.FocalLossCost', weight=2.0),
                reg_cost=dict(type='RBBoxL1Cost', weight=2.0,
                            box_format='xywht', angle_version=angle_version),
                iou_cost=dict(type='RotatedIoUCost', iou_mode='iou', weight=5.0)
                ),
                sampler=dict(type='mmdet.PseudoSampler'),
                pos_weight=1)
        ] * hbox2rbox + \
        [
            dict(
                assigner=dict(
                    type='TopkHungarianAssigner',
                    topk=2,
                    cls_cost=dict(type='mmdet.FocalLossCost', weight=2.0),
                    reg_cost=dict(type='RBBoxL1Cost', weight=2.0,
                                  box_format='xywht', angle_version=angle_version),
                    iou_cost=dict(type='RotatedIoUCost', iou_mode='iou', weight=5.0)
                ),
                sampler=dict(type='mmdet.PseudoSampler'),
                pos_weight=1)
        ] * rbox2rbox
    ),

    test_cfg=dict(
        rpn=None,
        rcnn=dict(
            use_nms=True,
            nms=dict(type='nms_rotated', iou_threshold=0.6),
            max_per_img=num_proposals)))

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        _delete_=True, type='AdamW', lr=2.5e-5 / 4, weight_decay=1e-4),
    clip_grad=dict(max_norm=1.0, norm_type=2))

train_cfg=dict(val_interval=2)
default_hooks = dict(checkpoint=dict(interval=1))

# base_batch_size = (2 GPUs) x (2 samples per GPU)
auto_scale_lr = dict(base_batch_size=4)