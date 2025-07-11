_base_ = [
    '../../../../configs/_base_/datasets/sardet_100k.py',
    '../../../../configs/_base_/schedules/schedule_1x.py',
    '../../../../configs/_base_/default_runtime.py',
]
custom_imports = dict(
    imports=['projects.SARDet_100K.sardet_100k'], allow_failed_imports=False)

num_classes = 6
num_stages = 6
num_proposals = 100
model = dict(
    type='mmdet.SparseRCNN',
    init_cfg=dict(type='Pretrained',
                  checkpoint='https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/'
                             'resolve/master/MSFA/pretrain_sparse-rcnn_r50_sar_wavelet/'
                             'best_coco_bbox_mAP_epoch_12.pth'),
    data_preprocessor=dict(
        type='mmdet.DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    backbone=dict(
        type='MSFA',
        use_sar=True,
        use_wavelet=True,
        backbone=dict(
            type='mmdet.ResNet',
            depth=50,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            frozen_stages=1,
            norm_cfg=dict(type='BN', requires_grad=True),
            norm_eval=True,
            style='pytorch',
            init_cfg=None
        ),
    ),
    neck=dict(
        type='mmdet.FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=0,
        add_extra_convs='on_input',
        num_outs=4),
    rpn_head=dict(
        type='mmdet.EmbeddingRPNHead',
        num_proposals=num_proposals,
        proposal_feature_channel=256),
    roi_head=dict(
        type='mmdet.SparseRoIHead',
        num_stages=num_stages,
        stage_loss_weights=[1] * num_stages,
        proposal_feature_channel=256,
        bbox_roi_extractor=dict(
            type='mmdet.SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=2),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=[
            dict(
                type='mmdet.DIIHead',
                reg_predictor_cfg=dict(type='mmdet.Linear'),
                cls_predictor_cfg=dict(type='mmdet.Linear'),
                num_classes=num_classes,
                num_ffn_fcs=2,
                num_heads=8,
                num_cls_fcs=1,
                num_reg_fcs=3,
                feedforward_channels=2048,
                in_channels=256,
                dropout=0.0,
                ffn_act_cfg=dict(type='ReLU', inplace=True),
                dynamic_conv_cfg=dict(
                    type='mmdet.DynamicConv',
                    in_channels=256,
                    feat_channels=64,
                    out_channels=256,
                    input_feat_shape=7,
                    act_cfg=dict(type='ReLU', inplace=True),
                    norm_cfg=dict(type='LN')),
                loss_bbox=dict(type='mmdet.L1Loss', loss_weight=5.0),
                loss_iou=dict(type='mmdet.GIoULoss', loss_weight=2.0),
                loss_cls=dict(
                    type='mmdet.FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=2.0),
                bbox_coder=dict(
                    type='mmdet.DeltaXYWHBBoxCoder',
                    clip_border=False,
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.5, 0.5, 1., 1.])) for _ in range(num_stages)
        ]),
    # training and testing settings
    train_cfg=dict(
        rpn=None,
        rcnn=[
            dict(
                assigner=dict(
                    type='mmdet.HungarianAssigner',
                    match_costs=[
                        dict(type='mmdet.FocalLossCost', weight=2.0),
                        dict(type='mmdet.BBoxL1Cost', weight=5.0, box_format='xyxy'),
                        dict(type='mmdet.IoUCost', iou_mode='giou', weight=2.0)
                    ]),
                sampler=dict(type='mmdet.PseudoSampler'),
                pos_weight=1) for _ in range(num_stages)
        ]),
    test_cfg=dict(rpn=None, rcnn=dict(max_per_img=num_proposals)))

# optimizer
optim_wrapper = dict(
    optimizer=dict(
        _delete_=True, type='AdamW', lr=0.000025, weight_decay=0.0001),
    clip_grad=dict(max_norm=1, norm_type=2))

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (2 samples per GPU)
auto_scale_lr = dict(base_batch_size=16)