
_base_ = [
    'ai4rs::_base_/datasets/dota.py',
    'ai4rs::_base_/schedules/schedule_1x.py',
    'ai4rs::_base_/default_runtime.py'
]

custom_imports = dict(
    imports=['projects.OrientedFormer.orientedformer'], allow_failed_imports=False)

pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'  # noqa

num_stages = 2
num_proposals = 300
num_classes = 15
angle_version = 'le90'
depths = [2, 2, 6, 2]

model = dict(
    type='OrientedDDQRCNN',
    data_preprocessor=dict(
        type='mmdet.DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32,
        boxtype2tensor=False),
    backbone=dict(
        type='mmdet.SwinTransformer',
        embed_dims=96,
        depths=depths,
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(
        type='ChannelMapperWithGN',
        kernel_size=1,
        in_channels=[96, 192, 384, 768],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='OrientedAdaMixerDDQ',
        angle_version=angle_version,
        ddq_num_classes=num_classes,
        num_proposals=num_proposals,
        in_channels=256,
        feat_channels=256,
        strides=[4, 8, 16, 32, 64],
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
        dqs_cfg=dict(type='nms_rotated', iou_threshold=0.7, nms_pre=1000),
        offset=0.5,
        aux_loss=dict(
            loss_cls=dict(
                type='mmdet.QualityFocalLoss',
                use_sigmoid=True,
                activated=True,  # use probability instead of logit as input
                beta=2.0,
                loss_weight=1.0),
            loss_bbox=dict(type='RotatedIoULoss', mode='linear', loss_weight=5.0),
            train_cfg=dict(
                assigner=dict(
                    type='TopkHungarianAssigner',
                    topk=8,
                    iou_calculator = dict(type='RBboxOverlaps2D'),
                    cls_cost=dict(type='mmdet.FocalLossCost', weight=2.0),
                    reg_cost=dict(type='RBBoxL1Cost', weight=2.0,
                             box_format='xywht', angle_version=angle_version),
                    iou_cost=dict(type='RotatedIoUCost', iou_mode='iou', weight=5.0)
                ),
                alpha=1,
                beta=6)),
        main_loss=dict(
            loss_cls=dict(
                type='mmdet.QualityFocalLoss',
                use_sigmoid=True,
                activated=True,  # use probability instead of logit as input
                beta=2.0,
                loss_weight=1.0),
            loss_bbox=dict(type='RotatedIoULoss', mode='linear', loss_weight=5.0),
            train_cfg=dict(
                assigner=dict(
                    type='TopkHungarianAssigner',
                    topk=8,
                    iou_calculator=dict(type='RBboxOverlaps2D'),
                    cls_cost=dict(type='mmdet.FocalLossCost', weight=2.0),
                    reg_cost=dict(type='RBBoxL1Cost', weight=2.0,
                             box_format='xywht', angle_version=angle_version),
                    iou_cost=dict(type='RotatedIoUCost', iou_mode='iou', weight=5.0)
                ),
                alpha=1,
                beta=6))),
    roi_head=dict(
        type='OrientedAdaMixerDecoder',
        num_stages=num_stages,
        stage_loss_weights=[1] * num_stages,
        content_dim=256,
        featmap_strides=[4, 8, 16, 32, 64],
        bbox_head=[
            dict(
                type='OrientedFormerDecoderLayer',
                num_classes=num_classes,
                angle_version=angle_version,
                reg_predictor_cfg=dict(type='mmdet.Linear'),
                cls_predictor_cfg=dict(type='mmdet.Linear'),
                num_cls_fcs=1,
                num_reg_fcs=1,
                content_dim=256,
                target_means=(.0, .0, .0, .0, .0),
                target_stds=(1., 1., 1., 1., 1.),
                self_attn_cfg=dict(
                     embed_dims=256,
                     num_heads=8,
                     dropout=0.0),
                o3d_attn_cfg=dict(
                    type='OrientedAttention',
                    n_points=32,
                    n_heads=64,
                    embed_dims=256,
                    reduction=4
                ),
                ffn_cfg=dict(
                    embed_dims=256,
                    feedforward_channels=2048,
                    num_fcs=2,
                    ffn_drop=0.0,
                    act_cfg=dict(type='ReLU', inplace=True)),
                loss_bbox=dict(type='mmdet.L1Loss', loss_weight=2.0),
                loss_iou=dict(type='RotatedIoULoss', mode='linear', loss_weight=5.0),
                loss_cls=dict(
                    type='mmdet.FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=2.0),
                # NOTE: The following argument is a placeholder to hack the code. No real effects for decoding or updating bounding boxes.
                bbox_coder=dict(
                    type='DeltaXYWHTRBBoxCoder')) for stage_idx in range(num_stages)
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
                        dict(type='RBBoxL1Cost', weight=2.0,
                             box_format='xywht', angle_version=angle_version),
                        dict(type='RotatedIoUCost', iou_mode='iou', weight=5.0)
                    ]),
                sampler=dict(type='mmdet.PseudoSampler'),
                pos_weight=1) for _ in range(num_stages)
        ]),
    test_cfg=dict(rpn=None, rcnn=dict(max_per_img=num_proposals)))

# set all layers in backbone to lr_mult=0.1
# set all norm layers, position_embeding,
# query_embeding, level_embeding to decay_multi=0.0
backbone_norm_multi = dict(lr_mult=0.1, decay_mult=0.0)
backbone_embed_multi = dict(lr_mult=0.1, decay_mult=0.0)
embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
custom_keys = {
    'backbone': dict(lr_mult=0.1, decay_mult=1.0),
    'backbone.patch_embed.norm': backbone_norm_multi,
    'backbone.norm': backbone_norm_multi,
    'absolute_pos_embed': backbone_embed_multi,
    'relative_position_bias_table': backbone_embed_multi,
    'query_embed': embed_multi,
    'query_feat': embed_multi,
    'level_embed': embed_multi
}
custom_keys.update({
    f'backbone.stages.{stage_id}.blocks.{block_id}.norm': backbone_norm_multi
    for stage_id, num_blocks in enumerate(depths)
    for block_id in range(num_blocks)
})
custom_keys.update({
    f'backbone.stages.{stage_id}.downsample.norm': backbone_norm_multi
    for stage_id in range(len(depths) - 1)
})

# optimizer
optim_wrapper = dict(
    optimizer=dict(
        _delete_=True, type='AdamW', lr=5e-5, weight_decay=1e-6),  # 2 RTX 2080ti
    clip_grad=dict(max_norm=1, norm_type=2))

train_cfg=dict(val_interval=6)
