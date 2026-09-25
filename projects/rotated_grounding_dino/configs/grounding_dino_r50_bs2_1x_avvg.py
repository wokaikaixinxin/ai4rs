from torch.optim.adamw import AdamW
from mmengine.config import read_base
from mmengine.optim.optimizer import OptimWrapper
from mmengine.optim.scheduler import MultiStepLR
from mmdet.models.data_preprocessors import DetDataPreprocessor
from mmdet.models import BertModel, ResNet, ChannelMapper
from mmdet.models.losses import FocalLoss, L1Loss
from mmdet.models.task_modules import HungarianAssigner, BinaryFocalLossCost
from mmrotate.models.losses import GDLoss
from projects.rotated_dino.rotated_dino.match_cost import RBoxL1Cost, GDCost
from projects.rotated_grounding_dino.rotated_grounding_dino import (RotatedGroundingDINO,
                                                                   RotatedGroundingDINOHead,
                                                                   VGLocalVisualizer)

with read_base():
    from .avvg import *
    from configs._base_.schedules.schedule_1x import *
    from configs._base_.default_runtime import *

angle_cfg = dict(
    width_longer=True,
    start_angle=0,
)
angle_factor = 3.1415926535897932384626433832795

model = dict(
    type=RotatedGroundingDINO,
    num_queries=900,
    with_box_refine=True,
    as_two_stage=True,
    data_preprocessor=dict(
        type=DetDataPreprocessor,
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_mask=False,
        boxtype2tensor=False
    ),
    language_model=dict(
        type=BertModel,
        name=lang_model_name,
        pad_to_max=False,
        use_sub_sentence_represent=True,
        special_tokens_list=['[CLS]', '[SEP]', '.', '?'],
        add_pooling_layer=False,
    ),
    backbone=dict(
        type=ResNet,
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type=ChannelMapper,
        in_channels=[512, 1024, 2048],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        bias=True,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=4),
    encoder=dict(
        num_layers=6,
        num_cp=6,
        # visual layer config
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_levels=4, dropout=0.0),
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=2048, ffn_drop=0.0)),
        # text layer config
        text_layer_cfg=dict(
            self_attn_cfg=dict(num_heads=4, embed_dims=256, dropout=0.0),
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=1024, ffn_drop=0.0)),
        # fusion layer config
        fusion_layer_cfg=dict(
            v_dim=256,
            l_dim=256,
            embed_dim=1024,
            num_heads=4,
            init_values=1e-4),
    ),
    decoder=dict(
        num_layers=6,
        return_intermediate=True,
        angle_factor=angle_factor,
        layer_cfg=dict(
            # query self attention layer
            self_attn_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
            # cross attention layer query to text
            cross_attn_text_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
            # cross attention layer query to image
            cross_attn_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=2048, ffn_drop=0.0)),
        post_norm_cfg=None),
    positional_encoding=dict(
        num_feats=128, normalize=True, offset=0.0, temperature=20),
    bbox_head=dict(
        type=RotatedGroundingDINOHead,
        num_classes=256, # NOTE: 256 because this is the (max) number of tokens in the text.
        angle_cfg=angle_cfg,
        angle_factor=angle_factor,
        sync_cls_avg_factor=True,
        contrastive_cfg=dict(max_text_len=256, log_scale='auto', bias=True),
        loss_cls=dict(
            type=FocalLoss,
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),  # 2.0 in DeformDETR
        loss_bbox=dict(type=L1Loss, loss_weight=5.0),
        loss_iou=dict(
            type=GDLoss,
            loss_type='kld',
            fun='log1p',
            tau=1,
            sqrt=False,
            loss_weight=2.0)),
    dn_cfg=dict(  # TODO: Move to model.train_cfg ?
        label_noise_scale=0.5,
        box_noise_scale=1.0,  # 0.4 for DN-DETR
        angle_cfg=angle_cfg,
        angle_factor=angle_factor,
        noise_mode='only_xyxy',  # 'only_xyxy', 'only_angle', 'only_xywh', 'all_xyxya'
        group_cfg=dict(dynamic=True, num_groups=None,
                       num_dn_queries=100)),  # TODO: half num_dn_queries
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type=HungarianAssigner,
            match_costs=[
                dict(type=BinaryFocalLossCost, weight=2.0),
                dict(type=RBoxL1Cost, weight=5.0, box_format='xywha', angle_factor=angle_factor),
                dict(type=GDCost, loss_type='kld', fun='log1p', tau=1, sqrt=False, weight=2.0)
            ])),
    test_cfg=dict(max_per_img=300))


optim_wrapper = dict(
    type=OptimWrapper,
    optimizer=dict(
        type=AdamW,
        lr=0.0001,  # 0.0002 for DeformDETR
        weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(custom_keys={
        'absolute_pos_embed': dict(decay_mult=0.),
        'backbone': dict(lr_mult=0.1)
    }))

param_scheduler = [
    dict(
        type=MultiStepLR,
        begin=0,
        end=train_cfg['max_epochs'],
        by_epoch=True,
        milestones=[11],
        gamma=0.1)
]

visualizer.update(type=VGLocalVisualizer)

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (4 GPUs) x (2 samples per GPU)
auto_scale_lr = dict(base_batch_size=2)
