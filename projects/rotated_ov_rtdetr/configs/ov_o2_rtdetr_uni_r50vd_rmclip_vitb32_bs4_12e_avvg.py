from torch.optim.adamw import AdamW
from mmengine.config import read_base
from mmengine.runner.loops import EpochBasedTrainLoop, TestLoop, ValLoop
from mmengine.optim.optimizer import OptimWrapper
from mmengine.optim.scheduler.lr_scheduler import LinearLR
from mmengine.hooks.ema_hook import EMAHook
from mmdet.models.data_preprocessors import DetDataPreprocessor
from mmdet.models.necks import ChannelMapper
from mmdet.models.losses import L1Loss
from mmdet.models.task_modules import FocalLossCost, HungarianAssigner
from mmdet.models.layers.ema import ExpMomentumEMA
from mmrotate.models.losses import GDLoss
from projects.rotated_dino.rotated_dino.match_cost import ChamferCost, GDCost
from projects.rotated_rtdetr.rotated_rtdetr import (RTDETRFPN, ResNetV1dPaddle, RTDETRVarifocalLoss)
from projects.rotated_ov_rtdetr.rotated_ov_rtdetr import (
    MultiModalYOLOBackbone, RotatedOVRTDETRHead, UniRotatedOVRTDETR,
    AVVGDetOneClassDataset, OBBRecallMetric)

with read_base():
    from configs._base_.default_runtime import *

load_from = 'work_dirs/ov_o2_rtdetr_r50vd_rmclip_vitb32_bs4_72e_avvg/epoch_72.pth'

angle_cfg = dict(
    width_longer=True,
    start_angle=0,
)
angle_factor=3.1415926535897932384626433832795

num_classes = 1
num_training_classes = 1

model = dict(
    type=UniRotatedOVRTDETR,
    num_queries=300,  # num_matching_queries, 900 for DINO
    with_box_refine=True,
    as_two_stage=True,
    text_channels=512,
    freeze_prompt=False,
    use_mlp_adapter=False,
    data_preprocessor=dict(
        type=DetDataPreprocessor,
        mean=[0, 0, 0],
        std=[255, 255, 255],
        bgr_to_rgb=True,
        pad_size_divisor=1,
        boxtype2tensor=False),
    backbone=dict(
        type=MultiModalYOLOBackbone,
        image_model=dict(
            type=ResNetV1dPaddle,  # ResNet for DINO
            depth=50,
            num_stages=4,
            out_indices=(1, 2, 3),
            frozen_stages=0,  # -1 for DINO
            norm_cfg=dict(type='BN', requires_grad=False),  # BN for DINO
            norm_eval=True,
            style='pytorch'),
        text_model=None,
        with_text_model=False),
    neck=dict(
        type=ChannelMapper,
        in_channels=[512, 1024, 2048],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='BN', requires_grad=True)),
    encoder=dict(
        use_encoder_idx=[-1],
        num_encoder_layers=1,
        in_channels=[256, 256, 256],
        fpn_cfg=dict(
            type=RTDETRFPN,
            in_channels=[256, 256, 256],
            out_channels=256,
            expansion=1.0,
            norm_cfg=dict(type='BN', requires_grad=True)),
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=1024,  # 2048 for DINO
                ffn_drop=0.0,
                act_cfg=dict(type='GELU')))),  # ReLU for DINO
    decoder=dict(
        num_layers=6,
        return_intermediate=True,
        angle_factor=angle_factor,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
            cross_attn_cfg=dict(
                embed_dims=256,
                num_levels=3,  # 4 for DINO
                dropout=0.0),
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=1024,  # 2048 for DINO
                ffn_drop=0.0)),
        post_norm_cfg=None),
    bbox_head=dict(
        type=RotatedOVRTDETRHead,
        num_classes=num_training_classes,
        angle_cfg=angle_cfg,
        angle_factor=angle_factor,
        sync_cls_avg_factor=True,
        loss_cls=dict(
            type=RTDETRVarifocalLoss,
            varifocal_loss_iou_type='hbox_iou',  # hbox_iou, rbox_iou, prob_iou
            use_sigmoid=True,
            alpha=0.75,
            gamma=2.0,
            iou_weighted=True,
            loss_weight=1.0),
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
        box_noise_scale=1.0,
        angle_cfg=angle_cfg,
        angle_factor=angle_factor,
        noise_mode='only_xyxy',  # 'only_xyxy', 'only_angle', 'only_xywh', 'all_xyxya'
        group_cfg=dict(dynamic=True,
                       num_groups=None,
                       num_dn_queries=100)),  # TODO: half num_dn_queries
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type=HungarianAssigner,
            match_costs=[
                dict(type=FocalLossCost, weight=2.0),
                dict(type=ChamferCost, weight=5.0, box_format='xywha'),
                dict(
                    type=GDCost,
                    loss_type='kld',
                    fun='log1p',
                    tau=1,
                    sqrt=False,
                    weight=2.0)])),
    test_cfg=dict(max_per_img=300))


backend_args = None
train_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args=backend_args),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(type='mmdet.Resize', scale=(1024, 576), keep_ratio=True),
    dict(
        type='mmdet.RandomFlip',
        prob=0.75,
        direction=['horizontal', 'vertical', 'diagonal']),
    dict(type='mmdet.PackDetInputs')
]

avvg_data_root = 'data/refGeo'

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=None,
    pin_memory=True,
    dataset=dict(
        type=AVVGDetOneClassDataset,
        data_root=avvg_data_root,
        ann_file='metainfo/avvg_detection_train.jsonl',
        data_prefix=dict(img_path='images/avvg'),
        pipeline=train_pipeline))

val_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args=backend_args),
    dict(type='mmdet.Resize', scale=(1024, 576), keep_ratio=True),
    # avoid bboxes being resized
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

val_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=AVVGDetOneClassDataset,
        data_root=avvg_data_root,
        ann_file='metainfo/avvg_detection_test.jsonl',
        data_prefix=dict(img_path='images/avvg'),
        test_mode=True,
        pipeline=val_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(type=OBBRecallMetric, metric='proposal_fast',
                     iou_thrs=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])
test_evaluator = val_evaluator

# optimizer
optim_wrapper = dict(
    type=OptimWrapper,
    optimizer=dict(type=AdamW, lr=0.0001, weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={'backbone': dict(lr_mult=0.1)},
        norm_decay_mult=0,
        bypass_duplicate=True))

# learning policy
max_epochs = 12
train_cfg = dict(
    type=EpochBasedTrainLoop, max_epochs=max_epochs, val_interval=2)
val_cfg = dict(type=ValLoop)
test_cfg = dict(type=TestLoop)
param_scheduler = [
    dict(
        type=LinearLR, start_factor=0.001, by_epoch=False, begin=0, end=50)
]
custom_hooks = [
    dict(
        type=EMAHook,
        ema_type=ExpMomentumEMA,
        momentum=0.0001,
        update_buffers=True,
        priority=49)
]

find_unused_parameters=True

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (2 GPUs) x (4 samples per GPU)
auto_scale_lr = dict(enable=False, base_batch_size=4)