# Copyright (c) ai4rs. All rights reserved.
from mmdet.utils import OptConfigType, ConfigType
from ai4rs.registry import MODELS
from .hbox2rbox_layer import Hbox2RboxLayer

@MODELS.register_module()
class Rbox2RboxLayer(Hbox2RboxLayer):
    r"""
    Args:
        num_classes (int): Number of class in dataset.
            Defaults to 80.
        num_cls_fcs (int): The number of fully-connected
            layers in classification subnet. Defaults to 1.
        num_reg_fcs (int): The number of fully-connected
            layers in regression subnet. Defaults to 3.
        dropout (float): Probability of drop the channel.
            Defaults to 0.0
        self_attn_cfg (:obj: `ConfigDict` or dict): Config for self
            attention.
        rroi_attn_cfg (:obj: `ConfigDict` or dict): Config for dy attention.
        ffn_cfg (:obj: `ConfigDict` or dict): Config for FFN.
        loss_iou (:obj:`ConfigDict` or dict): The config for iou or
            giou loss.
        init_cfg (:obj:`ConfigDict` or dict or list[:obj:`ConfigDict` or \
            dict]): Initialization config dict. Defaults to None.
    """

    def __init__(self,
                 num_classes=80,
                 angle_version='le90',
                 feat_channels=256,
                 dim_feedforward=2048,
                 num_cls_convs=1,
                 num_reg_convs=3,
                 num_heads=8,
                 dropout=0.0,
                 pooler_resolution=7,
                 act_cfg=dict(type='ReLU', inplace=True),
                 dynamic_conv=dict(dynamic_dim=64, dynamic_num=2),
                 loss_iou: ConfigType = dict(type='GIoULoss', loss_weight=2.0),
                 init_cfg: OptConfigType = None,
                 **kwargs) -> None:
        assert init_cfg is None, 'To prevent abnormal initialization ' \
                                 'behavior, init_cfg is not allowed to be set'
        super().__init__(
            num_classes=num_classes,
            angle_version=angle_version,
            feat_channels=feat_channels,
            dim_feedforward=dim_feedforward,
            num_cls_convs=num_cls_convs,
            num_reg_convs=num_reg_convs,
            num_heads=num_heads,
            dropout=dropout,
            pooler_resolution=pooler_resolution,
            act_cfg=act_cfg,
            dynamic_conv=dynamic_conv,
            loss_iou=loss_iou,
            init_cfg=init_cfg,
            **kwargs)