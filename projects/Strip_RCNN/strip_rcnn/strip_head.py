# Copyright (c) ai4rs. All rights reserved.
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmengine.config import ConfigDict
from torch import Tensor
from mmdet.models.roi_heads.bbox_heads import BBoxHead
from ai4rs.registry import MODELS
from .reg_block import StripBlock

@MODELS.register_module()
class StripHead_(BBoxHead):
    r"""More general bbox head, with shared conv and fc layers and two optional
    separated branches.

    .. code-block:: none

                                    /-> cls convs -> cls fcs -> cls
        shared convs -> shared fcs
                                    \-> reg convs -> reg fcs -> reg
    """  # noqa: W605

    def __init__(self,
                 num_shared_convs: int = 0,
                 num_shared_fcs: int = 0,
                 num_cls_convs: int = 0,
                 num_cls_fcs: int = 0,
                 num_reg_xy_wh_convs=0,
                 num_reg_xy_wh_fcs=0,
                 num_reg_theta_convs=0,
                 num_reg_theta_fcs=0,
                 conv_out_channels: int = 256,
                 fc_out_channels: int = 1024,
                 conv_cfg: Optional[Union[dict, ConfigDict]] = None,
                 norm_cfg: Optional[Union[dict, ConfigDict]] = None,
                 init_cfg: Optional[Union[dict, ConfigDict]] = None,
                 *args,
                 **kwargs) -> None:
        super(StripHead_, self).__init__(*args, init_cfg=init_cfg, **kwargs)
        self.num_shared_convs = num_shared_convs
        self.num_shared_fcs = num_shared_fcs
        self.num_cls_convs = num_cls_convs
        self.num_cls_fcs = num_cls_fcs
        self.num_reg_xy_wh_convs = num_reg_xy_wh_convs
        self.num_reg_xy_wh_fcs = num_reg_xy_wh_fcs
        self.num_reg_theta_convs = num_reg_theta_convs
        self.num_reg_theta_fcs = num_reg_theta_fcs
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        # add shared convs and fcs
        self.shared_convs, self.shared_fcs, last_layer_dim = \
            self._add_conv_fc_branch(
                self.num_shared_convs, self.num_shared_fcs, self.in_channels,
                True)
        self.shared_out_channels = last_layer_dim

        # add cls specific branch
        self.cls_convs, self.cls_fcs, self.cls_last_dim = \
            self._add_conv_fc_branch(
                self.num_cls_convs, self.num_cls_fcs, self.shared_out_channels)

        # add reg xy specific branch
        self.reg_xy_wh_convs, self.reg_xy_wh_fcs, self.reg_xy_wh_last_dim = \
            self._add_conv_strip_fc_branch(
                self.num_reg_xy_wh_convs, self.num_reg_xy_wh_fcs, self.shared_out_channels)

        # add reg theta specific branch
        self.reg_theta_convs, self.reg_theta_fcs, self.reg_theta_last_dim = \
            self._add_conv_fc_branch(
                self.num_reg_theta_convs, self.num_reg_theta_fcs, self.shared_out_channels)

        if self.num_reg_xy_wh_fcs == 0:
            self.reg_xy_wh_last_dim *= self.roi_feat_area
        if self.num_reg_theta_fcs == 0:
            self.reg_theta_last_dim *= self.roi_feat_area

        self.relu = nn.ReLU(inplace=True)
        # reconstruct fc_cls and fc_reg since input channels are changed
        if self.with_cls:
            if self.custom_cls_channels:
                cls_channels = self.loss_cls.get_cls_channels(self.num_classes)
            else:
                cls_channels = self.num_classes + 1
            cls_predictor_cfg_ = self.cls_predictor_cfg.copy()
            cls_predictor_cfg_.update(
                in_features=self.cls_last_dim, out_features=cls_channels)
            self.fc_cls = MODELS.build(cls_predictor_cfg_)
        if self.with_reg:
            box_dim = self.bbox_coder.encode_size
            #todo: 判断bbox_coder是否是DeltaXYWHTRBBoxCoder
            out_dim_reg_xy_wh = 4 if self.reg_class_agnostic else \
                4 * self.num_classes
            out_dim_reg_theta = 1 if self.reg_class_agnostic else \
                1 * self.num_classes
            reg_xy_wh_predictor_cfg_ = self.reg_predictor_cfg.copy()
            if isinstance(reg_xy_wh_predictor_cfg_, (dict, ConfigDict)):
                reg_xy_wh_predictor_cfg_.update(
                    in_features=self.reg_xy_wh_last_dim, out_features=out_dim_reg_xy_wh)
            self.fc_reg_xy_wh = MODELS.build(reg_xy_wh_predictor_cfg_)

            reg_theta_predictor_cfg_ = self.reg_predictor_cfg.copy()
            if isinstance(reg_theta_predictor_cfg_, (dict, ConfigDict)):
                reg_theta_predictor_cfg_.update(
                    in_features=self.reg_theta_last_dim, out_features=out_dim_reg_theta)
            self.fc_reg_theta = MODELS.build(reg_theta_predictor_cfg_)

        if init_cfg is None:
            # when init_cfg is None,
            # It has been set to
            # [[dict(type='Normal', std=0.01, override=dict(name='fc_cls'))],
            #  [dict(type='Normal', std=0.001, override=dict(name='fc_reg'))]
            # after `super(ConvFCBBoxHead, self).__init__()`
            # we only need to append additional configuration
            # for `shared_fcs`, `cls_fcs` and `reg_fcs`
            self.init_cfg += [
                dict(
                    type='Xavier',
                    distribution='uniform',
                    override=[
                        dict(name='shared_fcs'),
                        dict(name='cls_fcs'),
                        dict(name='reg_xy_wh_fcs'),
                        dict(name='reg_theta_fcs')
                    ])
            ]

    def _add_conv_strip_fc_branch(self,
                                  num_branch_convs,
                                  num_branch_fcs,
                                  in_channels,
                                  is_shared=False):
        """Add shared or separable branch.

        convs -> avg pool (optional) -> fcs
        """
        last_layer_dim = in_channels
        # add branch specific conv layers
        branch_convs = nn.ModuleList()
        if num_branch_convs > 0:
            for i in range(num_branch_convs):
                conv_in_channels = (
                    last_layer_dim if i == 0 else self.conv_out_channels)
                branch_convs.append(
                    ConvModule(
                        conv_in_channels,
                        self.conv_out_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
                branch_convs.append(
                    StripBlock(self.conv_out_channels)
                )
            last_layer_dim = self.conv_out_channels
        # add branch specific fc layers
        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            # for shared branch, only consider self.with_avg_pool
            # for separated branches, also consider self.num_shared_fcs
            if (is_shared
                or self.num_shared_fcs == 0) and not self.with_avg_pool:
                last_layer_dim *= self.roi_feat_area
            for i in range(num_branch_fcs):
                fc_in_channels = (
                    last_layer_dim if i == 0 else self.fc_out_channels)
                branch_fcs.append(
                    nn.Linear(fc_in_channels, self.fc_out_channels))
            last_layer_dim = self.fc_out_channels
        return branch_convs, branch_fcs, last_layer_dim

    def _add_conv_fc_branch(self,
                            num_branch_convs: int,
                            num_branch_fcs: int,
                            in_channels: int,
                            is_shared: bool = False) -> tuple:
        """Add shared or separable branch.

        convs -> avg pool (optional) -> fcs
        """
        last_layer_dim = in_channels
        # add branch specific conv layers
        branch_convs = nn.ModuleList()
        if num_branch_convs > 0:
            for i in range(num_branch_convs):
                conv_in_channels = (
                    last_layer_dim if i == 0 else self.conv_out_channels)
                branch_convs.append(
                    ConvModule(
                        conv_in_channels,
                        self.conv_out_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
            last_layer_dim = self.conv_out_channels
        # add branch specific fc layers
        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            # for shared branch, only consider self.with_avg_pool
            # for separated branches, also consider self.num_shared_fcs
            if (is_shared
                    or self.num_shared_fcs == 0) and not self.with_avg_pool:
                last_layer_dim *= self.roi_feat_area
            for i in range(num_branch_fcs):
                fc_in_channels = (
                    last_layer_dim if i == 0 else self.fc_out_channels)
                branch_fcs.append(
                    nn.Linear(fc_in_channels, self.fc_out_channels))
            last_layer_dim = self.fc_out_channels
        return branch_convs, branch_fcs, last_layer_dim

    def forward(self, x: Tuple[Tensor]) -> tuple:
        """Forward features from the upstream network.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: A tuple of classification scores and bbox prediction.

                - cls_score (Tensor): Classification scores for all \
                    scale levels, each is a 4D-tensor, the channels number \
                    is num_base_priors * num_classes.
                - bbox_pred (Tensor): Box energies / deltas for all \
                    scale levels, each is a 4D-tensor, the channels number \
                    is num_base_priors * 4.
        """
        # shared part
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = x.flatten(1)

            for fc in self.shared_fcs:
                x = self.relu(fc(x))
        # separate branches
        x_cls = x
        x_reg = x

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        x_reg_xy_wh = x_reg
        for conv in self.reg_xy_wh_convs:
            x_reg_xy_wh = conv(x_reg_xy_wh)
        if x_reg_xy_wh.dim() > 2:
            if self.with_avg_pool:
                x_reg_xy_wh = self.avg_pool(x_reg_xy_wh)
            x_reg_xy_wh = x_reg_xy_wh.flatten(1)
        for fc in self.reg_xy_wh_fcs:
            x_reg_xy_wh = self.relu(fc(x_reg_xy_wh))

        x_reg_theta = x_reg
        for conv in self.reg_theta_convs:
            x_reg_theta = conv(x_reg_theta)
        if x_reg_theta.dim() > 2:
            if self.with_avg_pool:
                x_reg_theta = self.avg_pool(x_reg_theta)
            x_reg_theta = x_reg_theta.flatten(1)
        for fc in self.reg_theta_fcs:
            x_reg_theta = self.relu(fc(x_reg_theta))

        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        xy_wh_pred = self.fc_reg_xy_wh(x_reg_xy_wh) if self.with_reg else None
        theta_pred = self.fc_reg_theta(x_reg_theta) if self.with_reg else None
        bbox_pred = torch.cat((xy_wh_pred, theta_pred), dim=1)
        return cls_score, bbox_pred


@MODELS.register_module()
class StripHead(StripHead_):

    def __init__(self, fc_out_channels: int = 1024, *args, **kwargs) -> None:
        super(StripHead, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=0,
            num_cls_convs=0,
            num_cls_fcs=2,
            num_reg_xy_wh_convs=1,
            num_reg_xy_wh_fcs=0,
            num_reg_theta_convs=0,
            num_reg_theta_fcs=2,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)