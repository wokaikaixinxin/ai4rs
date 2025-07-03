# Copyright (c) ai4rs. All rights reserved.
import torch
import torch.nn as nn
from torch import Tensor
from mmengine.config import ConfigDict
from mmengine.model import bias_init_with_prob
from mmcv.cnn import build_activation_layer
from mmdet.utils import OptConfigType, reduce_mean, ConfigType
from mmdet.models.roi_heads.bbox_heads import BBoxHead
from mmdet.models.losses import accuracy
from mmdet.models.task_modules import SamplingResult
from mmdet.models.utils import multi_apply
from ai4rs.registry import MODELS
from typing import List, Tuple
from .dynamic_cov import DynamicConv
from .match_cost import normalize_angle

@MODELS.register_module()
class Hbox2RboxLayer(BBoxHead):
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
            reg_decoded_bbox=True,
            reg_class_agnostic=True,
            init_cfg=init_cfg,
            **kwargs)
        self.angle_version = angle_version
        self.feat_channels = feat_channels

        # IoU Loss
        self.loss_iou = MODELS.build(loss_iou)

        # Dynamic
        self.self_attn = nn.MultiheadAttention(
            feat_channels, num_heads, dropout=dropout)
        self.inst_interact = DynamicConv(
            feat_channels=feat_channels,
            pooler_resolution=pooler_resolution,
            dynamic_dim=dynamic_conv['dynamic_dim'],
            dynamic_num=dynamic_conv['dynamic_num'])

        self.linear1 = nn.Linear(feat_channels, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, feat_channels)

        self.norm1 = nn.LayerNorm(feat_channels)
        self.norm2 = nn.LayerNorm(feat_channels)
        self.norm3 = nn.LayerNorm(feat_channels)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = build_activation_layer(act_cfg)

        # cls.
        cls_module = list()
        for _ in range(num_cls_convs):
            cls_module.append(nn.Linear(feat_channels, feat_channels, False))
            cls_module.append(nn.LayerNorm(feat_channels))
            cls_module.append(nn.ReLU(inplace=True))
        self.cls_module = nn.ModuleList(cls_module)

        # reg.
        reg_module = list()
        for _ in range(num_reg_convs):
            reg_module.append(nn.Linear(feat_channels, feat_channels, False))
            reg_module.append(nn.LayerNorm(feat_channels))
            reg_module.append(nn.ReLU(inplace=True))
        self.reg_module = nn.ModuleList(reg_module)

        # pred.
        if self.loss_cls.use_sigmoid:
            self.fc_cls = nn.Linear(feat_channels, num_classes)
        else:
            self.fc_cls = nn.Linear(feat_channels, num_classes + 1)
        self.fc_reg = nn.Linear(feat_channels, self.bbox_coder.encode_size)

        assert self.reg_class_agnostic, 'only ' \
            'suppport `reg_class_agnostic=True` '
        assert self.reg_decoded_bbox, 'only ' \
            'suppport `reg_decoded_bbox=True`'

    def init_weights(self) -> None:
        """Use xavier initialization for all weight parameter and set
        classification head bias as a specific value when use focal loss."""
        super().init_weights()
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                # adopt the default initialization for
                # the weight and bias of the layer norm
                pass
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            nn.init.constant_(self.fc_cls.bias, bias_init)

    def forward(self, roi_feat: Tensor, weight: Tensor, bs: int) -> tuple:
        """Forward function of RRoIFormer Decoder Layer.

        Args:
            roi_feat (Tensor): Roi-pooling features with shape
                (bs * num_box, embed_dims, pooling_h , pooling_w).
            weight (Tensor): Intermediate feature get from
                decoder in last stage, has shape (bs, num_box, embed_dims).
            bs (int): batch size.

        Returns:
            tuple[Tensor]: Usually a tuple of classification scores
            and bbox prediction and a intermediate feature.

            - cls_scores (Tensor): Classification scores for
              all queries, has shape (bs, num_queries, num_classes).
            - bbox_preds (Tensor): Box energies / deltas for
              all queries, has shape (bs, num_queries, 4).
            - obj_feat (Tensor): Object feature before classification
              and regression subnet, has shape (bs, num_queries, embed_dims).
        """
        num_boxes = roi_feat.shape[0] // bs

        if weight is None:
            weight = roi_feat.view(bs, num_boxes, self.feat_channels,
                                             -1).mean(-1)
        roi_features = roi_feat.view(bs * num_boxes, self.feat_channels,
                                         -1).permute(2, 0, 1)

        # self_att.
        weight = weight.view(bs, num_boxes,
                                         self.feat_channels).permute(1, 0, 2)
        weight2 = self.self_attn(
            weight, weight, value=weight)[0]
        weight = weight + self.dropout1(weight2)
        weight = self.norm1(weight)

        # inst_interact.
        weight = weight.view(
            num_boxes, bs,
            self.feat_channels).permute(1, 0,
                                        2).reshape(1, bs * num_boxes,
                                                   self.feat_channels)
        weight2 = self.inst_interact(weight, roi_features)
        weight = weight + self.dropout2(weight2)
        obj_features = self.norm2(weight)

        # obj_feature.
        obj_features2 = self.linear2(
            self.dropout(self.activation(self.linear1(obj_features))))
        obj_features = obj_features + self.dropout3(obj_features2)
        obj_features = self.norm3(obj_features)

        fc_feature = obj_features.transpose(0, 1).reshape(bs * num_boxes, -1)

        cls_feature = fc_feature.clone()
        reg_feature = fc_feature.clone()
        for cls_layer in self.cls_module:
            cls_feature = cls_layer(cls_feature)
        for reg_layer in self.reg_module:
            reg_feature = reg_layer(reg_feature)

        class_logits = self.fc_cls(cls_feature)
        bboxes_deltas = self.fc_reg(reg_feature)
        return  (class_logits.view(bs, num_boxes, -1),
                 bboxes_deltas.view(bs, num_boxes, -1),
                 obj_features)

    def loss_and_target(self,
                        cls_score: Tensor,
                        bbox_pred: Tensor,
                        sampling_results: List[SamplingResult],
                        rcnn_train_cfg: ConfigType,
                        imgs_whwht: Tensor,
                        concat: bool = True,
                        reduction_override: str = None) -> dict:
        """Calculate the loss based on the features extracted by the DIIHead.

        Args:
            cls_score (Tensor): Classification prediction
                results of all class, has shape
                (N, num_classes)
            bbox_pred (Tensor): Regression prediction results, has shape
                (N, 5), the last
                dimension 5 represents [cx, cy, w, h, radian].
            sampling_results (List[obj:SamplingResult]): Assign results of
                all images in a batch after sampling.
            rcnn_train_cfg (obj:ConfigDict): `train_cfg` of RCNN.
            imgs_whwht (Tensor): imgs_whwh (Tensor): Tensor with\
                shape (bs, num_query, 5), the last
                dimension means
                [img_width, img_height, img_width, img_height, 1.].
            concat (bool): Whether to concatenate the results of all
                the images in a single batch. Defaults to True.
            reduction_override (str, optional): The reduction
                method used to override the original reduction
                method of the loss. Options are "none",
                "mean" and "sum". Defaults to None.

        Returns:
            dict: A dictionary of loss and targets components.
            The targets are only used for cascade rcnn.
        """
        cls_reg_targets = self.get_targets(
            sampling_results=sampling_results,
            rcnn_train_cfg=rcnn_train_cfg,
            concat=concat)
        (labels, label_weights, bbox_targets, bbox_weights) = cls_reg_targets

        losses = dict()
        bg_class_ind = self.num_classes
        # note in spare rcnn num_gt == num_pos
        pos_inds = (labels >= 0) & (labels < bg_class_ind)
        num_pos = pos_inds.sum().float()
        avg_factor = reduce_mean(num_pos)
        if cls_score is not None:
            if cls_score.numel() > 0:
                losses['loss_cls'] = self.loss_cls(
                    cls_score,
                    labels,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)
                losses['pos_acc'] = accuracy(cls_score[pos_inds],
                                             labels[pos_inds])
        if bbox_pred is not None:
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            # do not perform bounding box regression for BG anymore.
            if pos_inds.any():
                pos_bbox_pred = bbox_pred.reshape(bbox_pred.size(0),
                                                  5)[pos_inds.type(torch.bool)]
                imgs_whwht = imgs_whwht.reshape(bbox_pred.size(0),
                                              5)[pos_inds.type(torch.bool)]
                pos_bbox_pred_temp = pos_bbox_pred / imgs_whwht
                bbox_targets_temp = bbox_targets[pos_inds.type(torch.bool)] / imgs_whwht
                pos_bbox_pred_temp[..., -1] = normalize_angle(pos_bbox_pred_temp[..., -1], self.angle_version)
                bbox_targets_temp[..., -1] = normalize_angle(bbox_targets_temp[..., -1], self.angle_version)
                losses['loss_bbox'] = self.loss_bbox(
                    pos_bbox_pred_temp,
                    bbox_targets_temp,
                    bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=avg_factor)
                losses['loss_iou'] = self.loss_iou(
                    pos_bbox_pred,
                    bbox_targets[pos_inds.type(torch.bool)],
                    bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=avg_factor)
            else:
                losses['loss_bbox'] = bbox_pred.sum() * 0
                losses['loss_iou'] = bbox_pred.sum() * 0
        return dict(loss_bbox=losses, bbox_targets=cls_reg_targets)

    def _get_targets_single(self, pos_inds: Tensor, neg_inds: Tensor,
                            pos_priors: Tensor, neg_priors: Tensor,
                            pos_gt_bboxes: Tensor, pos_gt_labels: Tensor,
                            cfg: ConfigDict) -> tuple:
        """Calculate the ground truth for proposals in the single image
        according to the sampling results.

        Almost the same as the implementation in `bbox_head`,
        we add pos_inds and neg_inds to select positive and
        negative samples instead of selecting the first num_pos
        as positive samples.

        Args:
            pos_inds (Tensor): The length is equal to the
                positive sample numbers contain all index
                of the positive sample in the origin proposal set.
            neg_inds (Tensor): The length is equal to the
                negative sample numbers contain all index
                of the negative sample in the origin proposal set.
            pos_priors (Tensor): Contains all the positive boxes,
                has shape (num_pos, 5), the last dimension 5
                represents [cx, cy, w, h, radian].
            neg_priors (Tensor): Contains all the negative boxes,
                has shape (num_neg, 5), the last dimension 5
                represents [cx, cy, w, h, radian].
            pos_gt_bboxes (Tensor): Contains gt_boxes for
                all positive samples, has shape (num_pos, 5),
                the last dimension 5
                represents [cx, cy, w, h, radian].
            pos_gt_labels (Tensor): Contains gt_labels for
                all positive samples, has shape (num_pos, ).
            cfg (obj:`ConfigDict`): `train_cfg` of R-CNN.

        Returns:
            Tuple[Tensor]: Ground truth for proposals in a single image.
            Containing the following Tensors:

            - labels(Tensor): Gt_labels for all proposals, has
              shape (num_proposals,).
            - label_weights(Tensor): Labels_weights for all proposals, has
              shape (num_proposals,).
            - bbox_targets(Tensor):Regression target for all proposals, has
              shape (num_proposals, 5), the last dimension 5
              represents [cx, cy, w, h, radian].
            - bbox_weights(Tensor):Regression weights for all proposals,
              has shape (num_proposals, 5).
        """
        num_pos = pos_priors.size(0)
        num_neg = neg_priors.size(0)
        num_samples = num_pos + num_neg

        # original implementation uses new_zeros since BG are set to be 0
        # now use empty & fill because BG cat_id = num_classes,
        # FG cat_id = [0, num_classes-1]
        labels = pos_priors.new_full((num_samples, ),
                                     self.num_classes,
                                     dtype=torch.long)
        label_weights = pos_priors.new_zeros(num_samples)
        bbox_targets = pos_priors.new_zeros(num_samples, 5)
        bbox_weights = pos_priors.new_zeros(num_samples, 5)
        if num_pos > 0:
            labels[pos_inds] = pos_gt_labels
            pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight
            label_weights[pos_inds] = pos_weight
            if not self.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_coder.encode(
                    pos_priors, pos_gt_bboxes)
            else:
                pos_bbox_targets = pos_gt_bboxes
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1
        if num_neg > 0:
            label_weights[neg_inds] = 1.0

        return labels, label_weights, bbox_targets, bbox_weights

    def get_targets(self,
                    sampling_results: List[SamplingResult],
                    rcnn_train_cfg: ConfigDict,
                    concat: bool = True) -> tuple:
        """Calculate the ground truth for all samples in a batch according to
        the sampling_results.

        Almost the same as the implementation in bbox_head, we passed
        additional parameters pos_inds_list and neg_inds_list to
        `_get_targets_single` function.

        Args:
            sampling_results (List[obj:SamplingResult]): Assign results of
                all images in a batch after sampling.
            rcnn_train_cfg (obj:ConfigDict): `train_cfg` of RCNN.
            concat (bool): Whether to concatenate the results of all
                the images in a single batch.

        Returns:
            Tuple[Tensor]: Ground truth for proposals in a single image.
            Containing the following list of Tensors:

            - labels (list[Tensor],Tensor): Gt_labels for all
              proposals in a batch, each tensor in list has
              shape (num_proposals,) when `concat=False`, otherwise just
              a single tensor has shape (num_all_proposals,).
            - label_weights (list[Tensor]): Labels_weights for
              all proposals in a batch, each tensor in list has shape
              (num_proposals,) when `concat=False`, otherwise just a
              single tensor has shape (num_all_proposals,).
            - bbox_targets (list[Tensor],Tensor): Regression target
              for all proposals in a batch, each tensor in list has
              shape (num_proposals, 4) when `concat=False`, otherwise
              just a single tensor has shape (num_all_proposals, 4),
              the last dimension 4 represents [tl_x, tl_y, br_x, br_y].
            - bbox_weights (list[tensor],Tensor): Regression weights for
              all proposals in a batch, each tensor in list has shape
              (num_proposals, 4) when `concat=False`, otherwise just a
              single tensor has shape (num_all_proposals, 4).
        """
        pos_inds_list = [res.pos_inds for res in sampling_results]
        neg_inds_list = [res.neg_inds for res in sampling_results]
        pos_priors_list = [res.pos_priors for res in sampling_results]
        neg_priors_list = [res.neg_priors for res in sampling_results]
        pos_gt_bboxes_list = [res.pos_gt_bboxes for res in sampling_results]
        pos_gt_labels_list = [res.pos_gt_labels for res in sampling_results]
        labels, label_weights, bbox_targets, bbox_weights = multi_apply(
            self._get_targets_single,
            pos_inds_list,
            neg_inds_list,
            pos_priors_list,
            neg_priors_list,
            pos_gt_bboxes_list,
            pos_gt_labels_list,
            cfg=rcnn_train_cfg)
        if concat:
            labels = torch.cat(labels, 0)
            label_weights = torch.cat(label_weights, 0)
            bbox_targets = torch.cat(bbox_targets, 0)
            bbox_weights = torch.cat(bbox_weights, 0)
        return labels, label_weights, bbox_targets, bbox_weights