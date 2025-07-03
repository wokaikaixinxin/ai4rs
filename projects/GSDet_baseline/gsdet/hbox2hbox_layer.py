# Copyright (c) ai4rs. All rights reserved.
from typing import Tuple
import warnings
import copy
import math
import torch
import torch.nn as nn
from torch import Tensor
from mmengine.structures import InstanceData
from mmcv.cnn import build_activation_layer
from mmdet.utils import InstanceList
from mmdet.structures.bbox import get_box_tensor
from mmdet.structures import SampleList
from mmdet.structures.bbox import (bbox_xyxy_to_cxcywh, bbox_cxcywh_to_xyxy, bbox2roi)
from ai4rs.structures.bbox import rbox2hbox
from ai4rs.registry import MODELS, TASK_UTILS
from .dynamic_cov import DynamicConv

_DEFAULT_SCALE_CLAMP = math.log(100000.0 / 16)

@MODELS.register_module()
class Hbox2HboxLayer(nn.Module):

    def __init__(self,
                 num_classes=80,
                 feat_channels=256,
                 num_proposals=900,
                 num_heads=6,
                 prior_prob=0.01,
                 deep_supervision=True,
                 single_head=dict(
                     type='SingleHbox2HboxHead',
                     num_cls_convs=1,
                     num_reg_convs=3,
                     dim_feedforward=2048,
                     num_heads=8,
                     dropout=0.0,
                     act_cfg=dict(type='ReLU'),
                     dynamic_conv=dict(dynamic_dim=64, dynamic_num=2)),
                 criterion=dict(
                     type='Hbox2HboxLayerCriterion',
                     num_classes=80,
                     assigner=dict(
                         type='Hbox2HboxLayerMatcher',
                         match_costs=[
                             dict(
                                 type='FocalLossCost',
                                 alpha=2.0,
                                 gamma=0.25,
                                 weight=2.0),
                             dict(
                                 type='BBoxL1Cost',
                                 weight=5.0,
                                 box_format='xyxy'),
                             dict(type='IoUCost', iou_mode='giou', weight=2.0)
                         ],
                         center_radius=2.5,
                         candidate_topk=5),
                 ),
                 roi_extractor=dict(
                     type='SingleRoIExtractor',
                     roi_layer=dict(
                         type='RoIAlign', output_size=7, sampling_ratio=2),
                     out_channels=256,
                     featmap_strides=[4, 8, 16, 32]),
                 test_cfg=None,
                 **kwargs) -> None:
        super().__init__()
        self.roi_extractor = MODELS.build(roi_extractor)

        self.num_classes = num_classes
        self.feat_channels = feat_channels
        self.num_proposals = num_proposals
        self.num_heads = num_heads

        # build criterion
        criterion.update(deep_supervision=deep_supervision)
        self.criterion = TASK_UTILS.build(criterion)
        self.use_focal_loss = self.criterion.loss_cls.use_sigmoid

        # Build Dynamic Head.
        single_head_ = single_head.copy()
        single_head_num_classes = single_head_.get('num_classes', None)
        if single_head_num_classes is None:
            single_head_.update(num_classes=num_classes)
        else:
            if single_head_num_classes != num_classes:
                warnings.warn(
                    'The `num_classes` of `DynamicDiffusionDetHead` and '
                    '`SingleDiffusionDetHead` should be same, changing '
                    f'`single_head.num_classes` to {num_classes}')
                single_head_.update(num_classes=num_classes)

        single_head_feat_channels = single_head_.get('feat_channels', None)
        if single_head_feat_channels is None:
            single_head_.update(feat_channels=feat_channels)
        else:
            if single_head_feat_channels != feat_channels:
                warnings.warn(
                    'The `feat_channels` of `DynamicDiffusionDetHead` and '
                    '`SingleDiffusionDetHead` should be same, changing '
                    f'`single_head.feat_channels` to {feat_channels}')
                single_head_.update(feat_channels=feat_channels)

        default_pooler_resolution = roi_extractor['roi_layer'].get(
            'output_size')
        assert default_pooler_resolution is not None
        single_head_pooler_resolution = single_head_.get('pooler_resolution')
        if single_head_pooler_resolution is None:
            single_head_.update(pooler_resolution=default_pooler_resolution)
        else:
            if single_head_pooler_resolution != default_pooler_resolution:
                warnings.warn(
                    'The `pooler_resolution` of `DynamicDiffusionDetHead` '
                    'and `SingleDiffusionDetHead` should be same, changing '
                    f'`single_head.pooler_resolution` to {num_classes}')
                single_head_.update(
                    pooler_resolution=default_pooler_resolution)

        single_head_.update(use_focal_loss=self.use_focal_loss)
        single_head_module = MODELS.build(single_head_)

        self.num_heads = num_heads
        self.head_series = nn.ModuleList(
            [copy.deepcopy(single_head_module) for _ in range(num_heads)])

        self.deep_supervision = deep_supervision

        self.prior_prob = prior_prob
        self._init_weights()

    def _init_weights(self):
        # init all parameters.
        bias_value = -math.log((1 - self.prior_prob) / self.prior_prob)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

            # initialize the bias for focal loss and fed loss.
            if self.use_focal_loss or self.use_fed_loss:
                if p.shape[-1] == self.num_classes or \
                        p.shape[-1] == self.num_classes + 1:
                    nn.init.constant_(p, bias_value)

    def initial_box(self, batch_img_metas, device):
        '''
        Args:
            batch_img_metas:
            device:
        Returns:
            diff_bboxes_abs (Tensor): [bs, num_boxes, 4], where 4 is (x1,y1,x2,y2)
        '''
        bs = len(batch_img_metas)
        box_placeholder = torch.randn(
            [bs, self.num_proposals, 4], device=device) / 6. + 0.5  # [bs, num_boxes, 4]
        box_placeholder[..., 2:] = torch.clip(
            box_placeholder[..., 2:], min=1e-4)
        random_bboxes = bbox_cxcywh_to_xyxy(box_placeholder)

        imgs_whwh = []
        for meta in batch_img_metas:
            h, w = meta['img_shape']
            imgs_whwh.append(torch.tensor([[w, h, w, h]], device=device))
        imgs_whwh = torch.cat(imgs_whwh, dim=0) # [bs, 4]
        imgs_whwh = imgs_whwh[:, None, :]       # [bs, 1, 4]

        # convert to abs bboxes
        random_bboxes_abs = random_bboxes * imgs_whwh

        return random_bboxes_abs

    def forward(self, features, init_bboxes, init_features=None):
        '''
        Args:
            features:
            init_bboxes (Tensor): bs, num_box, 4
            init_features:

        Returns:
            class_logits (Tensor): [n_layer, bs, n_boxes, class]
            pred_bboxes (Tensor): [n_layer, bs, n_boxes, 4]
            proposal_features (Tensor): [bs, n_boxes, 256]
        '''

        inter_class_logits = []
        inter_pred_bboxes = []

        bs, n_box = init_bboxes.shape[:2]
        bboxes = init_bboxes

        if init_features is not None:
            init_features = init_features[None].repeat(1, bs, 1)
            proposal_features = init_features.clone()
        else:
            proposal_features = None

        for head_idx, single_head in enumerate(self.head_series):
            class_logits, pred_bboxes, proposal_features = single_head(
                features, bboxes, proposal_features, self.roi_extractor)
            if self.deep_supervision:
                inter_class_logits.append(class_logits)
                inter_pred_bboxes.append(pred_bboxes)
            bboxes = pred_bboxes.detach()

        if self.deep_supervision:
            return torch.stack(inter_class_logits), torch.stack(
                inter_pred_bboxes), proposal_features.view(bs, n_box, -1)
        else:
            return class_logits[None, ...], pred_bboxes[None, ...], proposal_features.view(bs, n_box, -1)

    def loss(self, x: Tuple[Tensor], batch_data_samples: SampleList) -> dict:
        """Perform forward propagation and loss calculation of the detection
        head on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        """
        device = x[0].device

        batch_gt_instances = []
        batch_img_metas = []
        for data_sample in batch_data_samples:
            img_meta = data_sample.metainfo
            gt_instances = copy.deepcopy(data_sample.gt_instances)

            gt_bboxes = rbox2hbox(get_box_tensor(gt_instances.bboxes))
            gt_instances.bboxes = gt_bboxes
            h, w = img_meta['img_shape']
            image_size = gt_bboxes.new_tensor([w, h, w, h])

            norm_gt_bboxes = gt_bboxes / image_size
            norm_gt_bboxes_cxcywh = bbox_xyxy_to_cxcywh(norm_gt_bboxes)

            gt_instances.set_metainfo(dict(image_size=image_size))
            gt_instances.norm_bboxes_cxcywh = norm_gt_bboxes_cxcywh

            batch_gt_instances.append(gt_instances)
            batch_img_metas.append(data_sample.metainfo)

        random_bboxes_abs = self.initial_box(batch_img_metas, device)   # bs, num_box, 4

        pred_logits, pred_bboxes, weight = self.forward(
            x, random_bboxes_abs, init_features=None)   # x,y,x,y

        imgs_whwht = []
        for meta in batch_img_metas:
            h, w = meta['img_shape']
            imgs_whwht.append(torch.tensor([[w, h, w, h, 1.]], device=device))
        imgs_whwht = torch.cat(imgs_whwht, dim=0)
        imgs_whwht = imgs_whwht[:, None, :] # bs, 1, 5
        results_list = []
        for idx in range(len(batch_img_metas)):
            results = InstanceData()
            results.bboxes = pred_bboxes[-1][idx]
            results.weight = weight[idx]
            results.imgs_whwht = imgs_whwht[idx].repeat(
                weight.size(1), 1)
            results_list.append(results)

        output = {
            'pred_logits': pred_logits[-1],
            'pred_boxes': pred_bboxes[-1]
        }
        if self.deep_supervision:
            output['aux_outputs'] = [{
                'pred_logits': a,
                'pred_boxes': b
            } for a, b in zip(pred_logits[:-1], pred_bboxes[:-1])]

        losses = self.criterion(output, batch_gt_instances, batch_img_metas)
        return losses, results_list

    def predict(self,
                x: Tuple[Tensor],
                batch_data_samples: SampleList,
                # rescale: bool = False
                ) -> InstanceList:
        """Perform forward propagation of the detection head and predict
        detection results on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[obj:`InstanceData`]: Detection results of each image
            after the post process.
        """
        device = x[-1].device
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]
        random_bboxes_abs = self.initial_box(batch_img_metas, device)
        results_list = self.predict_by_feat(
            x,
            random_bboxes_abs=random_bboxes_abs,
            device=device,
            batch_img_metas=batch_img_metas)
        return results_list

    def predict_by_feat(self,
                        x,
                        random_bboxes_abs,
                        device,
                        batch_img_metas=None,
                        ):
        pred_logits, pred_bboxes, weight = self(x, random_bboxes_abs, init_features=None)

        imgs_whwht = []
        for meta in batch_img_metas:
            h, w = meta['img_shape']
            imgs_whwht.append(torch.tensor([[w, h, w, h, 1.]], device=device))
        imgs_whwht = torch.cat(imgs_whwht, dim=0)
        imgs_whwht = imgs_whwht[:, None, :] # bs, 1, 5
        results_list = []
        for idx in range(len(batch_img_metas)):
            results = InstanceData()
            results.bboxes = pred_bboxes[-1][idx]
            results.weight = weight[idx]
            results.imgs_whwht = imgs_whwht[idx].repeat(
                weight.size(1), 1)
            results_list.append(results)

        return results_list


@MODELS.register_module()
class SingleHbox2HboxHead(nn.Module):

    def __init__(
        self,
        num_classes=80,
        feat_channels=256,
        dim_feedforward=2048,
        num_cls_convs=1,
        num_reg_convs=3,
        num_heads=8,
        dropout=0.0,
        pooler_resolution=7,
        scale_clamp=_DEFAULT_SCALE_CLAMP,
        bbox_weights=(2.0, 2.0, 1.0, 1.0),
        use_focal_loss=True,
        act_cfg=dict(type='ReLU', inplace=True),
        dynamic_conv=dict(dynamic_dim=64, dynamic_num=2)
    ) -> None:
        super().__init__()
        self.feat_channels = feat_channels

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
        self.use_focal_loss = use_focal_loss
        if self.use_focal_loss:
            self.class_logits = nn.Linear(feat_channels, num_classes)
        else:
            self.class_logits = nn.Linear(feat_channels, num_classes + 1)
        self.bboxes_delta = nn.Linear(feat_channels, 4)
        self.scale_clamp = scale_clamp
        self.bbox_weights = bbox_weights

    def forward(self, features, bboxes, pro_features, pooler):
        """
        :param bboxes: (N, num_boxes, 4)
        :param pro_features: (N, num_boxes, feat_channels)
        """

        N, num_boxes = bboxes.shape[:2]

        # roi_feature.
        proposal_boxes = list()
        for b in range(N):
            proposal_boxes.append(bboxes[b])
        rois = bbox2roi(proposal_boxes)

        roi_features = pooler(features, rois)

        if pro_features is None:
            pro_features = roi_features.view(N, num_boxes, self.feat_channels,
                                             -1).mean(-1)

        roi_features = roi_features.view(N * num_boxes, self.feat_channels,
                                         -1).permute(2, 0, 1)

        # self_att.
        pro_features = pro_features.view(N, num_boxes,
                                         self.feat_channels).permute(1, 0, 2)
        pro_features2 = self.self_attn(
            pro_features, pro_features, value=pro_features)[0]
        pro_features = pro_features + self.dropout1(pro_features2)
        pro_features = self.norm1(pro_features)

        # inst_interact.
        pro_features = pro_features.view(
            num_boxes, N,
            self.feat_channels).permute(1, 0,
                                        2).reshape(1, N * num_boxes,
                                                   self.feat_channels)
        pro_features2 = self.inst_interact(pro_features, roi_features)
        pro_features = pro_features + self.dropout2(pro_features2)
        obj_features = self.norm2(pro_features)

        # obj_feature.
        obj_features2 = self.linear2(
            self.dropout(self.activation(self.linear1(obj_features))))
        obj_features = obj_features + self.dropout3(obj_features2)
        obj_features = self.norm3(obj_features)

        fc_feature = obj_features.transpose(0, 1).reshape(N * num_boxes, -1)

        cls_feature = fc_feature.clone()
        reg_feature = fc_feature.clone()
        for cls_layer in self.cls_module:
            cls_feature = cls_layer(cls_feature)
        for reg_layer in self.reg_module:
            reg_feature = reg_layer(reg_feature)
        class_logits = self.class_logits(cls_feature)
        bboxes_deltas = self.bboxes_delta(reg_feature)
        pred_bboxes = self.apply_deltas(bboxes_deltas, bboxes.view(-1, 4))

        return (class_logits.view(N, num_boxes,
                                  -1), pred_bboxes.view(N, num_boxes,
                                                        -1), obj_features)

    def apply_deltas(self, deltas, boxes):
        """Apply transformation `deltas` (dx, dy, dw, dh) to `boxes`.

        Args:
            deltas (Tensor): transformation deltas of shape (N, k*4),
                where k >= 1. deltas[i] represents k potentially
                different class-specific box transformations for
                the single box boxes[i].
            boxes (Tensor): boxes to transform, of shape (N, 4)
        """
        boxes = boxes.to(deltas.dtype)

        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights

        wx, wy, ww, wh = self.bbox_weights
        dx = deltas[:, 0::4] / wx
        dy = deltas[:, 1::4] / wy
        dw = deltas[:, 2::4] / ww
        dh = deltas[:, 3::4] / wh

        # Prevent sending too large values into torch.exp()
        dw = torch.clamp(dw, max=self.scale_clamp)
        dh = torch.clamp(dh, max=self.scale_clamp)

        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]

        pred_boxes = torch.zeros_like(deltas)
        pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w  # x1
        pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h  # y1
        pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w  # x2
        pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h  # y2

        return pred_boxes