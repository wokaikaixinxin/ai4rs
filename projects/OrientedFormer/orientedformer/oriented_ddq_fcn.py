import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from mmcv.cnn import ConvModule, Scale
from mmcv.ops import batched_nms
from mmdet.models.dense_heads import AnchorFreeHead
from mmdet.structures.bbox import get_box_tensor
from mmengine.config import ConfigDict
from mmengine.structures import InstanceData
from mmengine.model import bias_init_with_prob, normal_init

from mmrotate.registry import MODELS, TASK_UTILS
from mmrotate.structures.bbox import distance2obb, RotatedBoxes
from mmdet.models.utils import (multi_apply, filter_scores_and_topk,
                                select_single_mlvl, sigmoid_geometric_mean)
from mmdet.utils import reduce_mean
from mmdet.models.utils import levels_to_images
from mmdet.models.task_modules.prior_generators.point_generator import \
    MlvlPointGenerator


@MODELS.register_module()
class OrientedDDQFCN(AnchorFreeHead):
    """Oriented DDQ FCN RPN head for OrientedFormer."""
    def __init__(self,
                 *args,
                 angle_version: str='le90',
                 ddq_num_classes: int = 15,
                 num_proposals: int = 300,
                 shuffle_channles=64,
                 dqs_cfg=dict(type='nms_rotated', iou_threshold=0.7, nms_pre=1000),
                 offset=0.5,
                 strides=(8, 16, 32, 64, 128),
                 aux_loss=ConfigDict(
                     loss_cls=dict(
                         type='QualityFocalLoss',
                         use_sigmoid=True,
                         activated=True,  # use probability instead of logit as input
                         beta=2.0,
                         loss_weight=1.0),
                     loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
                     train_cfg=dict(assigner=dict(type='TopkHungarianAssigner',
                                                  topk=8),
                                    alpha=1,
                                    beta=6),
                 ),
                 main_loss=ConfigDict(
                     loss_cls=dict(
                         type='QualityFocalLoss',
                         use_sigmoid=True,
                         activated=True,  # use probability instead of logit as input
                         beta=2.0,
                         loss_weight=1.0),
                     loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
                     train_cfg=dict(assigner=dict(type='TopkHungarianAssigner',
                                                  topk=1),
                                    alpha=1,
                                    beta=6),
                 ),
                 **kwargs) -> None:
        self.ddq_num_classes = ddq_num_classes
        self.num_proposals = num_proposals      # before super().__init__(), because of _init_layers
        super().__init__(*args,
                         strides=strides,
                         loss_cls = dict(type='mmdet.FocalLoss'),   # just to pass the stupid registry
                         loss_bbox = dict(type='mmdet.IoULoss'),    # just to pass the stupid registry
                         bbox_coder = dict(type='mmdet.DistancePointBBoxCoder'),
                         **kwargs)
        self.loss_bbox, self.loss_cls, self.bbox_coder = None, None, None   # donnot use them, just to pass the stupid registry
        self.angle_version = angle_version
        self.aux_loss = AuxLoss(**aux_loss)
        self.main_loss = AuxLoss(**main_loss)
        self.dqs_cfg = dqs_cfg

        self.shuffle_channles = shuffle_channles

        # contains the tuple of level indices that will do the interaction
        self.fuse_lvl_list = []
        num_levels = len(self.prior_generator.strides)
        for lvl in range(num_levels):
            top_lvl = min(lvl + 1, num_levels - 1)
            dow_lvl = max(lvl - 1, 0)
            tar_lvl = lvl
            self.fuse_lvl_list.append((tar_lvl, top_lvl, dow_lvl))

        self.remain_chs = self.in_channels - self.shuffle_channles * 2
        self.init_weights()
        self.prior_generator = MlvlPointGenerator(strides, offset=offset)


    def _init_layers(self):
        self.inter_convs = nn.ModuleList()
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(chn,
                           self.feat_channels,
                           3,
                           stride=1,
                           padding=3 // 2,
                           conv_cfg=self.conv_cfg,
                           norm_cfg=self.norm_cfg))
            self.reg_convs.append(
                ConvModule(chn,
                           self.feat_channels,
                           3,
                           stride=1,
                           padding=3 // 2,
                           conv_cfg=self.conv_cfg,
                           norm_cfg=self.norm_cfg))

        self.objectness = nn.Sequential(
            nn.Conv2d(self.feat_channels, self.feat_channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.feat_channels // 4, 1, 3, padding=3 // 2))

        cls_out_channels = self.ddq_num_classes

        self.conv_cls = nn.Conv2d(self.feat_channels,
                                  self.num_base_priors * cls_out_channels,
                                  3,
                                  padding=3 // 2)

        self.conv_reg = nn.Conv2d(self.feat_channels,
                                  self.num_base_priors * 5,
                                  3,
                                  padding=3 // 2)
        self.scales = nn.ModuleList(
            [Scale(1.0) for _ in self.prior_generator.strides])

        self.aux_conv_objectness = nn.Sequential(
            nn.Conv2d(self.feat_channels, self.feat_channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.feat_channels // 4, 1, 3, padding=3 // 2))

        cls_out_channels = self.ddq_num_classes

        self.aux_conv_cls = nn.Conv2d(self.feat_channels,
                                      self.num_base_priors * cls_out_channels,
                                      3,
                                      padding=3 // 2)

        self.aux_conv_reg = nn.Conv2d(self.feat_channels,
                                      self.num_base_priors * 5,
                                      3,
                                      padding=3 // 2)
        self.aux_scales = nn.ModuleList(
            [Scale(1.0) for _ in self.prior_generator.strides])

        self.compress = nn.Linear(self.feat_channels * 2, self.feat_channels)

    def init_weights(self):
        """Initialize weights of the head."""
        bias_cls = bias_init_with_prob(0.01)
        for m in self.inter_convs.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)
        for m in self.cls_convs.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)
        for m in self.reg_convs.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)
        for layer in self.objectness.modules():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                torch.nn.init.constant_(layer.bias, 0)
            if isinstance(layer, nn.GroupNorm):
                torch.nn.init.constant_(layer.weight, 1)
                torch.nn.init.constant_(layer.bias, 0)

        normal_init(self.conv_cls, std=0.01, bias=bias_cls)
        normal_init(self.conv_reg, std=0.01)

        # only be used in training
        normal_init(self.aux_conv_objectness[-1], std=0.01, bias=bias_cls)
        normal_init(self.aux_conv_cls, std=0.01, bias=bias_cls)
        normal_init(self.aux_conv_reg, std=0.01)

    def _single_shuffle(self, inputs, conv_module):
        if not isinstance(conv_module, (nn.ModuleList, list)):
            conv_module = [conv_module]
        for single_conv_m in conv_module:
            fused_inputs = []
            for fuse_lvl_tuple in self.fuse_lvl_list:
                tar_lvl, top_lvl, dow_lvl = fuse_lvl_tuple
                tar_input = inputs[tar_lvl]
                top_input = inputs[top_lvl]
                down_input = inputs[dow_lvl]
                remain = tar_input[:, :self.remain_chs]
                from_top = top_input[:,
                                     self.remain_chs:][:,
                                                       self.shuffle_channles:]
                from_top = F.interpolate(from_top,
                                         size=tar_input.shape[-2:],
                                         mode='bilinear',
                                         align_corners=True)
                from_down = down_input[:, self.remain_chs:][:, :self.
                                                            shuffle_channles]
                from_down = F.interpolate(from_down,
                                          size=tar_input.shape[-2:],
                                          mode='bilinear',
                                          align_corners=True)
                fused_inputs.append(
                    torch.cat([remain, from_top, from_down], dim=1))
            fused_inputs = [single_conv_m(item) for item in fused_inputs]
            inputs = fused_inputs
        return inputs

    def loss_and_predict(
            self,
            x: list,
            img_metas: list,
            gt_bboxes: list,
            gt_labels=None,
            gt_bboxes_ignore=None,
            **kwargs):
        '''
        Args:
            x (list[Tensor]): Features from the upstream network, each is
                a 4D-tensor (bs, c, h, w).
            img_metas (list[dict]): Includes 'batch_input_shape', 'pad_shape', 'flip_direction',
                'ori_shape', 'img_id', 'flip', 'img_shape', 'scale_factor', 'img_path'.
            gt_bboxes (list[RotatedBoxes]):  (n, 5), where 5 represent (c_x, c_y, w, h, radian)
            gt_labels (list[Tensor]): (n)
            gt_bboxes_ignore: None
            **kwargs:

        Returns:

        '''
        self.img_metas = img_metas
        self.gt_bboxes = gt_bboxes
        loss = dict()

        main_results, aux_results = self.forward(x)

        main_loss_inputs, aux_loss_inputs, distinc_query_dict = \
            self.get_inputs(main_results, aux_results, img_metas=img_metas)

        aux_loss = self.aux_loss(*aux_loss_inputs,
                                 gt_bboxes=gt_bboxes,
                                 gt_labels=gt_labels,
                                 img_metas=img_metas)
        for k, v in aux_loss.items():
            loss[f'aux_{k}'] = v

        main_loss = self.main_loss(*main_loss_inputs,
                                   gt_bboxes=gt_bboxes,
                                   gt_labels=gt_labels,
                                   img_metas=img_metas)

        loss.update(main_loss)

        imgs_whwht = []
        for meta in img_metas:
            h, w = meta['img_shape']
            imgs_whwht.append(x[0].new_tensor([[w, h, w, h, 1.]]))
        imgs_whwht = torch.cat(imgs_whwht, dim=0)
        imgs_whwht = imgs_whwht[:, None, :]

        return loss, imgs_whwht, distinc_query_dict

    def forward(self, inputs: list, **kwargs):
        '''
        Args:
            inputs (list[Tensor]): Features from the upstream network, each is
                a 4D-tensor (bs, c, h, w).
            **kwargs:

        Returns:
            main_results (dict[list]): 'cls_scores_list' (list[Tensor]): (bs, num_class, h, w),
                'bbox_preds_list' (list[Tensor]): (bs, 5, h, w),
                'cls_feats' (list[Tensor]): (bs, c, h, w),
                'reg_feats' (list[Tensor]): (bs, c, h, w).
            aux_results (dict[list]): 'cls_scores_list' (list[Tensor]): (bs, num_class, h, w),
                'bbox_preds_list' (list[Tensor]): (bs, 5, h, w),
                'cls_feats' (list[Tensor]): (bs, c, h, w),
                'reg_feats' (list[Tensor]): (bs, c, h, w).
        '''
        cls_convs = self.cls_convs
        reg_convs = self.reg_convs
        scales = self.scales
        conv_objectness = self.objectness
        conv_cls = self.conv_cls
        conv_reg = self.conv_reg
        cls_feats = inputs
        reg_feats = inputs

        cls_scores_list = []
        bbox_preds_list = []

        for layer_index, conv_m in enumerate(cls_convs):
            # shuffle last 2 feature maps
            if layer_index > 1:
                cls_feats = self._single_shuffle(cls_feats, [conv_m])
            else:
                cls_feats = [conv_m(item) for item in cls_feats]

        for layer_index, conv_m in enumerate(reg_convs):
            # shuffle last feature maps
            if layer_index > 2:
                reg_feats = self._single_shuffle(reg_feats, [conv_m])
            else:
                reg_feats = [conv_m(item) for item in reg_feats]
        for idx, (cls_feat, reg_feat,
                  scale) in enumerate(zip(cls_feats, reg_feats, scales)):
            cls_logits = conv_cls(cls_feat)
            object_nesss = conv_objectness(reg_feat)
            cls_score = sigmoid_geometric_mean(cls_logits, object_nesss)
            reg_dist = scale(conv_reg(reg_feat).exp()).float()
            cls_scores_list.append(cls_score)
            bbox_preds_list.append(reg_dist)

        main_results = dict(cls_scores_list=cls_scores_list,
                            bbox_preds_list=bbox_preds_list,
                            cls_feats=cls_feats,
                            reg_feats=reg_feats)
        if self.training:
            cls_scores_list = []
            bbox_preds_list = []

            for idx, (cls_feat, reg_feat, scale) in enumerate(
                    zip(cls_feats, reg_feats, self.aux_scales)):
                cls_logits = self.aux_conv_cls(cls_feat)
                object_nesss = self.aux_conv_objectness(reg_feat)
                cls_score = sigmoid_geometric_mean(cls_logits, object_nesss)
                reg_dist = scale(self.aux_conv_reg(reg_feat).exp()).float()
                cls_scores_list.append(cls_score)
                bbox_preds_list.append(reg_dist)
            aux_results = dict(cls_scores_list=cls_scores_list,
                               bbox_preds_list=bbox_preds_list,
                               cls_feats=cls_feats,
                               reg_feats=reg_feats)
        else:
            aux_results = None
        return main_results, aux_results

    def get_inputs(self, main_results, aux_results, img_metas=None):
        '''
        Args:
            main_results (dict[list]): 'cls_scores_list' (list[Tensor]): (bs, num_class, h, w),
                'bbox_preds_list' (list[Tensor]): (bs, 5, h, w),
                'cls_feats' (list[Tensor]): (bs, c, h, w),
                'reg_feats' (list[Tensor]): (bs, c, h, w).
            aux_results (dict[list]): 'cls_scores_list' (list[Tensor]): (bs, num_class, h, w),
                'bbox_preds_list' (list[Tensor]): (bs, 5, h, w),
                'cls_feats' (list[Tensor]): (bs, c, h, w),
                'reg_feats' (list[Tensor]): (bs, c, h, w).
            img_metas:

        Returns:

        '''
        mlvl_score = main_results['cls_scores_list']

        num_levels = len(mlvl_score)
        featmap_sizes = [mlvl_score[i].shape[-2:] for i in range(num_levels)]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=mlvl_score[0].dtype,
            device=mlvl_score[0].device)

        all_cls_scores, all_bbox_preds, all_query_ids = self.pre_dqs(
            **main_results, mlvl_priors=mlvl_priors, img_metas=img_metas)
        if aux_results is None:
            (aux_cls_scores, aux_bbox_preds) = (all_cls_scores, all_bbox_preds)
        else:
            aux_cls_scores, aux_bbox_preds, aux_query_ids = self.pre_dqs(
                **aux_results, mlvl_priors=mlvl_priors, img_metas=img_metas)

        dqs_all_cls_scores, dqs_all_bbox_preds, dqs_query_ids = self.dqs(
            all_cls_scores, all_bbox_preds, all_query_ids)

        distinct_query_dict = self.construct_query(main_results,
                                                   dqs_all_cls_scores,
                                                   dqs_all_bbox_preds,
                                                   dqs_query_ids,
                                                   self.num_proposals)

        return (dqs_all_cls_scores, dqs_all_bbox_preds), \
                (aux_cls_scores, aux_bbox_preds), \
                    distinct_query_dict

    def pre_dqs(self,
                cls_scores_list=None,
                bbox_preds_list=None,
                mlvl_priors=None,
                img_metas=None,
                **kwargs):

        num_imgs = cls_scores_list[0].size(0)
        all_cls_scores = []
        all_bbox_preds = []
        all_query_ids = []
        for img_id in range(num_imgs):

            single_cls_score_list = select_single_mlvl(cls_scores_list,
                                                       img_id,
                                                       detach=False)
            sinlge_bbox_pred_list = select_single_mlvl(bbox_preds_list,
                                                       img_id,
                                                       detach=False)
            cls_score, bbox_pred, query_inds = self._get_topk(
                single_cls_score_list, sinlge_bbox_pred_list, mlvl_priors)
            all_cls_scores.append(cls_score)
            all_bbox_preds.append(bbox_pred)
            all_query_ids.append(query_inds)
        return all_cls_scores, all_bbox_preds, all_query_ids

    def _get_topk(self, cls_score_list, bbox_pred_list, mlvl_priors,
                  **kwargs):
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_query_inds = []
        start_inds = 0
        for level_idx, (cls_score, bbox_pred, priors, stride) in \
                enumerate(zip(cls_score_list, bbox_pred_list,
                     mlvl_priors, \
                        self.prior_generator.strides)):

            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 5)

            cls_score = cls_score.permute(1, 2,
                                          0).reshape(-1, self.ddq_num_classes)

            binary_cls_score = cls_score.max(-1).values.reshape(-1, 1)
            if self.dqs_cfg:
                nms_pre = self.dqs_cfg.pop('nms_pre', 1000)
            else:
                if self.training:
                    nms_pre = len(binary_cls_score)
                else:
                    nms_pre = 1000
            results = filter_scores_and_topk(
                binary_cls_score, 0, nms_pre,
                dict(bbox_pred=bbox_pred, priors=priors, cls_score=cls_score))
            scores, labels, keep_idxs, filtered_results = results
            keep_idxs = (keep_idxs + start_inds) // self.num_base_priors
            start_inds = start_inds + len(cls_score)
            bbox_pred = filtered_results['bbox_pred']
            priors = filtered_results['priors']
            cls_score = filtered_results['cls_score']
            bbox_pred[..., :4] = bbox_pred[..., :4] * stride[0]
            bbox_pred = distance2obb(priors, bbox_pred, angle_version=self.angle_version)
            mlvl_bboxes.append(bbox_pred)
            mlvl_scores.append(cls_score)
            mlvl_query_inds.append(keep_idxs)

        return torch.cat(mlvl_scores), torch.cat(mlvl_bboxes), torch.cat(
            mlvl_query_inds)


    def dqs(self, all_mlvl_scores, all_mlvl_bboxes, all_query_ids, **kwargs):
        all_distinct_bboxes = []
        all_distinct_scores = []
        all_distinct_query_ids = []
        for mlvl_bboxes, mlvl_scores, query_id in zip(all_mlvl_bboxes,
                                                      all_mlvl_scores,
                                                      all_query_ids):
            if mlvl_bboxes.numel() == 0:
                return mlvl_bboxes, mlvl_scores, query_id

            det_bboxes, keep_idxs = batched_nms(mlvl_bboxes,
                                                mlvl_scores.max(-1).values,
                                                torch.ones(len(mlvl_scores)),
                                                self.dqs_cfg)

            all_distinct_bboxes.append(mlvl_bboxes[keep_idxs])
            all_distinct_scores.append(mlvl_scores[keep_idxs])
            all_distinct_query_ids.append(query_id[keep_idxs])
        return all_distinct_scores, all_distinct_bboxes, all_distinct_query_ids


    def construct_query(self, feat_dict: dict, scores: list,
                        bboxes: list, query_ids: list, num: int=300):
        '''
        Args:
            feat_dict (dict): 'cls_scores_list' (list[Tensor]): (bs, num_class, h, w),
                'bbox_preds_list' (list[Tensor]): (bs, 5, h, w),
                'cls_feats' (list[Tensor]): (bs, c, h, w),
                'reg_feats' (list[Tensor]): (bs, c, h, w).
            scores (list[Tensor]): (n, num_class).
            bboxes (list[Tensor]): (n, 5), where 5 represent (c_x, c_y, w, h, radian).
            query_ids (list[Tensor]): (n, )
            num (int): number of queries.
        Returns:
            query_xyzrt  (Tensor): (bs, num_query, 5), where 5 represent (c_x, c_y, z, r, radian)
            query_content (Tensor): (bs, num_query, 256).
        '''
        cls_feats = feat_dict['cls_feats']
        reg_feats = feat_dict['reg_feats']
        cls_feats = levels_to_images(cls_feats)
        reg_feats = levels_to_images(reg_feats)
        num_img = len(cls_feats)
        all_img_proposals = []
        all_img_object_feats = []
        for img_id in range(num_img):
            singl_scores = scores[img_id].max(-1).values
            singl_bboxes = bboxes[img_id]
            single_ids = query_ids[img_id]
            singl_cls_feats = cls_feats[img_id]
            singl_reg_feats = reg_feats[img_id]

            object_feats = torch.cat([singl_cls_feats, singl_reg_feats],
                                     dim=-1)

            object_feats = object_feats.detach()
            singl_bboxes = singl_bboxes.detach()

            object_feats = self.compress(object_feats)

            select_ids = torch.sort(singl_scores,
                                    descending=True).indices[:num]
            single_ids = single_ids[select_ids]
            singl_bboxes = singl_bboxes[select_ids]

            object_feats = object_feats[single_ids]
            all_img_object_feats.append(object_feats)
            all_img_proposals.append(singl_bboxes)

        all_img_object_feats = align_tensor(all_img_object_feats, self.num_proposals)   # (bs, num_query, 256)
        all_img_proposals = align_tensor(all_img_proposals, self.num_proposals)         # (bs, num_query, 5)

        return dict(proposals=all_img_proposals,
                    object_feats=all_img_object_feats)

    def loss_by_feat(self, cls_scores, bbox_preds, gt_bboxes, gt_labels, img_metas):

        flatten_cls_scores = cls_scores
        flatten_bbox_preds = bbox_preds

        cls_reg_targets = self.get_targets(
            flatten_cls_scores,
            flatten_bbox_preds,
            gt_bboxes,
            img_metas,
            gt_labels_list=gt_labels,
        )
        (labels_list, label_weights_list, bbox_targets_list,
         alignment_metrics_list) = cls_reg_targets

        losses_cls, losses_bbox,\
            cls_avg_factors, bbox_avg_factors = multi_apply(
                self.loss_single,
                flatten_cls_scores,
                flatten_bbox_preds,
                labels_list,
                label_weights_list,
                bbox_targets_list,
                alignment_metrics_list,
                )

        cls_avg_factor = reduce_mean(sum(cls_avg_factors)).clamp_(min=1).item()
        losses_cls = list(map(lambda x: x / cls_avg_factor, losses_cls))

        bbox_avg_factor = reduce_mean(
            sum(bbox_avg_factors)).clamp_(min=1).item()
        losses_bbox = list(map(lambda x: x / bbox_avg_factor, losses_bbox))
        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox)

    def loss_single(self, cls_score, bbox_pred, labels, label_weights,
                    bbox_targets, alignment_metrics):

        bbox_targets = bbox_targets.reshape(-1, 5)
        labels = labels.reshape(-1)
        alignment_metrics = alignment_metrics.reshape(-1)
        label_weights = label_weights.reshape(-1)
        targets = (labels, alignment_metrics)
        cls_loss_func = self.loss_cls

        loss_cls = cls_loss_func(cls_score,
                                 targets,
                                 label_weights,
                                 avg_factor=1.0)

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = cls_score.size(-1)
        pos_inds = ((labels >= 0)
                    & (labels < bg_class_ind)).nonzero().squeeze(1)

        if len(pos_inds) > 0:
            pos_bbox_targets = bbox_targets[pos_inds]
            pos_bbox_pred = bbox_pred[pos_inds]

            pos_decode_bbox_pred = pos_bbox_pred
            pos_decode_bbox_targets = pos_bbox_targets

            # regression loss
            pos_bbox_weight = alignment_metrics[pos_inds]

            loss_bbox = self.loss_bbox(pos_decode_bbox_pred,
                                       pos_decode_bbox_targets,
                                       weight=pos_bbox_weight,
                                       avg_factor=1.0)
        else:
            loss_bbox = bbox_pred.sum() * 0
            pos_bbox_weight = bbox_targets.new_tensor(0.)

        return loss_cls, loss_bbox, alignment_metrics.sum(
        ), pos_bbox_weight.sum()


    def get_targets(self,
                    cls_scores,
                    bbox_preds,
                    gt_bboxes_list,
                    img_metas,
                    gt_labels_list=None,
                    **kwargs):

        (all_labels, all_label_weights, all_bbox_targets,
         all_assign_metrics) = multi_apply(self._get_target_single, cls_scores,
                                           bbox_preds, gt_bboxes_list,
                                           gt_labels_list, img_metas)

        return (all_labels, all_label_weights, all_bbox_targets,
                all_assign_metrics)

    def _get_target_single(self, cls_scores, bbox_preds, gt_bboxes, gt_labels,
                           img_meta):
        bbox_preds = get_box_tensor(bbox_preds)
        gt_bboxes = get_box_tensor(gt_bboxes)
        if len(gt_labels) == 0:
            num_valid_anchors = len(cls_scores)
            bbox_targets = torch.zeros_like(bbox_preds)
            labels = bbox_preds.new_full((num_valid_anchors, ),
                                         cls_scores.size(-1),
                                         dtype=torch.long)
            label_weights = bbox_preds.new_zeros(num_valid_anchors,
                                                 dtype=torch.float)
            norm_alignment_metrics = bbox_preds.new_zeros(num_valid_anchors,
                                                          dtype=torch.float)
            return (labels, label_weights, bbox_targets,
                    norm_alignment_metrics)

        assign_result = self.assigner.assign(cls_scores,
                                             bbox_preds, gt_bboxes,
                                             gt_labels, img_meta, 1, 6)
        assign_ious = assign_result.max_overlaps
        assign_metrics = assign_result.assign_metrics

        pred_instances = InstanceData()
        pred_instances.priors = bbox_preds

        gt_instances = InstanceData()
        gt_instances.bboxes = gt_bboxes

        sampling_result = self.sampler.sample(assign_result, pred_instances,
                                              gt_instances)

        num_valid_anchors = len(cls_scores)
        bbox_targets = torch.zeros_like(bbox_preds)
        labels = bbox_preds.new_full((num_valid_anchors, ),
                                     cls_scores.size(-1),
                                     dtype=torch.long)
        label_weights = bbox_preds.new_zeros(num_valid_anchors,
                                             dtype=torch.float)
        norm_alignment_metrics = bbox_preds.new_zeros(num_valid_anchors,
                                                      dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            # point-based
            pos_bbox_targets = sampling_result.pos_gt_bboxes
            bbox_targets[pos_inds, :] = pos_bbox_targets

            if gt_labels is None:
                # Only dense_heads gives gt_labels as None
                # Foreground is the first class since v2.5.0
                labels[pos_inds] = 0
            else:
                labels[pos_inds] = gt_labels[
                    sampling_result.pos_assigned_gt_inds]

            label_weights[pos_inds] = 1.0

        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        class_assigned_gt_inds = torch.unique(
            sampling_result.pos_assigned_gt_inds)
        for gt_inds in class_assigned_gt_inds:
            gt_class_inds = sampling_result.pos_assigned_gt_inds == gt_inds
            pos_alignment_metrics = assign_metrics[gt_class_inds]
            pos_ious = assign_ious[gt_class_inds]
            pos_norm_alignment_metrics = pos_alignment_metrics / (
                pos_alignment_metrics.max() + 10e-8) * pos_ious.max()
            norm_alignment_metrics[
                pos_inds[gt_class_inds]] = pos_norm_alignment_metrics

        return (labels, label_weights, bbox_targets, norm_alignment_metrics)

    def predict(self, x, img_metas, **kwargs):

        loss = dict()

        main_results, aux_results = self.forward(x)
        main_loss_inputs, aux_loss_inputs, \
            distinc_query_dict = self.get_inputs(
            main_results, aux_results, img_metas=img_metas)


        imgs_whwht = []
        for meta in img_metas:
            h, w = meta['img_shape']
            imgs_whwht.append(x[0].new_tensor([[w, h, w, h, 1.]]))
        imgs_whwht = torch.cat(imgs_whwht, dim=0)
        imgs_whwht = imgs_whwht[:, None, :]

        return loss, imgs_whwht, distinc_query_dict

class AuxLoss(nn.Module):
    def __init__(
        self,
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            activated=True,  # use probability instead of logit as input
            beta=2.0,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
        train_cfg=dict(assigner=dict(type='TopkHungarianAssigner', topk=8),
                       alpha=1,
                       beta=6),
    ):
        super(AuxLoss, self).__init__()
        self.train_cfg = train_cfg
        self.loss_cls = MODELS.build(loss_cls)
        self.loss_bbox = MODELS.build(loss_bbox)
        self.assigner = TASK_UTILS.build(self.train_cfg['assigner'])

        sampler_cfg = dict(type='mmdet.PseudoSampler')
        self.sampler = TASK_UTILS.build(sampler_cfg)

    def loss_single(self, cls_score, bbox_pred, labels, label_weights,
                    bbox_targets, alignment_metrics):

        bbox_targets = bbox_targets.reshape(-1, 5)
        labels = labels.reshape(-1)
        alignment_metrics = alignment_metrics.reshape(-1)
        label_weights = label_weights.reshape(-1)
        targets = (labels, alignment_metrics)
        cls_loss_func = self.loss_cls

        loss_cls = cls_loss_func(cls_score,
                                 targets,
                                 label_weights,
                                 avg_factor=1.0)

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = cls_score.size(-1)
        pos_inds = ((labels >= 0)
                    & (labels < bg_class_ind)).nonzero().squeeze(1)

        if len(pos_inds) > 0:
            pos_bbox_targets = bbox_targets[pos_inds]
            pos_bbox_pred = bbox_pred[pos_inds]

            pos_decode_bbox_pred = pos_bbox_pred
            pos_decode_bbox_targets = pos_bbox_targets

            # regression loss
            pos_bbox_weight = alignment_metrics[pos_inds]

            loss_bbox = self.loss_bbox(pos_decode_bbox_pred,
                                       pos_decode_bbox_targets,
                                       weight=pos_bbox_weight,
                                       avg_factor=1.0)
        else:
            loss_bbox = bbox_pred.sum() * 0
            pos_bbox_weight = bbox_targets.new_tensor(0.)

        return loss_cls, loss_bbox, alignment_metrics.sum(
        ), pos_bbox_weight.sum()

    def __call__(self, cls_scores, bbox_preds, gt_bboxes, gt_labels, img_metas,
                 **kwargs):

        flatten_cls_scores = cls_scores
        flatten_bbox_preds = bbox_preds

        cls_reg_targets = self.get_targets(
            flatten_cls_scores,
            flatten_bbox_preds,
            gt_bboxes,
            img_metas,
            gt_labels_list=gt_labels,
        )
        (labels_list, label_weights_list, bbox_targets_list,
         alignment_metrics_list) = cls_reg_targets

        losses_cls, losses_bbox,\
            cls_avg_factors, bbox_avg_factors = multi_apply(
                self.loss_single,
                flatten_cls_scores,
                flatten_bbox_preds,
                labels_list,
                label_weights_list,
                bbox_targets_list,
                alignment_metrics_list,
                )

        cls_avg_factor = reduce_mean(sum(cls_avg_factors)).clamp_(min=1).item()
        losses_cls = list(map(lambda x: x / cls_avg_factor, losses_cls))

        bbox_avg_factor = reduce_mean(
            sum(bbox_avg_factors)).clamp_(min=1).item()
        losses_bbox = list(map(lambda x: x / bbox_avg_factor, losses_bbox))
        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox)

    def get_targets(self,
                    cls_scores,
                    bbox_preds,
                    gt_bboxes_list,
                    img_metas,
                    gt_labels_list=None,
                    **kwargs):

        (all_labels, all_label_weights, all_bbox_targets,
         all_assign_metrics) = multi_apply(self._get_target_single, cls_scores,
                                           bbox_preds, gt_bboxes_list,
                                           gt_labels_list, img_metas)

        return (all_labels, all_label_weights, all_bbox_targets,
                all_assign_metrics)

    def _get_target_single(self, cls_scores, bbox_preds, gt_bboxes, gt_labels,
                           img_meta, **kwargs):
        bbox_preds = get_box_tensor(bbox_preds)
        gt_bboxes = get_box_tensor(gt_bboxes)
        num_gt = len(gt_labels)
        if num_gt == 0:
            num_valid_anchors = len(cls_scores)
            bbox_targets = torch.zeros_like(bbox_preds)
            labels = bbox_preds.new_full((num_valid_anchors, ),
                                         cls_scores.size(-1),
                                         dtype=torch.long)
            label_weights = bbox_preds.new_zeros(num_valid_anchors,
                                                 dtype=torch.float)
            norm_alignment_metrics = bbox_preds.new_zeros(num_valid_anchors,
                                                          dtype=torch.float)

            return (labels, label_weights, bbox_targets,
                    norm_alignment_metrics)

        assign_result = self.assigner.assign(cls_scores,
                                             bbox_preds,
                                             gt_bboxes,
                                             gt_labels, img_meta,
                                             self.train_cfg.get('alpha', 1),
                                             self.train_cfg.get('beta', 6))
        assign_ious = assign_result.max_overlaps
        assign_metrics = assign_result.assign_metrics

        pred_instances = InstanceData()
        pred_instances.priors = bbox_preds

        gt_instances = InstanceData()
        gt_instances.bboxes = gt_bboxes

        sampling_result = self.sampler.sample(assign_result, pred_instances,
                                              gt_instances)

        num_valid_anchors = len(cls_scores)
        bbox_targets = torch.zeros_like(bbox_preds)
        labels = bbox_preds.new_full((num_valid_anchors, ),
                                     cls_scores.size(-1),
                                     dtype=torch.long)
        label_weights = bbox_preds.new_zeros(num_valid_anchors,
                                             dtype=torch.float)
        norm_alignment_metrics = bbox_preds.new_zeros(num_valid_anchors,
                                                      dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            # point-based
            pos_bbox_targets = sampling_result.pos_gt_bboxes
            bbox_targets[pos_inds, :] = pos_bbox_targets

            if gt_labels is None:
                # Only dense_heads gives gt_labels as None
                # Foreground is the first class since v2.5.0
                labels[pos_inds] = 0
            else:
                labels[pos_inds] = gt_labels[
                    sampling_result.pos_assigned_gt_inds]

            label_weights[pos_inds] = 1.0

        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        class_assigned_gt_inds = torch.unique(
            sampling_result.pos_assigned_gt_inds)
        for gt_inds in class_assigned_gt_inds:
            gt_class_inds = sampling_result.pos_assigned_gt_inds == gt_inds
            pos_alignment_metrics = assign_metrics[gt_class_inds]
            pos_ious = assign_ious[gt_class_inds]
            pos_norm_alignment_metrics = pos_alignment_metrics / (
                pos_alignment_metrics.max() + 10e-8) * pos_ious.max()
            norm_alignment_metrics[
                pos_inds[gt_class_inds]] = pos_norm_alignment_metrics

        return (labels, label_weights, bbox_targets, norm_alignment_metrics)

def padding_to(inputs, max=300):
    if max is None:
        return inputs
    num_padding = max - len(inputs)
    if inputs.dim() > 1:
        padding = inputs.new_zeros(num_padding,
                                   *inputs.size()[1:],
                                   dtype=inputs.dtype)
    else:
        padding = inputs.new_zeros(num_padding, dtype=inputs.dtype)
    inputs = torch.cat([inputs, padding], dim=0)
    return inputs


def align_tensor(inputs, max_len=None):
    if max_len is None:
        max_len = max([len(item) for item in inputs])

    return torch.stack([padding_to(item, max_len) for item in inputs])