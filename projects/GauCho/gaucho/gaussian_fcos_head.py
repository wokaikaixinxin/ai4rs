# Copyright (c) ai4rs. All rights reserved.
import copy
from typing import Dict, List

import torch
from torch import Tensor

from mmdet.models.dense_heads import FCOSHead
from mmdet.models.utils import filter_scores_and_topk
from mmdet.structures.bbox import cat_boxes
from mmdet.utils import (InstanceList, OptInstanceList, reduce_mean)
from mmengine import ConfigDict
from mmengine.structures import InstanceData

from ai4rs.registry import MODELS
from ai4rs.structures import RotatedBoxes, norm_angle
from ai4rs.models import RotatedFCOSHead

def decode_gaucho_offset(points, deltas, angle_range, to_obb=False):
    ctr = deltas[..., :2] + points
    cholesky = deltas[..., 2:]
    cov_a = cholesky[..., 0] ** 2
    cov_b = (cholesky[..., 1] ** 2) + (cholesky[..., 2] ** 2)
    cov_c = cholesky[..., 0] * cholesky[..., 2]

    if to_obb:
        delta = torch.sqrt(4 * torch.abs(cov_c).square() + (cov_a - cov_b).square())
        eig1 = 0.5 * (cov_a + cov_b - delta)
        eig2 = 0.5 * (cov_a + cov_b + delta)
        gw = 2 * torch.sqrt(eig2).unsqueeze(-1)
        gh = 2 * torch.sqrt(eig1).unsqueeze(-1)
        gt = torch.atan(cov_c / (eig2 - cov_b)).unsqueeze(-1)
        gt = norm_angle(gt, angle_range)
        return torch.cat([ctr, gw, gh, gt], 1)
    else:
        return torch.cat([ctr, cov_a.unsqueeze(-1), cov_b.unsqueeze(-1), cov_c.unsqueeze(-1)], 1)

@MODELS.register_module()
class GaussianFCOSHead(RotatedFCOSHead):

    def __init__(self,
                 gaucho_encoding=False,
                 kfiou_loss=False,
                 gaussian_centerness=False,
                 **kwargs):

        super().__init__(**kwargs)
        self.gaucho_encoding = gaucho_encoding
        self.kfiou_loss = kfiou_loss
        self.gaussian_centerness = gaussian_centerness
        self.angle_version = self.bbox_coder.angle_version

    def forward_single(self, x, scale, stride):
        """Forward features of a single scale level.

        Args:
            x (Tensor): FPN feature maps of the specified stride.
            scale (:obj: `mmcv.cnn.Scale`): Learnable scale module to resize
                the bbox prediction.
            stride (int): The corresponding stride for feature maps, only
                used to normalize the bbox prediction when self.norm_on_bbox
                is True.
        Returns:
            tuple: scores for each class, bbox predictions, angle predictions \
                and centerness predictions of input feature maps.
        """
        cls_score, bbox_pred, cls_feat, reg_feat = super(
            FCOSHead, self).forward_single(x)
        if self.centerness_on_reg:
            centerness = self.conv_centerness(reg_feat)
        else:
            centerness = self.conv_centerness(cls_feat)
        # scale the bbox_pred of different level
        # float to avoid overflow when enabling FP16
        bbox_pred = scale(bbox_pred).float()

        if self.gaucho_encoding:
            gauss_xy = bbox_pred[:, 0:2] * stride
            gaucho_ab = bbox_pred[:, 2:4].exp() * stride  # cholesky a, b > 0
            gaucho_c = self.conv_angle(reg_feat) * stride
            if self.norm_on_bbox:
                raise NotImplementedError
            bbox_pred = torch.cat([gauss_xy, gaucho_ab, gaucho_c], dim=1)
            angle_pred = torch.zeros_like(centerness)
        else:
            if self.norm_on_bbox:
                # bbox_pred needed for gradient computation has been modified
                # by F.relu(bbox_pred) when run with PyTorch 1.10. So replace
                # F.relu(bbox_pred) with bbox_pred.clamp(min=0)
                bbox_pred = bbox_pred.clamp(min=0)
                if not self.training:
                    bbox_pred *= stride
            else:
                bbox_pred = bbox_pred.exp()
            angle_pred = self.conv_angle(reg_feat)
            if self.is_scale_angle:
                angle_pred = self.scale_angle(angle_pred).float()
        return cls_score, bbox_pred, angle_pred, centerness

    def loss_by_feat(
            self,
            cls_scores: List[Tensor],
            bbox_preds: List[Tensor],
            angle_preds: List[Tensor],
            centernesses: List[Tensor],
            batch_gt_instances: InstanceList,
            batch_img_metas: List[dict],
            batch_gt_instances_ignore: OptInstanceList = None
    ) -> Dict[str, Tensor]:
        """Calculate the loss based on the features extracted by the detection
        head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_points * num_classes.
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_points * 4.
            angle_preds (list[Tensor]): Box angle for each scale level, each \
                is a 4D-tensor, the channel number is num_points * encode_size.
            centernesses (list[Tensor]): centerness for each scale level, each
                is a 4D-tensor, the channel number is num_points * 1.
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance.  It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], Optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert len(cls_scores) == len(bbox_preds) \
               == len(angle_preds) == len(centernesses)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=bbox_preds[0].dtype,
            device=bbox_preds[0].device)
        # bbox_targets here is in format t,b,l,r
        # angle_targets is not coded here
        labels, bbox_targets, angle_targets = self.get_targets(
            all_level_points, batch_gt_instances)

        num_imgs = cls_scores[0].size(0)
        # flatten cls_scores, bbox_preds, angle_preds and centerness
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        if self.gaucho_encoding:
            flatten_bbox_preds = [
                bbox_pred.permute(0, 2, 3, 1).reshape(-1, 5)
                for bbox_pred in bbox_preds
            ]
        else:
            flatten_bbox_preds = [
                bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
                for bbox_pred in bbox_preds
            ]
        angle_dim = self.angle_coder.encode_size
        flatten_angle_preds = [
            angle_pred.permute(0, 2, 3, 1).reshape(-1, angle_dim)
            for angle_pred in angle_preds
        ]
        flatten_centerness = [
            centerness.permute(0, 2, 3, 1).reshape(-1)
            for centerness in centernesses
        ]
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_angle_preds = torch.cat(flatten_angle_preds)
        flatten_centerness = torch.cat(flatten_centerness)
        flatten_labels = torch.cat(labels)
        flatten_bbox_targets = torch.cat(bbox_targets)
        flatten_angle_targets = torch.cat(angle_targets)
        # repeat points to align with bbox_preds
        flatten_points = torch.cat(
            [points.repeat(num_imgs, 1) for points in all_level_points])

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((flatten_labels >= 0)
                    & (flatten_labels < bg_class_ind)).nonzero().reshape(-1)
        num_pos = torch.tensor(
            len(pos_inds), dtype=torch.float, device=bbox_preds[0].device)
        num_pos = max(reduce_mean(num_pos), 1.0)
        loss_cls = self.loss_cls(
            flatten_cls_scores, flatten_labels, avg_factor=num_pos)

        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_angle_preds = flatten_angle_preds[pos_inds]
        pos_centerness = flatten_centerness[pos_inds]
        pos_bbox_targets = flatten_bbox_targets[pos_inds]
        pos_angle_targets = flatten_angle_targets[pos_inds]

        if len(pos_inds) > 0:
            pos_points = flatten_points[pos_inds]
            if self.use_hbbox_loss:
                bbox_coder = self.h_bbox_coder
            else:
                bbox_coder = self.bbox_coder
                pos_decoded_angle_preds = self.angle_coder.decode(
                    pos_angle_preds, keepdim=True)
                pos_bbox_preds = torch.cat(
                    [pos_bbox_preds, pos_decoded_angle_preds], dim=-1)
                pos_bbox_targets = torch.cat(
                    [pos_bbox_targets, pos_angle_targets], dim=-1)

            if self.gaucho_encoding:
                pos_decoded_bbox_preds = decode_gaucho_offset(pos_points, pos_bbox_preds, self.angle_version)
            else:
                pos_decoded_bbox_preds = bbox_coder.decode(pos_points,
                                                           pos_bbox_preds)

            pos_decoded_target_preds = bbox_coder.decode(pos_points, pos_bbox_targets)
            pos_centerness_targets = self.centerness_target(pos_bbox_targets)

            centerness_denorm = max(reduce_mean(pos_centerness_targets.sum().detach()), 1e-6)

            if self.kfiou_loss:
                pos_pred_xy_offset = pos_bbox_preds[..., :2] - pos_points
                pos_target_xy_offset = pos_decoded_target_preds[..., :2] - pos_points
                loss_bbox = self.loss_bbox(
                    pos_pred_xy_offset,
                    pos_target_xy_offset,
                    pred_decode=pos_decoded_bbox_preds,
                    targets_decode=pos_decoded_target_preds,
                    weight=pos_centerness_targets,
                    avg_factor=centerness_denorm)
            else:
                loss_bbox = self.loss_bbox(
                    pos_decoded_bbox_preds,
                    pos_decoded_target_preds,
                    weight=pos_centerness_targets,
                    avg_factor=centerness_denorm)

            if self.loss_angle is not None:
                pos_angle_targets = self.angle_coder.encode(pos_angle_targets)
                loss_angle = self.loss_angle(
                    pos_angle_preds, pos_angle_targets, avg_factor=num_pos)
            loss_centerness = self.loss_centerness(
                pos_centerness, pos_centerness_targets, avg_factor=num_pos)
        else:
            loss_bbox = pos_bbox_preds.sum()
            loss_centerness = pos_centerness.sum()
            if self.loss_angle is not None:
                loss_angle = pos_angle_preds.sum()

        if not self.gaucho_encoding and self.loss_angle:
            return dict(
                loss_cls=loss_cls,
                loss_bbox=loss_bbox,
                loss_angle=loss_angle,
                loss_centerness=loss_centerness)
        else:
            return dict(
                loss_cls=loss_cls,
                loss_bbox=loss_bbox,
                loss_centerness=loss_centerness)

    def _predict_by_feat_single(self,
                                cls_score_list: List[Tensor],
                                bbox_pred_list: List[Tensor],
                                angle_pred_list: List[Tensor],
                                score_factor_list: List[Tensor],
                                mlvl_priors: List[Tensor],
                                img_meta: dict,
                                cfg: ConfigDict,
                                rescale: bool = False,
                                with_nms: bool = True) -> InstanceData:
        """Transform a single image's features extracted from the head into
        bbox results.
        Args:
            cls_score_list (list[Tensor]): Box scores from all scale
                levels of a single image, each item has shape
                (num_priors * num_classes, H, W).
            bbox_pred_list (list[Tensor]): Box energies / deltas from
                all scale levels of a single image, each item has shape
                (num_priors * 4, H, W).
            angle_pred_list (list[Tensor]): Box angle for a single scale
                level with shape (N, num_points * encode_size, H, W).
            score_factor_list (list[Tensor]): Score factor from all scale
                levels of a single image, each item has shape
                (num_priors * 1, H, W).
            mlvl_priors (list[Tensor]): Each element in the list is
                the priors of a single level in feature pyramid. In all
                anchor-based methods, it has shape (num_priors, 4). In
                all anchor-free methods, it has shape (num_priors, 2)
                when `with_stride=True`, otherwise it still has shape
                (num_priors, 4).
            img_meta (dict): Image meta info.
            cfg (mmengine.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.
            with_nms (bool): If True, do nms before return boxes.
                Defaults to True.
        Returns:
            :obj:`InstanceData`: Detection results of each image
            after the post process.
            Each item usually contains following keys.
                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 5),
                  the last dimension 5 arrange as (x, y, w, h, t).
        """
        if score_factor_list[0] is None:
            # e.g. Retina, FreeAnchor, etc.
            with_score_factors = False
        else:
            # e.g. FCOS, PAA, ATSS, etc.
            with_score_factors = True

        cfg = self.test_cfg if cfg is None else cfg
        cfg = copy.deepcopy(cfg)
        img_shape = img_meta['img_shape']
        nms_pre = cfg.get('nms_pre', -1)

        mlvl_bbox_preds = []
        mlvl_valid_priors = []
        mlvl_scores = []
        mlvl_labels = []
        if with_score_factors:
            mlvl_score_factors = []
        else:
            mlvl_score_factors = None
        for level_idx, (
                cls_score, bbox_pred, angle_pred, score_factor, priors) in \
                enumerate(zip(cls_score_list, bbox_pred_list, angle_pred_list,
                              score_factor_list, mlvl_priors)):

            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]

            # dim = self.bbox_coder.encode_size
            if self.gaucho_encoding:
                bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 5)
            else:
                bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            angle_pred = angle_pred.permute(1, 2, 0).reshape(
                -1, self.angle_coder.encode_size)
            if with_score_factors:
                score_factor = score_factor.permute(1, 2,
                                                    0).reshape(-1).sigmoid()
            cls_score = cls_score.permute(1, 2,
                                          0).reshape(-1, self.cls_out_channels)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                # remind that we set FG labels to [0, num_class-1]
                # since mmdet v2.0
                # BG cat_id: num_class
                scores = cls_score.softmax(-1)[:, :-1]

            # After https://github.com/open-mmlab/mmdetection/pull/6268/,
            # this operation keeps fewer bboxes under the same `nms_pre`.
            # There is no difference in performance for most models. If you
            # find a slight drop in performance, you can set a larger
            # `nms_pre` than before.
            score_thr = cfg.get('score_thr', 0)

            results = filter_scores_and_topk(
                scores, score_thr, nms_pre,
                dict(
                    bbox_pred=bbox_pred, angle_pred=angle_pred, priors=priors))
            scores, labels, keep_idxs, filtered_results = results

            bbox_pred = filtered_results['bbox_pred']
            angle_pred = filtered_results['angle_pred']
            priors = filtered_results['priors']

            decoded_angle = self.angle_coder.decode(angle_pred, keepdim=True)
            bbox_pred = torch.cat([bbox_pred, decoded_angle], dim=-1)

            if with_score_factors:
                score_factor = score_factor[keep_idxs]

            mlvl_bbox_preds.append(bbox_pred)
            mlvl_valid_priors.append(priors)
            mlvl_scores.append(scores)
            mlvl_labels.append(labels)

            if with_score_factors:
                mlvl_score_factors.append(score_factor)

        bbox_pred = torch.cat(mlvl_bbox_preds)
        priors = cat_boxes(mlvl_valid_priors)
        if self.gaucho_encoding:
            bboxes = decode_gaucho_offset(priors, bbox_pred, self.angle_version, to_obb=True)
        else:
            bboxes = self.bbox_coder.decode(priors, bbox_pred, max_shape=img_shape)

        results = InstanceData()
        results.bboxes = RotatedBoxes(bboxes)
        results.scores = torch.cat(mlvl_scores)
        results.labels = torch.cat(mlvl_labels)
        if with_score_factors:
            results.score_factors = torch.cat(mlvl_score_factors)

        return self._bbox_post_process(
            results=results,
            cfg=cfg,
            rescale=rescale,
            with_nms=with_nms,
            img_meta=img_meta)