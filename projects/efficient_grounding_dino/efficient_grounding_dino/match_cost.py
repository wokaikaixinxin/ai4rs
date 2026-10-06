from typing import Optional, Union

import torch
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.structures.bbox import bbox_overlaps, bbox_xyxy_to_cxcywh
from mmdet.models.task_modules import FocalLossCost


class IoUBinaryFocalLossCost(FocalLossCost):
    def __init__(self, mode='iou',gamma_num=4, **kwargs):
        self.mode = mode
        assert self.mode in ['giou', 'iou'], 'mode should in [giou, iou], but mode get: {}'.format(mode)
        self.gamma_num = gamma_num
        super().__init__(**kwargs)

    def _focal_loss_cost(self, cls_pred: Tensor, gt_labels: Tensor, iou_score: Tensor) -> Tensor:
        """
        Args:
            cls_pred (Tensor): Predicted classification logits, shape
                (num_queries, num_class).
            gt_labels (Tensor): Label of `gt_bboxes`, shape (num_gt,).

        Returns:
            torch.Tensor: cls_cost value with weight
        """
        cls_pred = cls_pred.flatten(1)
        gt_labels = gt_labels.flatten(1).float()
        cls_pred = cls_pred.sigmoid()
        cls_pred = cls_pred.unsqueeze(1) * iou_score.unsqueeze(-1)
        neg_cost = -(1 - cls_pred + self.eps).log() * (
            1 - self.alpha) * cls_pred.pow(self.gamma)
        pos_cost = -(cls_pred + self.eps).log() * self.alpha * (
            1 - cls_pred).pow(self.gamma)

        cls_cost = (neg_cost * (1 - gt_labels[None])).sum(-1) + (pos_cost * gt_labels[None]).sum(-1)
        # cls_cost = torch.einsum('nc,mc->nm', pos_cost, gt_labels) + \
        #     torch.einsum('nc,mc->nm', neg_cost, (1 - gt_labels))
        return cls_cost * self.weight

    def __call__(self,
                 pred_instances: InstanceData,
                 gt_instances: InstanceData,
                 img_meta: Optional[dict] = None,
                 **kwargs) -> Tensor:
        """Compute match cost.

        Args:
            pred_instances (:obj:`InstanceData`): Predicted instances which
                must contain ``scores`` or ``masks``.
            gt_instances (:obj:`InstanceData`): Ground truth which must contain
                ``labels`` or ``mask``.
            img_meta (Optional[dict]): Image information. Defaults to None.

        Returns:
            Tensor: Match Cost matrix of shape (num_preds, num_gts).
        """
        # gt_instances.text_token_mask is a repeated tensor of the same length
        # of instances. Only gt_instances.text_token_mask[0] is useful
        text_token_mask = torch.nonzero(
            gt_instances.text_token_mask[0]).squeeze(-1)
        pred_scores = pred_instances.scores[:, text_token_mask]
        gt_labels = gt_instances.positive_maps[:, text_token_mask]
        pred_bboxes = pred_instances.bboxes
        gt_bboxes = gt_instances.bboxes
        overlaps = bbox_overlaps(
            pred_bboxes, gt_bboxes, mode=self.mode, is_aligned=False)
        if self.mode == 'giou':
            iou_score = ((overlaps + 1) / 2.0).pow(self.gamma_num)
        elif self.mode == 'iou':
            iou_score = overlaps.pow(self.gamma_num)
        return self._focal_loss_cost(pred_scores, gt_labels, iou_score)
