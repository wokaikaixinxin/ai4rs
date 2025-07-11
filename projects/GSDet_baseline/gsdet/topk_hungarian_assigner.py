# Copyright (c) ai4rs. All rights reserved.
import torch
from scipy.optimize import linear_sum_assignment
from mmengine.structures import BaseDataElement
from mmdet.models.task_modules import AssignResult, TaskAlignedAssigner
from ai4rs.structures.bbox import rbox2hbox
from ai4rs.registry import TASK_UTILS

@TASK_UTILS.register_module()
class TopkHungarianAssigner(TaskAlignedAssigner):
    """Computes 1-to-k matching between ground truth and predictions.

    This class computes an assignment between the targets and the predictions
    based on the costs. The costs are weighted sum of some components.
    For DETR the costs are weighted sum of classification cost, regression L1
    cost and regression iou cost. The targets don't include the no_object, so
    generally there are more predictions than targets. After the 1-to-k
    gt-pred matching, the un-matched are treated as backgrounds. Thus each
    query prediction will be assigned with `0` or a positive integer
    indicating the ground truth index:

    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt

    Args:
        cls_cost (dict): Classification cost configuration.
        reg_cost (dict): Regression L1  cost configuration.
        iou_cost (dict): Regression iou cost configuration.
    """

    def __init__(self,
                 cls_cost=dict(type='FocalLossCost', weight=2.0),
                 reg_cost=dict(type='BBoxL1Cost', weight=5.0),
                 iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0),
                 **kwargs):
        super(TopkHungarianAssigner, self).__init__(
            iou_calculator= dict(type='RBboxOverlaps2D'),
            **kwargs)

        self.cls_cost = TASK_UTILS.build(cls_cost)
        self.reg_cost = TASK_UTILS.build(reg_cost)
        self.iou_cost = TASK_UTILS.build(iou_cost)

    def assign(self,
               pred_instances,
               gt_instances,
               img_meta,
               **kwargs):
        """Computes 1-to-k gt-pred matching based on the weighted costs.

        This method assign each query prediction to a ground truth or
        background. The `assigned_gt_inds` with -1 means don't care,
        0 means negative sample, and positive number is the index (1-based)
        of assigned gt.
        The assignment is done in the following steps, the order matters.

        1. Assign every prediction to -1.
        2. Compute the weighted costs, each cost has shape (num_pred, num_gt).
        3. Update topk to be min(topk, int(num_pred / num_gt)), then repeat
            costs topk times to shape: (num_pred, num_gt * topk), so that each
            gt will match topk predictions.
        3. Do Hungarian matching on CPU based on the costs.
        4. Assign all to 0 (background) first, then for each matched pair
           between predictions and gts, treat this prediction as foreground
           and assign the corresponding gt index (plus 1) to it.
        5. Calculate alignment metrics and overlaps of each matched pred-gt
            pair.

        Args:
            pred_scores (Tensor): Predicted normalized classification
                scores for one image, has shape (num_dense_queries,
                cls_out_channels).
            decode_bboxes (Tensor): Predicted unnormalized bbox coordinates
                for one image, has shape (num_dense_queries, 4) with the
                last dimension arranged as (x1, y1, x2, y2).
            gt_bboxes (Tensor): Unnormalized ground truth
                bboxes for one image, has shape (num_gt, 4) with the
                last dimension arranged as (x1, y1, x2, y2).
                NOTE: num_gt is dynamic for each image.
            gt_labels (Tensor): Ground truth classification
                    index for the image, has shape (num_gt,).
                    NOTE: num_gt is dynamic for each image.
            img_meta (dict): Meta information for one image.
            alpha (int): Hyper-parameters related to alignment_metrics.
                Defaults to 1.
            beta (int): Hyper-parameters related to alignment_metrics.
                Defaults to 6.

        Returns:
            :obj:`AssignResult`: The assigned result.
        """
        pred_scores = pred_instances.scores
        decode_bboxes = pred_instances.bboxes
        gt_bboxes = gt_instances.bboxes
        gt_labels = gt_instances.labels
        if decode_bboxes.shape[-1] == 4:
            gt_bboxes = rbox2hbox(gt_bboxes)


        pred_instances = BaseDataElement()
        gt_instances = BaseDataElement()

        pred_instances.bboxes = decode_bboxes
        gt_instances.bboxes = gt_bboxes

        pred_instances.scores = pred_scores
        gt_instances.labels = gt_labels

        reg_cost = self.reg_cost(pred_instances, gt_instances, img_meta)
        iou_cost = self.iou_cost(pred_instances, gt_instances, img_meta)
        cls_cost = self.cls_cost(pred_instances, gt_instances, img_meta)
        all_cost = cls_cost + reg_cost + iou_cost

        num_gt, num_bboxes = gt_bboxes.size(0), pred_scores.size(0)
        if num_gt > 0:
            # 1. assign 0 by default
            assigned_gt_inds = pred_scores.new_full((num_bboxes, ),
                                                    0,
                                                    dtype=torch.long)
            assigned_labels = pred_scores.new_full((num_bboxes, ),
                                                   -1,
                                                   dtype=torch.long)

            # 2. compute weighted cost
            select_cost = all_cost

            # topk = min(self.topk, int(len(select_cost) / num_gt))

            # Repeat the ground truth `topk` times to perform 1-to-k gt-pred
            #   matching. For example, if `num_pred` = 900, `num_gt` = 3, then
            #   there are only 3 gt-pred pairs in sum for 1-1 matching.
            #   However, for 1-k gt-pred matching, if `topk` = 4, then each
            #   gt is assigned 4 unique predictions, so there would be 12
            #   gt-pred pairs in sum.
            repeat_select_cost = select_cost[...,
                                             None].repeat(1, 1, self.topk).view(
                                                 select_cost.size(0), -1)

            # 3. do Hungarian matching on CPU using linear_sum_assignment
            # anchor index and gt index
            matched_row_inds, matched_col_inds = linear_sum_assignment(
                repeat_select_cost.detach().cpu().numpy())
            matched_row_inds = torch.from_numpy(matched_row_inds).to(
                pred_scores.device)
            matched_col_inds = torch.from_numpy(matched_col_inds).to(
                pred_scores.device)

            match_gt_ids = matched_col_inds // self.topk
            candidate_idxs = matched_row_inds

            # 4. assign backgrounds and foregrounds
            if candidate_idxs.numel() > 0:
                assigned_labels[candidate_idxs] = gt_labels[match_gt_ids]
            else:
                assigned_labels = None

            # why plus 1 ? Because the `assigned_gt_inds` with 0 means negative sample.
            # Assign foregrounds based on matching results.
            assigned_gt_inds[candidate_idxs] = match_gt_ids + 1

            assign_result = AssignResult(
                num_gt, assigned_gt_inds, max_overlaps=None, labels=assigned_labels)
            return assign_result
        else:

            assigned_gt_inds = pred_scores.new_full((num_bboxes, ),
                                                    -1,
                                                    dtype=torch.long)

            assigned_labels = pred_scores.new_full((num_bboxes, ),
                                                   -1,
                                                   dtype=torch.long)

            assigned_gt_inds[:] = 0
            return AssignResult(
                0, assigned_gt_inds, max_overlaps=None, labels=assigned_labels)
