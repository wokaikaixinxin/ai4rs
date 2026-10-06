# Copyright (c) OpenMMLab. All rights reserved.
from typing import Sequence

import torch
from mmengine.evaluator import BaseMetric

import numpy as np

def fp16_clamp(x, min=None, max=None):
    if not x.is_cuda and x.dtype == torch.float16:
        # clamp for cpu float16, tensor fp16 has no clamp implementation
        return x.float().clamp(min, max).half()

    return x.clamp(min, max)


class RSVGMetric_top1(BaseMetric):
    """Referring Expression Metric."""

    def __init__(self, metric: Sequence = ('Pr@0.5', 'Pr@0.6', 'Pr@0.7', 'Pr@0.8', 'Pr@0.9', 'meanIoU', 'cumIoU'), **kwargs):
        super().__init__(**kwargs)
        assert set(metric).issubset(['Pr@0.5', 'Pr@0.6', 'Pr@0.7', 'Pr@0.8', 'Pr@0.9', 'meanIoU', 'cumIoU']), \
            f'Only support cIoU, mIoU, Pr@0.5, Pr@0.6, Pr@0.7, Pr@0.8, Pr@0.9, but got {metric}'
        assert len(metric) > 0, 'metrics should not be empty'
        self.metrics = metric

    def compute_iou(self, pred_bbox: torch.Tensor,
                    gt_bbox: torch.Tensor) -> tuple:

        area1 = (pred_bbox[..., 2] - pred_bbox[..., 0]) * (
                pred_bbox[..., 3] - pred_bbox[..., 1])
        area2 = (gt_bbox[..., 2] - gt_bbox[..., 0]) * (
                gt_bbox[..., 3] - gt_bbox[..., 1])

        lt = torch.max(pred_bbox[..., :, None, :2],
                       gt_bbox[..., None, :, :2])  # [B, rows, cols, 2]
        rb = torch.min(pred_bbox[..., :, None, 2:],
                       gt_bbox[..., None, :, 2:])  # [B, rows, cols, 2]

        wh = fp16_clamp(rb - lt, min=0)
        overlap = wh[..., 0] * wh[..., 1]

        union = area1[..., None] + area2[..., None, :] - overlap
        return overlap, union

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data and data_samples.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        for data_sample in data_samples:
            pred_label = data_sample['pred_instances']['bboxes']
            pred_score = data_sample['pred_instances']['scores']
            # pred_label = pred_label[pred_score.sort(descending=True)[1]]
            pred_label = pred_label[pred_score.sort(descending=True)[1][0]][None]
            # label = data_sample['gt_masks'].to_tensor(
            #     pred_label.dtype, pred_label.device).bool()
            label = data_sample['gt_instances']['bboxes']
            # calculate iou
            overlap, union = self.compute_iou(pred_label, label)

            bs = len(pred_label)
            # bs = len(pred_label)
            gt_num = len(label)
            Pr5, Pr6, Pr7, Pr8, Pr9 = 0, 0, 0, 0, 0

            if overlap.shape[0] == 0:
                return self.results.append((overlap.sum(), union.sum(), torch.tensor(0, device=label.device), gt_num,
                                            Pr5, Pr6, Pr7, Pr8, Pr9))
            iou = overlap.reshape(bs, -1).sum(-1) * 1.0 / union.reshape(
                bs, -1).sum(-1)
            iou = torch.nan_to_num_(iou, nan=0.0)
            overlap = overlap[iou.sort(descending=True)[1]][0]
            union = union[iou.sort(descending=True)[1]][0]

            if iou.shape[0] > 0:
                iou_max = iou[0].max()
                # Pr@0.5
                if iou_max > 0.5:
                    Pr5 += 1
                # Pr@0.6
                if iou_max > 0.6:
                    Pr6 += 1
                # Pr@0.7
                if iou_max > 0.7:
                    Pr7 += 1
                # Pr@0.8
                if iou_max > 0.8:
                    Pr8 += 1
                # Pr@0.9
                if iou_max > 0.9:
                    Pr9 += 1

            self.results.append((overlap.sum().item(), union.sum().item(), iou.sum().item(),
                                 gt_num, Pr5, Pr6, Pr7, Pr8, Pr9))

    def compute_metrics(self, results: list) -> dict:
        results = tuple(zip(*results))
        # assert len(results) == 10
        cum_i = np.array(results[0])
        cum_u = np.array(results[1])
        iou = np.array(results[2])
        cum_gt_total = sum(results[3])
        cum_Pr5 = sum(results[4])
        cum_Pr6 = sum(results[5])
        cum_Pr7 = sum(results[6])
        cum_Pr8 = sum(results[7])
        cum_Pr9 = sum(results[8])


        metrics = {}

        if 'Pr@0.5' in self.metrics:
            metrics['Pr1@0.5'] = cum_Pr5 * 100 / cum_gt_total
        if 'Pr@0.6' in self.metrics:
            metrics['Pr1@0.6'] = cum_Pr6 * 100 / cum_gt_total
        if 'Pr@0.7' in self.metrics:
            metrics['Pr1@0.7'] = cum_Pr7 * 100 / cum_gt_total
        if 'Pr@0.8' in self.metrics:
            metrics['Pr1@0.8'] = cum_Pr8 * 100 / cum_gt_total
        if 'Pr@0.9' in self.metrics:
            metrics['Pr1@0.9'] = cum_Pr9 * 100 / cum_gt_total
        if 'meanIoU' in self.metrics:
            metrics['meanIoU'] = iou.mean() * 100
        if 'cumIoU' in self.metrics:
            metrics['cumIoU'] = cum_i.sum() / cum_u.sum() * 100
        return metrics
