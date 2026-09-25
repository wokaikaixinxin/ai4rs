from typing import Sequence
import numpy as np
import torch
from mmcv.ops import box_iou_rotated
from mmengine.evaluator import BaseMetric


class RSVGMetric_top1(BaseMetric):
    """Top-1 referring expression metric for rotated boxes."""

    def __init__(self, metric: Sequence = ('Pr@0.5', 'Pr@0.6', 'Pr@0.7', 'Pr@0.8', 'Pr@0.9', 'meanIoU', 'cumIoU'), **kwargs):
        super().__init__(**kwargs)
        assert set(metric).issubset(['Pr@0.5', 'Pr@0.6', 'Pr@0.7', 'Pr@0.8', 'Pr@0.9', 'meanIoU', 'cumIoU']), \
            f'Only support meanIoU, cumIoU, Pr@0.5, Pr@0.6, Pr@0.7, Pr@0.8, Pr@0.9, but got {metric}'
        assert len(metric) > 0, 'metrics should not be empty'
        self.metrics = metric

    def _to_rbox_tensor(self, bboxes) -> torch.Tensor:
        if hasattr(bboxes, 'tensor'):
            bboxes = bboxes.tensor
        assert isinstance(bboxes, torch.Tensor), \
            f'Only support rotated box tensor, but got {type(bboxes)}'
        assert bboxes.size(-1) == 5, \
            f'Only support rotated boxes in xywha format, but got {bboxes.shape}'
        return bboxes

    def compute_iou(self, pred_bbox: torch.Tensor,
                    gt_bbox: torch.Tensor) -> tuple:
        """Compute pairwise rotated intersections and unions."""
        pred_bbox = self._to_rbox_tensor(pred_bbox)
        gt_bbox = self._to_rbox_tensor(gt_bbox)

        ious = box_iou_rotated(pred_bbox.float(), gt_bbox.float())
        area1 = pred_bbox[..., 2] * pred_bbox[..., 3]
        area2 = gt_bbox[..., 2] * gt_bbox[..., 3]
        union = (area1[:, None] + area2[None, :]) / (1 + ious).clamp(min=1e-7)
        overlap = ious * union
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
            pred_bboxes = self._to_rbox_tensor(
                data_sample['pred_instances']['bboxes'])
            pred_score = data_sample['pred_instances']['scores']
            label = self._to_rbox_tensor(data_sample['gt_instances']['bboxes'])

            gt_num = len(label)
            Pr5, Pr6, Pr7, Pr8, Pr9 = 0, 0, 0, 0, 0

            if len(pred_bboxes) == 0 or gt_num == 0:
                self.results.append((0., 0., 0., gt_num,
                                     Pr5, Pr6, Pr7, Pr8, Pr9))
                continue

            top1_idx = pred_score.sort(descending=True)[1][0]
            pred_bbox = pred_bboxes[top1_idx][None]
            overlap, union = self.compute_iou(pred_bbox, label)

            iou = overlap.sum() / union.sum().clamp(min=1e-7)
            iou = torch.nan_to_num(iou, nan=0.0)

            if iou > 0.5:
                Pr5 += 1
            if iou > 0.6:
                Pr6 += 1
            if iou > 0.7:
                Pr7 += 1
            if iou > 0.8:
                Pr8 += 1
            if iou > 0.9:
                Pr9 += 1

            self.results.append((overlap.sum().item(), union.sum().item(), iou.item(),
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