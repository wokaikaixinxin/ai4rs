# Copyright (c) ai4rs. All rights reserved.
import torch
from torch import Tensor
from mmdet.models import SingleStageDetector
from ai4rs.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmdet.structures import OptSampleList
from typing import List, Tuple

@MODELS.register_module()
class DiffusionDet(SingleStageDetector):
    """Implementation of `DiffusionDet <>`_"""

    def __init__(self,
                 backbone: ConfigType,
                 neck: ConfigType,
                 bbox_head: ConfigType,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)

    def _forward(
            self,
            batch_inputs: Tensor,
            batch_data_samples: OptSampleList = None) -> Tuple[List[Tensor]]:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

         Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (list[:obj:`DetDataSample`]): Each item contains
                the meta information of each image and corresponding
                annotations.

        Returns:
            tuple[list]: A tuple of features from ``bbox_head`` forward.
        """
        x = self.extract_feat(batch_inputs)
        for data_samples in batch_data_samples:
            gt_instances = data_samples.gt_instances
            if gt_instances.bboxes.shape[-1] == 4 and 'gt_angles' in gt_instances.keys():
                gt_instances.bboxes = torch.cat([gt_instances.bboxes, gt_instances.gt_angles], dim=-1)
                gt_instances.pop('gt_angles')
        prepare_outputs = self.bbox_head.prepare_training_targets(batch_data_samples)
        (batch_gt_instances, batch_pred_instances, batch_gt_instances_ignore,
         batch_img_metas) = prepare_outputs

        batch_diff_bboxes = torch.stack([
            pred_instances.diff_bboxes_abs
            for pred_instances in batch_pred_instances
        ])
        batch_time = torch.stack(
            [pred_instances.time for pred_instances in batch_pred_instances])

        pred_logits, pred_bboxes, angle_cls = self.bbox_head._bbox_forward(x, batch_diff_bboxes, batch_time)
        results = (pred_logits, pred_bboxes, angle_cls)

        return results
