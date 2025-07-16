# Copyright (c) ai4rs. All rights reserved.
import torch
from mmcv.transforms import to_tensor, BaseTransform
from ai4rs.registry import TRANSFORMS

@TRANSFORMS.register_module()
class ConvertWeakSupervision(BaseTransform):
    """Convert RBoxes to Single Center Points."""

    def __init__(self,
                 point_proportion: float = 0.3,
                 hbox_proportion: float = 0.3,
                 point_dummy: float = 1,
                 hbox_dummy: float = 0,
                 modify_labels: bool = False) -> None:
        self.point_proportion = point_proportion
        self.hbox_proportion = hbox_proportion
        self.point_dummy = point_dummy
        self.hbox_dummy = hbox_dummy
        self.modify_labels = modify_labels

    def transform(self, results: dict) -> dict:
        """The transform function."""

        max_idx_p = int(round(results['gt_bboxes'].tensor.shape[0] * self.point_proportion))
        results['gt_bboxes'].tensor[:max_idx_p, 2] = self.point_dummy
        results['gt_bboxes'].tensor[:max_idx_p, 3] = self.point_dummy
        results['gt_bboxes'].tensor[:max_idx_p, 4] = 0

        max_idx_h = max_idx_p + int(round(results['gt_bboxes'].tensor.shape[0] * self.hbox_proportion))
        results['gt_bboxes'][max_idx_p:max_idx_h] = \
            results['gt_bboxes'][max_idx_p:max_idx_h].convert_to('hbox').convert_to('rbox')
        results['gt_bboxes'].tensor[max_idx_p:max_idx_h, 4] = self.hbox_dummy

        if self.modify_labels:
            ws_types = torch.zeros(results['gt_bboxes'].tensor.shape[0], dtype=torch.long)
            ws_types[:max_idx_p] = 2
            ws_types[max_idx_p:max_idx_h] = 1
            labels = to_tensor(results['gt_bboxes_labels'])
            results['gt_bboxes_labels'] = torch.stack((labels, ws_types), -1)

        return results