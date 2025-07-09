# Copyright (c) ai4rs. All rights reserved.
from mmcv.transforms import BaseTransform
from ai4rs.registry import TRANSFORMS

@TRANSFORMS.register_module()
class RBox2Point(BaseTransform):
    """Convert RBoxes to Single Center Points."""

    def __init__(self) -> None:
        pass

    def transform(self, results: dict) -> dict:
        """The transform function."""

        results['gt_bboxes'].tensor[:, 2] = 0.1
        results['gt_bboxes'].tensor[:, 3] = 0.1
        results['gt_bboxes'].tensor[:, 4] = 0

        return results