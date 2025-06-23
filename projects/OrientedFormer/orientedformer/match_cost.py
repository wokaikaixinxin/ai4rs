
from typing import Optional, Union
import torch
from mmengine.structures import InstanceData
from torch import Tensor

from ai4rs.registry import TASK_UTILS
from mmcv.ops import box_iou_rotated
from mmdet.models.task_modules.assigners.match_cost import BaseMatchCost
import math


@TASK_UTILS.register_module()
class RBBoxL1Cost(BaseMatchCost):
    def __init__(self,
                 box_format: str = 'xywht',
                 weight: Union[float, int] = 1.,
                 angle_version='le90',) -> None:
        super().__init__(weight=weight)
        assert box_format in ['xywht']
        assert angle_version in ['oc', 'le135', 'le90']
        self.box_format = box_format
        self.angle_version = angle_version
    def __call__(self,
                 pred_instances: InstanceData,
                 gt_instances: InstanceData,
                 img_meta: Optional[dict] = None,
                 **kwargs) -> Tensor:
        """
        Args:
            pred_instances (InstanceData):
                pred_instances.bboxes (Tensor):
                Predicted boxes with unnormalized coordinates
                (cx, cy, w, h, radian), which are all in range [0, 1].
                Shape [num_query, 5].
            gt_instances (InstanceData): Ground truth boxes with unnormalized
                coordinates (cx, cy, w, h, radian). Shape [num_gt, 5].

        Returns:
            torch.Tensor: bbox_cost value with weight
        """
        pred_bboxes = pred_instances.bboxes
        gt_bboxes = gt_instances.bboxes
        # when normalized, the gt_bboxes and pred_bboxes will change, because of pointer
        pred_bboxes_ = torch.empty_like(pred_bboxes)
        gt_bboxes_ = torch.empty_like(gt_bboxes)

        # normalized
        img_h, img_w = img_meta['img_shape']
        factor = gt_bboxes.new_tensor([img_w, img_h, img_w,
                                        img_h]).unsqueeze(0)
        gt_bboxes_[..., :-1] = gt_bboxes[..., :-1] / factor
        pred_bboxes_[..., :-1] = pred_bboxes[..., :-1] / factor
        gt_bboxes_[..., -1] = normalize_angle(gt_bboxes[..., -1], self.angle_version)
        pred_bboxes_[..., -1] = normalize_angle(pred_bboxes[..., -1], self.angle_version)

        bbox_cost = torch.cdist(pred_bboxes_, gt_bboxes_, p=1)
        return bbox_cost * self.weight

@TASK_UTILS.register_module()
class RotatedIoUCost(BaseMatchCost):
    def __init__(self, iou_mode: str = 'iou', weight: Union[float, int] = 1.):
        super().__init__(weight=weight)
        assert iou_mode in ['iou', 'iof']
        self.iou_mode = iou_mode
    def __call__(self,
                 pred_instances: InstanceData,
                 gt_instances: InstanceData,
                 img_meta: Optional[dict] = None,
                 **kwargs):
        pred_bboxes = pred_instances.bboxes
        gt_bboxes = gt_instances.bboxes
        overlaps = box_iou_rotated(pred_bboxes, gt_bboxes, mode=self.iou_mode, aligned=False)
        iou_cost = -overlaps
        return iou_cost * self.weight


def normalize_angle(angle, angle_version):
    '''
    Normalize the angle from radians to [0,1] with Min-Max Normalization.
    Min-Max Normalization formulation with variable x: x' = (x - x_min) / (x_max - x_min).

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!NOTICE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !!!!!!!!!!!!!!!!the defination of angle in 1.x mmrotate is different from 0.x!!!!!!!!!!!!!!!!!!
    !!!!!!!!!!!!!!!!!!!!!! refer to mmrotate/structures/bbox/rotated_boxes.py!!!!!!!!!!!!!!!!!!!!!!

     For convenience, three commonly used patterns are preset in
        ``regualrize_boxes``:

        - 'oc': OpenCV Definition. Has the same box representation as
          ``cv2.minAreaRect`` the angle ranges in [-pi/2, 0). Equal to set
          width_longer=False and start_angle=-90.
        - 'le90': Long Edge Definition (90). the angle ranges in [-pi/2, pi/2).
          The width is always longer than the height. Equal to set
          width_longer=True and start_angle=-90.
        - 'le135': Long Edge Definition (135). the angle ranges in [-pi/4, 3pi/4).
          The width is always longer than the height. Equal to set
          width_longer=True and start_angle=-45.
    Args:
        angle (Tensor): the angle of obbox. the type of angle is radians.
        angle_version (Str): 3 types of angle representations: 'oc', 'le90', 'le135',
            'oc' range [-pi/2, 0], 'le90' range [-pi/2, pi/2], 'le135' range [-pi/4, 3pi/4].
    Returns:
        angle (Tensor): the normalized angles [0, 1].
    '''
    if  isinstance(angle, torch.Tensor):
        if angle_version == 'oc':
            return (angle + math.pi / 2) / (math.pi / 2)
        elif angle_version == 'le90':
            return (angle + math.pi / 2) / math.pi
        elif angle_version == 'le135':
            return (angle + math.pi / 4) / math.pi
        else:
            raise NotImplementedError(f'The angle version {angle_version} not implement!')

