# Copyright (c) ai4rs. All rights reserved.
# /ai4rs/structures/bbox/transforms.py
from typing import Tuple, Union
from torch import Tensor
from mmdet.structures.bbox import BaseBoxes


def scale_rboxes(rboxes: Union[Tensor, BaseBoxes],
                 scale_factor: Tuple[float, float]) -> Union[Tensor, BaseBoxes]:
    """Scale rboxes with type of tensor or box type.
    Notes: Scale rboxes, but the angles remain the same.

    Args:
        rboxes (Tensor or :obj:`BaseBoxes`): boxes need to be scaled. Its type
            can be a tensor or a box type. [n, 5]
        scale_factor (Tuple[float, float]): factors for scaling boxes.
            The length should be 2.

    Returns:
        Union[Tensor, :obj:`BaseBoxes`]: Scaled boxes. [n, 5]
    """
    if isinstance(rboxes, BaseBoxes):
        rboxes[..., :-1].rescale_(scale_factor)
        assert rboxes.shape[-1] == 5
        return rboxes
    else:
        # Tensor boxes will be treated as horizontal boxes
        repeat_num = int(rboxes[..., :-1].size(-1) / 2)
        scale_factor = rboxes[..., :-1].new_tensor(scale_factor).repeat((1, repeat_num))
        rboxes[..., :-1] = rboxes[..., :-1] * scale_factor
        return rboxes


def get_rbox_wh(rboxes: Union[Tensor, BaseBoxes]) -> Tuple[Tensor, Tensor]:
    """Get the width and height of rboxes with type of tensor or box type.

    Args:
        boxes (Tensor or :obj:`BaseBoxes`): boxes with type of tensor
            or box type.

    Returns:
        Tuple[Tensor, Tensor]: the width and height of boxes.
    """
    if isinstance(rboxes, BaseBoxes):
        w = rboxes.widths
        h = rboxes.heights
    else:
        # Tensor boxes will be treated as horizontal boxes by defaults
        w = rboxes[:, 2]
        h = rboxes[:, 3]
    return w, h

