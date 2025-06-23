from typing import List, Tuple

import torch.nn as nn
from torch import Tensor
from mmdet.models.necks.channel_mapper import ChannelMapper
from mmrotate.registry import MODELS

@MODELS.register_module()
class ChannelMapperWithGN(ChannelMapper):
    def __init__(
        self, in_channels: List[int], out_channels: int, *args, **kwargs,
    ) -> None:
        super().__init__(in_channels, out_channels, *args, **kwargs)
        # add a gn
        self.gn_out = nn.GroupNorm(32, out_channels)

    def forward(self, inputs: Tuple[Tensor]) -> Tuple[Tensor]:
        outs = super().forward(inputs)
        outs = [self.gn_out(out) for out in outs]
        return outs
