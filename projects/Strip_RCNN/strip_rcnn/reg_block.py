# Copyright (c) ai4rs. All rights reserved.
import torch.nn as nn

class StripBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.strip_conv1 = nn.Conv2d(dim,dim,kernel_size=(1, 19), stride=1, padding=(0, 9), groups=dim)
        self.strip_conv2 = nn.Conv2d(dim,dim,kernel_size=(19, 1), stride=1, padding=(9, 0), groups=dim)
        self.conv1 = nn.Conv2d(dim, dim, 1)


    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.strip_conv1(attn)
        attn = self.strip_conv2(attn)
        attn = self.conv1(attn)

        return u * attn