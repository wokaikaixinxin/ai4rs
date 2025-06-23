import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from ai4rs.registry import MODELS


@MODELS.register_module()
class OrientedAttention(nn.Module):
    def __init__(self,
                 n_points: int = 32,
                 n_heads: int = 4,
                 embed_dims: int = 256,
                 reduction: int = 4):
        super(OrientedAttention, self).__init__()
        self.n_points = n_points
        self.n_heads = n_heads
        self.embed_dims = embed_dims

        self.sampling_offset_generator = nn.Sequential(
            nn.Linear(embed_dims, n_points * n_heads * 3))

        # channel attention
        self.ca_attn = nn.Sequential(
            nn.Linear(embed_dims, embed_dims//reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims//reduction, embed_dims, bias=False),
            nn.Sigmoid())

        # spatial attention
        self.sa_attn = nn.Sequential(
            nn.Linear(embed_dims, embed_dims // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims // reduction, n_heads*n_points, bias=False),
            nn.Sigmoid())

        self.out_proj = nn.Linear(embed_dims*n_points, embed_dims)

        self.init_weights()

    @torch.no_grad()
    def init_weights(self):
        nn.init.zeros_(self.sampling_offset_generator[-1].weight)
        nn.init.zeros_(self.sampling_offset_generator[-1].bias)

        bias = self.sampling_offset_generator[-1].bias.data.view(
            self.n_heads, self.n_points, 3)

        # if n_points are squared number, then initialize
        # to sampling on grids regularly, not used in most
        # of our experiments.
        if int(self.n_points ** 0.5) ** 2 == self.n_points:
            h = int(self.n_points ** 0.5)
            y = torch.linspace(-0.5, 0.5, h + 1) + 0.5 / h
            yp = y[:-1]
            y = yp.view(-1, 1).repeat(1, h)
            x = yp.view(1, -1).repeat(h, 1)
            y = y.flatten(0, 1)[None, :, None]
            x = x.flatten(0, 1)[None, :, None]
            bias[:, :, 0:2] = torch.cat([y, x], dim=-1)
        else:
            bandwidth = 0.5 * 1.0
            nn.init.uniform_(bias, -bandwidth, bandwidth)

        # initialize sampling delta z
        nn.init.constant_(bias[:, :, 2:3], -1.0)


        nn.init.zeros_(self.ca_attn[0].weight)
        nn.init.zeros_(self.ca_attn[2].weight)
        nn.init.zeros_(self.sa_attn[0].weight)
        nn.init.zeros_(self.sa_attn[2].weight)


    def forward(self, value: list, query_content: Tensor, query_xyzrt: Tensor,
                featmap_strides: list):
        '''
        Args:
            value (list): List of multi-level img features.
                        Each level feature has shape (bs, c, h, w).
            query_content (Tensor): (bs, num_query, 256)
            query_xyzrt (Tensor): (bs, num_query, 5)
            spatial_shapes (torch.Tensor): Spatial shape of features in
                different levels. With shape (num_levels, 2),
                last dimension represents (h, w).
            featmap_strides (list): [4, 8, 16, 32, 64]
        Return:
        '''
        identity = query_content
        bs, num_query, _ = query_content.shape

        # compute offset and sampling points (x', y', z')
        offset = self.sampling_offset_generator(query_content)  # (bs, num_query, points*head*3)
        sample_points_xyz = make_sample_points(
            offset, self.n_heads * self.n_points,
            query_xyzrt)                                        # (bs, num_query, 1, points*head, 3)
        sample_points_xy = sample_points_xyz[..., 0:2]          # (bs, num_query, 1, points*head, 2)
        sample_points_z = sample_points_xyz[..., 2].clone()     # (bs, num_query, 1, points*head)

        # scale-aware attention
        # weight of scale
        sample_points_lvl_weight = translate_to_linear_weight(
            sample_points_z, num_levels=len(featmap_strides),
            featmap_strides=featmap_strides)                    # (bs, num_query, 1, points*head, level)
        sample_points_lvl_weight_list = sample_points_lvl_weight.unbind(
            -1)                                                 # list[(bs, num_query, 1, points*head) * level]
        # weight of level * feature
        sampling_value = value[0].new_zeros(
            bs, num_query, self.n_heads, self.n_points, self.embed_dims//self.n_heads)
        for level, value_l_ in enumerate(value):
            lvl_weights = sample_points_lvl_weight_list[level]  # (bs, num_query, 1, points*head)
            stride = featmap_strides[level]
            mapping_size = value_l_.new_tensor(
                [value_l_.size(3), value_l_.size(2)]).view(1, 1, 1, 1, -1) * stride  # (1, 1, 1, 1, 2)
            normalized_xy = sample_points_xy / mapping_size     # (bs, num_query, 1, points*head, 2)
            sampling_value += sampling_each_level(normalized_xy, value_l_,
                                weight=lvl_weights,
                                n_points=self.n_points)        # (bs, num_query, head, points, embed_dim//head)

        # channel-aware attention
        ca_weights = self.ca_attn(query_content).view(
            bs, num_query, self.n_heads, 1, self.embed_dims // self.n_heads)
        output = sampling_value * ca_weights    # (bs, num_query, head, points, embed_dim//head)

        # spatial-aware attention
        sa_weight = self.sa_attn(query_content).view(
            bs, num_query, self.n_heads, self.n_points, 1)
        output = output * sa_weight             # (bs, num_query, head, points, embed_dim//head)

        output = self.out_proj(output.reshape(bs, num_query, -1))
        return output + identity


def make_sample_points(offset: Tensor, num_points: int, xyzrt: Tensor):
    '''
        Args
            offset (Tensor): (bs, num_query, num_points*3), normalized by stride
            num_points (int): 128.
            xyzrt (Tensor): (bs, num_query, 5)
        Returns:
            [B, num_query, 1, num_points, 3]
    '''
    B, L, _ = offset.shape

    offset = offset.view(B, L, 1, num_points, 3)                 # (bs, num_query, 1, num_points, 3)

    roi_cc = xyzrt[..., :2]                                     # (bs, num_query, 2)
    scale = 2.00 ** xyzrt[..., 2:3]                             # (bs, num_query, 4)
    ratio = 2.00 ** torch.cat([xyzrt[..., 3:4] * -0.5,
                               xyzrt[..., 3:4] * 0.5], dim=-1)
    roi_wh = scale * ratio                                      # (bs, num_query, 2)

    theta = xyzrt[..., 4:]                                      # (bs, num_query, 1)
    cos_theta = torch.cos(theta)                                # (bs, num_query, 1)
    sin_theta = torch.sin(theta)                                # (bs, num_query, 1)
    rotation_part1 = torch.cat((cos_theta, -sin_theta), dim=-1) # (bs, num_query, 2)
    rotation_part2 = torch.cat((sin_theta,  cos_theta), dim=-1) # (bs, num_query, 2)
    rotation_matrix = torch.stack((rotation_part1, rotation_part2), dim=-2) # (bs, num_query, 2, 2)

    offset_yx = offset[..., :2] * roi_wh.view(B, L, 1, 1, 2)    # (bs, num_query, 1, num_points, 2)
    offset_yx = torch.matmul(offset_yx, rotation_matrix[:, :, None, :, :].transpose(-2, -1)) # (bs, num_query, 1, num_points, 2)
    sample_yx = roi_cc.contiguous().view(B, L, 1, 1, 2) + offset_yx  # (bs, num_query, 1, num_points, 2)

    roi_lvl = xyzrt[..., 2:3].view(B, L, 1, 1, 1)               # (bs, num_query, 1, 1, 1)
    sample_lvl = roi_lvl + offset[..., 2:3]                     # (bs, num_query, 1, num_points, 1)

    return torch.cat([sample_yx, sample_lvl], dim=-1)           # (bs, num_query, 1, num_points, 3)

def translate_to_linear_weight(ref: torch.Tensor, num_levels,
                               featmap_strides=None, tau=2.0):
    '''
    Args:
        ref (Tensor): sample_points_z, (bs, num_query, 1, n_points*group)
        num_levels (int): num_level, 4
        tau (float): 2.0
        featmap_strides (list): [4, 8, 16, 32]
    Return:

    '''
    if featmap_strides is None:
        grid = torch.arange(num_levels, device=ref.device, dtype=ref.dtype).view(
            *[len(ref.shape)*[1, ]+[-1, ]])
    else:
        grid = torch.as_tensor(
            featmap_strides, device=ref.device, dtype=ref.dtype)    # (level,)
        grid = grid.log2().view(*[len(ref.shape)*[1, ]+[-1, ]])     # (1, 1, 1, level)

    ref = ref.unsqueeze(-1).clone()                 # (bs, num_query, 1, n_points*group, 1)
    l2 = (ref-grid).pow(2.0).div(tau).abs().neg()   # (bs, num_query, 1, n_points*group, level)
    weight = torch.sigmoid(l2)              # (bs, num_query, 1, n_points*group, level)

    return weight

def sampling_each_level(sample_points: Tensor,
                        value: Tensor,
                        weight=None,
                        n_points=1):
    '''
    Args:
        sample_points (Tensor): (bs, num_query, 1, n_points*group, 2)
        value (Tensor): (bs, c, h, w)
        weight (Tensor): (bs, num_query, 1, n_points*group)
        n_points (int): 32
    Return:
        out (Tensor): (bs, n_queries, n_groups, n_points, c//n_groups)
    '''
    B1, n_queries, _t, n_groups_points, _ = sample_points.shape
    assert _t == 1
    B2, C_feat, H_feat, W_feat = value.shape
    assert B1 == B2
    B = B1

    n_groups = n_groups_points//n_points
    n_channels = C_feat//n_groups

    sample_points = sample_points \
        .view(B, n_queries, n_groups, n_points, 2) \
        .permute(0, 2, 1, 3, 4).flatten(0, 1)   # (bs*n_groups, num_query, n_points, 2)
    sample_points = sample_points*2.0-1.0       # (bs*n_groups, num_query, n_points, 2)

    value = value.reshape(B*n_groups, n_channels, H_feat, W_feat)
    out = F.grid_sample(
        value, sample_points,
        mode='bilinear', padding_mode='zeros', align_corners=False) # (bs*n_groups, c//n_groups, num_query, n_points)

    if weight is not None:
        weight = weight.view(B, n_queries, n_groups, n_points) \
            .permute(0, 2, 1, 3).flatten(0, 1).unsqueeze(1)         # (bs*n_groups, 1, n_queries, n_points)
        out *= weight                                               # (bs*n_groups, c//n_groups, num_query, n_points)

    return out \
        .view(B, n_groups, n_channels, n_queries, n_points) \
        .permute(0, 3, 1, 4, 2)