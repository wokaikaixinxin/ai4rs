import torch
import torch.nn as nn
import math
from functools import partial
import warnings
from mmcv.cnn.bricks import DropPath
from mmengine.utils import to_2tuple
from mmengine.model import BaseModule
from mmrotate.registry import MODELS

# --- Dependency handling ---
try:
    from mmcv.cnn.utils.weight_init import (constant_init, normal_init, trunc_normal_init)
    from mmcv.cnn import build_norm_layer

except ImportError:

    def build_norm_layer(cfg, num_features, postfix=''):
        return 'BN', nn.BatchNorm2d(num_features)


    def trunc_normal_init(module, std=0.02, bias=0):
        nn.init.trunc_normal_(module.weight, std=std); nn.init.constant_(module.bias, bias)


    def constant_init(module, val, bias=0):
        nn.init.constant_(module.weight, val); nn.init.constant_(module.bias, bias)


    def normal_init(module, mean=0, std=1, bias=0):
        nn.init.normal_(module.weight, mean, std); nn.init.constant_(module.bias, bias)


# =============================================================================
# Core: PKSModule (PKINet-v2)
# =============================================================================

def _fuse_bn_tensor(conv, bn):
    """Fuses BN tensor into Conv tensor."""
    if conv is None:
        kernel = torch.zeros(bn.num_features, 1, 1, 1, device=bn.weight.device)
    else:
        kernel = conv.weight
    running_mean, running_var = bn.running_mean, bn.running_var
    gamma, beta, eps = bn.weight, bn.bias, bn.eps
    std = (running_var + eps).sqrt()
    t = (gamma / std).reshape(-1, 1, 1, 1)
    if conv is None:
        kernel = torch.ones(bn.num_features, 1, 1, 1, device=bn.weight.device)
        return kernel * t, beta - running_mean * gamma / std
    return kernel * t, beta - running_mean * gamma / std


class PKSModule(nn.Module):
    """
    [PKINet-v2] Poly-Kernel Scope (PKS) Module
    Branches:
    1. Axial Dense: 1x19 + 19x1 (Series) -> RF 19 [Global Backbone]
    2. Sparse: 7x7, d=3 -> RF 19 [Wide Context]
    3. Sparse: 5x5, d=3 -> RF 13 [Medium Transition]
    4. Sparse: 3x3, d=3 -> RF 7  [New: Sub-Medium]
    5. Dense:  3x3, d=1 -> RF 3  [Micro Texture]
    """

    def __init__(self, dim, deploy=False, auto_reparam=False,
                 norm_cfg=dict(type='BN'), branch_scale=1.0):
        super().__init__()
        self.deploy = deploy
        self.auto_reparam = auto_reparam
        self.dim = dim
        self.branch_scale = branch_scale

        # Max Kernel Size for Fusion (Determined by RF of Branch 1 & 2)
        # Branch 1: 19x19
        # Branch 2: 7 + (7-1)*(3-1) = 7 + 12 = 19
        self.max_k = 19

        # [Head] Pre-process Conv (5x5) - Kept from V5 structure
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)

        # [Tail] 1x1 Mixing
        self.conv1 = nn.Conv2d(dim, dim, 1)

        if deploy:
            # Deployment: Single Fused Large Kernel (19x19)
            self.fused_parallel_conv = nn.Conv2d(dim, dim, kernel_size=self.max_k,
                                                 padding=self.max_k // 2,
                                                 groups=dim, bias=True)
        else:
            # --- Branch 1: Axial Dense (1x19 + 19x1) ---
            k_axial = 19
            self.branch1_axial = nn.Sequential(
                nn.Conv2d(dim, dim, (1, k_axial), stride=1, padding=(0, k_axial // 2), groups=dim, bias=False),
                nn.Conv2d(dim, dim, (k_axial, 1), stride=1, padding=(k_axial // 2, 0), groups=dim, bias=False),
                build_norm_layer(norm_cfg, dim)[1]
            )

            # --- Branch 2: Sparse (7x7, d=3) ---
            k_b2, d_b2 = 7, 3
            pad_b2 = (k_b2 - 1) * d_b2 // 2
            self.branch2_sparse = nn.Sequential(
                nn.Conv2d(dim, dim, k_b2, stride=1, padding=pad_b2, dilation=d_b2, groups=dim, bias=False),
                build_norm_layer(norm_cfg, dim)[1]
            )

            # --- Branch 3: Sparse (5x5, d=3) ---
            k_b3, d_b3 = 5, 3
            pad_b3 = (k_b3 - 1) * d_b3 // 2
            self.branch3_sparse = nn.Sequential(
                nn.Conv2d(dim, dim, k_b3, stride=1, padding=pad_b3, dilation=d_b3, groups=dim, bias=False),
                build_norm_layer(norm_cfg, dim)[1]
            )

            # --- Branch 4: Sparse (3x3, d=3) [NEW] ---
            k_b4, d_b4 = 3, 3
            pad_b4 = (k_b4 - 1) * d_b4 // 2
            self.branch4_sparse = nn.Sequential(
                nn.Conv2d(dim, dim, k_b4, stride=1, padding=pad_b4, dilation=d_b4, groups=dim, bias=False),
                build_norm_layer(norm_cfg, dim)[1]
            )

            # --- Branch 5: Dense (3x3, d=1) ---
            self.branch5_dense = nn.Sequential(
                nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
                build_norm_layer(norm_cfg, dim)[1]
            )

    def forward(self, x):
        if self.auto_reparam and not self.training and not self.deploy:
            self.switch_to_deploy()

        if self.deploy:
            attn = self.conv0(x)
            attn = self.fused_parallel_conv(attn)
            attn = self.conv1(attn)
            return x * attn

        # Training Mode
        x_feat = self.conv0(x)

        # Accumulate all 5 branches
        # 1. Axial 19x19
        attn = self.branch1_axial(x_feat)
        # 2. Sparse 7x7 (d=3)
        attn = attn + self.branch2_sparse(x_feat)
        # 3. Sparse 5x5 (d=3)
        attn = attn + self.branch3_sparse(x_feat)
        # 4. Sparse 3x3 (d=3)
        attn = attn + self.branch4_sparse(x_feat)
        # 5. Dense 3x3
        attn = attn + self.branch5_dense(x_feat)
        attn = attn * self.branch_scale

        # Tail Proj
        attn = self.conv1(attn)

        return x * attn

    def switch_to_deploy(self):
        if self.deploy: return

        device = self.branch1_axial[0].weight.device

        # Initialize the master fused kernel (19x19)
        fused_kernel = torch.zeros(self.dim, 1, self.max_k, self.max_k, device=device)
        fused_bias = torch.zeros(self.dim, device=device)

        center_k = self.max_k // 2  # Center index of 19x19 (i.e., 9)

        # === Helper for Dilated Fusion ===
        def fuse_dilated_branch(branch, k_size, dilation):
            # Fuse Conv+BN
            k_w, b_w = _fuse_bn_tensor(branch[0], branch[1])  # (D, 1, k, k)

            center_small = k_size // 2

            # Scatter weights into the large fused_kernel
            for i in range(k_size):
                for j in range(k_size):
                    # Calculate position in large kernel
                    offset_h = (i - center_small) * dilation
                    offset_w = (j - center_small) * dilation

                    h_idx = center_k + offset_h
                    w_idx = center_k + offset_w

                    # Safe check (though design guarantees fit)
                    if 0 <= h_idx < self.max_k and 0 <= w_idx < self.max_k:
                        fused_kernel[:, :, h_idx, w_idx] += k_w[:, :, i, j]

            return b_w

        # 1. Fuse Branch 1 (Axial Series: 1x19 * 19x1)
        k1 = self.branch1_axial[0].weight  # 1x19
        k2, b2 = _fuse_bn_tensor(self.branch1_axial[1], self.branch1_axial[2])  # 19x1 fused with BN

        # Matrix Mul: (D, 1, 19, 1) @ (D, 1, 1, 19) -> (D, 1, 19, 19)
        k_axial_eff = torch.matmul(k2, k1)

        # Branch 1 is exactly 19x19, so direct addition
        fused_kernel += k_axial_eff
        fused_bias += b2

        # 2. Fuse Branch 2 (7x7, d=3)
        fused_bias += fuse_dilated_branch(self.branch2_sparse, k_size=7, dilation=3)

        # 3. Fuse Branch 3 (5x5, d=3)
        fused_bias += fuse_dilated_branch(self.branch3_sparse, k_size=5, dilation=3)

        # 4. Fuse Branch 4 (3x3, d=3)
        fused_bias += fuse_dilated_branch(self.branch4_sparse, k_size=3, dilation=3)

        # 5. Fuse Branch 5 (3x3, d=1)
        fused_bias += fuse_dilated_branch(self.branch5_dense, k_size=3, dilation=1)
        fused_kernel *= self.branch_scale
        fused_bias *= self.branch_scale

        # Finalize
        self.fused_parallel_conv = nn.Conv2d(self.dim, self.dim, self.max_k,
                                             padding=self.max_k // 2,
                                             groups=self.dim, bias=True)
        self.fused_parallel_conv.weight.data = fused_kernel
        self.fused_parallel_conv.bias.data = fused_bias

        # Cleanup
        del self.branch1_axial, self.branch2_sparse, self.branch3_sparse, self.branch4_sparse, self.branch5_dense
        self.deploy = True


# =============================================================================
# Wrapper: PKSBlock
# =============================================================================

class PKSBlock(nn.Module):
    def __init__(self, dim, deploy=False, auto_reparam=False,
                 norm_cfg=dict(type='BN'), branch_scale=1.0):
        super().__init__()
        self.proj_1 = nn.Conv2d(dim, dim, 1)
        self.activation = nn.GELU()
        # Uses PKS module (fixed branch design from PKINet-v2)
        self.spatial_gating_unit = PKSModule(
            dim, deploy=deploy, auto_reparam=auto_reparam,
            norm_cfg=norm_cfg, branch_scale=branch_scale)
        self.proj_2 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x

    def switch_to_deploy(self):
        self.spatial_gating_unit.switch_to_deploy()


# =============================================================================
# Standard Components (Mlp, PatchEmbed) - Unchanged
# =============================================================================

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, bias=True, groups=hidden_features)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768, norm_cfg=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = build_norm_layer(norm_cfg, embed_dim)[1] if norm_cfg else nn.BatchNorm2d(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = self.norm(x)
        return x, H, W


# =============================================================================
# Block & Backbone: PKINetV2
# =============================================================================

class PKINetV2Block(nn.Module):
    def __init__(self, dim, mlp_ratio=4.,
                 drop=0., drop_path=0., act_layer=nn.GELU,
                 auto_reparam=False, norm_cfg=dict(type='BN'),
                 branch_scale=1.0):
        super().__init__()
        self.norm1 = build_norm_layer(norm_cfg, dim)[1]
        self.norm2 = build_norm_layer(norm_cfg, dim)[1]

        # Use PKS Block
        self.attn = PKSBlock(
            dim, auto_reparam=auto_reparam, norm_cfg=norm_cfg,
            branch_scale=branch_scale)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(self.layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(self.layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
        return x

    def switch_to_deploy(self):
        self.attn.switch_to_deploy()


@MODELS.register_module()
class PKINetV2(BaseModule):
    """
    PKINet-v2 backbone.
    - Fixed 5-branch architecture in PKS Module.
    - Max Kernel 19x19.
    """

    def __init__(self, img_size=224, in_chans=3, embed_dims=[64, 128, 256, 512],
                 mlp_ratios=[8, 8, 4, 4],
                 # kernel_sizes arg is deprecated/ignored in V6 but kept for compatibility if needed
                 kernel_sizes=None,
                 drop_rate=0., drop_path_rate=0.,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 depths=[3, 4, 6, 3], num_stages=4,
                 pretrained=None, init_cfg=None,
                 deploy=False, auto_reparam=False,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 branch_scale=1.0):
        super().__init__(init_cfg=init_cfg)

        assert not (init_cfg and pretrained), 'init_cfg and pretrained cannot be set at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is not None:
            raise TypeError('pretrained must be a str or None')

        self.depths = depths
        self.num_stages = num_stages
        if deploy: auto_reparam = False

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0

        for i in range(num_stages):
            patch_embed = OverlapPatchEmbed(img_size=img_size if i == 0 else img_size // (2 ** (i + 1)),
                                            patch_size=7 if i == 0 else 3, stride=4 if i == 0 else 2,
                                            in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                            embed_dim=embed_dims[i], norm_cfg=norm_cfg)

            block = nn.ModuleList([PKINetV2Block(
                dim=embed_dims[i],
                mlp_ratio=mlp_ratios[i],
                drop=drop_rate,
                drop_path=dpr[cur + j],
                auto_reparam=auto_reparam,
                norm_cfg=norm_cfg,
                branch_scale=branch_scale)
                for j in range(depths[i])])

            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

    def init_weights(self):
        if self.init_cfg is None:
            print("[PKINet++ V6] Using Random Initialization.")
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, val=1.0, bias=0.)
                elif isinstance(m, nn.Conv2d):
                    fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    fan_out //= m.groups
                    normal_init(m, mean=0, std=math.sqrt(2.0 / fan_out), bias=0)
        else:
            print(f"[PKINet++ V6] Loading pretrained weights from {self.init_cfg.get('checkpoint')}...")
            super(PKINetV2, self).init_weights()

    def switch_to_deploy(self):
        for m in self.modules():
            if m is not self and hasattr(m, 'switch_to_deploy'):
                m.switch_to_deploy()

    def forward(self, x):
        B = x.shape[0]
        outs = []
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embed(x)
            for blk in block: x = blk(x)
            x = x.flatten(2).transpose(1, 2)
            x = norm(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)
        return outs


# Register PKINet-v2 variants
@MODELS.register_module()
class PKINetV2_T(PKINetV2):
    def __init__(self, **kwargs):
        cfg = dict(
            embed_dims=[32, 64, 160, 256], depths=[3, 3, 5, 2],
            mlp_ratios=[8, 8, 4, 4],
            drop_rate=0.1, drop_path_rate=0.1, norm_cfg=dict(type='BN', requires_grad=True)
        )
        cfg.update(kwargs)
        super().__init__(**cfg)


@MODELS.register_module()
class PKINetV2_S(PKINetV2):
    def __init__(self, **kwargs):
        cfg = dict(
            embed_dims=[64, 128, 320, 512], depths=[2, 2, 4, 2],
            mlp_ratios=[8, 8, 4, 4],
            drop_rate=0.1, drop_path_rate=0.15, norm_cfg=dict(type='BN', requires_grad=True)
        )
        cfg.update(kwargs)
        super().__init__(**cfg)


@MODELS.register_module()
class PKINetV2_S_BranchSqrt5(PKINetV2):
    """PKINetV2-S with five PKS branches scaled by 1/sqrt(5)."""

    def __init__(self, **kwargs):
        cfg = dict(
            embed_dims=[64, 128, 320, 512], depths=[2, 2, 4, 2],
            mlp_ratios=[8, 8, 4, 4],
            drop_rate=0.1, drop_path_rate=0.15,
            norm_cfg=dict(type='BN', requires_grad=True),
            branch_scale=1.0 / math.sqrt(5.0)
        )
        cfg.update(kwargs)
        super().__init__(**cfg)