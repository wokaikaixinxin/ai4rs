# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch
from mmdet.models.task_modules.coders.base_bbox_coder import BaseBBoxCoder
from torch import Tensor

from ai4rs.registry import TASK_UTILS


@TASK_UTILS.register_module()
class CSLCoder(BaseBBoxCoder):
    """Circular Smooth Label Coder.

    `Circular Smooth Label (CSL)
    <https://link.springer.com/chapter/10.1007/978-3-030-58598-3_40>`_ .

    Args:
        angle_version (str): Angle definition.
        omega (float, optional): Angle discretization granularity.
            Default: 1.
        window (str, optional): Window function. Default: gaussian.
        radius (int/float): window radius, int type for
            ['triangle', 'rect', 'pulse'], float type for
            ['gaussian']. Default: 6.
    """

    def __init__(self, angle_version, omega=1, window='gaussian', radius=6):
        super().__init__()
        self.angle_version = angle_version
        assert angle_version in ['oc', 'le90', 'le135']
        assert window in ['gaussian', 'triangle', 'rect', 'pulse']
        self.angle_range = 90 if angle_version == 'oc' else 180
        self.angle_offset_dict = {'oc': 0, 'le90': 90, 'le135': 45}
        self.angle_offset = self.angle_offset_dict[angle_version]
        self.omega = omega
        self.window = window
        self.radius = radius
        self.encode_size = int(self.angle_range // omega)

    def encode(self, angle_targets: Tensor) -> Tensor:
        """Circular Smooth Label Encoder.

        Args:
            angle_targets (Tensor): Angle offset for each scale level
                Has shape (num_anchors * H * W, 1)

        Returns:
            Tensor: The csl encoding of angle offset for each scale
            level. Has shape (num_anchors * H * W, encode_size)
        """

        # radius to degree
        angle_targets_deg = angle_targets * (180 / math.pi)
        # empty label
        smooth_label = torch.zeros_like(angle_targets).repeat(
            1, self.encode_size)
        angle_targets_deg = (angle_targets_deg +
                             self.angle_offset) / self.omega
        # Float to Int
        angle_targets_long = angle_targets_deg.long()

        if self.window == 'pulse':
            radius_range = angle_targets_long % self.encode_size
            smooth_value = 1.0
        elif self.window == 'rect':
            base_radius_range = torch.arange(
                -self.radius, self.radius, device=angle_targets_long.device)
            radius_range = (base_radius_range +
                            angle_targets_long) % self.encode_size
            smooth_value = 1.0
        elif self.window == 'triangle':
            base_radius_range = torch.arange(
                -self.radius, self.radius, device=angle_targets_long.device)
            radius_range = (base_radius_range +
                            angle_targets_long) % self.encode_size
            smooth_value = 1.0 - torch.abs(
                (1 / self.radius) * base_radius_range)

        elif self.window == 'gaussian':
            base_radius_range = torch.arange(
                -self.angle_range // 2,
                self.angle_range // 2,
                device=angle_targets_long.device)

            radius_range = (base_radius_range +
                            angle_targets_long) % self.encode_size
            smooth_value = torch.exp(-torch.pow(base_radius_range.float(), 2.)
                                     ) / (2 * self.radius**2)

        else:
            raise NotImplementedError

        if isinstance(smooth_value, torch.Tensor):
            smooth_value = smooth_value.unsqueeze(0).repeat(
                smooth_label.size(0), 1)

        return smooth_label.scatter(1, radius_range, smooth_value)

    def decode(self, angle_preds: Tensor, keepdim: bool = False) -> Tensor:
        """Circular Smooth Label Decoder.

        Args:
            angle_preds (Tensor): The csl encoding of angle offset for each
                scale level. Has shape (num_anchors * H * W, encode_size) or
                (B, num_anchors * H * W, encode_size)
            keepdim (bool): Whether the output tensor has dim retained or not.


        Returns:
            Tensor: Angle offset for each scale level. When keepdim is true,
            return (num_anchors * H * W, 1) or (B, num_anchors * H * W, 1),
            otherwise (num_anchors * H * W,) or (B, num_anchors * H * W)
        """
        if angle_preds.shape[0] == 0:
            shape = list(angle_preds.size())
            if keepdim:
                shape[-1] = 1
            else:
                shape = shape[:-1]
            return angle_preds.new_zeros(shape)
        angle_cls_inds = torch.argmax(angle_preds, dim=-1, keepdim=keepdim)
        angle_pred = ((angle_cls_inds + 0.5) *
                      self.omega) % self.angle_range - self.angle_offset
        return angle_pred * (math.pi / 180)


@TASK_UTILS.register_module()
class PSCCoder(BaseBBoxCoder):
    """Phase-Shifting Coder.

    `Phase-Shifting Coder (PSC)
    <https://arxiv.org/abs/2211.06368>`.

    Args:
        angle_version (str): Angle definition.
            Only 'le90' is supported at present.
        dual_freq (bool, optional): Use dual frequency. Default: True.
        num_step (int, optional): Number of phase steps. Default: 3.
        thr_mod (float): Threshold of modulation. Default: 0.47.
    """

    def __init__(self,
                 angle_version: str,
                 dual_freq: bool = True,
                 num_step: int = 3,
                 thr_mod: float = 0.47):
        super().__init__()
        self.angle_version = angle_version
        assert angle_version in ['le90']
        self.dual_freq = dual_freq
        self.num_step = num_step
        self.thr_mod = thr_mod
        if self.dual_freq:
            self.encode_size = 2 * self.num_step
        else:
            self.encode_size = self.num_step

        self.coef_sin = torch.tensor(
            tuple(
                torch.sin(torch.tensor(2 * k * torch.pi / self.num_step))
                for k in range(self.num_step)))
        self.coef_cos = torch.tensor(
            tuple(
                torch.cos(torch.tensor(2 * k * torch.pi / self.num_step))
                for k in range(self.num_step)))

    def encode(self, angle_targets: Tensor) -> Tensor:
        """Phase-Shifting Encoder.

        Args:
            angle_targets (Tensor): Angle offset for each scale level.
                Has shape (num_anchors * H * W, 1)

        Returns:
            list[Tensor]: The psc coded data (phase-shifting patterns)
                for each scale level.
                Has shape (num_anchors * H * W, encode_size)
        """
        phase_targets = angle_targets * 2
        phase_shift_targets = tuple(
            torch.cos(phase_targets + 2 * torch.pi * x / self.num_step)
            for x in range(self.num_step))

        # Dual-freq PSC for square-like problem
        if self.dual_freq:
            phase_targets = angle_targets * 4
            phase_shift_targets += tuple(
                torch.cos(phase_targets + 2 * torch.pi * x / self.num_step)
                for x in range(self.num_step))

        return torch.cat(phase_shift_targets, axis=-1)

    def decode(self, angle_preds: Tensor, keepdim: bool = False) -> Tensor:
        """Phase-Shifting Decoder.

        Args:
            angle_preds (Tensor): The psc coded data (phase-shifting patterns)
                for each scale level.
                Has shape (num_anchors * H * W, encode_size)
            keepdim (bool): Whether the output tensor has dim retained or not.

        Returns:
            list[Tensor]: Angle offset for each scale level.
                Has shape (num_anchors * H * W, 1) when keepdim is true,
                (num_anchors * H * W) otherwise
        """
        self.coef_sin = self.coef_sin.to(angle_preds)
        self.coef_cos = self.coef_cos.to(angle_preds)

        phase_sin = torch.sum(
            angle_preds[:, 0:self.num_step] * self.coef_sin,
            dim=-1,
            keepdim=keepdim)
        phase_cos = torch.sum(
            angle_preds[:, 0:self.num_step] * self.coef_cos,
            dim=-1,
            keepdim=keepdim)
        phase_mod = phase_cos**2 + phase_sin**2
        phase = -torch.atan2(phase_sin, phase_cos)  # In range [-pi,pi)

        if self.dual_freq:
            phase_sin = torch.sum(
                angle_preds[:, self.num_step:(2 * self.num_step)] *
                self.coef_sin,
                dim=-1,
                keepdim=keepdim)
            phase_cos = torch.sum(
                angle_preds[:, self.num_step:(2 * self.num_step)] *
                self.coef_cos,
                dim=-1,
                keepdim=keepdim)
            phase_mod = phase_cos**2 + phase_sin**2
            phase2 = -torch.atan2(phase_sin, phase_cos) / 2

            # Phase unwarpping, dual freq mixing
            # Angle between phase and phase2 is obtuse angle
            idx = torch.cos(phase) * torch.cos(phase2) + torch.sin(
                phase) * torch.sin(phase2) < 0
            # Add pi to phase2 and keep it in range [-pi,pi)
            phase2[idx] = phase2[idx] % (2 * torch.pi) - torch.pi
            phase = phase2

        # Set the angle of isotropic objects to zero
        phase[phase_mod < self.thr_mod] *= 0
        angle_pred = phase / 2
        return angle_pred


@TASK_UTILS.register_module()
class PseudoAngleCoder(BaseBBoxCoder):
    """Pseudo Angle Coder."""

    encode_size = 1

    def encode(self, angle_targets: Tensor) -> Tensor:
        return angle_targets

    def decode(self, angle_preds: Tensor, keepdim: bool = False) -> Tensor:
        if keepdim:
            return angle_preds
        else:
            return angle_preds.squeeze(-1)

@TASK_UTILS.register_module()
class ACMCoder(BaseBBoxCoder):
    """Coordinate Decomposition Method(CDM) Coder.

    Encoding formula:
        z = f (\theta) = e^{j\omega\theta}
          = \cos(\omega\theta) + j \sin(\omega\theta)

    Decoding formula:
        \theta = f^{-1}(z) = -\frac{j}{\omega} \ln z
               = \frac{1}{\omega}((\mathrm{arctan2}(z_{im}, z_{re}) + 2\pi) \bmod 2\pi)

    Args:
        angle_version (str): Angle definition.
            Only 'le90' is supported at present.
        dual_freq (bool, optional): Use dual frequency. Default: True.
        N (int, optional): just for aligning coder interface.
    """

    def __init__(self,
                 angle_version: str = 'le90',
                 base_omega=2,
                 dual_freq: bool = True):
        super().__init__()
        self.angle_version = angle_version
        assert angle_version in ['le90']
        self.dual_freq = dual_freq
        self.encode_size = 4 if self.dual_freq else 2
        self.base_omega = base_omega

    def encode(self, angle_targets: Tensor) -> Tensor:
        """Angle Correct Moule (ACM) Encoder.

        Args:
            angle_targets (Tensor): Angle offset for each scale level.
                Has shape (num_anchors * H * W, 1)

        Returns:
            list[Tensor]: The psc coded data (phase-shifting patterns)
                for each scale level.
                Has shape (num_anchors * H * W, encode_size)
        """
        angle2 = self.base_omega * angle_targets
        cos2 = torch.cos(angle2)
        sin2 = torch.sin(angle2)

        if self.dual_freq:
            angle4 = 2 * self.base_omega * angle_targets
            cos4 = torch.cos(angle4)
            sin4 = torch.sin(angle4)

        return torch.cat([cos2, sin2, cos4, sin4], dim=-1) if self.dual_freq else torch.cat([cos2, sin2], dim=-1)

    def decode(self, angle_preds: Tensor, keepdim: bool = False) -> Tensor:
        """Angle Correct Moule (ACM) Decoder.

        Args:
            angle_preds (Tensor): The acm encoded-angle
                for each scale level.
                Has shape (num_anchors * H * W, encode_size)
            keepdim (bool): Whether the output tensor has dim retained or not.

        Returns:
            list[Tensor]: Angle offset for each scale level.
                Has shape (num_anchors * H * W, 1) when keepdim is true,
                (num_anchors * H * W) otherwise
        """
        if self.dual_freq:
            cos2, sin2, cos4, sin4 = angle_preds.unbind(dim=-1)
            angle_a = torch.atan2(sin2, cos2)  # 2x
            angle_b = torch.atan2(sin4, cos4) / 2  # 4x -> 2x

            idx = torch.cos(angle_a) * torch.cos(angle_b) + torch.sin(
                angle_a) * torch.sin(angle_b) < 0
            angle_b[idx] = angle_b[idx] % (2 * math.pi) - math.pi

            angles = angle_b / 2

        else:
            sin2, cos2 = angle_preds.unbind(dim=-1)
            cos2, sin2 = sin2, cos2
            angles = torch.atan2(sin2, cos2) / self.base_omega

        if keepdim:
            angles = angles.unsqueeze(dim=-1)

        return angles