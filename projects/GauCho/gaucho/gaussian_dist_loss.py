# Copyright (c) ai4rs. All rights reserved.
from copy import deepcopy

import torch
from torch import nn

from ai4rs.registry import MODELS
from ai4rs.models.losses.gaussian_dist_loss_v1 import kld_loss, bcd_loss, gwd_loss, xy_wh_r_2_xy_sigma

def probiou_loss(pred, target, fun='log1p', tau=1.0):
    """ProbIoU loss.

    Args:
        pred (torch.Tensor): Predicted bboxes.
        target (torch.Tensor): Corresponding gt bboxes.
        fun (str): The function applied to distance. Defaults to 'log1p'. # Unused
        tau (float): Defaults to 1.0. # Unused

    Returns:
        loss (torch.Tensor)
    """

    xy_p, Sigma_p = pred
    xy_t, Sigma_t = target
    Sigma = 0.5 * (Sigma_p + Sigma_t)

    # Refer to https://arxiv.org/abs/2106.06072

    xy_diffs = xy_p - xy_t
    det_p = Sigma_p[:, 0, 0] * Sigma_p[:, 1, 1] - (Sigma_p[:, 0, 1] ** 2)
    det_t = Sigma_t[:, 0, 0] * Sigma_t[:, 1, 1] - (Sigma_t[:, 0, 1] ** 2)

    x = xy_diffs[:, 0]
    y = xy_diffs[:, 1]
    a = Sigma[:, 0, 0]
    b = Sigma[:, 1, 1]
    c = Sigma[:, 0, 1]
    det = a*b - (c**2)

    B1 = 0.125 * ((a * y**2) + (b * x**2) + 2*(-c * x * y)) / det
    B2 = 0.5 * torch.log(det / (torch.sqrt((det_p * det_t) + 1e-7)))
    Bd = B1 + B2
    Bc = torch.exp(-Bd)
    loss = torch.sqrt(1 - Bc.clamp(max=1.0))

    return loss

def gaussian_prediction_2_xy_sigma(xyabc):
    """Extract xy and sigma elements from a gaussian prediction

    Args:
        xyabc (torch.Tensor): gaussian bboxes with shape (N, 5).

    Returns:
        xy (torch.Tensor): center point of 2-D Gaussian distribution
            with shape (N, 2).
        sigma (torch.Tensor): covariance matrix of 2-D Gaussian distribution
            with shape (N, 2, 2).
    """
    xy = xyabc[..., :2]
    sigma = torch.stack((xyabc[..., 2], xyabc[..., 4],
                         xyabc[..., 4], xyabc[..., 3]),
                        dim=-1).reshape(xyabc.shape[:-1] + (2, 2))
    return xy, sigma

@MODELS.register_module()
class GDLoss_GauCho(nn.Module):
    """Gaussian based loss.

    Args:
        loss_type (str):  Type of loss.
        fun (str, optional): The function applied to distance.
            Defaults to 'log1p'.
        tau (float, optional): Defaults to 1.0.
        reduction (str, optional): The reduction method of the
            loss. Defaults to 'mean'.
        loss_weight (float, optional): The weight of loss. Defaults to 1.0.
        gaussian_prediction (bool, optional): If True, treat the input as in the form (x,y,a,b,c) where a,b,c are the elements of the 2D Covariance Matrix. Defaults to False.

    Returns:
        loss (torch.Tensor)
    """
    BAG_GD_LOSS = {'kld': kld_loss, 'bcd': bcd_loss, 'gwd': gwd_loss, 'probiou': probiou_loss}

    def __init__(self,
                 loss_type,
                 fun='sqrt',
                 tau=1.0,
                 reduction='mean',
                 loss_weight=1.0,
                 gaussian_prediction=False,
                 **kwargs):
        super(GDLoss_GauCho, self).__init__()
        assert reduction in ['none', 'sum', 'mean']
        assert fun in ['log1p', 'sqrt', '']
        assert loss_type in self.BAG_GD_LOSS
        self.loss = self.BAG_GD_LOSS[loss_type]
        self.preprocess = xy_wh_r_2_xy_sigma
        self.fun = fun
        self.tau = tau
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.gaussian_prediction = gaussian_prediction
        self.kwargs = kwargs

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function.

        Args:
            pred (torch.Tensor): Predicted convexes.
            target (torch.Tensor): Corresponding gt convexes.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
               override the original reduction method of the loss.
               Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if (weight is not None) and (not torch.any(weight > 0)) and (
                reduction != 'none'):
            return (pred * weight).sum()
        if weight is not None and weight.dim() > 1:
            assert weight.shape == pred.shape
            weight = weight.mean(-1)
        _kwargs = deepcopy(self.kwargs)
        _kwargs.update(kwargs)

        mask = (weight > 0).detach()
        pred = pred[mask]
        target = target[mask]
        if self.gaussian_prediction:
            pred = gaussian_prediction_2_xy_sigma(pred)
        else:
            pred = self.preprocess(pred)
        target = self.preprocess(target)

        return self.loss(
            pred, target, fun=self.fun, tau=self.tau, **
            _kwargs) * self.loss_weight
