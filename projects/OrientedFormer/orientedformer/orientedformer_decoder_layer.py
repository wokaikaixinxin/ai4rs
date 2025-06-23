from typing import List, Sequence
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from mmcv.cnn import build_activation_layer, build_norm_layer
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention
from mmengine.config import ConfigDict
from mmengine.model import bias_init_with_prob
from mmdet.models.losses import accuracy
from mmdet.models.task_modules import SamplingResult
from mmdet.models.utils import multi_apply
from mmdet.utils import ConfigType, OptConfigType, reduce_mean
from mmdet.models.roi_heads.bbox_heads import BBoxHead
from mmrotate.registry import MODELS
from mmrotate.structures.bbox.transforms import norm_angle
from mmrotate.models.losses.gaussian_dist_loss import xy_wh_r_2_xy_sigma
from .match_cost import normalize_angle


@MODELS.register_module()
class OrientedFormerDecoderLayer(BBoxHead):
    r"""
    Args:
        num_classes (int): Number of class in dataset.
            Defaults to 80.
        num_ffn_fcs (int): The number of fully-connected
            layers in FFNs. Defaults to 2.
        num_heads (int): The hidden dimension of FFNs.
            Defaults to 8.
        num_cls_fcs (int): The number of fully-connected
            layers in classification subnet. Defaults to 1.
        num_reg_fcs (int): The number of fully-connected
            layers in regression subnet. Defaults to 3.
        feedforward_channels (int): The hidden dimension
            of FFNs. Defaults to 2048
        in_channels (int): Hidden_channels of MultiheadAttention.
            Defaults to 256.
        dropout (float): Probability of drop the channel.
            Defaults to 0.0
        ffn_act_cfg (:obj:`ConfigDict` or dict): The activation config
            for FFNs.
        dynamic_conv_cfg (:obj:`ConfigDict` or dict): The convolution
            config for DynamicConv.
        loss_iou (:obj:`ConfigDict` or dict): The config for iou or
            giou loss.
        init_cfg (:obj:`ConfigDict` or dict or list[:obj:`ConfigDict` or \
            dict]): Initialization config dict. Defaults to None.
    """

    def __init__(self,
                 num_classes: int = 80,
                 angle_version: str = 'le90',
                 num_cls_fcs: int = 1,
                 num_reg_fcs: int = 1,
                 content_dim: int = 256,
                 target_means: Sequence[float] = (0., 0., 0., 0., 0.),
                 target_stds: Sequence[float] = (1., 1., 1., 1., 1.),
                 self_attn_cfg: ConfigType = dict(
                     embed_dims=256,
                     num_heads=8,
                     dropout=0.0),
                 o3d_attn_cfg: ConfigType = dict(
                     type='Oriented3dAttention',
                     in_points=32,
                     out_points=128,
                     n_heads=4,
                     embed_dims=256),
                 ffn_cfg: ConfigType = dict(
                     embed_dims=256,
                     feedforward_channels=2048,
                     num_fcs=2,
                     ffn_drop=0.0,
                     act_cfg=dict(type='ReLU', inplace=True)),
                 loss_iou: ConfigType = dict(type='GIoULoss', loss_weight=2.0),
                 init_cfg: OptConfigType = None,
                 **kwargs) -> None:
        assert init_cfg is None, 'To prevent abnormal initialization ' \
                                 'behavior, init_cfg is not allowed to be set'
        super(OrientedFormerDecoderLayer, self).__init__(
            num_classes=num_classes,
            reg_decoded_bbox=True,
            reg_class_agnostic=True,
            init_cfg=init_cfg,
            **kwargs)
        self.loss_iou = MODELS.build(loss_iou)
        self.content_dim = content_dim
        self.fp16_enabled = False
        self.angle_version = angle_version
        self.means = target_means
        self.stds = target_stds

        # self-attention
        self.tau = nn.Parameter(torch.ones(self_attn_cfg.get('num_heads'), ))
        self.self_attn = MultiheadAttention(**self_attn_cfg)
        self.self_attn_norm = build_norm_layer(dict(type='LN'), content_dim)[1]

        # cross-attention
        self.o3d_attn = MODELS.build(o3d_attn_cfg)
        self.o3d_attn_norm = build_norm_layer(dict(type='LN'), content_dim)[1]

        # FFN
        self.ffn = FFN(**ffn_cfg)
        self.ffn_norm = build_norm_layer(dict(type='LN'), content_dim)[1]

        self.cls_fcs = nn.ModuleList()
        for _ in range(num_cls_fcs):
            self.cls_fcs.append(
                nn.Linear(content_dim, content_dim, bias=True))
            self.cls_fcs.append(
                build_norm_layer(dict(type='LN'), content_dim)[1])
            self.cls_fcs.append(
                build_activation_layer(dict(type='ReLU', inplace=True)))

        # over load the self.fc_cls in BBoxHead
        if self.loss_cls.use_sigmoid:
            self.fc_cls = nn.Linear(content_dim, self.num_classes)
        else:
            self.fc_cls = nn.Linear(content_dim, self.num_classes + 1)

        self.reg_fcs = nn.ModuleList()
        for _ in range(num_reg_fcs):
            self.reg_fcs.append(
                nn.Linear(content_dim, content_dim, bias=True))
            self.reg_fcs.append(
                build_norm_layer(dict(type='LN'), content_dim)[1])
            self.reg_fcs.append(
                build_activation_layer(dict(type='ReLU', inplace=True)))
        # over load the self.fc_cls in BBoxHead
        self.fc_reg = nn.Linear(content_dim, 5)

        assert self.reg_class_agnostic, 'OrientedFormer only ' \
            'suppport `reg_class_agnostic=True` '
        assert self.reg_decoded_bbox, 'OrientedFormer only ' \
            'suppport `reg_decoded_bbox=True`'

    @torch.no_grad()
    def init_weights(self):
        super(OrientedFormerDecoderLayer, self).init_weights()
        for n, m in self.named_modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()
                nn.init.xavier_uniform_(m.weight)

        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            nn.init.constant_(self.fc_cls.bias, bias_init)

        nn.init.zeros_(self.fc_reg.weight)
        nn.init.zeros_(self.fc_reg.bias)

        nn.init.uniform_(self.tau, 0.0, 4.0)

    def forward(self,
                x: list,
                query_xyzrt: Tensor,
                query_content: Tensor,
                featmap_strides: list):
        '''
        Args:
            x (list): List of multi-level img features. Each level feature has shape (bs, c, h, w).
            query_xyzrt (Tensor): (bs, num_query, 5), where 5 represents (x, y, w, h, radian).
            query_content (Tensor): (bs, num_query, 256)
            featmap_strides (list): [4, 8, 16, 32]
        Returns:

        '''
        # self-attention
        N, n_query = query_content.shape[:2]
        with torch.no_grad():
            rboxes = self.decode_box(query_xyzrt)           # (bs, num_query, 5)
            gau_rboxes = [xy_wh_r_2_xy_sigma(rbox)
                          for rbox in rboxes]               # [tuple((bs,num_query,2),(bs,num_query,2,2))*bs]
            gau_scores = [get_gau_scores(rbox, rbox)
                          for rbox in gau_rboxes]           # [(num_query, num_query) * bs]
            gau_scores = torch.stack(gau_scores, dim=0)[:, None, :, :]    # (bs, 1, num_query, num_query)
            gau_scores = (gau_scores + 1e-7).log()
            pe = position_embedding(self.decode_box(query_xyzrt), query_content.size(-1) // 4)    # (bs, num_query, 256)
        attn_bias = (gau_scores * self.tau.view(
            1, -1, 1, 1)).flatten(0, 1)                     # (bs*head, num_query, num_query)
        query_content = query_content.permute(1, 0, 2)      # (num_query, bs, 256)
        pe = pe.permute(1, 0, 2)                            # (num_query, bs, 256)
        '''sinusoidal positional embedding'''
        query_content_attn = query_content + pe             # (num_query, bs, 256)
        query_content = self.self_attn(
            query_content_attn,                            # (num_query, bs, 256)
            attn_mask=attn_bias)                            # (num_query, bs, 256)
        query_content = self.self_attn_norm(query_content)  # (num_query, bs, 256)
        query_content = query_content.permute(1, 0, 2)      # (bs, num_query, 256)

        # cross-attention
        query_content = self.o3d_attn(
            x, query_content, query_xyzrt, featmap_strides) # (bs, num_query, 256)

        query_content = self.o3d_attn_norm(query_content)

        # FFN
        query_content = self.ffn_norm(self.ffn(query_content)) # (bs, num_query, 256)

        cls_feat = query_content
        reg_feat = query_content

        for cls_layer in self.cls_fcs:
            cls_feat = cls_layer(cls_feat)
        for reg_layer in self.reg_fcs:
            reg_feat = reg_layer(reg_feat)

        cls_score = self.fc_cls(cls_feat).view(N, n_query, -1)
        xyzrt_delta = self.fc_reg(reg_feat).view(N, n_query, -1)

        return cls_score, xyzrt_delta, query_content.view(N, n_query, -1)

    def refine_xyzrt(self, xyzrt: Tensor, xyzrt_delta: Tensor, return_bbox: bool=True):
        '''
        Args:
            xyzrt (Tensor): (bs, num_query, 5)
            xyzrt_delta (Tensor): (bs, num_query, 5)
            return_bbox (bool): True
        Returns:
            xyzrt (Tensor): (bs, num_query, 5), where 5 represents (x, y, z, r, radian).
            decoded_bbox (Tensor): (bs, num_query, 5), where 5 represents (x, y, w, h, radian).
        '''
        means = xyzrt_delta.new_tensor(self.means).view(1, -1)
        stds = xyzrt_delta.new_tensor(self.stds).view(1, -1)
        xyzrt_delta = xyzrt_delta * stds + means

        z = xyzrt[..., 2:3]
        new_xy = xyzrt[..., 0:2] + xyzrt_delta[..., 0:2] * (2 ** z)
        new_zr = xyzrt[..., 2:4] + xyzrt_delta[..., 2:4]
        new_theta = norm_angle(xyzrt[..., 4:] + xyzrt_delta[..., 4:], self.angle_version)
        xyzrt = torch.cat([new_xy, new_zr, new_theta], dim=-1)
        if return_bbox:
            return xyzrt, self.decode_box(xyzrt)
        else:
            return xyzrt

    def decode_box(self, xyzrt: Tensor):
        '''
        Args:
            xyzrt (Tensor): (bs, num_query, 5), where 5 represents (x, y, z, r, radian).
        Returns:
            decoded_bbox (Tensor): (bs, num_query, 5), where 5 represents (x, y, w, h, radian).
        '''
        scale = 2.00 ** xyzrt[..., 2:3] # (bs, num_query, 1)
        ratio = 2.00 ** torch.cat([xyzrt[..., 3:4] * -0.5,
                                  xyzrt[..., 3:4] * 0.5], dim=-1)   # (bs, num_query, 2)
        wh = scale * ratio              # (bs, num_query, 2)
        xy = xyzrt[..., 0:2]            # (bs, num_query, 2)
        gt = xyzrt[..., 4:]             # (bs, num_query, 1)
        gw = wh[..., 0:1]
        gh = wh[..., 1:]

        w_regular = torch.where(gw > gh, gw, gh)
        h_regular = torch.where(gw > gh, gh, gw)
        theta_regular = torch.where(gw > gh, gt, gt + np.pi / 2)
        theta_regular = norm_angle(theta_regular, self.angle_version)
        decoded_bbox = torch.cat(
            [xy, w_regular, h_regular, theta_regular], dim=-1) # (bs, num_query, 5)
        return decoded_bbox

    def loss_and_target(self,
                        cls_score: Tensor,
                        bbox_pred: Tensor,
                        sampling_results: List[SamplingResult],
                        rcnn_train_cfg: ConfigType,
                        imgs_whwht: Tensor,
                        concat: bool = True,
                        reduction_override: str = None) -> dict:
        """Calculate the loss based on the features extracted by the DIIHead.

        Args:
            cls_score (Tensor): Classification prediction
                results of all class, has shape
                (N, num_classes)
            bbox_pred (Tensor): Regression prediction results, has shape
                (N, 5), the last
                dimension 5 represents [cx, cy, w, h, radian].
            sampling_results (List[obj:SamplingResult]): Assign results of
                all images in a batch after sampling.
            rcnn_train_cfg (obj:ConfigDict): `train_cfg` of RCNN.
            imgs_whwht (Tensor): imgs_whwh (Tensor): Tensor with\
                shape (bs, num_query, 5), the last
                dimension means
                [img_width, img_height, img_width, img_height, 1.].
            concat (bool): Whether to concatenate the results of all
                the images in a single batch. Defaults to True.
            reduction_override (str, optional): The reduction
                method used to override the original reduction
                method of the loss. Options are "none",
                "mean" and "sum". Defaults to None.

        Returns:
            dict: A dictionary of loss and targets components.
            The targets are only used for cascade rcnn.
        """
        cls_reg_targets = self.get_targets(
            sampling_results=sampling_results,
            rcnn_train_cfg=rcnn_train_cfg,
            concat=concat)
        (labels, label_weights, bbox_targets, bbox_weights) = cls_reg_targets

        losses = dict()
        bg_class_ind = self.num_classes
        # note in spare rcnn num_gt == num_pos
        pos_inds = (labels >= 0) & (labels < bg_class_ind)
        num_pos = pos_inds.sum().float()
        avg_factor = reduce_mean(num_pos)
        if cls_score is not None:
            if cls_score.numel() > 0:
                losses['loss_cls'] = self.loss_cls(
                    cls_score,
                    labels,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)
                losses['pos_acc'] = accuracy(cls_score[pos_inds],
                                             labels[pos_inds])
        if bbox_pred is not None:
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            # do not perform bounding box regression for BG anymore.
            if pos_inds.any():
                pos_bbox_pred = bbox_pred.reshape(bbox_pred.size(0),
                                                  5)[pos_inds.type(torch.bool)]
                imgs_whwht = imgs_whwht.reshape(bbox_pred.size(0),
                                              5)[pos_inds.type(torch.bool)]
                pos_bbox_pred_temp = pos_bbox_pred / imgs_whwht
                bbox_targets_temp = bbox_targets[pos_inds.type(torch.bool)] / imgs_whwht
                pos_bbox_pred_temp[..., -1] = normalize_angle(pos_bbox_pred_temp[..., -1], self.angle_version)
                bbox_targets_temp[..., -1] = normalize_angle(bbox_targets_temp[..., -1], self.angle_version)
                losses['loss_bbox'] = self.loss_bbox(
                    pos_bbox_pred_temp,
                    bbox_targets_temp,
                    bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=avg_factor)
                losses['loss_iou'] = self.loss_iou(
                    pos_bbox_pred,
                    bbox_targets[pos_inds.type(torch.bool)],
                    bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=avg_factor)
            else:
                losses['loss_bbox'] = bbox_pred.sum() * 0
                losses['loss_iou'] = bbox_pred.sum() * 0
        return dict(loss_bbox=losses, bbox_targets=cls_reg_targets)

    def _get_targets_single(self, pos_inds: Tensor, neg_inds: Tensor,
                            pos_priors: Tensor, neg_priors: Tensor,
                            pos_gt_bboxes: Tensor, pos_gt_labels: Tensor,
                            cfg: ConfigDict) -> tuple:
        """Calculate the ground truth for proposals in the single image
        according to the sampling results.

        Almost the same as the implementation in `bbox_head`,
        we add pos_inds and neg_inds to select positive and
        negative samples instead of selecting the first num_pos
        as positive samples.

        Args:
            pos_inds (Tensor): The length is equal to the
                positive sample numbers contain all index
                of the positive sample in the origin proposal set.
            neg_inds (Tensor): The length is equal to the
                negative sample numbers contain all index
                of the negative sample in the origin proposal set.
            pos_priors (Tensor): Contains all the positive boxes,
                has shape (num_pos, 5), the last dimension 5
                represents [cx, cy, w, h, radian].
            neg_priors (Tensor): Contains all the negative boxes,
                has shape (num_neg, 5), the last dimension 5
                represents [cx, cy, w, h, radian].
            pos_gt_bboxes (Tensor): Contains gt_boxes for
                all positive samples, has shape (num_pos, 5),
                the last dimension 5
                represents [cx, cy, w, h, radian].
            pos_gt_labels (Tensor): Contains gt_labels for
                all positive samples, has shape (num_pos, ).
            cfg (obj:`ConfigDict`): `train_cfg` of R-CNN.

        Returns:
            Tuple[Tensor]: Ground truth for proposals in a single image.
            Containing the following Tensors:

            - labels(Tensor): Gt_labels for all proposals, has
              shape (num_proposals,).
            - label_weights(Tensor): Labels_weights for all proposals, has
              shape (num_proposals,).
            - bbox_targets(Tensor):Regression target for all proposals, has
              shape (num_proposals, 5), the last dimension 5
              represents [cx, cy, w, h, radian].
            - bbox_weights(Tensor):Regression weights for all proposals,
              has shape (num_proposals, 5).
        """
        num_pos = pos_priors.size(0)
        num_neg = neg_priors.size(0)
        num_samples = num_pos + num_neg

        # original implementation uses new_zeros since BG are set to be 0
        # now use empty & fill because BG cat_id = num_classes,
        # FG cat_id = [0, num_classes-1]
        labels = pos_priors.new_full((num_samples, ),
                                     self.num_classes,
                                     dtype=torch.long)
        label_weights = pos_priors.new_zeros(num_samples)
        bbox_targets = pos_priors.new_zeros(num_samples, 5)
        bbox_weights = pos_priors.new_zeros(num_samples, 5)
        if num_pos > 0:
            labels[pos_inds] = pos_gt_labels
            pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight
            label_weights[pos_inds] = pos_weight
            if not self.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_coder.encode(
                    pos_priors, pos_gt_bboxes)
            else:
                pos_bbox_targets = pos_gt_bboxes
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1
        if num_neg > 0:
            label_weights[neg_inds] = 1.0

        return labels, label_weights, bbox_targets, bbox_weights

    def get_targets(self,
                    sampling_results: List[SamplingResult],
                    rcnn_train_cfg: ConfigDict,
                    concat: bool = True) -> tuple:
        """Calculate the ground truth for all samples in a batch according to
        the sampling_results.

        Almost the same as the implementation in bbox_head, we passed
        additional parameters pos_inds_list and neg_inds_list to
        `_get_targets_single` function.

        Args:
            sampling_results (List[obj:SamplingResult]): Assign results of
                all images in a batch after sampling.
            rcnn_train_cfg (obj:ConfigDict): `train_cfg` of RCNN.
            concat (bool): Whether to concatenate the results of all
                the images in a single batch.

        Returns:
            Tuple[Tensor]: Ground truth for proposals in a single image.
            Containing the following list of Tensors:

            - labels (list[Tensor],Tensor): Gt_labels for all
              proposals in a batch, each tensor in list has
              shape (num_proposals,) when `concat=False`, otherwise just
              a single tensor has shape (num_all_proposals,).
            - label_weights (list[Tensor]): Labels_weights for
              all proposals in a batch, each tensor in list has shape
              (num_proposals,) when `concat=False`, otherwise just a
              single tensor has shape (num_all_proposals,).
            - bbox_targets (list[Tensor],Tensor): Regression target
              for all proposals in a batch, each tensor in list has
              shape (num_proposals, 4) when `concat=False`, otherwise
              just a single tensor has shape (num_all_proposals, 4),
              the last dimension 4 represents [tl_x, tl_y, br_x, br_y].
            - bbox_weights (list[tensor],Tensor): Regression weights for
              all proposals in a batch, each tensor in list has shape
              (num_proposals, 4) when `concat=False`, otherwise just a
              single tensor has shape (num_all_proposals, 4).
        """
        pos_inds_list = [res.pos_inds for res in sampling_results]
        neg_inds_list = [res.neg_inds for res in sampling_results]
        pos_priors_list = [res.pos_priors for res in sampling_results]
        neg_priors_list = [res.neg_priors for res in sampling_results]
        pos_gt_bboxes_list = [res.pos_gt_bboxes for res in sampling_results]
        pos_gt_labels_list = [res.pos_gt_labels for res in sampling_results]
        labels, label_weights, bbox_targets, bbox_weights = multi_apply(
            self._get_targets_single,
            pos_inds_list,
            neg_inds_list,
            pos_priors_list,
            neg_priors_list,
            pos_gt_bboxes_list,
            pos_gt_labels_list,
            cfg=rcnn_train_cfg)
        if concat:
            labels = torch.cat(labels, 0)
            label_weights = torch.cat(label_weights, 0)
            bbox_targets = torch.cat(bbox_targets, 0)
            bbox_weights = torch.cat(bbox_weights, 0)
        return labels, label_weights, bbox_targets, bbox_weights


def position_embedding(xywht: Tensor, num_feats: int, temperature: int = 10000):
    '''
    Args:
        xywht (Tensor): (bs, num_query, 5), where 5 represents (c_x, c_y, w, h, radian).
        num_feats (int): 64
        temperature (int): 10000
    Returns:
        pos_x (Tensor): (bs, num_query, 256)
    '''
    assert xywht.size(-1) == 5
    mean, var = xy_wh_r_2_xy_sigma(xywht)                      # (bs, num_query, 2), (bs, num_query, 2, 2)
    var_diag = torch.cat([var[..., 0, 0].unsqueeze(-1), var[..., 1, 1].unsqueeze(-1)], dim=-1)  # (bs, num_query, 2)

    dim_t = torch.arange(
        num_feats, dtype=torch.float32, device=xywht.device)   # (64)
    dim_t = (temperature ** (2 * (dim_t // 2) / num_feats)).view(1, 1, 1, -1)   # (1, 1, 1, 64)
    new_mean = mean[..., None] / dim_t                         # (bs, num_query, 2, 64)
    new_var_diag = var_diag[..., None] / (dim_t ** 2)          # (bs, num_query, 2, 64)
    pe = torch.stack((torch.exp(-0.5 * new_var_diag) * torch.sin(new_mean),
                      torch.exp(-0.5 * new_var_diag) * torch.cos(new_mean)), dim=4)  # (bs, num_query, 2, 64, 2)
    pe = pe.flatten(2)                                         # (bs, num_query, 256)
    return pe

def get_gau_scores(pred, target, tau=1.0, alpha=1.0, normalize=True):
    '''
    Args:
        pred (tuple): first, (n1, 2), where, 2 represents (x, y). second, (n1, 2, 2)
        target (tuple): first, (n2, 2), where, 2 represents (x, y). second, (n2, 2, 2)
        tau (float): 1.0
        alpha (float): 1.0
        normalize (bool): True
    Returns:
        scores (Tensor): (n1, n2)
    '''
    xy_p, Sigma_p = pred        # (n1, 2), (n1, 2, 2)
    xy_t, Sigma_t = target      # (n2, 2), (n2, 2, 2)
    assert xy_p.shape[-1] == 2 and xy_t.shape[-1] == 2
    assert Sigma_p.shape[-2:] == (2, 2) and Sigma_t.shape[-2:] == (2, 2)

    xy_distance = (xy_p.unsqueeze(1) - xy_t.unsqueeze(0)).square().sum(dim=-1)    # (n1, n2)

    whr_distance = Sigma_p.diagonal(dim1=-2, dim2=-1).sum(dim=-1).unsqueeze(1) \
                   + Sigma_t.diagonal(dim1=-2, dim2=-1).sum(dim=-1).unsqueeze(0)  # (n1, n2)

    _t_tr_tmp = torch.einsum('aij, bjk -> abik', Sigma_p, Sigma_t)  # (n1, n2, 2, 2)
    _t_tr = _t_tr_tmp[..., 0, 0] + _t_tr_tmp[..., 1, 1]             # (n1, n2)
    _t_det_sqrt = (Sigma_p.det().unsqueeze(1)
                   * Sigma_t.det().unsqueeze(0)).clamp(1e-7).sqrt() # (n1, n2)
    whr_distance = whr_distance + (-2) * ((_t_tr + 2 * _t_det_sqrt).clamp(1e-7).sqrt())

    distance = (xy_distance + alpha * alpha * whr_distance).clamp(1e-7).sqrt()

    if normalize:
        scale = 2 * (
            _t_det_sqrt.clamp(1e-7).sqrt().clamp(1e-7).sqrt()).clamp(1e-7)
        distance = distance / scale

    scores = 1 / (tau + distance)                                   # (n1, n2)
    return scores

