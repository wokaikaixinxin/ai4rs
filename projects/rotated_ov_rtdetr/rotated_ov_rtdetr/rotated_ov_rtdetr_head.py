import math
import copy
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from mmengine.model import BaseModule
from mmengine.structures import InstanceData
from mmcv.cnn import Linear
from mmcv.ops import batched_nms
from mmdet.structures import SampleList
from mmdet.utils import InstanceList, OptConfigType
from projects.rotated_rtdetr.rotated_rtdetr import RotatedRTDETRHead


class ContrastiveEmbed(BaseModule):
    """Contrastive Head for Geo Rotated RTDETR
    compute the region-text scores according to the
    similarity between image and text features
    Args:
        bias : Learnable bias added to logits.
            Initialized as -log(100), which corresponds to
            a low foreground prior probability (~1%).
        logit_scale : Learnable temperature scaling factor.
            Initialized as log(1/0.07), following CLIP.
            exp(logit_scale) gives the inverse temperature.
    """
    def __init__(self, init_cfg: OptConfigType = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.bias = nn.Parameter(torch.full((), -math.log(100)))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, x: Tensor, w: Tensor) -> Tensor:
        """Forward function of contrastive learning.
        Args:
            x (Tensor): cls_embed. shape [bs, num_query, 512].
            w (Tensor): txt_feats. shape [bs, num_class, 512].
        Returns:
            Tensor: scores. shape [bs, num_query, num_class].
        """
        x = F.normalize(x, dim=-1, p=2)
        w = F.normalize(w, dim=-1, p=2)
        x = torch.einsum('bqc,bnc->bqn', x, w)
        x = x * self.logit_scale.exp() + self.bias
        return x


class RotatedOVRTDETRHead(RotatedRTDETRHead):
    def __init__(self,
                 *args,
                 text_channels: int = 512,
                 **kwargs):
        self.text_channels = text_channels
        super(RotatedOVRTDETRHead, self).__init__(*args, **kwargs)


    def _init_layers(self) -> None:
        """Initialize classification branch and regression branch of head.

        The difference from the parent method is the dimension of
        `self.fc_reg` which is 5 to predict [cx, cy, w, h, rad],
        and the cls_cntrst.
        """
        cls_cntrst = ContrastiveEmbed()
        fc_cls = Linear(self.embed_dims, self.text_channels)
        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, self.reg_dim))
        reg_branch = nn.Sequential(*reg_branch)

        if self.share_pred_layer:
            self.cls_branches = nn.ModuleList(
                [fc_cls for _ in range(self.num_pred_layer)])
            self.reg_branches = nn.ModuleList(
                [reg_branch for _ in range(self.num_pred_layer)])
            self.cls_contrasts = nn.ModuleList(
                [cls_cntrst for _ in range(self.num_pred_layer)])
        else:
            self.cls_branches = nn.ModuleList(
                [copy.deepcopy(fc_cls) for _ in range(self.num_pred_layer)])
            self.reg_branches = nn.ModuleList([
                copy.deepcopy(reg_branch) for _ in range(self.num_pred_layer)
            ])
            self.cls_contrasts = nn.ModuleList(
                [copy.deepcopy(cls_cntrst) for _ in range(self.num_pred_layer)])

    def loss(self, hidden_states: Tensor, references: List[Tensor],
             enc_outputs_class: Tensor, enc_outputs_coord: Tensor,
             txt_feats: Tensor, batch_data_samples: SampleList,
             dn_meta: Dict[str, int]) -> dict:
        """Perform forward propagation and loss calculation of the detection
        head on the queries of the upstream network.

        Args:
            hidden_states (Tensor): Hidden states output from each decoder
                layer, has shape (num_decoder_layers, bs, num_queries_total,
                dim), where `num_queries_total` is the sum of
                `num_denoising_queries` and `num_matching_queries` when
                `self.training` is `True`, else `num_matching_queries`.
            references (list[Tensor]): List of the reference from the decoder.
                The first reference is the `init_reference` (initial) and the
                other num_decoder_layers(6) references are `inter_references`
                (intermediate). The `init_reference` has shape (bs,
                num_queries_total, 5) and each `inter_reference` has shape
                (bs, num_queries, 5) with the last dimension arranged as
                (cx, cy, w, h, radian).
            enc_outputs_class (Tensor): The score of each point on encode
                feature map, has shape (bs, num_feat_points, cls_out_channels).
            enc_outputs_coord (Tensor): The proposal generate from the
                encode feature map, has shape (bs, num_feat_points, 5) with the
                last dimension arranged as (cx, cy, w, h, radian).
            batch_data_samples (list[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            dn_meta (Dict[str, int]): The dictionary saves information about
              group collation, including 'num_denoising_queries' and
              'num_denoising_groups'. It will be used for split outputs of
              denoising and matching parts and loss calculation.

        Returns:
            dict: A dictionary of loss components.
        """
        batch_gt_instances = []
        batch_img_metas = []
        for data_sample in batch_data_samples:
            batch_img_metas.append(data_sample.metainfo)
            batch_gt_instances.append(data_sample.gt_instances)

        outs = self(hidden_states, references, txt_feats)
        loss_inputs = outs + (enc_outputs_class, enc_outputs_coord,
                              batch_gt_instances, batch_img_metas, dn_meta)
        losses = self.loss_by_feat(*loss_inputs)
        return losses

    def forward(self, hidden_states: List[Tensor],
                references: List[Tensor],
                txt_feats) -> Tuple[Tensor, Tensor]:
        """Forward function.

        Args:
            hidden_states (list[Tensor]): List of the class embed from each
                decoder layer, has shape [(bs, num_queries, txt_channel), ...].
            references (list[Tensor]): List of the reference from the decoder.
                The first reference is the `init_reference` (initial) and the
                other num_decoder_layers(6) references are `inter_references`
                (intermediate). The `init_reference` has shape (bs,
                num_queries, 5) when `as_two_stage` of the detector is `True`,
                otherwise (bs, num_queries, 2). Each `inter_reference` has
                shape (bs, num_queries, 5) when `with_box_refine` of the
                detector is `True`, otherwise (bs, num_queries, 2). The
                coordinates are arranged as (cx, cy) when the last dimension is
                2, and (cx, cy, w, h, rad) when it is 5.
            txt_feats ([Tensor]): shape (bs, num_class, txt_channel).

        Returns:
            tuple[Tensor]: results of head containing the following tensor.

            - all_layers_outputs_classes (list[Tensor]): Outputs from the
              classification head, has shape [(bs, num_queries, num_classes),...].
            - all_layers_outputs_coords (list[Tensor]): Sigmoid outputs from the
              regression head with normalized coordinate format (cx, cy, w, h,
              rad), has shape [(bs, num_queries, 5),...] with the last dimension
              arranged as (cx, cy, w, h, rad).
        """
        all_layers_outputs_classes = []
        for idx, hidden_state in enumerate(hidden_states):  # (bs, num_queries, txt_channel)
            cls_embed = self.cls_branches[idx](hidden_state)
            outputs_class = self.cls_contrasts[idx](cls_embed, txt_feats)   # (bs, num_queries, num_classes)
            all_layers_outputs_classes.append(outputs_class)
        all_layers_outputs_coords = references
        return all_layers_outputs_classes, all_layers_outputs_coords

    def predict(self,
                hidden_states: Tensor,
                references: List[Tensor],
                txt_feats: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> InstanceList:
        """Perform forward propagation and loss calculation of the detection
        head on the queries of the upstream network.

        Args:
            hidden_states (Tensor): Hidden states output from each decoder
                layer, has shape (num_decoder_layers, num_queries, bs, dim).
            references (list[Tensor]): List of the reference from the decoder.
                The first reference is the `init_reference` (initial) and the
                other num_decoder_layers(6) references are `inter_references`
                (intermediate). The `init_reference` has shape (bs,
                num_queries, 4) when `as_two_stage` of the detector is `True`,
                otherwise (bs, num_queries, 2). Each `inter_reference` has
                shape (bs, num_queries, 4) when `with_box_refine` of the
                detector is `True`, otherwise (bs, num_queries, 2). The
                coordinates are arranged as (cx, cy) when the last dimension is
                2, and (cx, cy, w, h) when it is 4.
            txt_feats ([Tensor]): shape (bs, num_class, txt_channel).
            batch_data_samples (list[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool, optional): If `True`, return boxes in original
                image space. Defaults to `True`.

        Returns:
            list[obj:`InstanceData`]: Detection results of each image
            after the post process.
        """
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]
        self.num_test_classes = txt_feats[0].shape[0]
        outs = self(hidden_states, references, txt_feats)
        predictions = self.predict_by_feat(
            *outs, batch_img_metas=batch_img_metas, rescale=rescale)
        return predictions

    def _predict_by_feat_single(self,
                                cls_score: Tensor,
                                bbox_pred: Tensor,
                                img_meta: dict,
                                rescale: bool = True) -> InstanceData:
        """Transform outputs from the last decoder layer into bbox predictions
        for each image.

        The only difference from the parent method is the normalization factor
        which has 5 dimension for rotated boxes.

        Args:
            cls_score (Tensor): Box score logits from the last decoder layer
                for each image. Shape [num_queries, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from the last decoder layer
                for each image, with coordinate format (cx, cy, w, h, rad) and
                shape [num_queries, 5].
            img_meta (dict): Image meta info.
            rescale (bool): If True, return boxes in original image
                space. Default True.

        Returns:
            :obj:`InstanceData`: Detection results of each image
            after the post process.
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 5),
                  the last dimension 4 arrange as (cx, cy, w, h, rad).
        """
        assert len(cls_score) == len(bbox_pred)  # num_queries
        max_per_img = self.test_cfg.get('max_per_img', len(cls_score))
        img_shape = img_meta['img_shape']
        # exclude background
        if self.loss_cls.use_sigmoid:
            cls_score = cls_score.sigmoid()
            scores, indexes = cls_score.view(-1).topk(max_per_img)
            det_labels = indexes % self.num_test_classes
            bbox_index = indexes // self.num_test_classes
            bbox_pred = bbox_pred[bbox_index]
        else:
            scores, det_labels = F.softmax(cls_score, dim=-1)[..., :-1].max(-1)
            scores, bbox_index = scores.topk(max_per_img)
            bbox_pred = bbox_pred[bbox_index]
            det_labels = det_labels[bbox_index]

        det_bboxes = bbox_pred
        det_bboxes[:, 0:4:2] = det_bboxes[:, 0:4:2] * img_shape[1]
        det_bboxes[:, 1:4:2] = det_bboxes[:, 1:4:2] * img_shape[0]
        # denormalize the angle dimension
        det_bboxes[:, 4] = det_bboxes[:, 4] * self.angle_factor
        det_bboxes[:, 0:4:2].clamp_(min=0, max=img_shape[1])
        det_bboxes[:, 1:4:2].clamp_(min=0, max=img_shape[0])
        if rescale:
            assert img_meta.get('scale_factor') is not None
            scale_factor = np.array(img_meta['scale_factor']).repeat(2)
            if scale_factor.shape[0] == 4:
                # angle should not be rescaled
                scale_factor = np.append(scale_factor, 1)
            det_bboxes /= det_bboxes.new_tensor(scale_factor).repeat((1, 1))

        results = InstanceData()
        results.bboxes = det_bboxes
        results.scores = scores
        results.labels = det_labels

        nms_cfg = self.test_cfg.get('nms', None)
        if nms_cfg is not None:
            _, keeps = batched_nms(
                boxes=results.bboxes,
                scores=results.scores,
                idxs=results.labels,
                nms_cfg=nms_cfg)
            results = results[keeps]
        return results