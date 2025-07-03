# Copyright (c) ai4rs. All rights reserved.
# other
from typing import List, Sequence, Tuple, Union
import copy
# torch
import torch
from torch import Tensor
# mmengine
from mmengine.structures import InstanceData
# mmcv
from mmcv.ops import batched_nms
# mmdet
from mmdet.structures import SampleList
from mmdet.structures.bbox import get_box_tensor
from mmdet.models.utils import empty_instances, unpack_gt_instances
from mmdet.models.roi_heads import CascadeRoIHead
from mmdet.utils import (ConfigType, InstanceList, OptConfigType, OptMultiConfig)
from mmdet.models.test_time_augs import merge_aug_masks
from mmdet.structures.bbox import bbox2roi
# ai4rs
from ai4rs.registry import MODELS
from ai4rs.structures.bbox import rbox2hbox

@MODELS.register_module()
class Decoder(CascadeRoIHead):
    r"""
    Args:
        num_stages (int): Number of stage whole iterative process.
            Defaults to 6.
        stage_loss_weights (Tuple[float]): The loss
            weight of each stage. By default all stages have
            the same weight 1.
        bbox_roi_extractor (:obj:`ConfigDict` or dict): Config of box
            roi extractor.
        mask_roi_extractor (:obj:`ConfigDict` or dict): Config of mask
            roi extractor.
        bbox_head (:obj:`ConfigDict` or dict): Config of box head.
        mask_head (:obj:`ConfigDict` or dict): Config of mask head.
        train_cfg (:obj:`ConfigDict` or dict, Optional): Configuration
            information in train stage. Defaults to None.
        test_cfg (:obj:`ConfigDict` or dict, Optional): Configuration
            information in test stage. Defaults to None.
        init_cfg (:obj:`ConfigDict` or dict or list[:obj:`ConfigDict` or \
            dict]): Initialization config dict. Defaults to None.
    """

    def __init__(self,
                 num_stages: int = 6,
                 num_proposals: int = 300,
                 angle_version: str = 'le90',
                 stage_loss_weights: Union[List[float], Tuple[float]] = [1, 1, 1, 1, 1, 1],
                 bbox_roi_extractor: OptMultiConfig = None,
                 bbox_head: OptMultiConfig = None,
                 mask_roi_extractor: OptMultiConfig = None,
                 mask_head: OptMultiConfig = None,
                 shared_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        assert bbox_roi_extractor is not None
        assert bbox_head is not None
        assert shared_head is None, \
            'Shared head is not supported in Cascade RCNN anymore'

        self.num_stages = num_stages
        self.stage_loss_weights = stage_loss_weights
        super().__init__(
            num_stages=num_stages,
            stage_loss_weights=stage_loss_weights,
            bbox_roi_extractor=bbox_roi_extractor,
            bbox_head=bbox_head,
            mask_roi_extractor=mask_roi_extractor,
            mask_head=mask_head,
            shared_head=shared_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg)

        self.init_num_points = num_proposals
        self.angle_version = angle_version

    def _bbox_forward(self, stage: int, x: Tuple[Tensor],
                      rois: Tensor, weight, batch_img_metas) -> dict:
        """Box head forward function used in both training and testing.

        Args:
            stage (int): The current stage in Cascade RoI Head.
            x (tuple[Tensor]): List of multi-level img features.
            rois (Tensor): RoIs with the shape (n, 5) where the first
                column indicates batch id of each RoI.

        Returns:
             dict[str, Tensor]: Usually returns a dictionary with keys:

                - `cls_score` (Tensor): Classification scores.
                - `bbox_pred` (Tensor): Box energies / deltas.
                - `bbox_feats` (Tensor): Extract bbox RoI features.
        """
        num_imgs = len(batch_img_metas)
        bbox_roi_extractor = self.bbox_roi_extractor[stage]
        bbox_head = self.bbox_head[stage]
        roi_feats = bbox_roi_extractor(x[:bbox_roi_extractor.num_inputs],
                                        rois)   # bs*num_query, 256, 7, 7
        cls_score, bbox_pred, weight = bbox_head(
            roi_feats, weight, bs=num_imgs)   # [bs, num_query, 15], [bs, num_query, 5], [bs, num_query, 256]

        fake_bbox_results = dict(
            rois=rois,
            bbox_targets=(rois.new_zeros(len(rois), dtype=torch.long), None),
            bbox_pred=bbox_pred.view(-1, bbox_pred.size(-1)),
            cls_score=cls_score.view(-1, cls_score.size(-1)))
        fake_sampling_results = [
            InstanceData(pos_is_gt=rois.new_zeros(cls_score.size(1)))
            for _ in range(len(batch_img_metas))
        ]

        # refine_bboxes function: delta to bboxes
        results_list = bbox_head.refine_bboxes(
            sampling_results=fake_sampling_results,
            bbox_results=fake_bbox_results,
            batch_img_metas=batch_img_metas)
        proposal_list = [res.bboxes for res in results_list]
        bbox_results = dict(
            cls_score=cls_score,
            decoded_bboxes=torch.cat(proposal_list),
            weight=weight,
            # detach then use it in label assign
            detached_cls_scores=[
                cls_score[i].detach() for i in range(num_imgs)
            ],
            detached_proposals=[item.detach() for item in proposal_list])

        return bbox_results


    def bbox_loss(self, stage: int,
                  x: Tuple[Tensor],
                  weight: Tensor,
                  results_list: InstanceList,
                  batch_img_metas: List[dict],
                  batch_gt_instances: InstanceList) -> dict:
        """Run forward function and calculate loss for box head in training.

        Args:
            stage (int): The current stage in Cascade RoI Head.
            x (tuple[Tensor]): List of multi-level img features.
            sampling_results (list["obj:`SamplingResult`]): Sampling results.

        Returns:
            dict: Usually returns a dictionary with keys:

                - `cls_score` (Tensor): Classification scores.
                - `bbox_pred` (Tensor): Box energies / deltas.
                - `bbox_feats` (Tensor): Extract bbox RoI features.
                - `loss_bbox` (dict): A dictionary of bbox loss components.
                - `rois` (Tensor): RoIs with the shape (n, 5) where the first
                  column indicates batch id of each RoI.
                - `bbox_targets` (tuple):  Ground truth for proposals in a
                  single image. Containing the following list of Tensors:
                  (labels, label_weights, bbox_targets, bbox_weights)
        """
        proposal_list = [res.bboxes for res in results_list]
        rois = bbox2roi(proposal_list)  # [bs*num_query, 6] or [bs*num_query, 5]
        bbox_results = self._bbox_forward(stage, x, rois, weight, batch_img_metas)

        imgs_whwht = torch.cat(
            [res.imgs_whwht[None, ...] for res in results_list])
        cls_pred_list = bbox_results['detached_cls_scores']
        proposal_list = bbox_results['detached_proposals']

        sampling_results = []
        bbox_head = self.bbox_head[stage]
        for i in range(len(batch_img_metas)):
            pred_instances = InstanceData()
            # TODO: Enhance the logic
            pred_instances.bboxes = proposal_list[i]  # for assinger
            pred_instances.scores = cls_pred_list[i]
            pred_instances.priors = proposal_list[i]  # for sampler

            assign_result = self.bbox_assigner[stage].assign(
                pred_instances=pred_instances,
                gt_instances=batch_gt_instances[i],
                gt_instances_ignore=None,
                img_meta=batch_img_metas[i])

            if pred_instances.priors.shape[-1] == 4:
                gt_instances_sample = copy.deepcopy(batch_gt_instances[i])
                gt_instances_sample.bboxes = rbox2hbox(gt_instances_sample.bboxes)
            elif pred_instances.priors.shape[-1] == 5:
                gt_instances_sample = copy.deepcopy(batch_gt_instances[i])
            else:
                raise NotImplementedError

            sampling_result = self.bbox_sampler[stage].sample(
                # assign_result, pred_instances, batch_gt_instances[i])
                assign_result, pred_instances, gt_instances_sample)
            sampling_results.append(sampling_result)

        bbox_results.update(sampling_results=sampling_results)

        cls_score = bbox_results['cls_score']
        decoded_bboxes = bbox_results['decoded_bboxes']
        cls_score = cls_score.view(-1, cls_score.size(-1))
        bbox_loss_and_target = bbox_head.loss_and_target(
            cls_score,
            decoded_bboxes,
            sampling_results,
            self.train_cfg[stage],
            imgs_whwht=imgs_whwht,
            concat=True)
        bbox_results.update(bbox_loss_and_target)

        # propose for the new proposal_list
        proposal_list = []
        for idx in range(len(batch_img_metas)):
            results = InstanceData()
            results.imgs_whwht = results_list[idx].imgs_whwht
            results.bboxes = bbox_results['detached_proposals'][idx]
            proposal_list.append(results)
        bbox_results.update(results_list=proposal_list)
        return bbox_results


    def loss(self, x: Tuple[Tensor], results_list,
             batch_data_samples: SampleList) -> dict:
        """Perform forward propagation and loss calculation of the detection
        roi on the features of the upstream network.

        Args:
            x (tuple[Tensor]): List of multi-level img features.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: A dictionary of loss components
        """
        outputs = unpack_gt_instances(batch_data_samples)
        batch_gt_instances, batch_gt_instances_ignore, batch_img_metas \
            = outputs
        for item in batch_gt_instances:
            item.bboxes = get_box_tensor(item.bboxes)

        weight = []
        for idx in range(len(results_list)):
            weight.append(results_list[idx].weight)
        weight = torch.cat(weight, dim=0).unsqueeze(0) # bs*num_box, 256

        losses = {}
        for stage in range(self.num_stages):
            stage_loss_weight = self.stage_loss_weights[stage]

            # bbox head forward and loss
            bbox_results = self.bbox_loss(
                stage=stage,
                x=x,
                weight=weight,
                results_list=results_list,
                batch_img_metas=batch_img_metas,
                batch_gt_instances=batch_gt_instances)

            for name, value in bbox_results['loss_bbox'].items():
                losses[f's{stage}.{name}'] = (
                    value * stage_loss_weight if 'loss' in name else value)

            if self.with_mask:
                raise NotImplementedError('Segmentation not implement! Welcome to finish it!')

            weight = bbox_results['weight']
            results_list = bbox_results['results_list']
        return losses


    def predict_bbox(self,
                     x: Tuple[Tensor],
                     batch_img_metas: List[dict],
                     rpn_results_list: InstanceList,
                     rcnn_test_cfg: ConfigType,
                     rescale: bool = False,
                     **kwargs) -> InstanceList:
        """Perform forward propagation of the bbox head and predict detection
        results on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
            batch_img_metas (list[dict]): List of image information.
            rpn_results_list (list[:obj:`InstanceData`]): List of region
                proposals.
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of R-CNN.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.

        Returns:
            list[:obj:`InstanceData`]: Detection results of each image
            after the post process.
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        proposal_list = [res.bboxes for res in rpn_results_list]

        if all([proposal.shape[0] == 0 for proposal in proposal_list]):
            # There is no proposal in the whole batch
            return empty_instances(
                batch_img_metas, x[0].device, task_type='bbox')

        weight = []
        for idx in range(len(rpn_results_list)):
            weight.append(rpn_results_list[idx].weight)
        weight = torch.cat(weight, dim=0).unsqueeze(0) # bs*num_box, 256

        for stage in range(self.num_stages):
            rois = bbox2roi(proposal_list)
            bbox_results = self._bbox_forward(stage, x, rois, weight, batch_img_metas)
            weight = bbox_results['weight']
            cls_score = bbox_results['detached_cls_scores']
            proposal_list = bbox_results['detached_proposals']

        num_classes = self.bbox_head[-1].num_classes
        cls_score = torch.stack(cls_score, dim=0)
        if self.bbox_head[-1].loss_cls.use_sigmoid:
            cls_score = cls_score.sigmoid()
        else:
            cls_score = cls_score.softmax(-1)[..., :-1]

        topk_inds_list = []
        results_list = []
        for img_id in range(len(batch_img_metas)):
            cls_score_per_img = cls_score[img_id]
            scores_per_img, topk_inds = cls_score_per_img.flatten(0, 1).topk(
                self.test_cfg.max_per_img, sorted=False)
            labels_per_img = topk_inds % num_classes
            bboxes_per_img = proposal_list[img_id][topk_inds // num_classes]
            topk_inds_list.append(topk_inds)
            if rescale and bboxes_per_img.size(0) > 0:
                assert batch_img_metas[img_id].get('scale_factor') is not None
                scale_factor = bboxes_per_img.new_tensor(
                    batch_img_metas[img_id]['scale_factor']).repeat((1, 2))
                # Notice: Due to keep ratio when resize in data preparation,
                # the angle(radian) will not rescale.
                radian_factor = scale_factor.new_ones((scale_factor.size(0), 1))
                scale_factor = torch.cat([scale_factor, radian_factor], dim=-1)
                bboxes_per_img = (
                    bboxes_per_img.view(bboxes_per_img.size(0), -1, 5) /
                    scale_factor).view(bboxes_per_img.size()[0], -1)

            use_nms = self.test_cfg.use_nms
            if use_nms == True:
                nms_cfg = self.test_cfg.nms
                det_bboxes, keep_idxs = batched_nms(
                    bboxes_per_img, scores_per_img, labels_per_img,
                    nms_cfg)
                bboxes_per_img = bboxes_per_img[keep_idxs]
                labels_per_img = labels_per_img[keep_idxs]
                scores_per_img = det_bboxes[:, -1]

            results = InstanceData()
            results.bboxes = bboxes_per_img
            results.scores = scores_per_img
            results.labels = labels_per_img
            results_list.append(results)
        if self.with_mask:
            raise NotImplementedError('Segmentation not implement! Welcome to finish it!')
        return results_list

    def forward(self, x: Tuple[Tensor], rpn_results_list: InstanceList,
                batch_data_samples: SampleList) -> tuple:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

        Args:
            x (List[Tensor]): Multi-level features that may have different
                resolutions.
            rpn_results_list (list[:obj:`InstanceData`]): List of region
                proposals.
            batch_data_samples (list[:obj:`DetDataSample`]): Each item contains
                the meta information of each image and corresponding
                annotations.

        Returns
            tuple: A tuple of features from ``bbox_head`` and ``mask_head``
            forward.
        """
        # todo: finish this function
        results = ()
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]
        proposals = [rpn_results.bboxes for rpn_results in rpn_results_list]
        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = bbox2roi(proposals)
        # bbox head
        if self.with_bbox:
            rois, cls_scores, bbox_preds = self._refine_roi(
                x, rois, batch_img_metas, num_proposals_per_img)
            results = results + (cls_scores, bbox_preds)

            merged_masks = []
            for i in range(len(batch_img_metas)):
                aug_mask = [mask[i] for mask in aug_masks]
                merged_mask = merge_aug_masks(aug_mask, batch_img_metas[i])
                merged_masks.append(merged_mask)
            results = results + (merged_masks,)
        return results
