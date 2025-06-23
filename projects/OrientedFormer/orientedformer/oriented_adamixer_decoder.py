from typing import List, Tuple

import torch
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.models.task_modules.samplers import PseudoSampler
from mmrotate.registry import MODELS
from mmdet.structures import SampleList
from mmdet.structures.bbox import get_box_tensor
from mmdet.utils import ConfigType, InstanceList, OptConfigType
from mmdet.models.utils.misc import empty_instances, unpack_gt_instances
from mmdet.models.roi_heads.cascade_roi_head import CascadeRoIHead


@MODELS.register_module()
class OrientedAdaMixerDecoder(CascadeRoIHead):
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
                 stage_loss_weights: Tuple[float] = (1, 1, 1, 1, 1, 1),
                 content_dim: int = 256,
                 featmap_strides=[4, 8, 16, 32],
                 bbox_head: ConfigType = dict(
                     type='DIIHead',
                     num_classes=80,
                     num_fcs=2,
                     num_heads=8,
                     num_cls_fcs=1,
                     num_reg_fcs=3,
                     feedforward_channels=2048,
                     hidden_channels=256,
                     dropout=0.0,
                     roi_feat_size=7,
                     ffn_act_cfg=dict(type='ReLU', inplace=True)),
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 init_cfg: OptConfigType = None) -> None:
        assert bbox_head is not None
        assert len(stage_loss_weights) == num_stages
        self.num_stages = num_stages
        self.featmap_strides = featmap_strides
        self.stage_loss_weights = stage_loss_weights
        self.content_dim = content_dim
        super().__init__(
            num_stages=num_stages,
            stage_loss_weights=stage_loss_weights,
            bbox_roi_extractor=dict(
                # This does not mean that our method need RoIAlign. We put this
                # as a placeholder to satisfy the argument for the parent class
                # 'CascadeRoIHead'.
                type='RotatedSingleRoIExtractor',
                roi_layer=dict(
                    type='RoIAlignRotated',
                    out_size=7,
                    sample_num=2,
                    clockwise=True),
                out_channels=256,
                featmap_strides=[4, 8, 16, 32]),
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg)
        # train_cfg would be None when run the test.py
        if train_cfg is not None:
            for stage in range(num_stages):
                assert isinstance(self.bbox_sampler[stage], PseudoSampler), \
                    'Sparse R-CNN and QueryInst only support `PseudoSampler`'

    def bbox_loss(self, stage: int, x: Tuple[Tensor],
                  results_list: InstanceList, query_content: Tensor,
                  batch_img_metas: List[dict],
                  batch_gt_instances: InstanceList) -> dict:
        """Perform forward propagation and loss calculation of the bbox head on
        the features of the upstream network.

        Args:
            stage (int): The current stage in iterative process.
            x (tuple[Tensor]): List of multi-level img features.
            results_list (List[:obj:`InstanceData`]) : List of region
                proposals.
            query_content (Tensor): The object feature extracted from
                the previous stage.
            batch_img_metas (list[dict]): Meta information of each image.
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes``, ``labels``, and
                ``masks`` attributes.

        Returns:
            dict[str, Tensor]: Usually returns a dictionary with keys:

            - `cls_score` (Tensor): Classification scores.
            - `bbox_pred` (Tensor): Box energies / deltas.
            - `bbox_feats` (Tensor): Extract bbox RoI features.
            - `loss_bbox` (dict): A dictionary of bbox loss components.
        """
        xyzrt_list = [res.query_xyzrt for res in results_list]
        query_xyzrt = torch.stack(xyzrt_list)             # bs, num_query, 5
        bbox_results = self._bbox_forward(stage, x, query_xyzrt, query_content,
                                          batch_img_metas)
        imgs_whwht = torch.cat(
            [res.imgs_whwht[None, ...] for res in results_list])  # bs, num_query, 5
        cls_pred_list = bbox_results['detach_cls_score_list']     # bs*{num_query, 80}
        bboxes_list = bbox_results['detached_bboxes_list']        # bs*{num_query, 4}

        sampling_results = []
        bbox_head = self.bbox_head[stage]
        for i in range(len(batch_img_metas)):
            pred_instances = InstanceData()
            pred_instances.bboxes = bboxes_list[i]  # for assinger
            pred_instances.scores = cls_pred_list[i]
            pred_instances.priors = bboxes_list[i]  # for sampler

            assign_result = self.bbox_assigner[stage].assign(
                pred_instances=pred_instances,
                gt_instances=batch_gt_instances[i],
                gt_instances_ignore=None,
                img_meta=batch_img_metas[i])

            sampling_result = self.bbox_sampler[stage].sample(
                assign_result, pred_instances, batch_gt_instances[i])
            sampling_results.append(sampling_result)

        bbox_results.update(sampling_results=sampling_results)

        cls_score = bbox_results['cls_score']               # bs, num_query, num_class
        decoded_bboxes = bbox_results['decode_bbox_pred']   # bs, num_query, 5
        cls_score = cls_score.view(-1, cls_score.size(-1))  # bs, num_query, num_class
        decoded_bboxes = decoded_bboxes.view(-1, 5)         # bs*num_query, 5
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
            results.query_xyzrt = bbox_results['query_xyzrt'][idx].detach()
            proposal_list.append(results)
        bbox_results.update(results_list=proposal_list)
        return bbox_results

    def _bbox_forward(self, stage: int, x: Tuple[Tensor], query_xyzrt: Tensor,
                      query_content: Tensor,
                      batch_img_metas: List[dict]) -> dict:
        """Box head forward function used in both training and testing. Returns
        all regression, classification results and a intermediate feature.

        Args:
            stage (int): The current stage in iterative process.
            x (tuple[Tensor]): List of multi-level img features.
            query_xyzrt (Tensor): (bs, num_query, 5), where 5 represents (c_x, c_y, z, r, radian)
            query_content (Tensor): The object feature extracted from
                the previous stage. (bs, num_query, 256).
            batch_img_metas (list[dict]): Meta information of each image.

        Returns:
            dict[str, Tensor]: a dictionary of bbox head outputs,
            Containing the following results:

            - cls_score (Tensor): The score of each class, has
              shape (batch_size, num_proposals, num_classes)
              when use focal loss or
              (batch_size, num_proposals, num_classes+1)
              otherwise.
            - decoded_bboxes (Tensor): The regression results
              with shape (batch_size, num_proposal, 4).
              The last dimension 4 represents
              [tl_x, tl_y, br_x, br_y].
            - object_feats (Tensor): The object feature extracted
              from current stage
            - detached_cls_scores (list[Tensor]): The detached
              classification results, length is batch_size, and
              each tensor has shape (num_proposal, num_classes).
            - detached_proposals (list[tensor]): The detached
              regression results, length is batch_size, and each
              tensor has shape (num_proposal, 4). The last
              dimension 4 represents [tl_x, tl_y, br_x, br_y].
        """
        num_imgs = len(batch_img_metas)     # bs
        bbox_head = self.bbox_head[stage]
        cls_score, delta_xyzrt, query_content = bbox_head(x, query_xyzrt,
                                                         query_content,
                                                         featmap_strides=self.featmap_strides)  #(bs, num_query, 80),(bs,num_query,4),(bs,num_query,256)

        query_xyzrt, decoded_bboxes = self.bbox_head[stage].refine_xyzrt(
            query_xyzrt,
            delta_xyzrt) # (bs, num_query, 5), (bs, num_query, 5)

        bboxes_list = [bboxes for bboxes in decoded_bboxes]

        bbox_results = dict(
            cls_score=cls_score,                # bs, num_query, 80
            query_xyzrt=query_xyzrt,            # bs, num_query, 5
            decode_bbox_pred=decoded_bboxes,    # bs, num_query, 5
            query_content=query_content,        # bs, num_query, 256
            # detach then use it in label assign
            detach_cls_score_list=[
                cls_score[i].detach() for i in range(num_imgs)
            ],
            detached_bboxes_list=[item.detach() for item in bboxes_list],
        )

        return bbox_results


    def loss(self, x: Tuple[Tensor], rpn_results_list: InstanceList,
             batch_data_samples: SampleList) -> dict:
        """Perform forward propagation and loss calculation of the detection
        roi on the features of the upstream network.

        Args:
            x (tuple[Tensor]): List of multi-level img features.
            rpn_results_list (List[:obj:`InstanceData`]): List of region
                proposals.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: a dictionary of loss components of all stage.
        """
        outputs = unpack_gt_instances(batch_data_samples)
        batch_gt_instances, batch_gt_instances_ignore, batch_img_metas \
            = outputs
        for item in batch_gt_instances:
            item.bboxes = get_box_tensor(item.bboxes)

        query_content = torch.cat(
            [res.pop('query_content')[None, ...] for res in rpn_results_list])   # bs, num_query, 256
        results_list = rpn_results_list
        losses = {}
        for stage in range(self.num_stages):
            stage_loss_weight = self.stage_loss_weights[stage]

            # bbox head forward and loss
            bbox_results = self.bbox_loss(
                stage=stage,
                x=x,
                query_content=query_content,
                results_list=results_list,
                batch_img_metas=batch_img_metas,
                batch_gt_instances=batch_gt_instances)

            for name, value in bbox_results['loss_bbox'].items():
                losses[f's{stage}.{name}'] = (
                    value * stage_loss_weight if 'loss' in name else value)

            query_content = bbox_results['query_content']
            results_list = bbox_results['results_list']
        return losses

    def predict_bbox(self,
                     x: Tuple[Tensor],
                     batch_img_metas: List[dict],
                     rpn_results_list: InstanceList,
                     rcnn_test_cfg: ConfigType,
                     rescale: bool = False) -> InstanceList:
        """Perform forward propagation of the bbox head and predict detection
        results on the features of the upstream network.

        Args:
            x(tuple[Tensor]): Feature maps of all scale level.
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
            - bboxes (Tensor): Has a shape (num_instances, 5),
              the last dimension 4 arrange as (x, y, w, h, theta).
        """
        xyzrt_list = [res.query_xyzrt for res in rpn_results_list]
        query_xyzrt = torch.stack(xyzrt_list)  # bs, num_query, 5

        query_content = torch.cat(
            [res.pop('query_content')[None, ...] for res in rpn_results_list])   # bs, num_query, 256
        if all([xyzrt.shape[0] == 0 for xyzrt in xyzrt_list]):
            # There is no proposal in the whole batch
            return empty_instances(
                batch_img_metas, x[0].device, task_type='bbox')

        for stage in range(self.num_stages):
            bbox_results = self._bbox_forward(stage, x, query_xyzrt, query_content,
                                              batch_img_metas)
            query_content = bbox_results['query_content']
            cls_score = bbox_results['cls_score']
            bboxes_list = bbox_results['detached_bboxes_list']
            query_xyzrt = bbox_results['query_xyzrt']

        num_classes = self.bbox_head[-1].num_classes

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
            bboxes_per_img = bboxes_list[img_id][topk_inds // num_classes]
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

            results = InstanceData()
            results.bboxes = bboxes_per_img
            results.scores = scores_per_img
            results.labels = labels_per_img
            results_list.append(results)
        return results_list

    def forward(self, x: Tuple[Tensor], rpn_results_list: InstanceList,
                batch_data_samples: SampleList) -> tuple:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

        Args:
            x (List[Tensor]): Multi-level features that may have different
                resolutions.
            rpn_results_list (List[:obj:`InstanceData`]): List of region
                proposals.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns
            tuple: A tuple of features from ``bbox_head`` and ``mask_head``
            forward.
        """
        outputs = unpack_gt_instances(batch_data_samples)
        (batch_gt_instances, batch_gt_instances_ignore,
         batch_img_metas) = outputs

        all_stage_bbox_results = []
        query_content = torch.cat(
            [res.pop('query_content')[None, ...] for res in rpn_results_list])   # bs, num_query, 256
        results_list = rpn_results_list
        if self.with_bbox:
            for stage in range(self.num_stages):
                bbox_results = self.bbox_loss(
                    stage=stage,
                    x=x,
                    query_content=query_content,
                    results_list=results_list,
                    batch_img_metas=batch_img_metas,
                    batch_gt_instances=batch_gt_instances)
                bbox_results.pop('loss_bbox')
                # torch.jit does not support obj:SamplingResult
                bbox_results.pop('results_list')
                bbox_res = bbox_results.copy()
                bbox_res.pop('sampling_results')
                all_stage_bbox_results.append((bbox_res, ))

                if self.with_mask:
                    raise NotImplemented("Not Implement for Segmentation.")
        return tuple(all_stage_bbox_results)
