from typing import Dict, List, Tuple, Union, Optional
import torch
from torch import Tensor
from mmdet.structures import OptSampleList, SampleList
from projects.rotated_rtdetr.rotated_rtdetr import RotatedRTDETR
from .rotated_ov_rtdetr_layers import RotatedOVRTDETRTransformerDecoder


class RotatedOVRTDETR(RotatedRTDETR):

    def _init_layers(self) -> None:
        decoder_cfg = self.decoder
        super()._init_layers()
        self.decoder = RotatedOVRTDETRTransformerDecoder(**decoder_cfg)

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (bs, dim, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components
        """
        img_feats, txt_feats = self.extract_feat(
            batch_inputs, batch_data_samples)
        head_inputs_dict = self.forward_transformer(img_feats, txt_feats,
                                                    batch_data_samples)
        losses = self.bbox_head.loss(
            **head_inputs_dict, batch_data_samples=batch_data_samples)

        return losses

    def forward_transformer(
        self,
        img_feats: Tuple[Tensor],
        txt_feats: Tensor,
        batch_data_samples: OptSampleList = None,
    ) -> Dict:
        """Forward process of Transformer.

        The forward procedure of the transformer is defined as:
        'pre_transformer' -> 'encoder' -> 'pre_decoder' -> 'decoder'
        More details can be found at `TransformerDetector.forward_transformer`
        in `mmdet/detector/base_detr.py`.
        The difference is that the ground truth in `batch_data_samples` is
        required for the `pre_decoder` to prepare the query of DINO.
        Additionally, DINO inherits the `pre_transformer` method and the
        `forward_encoder` method of DeformableDETR. More details about the
        two methods can be found in `mmdet/detector/deformable_detr.py`.

        Args:
            img_feats (tuple[Tensor]): Tuple of feature maps from neck. Each
                feature map has shape (bs, dim, H, W).
            txt_feats (Tensor): Clip text feat.
                shape (bs, text_len, 512)
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
                Defaults to None.

        Returns:
            dict: The dictionary of bbox_head function inputs, which always
            includes the `hidden_states` of the decoder output and may contain
            `references` including the initial and intermediate references.
        """
        encoder_inputs_dict, decoder_inputs_dict = self.pre_transformer(
            img_feats, txt_feats, batch_data_samples)

        encoder_outputs_dict = self.forward_encoder(**encoder_inputs_dict)

        tmp_dec_in, head_inputs_dict = self.pre_decoder(
            **encoder_outputs_dict, batch_data_samples=batch_data_samples)
        decoder_inputs_dict.update(tmp_dec_in)

        decoder_outputs_dict = self.forward_decoder(**decoder_inputs_dict)
        head_inputs_dict.update(decoder_outputs_dict, txt_feats=txt_feats)
        return head_inputs_dict

    def pre_transformer(self, img_feats, txt_feats, batch_data_samples):
        encoder_inputs_dict, decoder_inputs_dict = super().pre_transformer(
            img_feats, batch_data_samples)
        encoder_inputs_dict.update(txt_feats=txt_feats)
        return encoder_inputs_dict, decoder_inputs_dict

    def forward_encoder(self, mlvl_feats: Tuple[Tensor],
                        txt_feats: Tensor,
                        spatial_shapes: Tensor) -> Dict:
        encoder_outputs_dict = super().forward_encoder(mlvl_feats, spatial_shapes)
        encoder_outputs_dict.update(txt_feats=txt_feats)
        return encoder_outputs_dict

    def pre_decoder(
        self,
        memory: Tensor,
        memory_mask: Tensor,
        spatial_shapes: Tensor,
        txt_feats: Tensor,
        batch_data_samples: OptSampleList = None,
    ) -> Tuple[Dict]:
        """Prepare intermediate variables before entering Transformer decoder,
        such as `query`, `query_pos`, and `reference_points`.

        Args:
            memory (Tensor): The output embeddings of the Transformer encoder,
                has shape (bs, num_feat_points, dim).
            memory_mask (Tensor): ByteTensor, the padding mask of the memory,
                has shape (bs, num_feat_points). Will only be used when
                `as_two_stage` is `True`.
            spatial_shapes (Tensor): Spatial shapes of features in all levels.
                With shape (num_levels, 2), last dimension represents (h, w).
                Will only be used when `as_two_stage` is `True`.
            txt_feats (Tensor): Clip text feat. Shape (bs, text_len, 512).
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
                Defaults to None.

        Returns:
            tuple[dict]: The decoder_inputs_dict and head_inputs_dict.

            - decoder_inputs_dict (dict): The keyword dictionary args of
              `self.forward_decoder()`, which includes 'query', 'memory',
              `reference_points`, and `dn_mask`. The reference points of
              decoder input here are 4D boxes, although it has `points`
              in its name.
            - head_inputs_dict (dict): The keyword dictionary args of the
              bbox_head functions, which includes `topk_score`, `topk_coords`,
              and `dn_meta` when `self.training` is `True`, else is empty.
        """
        bs, _, c = memory.shape
        cls_out_features = self.bbox_head.num_classes

        output_memory, output_proposals = self.gen_encoder_output_proposals(
            memory, memory_mask, spatial_shapes)
        cls_embed = self.bbox_head.cls_branches[
            self.decoder.num_layers](
                output_memory)
        enc_outputs_class =self.bbox_head.cls_contrasts[
            self.decoder.num_layers](cls_embed, txt_feats)

        # NOTE The DINO selects top-k proposals according to scores of
        # multi-class classification, while DeformDETR, where the input
        # is `enc_outputs_class[..., 0]` selects according to scores of
        # binary classification.
        topk_indices = torch.topk(
            enc_outputs_class.max(-1)[0], k=self.num_queries, dim=1)[1]

        query = torch.gather(output_memory, 1,
                             topk_indices.unsqueeze(-1).repeat(1, 1, c))
        topk_output_proposals = torch.gather(
            output_proposals, 1,
            topk_indices.unsqueeze(-1).repeat(1, 1, 5))
        topk_coords_unact = self.bbox_head.reg_branches[
            self.decoder.num_layers](query) + topk_output_proposals

        if self.training:
            topk_score = torch.gather(
                enc_outputs_class, 1,
                topk_indices.unsqueeze(-1).repeat(1, 1, cls_out_features))
            topk_coords = topk_coords_unact.sigmoid()
            topk_coords_unact = topk_coords_unact.detach()

            dn_label_query, dn_bbox_query, dn_mask, dn_meta = \
                self.dn_query_generator(batch_data_samples)
            query = query.detach()  # detach() is not used in DINO
            query = torch.cat([dn_label_query, query], dim=1)
            dn_bbox_query = dn_bbox_query.type_as(topk_coords_unact)
            reference_points = torch.cat([dn_bbox_query, topk_coords_unact],
                                         dim=1)
        else:
            reference_points = topk_coords_unact
            dn_mask, dn_meta = None, None
        reference_points = reference_points.sigmoid()

        decoder_inputs_dict = dict(
            query=query,
            memory=memory,
            reference_points=reference_points,
            dn_mask=dn_mask,
            cls_branches=self.bbox_head.cls_branches,
            eval_idx=self.eval_idx)
        # NOTE DINO calculates encoder losses on scores and coordinates
        # of selected top-k encoder queries, while DeformDETR is of all
        # encoder queries.
        head_inputs_dict = dict(
            enc_outputs_class=topk_score,
            enc_outputs_coord=topk_coords,
            dn_meta=dn_meta) if self.training else dict()
        return decoder_inputs_dict, head_inputs_dict

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs (Tensor): Inputs, has shape (bs, dim, H, W).
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
            rescale (bool): Whether to rescale the results.
                Defaults to True.

        Returns:
            list[:obj:`DetDataSample`]: Detection results of the input images.
            Each DetDataSample usually contain 'pred_instances'. And the
            `pred_instances` usually contains following keys.

            - scores (Tensor): Classification scores, has a shape
              (num_instance, )
            - labels (Tensor): Labels of bboxes, has a shape
              (num_instances, ).
            - bboxes (Tensor): Has a shape (num_instances, 4),
              the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        img_feats, txt_feats = self.extract_feat(batch_inputs, batch_data_samples)
        head_inputs_dict = self.forward_transformer(img_feats, txt_feats,
                                                    batch_data_samples)
        results_list = self.bbox_head.predict(
            **head_inputs_dict,
            rescale=rescale,
            batch_data_samples=batch_data_samples)
        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples

    def extract_feat(
            self, batch_inputs: Tensor,
            batch_data_samples: SampleList) -> Tuple[Tuple[Tensor], Tensor]:
        """Extract features."""
        txt_feats = None
        if batch_data_samples is None:
            texts = self.texts
            txt_feats = self.text_feats
        elif isinstance(batch_data_samples,
                        dict) and 'texts' in batch_data_samples:
            texts = batch_data_samples['texts']
        elif isinstance(batch_data_samples, list) and hasattr(
                batch_data_samples[0], 'texts'):
            texts = [data_sample.texts for data_sample in batch_data_samples]
        elif hasattr(self, 'text_feats'):
            texts = self.texts
            txt_feats = self.text_feats
        else:
            raise TypeError('batch_data_samples should be dict or list.')
        if txt_feats is not None:
            # forward image only
            img_feats = self.backbone.forward_image(batch_inputs)
        else:
            img_feats, txt_feats = self.backbone(batch_inputs, texts)
        if self.with_neck:
            img_feats = self.neck(img_feats)
        return img_feats, txt_feats