from typing import Tuple
from torch import Tensor, nn
from mmdet.models.layers.transformer import inverse_sigmoid
from projects.rotated_rtdetr.rotated_rtdetr import RotatedRTDETRTransformerDecoder


class RotatedOVRTDETRTransformerDecoder(RotatedRTDETRTransformerDecoder):
    """Transformer decoder of Open-Vocabulary Rotated RT-DETR."""


    def forward(self, query: Tensor, value: Tensor, key_padding_mask: Tensor,
                self_attn_mask: Tensor, reference_points: Tensor,
                spatial_shapes: Tensor, level_start_index: Tensor,
                valid_ratios: Tensor, reg_branches: nn.ModuleList,
                cls_branches: nn.ModuleList, **kwargs) -> Tuple[Tensor]:
        """Forward function of Transformer decoder.

        Args:
            query (Tensor): The input query, has shape (num_queries, bs, dim).
            value (Tensor): The input values, has shape (num_value, bs, dim).
            key_padding_mask (Tensor): The `key_padding_mask` of `self_attn`
                input. ByteTensor, has shape (num_queries, bs).
            self_attn_mask (Tensor): The attention mask to prevent information
                leakage from different denoising groups and matching parts, has
                shape (num_queries_total, num_queries_total). It is `None` when
                `self.training` is `False`.
            reference_points (Tensor): The initial reference, has shape
                (bs, num_queries, 4) with the last dimension arranged as
                (cx, cy, w, h).
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels, ) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
            valid_ratios (Tensor): The ratios of the valid width and the valid
                height relative to the width and the height of features in all
                levels, has shape (bs, num_levels, 2).
            reg_branches: (obj:`nn.ModuleList`): Used for refining the
                regression results.
            cls_branches: (obj:`nn.ModuleList`): Used for classification
                results.

        Returns:
            tuple[Tensor]: Output queries and references of Transformer
                decoder

            - query (Tensor): Output embeddings of the last decoder, has
              shape (num_queries, bs, embed_dims) when `return_intermediate`
              is `False`. Otherwise, Intermediate output embeddings of all
              decoder layers, has shape (num_decoder_layers, num_queries, bs,
              embed_dims).
            - reference_points (Tensor): The reference of the last decoder
              layer, has shape (bs, num_queries, 5)  when `return_intermediate`
              is `False`. Otherwise, Intermediate references of all decoder
              layers, has shape (num_decoder_layers, bs, num_queries, 5). The
              coordinates are arranged as (cx, cy, w, h, radian)
        """
        assert self.return_intermediate
        assert reg_branches is not None
        assert reference_points.shape[-1] == 5
        # Reference_points have been processed through sigmoid() in pre_decoder().

        eval_idx = kwargs.pop('eval_idx', -1)
        if eval_idx < 0:
            eval_idx = eval_idx + self.num_layers
            assert eval_idx >= 0

        hidden_states = []
        all_layers_outputs_coords = []
        for lid, layer in enumerate(self.layers):
            num_levels = layer.cross_attn_cfg.num_levels
            reference_points_input = reference_points.unsqueeze(2).repeat(1, 1, num_levels, 1)
            reference_points_input[..., -1] *= self.angle_factor
            query_pos = self.ref_point_head(reference_points)

            query = layer(
                query,
                query_pos=query_pos,
                value=value,
                key_padding_mask=key_padding_mask,
                self_attn_mask=self_attn_mask,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                reference_points=reference_points_input,
                **kwargs)

            tmp = reg_branches[lid](query)

            if self.training or lid == eval_idx:
                hidden_states.append(query)
                all_layers_outputs_coords.append(
                    (tmp + inverse_sigmoid(reference_points, eps=1e-3)).sigmoid())

                if not self.training or lid == self.num_layers - 1:
                    break

            unact_reference_points = tmp + inverse_sigmoid(reference_points, eps=1e-3).detach()
            reference_points = unact_reference_points.sigmoid().detach()

        return hidden_states, all_layers_outputs_coords