from typing import Tuple
import torch
from torch import Tensor
import torch.nn as nn
from mmdet.structures import SampleList
from .rotated_ov_rtdetr import RotatedOVRTDETR


class UniRotatedOVRTDETR(RotatedOVRTDETR):

    def __init__(self,
                 *args,
                 text_channels: int = 512,
                 num_txt_prompts: int = 1,
                 embedding_path: str = '',
                 freeze_prompt: bool = False,
                 use_mlp_adapter: bool = False,
                 freeze_backbone: bool = True,
                 freeze_neck: bool = True,
                 freeze_encoder: bool = True,
                 freeze_decoder: bool = True,
                 freeze_head: bool = True,
                 freeze_dn_query_generator: bool = True,
                 **kwargs) -> None:
        """Initialize the detector and its learnable text prompts.

        Args:
            text_channels (int): Dimension of each text prompt embedding.
                Defaults to 512.
            num_txt_prompts (int): Number of text embeddings. Defaults to 1.
            embedding_path (str): Optional path to a ``.npy`` file containing
                the initial prompt embeddings. Defaults to an empty string.
            freeze_prompt (bool): Whether to exclude the prompt embeddings
                from optimization. Defaults to False.
            use_mlp_adapter (bool): Whether to refine prompt embeddings with
                an MLP adapter. Defaults to False.
        """
        bbox_head_cfg = kwargs.get('bbox_head')
        if bbox_head_cfg is not None:
            bbox_head_cfg = bbox_head_cfg.copy()
            bbox_head_cfg['text_channels'] = text_channels
            kwargs['bbox_head'] = bbox_head_cfg
        super().__init__(*args, **kwargs)

        if self.bbox_head.num_classes != num_txt_prompts:
            raise ValueError(f'{num_txt_prompts} != {self.bbox_head.num_classes}')

        if len(embedding_path) > 0:
            import numpy as np
            self.embeddings = torch.nn.Parameter(
                torch.from_numpy(np.load(embedding_path)).float())
            if self.embeddings.shape != (num_txt_prompts, text_channels):
                raise ValueError(f'text embed shape {tuple(self.embeddings.shape)}')
        else:
            # random init
            embeddings = nn.functional.normalize(torch.randn(
                (num_txt_prompts, text_channels)),
                dim=-1)
            self.embeddings = nn.Parameter(embeddings)

        if freeze_prompt:
            self.embeddings.requires_grad = False
        else:
            self.embeddings.requires_grad = True

        if use_mlp_adapter:
            self.adapter = nn.Sequential(
                nn.Linear(text_channels, text_channels * 2), nn.ReLU(True),
                nn.Linear(text_channels * 2, text_channels))
        else:
            self.adapter = None

        self.freeze_backbone = freeze_backbone
        self.freeze_neck = freeze_neck
        self.freeze_encoder = freeze_encoder
        self.freeze_decoder = freeze_decoder
        self.freeze_head = freeze_head
        self.freeze_dn_query_generator = freeze_dn_query_generator

    @staticmethod
    def _freeze_module(module: nn.Module) -> None:
        module.eval()
        for param in module.parameters():
            param.requires_grad_(False)

    def _freeze_modules(self) -> None:
        if self.freeze_backbone:
            self._freeze_module(self.backbone)
        if self.freeze_neck and self.with_neck:
            self._freeze_module(self.neck)
        if self.freeze_encoder:
            self._freeze_module(self.encoder)
        if self.freeze_decoder:
            self._freeze_module(self.decoder)
            self._freeze_module(self.memory_trans_fc)
            self._freeze_module(self.memory_trans_norm)
        if self.freeze_head:
            self._freeze_module(self.bbox_head)
        if self.freeze_dn_query_generator:
            self._freeze_module(self.dn_query_generator)

    def train(self, mode: bool = True) -> 'UniRotatedOVRTDETR':
        """Keep frozen modules in evaluation mode during training."""
        super().train(mode)
        self._freeze_modules()
        return self

    def extract_feat(
            self, batch_inputs: Tensor,
            batch_data_samples: SampleList) -> Tuple[Tuple[Tensor], Tensor]:
        """Extract features."""
        # only image features
        img_feats, _ = self.backbone(batch_inputs, text=None)
        # use embeddings
        txt_feats = self.embeddings[None]
        if self.adapter is not None:
            txt_feats = self.adapter(txt_feats) + txt_feats
            txt_feats = nn.functional.normalize(txt_feats, dim=-1, p=2)
        txt_feats = txt_feats.repeat(img_feats[0].shape[0], 1, 1)
        if self.with_neck:
            img_feats = self.neck(img_feats)
        return img_feats, txt_feats