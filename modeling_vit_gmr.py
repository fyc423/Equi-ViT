# coding=utf-8
from __future__ import annotations
import os
import sys
from typing import Tuple, Union

import torch
import torch.nn as nn

# Reuse stock ViT pieces
from .modeling_vit import (
    ViTModel,
    ViTForImageClassification,
    ViTEmbeddings,
    ViTPatchEmbeddings,
)

__all__ = [
    "ViTPatchEmbeddingsGMR",
    "ViTEmbeddingsGMR",
    "ViTModelGMR",
    "ViTForImageClassificationGMR",
]

# Import GMR kernel
try:
    from gmr_conv import GMR_Conv2d
except ImportError as _e:
    raise ImportError(
        "FATAL: gmr_conv.GMR_Conv2d import failed.\n"
        "This model requires the GMR kernel.\n"
        "Install with: pip install gmr_conv\n"
        "Or follow README.md setup instructions.\n"
        f"Original error: {_e}"
    ) from _e


# -------------------------------
# Helpers
# -------------------------------
def _as_2tuple(x: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    return (x, x) if isinstance(x, int) else tuple(x)


def _assert_gmr_is_active(model: ViTForImageClassification) -> None:
    try:
        m = model.vit.embeddings.patch_embeddings.projection  # type: ignore[attr-defined]
        print(f"[INFO] ViT-GMR active. Patch projection layer: {m.__class__.__name__}")
    except Exception as e:
        print("[WARN] Could not confirm GMR patch projection:", repr(e))


# -------------------------------
# GMR-swapped modules
# -------------------------------

class ViTPatchEmbeddingsGMR(ViTPatchEmbeddings):
    """
    Replace ViT patch projection with GMRConv2d.
    Optionally, apply a 7x7 GMRConv2d + pooling within each 16x16 patch.
    """

    def __init__(self, config, image_size=None):
        super().__init__(config)

        # Normalize patch size to a 2-tuple
        ps = _as_2tuple(getattr(config, "patch_size", 16))

        self.patch_size = ps
        # Accept kernel_size1 and kernel_size2 as config args, default to 6 and 11
        self.kernel_size1 = _as_2tuple(getattr(config, "kernel_size1", 6))
        self.kernel_size2 = _as_2tuple(getattr(config, "kernel_size2", 11))
        self.inner_stride = 1  
        self.num_channels = config.num_channels
        self.hidden_size = config.hidden_size
        self.mid_channels = getattr(config, "mid_channels", 64)  # default 64


        # First GMR conv: in_channels -> mid_channels
        self.gmr1 = GMR_Conv2d(
            in_channels=self.num_channels,
            out_channels=self.mid_channels,
            kernel_size=self.kernel_size1,
            stride=1,
            padding=0,
            bias=False,
        )
        # Second GMR conv: mid_channels -> hidden_size
        self.gmr2 = GMR_Conv2d(
            in_channels=self.mid_channels,
            out_channels=self.hidden_size,
            kernel_size=self.kernel_size2,
            stride=1,
            padding=0,
            bias=False,
        )

    def forward(self, x, interpolate_pos_encoding=None):
        B, C, H, W = x.shape
        ps_h, ps_w = self.patch_size
        stride_h, stride_w = ps_h, ps_w
        patches = x.unfold(2, ps_h, stride_h).unfold(3, ps_w, stride_w) # [B, C, nH, nW, ps_h, ps_w]
        nH, nW = patches.shape[2], patches.shape[3]
        
        patches = patches.permute(0,2,3,1,4,5).reshape(-1, C, ps_h, ps_w) # [B*nH*nW, C, ps_h, ps_w]
        # Step 2: first GMRConv2d
        feats1 = self.gmr1(patches)  # [B*nH*nW, mid_channels, out_h, out_w]
        # Step 3: second GMRConv2d
        feats2 = self.gmr2(feats1)  # [B*nH*nW, hidden_size, out_h2, out_w2]
        # Step 4: global average pool within each patch
        pooled = feats2.mean(dim=[2,3])  # [B*nH*nW, hidden_size]
        # Step 5: reshape back to [B, num_patches, hidden_size]
        pooled = pooled.view(B, nH*nW, self.hidden_size)
        return pooled

# ...rest of the file unchanged...


class ViTEmbeddingsGMR(ViTEmbeddings):
    """
    Swap the patch embedding module to the GMR variant.
    """

    def __init__(self, config, use_mask_token: bool = False, image_size=None):
        # Match parent signature (HF versions may pass image_size)
        super().__init__(config, use_mask_token=use_mask_token)
        self.patch_embeddings = ViTPatchEmbeddingsGMR(config, image_size=image_size)
        # Position/class tokens and dropout remain identical.


class ViTModelGMR(ViTModel):
    """
    ViT backbone with GMR patch embeddings.
    """

    def __init__(self, config, add_pooling_layer: bool = False, use_mask_token: bool = False):
        # In HF ViTForImageClassification, add_pooling_layer=False is standard
        super().__init__(config, add_pooling_layer=add_pooling_layer, use_mask_token=use_mask_token)

        self.embeddings = ViTEmbeddingsGMR(config, use_mask_token=use_mask_token)
        # Encoder and layer norms remain unchanged.


class ViTForImageClassificationGMR(ViTForImageClassification):
    """
    Classification head atop ViTModelGMR. Classifier head is unchanged.
    """

    def __init__(self, config):
        super().__init__(config)
        # Swap the backbone to our GMR variant; classifier head stays the same.
        self.vit = ViTModelGMR(config, add_pooling_layer=False)
        _assert_gmr_is_active(self)