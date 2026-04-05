# models/vit_vanilla.py
import os
import torch.nn as nn
from transformers import AutoConfig

class HFViTWrapper(nn.Module):
    def __init__(
        self,
        num_labels: int,
        checkpoint: str = "google/vit-base-patch16-224",
        pretrained: bool = True,
        use_gmrx2: bool = True,
        patch_size: int = 16,    
        kernel_size1: int = 6,   
        kernel_size2: int = 11,   
        image_size: int = 224,
    ):
        super().__init__()



        print("[INFO] Using GMRx2 ViT model!")
        from transformers.models.vit.modeling_vit_gmr import ViTForImageClassificationGMR as ViTCls
        if pretrained:
            self.model = ViTCls.from_pretrained(
                checkpoint,
                num_labels=num_labels,
                ignore_mismatched_sizes=True)
        else:
            cfg = AutoConfig.from_pretrained(checkpoint)
            cfg.num_labels = num_labels
            cfg.patch_size = patch_size
            cfg.image_size = image_size
            cfg.kernel_size1 = kernel_size1
            cfg.kernel_size2 = kernel_size2
            self.model = ViTCls(cfg)
                
    def forward(self, x, interpolate_pos_encoding=False):
        return self.model(pixel_values=x, interpolate_pos_encoding=interpolate_pos_encoding).logits
