"""Frozen DINOv3 ViT-L/16 encoder used by the paper."""
import re

import torch
from torch import nn


class DINOv3Encoder(nn.Module):
    def __init__(self, layer_indices, device, resolution=256):
        super().__init__()
        from .models.dinov3_loader import load_dinov3
        self.layer_indices = list(layer_indices)
        if not self.layer_indices or any(i < 0 or i > 23 for i in self.layer_indices):
            raise ValueError("DINOv3 ViT-L layer indices must lie in 0..23")
        if self.layer_indices != sorted(set(self.layer_indices)):
            raise ValueError("Layer indices must be sorted and unique")
        self.model = load_dinov3().to(device).eval()
        self.hidden_size = self.model.embed_dim
        self.patch_size = 16
        self.resolution = resolution
        # oldnorm checkpoints use non-affine output LayerNorm.
        self.model.norm = nn.LayerNorm(self.hidden_size, elementwise_affine=False)
        self.requires_grad_(False)

    def forward(self, normalized_images):
        """Return the selected normalized block outputs, each shaped [B,N,C]."""
        return list(self.model.get_intermediate_layers(
            normalized_images, n=self.layer_indices, reshape=False,
            return_class_token=False, norm=True))


def create_encoder(encoder_string, device, resolution=256):
    match = re.fullmatch(r"dinov3mls-vit-l16\[layers=([0-9.]+)\]", encoder_string)
    if match is None:
        raise ValueError("Use dinov3mls-vit-l16[layers=1.2...23] with explicit block indices")
    return DINOv3Encoder([int(i) for i in match[1].split('.')], device, resolution)
