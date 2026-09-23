"""Frozen encoder/fusion/decoder wrapper used by stage-2 flow matching.

The oldnorm decoder predicts encoder-normalized pixels. The wrapper applies
ImageNet output denormalization and independent latent mean/variance
normalization. For sampling, load_encoder=False skips encoder construction.
"""

from math import sqrt
from typing import Optional

import torch
import torch.nn as nn

from encoders.vision_encoder import create_encoder
from .rae import _load_decoder, _load_normalization_stats

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


class RAECombine(nn.Module):
    """Shared plumbing: frozen DINOv3 encoder, decoder from a stage-1 train ckpt,
    optional latent stats normalization, ImageNet de-normalization on decode."""

    def __init__(self,
        encoder_name: str,
        combine_config,
        drop: bool = True,
        resolution: int = 256,
        decoder_config_path: str = 'configs/decoder/ViTXL',
        decoder_patch_size: int = 16,
        stage1_ckpt_path: Optional[str] = None,
        use_ema: bool = True,
        latent_dim: int = 1024,
        normalization_stat_path: Optional[str] = None,
        eps: float = 1e-5,
        load_encoder: bool = True,
        decoder_output_normalization: str = "encoder",
    ):
        super().__init__()
        if not load_encoder and stage1_ckpt_path is None:
            raise ValueError("Sampling requires a trained decoder stage1_ckpt_path")
        self.encoder = create_encoder(encoder_name,
                                      device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                                      resolution=resolution) if load_encoder else None
        self.resolution = resolution
        self.encoder_patch_size = self.encoder.patch_size if self.encoder is not None else decoder_patch_size
        self.latent_dim = latent_dim
        self.base_patches = (resolution // 16) ** 2
        self.eps = eps
        self.use_ema = use_ema
        if decoder_output_normalization not in ("encoder", "raw"):
            raise ValueError("decoder_output_normalization must be encoder or raw")
        self.decoder_output_normalization = decoder_output_normalization

        self.decoder = _load_decoder(
            decoder_config_path, latent_dim, decoder_patch_size,
            self.base_patches, pretrained_path=None,
        )
        self.latent_mean, self.latent_var, self.do_normalization = \
            _load_normalization_stats(normalization_stat_path)

        self.register_buffer('img_mean', torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer('img_std', torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

        self._combine_state = None
        self.checkpoint_metadata = {}
        if stage1_ckpt_path is not None:
            from utils.inference import load_decoder_checkpoint
            decoder_state, self._combine_state, self.checkpoint_metadata = load_decoder_checkpoint(stage1_ckpt_path, use_ema)
            self.decoder.load_state_dict(decoder_state, strict=True)
            print(f"{type(self).__name__}: loaded decoder from {stage1_ckpt_path}")

        from omegaconf import OmegaConf
        from .combine import MLSCombine
        from utils.inference import load_combine_weights
        cc = OmegaConf.to_container(combine_config, resolve=True) \
            if OmegaConf.is_config(combine_config) else dict(combine_config)
        if cc["target"] != "stage1.combine.MLSCombine":
            raise ValueError("The paper uses stage1.combine.MLSCombine")
        self.combine = MLSCombine(**cc["params"])
        load_combine_weights(self.combine, self._combine_state)
        self._combine_state = None
        checkpoint_layers = self.checkpoint_metadata.get("layers")
        if checkpoint_layers is not None and list(checkpoint_layers) != self.combine.layers:
            raise ValueError("Checkpoint and configured fusion layers do not match")
        if self.encoder is not None and self.encoder.layer_indices != self.combine.layers:
            raise ValueError("Encoder and fusion layers do not match")
        self.requires_grad_(False)
        self.drop = drop
        self._apply_combine_mode()

    def _imgs_to_norm(self, x: torch.Tensor) -> torch.Tensor:
        """[0,1] (or [0,255]) images at any size -> encoder-normalized at self.resolution."""
        if self.encoder is None:
            raise RuntimeError("Encoding requires load_encoder=True")
        if x.max() > 1.0:
            x = x / 255.0
        _, _, h, w = x.shape
        if h != self.resolution or w != self.resolution:
            x = nn.functional.interpolate(x, size=(self.resolution, self.resolution),
                                          mode='bicubic', align_corners=False)
        return (x - self.img_mean) / self.img_std

    def _layer_tokens(self, x_norm):
        return self.encoder(x_norm)

    def _tokens_to_latent(self, z: torch.Tensor) -> torch.Tensor:
        """[B, N, C] tokens -> [B, C, H, W] stats-normalized latent."""
        b, n, c = z.shape
        h = w = int(sqrt(n))
        z = z.transpose(1, 2).view(b, c, h, w)
        if self.do_normalization:
            latent_mean = self.latent_mean.to(z.device) if self.latent_mean is not None else 0
            latent_var = self.latent_var.to(z.device) if self.latent_var is not None else 1
            z = (z - latent_mean) / torch.sqrt(latent_var + self.eps)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        if self.do_normalization:
            latent_mean = self.latent_mean.to(z.device) if self.latent_mean is not None else 0
            latent_var = self.latent_var.to(z.device) if self.latent_var is not None else 1
            z = z * torch.sqrt(latent_var + self.eps) + latent_mean
        b, c, h, w = z.shape
        z = z.view(b, c, h * w).transpose(1, 2)
        output = self.decoder(z, drop_cls_token=False).logits
        # our stage-1 decoders predict in ImageNet-normalized space -> back to [0,1]
        return self.denormalize_output(self.decoder.unpatchify(output))

    def denormalize_output(self, image):
        return image * self.img_std + self.img_mean if self.decoder_output_normalization == "encoder" else image

    def forward(self, x: torch.Tensor, return_latent: bool = False):
        z = self.encode(x)
        x_rec = self.decode(z)
        return (x_rec, z) if return_latent else x_rec

    def _apply_combine_mode(self):
        if self.encoder is not None:
            self.encoder.eval()
        self.decoder.eval()
        self.combine.train(self.drop)

    def train(self, mode=True):
        # Frozen stage-1 weights; drop controls only the random layer subset.
        self.training = mode
        self._apply_combine_mode()
        return self

    @torch.no_grad()
    def encode(self, x):
        tokens = self._layer_tokens(self._imgs_to_norm(x))
        return self._tokens_to_latent(self.combine(tokens))

    @torch.no_grad()
    def encode_cond_target(self, x):
        """Subset input and deterministic full-fusion target from one encoder pass.

        The same fixed last-layer surrogate and normalization statistics apply
        to both. With p_drop=0, the two latent tensors are identical.
        """
        if not self.drop:
            raise ValueError("encode_cond_target requires drop=True")
        tokens = self._layer_tokens(self._imgs_to_norm(x))
        z_cond = self.combine(tokens)
        z_target = self.combine(tokens, idx=range(self.combine.K))
        return self._tokens_to_latent(z_cond), self._tokens_to_latent(z_target)
