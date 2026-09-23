"""Configuration for the paper DINOv3 fusion decoder training recipe."""
from dataclasses import dataclass, field
from typing import Optional
from .shared import ModelConfig

@dataclass
class DecoderModuleConfig:
    config_path: str = "configs/decoder/ViTXL"
    latent_dim: int = 1024
    patch_size: int = 16
    num_patches: int = 256

@dataclass
class GanConfig:
    disc_weight: float = 0.75
    disc_start: int = 1            # epoch GAN turns on
    disc_ckpt: str = "pretrained_models/encoders/dino/dino_vit_small_patch8_224.pth"

@dataclass
class LossConfig:
    lpips_w: float = 1.0
    gan: GanConfig = field(default_factory=GanConfig)

@dataclass
class DataConfig:
    data_dir: str = "data/imagenet-256"
    image_size: int = 256
    num_workers: int = 4
    val_npz: str = "data_eval/imagenet-256-val.npz"
    val_n: int = 1000

@dataclass
class TrainingConfig:
    epochs: int = 16
    batch_size: int = 32          # per GPU (micro-batch when grad_accum_steps > 1)
    grad_accum_steps: int = 1     # accumulate N micro-batches per optimizer step
    lr: float = 8.0e-4
    warmup_epochs: int = 2
    ema_decay: float = 0.9995
    clip_grad: float = 1.0
    precision: str = "bf16"       # fp32 | bf16
    ckpt_every: int = 1
    log_every: int = 50
    seed: int = 42
    out_dir: str = "output_full/decoder_run"

@dataclass
class ProbeConfig:
    loo_solo: str = "final"       # off | final | every
    val_image: Optional[str] = None

@dataclass
class WandbConfig:
    enabled: bool = False
    project: str = "fusereg"
    entity: Optional[str] = None
    name: Optional[str] = None

@dataclass
class DecoderConfig:
    encoder_name: str = "dinov3mls-vit-l16"
    combine: ModelConfig = field(default_factory=ModelConfig)
    decoder: DecoderModuleConfig = field(default_factory=DecoderModuleConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    probe: ProbeConfig = field(default_factory=ProbeConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
