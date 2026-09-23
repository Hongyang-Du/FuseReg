"""Shared config dataclasses used across all training scripts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple



@dataclass
class ModelConfig:
    """Generic model configuration for instantiate_from_config().
    Used for stage_1 (RAE) and stage_2 (DiT) model definitions.
    The params dict is passed as kwargs to the target class constructor.
    """
    target: str = ""
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MiscConfig:
    """Miscellaneous model-related parameters."""
    latent_size: List[int] = field(default_factory=lambda: [1024, 16, 16])  # [C, H, W]
    num_classes: int = 1000
    time_dist_shift_dim: int = 262144  # 16*16*1024
    time_dist_shift_base: int = 4096


@dataclass
class OptimizerConfig:
    """Optimizer configuration (shared across all training)."""
    lr: float = 2.0e-4
    betas: Tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.0
    eps: float = 1e-8
    # GMuon-specific
    momentum: float = 0.95
    nesterov: bool = True
    adamw_lr: Optional[float] = None
    ns_use_kernels: bool = False
    ns_coefficients_preset: str = "POLAR_EXPRESS_COEFFICIENTS"


@dataclass
class SchedulerConfig:
    """LR scheduler configuration."""
    warmup_epochs: float = 1.0
    warmup_steps: Optional[int] = None
    warmup_from_zero: bool = True
    decay_end_epoch: float = 16.0
    decay_end_steps: Optional[int] = None
    base_lr: float = 2.0e-4
    final_lr: float = 2.0e-5


@dataclass
class DatasetConfig:
    """Dataset configuration (shared across all training)."""
    type: str = "hf"  # Local Arrow dataset, or "imagefolder".
    data_dir: str = "data/imagenet-256"
    split: str = "train"


@dataclass
class TrainingConfig:
    """Base training configuration (shared across all)."""
    epochs: int = 16
    batch_size: int = 32
    global_batch_size: Optional[int] = None
    num_workers: int = 4
    global_seed: int = 0
    ema_decay: float = 0.9995
    clip_grad: Optional[float] = None
    log_interval: int = 100
    checkpoint_interval: int = 4
    sample_every: int = 2500
    grad_accum_steps: int = 1
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: Optional[SchedulerConfig] = None
    image_size: int = 256
