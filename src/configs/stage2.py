"""Class-conditional, full-target DiT configuration for the paper experiments."""
from dataclasses import dataclass, field

from .shared import DatasetConfig, MiscConfig, ModelConfig, TrainingConfig


@dataclass
class TransportConfig:
    # x-prediction and logit-normal time sampling are fixed by the oldnorm recipe.
    t_eps: float = 0.05


@dataclass
class SamplerConfig:
    num_steps: int = 50


@dataclass
class IGConfig:
    scale: float = 1.0
    t_min: float = 0.0
    t_max: float = 1.0


@dataclass
class GuidanceConfig:
    ig: IGConfig = field(default_factory=IGConfig)

    @property
    def use_ig(self):
        return self.ig.scale != 1.0

    @property
    def any_guidance_active(self):
        return self.use_ig


@dataclass
class ConditioningArchConfig:
    num_t_tokens: int = 4
    num_c_tokens: int = 8


@dataclass
class ConditioningConfig:
    cfg_dropout_prob: float = 0.1
    arch: ConditioningArchConfig = field(default_factory=ConditioningArchConfig)


@dataclass
class InternalGuidanceConfig:
    base_model_coeff: float = 1.0


@dataclass
class Stage2Config:
    stage_1: ModelConfig = field(default_factory=ModelConfig)
    stage_2: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    transport: TransportConfig = field(default_factory=TransportConfig)
    sampler: SamplerConfig = field(default_factory=SamplerConfig)
    guidance: GuidanceConfig = field(default_factory=GuidanceConfig)
    conditioning: ConditioningConfig = field(default_factory=ConditioningConfig)
    misc: MiscConfig = field(default_factory=MiscConfig)
    internal_guidance: InternalGuidanceConfig = field(default_factory=InternalGuidanceConfig)

    def prepare_model_params(self):
        self.stage_2.params.setdefault("num_classes", self.misc.num_classes)
        self.stage_2.params.setdefault("cond_arch", self.conditioning.arch)
