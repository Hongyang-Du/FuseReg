"""Validation and class-label dropout for full-target DiT training."""
import torch

from configs.stage2 import Stage2Config


def validate_stage2_config(config: Stage2Config) -> None:
    if not config.stage_1.target or not config.stage_2.target:
        raise ValueError("Provide stage_1.target and stage_2.target")
    if config.training.grad_accum_steps < 1:
        raise ValueError("training.grad_accum_steps must be >= 1")
    if not 0 <= config.conditioning.cfg_dropout_prob <= 1:
        raise ValueError("conditioning.cfg_dropout_prob must be in [0, 1]")


def apply_cfg_dropout(model_conds, model_conds_null, cfg_dropout_prob=0.1):
    labels = model_conds["context"]
    mask = torch.rand(labels.shape[0], device=labels.device) < cfg_dropout_prob
    return {"context": torch.where(mask, model_conds_null["context"], labels)}, mask
