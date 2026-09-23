"""The paper's internal guidance: base + scale * (full - base)."""

from functools import partial

import torch

from configs.stage2 import GuidanceConfig


def forward_with_internalguidance(model, x, t, ig_scale, ig_interval=(0, 1), **condition_kwargs):
    """Apply internal guidance to a single class-conditional batch."""
    full, base = model(x, t, **condition_kwargs)
    t_min, t_max = ig_interval
    if not t_min < t_max:
        raise ValueError("ig_interval must have min < max")
    active = ((t >= t_min) & (t <= t_max)).view(-1, *([1] * (full.ndim - 1)))
    return torch.where(active, base + ig_scale * (full - base), full)


def get_model_forward_fn(model, guid_cfg: GuidanceConfig):
    if guid_cfg.ig.scale != 1.0:
        return partial(forward_with_internalguidance, model), {
            "ig_scale": guid_cfg.ig.scale,
            "ig_interval": (guid_cfg.ig.t_min, guid_cfg.ig.t_max),
        }
    return model.forward, {}
