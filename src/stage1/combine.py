"""Paper fusion: a retained-layer mean plus the fixed final-layer token mean.

Each training image independently keeps layers with probability ``1 - p_drop``.
As in the released checkpoint's training code, an empty draw is repaired by
uniformly keeping one layer. Evaluation uses all layers, or an explicit subset.
The final-layer surrogate is retained even when that layer is outside the subset.
"""
import torch
from torch import nn


class MLSCombine(nn.Module):
    def __init__(self, layers, weighting="random_drop", p_drop=0.3):
        super().__init__()
        self.layers = list(layers)
        if not self.layers or len(set(self.layers)) != len(self.layers):
            raise ValueError("layers must be nonempty and unique")
        if weighting not in ("mean", "random_drop"):
            raise ValueError("weighting must be mean or random_drop")
        if not 0 <= p_drop <= 1:
            raise ValueError("p_drop must be between 0 and 1")
        self.K = len(self.layers)
        self.weighting = weighting
        self.p_drop = p_drop

    @property
    def has_params(self):
        return False

    def forward(self, layer_tokens, idx=None):
        if len(layer_tokens) != self.K:
            raise ValueError(f"Expected {self.K} layer tensors, got {len(layer_tokens)}")
        stk = torch.stack(layer_tokens, dim=0)
        if idx is not None:
            idx = list(idx)
            if not idx or len(set(idx)) != len(idx) or any(i < 0 or i >= self.K for i in idx):
                raise ValueError("Subset indices must be nonempty, unique positions in layers")
            fused = stk[idx].mean(0)
        elif self.training and self.weighting == "random_drop" and self.p_drop > 0:
            K, B = stk.shape[:2]
            keep = torch.rand(K, B, device=stk.device) > self.p_drop
            dead = ~keep.any(0)
            if dead.any():
                keep[torch.randint(K, (int(dead.sum()),), device=stk.device), dead] = True
            weights = keep.to(stk.dtype)
            weights = weights / weights.sum(0, keepdim=True).clamp_min(1e-6)
            fused = (weights.view(K, B, 1, 1) * stk).sum(0)
        else:
            fused = stk.mean(0)
        return fused + stk[-1].mean(dim=1, keepdim=True)
