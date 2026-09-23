"""Reconstruction metric used during decoder training."""
import torch

@torch.no_grad()
def psnr(x: torch.Tensor, target: torch.Tensor, eps: float = 1e-10) -> float:
    """Mean PSNR in dB between two [B, C, H, W] images in [0, 1].

    Per-image MSE -> per-image PSNR -> averaged over the batch (the standard
    reconstruction-quality convention). Higher is better.
    """
    mse = ((x.detach().float() - target.detach().float()) ** 2).flatten(1).mean(1)  # [B]
    return (-10.0 * torch.log10(mse + eps)).mean().item()
