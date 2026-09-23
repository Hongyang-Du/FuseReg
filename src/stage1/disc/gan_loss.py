"""Adapted and modified from https://github.com/CompVis/taming-transformers"""

import torch
import torch.nn.functional as F


def hinge_d_loss(logits_real, logits_fake) -> torch.Tensor:
    """Hinge discriminator loss used by VQGAN."""
    loss_real = torch.mean(F.relu(1.0 - logits_real))
    loss_fake = torch.mean(F.relu(1.0 + logits_fake))
    return 0.5 * (loss_real + loss_fake)


def vanilla_g_loss(logits_fake) -> torch.Tensor:
    """Generator loss used with the hinge discriminator."""
    return -torch.mean(logits_fake)


def calculate_adaptive_weight(
    recon_loss: torch.Tensor,
    gan_loss: torch.Tensor,
    layer: torch.nn.Parameter,
    max_d_weight: float = 1e4,
) -> torch.Tensor:
    """Calculate adaptive weight for GAN loss based on gradient norms.

    Args:
        recon_loss: Reconstruction loss tensor
        gan_loss: GAN loss tensor
        layer: The layer parameter to compute gradients with respect to
        max_d_weight: Maximum discriminator weight clamp value

    Returns:
        Adaptive weight for balancing reconstruction and GAN losses
    """
    torch._functorch.config.donated_buffer = False  # Allow retain_graph=True with torch.compile
    recon_grads = torch.autograd.grad(recon_loss, layer, retain_graph=True)[0]
    gan_grads = torch.autograd.grad(gan_loss, layer, retain_graph=True)[0]
    d_weight = torch.norm(recon_grads) / (torch.norm(gan_grads) + 1e-6)
    d_weight = torch.clamp(d_weight, 0.0, max_d_weight)
    return d_weight.detach()
