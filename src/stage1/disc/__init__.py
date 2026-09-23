"""Discriminator and losses used by the paper decoder training recipe."""
from .discriminator import DinoDiscriminator
from .gan_loss import hinge_d_loss, vanilla_g_loss, calculate_adaptive_weight

__all__ = ["DinoDiscriminator", "hinge_d_loss", "vanilla_g_loss", "calculate_adaptive_weight"]
