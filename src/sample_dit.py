#!/usr/bin/env python3
"""Sample an unguided class-conditional grid from an EMA DiT and paired decoder."""

import argparse
import dataclasses
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from omegaconf import OmegaConf
from torchvision.utils import save_image

from configs.stage2 import Stage2Config
from stage2.transport import create_sampler, create_transport
from utils.model_utils import instantiate_from_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",  required=True, help="Stage-2 training YAML")
    ap.add_argument("--ckpt",    required=True, help="Stage-2 checkpoint (ep-*.pt)")
    ap.add_argument("--classes", type=int, nargs="+", default=None,
                    help="ImageNet class ids; default: 16 spread over 0..999")
    ap.add_argument("--per-class", type=int, default=1)
    ap.add_argument("--steps",   type=int, default=50)
    ap.add_argument("--seed",    type=int, default=42)
    ap.add_argument("--raw",     action="store_true", help="Use raw weights instead of EMA")
    ap.add_argument("--out",     required=True)
    ap.add_argument("--device",  default="cuda:0")
    ap.add_argument("opts", nargs="*", help="Config overrides")
    args = ap.parse_args()

    device = torch.device(args.device)
    config: Stage2Config = OmegaConf.to_object(
        OmegaConf.merge(OmegaConf.structured(Stage2Config), OmegaConf.load(args.config), OmegaConf.from_dotlist(args.opts)))
    config.prepare_model_params()

    config.stage_1.params["load_encoder"] = False
    rae = instantiate_from_config(config.stage_1).to(device).eval()

    model = instantiate_from_config(config.stage_2).to(device).eval()
    from utils.inference import load_dit_checkpoint
    sd, metadata = load_dit_checkpoint(args.ckpt, use_ema=not args.raw)
    model.load_state_dict(sd, strict=True)
    epoch = metadata.get("epoch", "?")
    del sd
    print(f"Loaded {'raw' if args.raw else 'EMA'} DiT from {args.ckpt} (epoch {epoch})")

    latent_size = tuple(config.misc.latent_size)
    time_dist_shift = math.sqrt(
        (config.misc.time_dist_shift_dim or math.prod(latent_size)) / config.misc.time_dist_shift_base)
    transport = create_transport(config=config.transport, time_dist_shift=time_dist_shift)
    sampler = create_sampler(transport)
    ode = sampler.sample_ode(**{**dataclasses.asdict(config.sampler), "num_steps": args.steps})

    classes = args.classes or [i * 999 // 15 for i in range(16)]
    y = torch.tensor([c for c in classes for _ in range(args.per_class)], device=device)
    n = y.shape[0]
    g = torch.Generator(device=device).manual_seed(args.seed)
    zs = torch.randn(n, *latent_size, device=device, generator=g)

    with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        latents = ode(zs, model.forward, context=y)[-1]
        imgs = rae.decode(latents.float()).clamp(0, 1)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    save_image(imgs.cpu(), args.out, nrow=round(n ** 0.5))
    print(f"{n} samples (classes {classes}, {args.steps} ODE steps) -> {args.out}")


if __name__ == "__main__":
    main()
