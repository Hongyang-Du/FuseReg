"""Precompute encoder-combine latent normalization stats for the paper RAECombine.

  - mean / var : per-channel stats of the (raw, un-normalized) combine latent z [B,C,H,W],
                 used by RAECombine._tokens_to_latent to standardize the DiT target.
                 shape [1, C, 1, 1]. Format matches _load_normalization_stats().

Computed on the DETERMINISTIC full-mean latent (drop=False) with normalization disabled,
so the result is the raw combine-output statistics.

  torchrun --nproc_per_node=1 src/compute_latent_stats.py \
      --config configs/stage2/training/dinov3-k23-fulltarget-p00-ditxl.yaml \
      --num-samples 50000 --batch 256 --out checkpoints/latent_stats.pt
"""
import argparse
import dataclasses
from pathlib import Path

import torch
from omegaconf import OmegaConf

from configs.stage2 import Stage2Config
from data import prepare_imagenet_dataloader
from utils.dist_utils import setup_distributed
from utils.model_utils import instantiate_from_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--num-samples", type=int, default=50000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--out", required=True)
    ap.add_argument("opts", nargs="*", help="Config overrides")
    args = ap.parse_args()

    if args.num_samples < 1 or args.batch < 1:
        ap.error("--num-samples and --batch must be positive")
    rank, world_size, device = setup_distributed()
    if world_size != 1:
        raise ValueError("Statistics are computed on one rank; use torchrun --nproc_per_node=1 to avoid duplicated or incomplete moments")
    config: Stage2Config = OmegaConf.to_object(
        OmegaConf.merge(OmegaConf.structured(Stage2Config), OmegaConf.load(args.config), OmegaConf.from_dotlist(args.opts)))

    # Force raw, deterministic latent: no pre-existing normalization, no random layer-drop.
    config.stage_1.params["normalization_stat_path"] = None
    config.stage_1.params["drop"] = False

    torch.manual_seed(config.training.global_seed)
    rae = instantiate_from_config(config.stage_1).to(device).eval()

    loader = prepare_imagenet_dataloader(
        config=dataclasses.asdict(config.dataset),
        image_size=config.training.image_size,
        batch_size=args.batch,
        num_workers=config.training.num_workers,
        rank=rank, world_size=world_size,
        transform=None,
    )

    c = None
    s = ss = None
    n_spatial = 0
    seen = 0
    for images, _ in loader:
        images = images[:args.num_samples - seen].to(device)
        z = rae.encode(images)                    # [B, C, H, W], raw (do_normalization=False)
        b, c_, h, w = z.shape
        if s is None:
            c = c_
            s = torch.zeros(c, dtype=torch.float64, device=device)
            ss = torch.zeros(c, dtype=torch.float64, device=device)
        zf = z.double().permute(1, 0, 2, 3).reshape(c, -1)   # [C, B*H*W]
        s += zf.sum(dim=1); ss += (zf * zf).sum(dim=1); n_spatial += zf.shape[1]
        seen += b
        if rank == 0 and seen % (args.batch * 20) == 0:
            print(f"  {seen}/{args.num_samples}", flush=True)
        if seen >= args.num_samples:
            break

    if seen != args.num_samples:
        raise ValueError(f"Requested {args.num_samples} images, but loader supplied {seen}")

    mean = s / n_spatial
    var = (ss / n_spatial) - mean * mean

    if rank == 0:
        out = {
            "mean": mean.float().view(1, c, 1, 1).cpu(),
            "var": var.clamp_min(0).float().view(1, c, 1, 1).cpu(),
            "num_samples": seen,
            "drop": False,
            "seed": config.training.global_seed,
        }
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        torch.save(out, args.out)
        print(f"saved -> {args.out}  | mean {mean.mean():.4f} var {var.mean():.4f} "
              f"(C={c}, N_spatial={n_spatial})")


if __name__ == "__main__":
    main()
