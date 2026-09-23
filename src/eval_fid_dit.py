#!/usr/bin/env python3
"""Sample an EMA DiT and compute torch-fidelity FID/IS against explicit real images."""

import argparse
import dataclasses
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from omegaconf import OmegaConf
from torchvision import transforms
from torchvision.utils import save_image

from configs.stage2 import Stage2Config
from eval.fid import calculate_fid_isc
from stage2.transport import create_sampler, create_transport
from utils.model_utils import instantiate_from_config


def _to_uint8_nhwc(imgs01: torch.Tensor) -> np.ndarray:
    """[B,3,H,W] float in [0,1] -> [B,H,W,3] uint8."""
    return (imgs01.clamp(0, 1) * 255).round().byte().permute(0, 2, 3, 1).cpu().numpy()


def _build_guidance(args):
    from configs.stage2 import GuidanceConfig, IGConfig
    return GuidanceConfig(ig=IGConfig(scale=args.ig_scale, t_min=args.ig_tmin, t_max=args.ig_tmax))


def generate(args, config, device):
    from utils.guidance_utils import get_model_forward_fn
    config.stage_1.params["load_encoder"] = False
    rae = instantiate_from_config(config.stage_1).to(device).eval()
    model = instantiate_from_config(config.stage_2).to(device).eval()
    from utils.inference import load_dit_checkpoint
    sd, metadata = load_dit_checkpoint(args.ckpt, use_ema=not args.raw)
    model.load_state_dict(sd, strict=True)
    epoch = metadata.get("epoch", "?")
    args.checkpoint_metadata = metadata
    del sd
    print(f"Loaded {'raw' if args.raw else 'EMA'} DiT from {args.ckpt} (epoch {epoch})", flush=True)

    guid = _build_guidance(args)
    model_fn, sample_kwargs = get_model_forward_fn(model, guid)
    print(f"Internal guidance scale={guid.ig.scale}, interval=[{guid.ig.t_min}, {guid.ig.t_max}]", flush=True)

    latent_size = tuple(config.misc.latent_size)
    time_dist_shift = math.sqrt(
        (config.misc.time_dist_shift_dim or math.prod(latent_size)) / config.misc.time_dist_shift_base)
    transport = create_transport(config=config.transport, time_dist_shift=time_dist_shift)
    sampler = create_sampler(transport)
    ode = sampler.sample_ode(**{**dataclasses.asdict(config.sampler), "num_steps": args.steps})

    n = args.num_samples
    labels = torch.arange(n) % config.misc.num_classes        # even class coverage
    g = torch.Generator(device=device).manual_seed(args.seed)
    arr = np.empty((n, config.training.image_size, config.training.image_size, 3), dtype=np.uint8)

    with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        for i in range(0, n, args.batch):
            y = labels[i:i + args.batch].to(device)
            bs = len(y)
            zs = torch.randn(bs, *latent_size, device=device, generator=g)
            latents = ode(zs, model_fn, context=y, **sample_kwargs)[-1]
            imgs = rae.decode(latents.float()).clamp(0, 1)
            arr[i:i + bs] = _to_uint8_nhwc(imgs)
            if (i // args.batch) % 10 == 0:
                print(f"  generated {i + bs}/{n}", flush=True)

    if args.grid:
        os.makedirs(os.path.dirname(os.path.abspath(args.grid)), exist_ok=True)
        grid = torch.from_numpy(arr[:64]).permute(0, 3, 1, 2).float() / 255
        save_image(grid, args.grid, nrow=8)
        print(f"Sample grid -> {args.grid}", flush=True)

    del model, rae
    torch.cuda.empty_cache()
    return arr, epoch


def load_reference(args, image_size):
    """N real training images with the stage-2 training preprocessing."""
    from data.partial_imagenet import PartialImageNetDataset
    tf = transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ])
    ds = PartialImageNetDataset(args.data, split="train", transform=tf)
    g = torch.Generator().manual_seed(args.seed)
    idxs = torch.randperm(len(ds), generator=g)[:args.num_samples].tolist()
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(ds, idxs),
        batch_size=128, num_workers=args.num_workers, drop_last=False)

    arr = np.empty((len(idxs), image_size, image_size, 3), dtype=np.uint8)
    i = 0
    for imgs, _ in loader:
        arr[i:i + imgs.shape[0]] = _to_uint8_nhwc(imgs)
        i += imgs.shape[0]
        if (i // 128) % 20 == 0:
            print(f"  reference {i}/{len(idxs)}", flush=True)
    return arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",      required=True, help="Stage-2 training YAML")
    ap.add_argument("--ckpt",        required=True, help="Stage-2 checkpoint (ep-*.pt)")
    ap.add_argument("--data",        default="data/imagenet-256")
    ap.add_argument("--ref-npz",     default=None, help="If set, use this npz (key arr_0) as the reference (e.g. imagenet val 50k) instead of sampling train images")
    ap.add_argument("--num-samples", type=int, default=10000)
    ap.add_argument("--batch",       type=int, default=64)
    ap.add_argument("--steps",       type=int, default=50)
    ap.add_argument("--seed",        type=int, default=42)
    ap.add_argument("--raw",         action="store_true", help="Use raw weights instead of EMA")
    ap.add_argument("--grid",        default=None, help="Optional PNG path for a 64-sample grid")
    ap.add_argument("--out",         default=None, help="Optional JSON path for the result")
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--device",      default="cuda:0")
    # Guidance (defaults => plain conditional, identical to the no-guidance baseline)
    ap.add_argument("--ig-scale",   type=float, default=1.0, help="Internal-guidance scale (1.0 = off)")
    ap.add_argument("--ig-tmin",    type=float, default=0.0)
    ap.add_argument("--ig-tmax",    type=float, default=1.0)
    ap.add_argument("opts", nargs="*", help="Config overrides")
    args = ap.parse_args()
    if args.num_samples < 2 or args.batch < 1 or args.steps < 1:
        ap.error("Use num-samples >= 2, batch >= 1 and steps >= 1")

    device = torch.device(args.device)
    config: Stage2Config = OmegaConf.to_object(
        OmegaConf.merge(OmegaConf.structured(Stage2Config), OmegaConf.load(args.config), OmegaConf.from_dotlist(args.opts)))
    config.prepare_model_params()
    gen_arr, epoch = generate(args, config, device)
    if args.ref_npz:
        _z = np.load(args.ref_npz)
        ref_arr = _z["arr_0"] if "arr_0" in _z else _z[list(_z.keys())[0]]
        print(f"Loaded reference {ref_arr.shape} from {args.ref_npz}", flush=True)
    else:
        ref_arr = load_reference(args, config.training.image_size)

    expected_shape = (config.training.image_size, config.training.image_size, 3)
    if ref_arr.dtype != np.uint8 or ref_arr.ndim != 4 or tuple(ref_arr.shape[1:]) != expected_shape or len(ref_arr) < 2:
        raise ValueError(f"Reference must be uint8 [N,{expected_shape}] with N >= 2; got {ref_arr.shape} {ref_arr.dtype}")
    print("Computing FID + IS (torch-fidelity)...", flush=True)
    fid, isc = calculate_fid_isc(gen_arr, ref_arr, bs=64, device=device)

    _guid = _build_guidance(args)
    result = {
        "fid": fid,
        "is": isc,
        "ckpt": args.ckpt,
        "epoch": epoch,
        "num_samples": len(gen_arr),
        "num_reference": len(ref_arr),
        "metric_backend": "torch-fidelity-0.3.0, inception-v3-compat",
        "reference": args.ref_npz or args.data,
        "reference_split": "explicit_npz" if args.ref_npz else "train",
        "steps": args.steps,
        "seed": args.seed,
        "ig_scale": args.ig_scale, "ig_tmin": args.ig_tmin, "ig_tmax": args.ig_tmax,
        "guidance": ("none (plain conditional Euler)" if not _guid.any_guidance_active
                     else f"IG={args.ig_scale}@[{args.ig_tmin},{args.ig_tmax}]"),
        "weights": "raw" if args.raw else "ema",
    }
    from utils.inference import file_sha256, config_provenance
    result.update(config_provenance(dataclasses.asdict(config)))
    result['checkpoint_sha256'] = file_sha256(args.ckpt)
    result['checkpoint_metadata'] = args.checkpoint_metadata
    if args.ref_npz:
        result['reference_sha256'] = file_sha256(args.ref_npz)
    else:
        import hashlib
        digest = hashlib.sha256()
        for offset in range(0, len(ref_arr), 64):
            digest.update(np.ascontiguousarray(ref_arr[offset:offset + 64]).tobytes())
        result['sampled_reference_images_sha256'] = digest.hexdigest()
    print(f"FID = {fid:.3f}  IS = {isc:.3f}  ({len(gen_arr)} gen vs {len(ref_arr)} real, "
          f"ckpt={args.ckpt}, epoch={epoch})", flush=True)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Result -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
