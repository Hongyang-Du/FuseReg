#!/usr/bin/env python3
"""Train the paper DINOv3 fusion decoder with L1, LPIPS, GAN, and EMA.

    torchrun --nproc_per_node=4 src/train_decoder.py \
        --config configs/stage1/decoder/dinov3-k23-p095.yaml
"""
import argparse
import math
import os
import time
from copy import deepcopy
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from omegaconf import OmegaConf
from torchvision import transforms

from configs.stage1_decoder import DecoderConfig
from encoders.vision_encoder import create_encoder
from stage1.combine import MLSCombine
from stage1.rae import _load_decoder
from stage1.disc import DinoDiscriminator, hinge_d_loss, vanilla_g_loss, calculate_adaptive_weight
from stage1.disc.diffaug import DiffAug
from stage1.disc.lpips import LPIPS
from stage1.disc.utils import RandomWindowCrop
from stage1.metrics import psnr


@torch.no_grad()
def update_ema(ema, model, decay=0.9995):
    for ep, p in zip(ema.parameters(), model.parameters()):
        ep.data.mul_(decay).add_(p.data, alpha=1 - decay)
    for eb, b in zip(ema.buffers(), model.buffers()):
        eb.copy_(b)


def make_loader(data_dir, image_size, batch_size, num_workers, world_size, rank):
    t = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    arrow_dir = os.path.join(data_dir, "imagenet-latents-images")
    if os.path.isdir(arrow_dir):
        from data.partial_imagenet import PartialImageNetDataset
        ds = PartialImageNetDataset(data_dir, split="train", transform=t)
    else:
        from torchvision.datasets import ImageFolder
        ds = ImageFolder(os.path.join(data_dir, "train"), transform=t)
    sampler = torch.utils.data.distributed.DistributedSampler(
        ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=True, drop_last=True), sampler


@torch.no_grad()
def validate(encoder, combine, decoder, images, mean, std, batch=32):
    from torchmetrics.functional.image import structural_similarity_index_measure
    psnrs, ssims = [], []
    for offset in range(0, len(images), batch):
        x = images[offset:offset + batch].to(mean.device).float() / 255
        tokens = encoder((x - mean) / std)
        prediction = decoder(combine(tokens), drop_cls_token=False).logits
        rec = (decoder.unpatchify(prediction) * std + mean).clamp(0, 1).float()
        mse = (rec - x).square().flatten(1).mean(1)
        psnrs.append(-10 * torch.log10(mse + 1e-10))
        ssims.append(structural_similarity_index_measure(rec, x, data_range=1.).item() * len(x))
    return torch.cat(psnrs).mean().item(), sum(ssims) / len(images)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("opts", nargs="*", help="OmegaConf dotlist overrides")
    args = parser.parse_args()
    cfg = OmegaConf.to_object(OmegaConf.merge(
        OmegaConf.structured(DecoderConfig), OmegaConf.load(args.config), OmegaConf.from_dotlist(args.opts)))
    C, L, D, T = cfg.combine, cfg.loss, cfg.data, cfg.training
    if C.target != "stage1.combine.MLSCombine":
        raise ValueError("The paper uses stage1.combine.MLSCombine")
    if cfg.encoder_name != "dinov3mls-vit-l16":
        raise ValueError("The paper uses the DINOv3 ViT-L/16 encoder")
    if T.precision not in ("fp32", "bf16") or T.grad_accum_steps < 1:
        raise ValueError("precision must be fp32 or bf16 and grad_accum_steps must be positive")
    if cfg.probe.loo_solo not in ("off", "final", "every"):
        raise ValueError("probe.loo_solo must be off, final, or every")

    dist.init_process_group("nccl", timeout=timedelta(minutes=30))
    rank, world = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    torch.manual_seed(T.seed + rank)
    torch.backends.cudnn.benchmark = True
    is_main = rank == 0
    out_dir = Path(T.out_dir)
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"World {world}; batch/GPU {T.batch_size}; accumulation {T.grad_accum_steps}")
        OmegaConf.save(OmegaConf.structured(cfg), out_dir / "config.yaml")
    wb = None
    if cfg.wandb.enabled and is_main:
        import wandb as wb
        wb.init(project=cfg.wandb.project, entity=cfg.wandb.entity, name=cfg.wandb.name,
                config=OmegaConf.to_container(OmegaConf.structured(cfg)))

    train_loader, train_sampler = make_loader(
        D.data_dir, D.image_size, T.batch_size, D.num_workers, world, rank)
    layers = list(C.params["layers"])
    encoder = create_encoder(cfg.encoder_name + "[layers=" + ".".join(map(str, layers)) + "]",
                             device, D.image_size).eval()
    mean = torch.tensor((0.485, 0.456, 0.406), device=device).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), device=device).view(1, 3, 1, 1)
    combine = MLSCombine(**C.params).to(device).train()
    decoder = _load_decoder(cfg.decoder.config_path, cfg.decoder.latent_dim,
                            cfg.decoder.patch_size, cfg.decoder.num_patches).to(device)
    decoder_ddp = DDP(decoder, device_ids=[local_rank])
    ema_dec = deepcopy(decoder).requires_grad_(False).eval()
    eval_combine = deepcopy(combine).eval()
    trainable = list(decoder.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=T.lr, betas=(0.9, 0.95), weight_decay=0.)
    total_steps = T.epochs * len(train_loader)
    warmup_steps = T.warmup_epochs * len(train_loader)

    def lr_fn(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn)
    perceptual = LPIPS().to(device).requires_grad_(False).eval()
    disc = DinoDiscriminator(device=device, dino_ckpt_path=L.gan.disc_ckpt).to(device)
    disc.dino_proxy[0].crop = RandomWindowCrop(D.image_size, 224, 9, False)
    disc.dino_proxy[0].original_input_size = D.image_size
    disc_ddp = DDP(disc, device_ids=[local_rank])
    disc_aug = DiffAug(prob=0.5, cutout=True)
    disc_optimizer = torch.optim.Adam(disc.parameters(), lr=1e-4, betas=(0.5, 0.9))
    disc_start_step = L.gan.disc_start * len(train_loader)

    # Load the validation subset before training so disk I/O cannot stall an
    # epoch-end collective. The cohort is fixed across all decoder rates.
    val_images = demo_images = demo_tokens = None
    if is_main and D.val_npz and Path(D.val_npz).exists():
        import numpy as np
        archive = np.load(D.val_npz, mmap_mode="r", allow_pickle=False)
        arr = archive[archive.files[0]] if isinstance(archive, np.lib.npyio.NpzFile) else archive
        indices = torch.randperm(len(arr), generator=torch.Generator().manual_seed(0))[:D.val_n].tolist()
        val_images = torch.stack([torch.from_numpy(arr[i].copy()) for i in indices]).permute(0, 3, 1, 2)
        del arr, archive
    if is_main and cfg.probe.val_image:
        from PIL import Image
        transform = transforms.Compose([transforms.Resize(D.image_size + 32),
                                        transforms.CenterCrop(D.image_size), transforms.ToTensor()])
        demo_path = Path(cfg.probe.val_image)
        paths = sorted(p for p in demo_path.parent.iterdir()
                       if p.suffix.lower() in (".png", ".jpg") and "concat" not in p.name.lower()) or [demo_path]
        demo_images = torch.stack([transform(Image.open(p).convert("RGB")) for p in paths]).to(device)
        with torch.no_grad():
            demo_tokens = encoder((demo_images - mean) / std)

    latest = out_dir / "ckpt_latest.pt"
    start_epoch = global_step = 0
    if latest.exists():
        checkpoint = torch.load(latest, map_location="cpu", weights_only=False)
        combine.load_state_dict(checkpoint["combine"], strict=True)
        decoder.load_state_dict(checkpoint["decoder"], strict=True)
        ema_dec.load_state_dict(checkpoint["ema_dec"], strict=True)
        disc.load_state_dict(checkpoint["disc"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        disc_optimizer.load_state_dict(checkpoint["disc_optimizer"])
        start_epoch, global_step = checkpoint["epoch"], checkpoint["global_step"]
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        else:
            scheduler.last_epoch = global_step - 1
            scheduler.step()
        del checkpoint
        if is_main:
            print(f"Resumed epoch={start_epoch}, step={global_step}")

    def decode_images(module, z):
        out = module(z, drop_cls_token=False).logits
        plain = module.module if isinstance(module, DDP) else module
        return (plain.unpatchify(out) * std + mean).clamp(0, 1)

    autocast = torch.autocast("cuda", dtype=torch.bfloat16, enabled=T.precision == "bf16")
    accum = T.grad_accum_steps
    optimizer.zero_grad(set_to_none=True)
    disc_optimizer.zero_grad(set_to_none=True)
    for epoch in range(start_epoch, T.epochs):
        train_sampler.set_epoch(epoch)
        decoder_ddp.train()
        start_time = time.time()
        for micro_idx, (images, _) in enumerate(train_loader):
            images = images.to(device)
            with torch.no_grad():
                tokens = encoder((images - mean) / std)
            use_gan = global_step >= disc_start_step and L.gan.disc_weight > 0
            step_optimizer = (micro_idx + 1) % accum == 0
            with autocast:
                z = combine(tokens)
                reconstruction = decode_images(decoder_ddp, z)
                loss_l1 = F.l1_loss(reconstruction, images)
                loss_lpips = perceptual(reconstruction * 2 - 1, images * 2 - 1).mean()
            loss_rec = loss_l1 + L.lpips_w * loss_lpips
            loss_gan = torch.zeros(1, device=device)
            adaptive_weight = torch.tensor(0., device=device)
            if use_gan:
                disc_ddp.eval()
                half = max(1, len(images) // 2)
                with autocast:
                    fake_aug = disc_aug.aug(reconstruction[:half] * 2 - 1)
                    logits_fake, _ = disc_ddp(fake_aug, None)
                loss_gan = vanilla_g_loss(logits_fake)
                last_layer = next(reversed([p for p in decoder.parameters() if p.requires_grad]))
                adaptive_weight = calculate_adaptive_weight(loss_rec, loss_gan, last_layer).clamp(0, 1e4).detach()
            loss = loss_rec + L.gan.disc_weight * adaptive_weight * loss_gan
            (loss / accum).backward()
            if step_optimizer:
                if T.clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(trainable, T.clip_grad)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            # Preserve the released recipe: global_step and LR scheduling count
            # micro-batches; optimizer and EMA update at accumulation boundaries.
            scheduler.step()
            if use_gan:
                disc_ddp.train()
                with autocast:
                    real_aug = disc_aug.aug(images[:half] * 2 - 1)
                    fake_aug = disc_aug.aug(reconstruction[:half].detach() * 2 - 1)
                    logits_real, _ = disc_ddp(real_aug, None)
                    logits_fake, _ = disc_ddp(fake_aug, None)
                    loss_d = hinge_d_loss(logits_real, logits_fake)
                (loss_d / accum).backward()
                if step_optimizer:
                    disc_optimizer.step()
                    disc_optimizer.zero_grad(set_to_none=True)
            if step_optimizer:
                update_ema(ema_dec, decoder, T.ema_decay)
            global_step += 1
            if is_main and global_step % T.log_every == 0:
                metrics = {"loss": loss.item(), "l1": loss_l1.item(), "lpips": loss_lpips.item(),
                           "gan": loss_gan.item(), "psnr": psnr(reconstruction, images),
                           "lr": optimizer.param_groups[0]["lr"]}
                print(f"epoch={epoch + 1} step={global_step} {metrics}", flush=True)
                if wb:
                    wb.log({f"train/{k}": v for k, v in metrics.items()}, step=global_step)

        if is_main:
            print(f"Epoch {epoch + 1}/{T.epochs}: {time.time() - start_time:.0f}s", flush=True)
            if val_images is not None:
                val_psnr, val_ssim = validate(encoder, eval_combine, ema_dec, val_images, mean, std)
                print(f"Validation EMA: PSNR={val_psnr:.4f}, SSIM={val_ssim:.4f}", flush=True)
                if wb:
                    wb.log({"val/psnr": val_psnr, "val/ssim": val_ssim}, step=global_step)
            do_probe = cfg.probe.loo_solo == "every" or (cfg.probe.loo_solo == "final" and epoch + 1 == T.epochs)
            if do_probe and demo_tokens is not None:
                with torch.no_grad():
                    full = psnr(decode_images(ema_dec, eval_combine(demo_tokens)), demo_images)
                    loo = [full - psnr(decode_images(ema_dec, eval_combine(
                        demo_tokens, idx=[j for j in range(len(layers)) if j != i])), demo_images)
                        for i in range(len(layers))]
                    solo = [psnr(decode_images(ema_dec, eval_combine(demo_tokens, idx=[i])), demo_images)
                            for i in range(len(layers))]
                print(f"LOO dPSNR={loo}; solo PSNR={solo}; layers={layers}", flush=True)
            checkpoint = {"epoch": epoch + 1, "global_step": global_step, "layers": layers,
                          "combine": combine.state_dict(), "decoder": decoder.state_dict(),
                          "ema_combine": eval_combine.state_dict(), "ema_dec": ema_dec.state_dict(),
                          "disc": disc.state_dict(), "optimizer": optimizer.state_dict(),
                          "disc_optimizer": disc_optimizer.state_dict(), "scheduler": scheduler.state_dict()}
            temporary = latest.with_suffix(".tmp")
            torch.save(checkpoint, temporary)
            os.replace(temporary, latest)
            if (epoch + 1) % T.ckpt_every == 0:
                torch.save(checkpoint, out_dir / f"ckpt_ep{epoch + 1:03d}.pt")
        dist.barrier()
    if wb:
        wb.finish()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
