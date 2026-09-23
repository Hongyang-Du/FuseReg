"""Stage 2 training script for flow matching on RAE latents."""

import argparse
import dataclasses
import math
import os

import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from copy import deepcopy

import torch.distributed as dist
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import transforms
from tqdm.auto import tqdm

from configs.stage2 import Stage2Config
from data import prepare_imagenet_dataloader
from stage1 import RAECombine
from stage2.engine import train_one_epoch
from stage2.models import Stage2ModelProtocol
from stage2.transport import create_sampler, create_transport
from stage2.utils import validate_stage2_config
from utils.checkpoint import load_stage2_checkpoint, save_stage2_checkpoint
from utils.dist_utils import cleanup_distributed, setup_distributed
from utils.model_utils import instantiate_from_config
from utils.optim_utils import build_optimizer, build_scheduler
from utils.resume_utils import configure_experiment_dirs, find_resume_checkpoint, save_worktree
from utils.train_utils import center_crop_arr, get_autocast_kwargs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Stage-2 transport model on RAE latents.")
    parser.add_argument("--config", type=str, required=True, help="YAML config file.")
    parser.add_argument("--results-dir", type=str, default="ckpts")
    parser.add_argument("--precision", type=str, choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--compile", action="store_true", help="torch.compile the training loss function")
    parser.add_argument("opts", nargs="*", default=[],
                        help="optional dotlist config overrides, e.g. training.epochs=40")
    return parser.parse_args()


def main():
    """Train Stage 2 model using config-driven hyperparameters."""
    args = parse_args()

    #########################################################
    # Distributed + config setup
    #########################################################
    rank, world_size, device = setup_distributed()
    _merged = OmegaConf.merge(OmegaConf.structured(Stage2Config), OmegaConf.load(args.config))
    if args.opts:
        _merged = OmegaConf.merge(_merged, OmegaConf.from_dotlist(list(args.opts)))
    config: Stage2Config = OmegaConf.to_object(_merged)
    # GRAD_ACCUM_OVERRIDE lets a launch script set grad_accum_steps without editing the
    # shared config (e.g. accum=1 at 32 GPUs vs accum=2 at 8 GPUs); global_batch is unchanged.
    if os.environ.get("GRAD_ACCUM_OVERRIDE"):
        config.training.grad_accum_steps = int(os.environ["GRAD_ACCUM_OVERRIDE"])
    validate_stage2_config(config)

    seed = config.training.global_seed * world_size + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    experiment_dir, checkpoint_dir, logger = configure_experiment_dirs(args, rank)
    autocast_kwargs = get_autocast_kwargs(args)

    #########################################################
    # Data setup; train and eval
    #########################################################
    global_batch_size = config.training.global_batch_size or (config.training.batch_size * world_size * config.training.grad_accum_steps)
    if global_batch_size % (world_size * config.training.grad_accum_steps):
        raise ValueError("global_batch_size must be divisible by world_size * grad_accum_steps")
    micro_batch_size = global_batch_size // (world_size * config.training.grad_accum_steps)
    stage2_transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, config.training.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    needs_transform = config.dataset.type != "hf"
    # train dataloader
    dataloader = prepare_imagenet_dataloader(
        config=dataclasses.asdict(config.dataset),
        image_size=config.training.image_size,
        batch_size=micro_batch_size,
        num_workers=config.training.num_workers,
        rank=rank,
        world_size=world_size,
        transform=stage2_transform if needs_transform else None,
    )

    #########################################################
    # Models setup
    #########################################################
    latent_size = tuple(config.misc.latent_size)

    # stage1: rae - frozen
    rae: RAECombine = instantiate_from_config(config.stage_1).to(device)
    rae.eval()

    config.prepare_model_params()

    # stage2: model - trainable
    model: Stage2ModelProtocol = instantiate_from_config(config.stage_2).to(device)
    model.requires_grad_(True)
    # stage2 ema model
    ema_model = deepcopy(model).to(device)
    ema_model.requires_grad_(False)
    ema_model.eval()

    # ddp wrapper for stage2 model
    ddp_model = DDP(model, device_ids=[device.index], broadcast_buffers=False, find_unused_parameters=False)
    model = ddp_model.module
    ddp_model.train()
    logger.info(f"Model Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    if args.wandb and rank == 0:
        import wandb
        wandb.config.update({
            "model_params_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 1),
            "trainable_params_M": round(sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6, 1),
        }, allow_val_change=True)

    #########################################################
    # Optimizer + Scheduler setup
    #########################################################
    optimizer, _ = build_optimizer(
        [p for p in model.parameters() if p.requires_grad],
        config.training.optimizer,
    )

    #########################################################
    # Steps per epoch setup
    #########################################################
    steps_per_epoch = len(dataloader) // config.training.grad_accum_steps
    logger.info(f"Using {steps_per_epoch} optimizer steps per epoch")
    if steps_per_epoch < 1:
        raise ValueError("Dataset is too small for one full gradient accumulation group")

    # Build scheduler (needs steps_per_epoch)
    scheduler = None
    sched_msg = None
    if config.training.scheduler is not None:
        scheduler, sched_msg = build_scheduler(optimizer, steps_per_epoch, config.training.scheduler)

    #########################################################
    # Transport + Sampler setup
    #########################################################
    time_dist_shift = math.sqrt(
        (config.misc.time_dist_shift_dim or math.prod(latent_size)) / config.misc.time_dist_shift_base
    )
    transport = create_transport(
        config=config.transport,
        time_dist_shift=time_dist_shift,
    )
    transport_sampler = create_sampler(transport)
    eval_sampler = transport_sampler.sample_ode(**dataclasses.asdict(config.sampler))

    if args.compile:
        transport.training_losses = torch.compile(transport.training_losses)

    #########################################################
    # Resume setup
    #########################################################
    start_epoch = 0
    global_step = 0
    ckpt_path = find_resume_checkpoint(experiment_dir, args.ckpt)
    if ckpt_path:
        start_epoch, global_step = load_stage2_checkpoint(ckpt_path, ddp_model, ema_model, optimizer, scheduler)
        logger.info(f"[Rank {rank}] Resumed from {ckpt_path} (epoch={start_epoch}, step={global_step}).")
    else:
        if rank == 0:
            save_worktree(experiment_dir, config)
            logger.info(f"Saved training worktree and config to {experiment_dir}.")

    total_steps = config.training.epochs * steps_per_epoch
    progress_bar = tqdm(total=total_steps, initial=global_step, desc="Training", disable=rank != 0)

    # fixed state for consistent visualization across epochs (populated from first batch)
    num_viz_samples = min(micro_batch_size, 32)
    viz_fixed = {
        'zs': torch.randn(num_viz_samples, *latent_size, device=device, dtype=torch.float32,
                           generator=torch.Generator(device=device).manual_seed(seed)),
        'context': None,
    }

    #########################################################
    # Training loop
    #########################################################
    dist.barrier()
    for epoch in range(start_epoch, config.training.epochs):
        model.train()

        global_step = train_one_epoch(
            ddp_model=ddp_model,
            ema_model=ema_model,
            rae=rae,
            transport=transport,
            eval_sampler=eval_sampler,
            dataloader=dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            autocast_kwargs=autocast_kwargs,
            device=device,
            epoch=epoch,
            global_step=global_step,
            config=config,
            args=args,
            rank=rank,
            checkpoint_dir=checkpoint_dir,
            experiment_dir=experiment_dir,
            progress_bar=progress_bar,
            viz_fixed=viz_fixed,
        )
    progress_bar.close()

    #########################################################
    # final checkpoint setup and cleanup
    #########################################################
    if rank == 0:
        logger.info(f"Saving final checkpoint at epoch {config.training.epochs}...")
        ckpt_path = f"{checkpoint_dir}/ep-{config.training.epochs:07d}.pt"
        save_stage2_checkpoint(ckpt_path, global_step, config.training.epochs, ddp_model, ema_model, optimizer, scheduler)

    dist.barrier()
    logger.info("Done!")
    cleanup_distributed()


if __name__ == "__main__":
    main()
