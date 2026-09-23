"""Train the paper class-conditional DiT on subset inputs and full-mean targets."""
import logging
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
from torchvision.utils import save_image

from utils import wandb_utils
from utils.checkpoint import save_stage2_checkpoint
from utils.guidance_utils import get_model_forward_fn
from utils.train_utils import update_ema

logger = logging.getLogger("rae")


def train_one_epoch(*, ddp_model, ema_model, rae, transport, eval_sampler,
                    dataloader, optimizer, scheduler, autocast_kwargs, device,
                    epoch, global_step, config, args, rank, checkpoint_dir,
                    experiment_dir, progress_bar, viz_fixed):
    model = ddp_model.module
    model_fn, sample_kwargs = get_model_forward_fn(ema_model, config.guidance)
    training = config.training
    if training.checkpoint_interval > 0 and epoch % training.checkpoint_interval == 0 and rank == 0:
        save_stage2_checkpoint(f"{checkpoint_dir}/ep-{epoch:07d}.pt", global_step,
                               epoch, ddp_model, ema_model, optimizer, scheduler)

    dataloader.set_epoch(epoch)
    optimizer.zero_grad(set_to_none=True)
    total_loss = torch.zeros((), device=device)
    num_batches = 0
    # Only complete accumulation groups contribute optimizer updates.
    num_micro_batches = len(dataloader) // training.grad_accum_steps * training.grad_accum_steps
    for step, (images, labels) in enumerate(dataloader):
        if step >= num_micro_batches:
            break
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            z_subset, z_full = rae.encode_cond_target(images)
        if viz_fixed["context"] is None:
            viz_fixed["context"] = labels[:len(viz_fixed["zs"])].clone()

        model_kwargs = {"context": labels}
        null_kwargs = {"context": torch.full_like(labels, config.misc.num_classes)}
        boundary = (step + 1) % training.grad_accum_steps == 0
        # DDP requires both forward and backward inside no_sync during accumulation.
        with nullcontext() if boundary else ddp_model.no_sync():
            with torch.autocast(device.type, **autocast_kwargs):
                losses = transport.training_losses(
                    ddp_model, z_subset, model_kwargs, null_kwargs, x1_target=z_full,
                    base_model_coeff=config.internal_guidance.base_model_coeff,
                    cfg_dropout_prob=config.conditioning.cfg_dropout_prob,
                )
                loss = losses["loss"].mean()
            (loss / training.grad_accum_steps).backward()
        total_loss += loss.detach()
        num_batches += 1
        if not boundary:
            continue

        if training.clip_grad:
            torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), training.clip_grad)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if scheduler is not None:
            scheduler.step()
        update_ema(ema_model, model, decay=training.ema_decay)
        global_step += 1
        progress_bar.update(1)

        if training.log_interval > 0 and global_step % training.log_interval == 0 and rank == 0:
            stats = {"train/loss": loss.item(), "train/loss_base": losses["loss_base"].mean().item(),
                     "train/lr": optimizer.param_groups[0]["lr"]}
            logger.info(f"[Epoch {epoch} | Step {global_step}] {stats}")
            if args.wandb:
                wandb_utils.log(stats, step=global_step)
            progress_bar.set_postfix(loss=loss.item(), lr=stats["train/lr"])

        if training.sample_every > 0 and global_step % training.sample_every == 0:
            model.eval()
            if rank == 0:
                sample_dir = Path(experiment_dir) / "samples"
                sample_dir.mkdir(parents=True, exist_ok=True)
                n = len(viz_fixed["zs"])
                batches = {
                    "batch": (torch.randn(n, *config.misc.latent_size, device=device), labels[:n]),
                    "fixed": (viz_fixed["zs"].clone(), viz_fixed["context"]),
                }
                for name, (noise, context) in batches.items():
                    with torch.no_grad(), torch.autocast(device.type, **autocast_kwargs):
                        latents = eval_sampler(noise, model_fn, context=context, **sample_kwargs)[-1]
                        images = rae.decode(latents).cpu().float().clamp(0, 1)
                    save_image(images, sample_dir / f"samples_{name}_s{global_step:07d}.png",
                               nrow=round(n ** 0.5))
                    if args.wandb:
                        import wandb
                        wandb.log({f"samples/{name}": wandb.Image(wandb_utils.array2grid(images))}, step=global_step)
            dist.barrier()
            model.train()

    if rank == 0 and num_batches:
        stats = {"epoch/loss": total_loss.item() / num_batches}
        logger.info(f"[Epoch {epoch}] {stats}")
        if args.wandb:
            wandb_utils.log(stats, step=global_step)
    return global_step
