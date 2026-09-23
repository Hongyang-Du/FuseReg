import argparse
import hashlib
import logging
import math
import os
import sys

import torch
import torch.distributed as dist
import wandb
from torchvision.utils import make_grid


def create_logger(logging_dir: str, logger_name: str) -> logging.Logger:
    """
    Create a logger that writes to a log file and stdout.
    Only rank 0 writes; other ranks get a dummy logger.
    """
    rank = dist.get_rank() if dist.is_initialized() else 0
    logger = logging.getLogger(logger_name)  # use provided logger name

    if rank == 0:
        # Make sure log dir exists
        os.makedirs(logging_dir, exist_ok=True)

        # Clear any existing handlers so we can reconfigure
        for h in list(logger.handlers):
            logger.removeHandler(h)

        logger.setLevel(logging.INFO)
        logger.propagate = False  # don't double-log via root

        fmt = logging.Formatter(
            '[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
        )

        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(fmt)
        logger.addHandler(stream_handler)

        file_handler = logging.FileHandler(os.path.join(logging_dir, "log.txt"))
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    else:
        # Dummy logger: no handlers, no output
        logger.setLevel(logging.CRITICAL + 1)
        logger.propagate = False
        for h in list(logger.handlers):
            logger.removeHandler(h)

    return logger

def is_main_process():
    return dist.get_rank() == 0

def namespace_to_dict(namespace):
    return {
        k: namespace_to_dict(v) if isinstance(v, argparse.Namespace) else v
        for k, v in vars(namespace).items()
    }


def generate_run_id(exp_name):
    # https://stackoverflow.com/questions/16008670/how-to-hash-a-string-into-8-digits
    return str(int(hashlib.sha256(exp_name.encode('utf-8')).hexdigest(), 16) % 10 ** 8)


def initialize(args, entity, exp_name, project_name):
    config_dict = namespace_to_dict(args)
    if is_main_process():
        if "WANDB_KEY" in os.environ:
            wandb.login(key=os.environ["WANDB_KEY"])
        init_kwargs = dict(
            entity=entity,
            project=project_name,
            name=exp_name,
            config=config_dict,
            reinit=True,
        )
        # WANDB_FRESH_RUN=1: start a brand-new run each launch (wandb auto-generates a
        # unique id). Avoids resuming a deterministic-id run whose logged step got ahead
        # of the last checkpoint after a restart — that makes wandb ignore the re-logged
        # (lower) steps ("ignoring partial history record") and the run shows as crashed.
        if not os.environ.get("WANDB_FRESH_RUN"):
            init_kwargs.update(id=generate_run_id(exp_name), resume="allow")
        wandb.init(**init_kwargs)


def log(stats, step=None):
    if is_main_process():
        # print(f"WandB logging at step {step}: {stats}")
        wandb.log({k: v for k, v in stats.items()}, step=step)


def array2grid(x):
    nrow = round(math.sqrt(x.size(0)))
    x = make_grid(x, nrow=nrow, normalize=True, value_range=(0,1))
    x = x.clamp(0, 1).mul(255).permute(1,2,0).to('cpu', torch.uint8).numpy()
    return x
