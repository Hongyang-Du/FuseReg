import os
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.distributed as dist


@contextmanager
def _rank0_first():
    """Gate torch.hub download so only rank 0 fetches, others wait then read from cache."""
    initialized = dist.is_initialized()
    rank = dist.get_rank() if initialized else 0
    if initialized and rank != 0:
        dist.barrier()
    try:
        yield
    finally:
        if initialized and rank == 0:
            dist.barrier()


DINOV3_HUB_REF = "facebookresearch/dinov3:94a96ac83c2446f15f9bdcfae23cad3c6a9d4988"
DEFAULT_CKPT_DIR = Path(__file__).resolve().parents[3] / "pretrained_models" / "encoders" / "dinov3"

def load_dinov3():
    model_name = "dinov3_vitl16"
    ckpt_dir = os.environ.get("DINOV3_CKPT_DIR", str(DEFAULT_CKPT_DIR))
    weights = os.path.join(ckpt_dir, f"{model_name}_pretrain_lvd1689m-8aa4cbdd.pth")
    repo_dir = os.environ.get("DINOV3_REPO_DIR")
    with _rank0_first():
        if repo_dir and os.path.isfile(os.path.join(repo_dir, "hubconf.py")):
            return torch.hub.load(
                repo_dir,
                model_name,
                source="local",
                trust_repo=True,
                skip_validation=True,
                weights=weights,
            )
        return torch.hub.load(
            DINOV3_HUB_REF,
            model_name,
            source="github",
            trust_repo=True,
            skip_validation=True,
            weights=weights,
        )
