#!/usr/bin/env python3
"""Load a real decoder and run a synthetic latent. This is not a paper evaluation."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
from stage1.rae import _load_decoder
from utils.inference import load_decoder_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--config", default="configs/decoder/ViTXL")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(0)
    started = time.perf_counter()
    state, _, metadata = load_decoder_checkpoint(args.ckpt)
    latent_dim = state["decoder_embed.weight"].shape[1]
    model = _load_decoder(args.config, hidden_size=latent_dim, patch_size=16, num_patches=256).eval()
    model.load_state_dict(state, strict=True)
    del state
    with torch.inference_mode():
        output = model(torch.randn(1, 256, latent_dim), drop_cls_token=False).logits
        image = model.unpatchify(output)
    if image.shape != (1, 3, 256, 256) or not torch.isfinite(image).all():
        raise ValueError("Unexpected output shape or non-finite values")
    result = {
        "test": "real_weights_synthetic_latent_cpu_forward",
        "paper_evaluation": False,
        "checkpoint": Path(args.ckpt).name,
        "checkpoint_metadata": metadata,
        "device": "cpu", "dtype": "float32", "torch": torch.__version__,
        "output_shape": list(image.shape), "all_finite": True,
        "elapsed_seconds": time.perf_counter() - started,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
