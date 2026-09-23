<div align="center">

# FuseReg

**Regularizing Layer Fusion Mitigates the Reconstruction-Generation Gap<br>in Representation Autoencoders**

Anonymous supplementary code

<a href="#quick-start">Quick start</a> ·
<a href="#included-recipes">Recipes</a> ·
<a href="#reproduction">Reproduction</a> ·
<a href="#verification">Verification</a>

</div>

FuseReg trains representation-autoencoder decoders on random subsets of frozen DINOv3-L layers, then trains full-target diffusion models on the resulting representations. This archive includes training, sampling, evaluation, and the paper's experiment matrix.

> **Reproduction status:** All 175 unique table evaluations, including the 120 appendix evaluations, remain **not run**. The recorded numbers are paper-reported targets. Source checks, checkpoint checks and CPU forward passes do not establish numerical agreement with the paper.

## Quick start

Run commands from the extracted `FuseReg/` directory. Use Python 3.11 and a Linux CUDA machine for full experiments.

### 1. Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

DiT training additionally requires the pinned optimizer dependency: `python -m pip install '.[train]'`.

### 2. Prepare the assets

The archive contains source code only. ImageNet images, frozen encoder weights, decoder/DiT weights, and compatible latent normalization statistics are external assets. **An anonymous weight distribution must accompany the submission before full table reproduction is possible.**

For the reconstruction example, provide the frozen DINOv3-L encoder and these local files:

```text
checkpoints/
└── dinov3-vitl/decoder_k23/p0.95.safetensors
data_eval/
└── imagenet-256-val.npz
```

The NPZ's first array must contain RGB `uint8` images with shape `[N, 256, 256, 3]`. Follow the [reproduction guide](docs/reproduction.md) for preprocessing, readouts and asset requirements. Check the supplied decoder against the checkpoint manifest:

```bash
python scripts/validate_image_archive.py data_eval/imagenet-256-val.npz --min-images 100
python scripts/verify_checkpoints.py --root checkpoints \
  --file dinov3-vitl/decoder_k23/p0.95.safetensors
```

### 3. Reconstruct a small batch

```bash
python src/eval_reconstruction.py \
  --config configs/stage1/decoder/dinov3-k23-p095.yaml \
  --ckpt checkpoints/dinov3-vitl/decoder_k23/p0.95.safetensors \
  --val-npz data_eval/imagenet-256-val.npz \
  --num-images 100 --seed 0 --metrics psnr ssim \
  --out results/p095-smoke.json
```

This 100-image run checks loading and the data path. Formal reconstruction rows require `--num-images 50000 --metrics psnr ssim rfid` and the verified evaluation protocol.

## Included recipes

| Model | Configurations | Layer-drop rates | Epochs |
|---|---:|---|---:|
| K23 decoder | 8 | 0, .05, .1, .3, .5, .7, .9, .95 | 16 |
| K23 DiT-Base | 7 | 0, .05, .1, .3, .5, .7, .9 | 40 |
| K23 DiT-XL | 4 | 0, .5, .7, .9 | 40 |

The 19 configurations use DINOv3-L representations. Evaluation includes full K23, sparse K7, and layer 11 readouts; the fixed last-layer token-mean surrogate remains present for subset evaluation. Generation requires the latent statistics of the generator's training representation.

The no-drop decoder recipe is a derived template. Its original output normalization, the official reconstruction baselines, and the separate fixed-generator decoder-swap checkpoint mappings still require verification.

## Reproduction

The [experiment matrix](reproduction/experiments.json) records targets, required assets and execution status for every table row. The [reproduction guide](docs/reproduction.md) explains checkpoint mappings, protocol differences and result recording.

```bash
python scripts/plan_reproduction.py --table appendix --list
python scripts/run_table.py --table 4 --dry-run
```

`--table` takes a paper table number, a matrix key or a LaTeX label; the matrix's `paper_tables` block
records the numbering. The planner lists targets and checks asset presence. `run_table.py` materializes each reconstruction row's
evaluator command, runs it without `--dry-run`, and chains the comparison; it refuses generation rows,
official baselines and the unresolved `p_dec=0` normalization rather than substituting a default.

The appendix contains 24 reconstruction evaluations, 32 fixed-generator decoder-swap evaluations, and 64 DiT-XL evaluations. Generation uses 50,000 samples, 50 Euler steps, and either no guidance or internal guidance 1.78. Neither script certifies checkpoint mapping, backend equivalence or a tolerance.

## Verification

```bash
python -m pip install '.[dev]'
python -m pytest -q
```

[Validation evidence](reproduction/validation) records source compatibility checks and checkpoint integrity. Each executable exposes its arguments through `--help`.

Every archive member is listed in `MANIFEST.sha256`. Third-party attribution is retained in [LICENSE](LICENSE) and [THIRD_PARTY_NOTICE.md](THIRD_PARTY_NOTICE.md); encoder and checkpoint assets retain their separate terms.
