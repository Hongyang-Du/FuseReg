<div align="center">

# FuseReg: Regularizing Layer Fusion Mitigates the Reconstruction–Generation Gap in Representation Autoencoders

<p>
  <a href="https://hongyang-du.github.io/">Hongyang Du</a><sup>1,2</sup> &nbsp;&nbsp; <a href="https://yunfeixie233.github.io/">Yunfei Xie</a><sup>3</sup> &nbsp;&nbsp; <a href="https://junjieye.com/">Junjie Ye</a><sup>1</sup> &nbsp;&nbsp; <a href="https://jiawei-yang.github.io/">Jiawei Yang</a><sup>1</sup> &nbsp;&nbsp; <a href="https://oliver-cong02.github.io/">Xiaoyan Cong</a><sup>2</sup> &nbsp;&nbsp; <a href="https://cs.brown.edu/people/grad/hzhan351/">Haodong Zhang</a><sup>2</sup><br>
  <a href="https://www.abdn.ac.uk/people/yongchao.huang">Yongchao Huang</a><sup>4</sup> &nbsp;&nbsp; <a href="https://haiyuwu.github.io/">Haiyu Wu</a><sup>5</sup> &nbsp;&nbsp; <a href="https://zli12321.github.io/">Zongxia Li</a><sup>6</sup> &nbsp;&nbsp; <a href="https://www.linkedin.com/in/shihang-gui-06917727b/">Shihang Gui</a><sup>2</sup> &nbsp;&nbsp; <a href="https://davidliuk.github.io/">Dawei Liu</a><sup>7</sup> &nbsp;&nbsp; <a href="https://www.linkedin.com/in/runhao-li-lee021004/">Runhao Li</a><sup>1</sup><br>
  <a href="https://jingchengni.com/">Jingcheng Ni</a><sup>2</sup> &nbsp;&nbsp; <a href="https://weichen582.github.io/">Chen Wei</a><sup>3,†</sup> &nbsp;&nbsp; <a href="https://randallbalestriero.github.io/">Randall Balestriero</a><sup>2,†</sup> &nbsp;&nbsp; <a href="https://yuewang.xyz/">Yue Wang</a><sup>1,†</sup>
</p>

<p>
  <sup>1</sup>USC PSI Lab &nbsp;&nbsp; <sup>2</sup>Brown University &nbsp;&nbsp; <sup>3</sup>Rice University &nbsp;&nbsp; <sup>4</sup>University of Aberdeen<br>
  <sup>5</sup>University of Notre Dame &nbsp;&nbsp; <sup>6</sup>University of Maryland, College Park &nbsp;&nbsp; <sup>7</sup>University of Pennsylvania<br>
  <sup>†</sup>Equal advising &nbsp;·&nbsp; Corresponding to <a href="mailto:hongyang_du@brown.edu">hongyang_du@brown.edu</a>
</p>

<p>
  <a href="https://hongyang-du.github.io/FuseReg/"><img src="https://img.shields.io/badge/Project_Page-FuseReg-1769aa?style=for-the-badge&logo=googlechrome&logoColor=white" alt="Project page"></a>
  <a href="#"><img src="https://img.shields.io/badge/arXiv-coming_soon-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white" alt="arXiv"></a>
  <a href="https://huggingface.co/Hongyang-Du/FuseReg"><img src="https://img.shields.io/badge/Hugging_Face-Models-ffd21e?style=for-the-badge&logo=huggingface&logoColor=black" alt="Hugging Face models"></a>
</p>

</div>

<p align="center">
  <img src="assets/figures/cosine-similarity.png" width="100%" alt="Cosine-similarity maps of decoder intermediate-block features for FuseReg (top) and RAEv2 (bottom), fed K=23, K=7 and single-layer l11 fusions, with reconstructions.">
</p>

**One decoder, every fusion.** FuseReg (top) keeps stable spatial structure under *K*=23, *K*=7 and the single layer ℓ<sub>11</sub>; the fixed-fusion RAEv2 decoder (bottom) degrades away from its training fusion.

## Overview

Representation autoencoders ask one fused latent to serve two different goals: shallow encoder features preserve pixel detail, while deeper features tend to be easier to model generatively. A fixed layer fusion hard-codes this trade-off into both the decoder and the generator.

**FuseReg turns layer fusion from a fixed heuristic into a training distribution.** Each training sample averages a random non-empty subset of encoder layers; the released latent also retains a fixed final-layer token-mean surrogate. The decoder learns to reconstruct from changing layer compositions, and the diffusion transformer learns from noisy subset representations while targeting the full-layer fusion. Inference uses the full-layer mean by default, with **no architecture change or additional inference cost**.

Averaging over the retained layers keeps the deployment latent unchanged in expectation, so randomization varies only the directions along which encoder layers disagree. The two stages use separate rates, `p_dec` and `p_dit`.

> [!IMPORTANT]
> **Paper formulation vs. released implementation:** the manuscript conditions Bernoulli masks on retaining at least one layer. The checkpoint-compatible source instead repairs an all-dropped draw by retaining one uniformly selected layer. Both support every non-empty subset, but they assign different probabilities; at `K=23, p=.95`, about 30.7% of raw draws are empty before repair. Preserve the released behavior for checkpoint-compatible execution, and do not describe a retraining run as distribution-identical to the manuscript without resolving this difference.

## Results

### One regularizer, two stages, two scales

<p align="center">
  <img src="assets/figures/joint-regularization-table.png" width="100%" alt="Paper Table 2: gFID and IS grids over generator rate p_dit (rows) and decoder rate p_dec (columns) for DiT-Base and DiT-XL. Best DiT-Base gFID is 9.93 at p_dec=0.9, p_dit=0.7; best DiT-XL gFID is 2.38 at p_dec=0.95, p_dit=0.">
</p>

**Two stages, two scales.** Unguided ImageNet-256. Regularizing both stages gives the best gFID: **13.96 → 9.93** on DiT-Base and **2.91 → 2.38** on DiT-XL, against the fixed-fusion baseline in the green box.

### Fixed generator, swapped decoders

<p align="center">
  <img src="assets/figures/decoder-swap.png" width="100%" alt="Paper decoder-swap results: gFID and Inception Score across decoder layer-drop rates for K23 and K7 readouts, without guidance and with internal guidance 1.78.">
</p>

**Same generator, same latents.** Swapping in a regularized decoder alone lowers gFID from **3.01 → 2.21** at *k*=23 and from **27.73 → 1.92** at *k*=7.

### One decoder, different layer readouts

<p align="center">
  <img src="assets/figures/reconstruction-readouts.png" width="100%" alt="Paper reconstruction comparison: the same FuseReg decoder and fixed-fusion RAEv2 decoders reconstruct four images from last-seven-layer and single-layer L11 readouts. Insets show PSNR in dB.">
</p>

**Reconstruction off the training fusion.** Fixed-fusion RAEv2 decoders break down; FuseReg does not. Insets show PSNR in dB.

## Discussion

### FuseReg distributes reconstruction across layers

<p align="center">
  <img src="assets/figures/layer-usage.png" width="100%" alt="Paper figure: leave-one-layer-out PSNR drop (left) and single-layer PSNR (right) for the RAEv2 decoder and FuseReg with p_dec=0.95, frozen DINOv3 K=23.">
</p>

**Distributed reconstruction.** FuseReg has a flatter leave-one-layer-out profile (left) and stronger single-layer readouts (right), on 10,000 held-out ImageNet-256 images.

Because the encoder is frozen, this reflects the decoder's ability to recover information already available across depths: improving the readout of a fixed representation can improve generation even when the generator and sampled latents are unchanged.

## Quick start

The manifest and planner can be inspected without credentials, checkpoints, ImageNet, or a package install:

```bash
git clone https://github.com/Hongyang-Du/FuseReg.git
cd FuseReg
python3 scripts/plan_reproduction.py --table appendix --list
```

The planner only reads release metadata; it does not download assets, load a model, or claim that an experiment is ready. The reconstruction path below uses **separately licensed, access-controlled assets** and is not a zero-asset demo. The release was validated with **Python 3.11**; package metadata supports Python 3.10–3.13. Use a **Linux CUDA machine** for full experiments.

### 1. Install

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

`requirements.txt` installs the project and the exact pair `torch==2.10.0`, `torchvision==0.25.0`. If the default index is not appropriate for your CUDA runtime, install that pair from the matching [official PyTorch wheel index](https://pytorch.org/get-started/locally/) first, then install FuseReg. For example, for a system compatible with the CUDA 12.8 index:

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cu128 \
  'torch==2.10.0' 'torchvision==0.25.0'
python -m pip install -e . --extra-index-url https://download.pytorch.org/whl/cu128
python - <<'PY'
import torch, torchvision
print(torch.__version__, torchvision.__version__, torch.version.cuda)
PY
```

Replace `cu128` with the official index matching your system. The second command reuses the already installed exact pair while resolving the remaining project dependencies. Do not silently substitute another PyTorch pair for a paper-comparison run.

### 2. Download a decoder

The checkpoint repository is currently access-controlled. Request/obtain access on [Hugging Face](https://huggingface.co/Hongyang-Du/FuseReg), authenticate with `hf auth login`, then download and verify the paper's p = 0.95 robustness decoder:

```bash
hf auth login

hf download Hongyang-Du/FuseReg \
  dinov3-vitl/decoder_k23/p0.95.safetensors \
  --local-dir checkpoints

python scripts/verify_checkpoints.py \
  --root checkpoints \
  --file dinov3-vitl/decoder_k23/p0.95.safetensors
```

You also need the frozen encoder and evaluation images:

| Asset | Expected location or format |
|---|---|
| DINOv3-L weights | Obtain under the [DINOv3 access terms](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m); place at `pretrained_models/encoders/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth` |
| ImageNet evaluation images | `data_eval/imagenet-256-val.npz` — RGB `uint8`, shape `[N, 256, 256, 3]` |
| DINO S/8 discriminator weights (training only) | `pretrained_models/encoders/dino/dino_vit_small_patch8_224.pth` |

Set `DINOV3_CKPT_DIR` to use a different encoder-weight directory. The loader fetches the pinned DINOv3 source revision through `torch.hub`; set `DINOV3_REPO_DIR` to a local DINOv3 checkout containing `hubconf.py` for offline clusters. Obtain encoder weights and ImageNet under their respective access terms. The release intentionally does not redistribute ImageNet or claim a newly generated reference archive; the [data format guide](docs/reproduction.md#execution-order-and-result-recording) records the required shape and protocol boundary.

Before any model is loaded, validate the archive header and minimum sample count:

```bash
python scripts/validate_image_archive.py \
  data_eval/imagenet-256-val.npz --min-images 100
```

This checks the first array's dtype and `[N, 256, 256, 3]` shape without reading the image payload. It does **not** verify image values, crop/resize provenance, or agreement with the paper's evaluation cohort.

### 3. Reconstruct images

```bash
python src/eval_reconstruction.py \
  --config configs/stage1/decoder/dinov3-k23-p095.yaml \
  --ckpt checkpoints/dinov3-vitl/decoder_k23/p0.95.safetensors \
  --val-npz data_eval/imagenet-256-val.npz \
  --num-images 100 --seed 0 --metrics psnr ssim \
  --out results/p095-smoke.json
```

This checks loading and evaluation on 100 images. A 50k evaluation candidate uses `--num-images 50000 --metrics psnr ssim rfid`, but the exact checkpoint mapping, preprocessing and metric protocol must still be verified before calling the result a paper reproduction.

<details>
<summary><strong>Choose a reconstruction readout</strong></summary>

The paper and released source currently differ in how K7 is identified. Until that mapping is resolved, use K23 as the default checkpoint-compatible readout and label custom subsets explicitly as source-code probes:

| Readout | Argument | Encoder block IDs |
|---|---|---|
| K23 | No additional argument | 1–23 |
| Source-code sparse K7 probe | `--layers 11,13,15,17,19,21,23` | 11, 13, 15, 17, 19, 21, 23 |
| Layer 11 | `--layers 11` | 11 |

`--layers` uses encoder block IDs; `--idx` uses positions within the configured layer list. Do not silently equate this sparse source probe with every paper baseline labeled “last K7.” The fixed final-layer surrogate remains present for every readout. Use a distinct `--out` path for each experiment.

</details>

<details>
<summary><strong>Run the 50k K23 evaluation template</strong></summary>

The paired reference/reconstruction arrays require roughly **20 GB of temporary disk**. Point `--work-dir` at fast scratch storage:

```bash
python src/eval_reconstruction.py \
  --config configs/stage1/decoder/dinov3-k23-p095.yaml \
  --ckpt checkpoints/dinov3-vitl/decoder_k23/p0.95.safetensors \
  --val-npz data_eval/imagenet-256-val.npz \
  --num-images 50000 --seed 0 --metrics psnr ssim rfid \
  --decoder-output-normalization encoder \
  --work-dir /fast/scratch/fusereg-fid \
  --out results/p095-k23.json
```

</details>

## Checkpoints

The [checkpoint manifest](reproduction/checkpoints.json) records **13 paper-relevant candidates**, including file paths, SHA-256 hashes and verification scope.

| Family | Layer-drop rates | Checkpoint details |
|---|---|---|
| K23 decoder | 0, .05, .1, .3, .5, .7, .9, .95 | 8 EMA exports · ~1.55 GiB each · p=0 normalization unresolved |
| K23 DiT-XL | .5, .7, .9 | 40-epoch EMA exports · ~3.26 GiB each |
| No-drop generator | 0 | 40- and 80-epoch candidates · ~3.26 GiB each · table assignment unverified |

Released `safetensors` contain EMA inference weights. Original `.pt` training checkpoints are also supported; inference exports contain no optimizer state for resuming training.

<details>
<summary><strong>Baseline mapping and normalization</strong></summary>

- The released/source p = 0 decoder candidate and the official RAEv2 K23 decoder are distinct assets.
- The p = 0 decoder's source checkpoint does not record its output normalization. Confirm the original recipe, then explicitly select `--decoder-output-normalization raw` or `encoder`.
- Generation requires the latent statistics of the generator's training representation. Matching filenames or tensor shapes do not establish compatible normalization.
- The fixed-generator decoder-swap table and the XL-grid baseline have different paper targets. Their checkpoint mappings must be verified separately.

See the [checkpoint and protocol guide](docs/reproduction.md#checkpoint-mapping-and-protocol-checks) before a paper comparison.

</details>

## Reproducing the paper's tables

Each table in the paper maps to one command over a set of checkpoints. The planner prints the exact runs for a table, including paper targets and required assets, without loading a model:

```bash
python scripts/plan_reproduction.py --table tab:recon-main --list
python scripts/plan_reproduction.py --table appendix --list
```

| Paper table | Contents | How to reproduce |
|---|---|---|
| **Table 1** · reconstruction across fusions | PSNR, SSIM, rFID for the RAEv2 K7, RAEv2 K23 and FuseReg p=.95 decoders under `k=7`, `k=23` and `ℓ11` | `src/eval_reconstruction.py` once per decoder and readout, `--num-images 50000 --metrics psnr ssim rfid` |
| **Table 2** · two stages, two scales | gFID and IS over the `p_dec` × `p_dit` grid for DiT-Base and DiT-XL, no guidance | `src/eval_fid_dit.py --num-samples 50000 --steps 50 --ig-scale 1.0` per (decoder, generator) pair |
| **Table 4** (appendix) · decoder rate sweep | 8 decoders × 3 readouts | same command as Table 1, over the eight `p_dec` checkpoints |
| **Table 5** (appendix) · generation when only the decoder changes | 8 decoders × 2 readouts × 2 guidance settings, with one fixed generator | `src/eval_fid_dit.py` with the generator held fixed, `--ig-scale 1.0` and `--ig-scale 1.78` |
| **Table 6** (appendix) · DiT-XL rates under guidance | 4 generators × 8 decoders × 2 guidance settings | `src/eval_fid_dit.py --ig-scale 1.78` per pair |

Rather than issuing those commands by hand, `scripts/run_table.py` materializes them from the matrix, runs
them and chains the comparison, one row at a time:

```bash
python scripts/run_table.py --table 4 --dry-run
```

`--table` takes a paper table number, a matrix key or a LaTeX label; the
[numbering](docs/reproduction.md#paper-table-numbers) is recorded in the matrix. Drop `--dry-run` to execute.
Rows whose assets are unmapped, or whose protocol is unresolved, are reported as unrunnable with the missing
asset IDs rather than approximated.

Generation uses **50,000 class-conditional samples** and **50 Euler steps**. In `src/eval_fid_dit.py`, `--ig-scale 1.0` selects no guidance and `--ig-scale 1.78` selects internal guidance.

Figure 3 uses the Table 5 runs, and Figure 4 uses the leave-one-out and single-layer readouts of `src/eval_reconstruction.py`.

**Table 1, FuseReg p = 0.95 row**

| Readout | PSNR ↑ | SSIM ↑ | rFID ↓ |
|---|---:|---:|---:|
| `k=23` | 27.52 | 0.826 | 0.42 |
| last `k=7` | 23.77 | 0.678 | 0.60 |
| `ℓ11` | 25.13 | 0.735 | 0.45 |

Reconstruction uses **50,000 ImageNet-256 images**; generation uses **50,000 class-conditional samples** and **50 Euler steps**. In `src/eval_fid_dit.py`, `--ig-scale 1.0` is no guidance and `--ig-scale 1.78` is internal guidance.

Compare a finished run against its paper target:

```bash
python scripts/compare_result.py \
  --experiment recon-p0p95-k23 \
  --result results/p095-k23.json \
  --out results/p095-k23-comparison.json
```

The [reproduction guide](docs/reproduction.md) covers execution order, asset bindings, baseline mappings and known differences between the manuscript and the source training recipe.

## Training

The **19 training configurations** follow the paper's rate grids:

| Model | Layer-drop rates | Epochs | Recipes |
|---|---|---:|---|
| K23 decoder | 0, .05, .1, .3, .5, .7, .9, .95 | 16 | [8 configs](configs/stage1/decoder) |
| K23 DiT-Base | 0, .05, .1, .3, .5, .7, .9 | 40 | [7 configs](configs/stage2/training) |
| K23 DiT-XL | 0, .5, .7, .9 | 40 | [4 configs](configs/stage2/training) |

The no-drop decoder config is a derived template; its exact original training recipe remains unresolved. The retained DiT recipe uses **x-prediction**, **logit-normal time sampling** and a **full-fusion target**. Preserve the effective global batch when changing GPU count or gradient accumulation.

<details>
<summary><strong>Stage 1 · Train a decoder</strong></summary>

```bash
torchrun --standalone --nproc_per_node=8 src/train_decoder.py \
  --config configs/stage1/decoder/dinov3-k23-p03.yaml \
  data.data_dir=data/imagenet \
  data.val_npz=data_eval/imagenet-256-val.npz \
  training.out_dir=outputs/decoder-p03 \
  loss.gan.disc_ckpt=pretrained_models/encoders/dino/dino_vit_small_patch8_224.pth \
  wandb.enabled=false
```

The training loader accepts ImageFolder data under `data/imagenet/train/` or Arrow shards under `data/imagenet/imagenet-latents-images/`. The DINO S/8 discriminator weights are an external dependency and are passed explicitly above.

</details>

<details>
<summary><strong>Stage 2 · Compute latent statistics and train DiT</strong></summary>

Install the pinned GMuon optimizer:

```bash
python -m pip install '.[train]'
```

Compute statistics for the exact full-fusion encoder readout:

```bash
torchrun --standalone --nproc_per_node=1 src/compute_latent_stats.py \
  --config configs/stage2/training/dinov3-k23-fulltarget-p09-ditxl.yaml \
  --num-samples 50000 --batch 32 \
  --out checkpoints/latent_stats.pt \
  dataset.data_dir=data/imagenet \
  stage_1.params.stage1_ckpt_path=checkpoints/dinov3-vitl/decoder_k23/p0.95.safetensors
```

Then train the generator. The recipe fixes a global batch of 1024, so the micro-batch is `1024 / (world_size × GRAD_ACCUM_OVERRIDE)`. For example, `GRAD_ACCUM_OVERRIDE=8` gives a micro-batch of 16 on eight processes; choose a divisor compatible with memory while preserving the global batch:

```bash
GRAD_ACCUM_OVERRIDE=8 torchrun --standalone --nproc_per_node=8 src/train.py \
  --config configs/stage2/training/dinov3-k23-fulltarget-p09-ditxl.yaml \
  --results-dir outputs/ditxl-p09 --precision bf16 \
  dataset.data_dir=data/imagenet \
  stage_1.params.stage1_ckpt_path=checkpoints/dinov3-vitl/decoder_k23/p0.95.safetensors \
  stage_1.params.normalization_stat_path=checkpoints/latent_stats.pt
```

The provided recipes default to a saved Hugging Face Arrow ImageNet dataset under `data/imagenet/imagenet-latents-images/`. Use the exact training representation's statistics for released generators; newly computed statistics require protocol verification before comparison with the paper.

</details>

## Verification and supplement

Release checks recorded on source revision [`cf07184`](https://github.com/Hongyang-Du/FuseReg/commit/cf071846f097ac4d79be56afacde99d751f612a7):

| Check | Recorded result (2026-09-19) |
|---|---|
| Test suite | 25 tests + 3 subtests passed |
| Configurations and CLI entry points | 19 configs + 6 entry points validated |
| Fusion, decoder and DiT compatibility | Exact agreement in recorded CPU comparisons |
| p = 0.95 decoder | Real-weight CPU forward passed; all 456 EMA tensors match the original |
| DiT-XL checkpoint structure | 572 tensor names and shapes match available headers |

Full evidence is in [reproduction/validation](reproduction/validation).

```bash
python -m pip install '.[dev]'
python -m pytest -q
python scripts/build_supplement.py --output dist/fusereg-supplement.zip
```

The anonymous archive contains source, configurations and a SHA-256 manifest. It substitutes an anonymous README and excludes Git history, credentials, data, checkpoints and local results. Full evaluation additionally requires an anonymous distribution of the external assets. Third-party attribution is preserved.

## Scope and limitations

- **Empirical scope.** Current evidence covers ImageNet-256 with a frozen DINOv3-L encoder and DiT-Base/XL under matched budgets. Transfer to other encoders, resolutions, domains and longer schedules has not been verified.
- **Two rates must be selected separately.** Preferred `p_dec` and `p_dit` values depend on model scale, metric and guidance; the heatmap is not a universal default.
- **Theory boundary.** The objective decomposition assumes homogeneous linear predictors and squared loss. Nonlinear-model benefits are empirical; mixed reconstruction losses and the complete guided trajectory remain open.
- **Release boundary.** Full numerical reproduction remains gated by separately licensed assets and the unresolved protocol items in the [reproduction guide](docs/reproduction.md).

## Repository map

| Path | Contents |
|---|---|
| [`src/stage1/`](src/stage1) | Layer fusion, reconstruction decoder and discriminator |
| [`src/stage2/`](src/stage2) | DiT architecture, full-target loss and Euler sampling |
| [`configs/`](configs) | Decoder architecture and paper training recipes |
| [`reproduction/`](reproduction) | Experiment matrix, checkpoint manifest and validation evidence |
| [`docs/reproduction.md`](docs/reproduction.md) | Evaluation protocol and remaining reproduction dependencies |
| [`scripts/`](scripts) | Planning, table execution, integrity checks, result comparison and ZIP packaging |

## License

See [LICENSE](LICENSE) for the repository's inherited code license and [THIRD_PARTY_NOTICE.md](THIRD_PARTY_NOTICE.md) for upstream attribution. Paper figures/PDF, released checkpoints, DINO weights and ImageNet data may have separate terms; users are responsible for following the terms attached to each asset.

## Citation

Provisional project citation; replace it with the stable venue/arXiv entry when one is available.

```bibtex
@misc{du2026fusereg,
  title  = {FuseReg: Regularizing Layer Fusion Mitigates the Reconstruction-Generation Gap in Representation Autoencoders},
  author = {Du, Hongyang and Xie, Yunfei and Ye, Junjie and Yang, Jiawei and Cong, Xiaoyan
            and Zhang, Haodong and Huang, Yongchao and Wu, Haiyu and Li, Zongxia
            and Gui, Shihang and Liu, Dawei and Li, Runhao and Ni, Jingcheng
            and Wei, Chen and Balestriero, Randall and Wang, Yue},
  year   = {2026},
  url    = {https://hongyang-du.github.io/FuseReg/}
}
```
