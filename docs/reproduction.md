# Reproducing the paper

The numerical values in [experiments.json](../reproduction/experiments.json) are **paper-reported targets**. All 175 unique evaluations currently have `status: "not_run"` and `reproduced: null`. Planning, checkpoint transfer and import checks do not establish numerical reproduction.

## What the matrix covers

| Table key | Evaluations | Protocol |
|---|---:|---|
| `appendix_decoder` | 24 | 8 decoder rates × 3 feeds: k7, k23, l11; PSNR, SSIM, rFID |
| `appendix_swap` | 32 | 8 decoders × k7/k23 fixed generators × no guidance/internal guidance 1.78; gFID, IS |
| `appendix_xl` | 64 | 4 generator rates × 8 decoder rates × 2 guidance settings; gFID, IS |
| `main_base` | 49 | 7 generator rates × 7 decoder rates, no guidance; gFID, IS |
| `main_reconstruction` | 6 | Official k7/k23 baseline decoders × 3 feeds; PSNR, SSIM, rFID |

The appendix requires 120 evaluations. The main p=.95 reconstruction row reuses three decoder-sweep evaluations; the main unguided XL grid reuses 32 appendix evaluations.

### Paper table numbers

`--table` accepts a paper table number, a matrix key, a LaTeX label, `appendix`, or `all`. The numbering
follows the current draft, as the README's reproduction table states it, and is recorded in the matrix's
`paper_tables` block.

| Paper table | Contents | Selector | Unique evaluations | Kind |
|---:|---|---|---:|---|
| 1 | Reconstruction across fusions | `--table 1` · `tab:recon-main` | 9 | reconstruction |
| 2 | Two stages, two scales: DiT-Base and DiT-XL grids, unguided | `--table 2` · `tab:scale-b`, `tab:scale-xl` | 81 | generation |
| 4 | Appendix: decoder random-drop rate sweep | `--table 4` · `tab:drop-sweep` | 24 | reconstruction |
| 5 | Appendix: one fixed generator, swapped decoders | `--table 5` · `tab:dropdecode` | 32 | generation |
| 6 | Appendix: DiT-XL rates at both guidance settings | `--table 6` · `tab:k23_dit_dec_heatmap` | 64 | generation |

Table 2 spans two matrix tables, the 49 DiT-Base cells and the 32 unguided DiT-XL cells, and each half carries
a second label for its IS panel — the same evaluations read for another metric, so either label selects the
whole half. Table 6 is a superset of table 2's DiT-XL half, adding the 32 guided rows. Tables 1 and 4 overlap
by three rows, because the main p=.95 reconstruction row is reused from the sweep rather than duplicated.
There is no table 3 here: the draft's table 3 is not an evaluation table.


## Make a plan and inspect missing assets

The planner uses only the Python standard library. It does not load checkpoints, download files or execute experiments.

```bash
python scripts/plan_reproduction.py --table appendix --list
python scripts/plan_reproduction.py --table appendix_decoder --p-dec 0.95 \
  --assets-root checkpoints --output results/reconstruction-plan.json
python scripts/plan_reproduction.py --table appendix_xl --p-dit 0 \
  --guidance none --output results/xl-baseline-plan.json
python scripts/plan_reproduction.py --table main_base --priority P0 --list
```

`--id` accepts an experiment-ID glob. Repeat `--table` or `--id` to select multiple groups; different filters are combined. An unmatched selection is an error.

The default asset layout accepts the released safetensors paths, for example `checkpoints/dinov3-vitl/decoder_k23/p0.95.safetensors`, and the original inventory paths such as `checkpoints/decoder-k23/decoder_k23_p0.95.pt`. To use different local filenames, create a bindings JSON object:

```json
{
  "decoder_k23_p0p95": "dinov3-vitl/decoder_k23/p0.95.safetensors",
  "encoder_dinov3_l": "encoder/dinov3_l_weights.pt",
  "imagenet_eval": "data/imagenet_eval.npz"
}
```

```bash
python scripts/plan_reproduction.py --id 'recon-p0p95-*' \
  --assets-root . --bindings local-assets.json --output results/plan.json
```

Paths in bindings are relative to `--assets-root`, or absolute. The names above are examples to replace with real assets. FuseReg reconstruction runs encoder → combiner → decoder directly and needs no separate latent statistics; reconstruction FID uses the same reference images. Generation and official RAEv2 models need their own compatible normalization assets. The current FID evaluator takes reference images, not a cached FID-statistics file.

The plan reports `missing`, `size_mismatch`, or `present_unverified` for each asset. Existing files remain unverified: the planner does not inspect model state, validate hashes or approve protocol compatibility. It intentionally leaves `runnable: false` until a separate model/protocol review and actual evaluation establish readiness. Local plans contain local paths and should not be included in the anonymous submission.

## Run a reconstruction table

`scripts/run_table.py` materializes the evaluator command for each selected row, runs it, and chains
`scripts/compare_result.py` on every result. It builds the same `src/eval_reconstruction.py` invocations
documented below and adds no protocol of its own.

```bash
python scripts/run_table.py --table 4 --dry-run
python scripts/run_table.py --table 4 \
  --assets-root checkpoints --val-npz data_eval/imagenet-256-val.npz \
  --results-dir results --summary results/drop-sweep-summary.json
```

It takes the planner's selection flags (`--table`, `--id`, `--priority`, `--p-dec`, `--p-dit`, `--guidance`)
and the same `--bindings` file. `--num-images` overrides the 50,000-image protocol for a smoke pass, which
`compare_result.py` then labels `smoke_only`. `--dry-run` prints the commands without running anything,
including for checkpoints that are not downloaded yet, so a selection can be inspected before any transfer.

Generation rows build `src/eval_fid_dit.py` invocations instead, and additionally need
`--latent-stats`. A table's shared latents come from fixing `--seed` and `--batch`: the sampled noise and
the class labels depend only on those, and the decoder enters after the ODE, so swapping decoders within one
`latent_reuse_group` renders the same latents without saving them. Changing either flag breaks that property.

```bash
python scripts/run_table.py --table 6 --latent-stats checkpoints/latent_stats.pt \
  --assets-root checkpoints --val-npz data_eval/imagenet-256-val.npz
```

The driver refuses a row rather than substituting a default when:

- the generator asset has no mapped checkpoint, as for the decoder-swap and DiT-Base tables;
- the table has no recorded generator architecture, as for the decoder-swap experiment whose architecture
  and epoch are unconfirmed;
- `--latent-stats` was not given, because a generator needs the statistics of its own training
  representation and freshly computed ones require protocol verification first;
- the decoder is an official RAEv2 baseline, which has no released training configuration here;
- the row is `p_dec=0` and no `--decoder-output-normalization` was given, because that lineage records no
  convention and the choice must not be made to improve agreement.

Each refusal prints the affected experiment IDs and, for generation rows, the unmapped asset IDs. A selection
in which no row is runnable exits non-zero. Execution does not verify checkpoint mapping, metric backend
equivalence or any tolerance: `protocol_verified` stays `false` in the summary it writes.

## Checkpoint mapping and protocol checks

The inventory contains filename candidates for all eight K23 decoder rates and XL p_dit=.5/.7/.9 epoch40. Base generators, official reconstruction baselines, encoder weights and metric assets were not mapped from that inventory.

The two no-drop generator epochs are now assigned by the source author: **`raev2_pdit0.0_ep040` is the XL grid baseline and `raev2_pdit0.0_ep080` is the fixed generator for the decoder-swap K23 columns.** That assignment explains why the two tables report different p_dit=0 baselines — 2.91/1.57 for the grid against 3.01/1.25 for the swap — and it replaces the earlier instruction to treat both epochs as interchangeable candidates. The assignment is authorship evidence, not a measured verification: the paper targets and each checkpoint's normalization still need checking.

The decoder-swap **K7 generator is the official RAEv2 model**: `stage2/imagenet/dinov3l-k7/checkpoint.pt` in `nyu-visionx/RAEv2-models`, the only stage-2 checkpoint that repository publishes. Download it together with its statistics:

```bash
hf download nyu-visionx/RAEv2-models --include "stage2/imagenet/dinov3l-k7/*" "stage1/imagenet/dinov3l-k7/stats.pt" \
  --local-dir checkpoints/raev2-models
```

Its [official sampling config](https://github.com/nanovisionx/RAEv2/blob/main/configs/stage2/sampling/imagenet-dinov3l-k7.yaml) makes it compatible with this release. The architecture is identical to the DiT-XL here — `DiTwDDTHeadIG`, hidden `[1440, 2048]`, depth `[28, 2]`, heads `[20, 16]`, `base_model_depth: 8` — and RAEv2's DINOv3-MLS encoder averages the selected layers and adds the final-layer token mean, exactly as `MLSCombine` does, although the RAEv2 README calls it multi-layer-sum. RAEv2's own name for layers `11,13,15,17,19,21,23` is "last 7 layers", so the paper's last-K7 readout and the source evaluator's sparse readout are the same set.

Two settings differ from this release's defaults and must be kept. **Normalization:** the official model was trained with per-position `[C, H, W]` statistics (`stage1/imagenet/dinov3l-k7/stats.pt`), whereas `compute_latent_stats.py` produces per-channel `[1, C, 1, 1]` statistics; the K7 columns must use the official file. **Guidance interval:** the paper names the official RAEv2 internal-guidance setting, which is scale 1.78 over `t ∈ [0.10, 1.0]`; the evaluator defaults to `[0, 1]`. The matrix records that interval on the sixteen guided decoder-swap rows and the driver passes it as `--ig-tmin 0.1 --ig-tmax 1.0`. The official sampling config also defaults to 100 steps; the matrix keeps the paper's 50.

The p=.95 original checkpoint was downloaded and its SHA256 matched the paper's layer-usage provenance: `49cf7f50310dd836f9d8282966c4ea5c188fbf03cbfe5504a09efe5fcec79a24`. All **456 EMA decoder tensors** exactly matched the released `p0.95.safetensors`, whose SHA256 is `f82adb05ca36c5bc22630a86c7439a630769bc2dd7c8f2a5e4043f0e38120476`. The checkpoint records epoch16/global-step80064, layers1–23 and an empty EMA combiner state; it uses `ema_dec` with `configs/decoder/ViTXL`. This verifies that export's identity, not the paper's metrics or evaluator protocol. Other filename candidates remain subject to their stated verification scope.

Before formal runs, verify these details against each checkpoint and original configuration:

- **Fusion and normalization:** exact layer IDs/indexing, sum versus mean, encoder/decoder normalization, latent stats, and the fixed final-layer token-mean surrogate. In particular, k7 must not silently become the first or last seven array positions. The layer-usage figure preserves the final-layer surrogate even for single-layer probes.
- **Model state:** architecture, original or EMA weights, training epoch and any combiner or semantic state used during inference. The official RAEv2 reconstruction baseline is different from the released/source no-drop decoder candidate.
- **Evaluation:** ImageNet split and preprocessing, FID/IS implementation and reference statistics, sampling seed, and internal-guidance algorithm/settings. These details are incompletely specified in the manuscript. `src/eval_reconstruction.py --help` describes the reconstruction interface; `src/eval_fid_dit.py` is the generation evaluation entry point.

The preserved `MLSCombine` implementation repairs an all-dropped mask by keeping one uniformly selected layer. The manuscript instead models Bernoulli sampling conditioned on a nonempty mask. These are different distributions: at K23/p=.95, about 30.7% of raw masks are empty. The release preserves the checkpoint's existing recipe; retraining with rejection sampling would change that recipe. The fixed last-layer token-mean surrogate is also additional information beyond the subset-mean mathematical model.

The DiT implementation fixes the oldnorm recipe: x-prediction, a random-subset input with a full-fusion target, logit-normal time sampling, and a latent-dimension time shift. The method text writes uniform time sampling. Preserve the source recipe and resolve that reporting difference before describing a retraining run as identical to the mathematical protocol. `compute_latent_stats.py` measures the deterministic full-fusion encoder latents used by the retained DiT training recipes. All rates share the full-fusion target, but each released generator still requires its verified normalization statistics.

Generation supports the paper's two sampling settings through internal guidance: `--ig-scale 1.0` selects the full model without guidance and `--ig-scale 1.78` uses the learned full/base outputs. Sampling uses Euler integration and table runs require `--steps 50`; `--ig-tmin` and `--ig-tmax` specify the guidance interval, whose match to the original evaluation must be verified.

The manuscript's default-rate paragraph says .9 while its main reconstruction row uses .95. Use each matrix row's explicit rate. The separate decoder-swap k23 baseline reports gFID 3.01 without guidance and 1.25 with guidance; the XL-grid baseline reports 2.91 and 1.57. Do not merge these checkpoints or targets merely because both have p_dit=0.

## Execution order and result recording

Reconstruction input is an NPZ whose first array is RGB `uint8` with shape `[N, 256, 256, 3]`; it must contain at least the requested image count. The evaluator samples image indices using `--seed` and does not resize the archive's images. An NPY array with the same shape is also accepted. Prepare the images using the verified paper crop/resize pipeline before packaging them.

Reject an incorrectly shaped or undersized archive before loading any checkpoints or models:

```bash
python scripts/validate_image_archive.py data_eval/imagenet-256-val.npz --min-images 50000
```

This is a header-only preflight. It does not inspect pixels or verify the archive's preprocessing provenance.

For the released p=.95 decoder, this is an executable loading/data smoke test:

```bash
python src/eval_reconstruction.py \
  --config configs/stage1/decoder/dinov3-k23-p095.yaml \
  --ckpt checkpoints/dinov3-vitl/decoder_k23/p0.95.safetensors \
  --val-npz data_eval/imagenet-256-val.npz \
  --num-images 100 --batch 32 --seed 0 --metrics psnr ssim \
  --decoder-output-normalization encoder --out results/p095-smoke.json
```

Omitting a subset selects all configured layers1–23. `--layers 11` selects actual encoder block11; `--layers 11,13,15,17,19,21,23` selects the sparse k7 readout recorded by the source implementation. Alternatively, `--idx 10,12,14,16,18,20,22` selects those positions in the configured K23 list. Do not pass both flags. The source implementation establishes this sparse readout, but its exact association with every reported k7 baseline still needs verification. The last-layer surrogate remains fixed even for `--layers 11`.

For a formal row, request `--num-images 50000 --metrics psnr ssim rfid` and save a separate output for each feed. Oldnorm decoders predict encoder-normalized pixels, so `--decoder-output-normalization encoder` applies the encoder's inverse pixel normalization. `raw` describes a different training convention and must not be selected to improve the comparison. Official RAEv2 baselines may need their original normalization/adapter; the custom decoder command is not automatically an official-baseline reproduction.

First run a small smoke test to check loading and preprocessing. Then run the selected rows on the full 50k protocol. Prioritize reconstruction at p_dec=0/.9/.95, decoder-swap at p_dec=0/.95, and the unguided XL p_dit=0 baseline and decoder changes. The Base `P0` selection contains the four cells used for the paper's joint-rate comparison.

For decoder-swap, save or deterministically reuse the same 50k sampled latents across all eight decoders. Base/XL grids can similarly reuse latents for each generator and guidance setting; `sampling.latent_reuse_group` records this grouping. Preserve the shared generator's latent convention while swapping decoders. The appendix alone entails 4.8 million generated-image renderings and 1.2 million reconstruction renderings; measure throughput on the actual CUDA machine before estimating completion time. The paper gives no GPU model, runtime or multi-seed tolerance.

Record actual metrics separately with experiment ID, checkpoint/data hashes, config, seed, sample count, code revision and raw output path. Reconstruction currently emits `psnr`, `ssim` and **`rfid_tf`** using torch-fidelity with paired reference images; generation emits **`fid`** and `is`. The matrix's `runtime_metric_names` records the correspondence to paper `rfid`/`gfid`, but a name mapping does not establish backend equivalence. Retain the actual backend and preprocessing when reporting differences.

Compare each metric against `paper_reported`, reporting the absolute difference and any protocol deviation. Do not copy targets into `reproduced`. Lower-sample smoke tests should retain their actual sample count and cannot count as a 50k paper reproduction. Main-table rFID is rounded more coarsely than the detailed decoder sweep; the manifest preserves the detailed values.

Subset/Shapley and qualitative appendix figures need additional sampling/image-selection protocols that are not fully specified. Replotting existing figure data is distinct from rerunning the underlying model evaluation.
