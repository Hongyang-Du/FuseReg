#!/usr/bin/env python3
"""Materialize one table's evaluator commands from the matrix, run them, and compare each result.

This is a driver, not new science: every command it builds is the documented
``src/eval_reconstruction.py`` invocation for one matrix row. It refuses rows whose protocol or
assets are unresolved rather than substituting a default, and running it does not establish
checkpoint association, backend equivalence or any acceptance tolerance.
"""

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plan_reproduction import inspect_asset, read_json, select_experiments

ROOT = Path(__file__).resolve().parents[1]
DECODER_CONFIG_DIR = ROOT / "configs" / "stage1" / "decoder"
OFFICIAL_DECODER_CONFIG = ROOT / "configs" / "stage1" / "official" / "raev2-dinov3l-eval.yaml"
GENERATOR_CONFIG_DIR = ROOT / "configs" / "stage2" / "training"
GENERATOR_ARCH = {"appendix_xl": "ditxl", "main_base": "ditb", "appendix_swap": "ditxl"}
GUIDANCE_SCALE = {None: "1.0", 1.78: "1.78"}
FEED_LAYERS = {"k23": None, "k7": "11,13,15,17,19,21,23", "l11": "11"}
SELECTORS = ("table", "id", "priority", "p_dec", "p_dit", "guidance")
METRIC_ORDER = ("psnr", "ssim", "rfid")


def rel(path):
    """Print repository paths the way the documentation writes them; commands run with cwd=ROOT."""
    path = Path(path)
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def decoder_configs():
    """Map each released decoder rate to its config by reading p_drop, not by filename math."""
    mapping = {}
    for path in sorted(DECODER_CONFIG_DIR.glob("*.yaml")):
        for line in path.read_text(encoding="utf-8").splitlines():
            key, _, value = line.strip().partition(":")
            if key == "p_drop":
                mapping[float(value)] = path
                break
    return mapping


def generator_configs(arch):
    """Map each generator rate to its stage-2 recipe by reading p_drop, as for decoders."""
    mapping = {}
    for path in sorted(GENERATOR_CONFIG_DIR.glob(f"*-{arch}.yaml")):
        for line in path.read_text(encoding="utf-8").splitlines():
            key, _, value = line.strip().partition(":")
            if key == "p_drop":
                mapping[float(value)] = path
                break
    return mapping


def resolve_asset(asset_id, manifest, assets_root, bindings):
    """Return (checked paths, a usable file or None, the sizes that disagree with the inventory).

    A file whose size contradicts the inventory is not usable: it is the wrong export, a truncated
    download or a same-named different run, and running it would produce plausible but wrong numbers.
    """
    report = inspect_asset(asset_id, manifest["assets"][asset_id], assets_root, bindings)
    checked = [c["path"] for c in report["checked_paths"]]
    present = [c for c in report["checked_paths"] if c["exists"]]
    mismatched = [c["path"] for c in present if c["size_matches_inventory"] is False]
    usable = [c["path"] for c in present if c["size_matches_inventory"] is not False]
    return checked, (Path(usable[0]) if usable else None), mismatched


def size_mismatch_reason(asset_id):
    return (f"{asset_id} is present but its size contradicts the inventory: it is a different export, "
            "a truncated download or another run. Re-fetch it, or bind the intended file.")


def resolve_checkpoint(exp, manifest, assets_root, bindings):
    """Return the row's decoder asset id, its checked paths, and the present file if there is one."""
    decoders = [a for a in exp["required_assets"] if a.startswith("decoder_")]
    if len(decoders) != 1:
        raise ValueError(f"{exp['id']} does not name exactly one decoder asset")
    asset_id = decoders[0]
    checked, present, mismatched = resolve_asset(asset_id, manifest, assets_root, bindings)
    return asset_id, checked, present, mismatched


def build_command(exp, ckpt, config, args, output_normalization=None):
    metrics = [m for m in METRIC_ORDER if m in exp["paper_reported"]]
    if len(metrics) != len(exp["paper_reported"]):
        raise ValueError(f"{exp['id']} reports a metric the reconstruction evaluator does not emit")
    command = [
        args.python, "src/eval_reconstruction.py",
        "--config", rel(config), "--ckpt", rel(ckpt), "--val-npz", str(args.val_npz),
        "--num-images", str(args.num_images or exp["num_samples"]),
        "--batch", str(args.batch), "--seed", str(args.seed),
        "--metrics", *metrics,
        "--device", args.device,
        "--out", str(args.results_dir / f"{exp['id']}.json"),
    ]
    layers = FEED_LAYERS[exp["fusion"]]
    if layers:
        command += ["--layers", layers]
    output_normalization = output_normalization or args.decoder_output_normalization
    if output_normalization:
        command += ["--decoder-output-normalization", output_normalization]
    if args.work_dir:
        command += ["--work-dir", str(args.work_dir)]
    return command


def guidance_interval(exp):
    """Pass the recorded internal-guidance interval; rows without one keep the evaluator default."""
    interval = exp["sampling"].get("guidance_interval")
    if exp["guidance"] is None or not interval:
        return []
    return ["--ig-tmin", str(interval[0]), "--ig-tmax", str(interval[1])]


def build_generation_command(exp, generator, decoder, config, latent_stats, args):
    """One eval_fid_dit.py invocation. A fixed seed and batch give the table its shared latents:
    the noise and labels depend only on those, and the decoder enters after the ODE.

    Sampling sets load_encoder=False, so a row's fusion is carried entirely by its generator and its
    normalization statistics. The decoders are the same K23-trained set in every column; feeding them a
    K7 generator's latents is the shifted fusion the decoder-swap table measures.
    """
    overrides = [
        f"stage_1.params.stage1_ckpt_path={rel(decoder)}",
        f"stage_1.params.normalization_stat_path={rel(latent_stats)}",
    ]
    if args.decoder_output_normalization:
        overrides.append(
            f"stage_1.params.decoder_output_normalization={args.decoder_output_normalization}")
    return [
        args.python, "src/eval_fid_dit.py",
        "--config", rel(config), "--ckpt", rel(generator),
        "--ref-npz", str(args.val_npz),
        "--num-samples", str(args.num_images or exp["num_samples"]),
        "--steps", str(exp["sampling"]["steps"]),
        "--batch", str(args.batch), "--seed", str(args.seed),
        "--ig-scale", GUIDANCE_SCALE[exp["guidance"]],
        *guidance_interval(exp),
        "--device", args.device,
        "--out", str(args.results_dir / f"{exp['id']}.json"),
        *overrides,
    ]


def prepare_generation(exp, manifest, args):
    """Generation rows need a mapped generator and verified latent statistics; never substitute either."""
    unmapped = sorted({a for a in exp["required_assets"]
                       if not manifest["assets"][a]["candidates"] and a not in args.bindings})
    if exp["table"] not in GENERATOR_ARCH:
        raise ValueError(f"No generator architecture recorded for table {exp['table']}")
    arch = GENERATOR_ARCH[exp["table"]]
    generators = [a for a in exp["required_assets"] if a.startswith("generator_")]
    if len(generators) != 1:
        raise ValueError(f"{exp['id']} does not name exactly one generator asset")
    asset_id = generators[0]
    if not manifest["assets"][asset_id]["candidates"] and asset_id not in args.bindings:
        return {"id": exp["id"], "action": "skip",
                "reason": f"{asset_id} has no mapped checkpoint in the manifest",
                "unmapped_assets": unmapped}
    stats_ids = [a for a in exp["required_assets"] if a.startswith("latent_stats")]
    if len(stats_ids) != 1:
        raise ValueError(f"{exp['id']} does not name exactly one latent-statistics asset")
    stats_id = stats_ids[0]
    if args.latent_stats is not None:
        latent_stats = args.latent_stats
    elif stats_id in args.bindings:
        _, latent_stats, stats_bad = resolve_asset(stats_id, manifest, args.assets_root, args.bindings)
        if stats_bad:
            return {"id": exp["id"], "action": "skip", "reason": size_mismatch_reason(stats_id),
                    "size_mismatched": stats_bad}
        if latent_stats is None and not args.dry_run:
            return {"id": exp["id"], "action": "skip", "reason": f"No local file for {stats_id}"}
        latent_stats = latent_stats or Path(args.bindings[stats_id])
    else:
        return {"id": exp["id"], "action": "skip", "reason":
                f"Generation needs {stats_id}: the statistics of the generator's own training "
                "representation. Bind it or pass --latent-stats; freshly computed statistics require "
                "protocol verification before comparison."}
    if exp["p_dec"] == 0 and not args.decoder_output_normalization:
        return {"id": exp["id"], "action": "skip", "reason":
                "The p_dec=0 lineage records no output normalization. Pass "
                "--decoder-output-normalization explicitly once the original recipe is confirmed; "
                "do not select the convention that improves agreement."}
    config = generator_configs(arch).get(float(exp["p_dit"]))
    if config is None:
        return {"id": exp["id"], "action": "skip",
                "reason": f"No {arch} config for p_dit={exp['p_dit']}"}
    gen_checked, generator, gen_bad = resolve_asset(asset_id, manifest, args.assets_root, args.bindings)
    dec_id, dec_checked, decoder, dec_bad = resolve_checkpoint(
        exp, manifest, args.assets_root, args.bindings)
    for name, found, checked, bad in ((asset_id, generator, gen_checked, gen_bad),
                                      (dec_id, decoder, dec_checked, dec_bad)):
        if bad:
            return {"id": exp["id"], "action": "skip", "reason": size_mismatch_reason(name),
                    "size_mismatched": bad}
        if found is None and not args.dry_run:
            return {"id": exp["id"], "action": "skip",
                    "reason": f"No local file for {name}", "searched": checked}
    return {"id": exp["id"], "action": "run", "checkpoint_asset": asset_id,
            "decoder_asset": dec_id,
            "checkpoint_present": generator is not None and decoder is not None,
            "latent_reuse_group": exp["sampling"]["latent_reuse_group"],
            "latent_stats_asset": stats_id,
            "command": build_generation_command(
                exp, generator or Path(gen_checked[0]), decoder or Path(dec_checked[0]),
                config, latent_stats, args)}


def prepare(exp, manifest, configs, args):
    """Decide whether a row can run, and with which command. Never guesses an unresolved setting."""
    if exp["kind"] == "generation":
        return prepare_generation(exp, manifest, args)
    if exp["kind"] != "reconstruction":
        return {"id": exp["id"], "action": "skip", "reason": f"Unknown experiment kind {exp['kind']}"}
    if exp["fusion"] not in FEED_LAYERS:
        return {"id": exp["id"], "action": "skip", "reason": f"Unknown feed {exp['fusion']}"}
    asset_id, checked, ckpt, mismatched = resolve_checkpoint(
        exp, manifest, args.assets_root, args.bindings)
    output_normalization = None
    if asset_id.startswith("decoder_official_"):
        # Official RAEv2 decoders carry their own recorded convention; --decoder-output-normalization
        # is for the unresolved p_dec=0 lineage and must not leak into them.
        output_normalization = manifest["assets"][asset_id].get("output_normalization")
        if output_normalization is None:
            return {"id": exp["id"], "action": "skip", "reason":
                    f"{asset_id} records no output normalization in the manifest."}
        config = OFFICIAL_DECODER_CONFIG
    elif asset_id.startswith("decoder_k23_p"):
        config = configs.get(float(exp["p_dec"]))
    else:
        return {"id": exp["id"], "action": "skip", "reason": f"Unknown decoder asset {asset_id}"}
    if config is None:
        return {"id": exp["id"], "action": "skip", "reason": f"No decoder config for p_dec={exp['p_dec']}"}
    if mismatched:
        return {"id": exp["id"], "action": "skip", "reason": size_mismatch_reason(asset_id),
                "size_mismatched": mismatched}
    if ckpt is None and not args.dry_run:
        return {"id": exp["id"], "action": "skip", "reason": f"No local file for {asset_id}",
                "searched": checked}
    if exp["p_dec"] == 0 and not (output_normalization or args.decoder_output_normalization):
        return {"id": exp["id"], "action": "skip", "reason":
                "The p_dec=0 lineage records no output normalization. Pass "
                "--decoder-output-normalization explicitly once the original recipe is confirmed; "
                "do not select the convention that improves agreement."}
    # A dry run shows the command for a checkpoint that is not downloaded yet; a real run does not.
    planned = ckpt or Path(checked[0])
    return {"id": exp["id"], "action": "run", "checkpoint_asset": asset_id,
            "checkpoint_present": ckpt is not None,
            "command": build_command(exp, planned, config, args, output_normalization)}


def group_skips(skipped):
    grouped = {}
    for plan in skipped:
        grouped.setdefault(plan["reason"], []).append(plan)
    return grouped


def execute(plan, args):
    command = plan["command"]
    print(f"\n=== {plan['id']} ===\n{shlex.join(command)}", flush=True)
    completed = subprocess.run(command, cwd=ROOT)
    if completed.returncode != 0:
        return {**plan, "status": "failed", "returncode": completed.returncode}
    result = args.results_dir / f"{plan['id']}.json"
    comparison = args.results_dir / f"{plan['id']}-comparison.json"
    compare = [args.python, "scripts/compare_result.py", "--experiment", plan["id"],
               "--result", str(result), "--out", str(comparison)]
    print(shlex.join(compare), flush=True)
    compared = subprocess.run(compare, cwd=ROOT)
    return {**plan, "status": "evaluated" if compared.returncode == 0 else "compare_failed",
            "result": str(result), "comparison": str(comparison) if compared.returncode == 0 else None}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=ROOT / "reproduction" / "experiments.json")
    parser.add_argument("--table", action="append",
                        help="Paper table number, matrix key, LaTeX label, appendix, or all")
    parser.add_argument("--id", action="append", help="Experiment ID glob; repeat to select several")
    parser.add_argument("--priority", choices=["P0", "P1"])
    parser.add_argument("--p-dec", type=float)
    parser.add_argument("--p-dit", type=float)
    parser.add_argument("--guidance", choices=["none", "1.78"])
    parser.add_argument("--assets-root", type=Path, default=ROOT / "checkpoints")
    parser.add_argument("--bindings", type=Path, help="JSON object mapping asset IDs to local files")
    parser.add_argument("--val-npz", type=Path, default=Path("data_eval/imagenet-256-val.npz"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--num-images", type=int,
                        help="Override the row's 50000-image protocol for a smoke run")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--decoder-output-normalization", choices=["encoder", "raw"])
    parser.add_argument("--latent-stats", type=Path,
                        help="Latent statistics for generation rows; must match the generator's training representation")
    parser.add_argument("--work-dir", type=Path, help="Scratch directory for reconstruction FID memmaps")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true", help="Print the commands without running anything")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--summary", type=Path, help="Write the run summary JSON to this path")
    args = parser.parse_args(argv)
    try:
        manifest = read_json(args.manifest)
        selected = select_experiments(manifest, SimpleNamespace(**{k: getattr(args, k) for k in SELECTORS}))
        if not selected:
            parser.error("No experiments matched; use plan_reproduction.py --list to inspect IDs")
        bindings = read_json(args.bindings) if args.bindings else {}
        if not isinstance(bindings, dict):
            raise ValueError("Bindings must be a JSON object mapping asset IDs to file paths")
        args.bindings = bindings
        args.assets_root = args.assets_root.expanduser().resolve()
        args.results_dir.mkdir(parents=True, exist_ok=True)
        plans = [prepare(exp, manifest, decoder_configs(), args) for exp in selected]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    runnable = [p for p in plans if p["action"] == "run"]
    skipped = [p for p in plans if p["action"] == "skip"]
    for reason, plans_for_reason in group_skips(skipped).items():
        ids = [p["id"] for p in plans_for_reason]
        shown = ", ".join(ids[:4]) + (f", ... (+{len(ids) - 4} more)" if len(ids) > 4 else "")
        print(f"skip {len(ids)} row(s): {reason}\n  rows: {shown}", file=sys.stderr)
        unmapped = sorted({a for p in plans_for_reason for a in p.get("unmapped_assets", [])})
        if unmapped:
            print(f"  unmapped assets: {', '.join(unmapped)}", file=sys.stderr)
    if not args.dry_run and not runnable:
        print(f"Error: no selected row is runnable ({len(skipped)} skipped)", file=sys.stderr)
        return 1
    if args.val_npz and not args.dry_run and not Path(args.val_npz).is_file():
        print(f"Error: reference images not found at {args.val_npz}", file=sys.stderr)
        return 1
    records = []
    for plan in runnable:
        if args.dry_run:
            print(shlex.join(plan["command"]))
            records.append({**plan, "status": "not_run"})
            continue
        record = execute(plan, args)
        records.append(record)
        if record["status"] != "evaluated" and args.stop_on_error:
            print(f"Error: stopping after {record['id']}", file=sys.stderr)
            break
    summary = {
        "schema_version": 1,
        "mode": "dry_run" if args.dry_run else "executed",
        "selected": len(plans), "runnable": len(runnable), "skipped": len(skipped),
        "evaluated": sum(1 for r in records if r["status"] == "evaluated"),
        "failed": sum(1 for r in records if r["status"] in ("failed", "compare_failed")),
        "protocol_verified": False,
        "note": "Execution does not verify checkpoint mapping, metric backend equivalence or any tolerance.",
        "runs": records, "skips": skipped,
    }
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("runs", "skips")}, indent=2))
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
