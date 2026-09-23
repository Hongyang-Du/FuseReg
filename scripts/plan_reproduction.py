#!/usr/bin/env python3
"""Select paper experiments and check local asset presence without running models."""

import argparse
from collections import Counter
from fnmatch import fnmatchcase
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def expand_table_selectors(manifest, tables):
    """Accept a paper table number wherever a matrix key or LaTeX label is accepted."""
    numbering = manifest.get("paper_tables", {}).get("tables", {})
    expanded = []
    for table in tables or []:
        number = table[len("table:"):] if table.startswith("table:") else table
        if number in numbering:
            expanded.extend(numbering[number]["labels"])
        elif table.startswith("table:") or table.isdigit():
            known = ", ".join(sorted(numbering)) or "none recorded"
            raise ValueError(f"No paper table {number} in paper_tables; known tables: {known}")
        else:
            expanded.append(table)
    return expanded


def select_experiments(manifest, args):
    tables = expand_table_selectors(manifest, args.table)
    selected = []
    for exp in manifest["experiments"]:
        if tables and not any(
            table == exp["table"]
            or table in exp["table_labels"]
            or (table == "appendix" and exp["table"].startswith("appendix_"))
            or table == "all"
            for table in tables
        ):
            continue
        if args.id and not any(fnmatchcase(exp["id"], pattern) for pattern in args.id):
            continue
        if args.priority and exp["priority"] != args.priority:
            continue
        if args.p_dec is not None and exp["p_dec"] != args.p_dec:
            continue
        if args.p_dit is not None and exp["p_dit"] != args.p_dit:
            continue
        if args.guidance is not None:
            wanted = None if args.guidance == "none" else 1.78
            if exp["kind"] != "generation" or exp["guidance"] != wanted:
                continue
        selected.append(exp)
    return selected


def inspect_asset(asset_id, asset, assets_root, bindings):
    """Existence is deliberately separate from checkpoint/protocol verification."""
    override = bindings.get(asset_id)
    candidates = []
    if override is not None:
        if not isinstance(override, str) or not override:
            raise ValueError(f"Binding for {asset_id} must be a nonempty path string")
        candidates = [{"path": override, "evidence": "explicit local binding"}]
    else:
        candidates = asset["candidates"]
    checked = []
    for candidate in candidates:
        path = Path(candidate["path"]).expanduser()
        if not path.is_absolute():
            path = assets_root / path
        path = path.resolve()
        is_file = path.is_file()
        size = path.stat().st_size if is_file else None
        expected = candidate.get("bytes")
        checked.append({
            "path": str(path),
            "exists": is_file,
            "bytes": size,
            "inventory_bytes": expected,
            "expected_sha256": candidate.get("sha256"),
            "hash_checked_by_planner": False,
            "size_matches_inventory": (size == expected) if is_file and expected else None,
        })
    present = [candidate for candidate in checked if candidate["exists"]]
    size_mismatches = [candidate for candidate in present if candidate["size_matches_inventory"] is False]
    return {
        "id": asset_id,
        "kind": asset["kind"],
        "description": asset["description"],
        "local_status": "missing" if not present else "size_mismatch" if len(size_mismatches) == len(present) else "present_unverified",
        "mapping_status": asset["mapping_status"],
        "checked_paths": checked,
        "paper_provenance": asset.get("paper_provenance"),
        "identity_verification": asset.get("identity_verification"),
    }


def make_plan(manifest, selected, assets_root, bindings):
    unknown = set(bindings) - set(manifest["assets"])
    if unknown:
        raise ValueError("Unknown asset binding IDs: " + ", ".join(sorted(unknown)))
    required = sorted({asset_id for exp in selected for asset_id in exp["required_assets"]})
    assets = {
        asset_id: inspect_asset(asset_id, manifest["assets"][asset_id], assets_root, bindings)
        for asset_id in required
    }
    experiments = []
    for exp in selected:
        missing = [asset_id for asset_id in exp["required_assets"] if assets[asset_id]["local_status"] == "missing"]
        mismatches = [asset_id for asset_id in exp["required_assets"] if assets[asset_id]["local_status"] == "size_mismatch"]
        experiments.append({**exp, "preflight": {
            "missing_assets": missing,
            "size_mismatched_assets": mismatches,
            "protocol_verified": False,
            "runnable": False,
            "reason": "Planning only: file presence does not verify checkpoint mapping, normalization, evaluator protocol or hardware.",
        }})
    return {
        "schema_version": 1,
        "mode": "plan_only",
        "summary": {
            "experiments": len(experiments),
            "by_table": dict(sorted(Counter(exp["table"] for exp in experiments).items())),
            "assets": len(assets),
            "by_asset_local_status": dict(sorted(Counter(asset["local_status"] for asset in assets.values()).items())),
            "latent_reuse_groups": len({exp["sampling"]["latent_reuse_group"] for exp in experiments if "sampling" in exp}),
        },
        "protocol_checks": manifest["protocol_checks"],
        "assets": assets,
        "experiments": experiments,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "reproduction" / "experiments.json")
    parser.add_argument("--table", action="append",
                        help="Paper table number, matrix key, LaTeX label, appendix, or all; repeat to select several")
    parser.add_argument("--id", action="append", help="Experiment ID glob; repeat to select several")
    parser.add_argument("--priority", choices=["P0", "P1"])
    parser.add_argument("--p-dec", type=float)
    parser.add_argument("--p-dit", type=float)
    parser.add_argument("--guidance", choices=["none", "1.78"])
    parser.add_argument("--assets-root", type=Path, default=ROOT / "checkpoints")
    parser.add_argument("--bindings", type=Path, help="JSON object mapping asset IDs to local files; relative paths use assets-root")
    parser.add_argument("--list", action="store_true", help="Print selected IDs and paper metrics without writing a plan")
    parser.add_argument("--output", type=Path, help="Write JSON plan to this path; otherwise print JSON to stdout")
    args = parser.parse_args(argv)
    if args.list and args.output:
        parser.error("--list and --output are mutually exclusive")
    try:
        manifest = read_json(args.manifest)
        selected = select_experiments(manifest, args)
        if not selected:
            parser.error("No experiments matched; use --list without filters to inspect IDs")
        if args.list:
            for exp in selected:
                print(f"{exp['id']}\t{exp['table']}\t{exp['status']}\t{json.dumps(exp['paper_reported'], sort_keys=True)}")
            return 0
        bindings = read_json(args.bindings) if args.bindings else {}
        if not isinstance(bindings, dict):
            raise ValueError("Bindings must be a JSON object mapping asset IDs to file paths")
        plan = make_plan(manifest, selected, args.assets_root.expanduser().resolve(), bindings)
        serialized = json.dumps(plan, indent=2) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(serialized, encoding="utf-8")
            print(json.dumps(plan["summary"], indent=2))
        else:
            print(serialized, end="")
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
