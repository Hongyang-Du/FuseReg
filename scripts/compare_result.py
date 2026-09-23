#!/usr/bin/env python3
"""Compare a measured evaluator JSON with one paper row, without certifying equivalence."""
import argparse
import json
import math
from pathlib import Path


def compare(experiment, measured):
    count = measured.get("num_images", measured.get("num_samples"))
    if not isinstance(count, int) or count <= 0:
        raise ValueError("Measured result must include a positive num_images or num_samples")
    aliases = {"rfid": "rfid_tf", "gfid": "fid"}
    rows = []
    for metric, target in experiment["paper_reported"].items():
        key = metric if metric in measured else aliases.get(metric, metric)
        value = measured.get(key)
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
            raise ValueError(f"Measured result is missing a finite {metric} ({key}) value")
        rows.append({"metric": metric, "measured_key": key, "paper": target,
                     "measured": value, "absolute_difference": abs(value - target)})
    return {
        "experiment_id": experiment["id"],
        "status": "measured_protocol_unverified" if count == experiment["num_samples"] else "smoke_only",
        "actual_samples": count, "paper_samples": experiment["num_samples"],
        "protocol_verified": False,
        "note": "This comparison does not establish checkpoint association, backend equivalence or an acceptance tolerance.",
        "metrics": rows,
        "measurement": measured,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).resolve().parents[1] / "reproduction/experiments.json")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    matches = [e for e in manifest["experiments"] if e["id"] == args.experiment]
    if len(matches) != 1:
        parser.error("--experiment must match exactly one experiment ID")
    result = compare(matches[0], json.loads(args.result.read_text()))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"{result['experiment_id']}: {result['status']} ({result['actual_samples']} samples)")
    print("\n| Metric | Paper target | Measured | Absolute difference |\n|---|---:|---:|---:|")
    for row in result["metrics"]:
        print(f"| {row['metric']} | {row['paper']:g} | {row['measured']:g} | {row['absolute_difference']:.6g} |")


if __name__ == "__main__":
    main()
