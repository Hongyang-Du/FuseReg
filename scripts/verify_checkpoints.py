#!/usr/bin/env python3
"""Verify locally downloaded inference assets against the release manifest."""
import argparse
import hashlib
import json
from pathlib import Path


def verify(manifest, root, selected=None):
    root = Path(root).resolve()
    results = []
    entries = manifest["files"]
    if selected:
        entries = [entry for entry in entries if entry["path"] in selected]
        unknown = set(selected) - {entry["path"] for entry in entries}
        if unknown:
            raise ValueError(f"Unknown checkpoint paths: {sorted(unknown)}")
    for entry in entries:
        path = (root / entry["path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Manifest path escapes checkpoint root")
        result = {"path": entry["path"]}
        if not path.is_file():
            result["status"] = "missing"
        elif path.stat().st_size != entry["size_bytes"]:
            result["status"] = "size_mismatch"
        else:
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            result["sha256"] = digest.hexdigest()
            result["status"] = "verified" if result["sha256"] == entry["sha256"] else "hash_mismatch"
        results.append(result)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).resolve().parents[1] / "reproduction/checkpoints.json")
    parser.add_argument("--root", type=Path, default=Path("checkpoints"))
    parser.add_argument("--file", action="append", help="Verify only this repository-relative checkpoint path; repeatable")
    args = parser.parse_args()
    results = verify(json.loads(args.manifest.read_text()), args.root, args.file)
    print(json.dumps(results, indent=2))
    raise SystemExit(0 if results and all(r["status"] == "verified" for r in results) else 1)


if __name__ == "__main__":
    main()
