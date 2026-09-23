#!/usr/bin/env python3
"""Build a deterministic, source-only anonymous review archive."""

import argparse
import hashlib
from pathlib import Path
import re
import zipfile


ROOT_FILES = {"README.md", "LICENSE", "THIRD_PARTY_NOTICE.md", "pyproject.toml", ".gitignore"}
ROOT_DIRS = {"src", "configs", "scripts", "tests", "reproduction", "docs"}
SUFFIXES = {".py", ".yaml", ".yml", ".json", ".md", ".txt", ".toml", ".sh"}
PRIVATE = re.compile(
    r"hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|"
    r"/Users/[^/\s]+/|/home/[^/\s]+/|/sensei-fs[^\s\"']*|"
    r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
)


def anonymize(content):
    # Omit only this submission's repository identity. Keep upstream attribution.
    content = content.replace("Hongyang-Du", "anonymous")
    content = content.replace("hongyangd", "anonymous")
    content = content.replace("davidliuk", "anonymous")
    return content


def build(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    anonymous_readme = root / "docs" / "README_ANONYMOUS.md"
    if not anonymous_readme.is_file():
        raise ValueError("Missing docs/README_ANONYMOUS.md")
    members = {}
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if not path.is_file() or path.is_symlink():
            continue
        if any(p.startswith(".") or p == "__pycache__" or p.endswith(".egg-info") for p in rel.parts):
            if rel.as_posix() != ".gitignore":
                continue
        if len(rel.parts) == 1:
            if rel.name not in ROOT_FILES and not (rel.name.startswith("requirements") and rel.suffix == ".txt"):
                continue
        elif rel.parts[0] not in ROOT_DIRS or path.suffix not in SUFFIXES:
            continue
        if rel.as_posix() in {"docs/README_ANONYMOUS.md", "docs/release_status.md", "reproduction/checkpoint_sources.json"}:
            continue
        source = anonymous_readme if rel.as_posix() == "README.md" else path
        content = anonymize(source.read_text(encoding="utf-8"))
        # The scanner source contains literal signatures by design.
        if rel.as_posix() != "scripts/build_supplement.py":
            match = PRIVATE.search(content)
            if match:
                raise ValueError(f"Identity or private path in {rel}; inspect before release")
        members["FuseReg/" + rel.as_posix()] = content.encode("utf-8")
    if "FuseReg/README.md" not in members:
        raise ValueError("Missing source README.md")
    manifest = "".join(f"{hashlib.sha256(data).hexdigest()}  {name}\n" for name, data in members.items())
    members["FuseReg/MANIFEST.sha256"] = manifest.encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, data in sorted(members.items()):
            info = zipfile.ZipInfo(name, date_time=(2020, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data)
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise ValueError("Archive integrity check failed")
    return {"files": len(members), "sha256": hashlib.sha256(output.read_bytes()).hexdigest(), "bytes": output.stat().st_size}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=Path("dist/fusereg-supplement.zip"))
    args = parser.parse_args()
    import json
    print(json.dumps(build(args.root, args.output), indent=2))


if __name__ == "__main__":
    main()
