#!/usr/bin/env python3
"""Validate an evaluation NPY/NPZ header before loading models or image data."""

import argparse
import json
from pathlib import Path
import struct
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from utils.image_archive import inspect_image_archive


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="RGB uint8 evaluation archive (.npy or .npz)")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--min-images", type=int, default=1)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(inspect_image_archive(args.archive, args.resolution, args.min_images), indent=2))
        return 0
    except (OSError, ValueError, zipfile.BadZipFile, struct.error, SyntaxError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
