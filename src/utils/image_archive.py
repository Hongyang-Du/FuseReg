"""Header-only inspection for reconstruction evaluation image archives."""

import ast
from contextlib import contextmanager
from pathlib import Path
import struct
import zipfile

import numpy as np


def read_npy_header(stream):
    """Read an NPY header without materializing the array payload."""
    if stream.read(6) != b"\x93NUMPY":
        raise ValueError("not a valid NPY stream")
    version = tuple(stream.read(2))
    if version == (1, 0):
        header_size = struct.unpack("<H", stream.read(2))[0]
        encoding = "latin1"
    elif version in {(2, 0), (3, 0)}:
        header_size = struct.unpack("<I", stream.read(4))[0]
        encoding = "utf-8" if version == (3, 0) else "latin1"
    else:
        raise ValueError(f"unsupported NPY format version {version}")
    if header_size > 1_000_000:
        raise ValueError(f"NPY header is unexpectedly large: {header_size} bytes")
    header = ast.literal_eval(stream.read(header_size).decode(encoding).strip())
    required_keys = {"descr", "fortran_order", "shape"}
    if not isinstance(header, dict) or set(header) != required_keys:
        raise ValueError("malformed NPY header")
    if type(header["fortran_order"]) is not bool:
        raise ValueError("fortran_order is not a valid bool")
    shape = header["shape"]
    if not isinstance(shape, tuple) or any(not isinstance(value, int) or value < 0 for value in shape):
        raise ValueError(f"invalid array shape {shape!r}")
    return shape, np.dtype(header["descr"]), header["fortran_order"], version


@contextmanager
def first_array_stream(path):
    """Yield the first array stream and member name, matching evaluator semantics."""
    path = Path(path)
    if path.suffix.lower() == ".npy":
        with path.open("rb") as stream:
            yield stream, path.name
        return
    if path.suffix.lower() != ".npz":
        raise ValueError("archive must end in .npy or .npz")
    with zipfile.ZipFile(path) as archive:
        members = [name for name in archive.namelist() if name.endswith(".npy") and not name.endswith("/")]
        if not members:
            raise ValueError("NPZ contains no NPY arrays")
        with archive.open(members[0]) as stream:
            yield stream, members[0]


def inspect_image_archive(path, resolution=256, min_images=1):
    """Validate dtype, NHWC shape, and count using only the first array header."""
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"archive does not exist: {path}")
    if resolution < 1 or min_images < 1:
        raise ValueError("resolution and min_images must be positive")
    with first_array_stream(path) as (stream, member):
        shape, dtype, fortran_order, version = read_npy_header(stream)
    expected = (resolution, resolution, 3)
    if dtype != np.dtype(np.uint8) or len(shape) != 4 or tuple(shape[1:]) != expected:
        raise ValueError(f"expected uint8 [N,{resolution},{resolution},3]; got {shape} {dtype}")
    if shape[0] < min_images:
        raise ValueError(f"requested at least {min_images} images; archive contains {shape[0]}")
    return {
        "status": "valid_header",
        "archive": str(path),
        "member": member,
        "shape": list(shape),
        "dtype": str(dtype),
        "fortran_order": fortran_order,
        "npy_version": ".".join(map(str, version)),
        "minimum_images": min_images,
        "scope": "Header validation only; image values and preprocessing provenance were not verified.",
    }
