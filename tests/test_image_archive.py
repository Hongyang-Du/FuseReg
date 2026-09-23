"""Low-cost validation of reconstruction evaluation archives."""

from pathlib import Path
import struct

import numpy as np
import pytest

from utils.image_archive import inspect_image_archive


@pytest.mark.parametrize("suffix", [".npy", ".npz"])
def test_accepts_uint8_nhwc_without_loading_models(tmp_path, suffix):
    path = tmp_path / f"images{suffix}"
    images = np.zeros((3, 8, 8, 3), dtype=np.uint8)
    (np.save if suffix == ".npy" else np.savez)(path, images)
    result = inspect_image_archive(path, resolution=8, min_images=3)
    assert result["shape"] == [3, 8, 8, 3]
    assert result["dtype"] == "uint8"
    assert result["scope"].startswith("Header validation only")


def test_rejects_wrong_dtype_shape_and_short_archive(tmp_path):
    wrong_dtype = tmp_path / "float.npy"
    np.save(wrong_dtype, np.zeros((3, 8, 8, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="expected uint8"):
        inspect_image_archive(wrong_dtype, resolution=8)

    wrong_shape = tmp_path / "chw.npy"
    np.save(wrong_shape, np.zeros((3, 3, 8, 8), dtype=np.uint8))
    with pytest.raises(ValueError, match="expected uint8"):
        inspect_image_archive(wrong_shape, resolution=8)

    too_short = tmp_path / "short.npz"
    np.savez(too_short, np.zeros((2, 8, 8, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="at least 3"):
        inspect_image_archive(too_short, resolution=8, min_images=3)


def test_missing_or_empty_archive_fails_cleanly(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        inspect_image_archive(tmp_path / "missing.npz")
    empty = tmp_path / "empty.npz"
    np.savez(empty)
    with pytest.raises(ValueError, match="no NPY arrays"):
        inspect_image_archive(empty)


@pytest.mark.parametrize(
    ("header", "message"),
    [
        ({"descr": "|u1", "fortran_order": "False", "shape": (3, 8, 8, 3)}, "valid bool"),
        ({"descr": "|u1", "fortran_order": False, "shape": (3, 8, 8, 3), "extra": 1}, "malformed"),
    ],
)
def test_rejects_headers_numpy_would_reject(tmp_path, header, message):
    path = Path(tmp_path) / "invalid.npy"
    encoded = repr(header).encode("latin1")
    padding = 16 - ((10 + len(encoded) + 1) % 16)
    encoded += b" " * padding + b"\n"
    path.write_bytes(b"\x93NUMPY\x01\x00" + struct.pack("<H", len(encoded)) + encoded)
    with pytest.raises(ValueError, match=message):
        inspect_image_archive(path, resolution=8)
    with pytest.raises(ValueError):
        np.load(path, allow_pickle=False)
