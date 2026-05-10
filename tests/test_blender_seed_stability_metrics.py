from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from scripts.benchmark.rendering import seed_stability_metrics as ssm


def _write_constant_exr(path: Path, value: float, shape: tuple[int, int, int] = (8, 8, 3)) -> None:
    arr = np.full(shape, value, dtype=np.float32)
    bgr = arr[..., ::-1]  # cv2 expects BGR
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), bgr)
    assert ok, f"cv2.imwrite failed for {path}"


def test_load_exr_stack_returns_float32_rgb_stack(tmp_path) -> None:
    paths = []
    for index, value in enumerate([0.1, 0.2, 0.3]):
        p = tmp_path / f"img_{index}.exr"
        _write_constant_exr(p, value)
        paths.append(p)
    stack = ssm.load_exr_stack(paths)
    assert stack.dtype == np.float32
    assert stack.shape == (3, 8, 8, 3)
    assert np.allclose(stack[0], 0.1, atol=1e-3)
    assert np.allclose(stack[2], 0.3, atol=1e-3)


def test_load_exr_stack_preserves_rgb_channel_order(tmp_path) -> None:
    arr = np.zeros((4, 4, 3), dtype=np.float32)
    arr[..., 0] = 0.9  # R
    arr[..., 1] = 0.5  # G
    arr[..., 2] = 0.1  # B
    path = tmp_path / "rgb.exr"
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = arr[..., ::-1]
    assert cv2.imwrite(str(path), bgr)
    stack = ssm.load_exr_stack([path])
    assert np.allclose(stack[0, ..., 0], 0.9, atol=1e-3)
    assert np.allclose(stack[0, ..., 1], 0.5, atol=1e-3)
    assert np.allclose(stack[0, ..., 2], 0.1, atol=1e-3)


def test_load_exr_stack_raises_on_shape_mismatch(tmp_path) -> None:
    a = tmp_path / "a.exr"
    b = tmp_path / "b.exr"
    _write_constant_exr(a, 0.1, shape=(8, 8, 3))
    _write_constant_exr(b, 0.1, shape=(16, 16, 3))
    with pytest.raises(ValueError) as excinfo:
        ssm.load_exr_stack([a, b])
    assert "shape mismatch" in str(excinfo.value).lower()
    assert "b.exr" in str(excinfo.value)


def test_load_exr_stack_raises_on_missing_file(tmp_path) -> None:
    a = tmp_path / "a.exr"
    _write_constant_exr(a, 0.1)
    missing = tmp_path / "missing.exr"
    with pytest.raises(FileNotFoundError) as excinfo:
        ssm.load_exr_stack([a, missing])
    assert "missing.exr" in str(excinfo.value)
