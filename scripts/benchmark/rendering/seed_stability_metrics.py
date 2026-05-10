"""Seed-variance metric computation. Pure functions where possible; render_std_map writes a PNG."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import numpy as np

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
import cv2


def _load_one(path: Path) -> np.ndarray:
    if not Path(path).exists():
        raise FileNotFoundError(f"EXR not found: {path}")
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise RuntimeError(f"cv2 failed to read EXR: {path}")
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    if arr.ndim != 3 or arr.shape[-1] not in (3, 4):
        raise ValueError(f"unexpected EXR shape {arr.shape} for {path}")
    rgb = arr[..., :3][..., ::-1]
    return np.ascontiguousarray(rgb, dtype=np.float32)


def load_exr_stack(paths: Sequence[Path]) -> np.ndarray:
    if not paths:
        raise ValueError("load_exr_stack: empty path list")
    images: list[np.ndarray] = []
    reference_shape: tuple[int, ...] | None = None
    for path in paths:
        img = _load_one(Path(path))
        if reference_shape is None:
            reference_shape = img.shape
        elif img.shape != reference_shape:
            raise ValueError(f"shape mismatch: expected {reference_shape}, got {img.shape} from {path}")
        images.append(img)
    return np.stack(images, axis=0).astype(np.float32, copy=False)
