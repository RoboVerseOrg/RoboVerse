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


def per_pixel_stack_stats(stack: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if stack.ndim != 4:
        raise ValueError(f"per_pixel_stack_stats expects (N, H, W, C); got {stack.shape}")
    if stack.shape[0] < 2:
        raise ValueError(f"per_pixel_stack_stats requires N >= 2; got N={stack.shape[0]}")
    mean = stack.mean(axis=0).astype(np.float32, copy=False)
    std = stack.std(axis=0, ddof=0).astype(np.float32, copy=False)
    return mean, std


def crop(image: np.ndarray, box: list[int]) -> np.ndarray:
    x0, y0, x1, y1 = (int(v) for v in box)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"empty crop box {box}")
    return image[y0:y1, x0:x1, :]


def group_means(per_roi: dict[str, float], grouping: dict[str, list[str]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for group, names in grouping.items():
        values = [per_roi[name] for name in names if name in per_roi]
        if not values:
            continue
        out[group] = float(sum(values) / len(values))
    return out


class MetricRunner:
    """Numerical + perceptual metrics. LPIPS and FLIP can be disabled for cheap tests."""

    def __init__(self, *, enable_lpips: bool, enable_flip: bool):
        self._enable_lpips = enable_lpips
        self._enable_flip = enable_flip
        self._lpips_model = None
        self._flip_module = None
        self._device = None

    @classmethod
    def numerical_only(cls) -> MetricRunner:
        return cls(enable_lpips=False, enable_flip=False)

    @classmethod
    def full(cls) -> MetricRunner:
        return cls(enable_lpips=True, enable_flip=True)

    @staticmethod
    def rmse(a: np.ndarray, b: np.ndarray) -> float:
        diff = a - b
        return float(np.sqrt(np.mean(diff * diff)))

    @staticmethod
    def mae(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.mean(np.abs(a - b)))

    @staticmethod
    def psnr(a: np.ndarray, b: np.ndarray) -> float:
        a_c = np.clip(a, 0.0, 1.0)
        b_c = np.clip(b, 0.0, 1.0)
        mse = float(np.mean((a_c - b_c) ** 2))
        if mse <= 1e-20:
            return 99.0
        return float(10.0 * np.log10(1.0 / mse))

    def _ensure_lpips(self):
        if self._lpips_model is None and self._enable_lpips:
            try:
                import lpips  # type: ignore
                import torch

                self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self._lpips_model = lpips.LPIPS(net="alex", verbose=False).to(self._device).eval()
            except Exception:
                self._enable_lpips = False
                self._lpips_model = None
        return self._lpips_model

    def lpips(self, a: np.ndarray, b: np.ndarray) -> float | None:
        if not self._enable_lpips:
            return None
        import torch

        model = self._ensure_lpips()
        if model is None:
            return None
        a_clip = np.clip(a, 0.0, 1.0).astype(np.float32)
        b_clip = np.clip(b, 0.0, 1.0).astype(np.float32)
        a_t = torch.from_numpy(a_clip).permute(2, 0, 1).unsqueeze(0).to(self._device)
        b_t = torch.from_numpy(b_clip).permute(2, 0, 1).unsqueeze(0).to(self._device)
        a_t = a_t * 2.0 - 1.0
        b_t = b_t * 2.0 - 1.0
        with torch.no_grad():
            d = model(a_t, b_t)
        return float(d.detach().cpu().item())

    def _ensure_flip(self):
        if self._flip_module is None and self._enable_flip:
            try:
                import flip_evaluator  # type: ignore

                self._flip_module = flip_evaluator
            except Exception:
                self._flip_module = None
        return self._flip_module

    def flip(self, a: np.ndarray, b: np.ndarray) -> float | None:
        if not self._enable_flip:
            return None
        module = self._ensure_flip()
        if module is None:
            return None
        _, mean_error, _ = module.evaluate(
            a.astype(np.float32), b.astype(np.float32), "HDR", applyMagma=False, computeMeanError=True
        )
        return float(mean_error)
