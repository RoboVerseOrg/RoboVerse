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


def _summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "n_pairs": 0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "n_pairs": int(arr.size),
    }


def _metrics_for_pair(runner: MetricRunner, a: np.ndarray, b: np.ndarray) -> dict[str, float | None]:
    return {
        "rmse": runner.rmse(a, b),
        "mae": runner.mae(a, b),
        "psnr": runner.psnr(a, b),
        "lpips": runner.lpips(a, b),
        "flip": runner.flip(a, b),
    }


_METRIC_NAMES: tuple[str, ...] = ("rmse", "mae", "psnr", "lpips", "flip")


def pairwise_metrics(
    stack: np.ndarray,
    *,
    runner: MetricRunner,
    roi_boxes: dict[str, list[int]] | None,
    roi_grouping: dict[str, list[str]] | None,
) -> dict[str, dict[str, dict[str, float]]]:
    if stack.ndim != 4:
        raise ValueError(f"pairwise_metrics expects (N, H, W, C); got {stack.shape}")
    n = stack.shape[0]
    if n < 2:
        raise ValueError(f"pairwise_metrics requires N >= 2; got N={n}")

    region_pairs: dict[str, dict[str, list[float]]] = {
        "global": {m: [] for m in _METRIC_NAMES},
    }
    if roi_boxes and roi_grouping:
        for group in roi_grouping:
            region_pairs[group] = {m: [] for m in _METRIC_NAMES}

    for i in range(n):
        for j in range(i + 1, n):
            a, b = stack[i], stack[j]
            global_metrics = _metrics_for_pair(runner, a, b)
            for m in _METRIC_NAMES:
                v = global_metrics[m]
                if v is not None:
                    region_pairs["global"][m].append(v)
            if roi_boxes and roi_grouping:
                per_roi: dict[str, dict[str, float | None]] = {
                    name: _metrics_for_pair(runner, crop(a, box), crop(b, box)) for name, box in roi_boxes.items()
                }
                for m in _METRIC_NAMES:
                    per_roi_scalar = {n_: v[m] for n_, v in per_roi.items() if v[m] is not None}
                    if not per_roi_scalar:
                        continue
                    grouped = group_means(per_roi_scalar, roi_grouping)
                    for group, value in grouped.items():
                        region_pairs[group][m].append(value)

    return {
        region: {m: _summarize(values) for m, values in metrics.items()} for region, metrics in region_pairs.items()
    }


def vs_reference_metrics(
    stack: np.ndarray,
    reference: np.ndarray,
    *,
    runner: MetricRunner,
    roi_boxes: dict[str, list[int]] | None,
    roi_grouping: dict[str, list[str]] | None,
) -> list[dict[str, float | int | None]]:
    if stack.ndim != 4 or reference.ndim != 3:
        raise ValueError(f"shape error: stack={stack.shape}, reference={reference.shape}")
    if stack.shape[1:] != reference.shape:
        raise ValueError(f"shape mismatch: stack frame {stack.shape[1:]} != reference {reference.shape}")
    out: list[dict[str, float | int | None]] = []
    for index in range(stack.shape[0]):
        seed_img = stack[index]
        global_metrics = _metrics_for_pair(runner, seed_img, reference)
        row: dict[str, float | int | None] = {
            "seed_index": index,
            **{m: global_metrics[m] for m in _METRIC_NAMES},
        }
        if roi_boxes and roi_grouping:
            per_roi_rmse: dict[str, float] = {}
            per_roi_psnr: dict[str, float] = {}
            for name, box in roi_boxes.items():
                a = crop(seed_img, box)
                b = crop(reference, box)
                per_roi_rmse[name] = runner.rmse(a, b)
                per_roi_psnr[name] = runner.psnr(a, b)
            grouped_rmse = group_means(per_roi_rmse, roi_grouping)
            grouped_psnr = group_means(per_roi_psnr, roi_grouping)
            for group, value in grouped_rmse.items():
                row[f"{group}_rmse"] = value
                row[f"{group}_psnr"] = grouped_psnr.get(group, 0.0)
        out.append(row)
    return out


def render_std_map(std_image: np.ndarray, out_path: Path, *, color_scale: float | None) -> float:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if std_image.ndim != 3 or std_image.shape[-1] != 3:
        raise ValueError(f"render_std_map expects (H, W, 3); got {std_image.shape}")
    intensity = std_image.mean(axis=-1)
    scale = color_scale if color_scale is not None else float(np.max(intensity))
    if scale <= 0:
        scale = 1.0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=120)
    im = ax.imshow(intensity, cmap="magma", vmin=0.0, vmax=scale)
    ax.set_axis_off()
    fig.colorbar(im, ax=ax, label="linear-EXR std")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return scale
