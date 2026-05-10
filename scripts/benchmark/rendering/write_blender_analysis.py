#!/usr/bin/env python3
"""Write interpretation-focused Blender benchmark analysis and figures."""

from __future__ import annotations

import argparse
import csv
import math
import os
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from scripts.benchmark.rendering import seed_stability_metrics as _seed_metrics
from scripts.benchmark.rendering import write_seed_stability_section as _seed_section

ROOT_DEFAULT = Path("outputs/rendering_benchmark/blender_2026-05-09_glass_only")
SCENES = ("complex_glass_cube_reach",)
SCENE_TITLES = {
    "complex_glass_cube_reach": "Complex glass cube reach",
}
CONTACT_SHEET_LABEL_HEIGHT = 88
CONTACT_SHEET_FONT_SIZE = 32
ROI_GROUPS = {
    "complex_glass_cube_reach": {
        "glass": {
            "left_display_case": [45, 325, 430, 615],
            "center_bottle": [620, 225, 720, 505],
            "front_hurricane_glass": [600, 420, 775, 695],
            "rear_tumbler": [880, 200, 985, 295],
        },
        "non_glass": {
            "robot_left_arm": [0, 40, 470, 430],
            "robot_right_arm": [300, 50, 720, 230],
            "target_cube": [545, 280, 595, 330],
            "table_plain_region": [770, 320, 1110, 520],
        },
    },
}
ROI_BOXES = {
    scene: {name: box for group in groups.values() for name, box in group.items()}
    for scene, groups in ROI_GROUPS.items()
}


@dataclass(frozen=True)
class QualityMetric:
    key: str
    label: str
    ylabel: str
    log_y: bool = False


QUALITY_METRICS = (
    QualityMetric("rmse", "RMSE", "Linear EXR RMSE vs reference", True),
    QualityMetric("psnr_clip", "PSNR", "Clipped PSNR (dB)"),
    QualityMetric("lpips_alex", "LPIPS Alex", "LPIPS Alex vs reference"),
    QualityMetric("flip_hdr", "HDR FLIP", "HDR FLIP vs reference"),
)


@dataclass(frozen=True)
class TimedRun:
    row: dict[str, str]
    mean_ms: float | None
    p95_ms: float | None
    frame_count: int
    rmse: float | None = None
    mae: float | None = None
    psnr_clip: float | None = None
    lpips_alex: float | None = None
    flip_hdr: float | None = None
    roi_rmse_max: float | None = None
    glass_rmse: float | None = None
    glass_psnr_clip: float | None = None
    glass_lpips_alex: float | None = None
    glass_flip_hdr: float | None = None
    non_glass_rmse: float | None = None
    non_glass_psnr_clip: float | None = None
    non_glass_lpips_alex: float | None = None
    non_glass_flip_hdr: float | None = None
    quality_error: str = ""


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, header: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in header})


def latest_successful_runs(rows: list[dict[str, str]], key_fields: tuple[str, ...]) -> list[dict[str, str]]:
    latest: dict[tuple[str, ...], dict[str, str]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        key = tuple(row.get(field, "") for field in key_fields)
        latest[key] = row
    return list(latest.values())


def latest_rows_by_run_id(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    latest: dict[str, dict[str, str]] = {}
    anonymous = []
    for row in rows:
        run_id = row.get("run_id", "")
        if not run_id:
            anonymous.append(row)
            continue
        latest[run_id] = row
    return list(latest.values()) + anonymous


def _float_values(rows: list[dict[str, str]], run_id: str) -> list[float]:
    latest_by_frame: dict[tuple[str, str], tuple[int, float]] = {}
    for index, row in enumerate(rows):
        if row.get("run_id") != run_id or row.get("status") != "ok":
            continue
        try:
            value = float(row.get("elapsed_ms", ""))
        except ValueError:
            continue
        frame = row.get("frame", "")
        camera = row.get("camera", "")
        key = (frame, camera) if frame or camera else (str(index), "")
        latest_by_frame[key] = (index, value)
    return [value for _, value in sorted(latest_by_frame.values(), key=lambda pair: pair[0])]


def mean_frame_ms(frame_time_path: Path, run_id: str) -> float | None:
    values = _float_values(read_csv(frame_time_path), run_id)
    if not values:
        return None
    return sum(values) / len(values)


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = (len(ordered) - 1) * pct / 100.0
    lo = math.floor(index)
    hi = math.ceil(index)
    if lo == hi:
        return ordered[int(index)]
    return ordered[lo] * (hi - index) + ordered[hi] * (index - lo)


def _timed_run(row: dict[str, str], frame_rows: list[dict[str, str]]) -> TimedRun:
    values = _float_values(frame_rows, row.get("run_id", ""))
    mean_ms = sum(values) / len(values) if values else None
    p95_ms = percentile(values, 95.0)
    return TimedRun(row=row, mean_ms=mean_ms, p95_ms=p95_ms, frame_count=len(values))


class QualityComputer:
    def __init__(self) -> None:
        self._cache: dict[str, Any] = {}
        self._metric_cache: dict[tuple[str, str, tuple[int, ...]], tuple[dict[str, float], str]] = {}
        self._lpips_model: Any | None = None
        self._lpips_error = ""
        self._flip_module: Any | None = None
        self._flip_error = ""
        self.error = ""

    def _load(self, path: str) -> Any:
        if path in self._cache:
            return self._cache[path]
        os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
        import cv2  # type: ignore
        import numpy as np

        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise RuntimeError(f"could not read EXR: {path}")
        image = image.astype("float32")
        if image.ndim == 3 and image.shape[2] >= 3:
            image = image[:, :, :3][:, :, ::-1]
        elif image.ndim == 2:
            image = np.repeat(image[:, :, None], 3, axis=2)
        self._cache[path] = image
        return image

    def _lpips(self) -> Any | None:
        if self._lpips_model is not None or self._lpips_error:
            return self._lpips_model
        try:
            os.environ.setdefault("TORCH_HOME", "/tmp/torch_home")
            import lpips  # type: ignore

            self._lpips_model = lpips.LPIPS(net="alex", verbose=False)
            self._lpips_model.eval()
        except Exception as exc:  # pragma: no cover - depends on local model cache/network
            self._lpips_error = repr(exc)
        return self._lpips_model

    def _flip(self) -> Any | None:
        if self._flip_module is not None or self._flip_error:
            return self._flip_module
        try:
            import flip_evaluator  # type: ignore

            self._flip_module = flip_evaluator
        except Exception as exc:  # pragma: no cover - depends on local optional package
            self._flip_error = repr(exc)
        return self._flip_module

    def _crop(self, image: Any, box: list[int] | None) -> Any:
        if not box:
            return image
        x0, y0, x1, y1 = [int(value) for value in box]
        return image[y0:y1, x0:x1, :]

    def _lpips_metric(self, reference: Any, test: Any) -> tuple[float | None, str]:
        model = self._lpips()
        if model is None:
            return None, f"LPIPS unavailable: {self._lpips_error}"
        try:
            import numpy as np
            import torch

            ref = np.clip(reference, 0.0, 1.0)
            tst = np.clip(test, 0.0, 1.0)
            ref_tensor = torch.from_numpy(ref).permute(2, 0, 1).unsqueeze(0).float() * 2.0 - 1.0
            tst_tensor = torch.from_numpy(tst).permute(2, 0, 1).unsqueeze(0).float() * 2.0 - 1.0
            with torch.no_grad():
                return float(model(ref_tensor, tst_tensor).item()), ""
        except Exception as exc:  # pragma: no cover - depends on local torch/lpips stack
            return None, repr(exc)

    def _flip_metric(self, reference: Any, test: Any) -> tuple[float | None, str]:
        module = self._flip()
        if module is None:
            return None, f"FLIP unavailable: {self._flip_error}"
        try:
            _errormap, mean_error, _params = module.evaluate(
                reference,
                test,
                "HDR",
                inputsRGB=True,
                applyMagma=False,
                computeMeanError=True,
            )
            return float(mean_error), ""
        except Exception as exc:  # pragma: no cover - depends on local flip stack
            return None, repr(exc)

    def metrics(self, reference: str, test: str, box: list[int] | None = None) -> tuple[dict[str, float], str]:
        if not reference or not test:
            return {}, "missing reference or test path"
        if not Path(reference).exists() or not Path(test).exists():
            return {}, "missing reference or test EXR"
        key = (reference, test, tuple(box or ()))
        if key in self._metric_cache:
            return self._metric_cache[key]
        try:
            import numpy as np

            ref = self._load(reference)
            tst = self._load(test)
            if ref.shape != tst.shape:
                return {}, f"shape mismatch: reference={ref.shape}, test={tst.shape}"
            ref = self._crop(ref, box)
            tst = self._crop(tst, box)
            diff = tst - ref
            mse = float(np.mean(diff * diff))
            rmse = math.sqrt(max(mse, 0.0))
            mae = float(np.mean(np.abs(diff)))
            clipped = np.clip(tst, 0.0, 1.0) - np.clip(ref, 0.0, 1.0)
            clipped_mse = float(np.mean(clipped * clipped))
            psnr = 99.0 if clipped_mse <= 1e-12 else 20.0 * math.log10(1.0 / math.sqrt(clipped_mse))
            result = {"rmse": rmse, "mae": mae, "psnr_clip": psnr}
            errors = []
            lpips_value, lpips_error = self._lpips_metric(ref, tst)
            if lpips_value is not None:
                result["lpips_alex"] = lpips_value
            elif lpips_error:
                errors.append(lpips_error)
            flip_value, flip_error = self._flip_metric(ref, tst)
            if flip_value is not None:
                result["flip_hdr"] = flip_value
            elif flip_error:
                errors.append(flip_error)
            response = (result, "; ".join(dict.fromkeys(errors)))
            self._metric_cache[key] = response
            return response
        except Exception as exc:  # pragma: no cover - depends on local EXR stack
            return {}, repr(exc)


def _reference_for_scene(root: Path, scene: str) -> str:
    path = root / scene / "reference" / "blender_4096spp_denoise_off.exr"
    return str(path) if path.exists() else ""


def render_preview_relpath(scene: str) -> Path:
    return Path(scene) / "analysis" / "render_result.png"


def write_render_preview(root: Path, scene: str) -> Path | None:
    target = root / render_preview_relpath(scene)
    curated_exr = root / scene / "analysis" / "render_result_source.exr"
    source = curated_exr if curated_exr.exists() else Path(_reference_for_scene(root, scene))
    if not source.exists():
        return None
    return write_exr_preview(source, target)


def write_exr_preview(source: Path, target: Path) -> Path | None:
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    import cv2  # type: ignore
    import numpy as np

    image = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
    if image is None:
        return None
    image = image.astype("float32")
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    elif image.ndim == 3 and image.shape[2] >= 3:
        image = image[:, :, :3][:, :, ::-1]
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    image = np.maximum(image, 0.0)

    finite_positive = image[np.isfinite(image) & (image > 0.0)]
    white = float(np.percentile(finite_positive, 99.5)) if finite_positive.size else 1.0
    if white <= 1e-6:
        white = 1.0
    preview = np.clip(image / white, 0.0, 1.0)
    preview = np.power(preview, 1.0 / 2.2)

    target.parent.mkdir(parents=True, exist_ok=True)
    preview_bgr = (preview[:, :, ::-1] * 255.0 + 0.5).astype("uint8")
    cv2.imwrite(str(target), preview_bgr)
    return target


def _load_rgb_image(path: Path) -> Any | None:
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    import cv2  # type: ignore
    import numpy as np

    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        return None
    image = image.astype("float32")
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    elif image.ndim == 3 and image.shape[2] >= 3:
        image = image[:, :, :3][:, :, ::-1]
    return np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)


def _difference_map(reference: Path, source: Path) -> Any | None:
    import numpy as np

    ref = _load_rgb_image(reference)
    test = _load_rgb_image(source)
    if ref is None or test is None or ref.shape != test.shape:
        return None

    return np.mean(np.abs(test - ref), axis=2)


def _difference_scale(reference: Path, sources: list[Path]) -> float:
    import numpy as np

    values = []
    for source in sources:
        diff = _difference_map(reference, source)
        if diff is None:
            continue
        positive = diff[np.isfinite(diff) & (diff > 0.0)]
        if positive.size:
            values.append(positive)
    if not values:
        return 1.0
    scale = float(np.percentile(np.concatenate(values), 99.5))
    if scale <= 1e-8:
        return 1.0
    return scale


def write_difference_preview(reference: Path, source: Path, target: Path, *, scale: float | None = None) -> Path | None:
    import cv2  # type: ignore
    import numpy as np

    diff = _difference_map(reference, source)
    if diff is None:
        return None

    diff_scale = scale if scale is not None and scale > 1e-8 else _difference_scale(reference, [source])
    normalized = np.clip(diff / diff_scale, 0.0, 1.0)
    heat = cv2.applyColorMap((normalized * 255.0 + 0.5).astype("uint8"), cv2.COLORMAP_INFERNO)
    target.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(target), heat)
    return target


def _safe_label(row: dict[str, str]) -> str:
    for field in ("profile", "adaptive_threshold", "samples"):
        value = row.get(field, "")
        if value:
            return str(value).replace(".", "p")
    return row.get("run_id", "run")


def _contact_sheet_font():
    from PIL import ImageFont

    for font_path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        if Path(font_path).exists():
            return ImageFont.truetype(font_path, CONTACT_SHEET_FONT_SIZE)
    return ImageFont.load_default()


def _write_contact_sheet(paths: list[Path], labels: list[str], target: Path) -> None:
    if not paths:
        return
    try:
        from PIL import Image, ImageDraw

        panels = [Image.open(path).convert("RGB") for path in paths]
        panel_width = max(image.width for image in panels)
        panel_height = max(image.height for image in panels)
        tile_height = panel_height + CONTACT_SHEET_LABEL_HEIGHT
        columns = min(3, len(panels))
        rows = math.ceil(len(panels) / columns)
        sheet = Image.new("RGB", (columns * panel_width, rows * tile_height), "white")
        draw = ImageDraw.Draw(sheet)
        font = _contact_sheet_font()
        for index, panel in enumerate(panels):
            x = (index % columns) * panel_width
            y = (index // columns) * tile_height
            sheet.paste(panel, (x, y))
            draw.text((x + 18, y + panel_height + 20), labels[index][:96], fill="black", font=font)
        target.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(target)
    except Exception:
        return


def write_comparison_previews(root: Path, analysis: dict[str, Any]) -> None:
    scene = analysis["scene"]
    out_dir = root / scene / "analysis" / "previews"
    reference_text = analysis.get("reference", "")
    reference = Path(reference_text) if reference_text else None
    groups = {
        "spp": analysis["spp"],
        "adaptive": analysis["adaptive"],
        "glass": analysis["glass"],
    }
    for group, items in groups.items():
        paths = []
        labels = []
        source_paths: list[Path] = []
        for item in items:
            source = item.row.get("output_path", "")
            source_path = Path(source) if source else None
            if source_path is None or not source_path.exists():
                continue
            label = _safe_label(item.row)
            if group == "spp":
                label = f"{item.row.get('samples')} spp"
            elif group == "adaptive":
                label = f"{item.row.get('samples')} spp, adaptive {item.row.get('adaptive_threshold')}"
            elif group == "glass":
                label = item.row.get("profile", "profile")
            target = out_dir / group / f"{label.replace(' ', '_').replace('.', 'p')}.png"
            if write_exr_preview(source_path, target) is not None:
                paths.append(target)
                labels.append(label)
                source_paths.append(source_path)
        _write_contact_sheet(paths, labels, out_dir / f"{group}_contact_sheet.png")
        diff_paths = []
        diff_labels = []
        if reference is not None and reference.exists() and source_paths:
            scale = _difference_scale(reference, source_paths)
            for source_path, preview_path, label in zip(source_paths, paths, labels, strict=False):
                diff_target = out_dir / f"{group}_difference" / preview_path.name
                if write_difference_preview(reference, source_path, diff_target, scale=scale) is not None:
                    diff_paths.append(diff_target)
                    diff_labels.append(f"{label} diff vs 4096 spp denoiser off")
        _write_contact_sheet(diff_paths, diff_labels, out_dir / f"{group}_difference_contact_sheet.png")


def _attach_quality(
    timed: list[TimedRun],
    scene: str,
    reference: str,
    boxes: dict[str, list[int]],
    quality: QualityComputer,
) -> list[TimedRun]:
    enriched = []
    for item in timed:
        output = item.row.get("output_path", "")
        metrics, error = quality.metrics(reference, output)
        roi_values = []
        for box in boxes.values():
            roi_metrics, roi_error = quality.metrics(reference, output, box)
            if roi_error:
                error = error or roi_error
                continue
            roi_values.append(roi_metrics["rmse"])
        glass_metrics, glass_error = _group_metrics(scene, "glass", reference, output, quality)
        non_glass_metrics, non_glass_error = _group_metrics(scene, "non_glass", reference, output, quality)
        error = error or glass_error or non_glass_error
        enriched.append(
            TimedRun(
                row=item.row,
                mean_ms=item.mean_ms,
                p95_ms=item.p95_ms,
                frame_count=item.frame_count,
                rmse=metrics.get("rmse"),
                mae=metrics.get("mae"),
                psnr_clip=metrics.get("psnr_clip"),
                lpips_alex=metrics.get("lpips_alex"),
                flip_hdr=metrics.get("flip_hdr"),
                roi_rmse_max=max(roi_values) if roi_values else None,
                glass_rmse=glass_metrics.get("rmse"),
                glass_psnr_clip=glass_metrics.get("psnr_clip"),
                glass_lpips_alex=glass_metrics.get("lpips_alex"),
                glass_flip_hdr=glass_metrics.get("flip_hdr"),
                non_glass_rmse=non_glass_metrics.get("rmse"),
                non_glass_psnr_clip=non_glass_metrics.get("psnr_clip"),
                non_glass_lpips_alex=non_glass_metrics.get("lpips_alex"),
                non_glass_flip_hdr=non_glass_metrics.get("flip_hdr"),
                quality_error=error,
            )
        )
    return enriched


def _mean_metric(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def json_box(box: list[int]) -> str:
    return "[" + ", ".join(str(int(value)) for value in box) + "]"


def _group_metrics(
    scene: str,
    group: str,
    reference: str,
    output: str,
    quality: QualityComputer,
) -> tuple[dict[str, float], str]:
    collected: dict[str, list[float]] = defaultdict(list)
    errors = []
    for box in ROI_GROUPS.get(scene, {}).get(group, {}).values():
        metrics, error = quality.metrics(reference, output, box)
        if error:
            errors.append(error)
        for key, value in metrics.items():
            collected[key].append(float(value))
    return (
        {key: value for key, values in collected.items() if (value := _mean_metric(values)) is not None},
        "; ".join(dict.fromkeys(errors)),
    )


def roi_detail_rows(
    scene: str,
    runs: list[dict[str, str]],
    *,
    reference: str,
    quality: QualityComputer,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in runs:
        if run.get("status") != "ok":
            continue
        output = run.get("output_path", "")
        for group, group_boxes in ROI_GROUPS.get(scene, {}).items():
            for roi_name, box in group_boxes.items():
                metrics, error = quality.metrics(reference, output, box)
                rows.append({
                    "scene": scene,
                    "kind": run.get("kind", ""),
                    "run_id": run.get("run_id", ""),
                    "samples": run.get("samples", ""),
                    "profile": run.get("profile", ""),
                    "adaptive_threshold": run.get("adaptive_threshold", ""),
                    "roi_group": group,
                    "roi_name": roi_name,
                    "box": json_box(box),
                    "rmse_linear": fmt(metrics.get("rmse"), 8),
                    "mae_linear": fmt(metrics.get("mae"), 8),
                    "psnr_clip_db": fmt(metrics.get("psnr_clip"), 3),
                    "lpips_alex": fmt(metrics.get("lpips_alex"), 6),
                    "flip_hdr": fmt(metrics.get("flip_hdr"), 6),
                    "metric_error": error,
                })
    return rows


def region_summary_rows(
    scene: str, runs: list[dict[str, str]], *, reference: str, quality: QualityComputer
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in runs:
        if run.get("status") != "ok":
            continue
        output = run.get("output_path", "")
        for group in ("glass", "non_glass"):
            metrics, error = _group_metrics(scene, group, reference, output, quality)
            rows.append({
                "scene": scene,
                "kind": run.get("kind", ""),
                "run_id": run.get("run_id", ""),
                "samples": run.get("samples", ""),
                "profile": run.get("profile", ""),
                "adaptive_threshold": run.get("adaptive_threshold", ""),
                "roi_group": group,
                "rmse_linear": fmt(metrics.get("rmse"), 8),
                "mae_linear": fmt(metrics.get("mae"), 8),
                "psnr_clip_db": fmt(metrics.get("psnr_clip"), 3),
                "lpips_alex": fmt(metrics.get("lpips_alex"), 6),
                "flip_hdr": fmt(metrics.get("flip_hdr"), 6),
                "metric_error": error,
            })
    return rows


def _ok_runs_by_kind(root: Path, scene: str, kind: str) -> list[dict[str, str]]:
    rows = latest_rows_by_run_id(read_csv(root / scene / "runs.csv"))
    return [row for row in rows if row.get("status") == "ok" and row.get("kind") == kind]


def _scene_timed_runs(root: Path, scene: str, kind: str) -> list[TimedRun]:
    frame_rows = read_csv(root / scene / "overhead" / "frame_time.csv")
    return [_timed_run(row, frame_rows) for row in _ok_runs_by_kind(root, scene, kind)]


def scene_analysis(root: Path, scene: str) -> dict[str, Any]:
    all_rows = read_csv(root / scene / "runs.csv")
    rows = [
        row
        for row in latest_rows_by_run_id(all_rows)
        if row.get("kind") != "adaptive_sampling" or row.get("samples") in {"8", "16", "64"}
    ]
    quality = QualityComputer()
    reference = _reference_for_scene(root, scene)
    boxes = ROI_BOXES[scene]

    spp_candidates = [
        row
        for row in rows
        if row.get("status") == "ok"
        and row.get("kind") == "spp_sweep"
        and row.get("camera_count") == "1"
        and row.get("denoiser") in {"on", "off"}
    ]
    spp_rows = latest_successful_runs(spp_candidates, ("samples", "denoiser", "denoiser_engine"))
    frame_rows = read_csv(root / scene / "overhead" / "frame_time.csv")
    spp = _attach_quality([_timed_run(row, frame_rows) for row in spp_rows], scene, reference, boxes, quality)
    spp.sort(key=lambda item: (int(item.row.get("samples", "0")), item.row.get("denoiser", "")))

    reference_rows = _attach_quality(
        [_timed_run(row, frame_rows) for row in _ok_runs_by_kind(root, scene, "reference")],
        scene,
        reference,
        boxes,
        quality,
    )
    reference_rows.sort(key=lambda item: int(item.row.get("samples", "0")))

    multi = _scene_timed_runs(root, scene, "multi_camera")
    multi.sort(key=lambda item: int(item.row.get("camera_count", "0")))
    denoiser = _attach_quality(_scene_timed_runs(root, scene, "denoiser_matrix"), scene, reference, boxes, quality)
    adaptive = _attach_quality(
        [
            item
            for item in _scene_timed_runs(root, scene, "adaptive_sampling")
            if item.row.get("samples") in {"8", "16", "64"}
        ],
        scene,
        reference,
        boxes,
        quality,
    )
    light = _attach_quality(_scene_timed_runs(root, scene, "light_sampling"), scene, reference, boxes, quality)
    glass = _attach_quality(_scene_timed_runs(root, scene, "glass_light_paths"), scene, reference, boxes, quality)
    stability = _scene_timed_runs(root, scene, "stability")
    included_run_ids = {row.get("run_id", "") for row in rows if row.get("run_id")}
    vram = [
        row
        for row in read_csv(root / scene / "overhead" / "vram_breakdown.csv")
        if not row.get("run_id") or row.get("run_id") in included_run_ids
    ]
    roi_source_runs = {
        item.row.get("run_id", ""): item.row
        for collection in (spp, reference_rows, adaptive, glass)
        for item in collection
        if item.row.get("run_id")
    }
    roi_detail = roi_detail_rows(scene, list(roi_source_runs.values()), reference=reference, quality=quality)
    region_summary = region_summary_rows(scene, list(roi_source_runs.values()), reference=reference, quality=quality)

    return {
        "scene": scene,
        "runs": rows,
        "status_counts": Counter(row.get("status", "") for row in rows),
        "kind_counts": Counter(row.get("kind", "") for row in rows if row.get("status") == "ok"),
        "reference": reference,
        "spp": spp,
        "reference_rows": reference_rows,
        "multi": multi,
        "denoiser": denoiser,
        "adaptive": adaptive,
        "light": light,
        "glass": glass,
        "stability": stability,
        "vram": vram,
        "roi_detail": roi_detail,
        "region_summary": region_summary,
    }


def fmt(value: float | None, digits: int = 2) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def fmt_ms(value: float | None) -> str:
    if value is None:
        return ""
    if value >= 1000.0:
        return f"{value / 1000.0:.2f} s"
    return f"{value:.0f} ms"


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def timed_rows_for_csv(scene: str, items: list[TimedRun]) -> list[dict[str, Any]]:
    rows = []
    for item in items:
        row = item.row
        rows.append({
            "scene": scene,
            "kind": row.get("kind", ""),
            "run_id": row.get("run_id", ""),
            "samples": row.get("samples", ""),
            "denoiser": row.get("denoiser", ""),
            "denoiser_engine": row.get("denoiser_engine", ""),
            "camera_count": row.get("camera_count", ""),
            "frames": row.get("frames", ""),
            "profile": row.get("profile", ""),
            "adaptive_threshold": row.get("adaptive_threshold", ""),
            "light_tree": row.get("light_tree", ""),
            "light_threshold": row.get("light_threshold", ""),
            "mean_ms": fmt(item.mean_ms, 3),
            "p95_ms": fmt(item.p95_ms, 3),
            "frame_count": item.frame_count,
            "rmse_linear": fmt(item.rmse, 8),
            "mae_linear": fmt(item.mae, 8),
            "psnr_clip_db": fmt(item.psnr_clip, 3),
            "lpips_alex": fmt(item.lpips_alex, 6),
            "flip_hdr": fmt(item.flip_hdr, 6),
            "roi_rmse_max": fmt(item.roi_rmse_max, 8),
            "glass_rmse_linear": fmt(item.glass_rmse, 8),
            "glass_psnr_clip_db": fmt(item.glass_psnr_clip, 3),
            "glass_lpips_alex": fmt(item.glass_lpips_alex, 6),
            "glass_flip_hdr": fmt(item.glass_flip_hdr, 6),
            "non_glass_rmse_linear": fmt(item.non_glass_rmse, 8),
            "non_glass_psnr_clip_db": fmt(item.non_glass_psnr_clip, 3),
            "non_glass_lpips_alex": fmt(item.non_glass_lpips_alex, 6),
            "non_glass_flip_hdr": fmt(item.non_glass_flip_hdr, 6),
            "quality_error": item.quality_error,
        })
    return rows


def write_analysis_csvs(root: Path, analyses: dict[str, dict[str, Any]]) -> None:
    header = [
        "scene",
        "kind",
        "run_id",
        "samples",
        "denoiser",
        "denoiser_engine",
        "camera_count",
        "frames",
        "profile",
        "adaptive_threshold",
        "light_tree",
        "light_threshold",
        "mean_ms",
        "p95_ms",
        "frame_count",
        "rmse_linear",
        "mae_linear",
        "psnr_clip_db",
        "lpips_alex",
        "flip_hdr",
        "roi_rmse_max",
        "glass_rmse_linear",
        "glass_psnr_clip_db",
        "glass_lpips_alex",
        "glass_flip_hdr",
        "non_glass_rmse_linear",
        "non_glass_psnr_clip_db",
        "non_glass_lpips_alex",
        "non_glass_flip_hdr",
        "quality_error",
    ]
    roi_header = [
        "scene",
        "kind",
        "run_id",
        "samples",
        "profile",
        "adaptive_threshold",
        "roi_group",
        "roi_name",
        "box",
        "rmse_linear",
        "mae_linear",
        "psnr_clip_db",
        "lpips_alex",
        "flip_hdr",
        "metric_error",
    ]
    region_header = [
        "scene",
        "kind",
        "run_id",
        "samples",
        "profile",
        "adaptive_threshold",
        "roi_group",
        "rmse_linear",
        "mae_linear",
        "psnr_clip_db",
        "lpips_alex",
        "flip_hdr",
        "metric_error",
    ]
    for scene, analysis in analyses.items():
        rows = []
        for key in ("spp", "reference_rows", "adaptive", "glass"):
            rows.extend(timed_rows_for_csv(scene, analysis[key]))
        write_csv(root / scene / "analysis" / "analysis_metrics.csv", header, rows)
        write_csv(root / scene / "analysis" / "roi_quality_detailed.csv", roi_header, analysis["roi_detail"])
        write_csv(root / scene / "analysis" / "region_quality_summary.csv", region_header, analysis["region_summary"])


def _setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "axes.grid": True,
        "grid.alpha": 0.25,
        "figure.dpi": 140,
        "savefig.dpi": 160,
        "font.size": 9,
    })
    return plt


def _save_no_data(path: Path, title: str, message: str) -> None:
    plt = _setup_matplotlib()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.text(0.5, 0.55, title, ha="center", va="center", fontsize=13, weight="bold")
    ax.text(0.5, 0.42, message, ha="center", va="center", wrap=True)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _positive(values: Iterable[float | None]) -> list[float]:
    return [float(value) for value in values if value is not None and value > 0.0]


def _metric(item: TimedRun, metric: QualityMetric) -> float | None:
    return getattr(item, metric.key)


def _region_metric(item: TimedRun, region: str, metric: QualityMetric) -> float | None:
    suffix = {
        "rmse": "rmse",
        "psnr_clip": "psnr_clip",
        "lpips_alex": "lpips_alex",
        "flip_hdr": "flip_hdr",
    }[metric.key]
    return getattr(item, f"{region}_{suffix}")


def _quality_axes(path: Path, title: str):
    plt = _setup_matplotlib()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.2))
    fig.suptitle(title)
    return plt, fig, list(axes.ravel())


def _finish_quality_axes(fig, axes: list[Any]) -> None:
    for ax in axes:
        handles, _labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.96))


def plot_time_to_quality(path: Path, title: str, items: list[TimedRun]) -> None:
    valid = [
        item
        for item in items
        if item.mean_ms is not None and any(_metric(item, metric) is not None for metric in QUALITY_METRICS)
    ]
    if not valid:
        _save_no_data(path, title, "No successful rows with both timing and image quality metrics.")
        return
    plt, fig, axes = _quality_axes(path, title)
    for ax, metric in zip(axes, QUALITY_METRICS, strict=True):
        for denoiser, marker in (("off", "s"), ("on", "o")):
            subset = [
                item for item in valid if item.row.get("denoiser") == denoiser and _metric(item, metric) is not None
            ]
            if not subset:
                continue
            x = [item.mean_ms for item in subset]
            y = [max(_metric(item, metric) or 0.0, 1e-8) if metric.log_y else _metric(item, metric) for item in subset]
            ax.scatter(x, y, label=f"denoiser {denoiser}", marker=marker, s=44)
            for item in subset:
                value = max(_metric(item, metric) or 0.0, 1e-8) if metric.log_y else _metric(item, metric)
                ax.annotate(
                    f"{item.row.get('samples')} spp",
                    (item.mean_ms, value),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=7,
                )
        if _positive(item.mean_ms for item in valid):
            ax.set_xscale("log")
        if metric.log_y and _positive(_metric(item, metric) for item in valid):
            ax.set_yscale("log")
        ax.set_title(metric.label)
        ax.set_xlabel("Mean camera-frame time (ms)")
        ax.set_ylabel(metric.ylabel)
    _finish_quality_axes(fig, axes)
    fig.savefig(path)
    plt.close(fig)


def plot_spp_quality_metrics(path: Path, title: str, items: list[TimedRun]) -> None:
    valid = [
        item
        for item in items
        if item.row.get("denoiser", "on") == "on"
        and item.row.get("samples")
        and any(_metric(item, metric) is not None for metric in QUALITY_METRICS)
    ]
    if not valid:
        _save_no_data(path, title, "No SPP rows with image quality metrics.")
        return
    plt, fig, axes = _quality_axes(path, title)
    ordered = sorted(valid, key=lambda item: int(item.row.get("samples", "0") or 0))
    spp = [int(item.row.get("samples", "0") or 0) for item in ordered]
    for ax, metric in zip(axes, QUALITY_METRICS, strict=True):
        y = [max(_metric(item, metric) or 0.0, 1e-8) if metric.log_y else _metric(item, metric) for item in ordered]
        ax.plot(spp, y, marker="o")
        ax.set_xscale("log")
        if metric.log_y and _positive(_metric(item, metric) for item in ordered):
            ax.set_yscale("log")
        ax.set_title(metric.label)
        ax.set_xlabel("Samples per pixel")
        ax.set_ylabel(metric.ylabel)
    _finish_quality_axes(fig, axes)
    fig.savefig(path)
    plt.close(fig)


def plot_reference(path: Path, title: str, items: list[TimedRun]) -> None:
    valid = [item for item in items if item.mean_ms is not None]
    if not valid:
        _save_no_data(path, title, "No successful reference rows.")
        return
    plt = _setup_matplotlib()
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = [f"{item.row.get('samples')} spp" for item in valid]
    times = [item.mean_ms / 1000.0 for item in valid]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    bars = ax.bar(labels, times, color="#4c78a8")
    for bar, item in zip(bars, valid, strict=False):
        label = "ref" if item.rmse == 0 else f"RMSE {fmt(item.rmse, 4)}"
        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), label, ha="center", va="bottom", fontsize=8)
    ax.set_title(title)
    ax.set_ylabel("Render time (s)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_roi_quality(path: Path, title: str, items: list[TimedRun]) -> None:
    valid = [
        item
        for item in items
        if any(
            _region_metric(item, region, metric) is not None
            for region in ("glass", "non_glass")
            for metric in QUALITY_METRICS
        )
    ]
    if not valid:
        _save_no_data(path, title, "No glass/non-glass ROI quality values were available.")
        return
    plt, fig, axes = _quality_axes(path, title)
    labels = [f"{item.row.get('samples')} spp" for item in valid]
    x = list(range(len(valid)))
    for ax, metric in zip(axes, QUALITY_METRICS, strict=True):
        glass_values = [_region_metric(item, "glass", metric) or 0.0 for item in valid]
        non_glass_values = [_region_metric(item, "non_glass", metric) or 0.0 for item in valid]
        ax.bar([i - 0.18 for i in x], glass_values, width=0.36, label="glass")
        ax.bar([i + 0.18 for i in x], non_glass_values, width=0.36, label="non-glass")
        ax.set_xticks(x, labels, rotation=35, ha="right")
        if metric.log_y and _positive(glass_values + non_glass_values):
            ax.set_yscale("log")
        ax.set_title(metric.label)
        ax.set_ylabel(metric.ylabel)
    _finish_quality_axes(fig, axes)
    fig.savefig(path)
    plt.close(fig)


def plot_multi_camera(path: Path, title: str, items: list[TimedRun]) -> None:
    valid = [item for item in items if item.mean_ms is not None]
    if not valid:
        _save_no_data(path, title, "No successful multi-camera timing rows.")
        return
    plt = _setup_matplotlib()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    cameras = [int(item.row.get("camera_count", "0")) for item in valid]
    per_camera = [item.mean_ms for item in valid]
    batch = [item.mean_ms * int(item.row.get("camera_count", "0")) / 1000.0 for item in valid]
    ax.plot(cameras, per_camera, marker="o", label="per camera frame")
    ax.set_xlabel("Camera count")
    ax.set_ylabel("Mean per-camera render time (ms)")
    ax2 = ax.twinx()
    ax2.plot(cameras, batch, marker="s", color="#f58518", label="per full frame batch")
    ax2.set_ylabel("Estimated full multi-camera frame time (s)")
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="upper left")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_bar_metric(
    path: Path, title: str, items: list[TimedRun], label_fn, *, ylabel: str = "Mean camera-frame time (ms)"
) -> None:
    valid = [item for item in items if item.mean_ms is not None]
    if not valid:
        _save_no_data(path, title, "No successful rows for this axis.")
        return
    plt = _setup_matplotlib()
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = [label_fn(item) for item in valid]
    values = [item.mean_ms for item in valid]
    fig_width = max(7.2, min(12.0, 0.42 * len(labels) + 4.0))
    fig, ax = plt.subplots(figsize=(fig_width, 4.8))
    bars = ax.bar(range(len(labels)), values, color="#54a24b")
    for bar, item in zip(bars, valid, strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            fmt_ms(item.mean_ms),
            ha="center",
            va="bottom",
            fontsize=7,
        )
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_adaptive(path: Path, title: str, items: list[TimedRun], unsupported_rows: list[dict[str, str]]) -> None:
    valid = [item for item in items if item.mean_ms is not None and item.row.get("adaptive_threshold")]
    if not valid:
        thresholds = [row.get("adaptive_threshold", "") for row in unsupported_rows if row.get("adaptive_threshold")]
        if thresholds:
            counts = Counter(thresholds)
            plt = _setup_matplotlib()
            path.parent.mkdir(parents=True, exist_ok=True)
            labels = list(counts)
            fig, ax = plt.subplots(figsize=(7.2, 4.4))
            ax.bar(labels, [counts[label] for label in labels], color="#e45756")
            ax.set_title(title)
            ax.set_ylabel("Unsupported rows")
            ax.set_xlabel("Adaptive threshold")
            ax.text(
                0.5,
                0.92,
                "No successful adaptive rows; chart shows unsupported coverage.",
                transform=ax.transAxes,
                ha="center",
            )
            fig.tight_layout()
            fig.savefig(path)
            plt.close(fig)
            return
        _save_no_data(path, title, "No adaptive rows.")
        return
    plt = _setup_matplotlib()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    for samples in sorted({item.row.get("samples", "") for item in valid}, key=lambda value: int(value or 0)):
        subset = sorted(
            [item for item in valid if item.row.get("samples") == samples],
            key=lambda item: float(item.row.get("adaptive_threshold", "0")),
        )
        x = [float(item.row.get("adaptive_threshold", "0")) for item in subset]
        y = [item.mean_ms for item in subset]
        ax.plot(x, y, marker="o", label=f"{samples} spp")
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_title(title)
    ax.set_xlabel("Adaptive threshold (looser to tighter)")
    ax.set_ylabel("Mean camera-frame time (ms)")
    ax.legend(title="Max SPP")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_adaptive_quality_metrics(path: Path, title: str, items: list[TimedRun]) -> None:
    valid = [
        item
        for item in items
        if item.row.get("adaptive_threshold")
        and item.row.get("samples")
        and any(_metric(item, metric) is not None for metric in QUALITY_METRICS)
    ]
    if not valid:
        _save_no_data(path, title, "No adaptive rows with image quality metrics.")
        return
    plt, fig, axes = _quality_axes(path, title)
    for ax, metric in zip(axes, QUALITY_METRICS, strict=True):
        for samples in sorted({item.row.get("samples", "") for item in valid}, key=lambda value: int(value or 0)):
            subset = sorted(
                [item for item in valid if item.row.get("samples") == samples and _metric(item, metric) is not None],
                key=lambda item: float(item.row.get("adaptive_threshold", "0")),
            )
            if not subset:
                continue
            x = [float(item.row.get("adaptive_threshold", "0")) for item in subset]
            y = [max(_metric(item, metric) or 0.0, 1e-8) if metric.log_y else _metric(item, metric) for item in subset]
            ax.plot(x, y, marker="o", label=f"{samples} spp")
        ax.set_xscale("log")
        ax.invert_xaxis()
        if metric.log_y and _positive(_metric(item, metric) for item in valid):
            ax.set_yscale("log")
        ax.set_title(metric.label)
        ax.set_xlabel("Adaptive threshold (looser to tighter)")
        ax.set_ylabel(metric.ylabel)
    _finish_quality_axes(fig, axes)
    fig.savefig(path)
    plt.close(fig)


def plot_profile_quality_metrics(path: Path, title: str, items: list[TimedRun], label_fn) -> None:
    valid = [item for item in items if any(_metric(item, metric) is not None for metric in QUALITY_METRICS)]
    if not valid:
        _save_no_data(path, title, "No profile rows with image quality metrics.")
        return
    plt, fig, axes = _quality_axes(path, title)
    labels = [label_fn(item) for item in valid]
    x = list(range(len(valid)))
    for ax, metric in zip(axes, QUALITY_METRICS, strict=True):
        values = [max(_metric(item, metric) or 0.0, 1e-8) if metric.log_y else _metric(item, metric) for item in valid]
        ax.bar(x, values, color="#72b7b2")
        ax.set_xticks(x, labels, rotation=35, ha="right")
        if metric.log_y and _positive(_metric(item, metric) for item in valid):
            ax.set_yscale("log")
        ax.set_title(metric.label)
        ax.set_ylabel(metric.ylabel)
    _finish_quality_axes(fig, axes)
    fig.savefig(path)
    plt.close(fig)


def plot_scene_figures(root: Path, analysis: dict[str, Any]) -> None:
    scene = analysis["scene"]
    scene_root = root / scene
    title = SCENE_TITLES[scene]
    plot_time_to_quality(
        scene_root / "core" / "time_to_quality.png", f"{title}: time to proxy quality", analysis["spp"]
    )
    plot_spp_quality_metrics(
        scene_root / "core" / "spp_quality_metrics.png", f"{title}: SPP quality metrics", analysis["spp"]
    )
    plot_reference(
        scene_root / "core" / "reference_convergence.png", f"{title}: reference render cost", analysis["reference_rows"]
    )
    plot_roi_quality(scene_root / "core" / "roi_quality.png", f"{title}: ROI proxy quality", analysis["spp"])
    plot_multi_camera(scene_root / "core" / "multi_camera.png", f"{title}: multi-camera scaling", analysis["multi"])
    plot_time_to_quality(
        scene_root / "core" / "pareto.png", f"{title}: timing/quality envelope", analysis["spp"] + analysis["denoiser"]
    )
    plot_bar_metric(
        scene_root / "controls" / "denoiser_matrix.png",
        f"{title}: denoiser matrix",
        analysis["denoiser"],
        lambda item: (
            "off"
            if item.row.get("denoiser") == "off"
            else f"{item.row.get('denoiser_engine')}\n{item.row.get('denoiser_passes')}"
        ),
    )
    unsupported_adaptive = [
        row for row in analysis["runs"] if row.get("kind") == "adaptive_sampling" and row.get("status") != "ok"
    ]
    plot_adaptive(
        scene_root / "controls" / "adaptive_sampling.png",
        f"{title}: adaptive sampling",
        analysis["adaptive"],
        unsupported_adaptive,
    )
    plot_adaptive_quality_metrics(
        scene_root / "controls" / "adaptive_sampling_quality.png",
        f"{title}: adaptive sampling quality metrics",
        analysis["adaptive"],
    )
    plot_bar_metric(
        scene_root / "controls" / "light_sampling.png",
        f"{title}: light sampling",
        analysis["light"],
        lambda item: f"tree {item.row.get('light_tree')}\nthr {item.row.get('light_threshold')}",
    )
    if scene == "complex_glass_cube_reach":
        plot_bar_metric(
            scene_root / "controls" / "glass_light_paths.png",
            f"{title}: glass light-path profiles",
            analysis["glass"],
            lambda item: item.row.get("profile", "profile"),
        )
        plot_profile_quality_metrics(
            scene_root / "controls" / "glass_light_paths_quality.png",
            f"{title}: glass light-path quality metrics",
            analysis["glass"],
            lambda item: item.row.get("profile", "profile"),
        )
    else:
        _save_no_data(
            scene_root / "controls" / "glass_light_paths.png",
            f"{title}: glass light-path profiles",
            "Not applicable: the normal task1 scenario is not the glass stress scene.",
        )
        _save_no_data(
            scene_root / "controls" / "glass_light_paths_quality.png",
            f"{title}: glass light-path quality metrics",
            "Not applicable: the normal task1 scenario is not the glass stress scene.",
        )


def plot_root_figures(root: Path, analyses: dict[str, dict[str, Any]]) -> None:
    plt = _setup_matplotlib()
    figures = root / "analysis_figures"
    figures.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    for scene, analysis in analyses.items():
        valid = [item for item in analysis["spp"] if item.mean_ms is not None and item.row.get("denoiser") == "on"]
        if not valid:
            continue
        ax.plot(
            [int(item.row["samples"]) for item in valid],
            [item.mean_ms for item in valid],
            marker="o",
            label=SCENE_TITLES[scene],
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Samples per pixel")
    ax.set_ylabel("Mean camera-frame time (ms)")
    ax.set_title("SPP timing comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "spp_timing_comparison.png")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.2))
    axes_list = list(axes.ravel())
    fig.suptitle("SPP quality metric comparison")
    for ax, metric in zip(axes_list, QUALITY_METRICS, strict=True):
        plotted_metric = False
        for scene, analysis in analyses.items():
            valid = [
                item
                for item in analysis["spp"]
                if item.row.get("denoiser") == "on" and item.row.get("samples") and _metric(item, metric) is not None
            ]
            if not valid:
                continue
            ordered = sorted(valid, key=lambda item: int(item.row.get("samples", "0") or 0))
            ax.plot(
                [int(item.row["samples"]) for item in ordered],
                [
                    max(_metric(item, metric) or 0.0, 1e-8) if metric.log_y else _metric(item, metric)
                    for item in ordered
                ],
                marker="o",
                label=SCENE_TITLES[scene],
            )
            plotted_metric = True
        ax.set_xscale("log")
        if metric.log_y:
            ax.set_yscale("log")
        ax.set_xlabel("Samples per pixel")
        ax.set_ylabel(metric.ylabel)
        ax.set_title(metric.label)
        if plotted_metric:
            ax.legend(fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(figures / "spp_quality_metrics_comparison.png")
    fig.savefig(figures / "spp_proxy_quality_comparison.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    plotted = False
    for scene, analysis in analyses.items():
        valid = [item for item in analysis["multi"] if item.mean_ms is not None]
        if not valid:
            continue
        ax.plot(
            [int(item.row["camera_count"]) for item in valid],
            [item.mean_ms for item in valid],
            marker="o",
            label=SCENE_TITLES[scene],
        )
        plotted = True
    ax.set_xlabel("Camera count")
    ax.set_ylabel("Mean per-camera render time (ms)")
    ax.set_title("Multi-camera per-camera timing")
    if plotted:
        ax.legend()
        fig.tight_layout()
        fig.savefig(figures / "multi_camera_comparison.png")
        plt.close(fig)
    else:
        plt.close(fig)
        _save_no_data(
            figures / "multi_camera_comparison.png",
            "Multi-camera per-camera timing",
            "No multi-camera rows in this reduced rerun.",
        )

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    labels = []
    peaks = []
    for scene, analysis in analyses.items():
        values = []
        for row in analysis["vram"]:
            try:
                values.append(float(row.get("peak_mb", "")))
            except ValueError:
                continue
        if values:
            labels.append(SCENE_TITLES[scene])
            peaks.append(max(values))
    if labels:
        ax.bar(labels, peaks, color="#b279a2")
        ax.set_ylabel("Peak VRAM observed (MB)")
        ax.set_title("Observed peak VRAM by scene")
        for label_index, value in enumerate(peaks):
            ax.text(label_index, value, f"{value:.0f} MB", ha="center", va="bottom")
        fig.tight_layout()
        fig.savefig(figures / "vram_peak_by_scene.png")
    else:
        plt.close(fig)
        _save_no_data(figures / "vram_peak_by_scene.png", "Observed peak VRAM by scene", "No VRAM rows available.")
        return
    plt.close(fig)


def _scene_summary_rows(analyses: dict[str, dict[str, Any]]) -> list[list[str]]:
    rows = []
    for scene, analysis in analyses.items():
        rows.append([
            SCENE_TITLES[scene],
            str(len(analysis["runs"])),
            ", ".join(f"{key}={value}" for key, value in sorted(analysis["status_counts"].items())),
            ", ".join(f"{key}={value}" for key, value in sorted(analysis["kind_counts"].items())),
        ])
    return rows


def _timing_table(items: list[TimedRun], *, limit: int | None = None) -> list[list[str]]:
    rows = []
    for item in items[:limit]:
        row = item.row
        rows.append([
            row.get("kind", ""),
            row.get("samples", ""),
            row.get("denoiser", ""),
            row.get("denoiser_engine", ""),
            row.get("camera_count", ""),
            row.get("frames", ""),
            str(item.frame_count),
            fmt_ms(item.mean_ms),
            fmt(item.rmse, 5),
            fmt(item.lpips_alex, 4),
            fmt(item.flip_hdr, 4),
            fmt(item.roi_rmse_max, 5),
            fmt(item.glass_psnr_clip, 2),
            fmt(item.non_glass_psnr_clip, 2),
        ])
    return rows


def _roi_box_table(scene: str) -> list[list[str]]:
    rows = []
    for group, boxes in ROI_GROUPS.get(scene, {}).items():
        for name, box in boxes.items():
            rows.append([group, name, json_box(box)])
    return rows


TIMING_HEADERS = [
    "Kind",
    "SPP",
    "Denoiser",
    "Engine",
    "Cameras",
    "Frames",
    "Timed frames",
    "Mean time",
    "RMSE",
    "LPIPS",
    "FLIP",
    "ROI max RMSE",
    "Glass PSNR",
    "Non-glass PSNR",
]


def _multi_table(items: list[TimedRun]) -> list[list[str]]:
    rows = []
    for item in items:
        cameras = int(item.row.get("camera_count", "0") or 0)
        batch_ms = item.mean_ms * cameras if item.mean_ms is not None else None
        rows.append([
            item.row.get("camera_count", ""),
            item.row.get("frames", ""),
            str(item.frame_count),
            fmt_ms(item.mean_ms),
            fmt_ms(batch_ms),
            fmt(item.p95_ms, 1),
        ])
    return rows


def _denoiser_table(items: list[TimedRun]) -> list[list[str]]:
    rows = []
    for item in items:
        row = item.row
        rows.append([
            row.get("denoiser", ""),
            row.get("denoiser_engine", ""),
            row.get("denoiser_passes", ""),
            row.get("denoising_prefilter", ""),
            row.get("denoising_quality", ""),
            fmt_ms(item.mean_ms),
            fmt(item.rmse, 5),
            fmt(item.roi_rmse_max, 5),
        ])
    return rows


def _adaptive_table(items: list[TimedRun]) -> list[list[str]]:
    rows = []
    for item in sorted(
        items,
        key=lambda run: (int(run.row.get("samples", "0") or 0), float(run.row.get("adaptive_threshold", "0") or 0)),
    ):
        row = item.row
        rows.append([
            row.get("samples", ""),
            row.get("adaptive_threshold", ""),
            row.get("adaptive_min_samples", ""),
            row.get("frames", ""),
            fmt_ms(item.mean_ms),
            fmt(item.rmse, 5),
            fmt(item.roi_rmse_max, 5),
        ])
    return rows


def _region_metric_bundle(rmse: float | None, psnr: float | None, lpips: float | None, flip: float | None) -> str:
    return f"{fmt(rmse, 5)} / {fmt(psnr, 2)} dB / {fmt(lpips, 4)} / {fmt(flip, 4)}"


def _lower_owner(glass: float | None, non_glass: float | None, *, tolerance: float = 1e-9) -> str:
    if glass is None or non_glass is None:
        return "unavailable"
    if math.isclose(glass, non_glass, abs_tol=tolerance):
        return "tie"
    return "glass" if glass < non_glass else "non-glass"


def _region_reading(item: TimedRun) -> str:
    rmse_owner = _lower_owner(item.glass_rmse, item.non_glass_rmse, tolerance=5e-6)
    flip_owner = _lower_owner(item.glass_flip_hdr, item.non_glass_flip_hdr, tolerance=5e-6)
    if rmse_owner == "unavailable" or flip_owner == "unavailable":
        return "region metrics unavailable"
    if rmse_owner == flip_owner:
        return f"{rmse_owner} has lower RMSE and FLIP"
    return f"{rmse_owner} has lower RMSE, {flip_owner} has lower FLIP"


def _spp_region_table(items: list[TimedRun]) -> list[list[str]]:
    denoised = [item for item in items if item.row.get("denoiser", "on") == "on"]
    selected = denoised or items
    rows = []
    for item in sorted(selected, key=lambda run: int(run.row.get("samples", "0") or 0)):
        rows.append([
            item.row.get("samples", ""),
            fmt_ms(item.mean_ms),
            _region_metric_bundle(item.glass_rmse, item.glass_psnr_clip, item.glass_lpips_alex, item.glass_flip_hdr),
            _region_metric_bundle(
                item.non_glass_rmse,
                item.non_glass_psnr_clip,
                item.non_glass_lpips_alex,
                item.non_glass_flip_hdr,
            ),
            _region_reading(item),
        ])
    return rows


def _glass_region_table(items: list[TimedRun]) -> list[list[str]]:
    rows = []
    for item in items:
        rows.append([
            item.row.get("profile", ""),
            fmt_ms(item.mean_ms),
            _region_metric_bundle(item.glass_rmse, item.glass_psnr_clip, item.glass_lpips_alex, item.glass_flip_hdr),
            _region_metric_bundle(
                item.non_glass_rmse,
                item.non_glass_psnr_clip,
                item.non_glass_lpips_alex,
                item.non_glass_flip_hdr,
            ),
            _region_reading(item),
        ])
    return rows


def _metric_spread(values: list[float | None], digits: int) -> str:
    numbers = [float(value) for value in values if value is not None]
    if not numbers:
        return ""
    return fmt(max(numbers) - min(numbers), digits)


def _metric_range(values: list[float | None], digits: int) -> str:
    numbers = [float(value) for value in values if value is not None]
    if not numbers:
        return ""
    return f"{fmt(min(numbers), digits)}-{fmt(max(numbers), digits)}"


def _adaptive_spread_bundle(items: list[TimedRun], prefix: str) -> str:
    return " / ".join([
        _metric_spread([getattr(item, f"{prefix}_rmse") for item in items], 5),
        f"{_metric_spread([getattr(item, f'{prefix}_psnr_clip') for item in items], 3)} dB",
        _metric_spread([getattr(item, f"{prefix}_lpips_alex") for item in items], 4),
        _metric_spread([getattr(item, f"{prefix}_flip_hdr") for item in items], 4),
    ])


def _adaptive_region_spread_table(items: list[TimedRun]) -> list[list[str]]:
    by_spp: dict[str, list[TimedRun]] = defaultdict(list)
    for item in items:
        if item.row.get("samples"):
            by_spp[item.row.get("samples", "")].append(item)
    rows = []
    for samples, group in sorted(by_spp.items(), key=lambda pair: int(pair[0] or 0)):
        ordered = sorted(group, key=lambda run: float(run.row.get("adaptive_threshold", "0") or 0))
        thresholds = ", ".join(item.row.get("adaptive_threshold", "") for item in ordered)
        best = min(ordered, key=lambda item: item.rmse if item.rmse is not None else float("inf"))
        glass_rmse_spread = max((item.glass_rmse or 0.0) for item in ordered) - min(
            (item.glass_rmse or 0.0) for item in ordered
        )
        non_glass_rmse_spread = max((item.non_glass_rmse or 0.0) for item in ordered) - min(
            (item.non_glass_rmse or 0.0) for item in ordered
        )
        reading = (
            "threshold changes are negligible"
            if max(glass_rmse_spread, non_glass_rmse_spread) < 1e-4
            else "threshold changes are visible"
        )
        rows.append([
            samples,
            thresholds,
            _adaptive_spread_bundle(ordered, "glass"),
            _adaptive_spread_bundle(ordered, "non_glass"),
            _metric_range([item.rmse for item in ordered], 5),
            best.row.get("adaptive_threshold", ""),
            reading,
        ])
    return rows


def _light_table(items: list[TimedRun]) -> list[list[str]]:
    rows = []
    for item in sorted(
        items, key=lambda run: (run.row.get("light_tree", ""), float(run.row.get("light_threshold", "0") or 0))
    ):
        row = item.row
        rows.append([
            row.get("light_tree", ""),
            row.get("light_threshold", ""),
            row.get("samples", ""),
            row.get("frames", ""),
            fmt_ms(item.mean_ms),
            fmt(item.rmse, 5),
            fmt(item.roi_rmse_max, 5),
        ])
    return rows


def _glass_table(items: list[TimedRun]) -> list[list[str]]:
    rows = []
    for item in items:
        row = item.row
        caustics = f"R:{row.get('caustics_reflective', '')} / T:{row.get('caustics_refractive', '')}"
        rows.append([
            row.get("profile", ""),
            row.get("max_bounces", ""),
            row.get("transmission_bounces", ""),
            row.get("transparent_bounces", ""),
            caustics,
            row.get("sample_clamp_indirect", ""),
            row.get("filter_glossy", ""),
            fmt_ms(item.mean_ms),
            fmt(item.rmse, 5),
            fmt(item.roi_rmse_max, 5),
        ])
    return rows


def _find_spp(items: list[TimedRun], samples: str) -> TimedRun | None:
    for item in items:
        if item.row.get("samples") == samples and item.row.get("denoiser") == "on":
            return item
    return None


def _find_adaptive(items: list[TimedRun], threshold: str) -> TimedRun | None:
    for item in items:
        if item.row.get("adaptive_threshold") == threshold:
            return item
    return None


def _find_glass_profile(items: list[TimedRun], profile: str) -> TimedRun | None:
    for item in items:
        if item.row.get("profile") == profile:
            return item
    return None


def _interpretation_rows(analysis: dict[str, Any]) -> list[list[str]]:
    rows: list[list[str]] = []
    spp_64 = _find_spp(analysis["spp"], "64")
    spp_256 = _find_spp(analysis["spp"], "256")
    spp_4096 = _find_spp(analysis["spp"], "4096")
    if spp_64 and spp_4096:
        speedup = (spp_4096.mean_ms / spp_64.mean_ms) if spp_4096.mean_ms and spp_64.mean_ms else None
        rows.append([
            "64 SPP baseline",
            f"64 spp is {fmt_ms(spp_64.mean_ms)} with RMSE {fmt(spp_64.rmse, 5)}, LPIPS {fmt(spp_64.lpips_alex, 4)}, and FLIP {fmt(spp_64.flip_hdr, 4)}; 4096 spp is {fmt_ms(spp_4096.mean_ms)} with RMSE {fmt(spp_4096.rmse, 5)}.",
            f"4096 spp costs {fmt(speedup, 1)}x more time than the 64 spp baseline for a {fmt((spp_64.rmse or 0) - (spp_4096.rmse or 0), 5)} RMSE reduction.",
        ])
    adaptive_64 = [item for item in analysis["adaptive"] if item.row.get("samples") == "64"]
    loose = next((item for item in adaptive_64 if item.row.get("adaptive_threshold") == "0.1"), None)
    mid = next((item for item in adaptive_64 if item.row.get("adaptive_threshold") == "0.03"), None)
    tight = next((item for item in adaptive_64 if item.row.get("adaptive_threshold") == "0.003"), None)
    if loose and mid and tight:
        rows.append([
            "Adaptive sampling",
            f"At 64 spp, threshold 0.1 is {fmt_ms(loose.mean_ms)} / RMSE {fmt(loose.rmse, 5)}; threshold 0.03 is {fmt_ms(mid.mean_ms)} / RMSE {fmt(mid.rmse, 5)}; threshold 0.003 is {fmt_ms(tight.mean_ms)} / RMSE {fmt(tight.rmse, 5)}.",
            "The adaptive comparison is now bounded by the practical 64/16/8 spp regimes instead of near-reference 4096 spp adaptive rows.",
        ])
    default = _find_glass_profile(analysis["glass"], "glass_default")
    fast = _find_glass_profile(analysis["glass"], "glass_fast")
    caustics_off = _find_glass_profile(analysis["glass"], "caustics_off")
    clamp = _find_glass_profile(analysis["glass"], "clamp_indirect_2")
    if default and fast and caustics_off and clamp:
        rows.append([
            "Glass paths",
            f"default is {fmt_ms(default.mean_ms)} / RMSE {fmt(default.rmse, 5)}; fast is {fmt_ms(fast.mean_ms)} / RMSE {fmt(fast.rmse, 5)}; caustics-off is {fmt_ms(caustics_off.mean_ms)} / RMSE {fmt(caustics_off.rmse, 5)}; clamp-2 is {fmt_ms(clamp.mean_ms)} / RMSE {fmt(clamp.rmse, 5)}.",
            "The fast and caustics-off rows reduce path complexity but visibly and numerically damage glass-heavy regions; clamp-2 is the smaller degradation among the tested shortcuts.",
        ])
    if spp_64:
        rows.append([
            "ROI split",
            f"At the 64 spp baseline, glass PSNR/LPIPS/FLIP are {fmt(spp_64.glass_psnr_clip, 2)} dB / {fmt(spp_64.glass_lpips_alex, 4)} / {fmt(spp_64.glass_flip_hdr, 4)}; non-glass PSNR/LPIPS/FLIP are {fmt(spp_64.non_glass_psnr_clip, 2)} dB / {fmt(spp_64.non_glass_lpips_alex, 4)} / {fmt(spp_64.non_glass_flip_hdr, 4)}.",
            "This split prevents transparent-object errors from being hidden by easier robot/table regions.",
        ])
    return rows


def _vram_table(analysis: dict[str, Any]) -> list[list[str]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in analysis["vram"]:
        try:
            grouped[row.get("run_id", "")].append(float(row.get("peak_mb", "")))
        except ValueError:
            continue
    values = [max(v) for v in grouped.values() if v]
    if not values:
        return [["Peak VRAM", "not available"]]
    return [
        ["rows", str(len(values))],
        ["min peak", f"{min(values):.0f} MB"],
        ["median peak", f"{statistics.median(values):.0f} MB"],
        ["max peak", f"{max(values):.0f} MB"],
    ]


SEED_STABILITY_REFERENCE_FALLBACK = Path(
    "outputs/rendering_benchmark/blender_2026-05-09_glass_only/complex_glass_cube_reach/reference/blender_4096spp_denoise_off.exr"
)


def _seed_stability_reference_for(scene: str) -> Path | None:
    if scene != "complex_glass_cube_reach":
        return None
    if SEED_STABILITY_REFERENCE_FALLBACK.exists():
        return SEED_STABILITY_REFERENCE_FALLBACK
    return None


def _seed_stability_grouping(scene: str) -> tuple[dict[str, list[int]] | None, dict[str, list[str]] | None]:
    groups = ROI_GROUPS.get(scene)
    if not groups:
        return None, None
    boxes: dict[str, list[int]] = {}
    grouping: dict[str, list[str]] = {}
    for group_name, group_boxes in groups.items():
        grouping[group_name] = list(group_boxes.keys())
        boxes.update(group_boxes)
    return boxes, grouping


def write_markdown(root: Path, analyses: dict[str, dict[str, Any]]) -> None:
    lines = [
        "# Blender Rendering Benchmark Analysis",
        "",
        f"This is an interpretation report for the Blender/Cycles benchmark runs under `{root}`.",
        "It is scoped to the complex glass cube-reach scenario only.",
        "",
        "## Scope And Validity",
        "",
        "- Scenario: `complex_glass_cube_reach`, derived from the cube-reach benchmark path with glass-heavy props and the recorded `cube_reach_default_initial.pt` state replay.",
        "- Renderer/runtime: Blender 5.0.1 through the `isaacsim` Python environment, Cycles/OptiX on the host GPU.",
        "- Quality metrics: global and ROI-split linear EXR RMSE/MAE, clipped PSNR, LPIPS Alex, and HDR FLIP.",
        "- ROI split: several glass boxes cover the display case, bottle, hurricane glass, and rear tumbler; non-glass boxes cover robot arms, cube, and plain table regions.",
        "- Time tradeoff: this rerun focuses on the 64 SPP baseline plus low-SPP adaptive sampling at 64, 16, and 8 SPP; other benchmark axes are intentionally excluded.",
        "",
        "## Coverage",
        "",
        markdown_table(["Scene", "Runs", "Status counts", "Successful kind counts"], _scene_summary_rows(analyses)),
        "",
        "## Overall Figures",
        "",
        "![SPP timing comparison](analysis_figures/spp_timing_comparison.png)",
        "",
        "![SPP quality metric comparison](analysis_figures/spp_quality_metrics_comparison.png)",
        "",
        "![Observed peak VRAM](analysis_figures/vram_peak_by_scene.png)",
        "",
        "## Main Interpretation",
        "",
        "- Read the SPP, adaptive, and glass-profile tables below as separate quality/cost axes against the retained `4096 spp, denoiser off` reference. The practical baseline row is `64 spp, denoiser on`.",
        "- Glass and non-glass quality can diverge. Glass ROIs include refraction-heavy transparent surfaces, while non-glass ROIs track ordinary robot/table/cube regions.",
        "- The PNG previews and contact sheets are generated from the same EXR outputs used for metrics, so visual comparison and numerical rows share provenance.",
        "",
    ]

    for scene, analysis in analyses.items():
        scene_dir = Path(scene)
        lines.extend([
            f"## {SCENE_TITLES[scene]}",
            "",
            "### Rendering Result",
            "",
            f"![{SCENE_TITLES[scene]} rendering result]({render_preview_relpath(scene).as_posix()})",
            "",
            "Visual preview for the scenario; timing and quality tables below use the same EXR render family.",
            "",
            "### ROI Boxes",
            "",
            markdown_table(["Group", "ROI", "Pixel box [x0, y0, x1, y1]"], _roi_box_table(scene)),
            "",
            "### Glass Vs Non-Glass Region Summary",
            "",
            "Metric bundles are ordered as `RMSE / PSNR / LPIPS / FLIP`. Lower is better for RMSE, LPIPS, and FLIP; higher is better for PSNR.",
            f"The complete raw per-run split remains in `{scene_dir}/analysis/region_quality_summary.csv`, and per-box values remain in `{scene_dir}/analysis/roi_quality_detailed.csv`.",
            "",
            "#### SPP Sweep Region Split",
            "",
            markdown_table(
                ["SPP", "Mean time", "Glass metrics", "Non-glass metrics", "Reading"],
                _spp_region_table(analysis["spp"]),
            ),
            "",
            "#### Adaptive Threshold Sensitivity By Region",
            "",
            "This table collapses the four adaptive thresholds into metric spreads at each SPP. Near-zero spreads mean the repeated threshold rows are numerically indistinguishable for the selected ROIs.",
            "",
            markdown_table(
                [
                    "SPP",
                    "Thresholds",
                    "Glass spread",
                    "Non-glass spread",
                    "Global RMSE range",
                    "Best threshold",
                    "Reading",
                ],
                _adaptive_region_spread_table(analysis["adaptive"]),
            ),
            "",
            "#### Glass Path Profile Region Split",
            "",
            markdown_table(
                ["Profile", "Mean time", "Glass metrics", "Non-glass metrics", "Reading"],
                _glass_region_table(analysis["glass"]),
            ),
            "",
            "### Human-Readable Comparison Sheets",
            "",
            "Normal comparison sheets show tone-mapped render previews at the original 1280x720 panel resolution. Difference sheets show mean absolute RGB EXR error against the retained `4096 spp, denoiser off` reference. The heatmap color scale is shared within each sheet, so colors are comparable inside one sheet but not across different sheets.",
            "",
            f"![SPP comparison]({scene_dir}/analysis/previews/spp_contact_sheet.png)",
            "",
            f"![SPP difference vs reference]({scene_dir}/analysis/previews/spp_difference_contact_sheet.png)",
            "",
            f"![Adaptive comparison]({scene_dir}/analysis/previews/adaptive_contact_sheet.png)",
            "",
            f"![Adaptive difference vs reference]({scene_dir}/analysis/previews/adaptive_difference_contact_sheet.png)",
            "",
            f"![Glass profile comparison]({scene_dir}/analysis/previews/glass_contact_sheet.png)",
            "",
            f"![Glass profile difference vs reference]({scene_dir}/analysis/previews/glass_difference_contact_sheet.png)",
            "",
            "### Key Figures",
            "",
            f"![Time to proxy quality]({scene_dir}/core/time_to_quality.png)",
            "",
            f"![SPP quality metrics]({scene_dir}/core/spp_quality_metrics.png)",
            "",
            f"![Glass/non-glass ROI quality]({scene_dir}/core/roi_quality.png)",
            "",
            f"![Adaptive sampling]({scene_dir}/controls/adaptive_sampling.png)",
            "",
            f"![Adaptive sampling quality metrics]({scene_dir}/controls/adaptive_sampling_quality.png)",
            "",
        ])
        if scene == "complex_glass_cube_reach":
            lines.extend([
                f"![Glass light paths]({scene_dir}/controls/glass_light_paths.png)",
                "",
                f"![Glass light-path quality metrics]({scene_dir}/controls/glass_light_paths_quality.png)",
                "",
            ])

        lines.extend([
            "### Interpretation",
            "",
            markdown_table(["Axis", "Measured result", "Interpretation"], _interpretation_rows(analysis)),
            "",
            "### SPP Timing And Proxy Quality",
            "",
            markdown_table(
                TIMING_HEADERS,
                _timing_table(analysis["spp"]),
            ),
            "",
            "### Reference Rows",
            "",
            markdown_table(
                TIMING_HEADERS,
                _timing_table(analysis["reference_rows"]),
            ),
            "",
        ])
        if analysis["adaptive"]:
            lines.extend([
                "### Adaptive Sampling Rows",
                "",
                markdown_table(
                    ["Max SPP", "Threshold", "Min samples", "Frames", "Mean time", "RMSE", "ROI max RMSE"],
                    _adaptive_table(analysis["adaptive"]),
                ),
                "",
            ])
        if analysis["glass"]:
            lines.extend([
                "### Glass Light-Path Rows",
                "",
                markdown_table(
                    [
                        "Profile",
                        "Max bounces",
                        "Transmission",
                        "Transparent",
                        "Caustics",
                        "Indirect clamp",
                        "Glossy filter",
                        "Mean time",
                        "RMSE",
                        "ROI max RMSE",
                    ],
                    _glass_table(analysis["glass"]),
                ),
                "",
            ])
        lines.extend([
            "### VRAM Summary",
            "",
            markdown_table(["Metric", "Value"], _vram_table(analysis)),
            "",
        ])
        boxes, grouping = _seed_stability_grouping(scene)
        if not (root / "matrix_driver.csv").exists():
            seed_md = ""
        else:
            try:
                seed_md = _seed_section.write_section(
                    root=root,
                    scene=scene,
                    reference_exr=_seed_stability_reference_for(scene),
                    roi_boxes=boxes,
                    roi_grouping=grouping,
                    runner=_seed_metrics.MetricRunner.full(),
                )
            except Exception as exc:
                seed_md = f"## Seed Stability\n\nSection generation failed: `{type(exc).__name__}: {exc}`. See logs for details.\n"
        if seed_md.strip():
            lines.append(seed_md)

    (root / "ANALYSIS.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_all(root: Path) -> None:
    analyses = {scene: scene_analysis(root, scene) for scene in SCENES}
    write_analysis_csvs(root, analyses)
    for analysis in analyses.values():
        plot_scene_figures(root, analysis)
        write_render_preview(root, analysis["scene"])
        write_comparison_previews(root, analysis)
    plot_root_figures(root, analyses)
    write_markdown(root, analyses)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write analysis report and meaningful figures for Blender benchmark outputs."
    )
    parser.add_argument("--root", type=Path, default=ROOT_DEFAULT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    write_all(args.root)
    print(args.root / "ANALYSIS.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
