"""Discover seed-variance runs, compute metrics, write CSVs/figures, return a markdown block."""

from __future__ import annotations

import csv
import re
import shlex
from pathlib import Path
from typing import Any

import numpy as np

from scripts.benchmark.rendering import seed_stability_metrics as ssm

_LABEL_RE = re.compile(
    r"^(?P<scene>[^:]+):seed_variance:(?P<group>spp|glass):(?P<key>[A-Za-z0-9_]+):seed(?P<seed>\d+)$"
)


def _parse_label(label: str) -> dict[str, str] | None:
    m = _LABEL_RE.match(label)
    if not m:
        return None
    return m.groupdict()


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _canon(cmd: str) -> str:
    try:
        return shlex.join(shlex.split(cmd))
    except ValueError:
        return cmd


def discover_seed_variance_runs(*, root: Path, scene: str) -> dict[str, list[dict[str, Any]]]:
    matrix_rows = _read_csv(root / "matrix_driver.csv")
    runs_rows = _read_csv(root / scene / "runs.csv")

    runs_by_command: dict[str, dict[str, str]] = {}
    for row in runs_rows:
        if row.get("status") != "ok":
            continue
        cmd = row.get("command", "")
        if cmd:
            runs_by_command[_canon(cmd)] = row

    by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for row in matrix_rows:
        if row.get("status") != "ok":
            continue
        parsed = _parse_label(row.get("label", ""))
        if not parsed or parsed["scene"] != scene:
            continue
        cmd = row.get("command", "")
        run_row = runs_by_command.get(_canon(cmd))
        if run_row is None:
            continue
        exr = run_row.get("output_path", "")
        if not exr:
            continue
        exr_path = Path(exr)
        if not exr_path.exists():
            continue
        setting_key = f"{parsed['group']}_{parsed['key']}"
        seed = int(parsed["seed"])
        by_key[(setting_key, seed)] = {
            "seed": seed,
            "exr": exr_path,
            "label": row["label"],
            "run_row": run_row,
        }

    grouped: dict[str, list[dict[str, Any]]] = {}
    for (setting_key, _seed), entry in by_key.items():
        grouped.setdefault(setting_key, []).append(entry)
    for entries in grouped.values():
        entries.sort(key=lambda e: e["seed"])
    return grouped


def compute_setting_metrics(
    *,
    root: Path,
    scene: str,
    reference_exr: Path | None,
    roi_boxes: dict[str, list[int]] | None,
    roi_grouping: dict[str, list[str]] | None,
    runner: ssm.MetricRunner,
) -> Path:
    grouped = discover_seed_variance_runs(root=root, scene=scene)
    out_dir = root / scene / "analysis" / "seed_stability"
    out_dir.mkdir(parents=True, exist_ok=True)

    reference_array: np.ndarray | None = None
    if reference_exr is not None and Path(reference_exr).exists():
        reference_array = ssm.load_exr_stack([Path(reference_exr)])[0]

    pairwise_rows: list[dict[str, Any]] = []
    vs_reference_rows: list[dict[str, Any]] = []
    std_rows: list[dict[str, Any]] = []
    skipped_settings: list[tuple[str, int]] = []

    for setting_key, entries in sorted(grouped.items()):
        if len(entries) < 3:
            skipped_settings.append((setting_key, len(entries)))
            continue
        exrs = [entry["exr"] for entry in entries]
        stack = ssm.load_exr_stack(exrs)

        pairwise = ssm.pairwise_metrics(stack, runner=runner, roi_boxes=roi_boxes, roi_grouping=roi_grouping)
        flat = {"setting": setting_key, "n_seeds": stack.shape[0]}
        for region, metrics in pairwise.items():
            for metric_name, summary in metrics.items():
                if summary["n_pairs"] == 0:
                    continue
                flat[f"{region}_{metric_name}_mean"] = summary["mean"]
                flat[f"{region}_{metric_name}_std"] = summary["std"]
                flat[f"{region}_{metric_name}_max"] = summary["max"]
        pairwise_rows.append(flat)

        if reference_array is not None and reference_array.shape == stack.shape[1:]:
            vs_ref = ssm.vs_reference_metrics(
                stack, reference_array, runner=runner, roi_boxes=roi_boxes, roi_grouping=roi_grouping
            )
            for r in vs_ref:
                vs_reference_rows.append({"setting": setting_key, **r})

        _, std_image = ssm.per_pixel_stack_stats(stack)
        std_path = out_dir / f"{setting_key}_std_map.png"
        max_intensity = ssm.render_std_map(std_image, std_path, color_scale=None)
        std_summary = {
            "setting": setting_key,
            "global_std_mean": float(np.mean(std_image)),
            "max_intensity": max_intensity,
            "std_map": std_path.name,
        }
        if roi_boxes and roi_grouping:
            per_roi: dict[str, float] = {}
            for name, box in roi_boxes.items():
                per_roi[name] = float(np.mean(ssm.crop(std_image, box)))
            grouped_std = ssm.group_means(per_roi, roi_grouping)
            for region, value in grouped_std.items():
                std_summary[f"{region}_std_mean"] = value
        std_rows.append(std_summary)

    _write_csv(out_dir / "pairwise_summary.csv", pairwise_rows)
    _write_csv(out_dir / "vs_reference_summary.csv", vs_reference_rows)
    _write_csv(out_dir / "std_summary.csv", std_rows)
    if skipped_settings:
        _write_csv(out_dir / "skipped_settings.csv", [{"setting": k, "n_seeds": n} for k, n in skipped_settings])
    return out_dir


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    field = sorted({k for row in rows for k in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=field)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in field})


def write_contact_sheet(out_dir: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    panels = sorted(out_dir.glob("*_std_map.png"))
    if not panels:
        target = out_dir / "seed_stability_contact_sheet.png"
        target.write_bytes(b"")
        return target
    cols = min(4, len(panels))
    rows = (len(panels) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.0, rows * 2.5), dpi=120, squeeze=False)
    for ax in axes.flat:
        ax.set_axis_off()
    for ax, panel in zip(axes.flat, panels):
        try:
            img = mpimg.imread(panel)
            ax.imshow(img)
        except Exception:
            ax.text(0.5, 0.5, panel.name, ha="center", va="center")
        ax.set_title(panel.stem.replace("_std_map", ""), fontsize=8)
    target = out_dir / "seed_stability_contact_sheet.png"
    fig.tight_layout()
    fig.savefig(target)
    plt.close(fig)
    return target


_SPP_KEY_RE = re.compile(r"^spp_(\d+)spp$")


def write_rmse_vs_spp_plot(pairs: dict[str, tuple[float, float | None]], out_path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    spp_pairs = []
    for setting, (pw, vr) in pairs.items():
        m = _SPP_KEY_RE.match(setting)
        if not m:
            continue
        spp_pairs.append((int(m.group(1)), pw, vr))
    spp_pairs.sort()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=120)
    if spp_pairs:
        spp_x = [s[0] for s in spp_pairs]
        ax.plot(spp_x, [s[1] for s in spp_pairs], marker="o", label="pairwise RMSE (precision)")
        vr_x = [s[0] for s in spp_pairs if s[2] is not None]
        vr_y = [s[2] for s in spp_pairs if s[2] is not None]
        if vr_x:
            ax.plot(vr_x, vr_y, marker="s", label="vs-reference RMSE (accuracy)")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("samples per pixel")
        ax.set_ylabel("linear-EXR RMSE")
        ax.set_title("Seed noise vs structural bias")
        ax.legend()
        ax.grid(True, which="both", linestyle=":", alpha=0.5)
    else:
        ax.text(0.5, 0.5, "no SPP rows", ha="center", va="center")
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def interpret_setting(*, setting_key: str, pairwise_rmse: float, vs_reference_rmse: float) -> str:
    if vs_reference_rmse <= 1e-9:
        return f"{setting_key}: vs-reference RMSE ≈ 0; the only residual is seed noise."
    ratio = pairwise_rmse / vs_reference_rmse
    if ratio > 0.8:
        verdict = "residual error is dominated by seed noise; more samples would help."
    elif ratio < 0.3:
        verdict = "residual error is structural; seed-averaging will not fix it."
    else:
        verdict = "mixed regime — both seed noise and structural bias contribute."
    return f"{setting_key}: pairwise/vs-ref RMSE ratio = {ratio:.2f}; {verdict}"


def _format_table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _fmt(value: Any, digits: int = 5) -> str:
    if value is None or value == "":
        return ""
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def write_section(
    *,
    root: Path,
    scene: str,
    reference_exr: Path | None,
    roi_boxes: dict[str, list[int]] | None,
    roi_grouping: dict[str, list[str]] | None,
    runner: ssm.MetricRunner,
) -> str:
    out_dir = compute_setting_metrics(
        root=root,
        scene=scene,
        reference_exr=reference_exr,
        roi_boxes=roi_boxes,
        roi_grouping=roi_grouping,
        runner=runner,
    )
    pairwise_rows = (
        list(csv.DictReader((out_dir / "pairwise_summary.csv").open(encoding="utf-8")))
        if (out_dir / "pairwise_summary.csv").read_text(encoding="utf-8").strip()
        else []
    )
    vs_ref_rows = (
        list(csv.DictReader((out_dir / "vs_reference_summary.csv").open(encoding="utf-8")))
        if (out_dir / "vs_reference_summary.csv").read_text(encoding="utf-8").strip()
        else []
    )
    std_rows = (
        list(csv.DictReader((out_dir / "std_summary.csv").open(encoding="utf-8")))
        if (out_dir / "std_summary.csv").read_text(encoding="utf-8").strip()
        else []
    )
    skipped_rows = (
        list(csv.DictReader((out_dir / "skipped_settings.csv").open(encoding="utf-8")))
        if (out_dir / "skipped_settings.csv").exists()
        else []
    )

    write_contact_sheet(out_dir)

    rmse_pairs: dict[str, tuple[float, float | None]] = {}
    for row in pairwise_rows:
        setting = row.get("setting", "")
        pw = float(row.get("global_rmse_mean", 0.0) or 0.0)
        vr_values = [float(v.get("rmse", 0.0) or 0.0) for v in vs_ref_rows if v.get("setting") == setting]
        vr_mean = sum(vr_values) / len(vr_values) if vr_values else None
        rmse_pairs[setting] = (pw, vr_mean)
    write_rmse_vs_spp_plot(rmse_pairs, out_dir / "rmse_vs_spp.png")

    md_lines = [
        "## Seed Stability",
        "",
        "Seed-stability tables decompose residual error into pairwise spread (precision), per-seed bias against the high-sample reference (accuracy), and per-pixel std (where in the image variance lives).",
        "",
        "### Table 1 — Pairwise stability across seeds",
        "",
    ]
    if pairwise_rows:
        cols = ["setting", "n_seeds", "global_rmse_mean", "global_psnr_mean", "global_lpips_mean", "global_flip_mean"]
        for region in ("glass", "non_glass"):
            if any(f"{region}_rmse_mean" in r for r in pairwise_rows):
                cols.append(f"{region}_rmse_mean")
        rows = [
            [_fmt(r.get(c), digits=5 if "rmse" in c or "lpips" in c or "flip" in c else 2) for c in cols]
            for r in pairwise_rows
        ]
        md_lines.append(_format_table(cols, rows))
    else:
        md_lines.append("(no settings with ≥ 3 seeds found)")
    md_lines.append("")

    md_lines.append("### Table 2 — Per-seed vs reference")
    md_lines.append("")
    if vs_ref_rows:
        cols = ["setting", "seed_index", "rmse", "psnr", "lpips", "flip"]
        rows = [[_fmt(r.get(c), digits=5 if c in ("rmse", "lpips", "flip") else 2) for c in cols] for r in vs_ref_rows]
        md_lines.append(_format_table(cols, rows))
    elif reference_exr is None or not Path(reference_exr).exists():
        md_lines.append("vs-reference block skipped (reference not available).")
    else:
        md_lines.append("(no rows)")
    md_lines.append("")

    md_lines.append("### Table 3 — Per-pixel std summary")
    md_lines.append("")
    if std_rows:
        cols = ["setting", "global_std_mean"]
        for region in ("glass", "non_glass"):
            if any(f"{region}_std_mean" in r for r in std_rows):
                cols.append(f"{region}_std_mean")
        cols.append("std_map")
        rows = [
            [_fmt(r.get(c), digits=5 if "std_mean" in c else 0) if c != "std_map" else r.get(c, "") for c in cols]
            for r in std_rows
        ]
        md_lines.append(_format_table(cols, rows))
        md_lines.append("")
        md_lines.append(f"![Seed-stability std maps]({scene}/analysis/seed_stability/seed_stability_contact_sheet.png)")
        md_lines.append("")
        md_lines.append(f"![Pairwise vs vs-reference RMSE]({scene}/analysis/seed_stability/rmse_vs_spp.png)")
    md_lines.append("")

    md_lines.append("### Interpretation")
    md_lines.append("")
    if pairwise_rows:
        for row in pairwise_rows:
            setting = row.get("setting", "")
            pw, vr = rmse_pairs.get(setting, (0.0, None))
            if vr is None:
                md_lines.append(f"- {setting}: vs-reference unavailable; pairwise RMSE = {pw:.5f}")
            else:
                md_lines.append(f"- {interpret_setting(setting_key=setting, pairwise_rmse=pw, vs_reference_rmse=vr)}")
    md_lines.append("")

    if skipped_rows:
        md_lines.append("### Notes")
        for row in skipped_rows:
            md_lines.append(f"- {row.get('setting', '')}: insufficient seeds (n={row.get('n_seeds', '?')}, needs ≥ 3)")
        md_lines.append("")

    return "\n".join(md_lines)
