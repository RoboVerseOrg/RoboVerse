from __future__ import annotations

import csv
import shlex
from pathlib import Path

import cv2
import numpy as np

from scripts.benchmark.rendering import seed_stability_metrics as ssm
from scripts.benchmark.rendering import write_seed_stability_section as section


def _write_constant_exr(path: Path, value: float, shape: tuple[int, int, int] = (8, 8, 3)) -> None:
    arr = np.full(shape, value, dtype=np.float32)
    bgr = arr[..., ::-1]
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), bgr)


def _write_runs_csv(scene_dir: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = list(rows[0].keys())
    scene_dir.mkdir(parents=True, exist_ok=True)
    with (scene_dir / "runs.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_matrix_driver_csv(root: Path, rows: list[dict[str, str]]) -> None:
    with (root / "matrix_driver.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["label", "command", "return_code", "elapsed_s", "status"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _command_with_seed(seed: int, samples: int, output_root: Path, scene: str = "complex_glass_cube_reach") -> str:
    args = [
        "/usr/bin/python",
        "render_benchmark.py",
        "render",
        "--scene",
        scene,
        "--samples",
        str(samples),
        "--denoiser",
        "off" if samples == 4096 else "on",
        "--seed",
        str(seed),
        "--output-root",
        str(output_root),
    ]
    return shlex.join(args)


def test_discover_seed_variance_runs_groups_by_setting(tmp_path) -> None:
    scene = "complex_glass_cube_reach"
    scene_dir = tmp_path / scene
    runs_rows = []
    driver_rows = []
    for samples in (64, 4096):
        for seed in range(2026, 2029):
            cmd = _command_with_seed(seed, samples, tmp_path)
            run_id = f"run_{samples}_{seed}"
            exr = scene_dir / "runs" / run_id / "overview_00" / "frame_000000.exr"
            _write_constant_exr(exr, 0.3)
            kind = "reference" if samples == 4096 else "spp_sweep"
            runs_rows.append({
                "scene": scene,
                "run_id": run_id,
                "kind": kind,
                "status": "ok",
                "samples": str(samples),
                "denoiser": "off" if samples == 4096 else "on",
                "seed": str(seed),
                "profile": "",
                "output_path": str(exr),
                "command": cmd,
            })
            label = f"{scene}:seed_variance:spp:{samples}spp:seed{seed}"
            driver_rows.append({"label": label, "command": cmd, "return_code": "0", "elapsed_s": "1.0", "status": "ok"})
    _write_runs_csv(scene_dir, runs_rows)
    _write_matrix_driver_csv(tmp_path, driver_rows)

    grouped = section.discover_seed_variance_runs(root=tmp_path, scene=scene)

    assert "spp_64spp" in grouped
    assert "spp_4096spp" in grouped
    assert len(grouped["spp_64spp"]) == 3
    assert len(grouped["spp_4096spp"]) == 3
    for entry in grouped["spp_64spp"]:
        assert entry["seed"] in {2026, 2027, 2028}
        assert entry["exr"].exists()


def test_discover_dedupes_same_setting_seed_pair(tmp_path) -> None:
    scene = "complex_glass_cube_reach"
    scene_dir = tmp_path / scene
    runs_rows = []
    driver_rows = []
    cmd = _command_with_seed(2026, 64, tmp_path)
    for run_index in range(2):
        run_id = f"r_dup_{run_index}"
        exr = scene_dir / "runs" / run_id / "overview_00" / "frame_000000.exr"
        _write_constant_exr(exr, 0.3)
        runs_rows.append({
            "scene": scene,
            "run_id": run_id,
            "kind": "spp_sweep",
            "status": "ok",
            "samples": "64",
            "denoiser": "on",
            "seed": "2026",
            "profile": "",
            "output_path": str(exr),
            "command": cmd,
        })
        driver_rows.append({
            "label": f"{scene}:seed_variance:spp:64spp:seed2026",
            "command": cmd,
            "return_code": "0",
            "elapsed_s": "1.0",
            "status": "ok",
        })
    _write_runs_csv(scene_dir, runs_rows)
    _write_matrix_driver_csv(tmp_path, driver_rows)

    grouped = section.discover_seed_variance_runs(root=tmp_path, scene=scene)
    assert len(grouped["spp_64spp"]) == 1
    assert grouped["spp_64spp"][0]["run_row"]["run_id"] == "r_dup_1"


def test_discover_skips_failed_rows(tmp_path) -> None:
    scene = "complex_glass_cube_reach"
    scene_dir = tmp_path / scene
    runs_rows = []
    driver_rows = []
    for seed in range(2026, 2029):
        cmd = _command_with_seed(seed, 64, tmp_path)
        run_id = f"run_{seed}"
        exr = scene_dir / "runs" / run_id / "overview_00" / "frame_000000.exr"
        if seed != 2027:
            _write_constant_exr(exr, 0.3)
        runs_rows.append({
            "scene": scene,
            "run_id": run_id,
            "kind": "spp_sweep",
            "status": "ok" if seed != 2027 else "failed",
            "samples": "64",
            "denoiser": "on",
            "seed": str(seed),
            "profile": "",
            "output_path": str(exr) if seed != 2027 else "",
            "command": cmd,
        })
        driver_rows.append({
            "label": f"{scene}:seed_variance:spp:64spp:seed{seed}",
            "command": cmd,
            "return_code": "0",
            "elapsed_s": "1.0",
            "status": "ok",
        })
    _write_runs_csv(scene_dir, runs_rows)
    _write_matrix_driver_csv(tmp_path, driver_rows)

    grouped = section.discover_seed_variance_runs(root=tmp_path, scene=scene)
    assert len(grouped["spp_64spp"]) == 2
    assert {entry["seed"] for entry in grouped["spp_64spp"]} == {2026, 2028}


def test_compute_setting_metrics_writes_pairwise_and_vs_reference_csvs(tmp_path) -> None:
    scene = "complex_glass_cube_reach"
    scene_dir = tmp_path / scene
    runs_rows = []
    driver_rows = []
    for seed in range(2026, 2030):
        cmd = _command_with_seed(seed, 64, tmp_path)
        run_id = f"r_{seed}"
        exr = scene_dir / "runs" / run_id / "overview_00" / "frame_000000.exr"
        rng = np.random.default_rng(seed)
        arr = np.full((8, 8, 3), 0.3, dtype=np.float32) + rng.normal(0, 0.005, (8, 8, 3)).astype(np.float32)
        exr.parent.mkdir(parents=True, exist_ok=True)
        assert cv2.imwrite(str(exr), arr[..., ::-1])
        runs_rows.append({
            "scene": scene,
            "run_id": run_id,
            "kind": "spp_sweep",
            "status": "ok",
            "samples": "64",
            "denoiser": "on",
            "seed": str(seed),
            "profile": "",
            "output_path": str(exr),
            "command": cmd,
        })
        driver_rows.append({
            "label": f"{scene}:seed_variance:spp:64spp:seed{seed}",
            "command": cmd,
            "return_code": "0",
            "elapsed_s": "1.0",
            "status": "ok",
        })
    _write_runs_csv(scene_dir, runs_rows)
    _write_matrix_driver_csv(tmp_path, driver_rows)

    reference = tmp_path / "ref.exr"
    _write_constant_exr(reference, 0.3)

    boxes = {"glass_a": [0, 0, 4, 4], "non_glass_a": [4, 4, 8, 8]}
    grouping = {"glass": ["glass_a"], "non_glass": ["non_glass_a"]}

    out_dir = section.compute_setting_metrics(
        root=tmp_path,
        scene=scene,
        reference_exr=reference,
        roi_boxes=boxes,
        roi_grouping=grouping,
        runner=ssm.MetricRunner.numerical_only(),
    )

    pairwise = list(csv.DictReader((out_dir / "pairwise_summary.csv").open(encoding="utf-8")))
    assert any(row["setting"] == "spp_64spp" for row in pairwise)
    vs_ref = list(csv.DictReader((out_dir / "vs_reference_summary.csv").open(encoding="utf-8")))
    assert len(vs_ref) == 4


def test_contact_sheet_collects_all_std_maps(tmp_path) -> None:
    out_dir = tmp_path / "complex_glass_cube_reach" / "analysis" / "seed_stability"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("spp_8spp", "spp_64spp", "glass_glass_default"):
        (out_dir / f"{name}_std_map.png").write_bytes(b"\x89PNG\r\n\x1a\n")  # placeholder
    sheet = section.write_contact_sheet(out_dir)
    assert sheet.exists()
    assert sheet.name == "seed_stability_contact_sheet.png"


def test_rmse_vs_spp_plot_writes_png(tmp_path) -> None:
    out = tmp_path / "rmse_vs_spp.png"
    pairs = {
        "spp_8spp": (0.020, 0.022),
        "spp_64spp": (0.005, 0.010),
        "spp_256spp": (0.002, 0.007),
        "spp_1024spp": (0.001, 0.005),
    }
    section.write_rmse_vs_spp_plot(pairs, out)
    assert out.exists()


def test_rmse_vs_spp_plot_handles_missing_vs_reference(tmp_path) -> None:
    out = tmp_path / "rmse_vs_spp.png"
    pairs = {"spp_8spp": (0.020, None), "spp_64spp": (0.005, None)}
    section.write_rmse_vs_spp_plot(pairs, out)
    assert out.exists()


def test_interpret_setting_classifies_noise_dominated() -> None:
    text = section.interpret_setting(setting_key="spp_8spp", pairwise_rmse=0.020, vs_reference_rmse=0.022)
    assert "noise" in text.lower() or "more samples" in text.lower()


def test_interpret_setting_classifies_structural() -> None:
    text = section.interpret_setting(setting_key="glass_glass_fast", pairwise_rmse=0.005, vs_reference_rmse=0.067)
    assert "structural" in text.lower()


def test_interpret_setting_handles_zero_reference() -> None:
    text = section.interpret_setting(setting_key="spp_4096spp", pairwise_rmse=0.001, vs_reference_rmse=0.0)
    assert text


def test_write_section_returns_markdown_with_three_tables_and_interpretation(tmp_path) -> None:
    scene = "complex_glass_cube_reach"
    scene_dir = tmp_path / scene
    runs_rows = []
    driver_rows = []
    for samples in (8, 64, 256):
        for seed in range(2026, 2029):
            cmd = _command_with_seed(seed, samples, tmp_path)
            run_id = f"r_{samples}_{seed}"
            exr = scene_dir / "runs" / run_id / "overview_00" / "frame_000000.exr"
            rng = np.random.default_rng(seed * samples)
            noise = 0.05 / np.sqrt(samples)
            arr = np.full((8, 8, 3), 0.3, dtype=np.float32) + rng.normal(0, noise, (8, 8, 3)).astype(np.float32)
            exr.parent.mkdir(parents=True, exist_ok=True)
            assert cv2.imwrite(str(exr), arr[..., ::-1])
            runs_rows.append({
                "scene": scene,
                "run_id": run_id,
                "kind": "spp_sweep",
                "status": "ok",
                "samples": str(samples),
                "denoiser": "on",
                "seed": str(seed),
                "profile": "",
                "output_path": str(exr),
                "command": cmd,
            })
            driver_rows.append({
                "label": f"{scene}:seed_variance:spp:{samples}spp:seed{seed}",
                "command": cmd,
                "return_code": "0",
                "elapsed_s": "1.0",
                "status": "ok",
            })
    _write_runs_csv(scene_dir, runs_rows)
    _write_matrix_driver_csv(tmp_path, driver_rows)

    reference = tmp_path / "ref.exr"
    _write_constant_exr(reference, 0.3)

    md = section.write_section(
        root=tmp_path,
        scene=scene,
        reference_exr=reference,
        roi_boxes=None,
        roi_grouping=None,
        runner=ssm.MetricRunner.numerical_only(),
    )
    assert "## Seed Stability" in md
    assert "Table 1" in md or "Pairwise" in md
    assert "Table 2" in md or "vs reference" in md.lower()
    assert "Table 3" in md or "Per-pixel std" in md
    assert "Interpretation" in md
    out_dir = scene_dir / "analysis" / "seed_stability"
    assert (out_dir / "seed_stability_contact_sheet.png").exists()
    assert (out_dir / "rmse_vs_spp.png").exists()


def test_write_section_handles_missing_reference(tmp_path) -> None:
    scene = "complex_glass_cube_reach"
    scene_dir = tmp_path / scene
    runs_rows = []
    driver_rows = []
    for seed in range(2026, 2029):
        cmd = _command_with_seed(seed, 64, tmp_path)
        run_id = f"r_{seed}"
        exr = scene_dir / "runs" / run_id / "overview_00" / "frame_000000.exr"
        _write_constant_exr(exr, 0.3)
        runs_rows.append({
            "scene": scene,
            "run_id": run_id,
            "kind": "spp_sweep",
            "status": "ok",
            "samples": "64",
            "denoiser": "on",
            "seed": str(seed),
            "profile": "",
            "output_path": str(exr),
            "command": cmd,
        })
        driver_rows.append({
            "label": f"{scene}:seed_variance:spp:64spp:seed{seed}",
            "command": cmd,
            "return_code": "0",
            "elapsed_s": "1.0",
            "status": "ok",
        })
    _write_runs_csv(scene_dir, runs_rows)
    _write_matrix_driver_csv(tmp_path, driver_rows)

    md = section.write_section(
        root=tmp_path,
        scene=scene,
        reference_exr=tmp_path / "missing.exr",
        roi_boxes=None,
        roi_grouping=None,
        runner=ssm.MetricRunner.numerical_only(),
    )
    assert "Seed Stability" in md
    assert "vs-reference block skipped" in md.lower() or "reference not available" in md.lower()


def test_write_section_skips_settings_with_fewer_than_three_seeds(tmp_path) -> None:
    scene = "complex_glass_cube_reach"
    scene_dir = tmp_path / scene
    runs_rows = []
    driver_rows = []
    for seed in (2026, 2027):
        cmd = _command_with_seed(seed, 64, tmp_path)
        run_id = f"r_{seed}"
        exr = scene_dir / "runs" / run_id / "overview_00" / "frame_000000.exr"
        _write_constant_exr(exr, 0.3)
        runs_rows.append({
            "scene": scene,
            "run_id": run_id,
            "kind": "spp_sweep",
            "status": "ok",
            "samples": "64",
            "denoiser": "on",
            "seed": str(seed),
            "profile": "",
            "output_path": str(exr),
            "command": cmd,
        })
        driver_rows.append({
            "label": f"{scene}:seed_variance:spp:64spp:seed{seed}",
            "command": cmd,
            "return_code": "0",
            "elapsed_s": "1.0",
            "status": "ok",
        })
    for seed in range(2026, 2029):
        cmd = _command_with_seed(seed, 256, tmp_path)
        run_id = f"r2_{seed}"
        exr = scene_dir / "runs" / run_id / "overview_00" / "frame_000000.exr"
        _write_constant_exr(exr, 0.3)
        runs_rows.append({
            "scene": scene,
            "run_id": run_id,
            "kind": "spp_sweep",
            "status": "ok",
            "samples": "256",
            "denoiser": "on",
            "seed": str(seed),
            "profile": "",
            "output_path": str(exr),
            "command": cmd,
        })
        driver_rows.append({
            "label": f"{scene}:seed_variance:spp:256spp:seed{seed}",
            "command": cmd,
            "return_code": "0",
            "elapsed_s": "1.0",
            "status": "ok",
        })
    _write_runs_csv(scene_dir, runs_rows)
    _write_matrix_driver_csv(tmp_path, driver_rows)

    md = section.write_section(
        root=tmp_path,
        scene=scene,
        reference_exr=None,
        roi_boxes=None,
        roi_grouping=None,
        runner=ssm.MetricRunner.numerical_only(),
    )
    assert "insufficient seeds" in md.lower() or "needs ≥ 3" in md.lower() or "skipped" in md.lower()
