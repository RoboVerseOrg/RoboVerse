"""Run the planned Blender rendering benchmark matrix serially.

Each benchmark row launches a fresh Python process so Blender/Cycles state and
persistent data do not leak across rows unless that row explicitly enables it.
"""

from __future__ import annotations

import argparse
import csv
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_ROOT = Path("outputs/rendering_benchmark/blender_2026-05-09_glass_only")
SCENES = ("complex_glass_cube_reach",)
SEED_VARIANCE_SEEDS: tuple[int, ...] = tuple(range(2026, 2036))
SEED_VARIANCE_SPP_LEVELS: tuple[int, ...] = (8, 16, 64, 256, 512, 1024, 4096)
SEED_VARIANCE_GLASS_PROFILES: tuple[str, ...] = (
    "glass_default",
    "glass_fast",
    "glass_reference",
    "caustics_off",
    "clamp_indirect_2",
)


@dataclass(frozen=True)
class MatrixRun:
    label: str
    args: list[str] = field(default_factory=list)


def _render_args(
    *,
    scene: str,
    samples: int,
    denoiser: str,
    device: str,
    camera_count: int = 1,
    frames: int = 1,
    warmup_frames: int = 0,
    seed: int = 0,
    output_root: Path,
    extra: list[str] | None = None,
) -> list[str]:
    args = [
        "render",
        "--scene",
        scene,
        "--samples",
        str(samples),
        "--denoiser",
        denoiser,
        "--device",
        device,
        "--camera-count",
        str(camera_count),
        "--frames",
        str(frames),
        "--warmup-frames",
        str(warmup_frames),
        "--seed",
        str(seed),
        "--output-root",
        str(output_root),
    ]
    if not extra or "--adaptive-sampling" not in extra:
        args[9:9] = ["--adaptive-sampling", "off"]
    if extra:
        args.extend(extra)
    return args


def _stability_args(
    *,
    scene: str,
    device: str,
    output_root: Path,
) -> list[str]:
    return [
        "stability",
        "--scene",
        scene,
        "--samples",
        "64",
        "--denoiser",
        "on",
        "--denoiser-engine",
        "OPTIX",
        "--denoiser-passes",
        "RGB_ALBEDO_NORMAL",
        "--adaptive-sampling",
        "off",
        "--device",
        device,
        "--camera-count",
        "1",
        "--repeats",
        "20",
        "--seed",
        "0",
        "--output-root",
        str(output_root),
    ]


def _glass_profile_args() -> dict[str, list[str]]:
    return {
        "glass_fast": [
            "--max-bounces",
            "4",
            "--diffuse-bounces",
            "2",
            "--glossy-bounces",
            "2",
            "--transmission-bounces",
            "4",
            "--transparent-bounces",
            "4",
            "--caustics-reflective",
            "off",
            "--caustics-refractive",
            "off",
            "--sample-clamp-indirect",
            "2.0",
            "--filter-glossy",
            "1.0",
        ],
        "glass_default": [
            "--max-bounces",
            "10",
            "--diffuse-bounces",
            "4",
            "--glossy-bounces",
            "4",
            "--transmission-bounces",
            "10",
            "--transparent-bounces",
            "10",
            "--caustics-reflective",
            "on",
            "--caustics-refractive",
            "on",
            "--sample-clamp-indirect",
            "10.0",
            "--filter-glossy",
            "1.0",
        ],
        "glass_reference": [
            "--max-bounces",
            "16",
            "--diffuse-bounces",
            "6",
            "--glossy-bounces",
            "8",
            "--transmission-bounces",
            "16",
            "--transparent-bounces",
            "16",
            "--caustics-reflective",
            "on",
            "--caustics-refractive",
            "on",
            "--sample-clamp-indirect",
            "0.0",
            "--filter-glossy",
            "0.0",
        ],
        "caustics_off": [
            "--max-bounces",
            "10",
            "--diffuse-bounces",
            "4",
            "--glossy-bounces",
            "4",
            "--transmission-bounces",
            "10",
            "--transparent-bounces",
            "10",
            "--caustics-reflective",
            "off",
            "--caustics-refractive",
            "off",
            "--sample-clamp-indirect",
            "10.0",
            "--filter-glossy",
            "1.0",
        ],
        "clamp_indirect_2": [
            "--max-bounces",
            "10",
            "--diffuse-bounces",
            "4",
            "--glossy-bounces",
            "4",
            "--transmission-bounces",
            "10",
            "--transparent-bounces",
            "10",
            "--caustics-reflective",
            "on",
            "--caustics-refractive",
            "on",
            "--sample-clamp-indirect",
            "2.0",
            "--filter-glossy",
            "1.0",
        ],
    }


def _seed_variance_rows(scene: str, output_root: Path, device: str) -> list[MatrixRun]:
    rows: list[MatrixRun] = []
    for samples in SEED_VARIANCE_SPP_LEVELS:
        denoiser = "off" if samples == 4096 else "on"
        extra: list[str] = []
        if denoiser == "on":
            extra = ["--denoiser-engine", "OPTIX", "--denoiser-passes", "RGB_ALBEDO_NORMAL"]
        for seed in SEED_VARIANCE_SEEDS:
            rows.append(
                MatrixRun(
                    f"{scene}:seed_variance:spp:{samples}spp:seed{seed}",
                    _render_args(
                        scene=scene,
                        samples=samples,
                        denoiser=denoiser,
                        device=device,
                        frames=1,
                        warmup_frames=0,
                        seed=seed,
                        output_root=output_root,
                        extra=extra,
                    ),
                )
            )
    for profile in SEED_VARIANCE_GLASS_PROFILES:
        profile_args = _glass_profile_args()[profile]
        for seed in SEED_VARIANCE_SEEDS:
            rows.append(
                MatrixRun(
                    f"{scene}:seed_variance:glass:{profile}:seed{seed}",
                    _render_args(
                        scene=scene,
                        samples=256,
                        denoiser="on",
                        device=device,
                        frames=1,
                        warmup_frames=0,
                        seed=seed,
                        output_root=output_root,
                        extra=[
                            "--denoiser-engine",
                            "OPTIX",
                            "--denoiser-passes",
                            "RGB_ALBEDO_NORMAL",
                            *profile_args,
                        ],
                    ),
                )
            )
    return rows


def build_matrix(scenes: list[str], output_root: Path, device: str, phase: str = "full") -> list[MatrixRun]:
    matrix: list[MatrixRun] = []
    for scene in scenes:
        if phase == "adaptive_low_spp":
            for samples in (64, 16, 8):
                matrix.append(
                    MatrixRun(
                        f"{scene}:spp_sweep:{samples}spp",
                        _render_args(
                            scene=scene,
                            samples=samples,
                            denoiser="on",
                            device=device,
                            frames=1,
                            warmup_frames=0,
                            seed=2026,
                            output_root=output_root,
                            extra=["--denoiser-engine", "OPTIX", "--denoiser-passes", "RGB_ALBEDO_NORMAL"],
                        ),
                    )
                )
            for samples in (64, 16, 8):
                for threshold in (0.1, 0.03, 0.01, 0.003):
                    matrix.append(
                        MatrixRun(
                            f"{scene}:adaptive:{samples}spp:{threshold}",
                            _render_args(
                                scene=scene,
                                samples=samples,
                                denoiser="on",
                                device=device,
                                frames=1,
                                warmup_frames=0,
                                seed=2026,
                                output_root=output_root,
                                extra=[
                                    "--denoiser-engine",
                                    "OPTIX",
                                    "--denoiser-passes",
                                    "RGB_ALBEDO_NORMAL",
                                    "--adaptive-sampling",
                                    "on",
                                    "--adaptive-threshold",
                                    str(threshold),
                                    "--adaptive-min-samples",
                                    "0",
                                ],
                            ),
                        )
                    )
            continue

        if phase == "seed_variance":
            matrix.extend(_seed_variance_rows(scene, output_root, device))
            continue

        if phase == "glass_quality":
            matrix.append(
                MatrixRun(
                    f"{scene}:reference:4096spp",
                    _render_args(
                        scene=scene,
                        samples=4096,
                        denoiser="off",
                        device=device,
                        frames=1,
                        warmup_frames=0,
                        seed=2026,
                        output_root=output_root,
                    ),
                )
            )
            for samples in (8, 16, 64, 256, 1024, 4096):
                matrix.append(
                    MatrixRun(
                        f"{scene}:spp_sweep:{samples}spp",
                        _render_args(
                            scene=scene,
                            samples=samples,
                            denoiser="on",
                            device=device,
                            frames=1,
                            warmup_frames=0,
                            seed=2026,
                            output_root=output_root,
                            extra=["--denoiser-engine", "OPTIX", "--denoiser-passes", "RGB_ALBEDO_NORMAL"],
                        ),
                    )
                )
            for samples in (64, 16, 8):
                for threshold in (0.1, 0.03, 0.01, 0.003):
                    matrix.append(
                        MatrixRun(
                            f"{scene}:adaptive:{samples}spp:{threshold}",
                            _render_args(
                                scene=scene,
                                samples=samples,
                                denoiser="on",
                                device=device,
                                frames=1,
                                warmup_frames=0,
                                seed=2026,
                                output_root=output_root,
                                extra=[
                                    "--denoiser-engine",
                                    "OPTIX",
                                    "--denoiser-passes",
                                    "RGB_ALBEDO_NORMAL",
                                    "--adaptive-sampling",
                                    "on",
                                    "--adaptive-threshold",
                                    str(threshold),
                                    "--adaptive-min-samples",
                                    "0",
                                ],
                            ),
                        )
                    )
            profiles = {
                "glass_fast": [
                    "--max-bounces",
                    "4",
                    "--diffuse-bounces",
                    "2",
                    "--glossy-bounces",
                    "2",
                    "--transmission-bounces",
                    "4",
                    "--transparent-bounces",
                    "4",
                    "--caustics-reflective",
                    "off",
                    "--caustics-refractive",
                    "off",
                    "--sample-clamp-indirect",
                    "2.0",
                    "--filter-glossy",
                    "1.0",
                ],
                "glass_default": [
                    "--max-bounces",
                    "10",
                    "--diffuse-bounces",
                    "4",
                    "--glossy-bounces",
                    "4",
                    "--transmission-bounces",
                    "10",
                    "--transparent-bounces",
                    "10",
                    "--caustics-reflective",
                    "on",
                    "--caustics-refractive",
                    "on",
                    "--sample-clamp-indirect",
                    "10.0",
                    "--filter-glossy",
                    "1.0",
                ],
                "glass_reference": [
                    "--max-bounces",
                    "16",
                    "--diffuse-bounces",
                    "6",
                    "--glossy-bounces",
                    "8",
                    "--transmission-bounces",
                    "16",
                    "--transparent-bounces",
                    "16",
                    "--caustics-reflective",
                    "on",
                    "--caustics-refractive",
                    "on",
                    "--sample-clamp-indirect",
                    "0.0",
                    "--filter-glossy",
                    "0.0",
                ],
                "caustics_off": [
                    "--max-bounces",
                    "10",
                    "--diffuse-bounces",
                    "4",
                    "--glossy-bounces",
                    "4",
                    "--transmission-bounces",
                    "10",
                    "--transparent-bounces",
                    "10",
                    "--caustics-reflective",
                    "off",
                    "--caustics-refractive",
                    "off",
                    "--sample-clamp-indirect",
                    "10.0",
                    "--filter-glossy",
                    "1.0",
                ],
                "clamp_indirect_2": [
                    "--max-bounces",
                    "10",
                    "--diffuse-bounces",
                    "4",
                    "--glossy-bounces",
                    "4",
                    "--transmission-bounces",
                    "10",
                    "--transparent-bounces",
                    "10",
                    "--caustics-reflective",
                    "on",
                    "--caustics-refractive",
                    "on",
                    "--sample-clamp-indirect",
                    "2.0",
                    "--filter-glossy",
                    "1.0",
                ],
            }
            for profile, profile_args in profiles.items():
                matrix.append(
                    MatrixRun(
                        f"{scene}:glass:{profile}",
                        _render_args(
                            scene=scene,
                            samples=256,
                            denoiser="on",
                            device=device,
                            frames=1,
                            warmup_frames=0,
                            seed=2026,
                            output_root=output_root,
                            extra=[
                                "--denoiser-engine",
                                "OPTIX",
                                "--denoiser-passes",
                                "RGB_ALBEDO_NORMAL",
                                *profile_args,
                            ],
                        ),
                    )
                )
            continue

        if phase == "full":
            for samples in (4096, 8192):
                matrix.append(
                    MatrixRun(
                        f"{scene}:reference:{samples}spp",
                        _render_args(
                            scene=scene,
                            samples=samples,
                            denoiser="off",
                            device=device,
                            frames=1,
                            warmup_frames=1,
                            seed=0,
                            output_root=output_root,
                        ),
                    )
                )

        for samples in (1, 4, 16, 64, 256, 1024, 4096):
            matrix.append(
                MatrixRun(
                    f"{scene}:spp_sweep:{samples}spp",
                    _render_args(
                        scene=scene,
                        samples=samples,
                        denoiser="on",
                        device=device,
                        frames=3,
                        warmup_frames=1,
                        seed=0,
                        output_root=output_root,
                        extra=["--denoiser-engine", "OPTIX", "--denoiser-passes", "RGB_ALBEDO_NORMAL"],
                    ),
                )
            )

        if phase == "full":
            matrix.append(
                MatrixRun(
                    f"{scene}:denoiser:off",
                    _render_args(
                        scene=scene,
                        samples=64,
                        denoiser="off",
                        device=device,
                        frames=5,
                        warmup_frames=1,
                        seed=0,
                        output_root=output_root,
                    ),
                )
            )
            for engine in ("AUTOMATIC", "OPENIMAGEDENOISE", "OPTIX"):
                for passes in ("RGB", "RGB_ALBEDO", "RGB_ALBEDO_NORMAL"):
                    matrix.append(
                        MatrixRun(
                            f"{scene}:denoiser:{engine}:{passes}",
                            _render_args(
                                scene=scene,
                                samples=64,
                                denoiser="on",
                                device=device,
                                frames=5,
                                warmup_frames=1,
                                seed=0,
                                output_root=output_root,
                                extra=[
                                    "--denoiser-engine",
                                    engine,
                                    "--denoiser-passes",
                                    passes,
                                    "--denoising-prefilter",
                                    "ACCURATE",
                                    "--denoising-quality",
                                    "HIGH",
                                ],
                            ),
                        )
                    )

        for threshold in (0.1, 0.03, 0.01, 0.003, 0.001):
            matrix.append(
                MatrixRun(
                    f"{scene}:adaptive:{threshold}",
                    _render_args(
                        scene=scene,
                        samples=4096,
                        denoiser="on",
                        device=device,
                        frames=5,
                        warmup_frames=1,
                        seed=0,
                        output_root=output_root,
                        extra=[
                            "--denoiser-engine",
                            "OPTIX",
                            "--denoiser-passes",
                            "RGB_ALBEDO_NORMAL",
                            "--adaptive-sampling",
                            "on",
                            "--adaptive-threshold",
                            str(threshold),
                            "--adaptive-min-samples",
                            "0",
                        ],
                    ),
                )
            )

        if scene == "complex_glass_cube_reach":
            profiles = {
                "glass_fast": [
                    "--max-bounces",
                    "4",
                    "--diffuse-bounces",
                    "2",
                    "--glossy-bounces",
                    "2",
                    "--transmission-bounces",
                    "4",
                    "--transparent-bounces",
                    "4",
                    "--caustics-reflective",
                    "off",
                    "--caustics-refractive",
                    "off",
                    "--sample-clamp-indirect",
                    "2.0",
                    "--filter-glossy",
                    "1.0",
                ],
                "glass_default": [
                    "--max-bounces",
                    "10",
                    "--diffuse-bounces",
                    "4",
                    "--glossy-bounces",
                    "4",
                    "--transmission-bounces",
                    "10",
                    "--transparent-bounces",
                    "10",
                    "--caustics-reflective",
                    "on",
                    "--caustics-refractive",
                    "on",
                    "--sample-clamp-indirect",
                    "10.0",
                    "--filter-glossy",
                    "1.0",
                ],
                "glass_reference": [
                    "--max-bounces",
                    "16",
                    "--diffuse-bounces",
                    "6",
                    "--glossy-bounces",
                    "8",
                    "--transmission-bounces",
                    "16",
                    "--transparent-bounces",
                    "16",
                    "--caustics-reflective",
                    "on",
                    "--caustics-refractive",
                    "on",
                    "--sample-clamp-indirect",
                    "0.0",
                    "--filter-glossy",
                    "0.0",
                ],
                "caustics_off": [
                    "--max-bounces",
                    "10",
                    "--diffuse-bounces",
                    "4",
                    "--glossy-bounces",
                    "4",
                    "--transmission-bounces",
                    "10",
                    "--transparent-bounces",
                    "10",
                    "--caustics-reflective",
                    "off",
                    "--caustics-refractive",
                    "off",
                    "--sample-clamp-indirect",
                    "10.0",
                    "--filter-glossy",
                    "1.0",
                ],
                "clamp_indirect_2": [
                    "--max-bounces",
                    "10",
                    "--diffuse-bounces",
                    "4",
                    "--glossy-bounces",
                    "4",
                    "--transmission-bounces",
                    "10",
                    "--transparent-bounces",
                    "10",
                    "--caustics-reflective",
                    "on",
                    "--caustics-refractive",
                    "on",
                    "--sample-clamp-indirect",
                    "2.0",
                    "--filter-glossy",
                    "1.0",
                ],
            }
            for profile, profile_args in profiles.items():
                matrix.append(
                    MatrixRun(
                        f"{scene}:glass:{profile}",
                        _render_args(
                            scene=scene,
                            samples=256,
                            denoiser="on",
                            device=device,
                            frames=5,
                            warmup_frames=1,
                            seed=0,
                            output_root=output_root,
                            extra=[
                                "--denoiser-engine",
                                "OPTIX",
                                "--denoiser-passes",
                                "RGB_ALBEDO_NORMAL",
                                *profile_args,
                            ],
                        ),
                    )
                )

        for light_tree in ("on", "off"):
            for threshold in (0.0, 0.001, 0.01, 0.05):
                matrix.append(
                    MatrixRun(
                        f"{scene}:light:{light_tree}:{threshold}",
                        _render_args(
                            scene=scene,
                            samples=256,
                            denoiser="on",
                            device=device,
                            frames=5,
                            warmup_frames=1,
                            seed=0,
                            output_root=output_root,
                            extra=[
                                "--denoiser-engine",
                                "OPTIX",
                                "--denoiser-passes",
                                "RGB_ALBEDO_NORMAL",
                                "--light-tree",
                                light_tree,
                                "--light-threshold",
                                str(threshold),
                            ],
                        ),
                    )
                )

        for cameras in (1, 2, 4, 8, 16):
            matrix.append(
                MatrixRun(
                    f"{scene}:multi_camera:{cameras}",
                    _render_args(
                        scene=scene,
                        samples=64,
                        denoiser="on",
                        device=device,
                        camera_count=cameras,
                        frames=100,
                        warmup_frames=10,
                        seed=0,
                        output_root=output_root,
                        extra=["--denoiser-engine", "OPTIX", "--denoiser-passes", "RGB_ALBEDO_NORMAL"],
                    ),
                )
            )

        matrix.append(
            MatrixRun(f"{scene}:stability", _stability_args(scene=scene, device=device, output_root=output_root))
        )

        for persistent in ("on", "off"):
            matrix.append(
                MatrixRun(
                    f"{scene}:persistent:{persistent}",
                    _render_args(
                        scene=scene,
                        samples=64,
                        denoiser="on",
                        device=device,
                        frames=1000,
                        warmup_frames=50,
                        seed=0,
                        output_root=output_root,
                        extra=[
                            "--denoiser-engine",
                            "OPTIX",
                            "--denoiser-passes",
                            "RGB_ALBEDO_NORMAL",
                            "--persistent-data",
                            persistent,
                        ],
                    ),
                )
            )
    return matrix


def _arg_value(args: list[str], name: str) -> str:
    with_value = f"{name}="
    for index, arg in enumerate(args):
        if arg == name and index + 1 < len(args):
            return args[index + 1]
        if arg.startswith(with_value):
            return arg[len(with_value) :]
    return ""


def _latest_child_status(output_root: Path, command: list[str], args: list[str], fallback: str) -> str:
    scene = _arg_value(args, "--scene")
    if not scene:
        return fallback
    runs_path = output_root / scene / "runs.csv"
    if not runs_path.exists():
        return fallback

    command_text = shlex.join(command)
    with runs_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in reversed(rows):
        if row.get("command") == command_text and row.get("status"):
            return str(row["status"])
    return fallback


def run_matrix(matrix: list[MatrixRun], output_root: Path) -> int:
    script = Path(__file__).with_name("blender_render_benchmark.py")
    log_path = output_root / "matrix_driver.log"
    csv_path = output_root / "matrix_driver.csv"
    output_root.mkdir(parents=True, exist_ok=True)
    csv_exists = csv_path.exists()
    failures = 0
    with log_path.open("a", encoding="utf-8") as log_file, csv_path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["label", "command", "return_code", "elapsed_s", "status"])
        if not csv_exists:
            writer.writeheader()
        for index, run in enumerate(matrix, 1):
            command = [sys.executable, str(script), *run.args]
            print(f"[{index}/{len(matrix)}] {run.label}", flush=True)
            log_file.write(f"\n=== {index}/{len(matrix)} {run.label} ===\n")
            log_file.write(" ".join(command) + "\n")
            log_file.flush()
            start = time.perf_counter()
            proc = subprocess.run(command, stdout=log_file, stderr=subprocess.STDOUT, text=True, check=False)
            elapsed_s = time.perf_counter() - start
            fallback_status = "ok" if proc.returncode == 0 else "failed"
            status = _latest_child_status(output_root, command, run.args, fallback_status)
            if proc.returncode != 0 or status == "failed":
                failures += 1
            writer.writerow({
                "label": run.label,
                "command": " ".join(command),
                "return_code": proc.returncode,
                "elapsed_s": f"{elapsed_s:.3f}",
                "status": status,
            })
            csv_file.flush()
            print(f"[{index}/{len(matrix)}] {run.label}: {status} ({elapsed_s:.1f}s)", flush=True)
    return 1 if failures else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the planned Blender benchmark matrix.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--device", default="OPTIX")
    parser.add_argument("--scene", choices=(*SCENES, "all"), default="all")
    parser.add_argument(
        "--phase",
        choices=("full", "practical", "glass_quality", "adaptive_low_spp", "seed_variance"),
        default="full",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scenes = list(SCENES if args.scene == "all" else (args.scene,))
    matrix = build_matrix(scenes, args.output_root, args.device, phase=args.phase)
    raise SystemExit(run_matrix(matrix, args.output_root))


if __name__ == "__main__":
    main()
