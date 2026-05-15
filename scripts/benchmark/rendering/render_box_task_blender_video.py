#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.benchmark.rendering.box_task_replay import (
    DEFAULT_BUNDLE_ROOT,
    DEFAULT_SRC_TOTAL_FRAMES,
    BoxTaskBundlePaths,
    compute_output_frames,
    frame_to_source_indices,
    parse_scene_tokens,
    scene_frame_bounds,
)


def parse_vec3(text: str) -> tuple[float, float, float]:
    values = [part.strip() for part in text.split(",") if part.strip()]
    if len(values) != 3:
        raise argparse.ArgumentTypeError(f"Expected vec3 as x,y,z, got {text!r}")
    return float(values[0]), float(values[1]), float(values[2])


def positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"Expected a positive integer, got {text!r}")
    return value


def frame_path(frame_dir: Path, frame_index: int) -> Path:
    return frame_dir / f"frame_{frame_index:06d}.png"


def select_work_dir(*, bundle_root: Path, tmp_dir: Path | None, pid: int | None = None) -> Path:
    selected_pid = os.getpid() if pid is None else pid
    root = tmp_dir.expanduser().resolve() if tmp_dir is not None else bundle_root / ".tmp_box_task_blender"
    return root / f"render_work_{selected_pid}"


def ffmpeg_command(*, ffmpeg_bin: str, frame_dir: Path, fps: int, out_video: Path) -> list[str]:
    return [
        ffmpeg_bin,
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(frame_dir / "frame_%06d.png"),
        "-pix_fmt",
        "yuv420p",
        "-vcodec",
        "libx264",
        str(out_video),
    ]


def assemble_video(*, frame_dir: Path, fps: int, out_video: Path) -> None:
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        raise RuntimeError("ffmpeg not found in PATH")
    out_video.parent.mkdir(parents=True, exist_ok=True)
    command = ffmpeg_command(ffmpeg_bin=ffmpeg_bin, frame_dir=frame_dir, fps=fps, out_video=out_video)
    proc = subprocess.run(command, text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        message = "\n".join(part for part in (proc.stderr.strip(), proc.stdout.strip()) if part)
        raise RuntimeError(f"ffmpeg failed with exit code {proc.returncode}: {message[-1000:]}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render box-task replay video in Blender.")
    parser.add_argument("--bundle-root", type=Path, default=DEFAULT_BUNDLE_ROOT)
    parser.add_argument("--traj-path", type=Path, default=None)
    parser.add_argument("--src-total-frames", type=int, default=DEFAULT_SRC_TOTAL_FRAMES)
    parser.add_argument("--scenes", default="0021,0022,0024,0025,0031")
    parser.add_argument("--fps", type=positive_int, default=30)
    parser.add_argument("--width", type=positive_int, default=800)
    parser.add_argument("--height", type=positive_int, default=800)
    parser.add_argument("--samples", type=positive_int, default=64)
    parser.add_argument("--device", default="OPTIX")
    parser.add_argument("--duration-sec", type=float, default=None)
    parser.add_argument("--out-frames", type=positive_int, default=None)
    parser.add_argument("--camera-pos", type=parse_vec3, default=(1.45, -1.0, 0.95))
    parser.add_argument("--camera-look-at", type=parse_vec3, default=(0.55, 0.0, 0.38))
    parser.add_argument("--head-light-intensity", type=float, default=10000.0)
    parser.add_argument("--out-video", type=Path, required=True)
    parser.add_argument("--tmp-dir", type=Path, default=None)
    parser.add_argument("--keep-intermediates", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def build_plan(args: argparse.Namespace) -> dict[str, object]:
    paths = BoxTaskBundlePaths.from_root(args.bundle_root, traj_path=args.traj_path)
    scenes = parse_scene_tokens(args.scenes)
    out_frames = compute_output_frames(
        src_total_frames=args.src_total_frames,
        fps=args.fps,
        duration_sec=args.duration_sec,
        out_frames=args.out_frames,
    )
    source_indices = frame_to_source_indices(src_total_frames=args.src_total_frames, out_frames=out_frames)
    return {
        "status": "dry_run" if args.dry_run else "planned",
        "bundle_root": str(paths.bundle_root),
        "traj_path": str(paths.traj_path),
        "scenes": scenes,
        "out_frames": out_frames,
        "fps": args.fps,
        "width": args.width,
        "height": args.height,
        "samples": args.samples,
        "device": args.device,
        "scene_frame_bounds": [list(pair) for pair in scene_frame_bounds(out_frames=out_frames, scene_count=len(scenes))],
        "first_source_frame": int(source_indices[0]),
        "last_source_frame": int(source_indices[-1]),
        "out_video": str(args.out_video.expanduser().resolve()),
    }


def render_video(args: argparse.Namespace) -> int:
    raise NotImplementedError("Real rendering is added in the next task")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    plan = build_plan(args)
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    return render_video(args)


if __name__ == "__main__":
    raise SystemExit(main())
