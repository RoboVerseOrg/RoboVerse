#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.benchmark.rendering.box_task_replay import (
    DEFAULT_BUNDLE_ROOT,
    DEFAULT_SRC_TOTAL_FRAMES,
    BoxTaskBundlePaths,
    build_box_task_scenario,
    compute_output_frames,
    decode_trajectory_tensor_states,
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
    paths = BoxTaskBundlePaths.from_root(args.bundle_root, traj_path=args.traj_path)
    scenes = parse_scene_tokens(args.scenes)
    out_frames = compute_output_frames(
        src_total_frames=args.src_total_frames,
        fps=args.fps,
        duration_sec=args.duration_sec,
        out_frames=args.out_frames,
    )
    source_indices = frame_to_source_indices(src_total_frames=args.src_total_frames, out_frames=out_frames)
    out_video = args.out_video.expanduser().resolve()
    work_dir = select_work_dir(bundle_root=paths.bundle_root, tmp_dir=args.tmp_dir)
    frame_dir = work_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)

    try:
        render_frames(
            args=args,
            paths=paths,
            scenes=scenes,
            frame_to_src=source_indices,
            frame_dir=frame_dir,
        )
        assemble_video(frame_dir=frame_dir, fps=args.fps, out_video=out_video)
    finally:
        if not args.keep_intermediates:
            shutil.rmtree(work_dir, ignore_errors=True)

    print(json.dumps({"status": "ok", "out_video": str(out_video), "frames": out_frames}))
    return 0


def render_frames(
    *,
    args: argparse.Namespace,
    paths: BoxTaskBundlePaths,
    scenes: Sequence[str],
    frame_to_src,
    frame_dir: Path,
) -> None:
    unique_source_indices = sorted({int(index) for index in frame_to_src})
    tensor_states_by_source = decode_trajectory_tensor_states(
        paths.traj_path,
        unique_source_indices,
        bundle_paths=paths,
    )

    for scene, (start_frame, end_frame) in zip(
        scenes,
        scene_frame_bounds(out_frames=len(frame_to_src), scene_count=len(scenes)),
        strict=True,
    ):
        scenario = build_box_task_scenario(
            paths=paths,
            simulator="blender",
            scene=scene,
            width=args.width,
            height=args.height,
            camera_pos=args.camera_pos,
            camera_look_at=args.camera_look_at,
            head_light_intensity=args.head_light_intensity,
        )
        render_scene_segment(
            scenario=scenario,
            tensor_states_by_src=tensor_states_by_source,
            frame_to_src=frame_to_src,
            frame_dir=frame_dir,
            out_start=start_frame,
            out_end=end_frame,
            samples=args.samples,
            device=args.device,
        )


def render_scene_segment(
    *,
    scenario: Any,
    tensor_states_by_src: dict[int, Any],
    frame_to_src,
    frame_dir: Path,
    out_start: int,
    out_end: int,
    samples: int,
    device: str,
) -> None:
    from metasim.sim.blender.blender import BlenderHandler

    handler = BlenderHandler(scenario)
    try:
        handler.launch()
        configure_blender_for_video(samples=samples, device=device)
        camera = scenario.cameras[0]
        for output_frame in range(out_start, out_end):
            source_index = int(frame_to_src[output_frame])
            apply_state_to_handler(handler, tensor_states_by_src[source_index])
            render_png_frame(handler=handler, camera=camera, path=frame_path(frame_dir, output_frame))
    finally:
        handler.close()


def configure_blender_for_video(*, samples: int, device: str) -> None:
    import bpy

    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = int(samples)
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0

    requested_device = device.upper()
    if requested_device == "CPU":
        scene.cycles.device = "CPU"
        return

    scene.cycles.device = "GPU"
    preferences = bpy.context.preferences.addons["cycles"].preferences
    device_types = ("OPTIX", "CUDA") if requested_device == "AUTO" else (requested_device,)
    last_error: Exception | None = None
    for device_type in device_types:
        try:
            preferences.compute_device_type = device_type
            devices = preferences.get_devices()
            _enable_cycles_devices(preferences, devices)
            return
        except Exception as exc:  # pragma: no cover - depends on Blender build/device support
            last_error = exc
    if last_error is not None:
        raise last_error


def _enable_cycles_devices(preferences: Any, devices: Any = None) -> None:
    if devices is None:
        devices = getattr(preferences, "devices", ())
    for cycles_device in devices:
        if isinstance(cycles_device, (list, tuple)):
            _enable_cycles_devices(preferences, cycles_device)
        else:
            cycles_device.use = True


def apply_state_to_handler(handler: Any, state: Any) -> None:
    try:
        handler.set_states(state)
    except TypeError:
        handler.set_states([state])
    refresh_render = getattr(handler, "refresh_render", None)
    if callable(refresh_render):
        refresh_render()


def camera_object_for(handler: Any, camera: Any) -> Any:
    camera_name = getattr(camera, "name", camera)
    for attr_name in ("camera_objs", "_camera_objs"):
        camera_objs = getattr(handler, attr_name, None)
        if camera_objs is not None and camera_name in camera_objs:
            return camera_objs[camera_name]
    raise KeyError(f"Camera object not found: {camera_name}")


def render_png_frame(*, handler: Any, camera: Any, path: Path) -> None:
    import bpy

    path.parent.mkdir(parents=True, exist_ok=True)
    scene = bpy.context.scene
    scene.camera = camera_object_for(handler, camera)
    scene.render.resolution_x = int(camera.width)
    scene.render.resolution_y = int(camera.height)
    scene.render.resolution_percentage = 100
    scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    plan = build_plan(args)
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    return render_video(args)


if __name__ == "__main__":
    raise SystemExit(main())
