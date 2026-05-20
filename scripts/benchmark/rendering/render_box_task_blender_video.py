#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
from contextlib import contextmanager, suppress
from copy import deepcopy
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
    load_decoded_tensor_states,
    parse_scene_tokens,
    save_decoded_tensor_states,
    scene_frame_bounds,
)

KUJIALE_WALL_MATERIALS = (
    "Wall",
    "MI_Stucco_Facade_wfnhfaq_2K",
)
KUJIALE_TV_MATERIALS = (
    "MI_554af95fe4b04227815a8fbd",
    "MI_554af95fe4b04227815a8fbe",
)
KUJIALE_CABINET_MATERIALS = (
    "MI_5a2788380c1ed46426c0f290",
    "MI_5a2788380c1ed46426c0f293",
    "MI_5a2788380c1ed46426c0f296",
    "MI_5a2788380c1ed46426c0f297",
    "MI_5a2788380c1ed46426c0f299",
    "MI_5a2788380c1ed46426c0f29b",
    "MI_5a2788380c1ed46426c0f29c",
    "MI_5a2788380c1ed46426c0f29d",
    "MI_5a2788380c1ed46426c0f29f",
    "MI_5a2788380c1ed46426c0f2a2",
    "MI_5a2788380c1ed46426c0f2a3",
    "MI_5a2788380c1ed46426c0f2a5",
    "MI_5a2788380c1ed46426c0f2a6",
    "MI_5a2788380c1ed46426c0f2a7",
    "MI_5a2788380c1ed46426c0f2a8",
    "MI_5a2788390c1ed46426c0f2a9",
)
KUJIALE_MATERIAL_OVERRIDES: dict[str, tuple[float, float, float, float]] = {
    **{name: (0.86, 0.84, 0.81, 1.0) for name in KUJIALE_WALL_MATERIALS},
    **{name: (0.02, 0.02, 0.02, 1.0) for name in KUJIALE_TV_MATERIALS},
    **{name: (0.78, 0.72, 0.64, 1.0) for name in KUJIALE_CABINET_MATERIALS},
    "Wood_05": (0.50, 0.38, 0.26, 1.0),
    "MI_Chevron_Walnut_Parquet_th4hfcfl_2K": (0.50, 0.38, 0.26, 1.0),
}


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
    parser.add_argument("--head-light-intensity", type=float, default=400.0)
    parser.add_argument("--exposure", type=float, default=-2.0)
    parser.add_argument(
        "--scene-task-offset",
        default="none",
        help="Task-frame translation: 'auto' uses -scenario.scene.default_position, 'none' keeps origin, or x,y,z.",
    )
    parser.add_argument(
        "--scene-import-offset",
        default="auto",
        help="Blender scene-root translation after import: 'auto' uses scenario.scene.default_position, 'none' disables, or x,y,z.",
    )
    parser.add_argument(
        "--no-clear-task-volume",
        dest="clear_task_volume",
        action="store_false",
        help="Do not hide imported scene meshes that intersect the replay workspace.",
    )
    parser.set_defaults(clear_task_volume=True)
    parser.add_argument("--out-video", type=Path, default=None)
    parser.add_argument("--tmp-dir", type=Path, default=None)
    parser.add_argument("--keep-intermediates", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--decode-source-indices-json", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--decode-states-out", type=Path, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    is_decode_worker = args.decode_source_indices_json is not None or args.decode_states_out is not None
    if is_decode_worker and (args.decode_source_indices_json is None or args.decode_states_out is None):
        parser.error("--decode-source-indices-json and --decode-states-out must be used together")
    if not is_decode_worker and args.out_video is None:
        parser.error("--out-video is required unless decoding states")
    return args


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
        "head_light_intensity": args.head_light_intensity,
        "exposure": args.exposure,
        "scene_task_offset": args.scene_task_offset,
        "scene_import_offset": args.scene_import_offset,
        "clear_task_volume": args.clear_task_volume,
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
    decoded_state_path = frame_dir.parent / "decoded_tensor_states.pt"
    decode_trajectory_tensor_states_subprocess(
        paths=paths,
        source_indices=unique_source_indices,
        state_path=decoded_state_path,
    )
    tensor_states_by_source = load_decoded_tensor_states(decoded_state_path)

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
        state_offset = resolve_scene_task_offset(getattr(args, "scene_task_offset", "auto"), scenario)
        scene_import_offset = resolve_scene_import_offset(getattr(args, "scene_import_offset", "auto"), scenario)
        apply_scene_task_offset_to_scenario(scenario, state_offset)
        render_scene_segment(
            scenario=scenario,
            tensor_states_by_src=tensor_states_by_source,
            frame_to_src=frame_to_src,
            frame_dir=frame_dir,
            out_start=start_frame,
            out_end=end_frame,
            samples=args.samples,
            device=args.device,
            exposure=getattr(args, "exposure", -2.0),
            state_offset=state_offset,
            scene_import_offset=scene_import_offset,
            clear_task_volume=bool(getattr(args, "clear_task_volume", True)),
        )


def decode_trajectory_tensor_states_subprocess(
    *,
    paths: BoxTaskBundlePaths,
    source_indices: Sequence[int],
    state_path: Path,
    python_bin: str = sys.executable,
    script_path: Path | None = None,
) -> None:
    worker_script = Path(__file__).resolve() if script_path is None else script_path
    command = [
        python_bin,
        str(worker_script),
        "--bundle-root",
        str(paths.bundle_root),
        "--traj-path",
        str(paths.traj_path),
        "--decode-source-indices-json",
        json.dumps([int(index) for index in source_indices]),
        "--decode-states-out",
        str(state_path),
    ]
    proc = subprocess.run(command, text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        message = "\n".join(part for part in (proc.stderr.strip(), proc.stdout.strip()) if part)
        raise RuntimeError(f"decode worker failed with exit code {proc.returncode}: {message[-1000:]}")


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
    exposure: float = -2.0,
    state_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    scene_import_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    clear_task_volume: bool = True,
) -> None:
    from metasim.sim.blender.blender import BlenderHandler

    with local_bundle_render_runtime():
        handler = BlenderHandler(scenario)
        try:
            handler.launch()
            apply_scene_import_offset_to_handler(handler, scene_import_offset)
            if clear_task_volume:
                clearance = clear_task_volume_for_replay(handler, center=state_offset)
                with suppress(Exception):
                    setattr(scenario, "_box_task_volume_clearance", clearance)
            material_repair = repair_imported_scene_materials()
            with suppress(Exception):
                setattr(scenario, "_box_task_room_material_repair", material_repair)
            configure_blender_for_video(samples=samples, device=device, exposure=exposure)
            camera = scenario.cameras[0]
            for output_frame in range(out_start, out_end):
                source_index = int(frame_to_src[output_frame])
                state = translate_tensor_state(tensor_states_by_src[source_index], state_offset)
                apply_state_to_handler(handler, state)
                render_png_frame(handler=handler, camera=camera, path=frame_path(frame_dir, output_frame))
        finally:
            handler.close()


def resolve_scene_task_offset(offset_text: str, scenario: Any) -> tuple[float, float, float]:
    normalized = offset_text.strip().lower()
    if normalized in {"none", "off", "false", "0"}:
        return 0.0, 0.0, 0.0
    if normalized != "auto":
        return parse_vec3(offset_text)

    scene_cfg = getattr(scenario, "scene", None)
    default_position = getattr(scene_cfg, "default_position", None)
    if default_position is None:
        return 0.0, 0.0, 0.0
    values = tuple(float(value) for value in default_position)
    if len(values) != 3:
        raise ValueError(f"Scene default_position must have 3 values, got {values!r}")
    return -values[0], -values[1], -values[2]


def resolve_scene_import_offset(offset_text: str, scenario: Any) -> tuple[float, float, float]:
    normalized = offset_text.strip().lower()
    if normalized in {"none", "off", "false", "0"}:
        return 0.0, 0.0, 0.0
    if normalized != "auto":
        return parse_vec3(offset_text)

    scene_cfg = getattr(scenario, "scene", None)
    default_position = getattr(scene_cfg, "default_position", None)
    if default_position is None:
        return 0.0, 0.0, 0.0
    values = tuple(float(value) for value in default_position)
    if len(values) != 3:
        raise ValueError(f"Scene default_position must have 3 values, got {values!r}")
    return values


def apply_scene_task_offset_to_scenario(scenario: Any, offset: tuple[float, float, float]) -> None:
    if _is_zero_vec3(offset):
        return
    for camera in getattr(scenario, "cameras", ()) or ():
        if hasattr(camera, "pos"):
            _set_object_attr(camera, "pos", _translated_vec3(camera.pos, offset))
        if hasattr(camera, "look_at"):
            _set_object_attr(camera, "look_at", _translated_vec3(camera.look_at, offset))
    for light in getattr(scenario, "lights", ()) or ():
        if hasattr(light, "pos"):
            _set_object_attr(light, "pos", _translated_vec3(light.pos, offset))


def apply_scene_import_offset_to_handler(handler: Any, offset: tuple[float, float, float]) -> None:
    if _is_zero_vec3(offset):
        return
    for root in getattr(handler, "_scene_objs", ()) or ():
        _translate_blender_object_location(root, offset)
    bpy = sys.modules.get("bpy")
    view_layer = getattr(getattr(bpy, "context", None), "view_layer", None) if bpy is not None else None
    update = getattr(view_layer, "update", None)
    if callable(update):
        update()


def _translate_blender_object_location(obj: Any, offset: tuple[float, float, float]) -> None:
    location = getattr(obj, "location", None)
    if location is None:
        return
    try:
        for index in range(3):
            location[index] = float(location[index]) + float(offset[index])
        return
    except (TypeError, AttributeError, IndexError, ValueError):
        pass
    if all(hasattr(location, attr_name) for attr_name in ("x", "y", "z")):
        location.x = float(location.x) + float(offset[0])
        location.y = float(location.y) + float(offset[1])
        location.z = float(location.z) + float(offset[2])
        return
    _set_object_attr(obj, "location", _translated_vec3(location, offset))


def clear_task_volume_for_replay(
    handler: Any,
    *,
    center: tuple[float, float, float],
    half_extents: tuple[float, float, float] = (2.0, 2.0, 2.4),
) -> dict[str, Any]:
    volume_min = tuple(float(center[index]) - float(half_extents[index]) for index in range(3))
    volume_max = tuple(float(center[index]) + float(half_extents[index]) for index in range(3))
    hidden_count = 0
    for obj in _iter_scene_tree(getattr(handler, "_scene_objs", ()) or ()):
        bounds = _object_world_bounds(obj)
        if bounds is None:
            continue
        if _looks_like_floor(bounds):
            continue
        if not _bounds_intersect(bounds, (volume_min, volume_max)):
            continue
        with suppress(Exception):
            obj.hide_render = True
        with suppress(Exception):
            obj.hide_viewport = True
        hidden_count += 1
    return {"status": "applied", "hidden_count": hidden_count}


def _iter_scene_tree(roots: Sequence[Any]):
    seen: set[int] = set()
    stack = list(roots)
    while stack:
        obj = stack.pop()
        obj_id = id(obj)
        if obj_id in seen:
            continue
        seen.add(obj_id)
        yield obj
        stack.extend(getattr(obj, "children", ()) or ())


def _object_world_bounds(obj: Any) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    if getattr(obj, "type", None) != "MESH":
        return None
    bound_box = getattr(obj, "bound_box", None)
    matrix_world = getattr(obj, "matrix_world", None)
    if bound_box is None or matrix_world is None:
        return None
    try:
        from mathutils import Vector
    except Exception:
        Vector = None
    points: list[tuple[float, float, float]] = []
    for corner in bound_box:
        try:
            world = matrix_world @ (Vector(corner) if Vector is not None else corner)
            points.append((float(world[0]), float(world[1]), float(world[2])))
        except Exception:
            return None
    if not points:
        return None
    return (
        tuple(min(point[index] for point in points) for index in range(3)),
        tuple(max(point[index] for point in points) for index in range(3)),
    )


def _looks_like_floor(bounds: tuple[tuple[float, float, float], tuple[float, float, float]]) -> bool:
    min_corner, max_corner = bounds
    return max_corner[2] <= 0.20 and (max_corner[2] - min_corner[2]) <= 0.25


def _bounds_intersect(
    first: tuple[tuple[float, float, float], tuple[float, float, float]],
    second: tuple[tuple[float, float, float], tuple[float, float, float]],
) -> bool:
    first_min, first_max = first
    second_min, second_max = second
    return all(first_min[index] <= second_max[index] and first_max[index] >= second_min[index] for index in range(3))


def translate_tensor_state(state: Any, offset: tuple[float, float, float]) -> Any:
    if _is_zero_vec3(offset):
        return state

    shifted = deepcopy(state)
    for collection_name in ("objects", "robots"):
        collection = getattr(shifted, collection_name, {}) or {}
        for entity_state in collection.values():
            _translate_state_attr(entity_state, "root_state", offset)
            _translate_state_attr(entity_state, "body_state", offset)
    return shifted


def _translate_state_attr(entity_state: Any, attr_name: str, offset: tuple[float, float, float]) -> None:
    tensor = getattr(entity_state, attr_name, None)
    if tensor is None:
        return
    translated = _clone_tensor_like(tensor)
    translated[..., 0:3] = translated[..., 0:3] + _offset_like(tensor, offset)
    setattr(entity_state, attr_name, translated)


def _clone_tensor_like(tensor: Any) -> Any:
    clone = getattr(tensor, "clone", None)
    if callable(clone):
        return clone()
    copy = getattr(tensor, "copy", None)
    if callable(copy):
        return copy()
    return deepcopy(tensor)


def _offset_like(tensor: Any, offset: tuple[float, float, float]) -> Any:
    new_tensor = getattr(tensor, "new_tensor", None)
    if callable(new_tensor):
        return new_tensor(offset)
    try:
        import numpy as np

        return np.asarray(offset, dtype=getattr(tensor, "dtype", None))
    except Exception:
        return offset


def _translated_vec3(value: Any, offset: tuple[float, float, float]) -> tuple[float, float, float]:
    values = tuple(float(item) for item in value)
    if len(values) != 3:
        raise ValueError(f"Expected vec3, got {values!r}")
    return tuple(values[index] + float(offset[index]) for index in range(3))


def _is_zero_vec3(value: tuple[float, float, float]) -> bool:
    return all(abs(float(item)) <= 1.0e-12 for item in value)


def _set_object_attr(obj: Any, name: str, value: Any) -> None:
    try:
        setattr(obj, name, value)
        return
    except (AttributeError, TypeError, ValueError):
        pass

    obj_dict = getattr(obj, "__dict__", None)
    if isinstance(obj_dict, dict):
        obj_dict[name] = value
        return

    raise AttributeError(f"{type(obj).__name__} does not allow setting {name!r}")


def configure_blender_for_video(*, samples: int, device: str, exposure: float = -2.0) -> None:
    import bpy

    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = int(samples)
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    _set_if_supported(scene.render, "use_compositing", False)
    _set_if_supported(scene.render, "use_sequencer", False)
    scene.view_settings.view_transform = "Standard"
    _set_if_supported(scene.view_settings, "look", "None")
    scene.view_settings.exposure = float(exposure)
    scene.view_settings.gamma = 1.0
    _set_if_supported(scene.cycles, "use_denoising", True)
    _set_if_supported(scene.cycles, "denoiser", "OPTIX")
    _set_if_supported(scene.cycles, "denoising_input_passes", "RGB_ALBEDO_NORMAL")
    _set_if_supported(scene.cycles, "denoising_prefilter", "ACCURATE")
    _set_if_supported(scene.cycles, "denoising_quality", "HIGH")
    _set_if_supported(scene.cycles, "use_adaptive_sampling", False)
    _set_if_supported(scene.cycles, "max_bounces", 10)
    _set_if_supported(scene.cycles, "diffuse_bounces", 4)
    _set_if_supported(scene.cycles, "glossy_bounces", 4)
    _set_if_supported(scene.cycles, "transmission_bounces", 10)
    _set_if_supported(scene.cycles, "transparent_max_bounces", 10)
    _set_if_supported(scene.cycles, "caustics_reflective", True)
    _set_if_supported(scene.cycles, "caustics_refractive", True)
    _set_if_supported(scene.cycles, "sample_clamp_indirect", 10.0)
    _set_if_supported(scene.cycles, "blur_glossy", 1.0)

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


def repair_imported_scene_materials() -> dict[str, Any]:
    import bpy

    objects = list(getattr(bpy.context.scene, "objects", ()))
    status: dict[str, Any] = {
        "status": "unknown",
        "object_count": len(objects),
        "override_count": len(KUJIALE_MATERIAL_OVERRIDES),
    }
    try:
        blender_module = importlib.import_module("metasim.sim.blender.blender")
        repair_imported_materials = getattr(blender_module, "_repair_imported_materials")
        repair_imported_materials(objects, KUJIALE_MATERIAL_OVERRIDES)
        status["status"] = "applied"
    except Exception as exc:
        status["status"] = "failed"
        status["error"] = repr(exc)
    with suppress(Exception):
        bpy.context.view_layer.update()
    return status


@contextmanager
def local_bundle_render_runtime():
    try:
        import metasim.utils.hf_util as hf_util
    except Exception:
        yield
        return

    original_single = getattr(hf_util, "check_and_download_single", None)
    original_recursive = getattr(hf_util, "check_and_download_recursive", None)
    hf_util.check_and_download_single = lambda filepath: None
    hf_util.check_and_download_recursive = lambda filepaths, n_processes=16: None
    try:
        yield
    finally:
        if original_single is not None:
            hf_util.check_and_download_single = original_single
        if original_recursive is not None:
            hf_util.check_and_download_recursive = original_recursive


def _set_if_supported(obj: Any, name: str, value: Any) -> None:
    try:
        setattr(obj, name, value)
    except (AttributeError, TypeError, ValueError):
        pass


def _enable_cycles_devices(preferences: Any, devices: Any = None) -> None:
    if devices is None:
        devices = getattr(preferences, "devices", ())
    for cycles_device in devices:
        if isinstance(cycles_device, (list, tuple)):
            _enable_cycles_devices(preferences, cycles_device)
        else:
            cycles_device.use = True


def apply_state_to_handler(handler: Any, state: Any) -> None:
    if apply_tensor_state_directly(handler, state):
        return

    set_blender_image_format_if_loaded(file_format="BMP", color_mode="RGB")
    try:
        handler.set_states(state)
    except TypeError:
        handler.set_states([state])
    if getattr(handler, "set_states_refreshes", False):
        return
    refresh_render = getattr(handler, "refresh_render", None)
    if callable(refresh_render):
        refresh_render()


def apply_tensor_state_directly(handler: Any, state: Any) -> bool:
    apply_tensor_state = getattr(handler, "_apply_tensor_state", None)
    if not callable(apply_tensor_state) or not _looks_like_tensor_state(state):
        return False

    invalidate = getattr(handler, "_invalidate_state_caches", None)
    if callable(invalidate):
        invalidate()
    apply_tensor_state(state)
    setattr(handler, "_last_tensor_state", state)
    setattr(handler, "_render_dirty", True)

    bpy = sys.modules.get("bpy")
    view_layer = getattr(getattr(bpy, "context", None), "view_layer", None) if bpy is not None else None
    update = getattr(view_layer, "update", None)
    if callable(update):
        update()
    return True


def _looks_like_tensor_state(state: Any) -> bool:
    return all(hasattr(state, attr_name) for attr_name in ("objects", "robots", "cameras"))


def set_blender_image_format_if_loaded(*, file_format: str, color_mode: str) -> None:
    bpy = sys.modules.get("bpy")
    if bpy is None:
        return
    scene = getattr(getattr(bpy, "context", None), "scene", None)
    image_settings = getattr(getattr(scene, "render", None), "image_settings", None)
    if image_settings is None:
        return
    image_settings.file_format = file_format
    image_settings.color_mode = color_mode


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
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.camera = camera_object_for(handler, camera)
    scene.render.resolution_x = int(camera.width)
    scene.render.resolution_y = int(camera.height)
    scene.render.resolution_percentage = 100
    scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)


def run_decode_worker(args: argparse.Namespace) -> int:
    paths = BoxTaskBundlePaths.from_root(args.bundle_root, traj_path=args.traj_path)
    source_indices = json.loads(args.decode_source_indices_json)
    states = decode_trajectory_tensor_states(paths.traj_path, source_indices, bundle_paths=paths)
    save_decoded_tensor_states(states, args.decode_states_out)
    print(
        json.dumps(
            {
                "status": "decoded",
                "states": str(args.decode_states_out),
                "source_frames": len(source_indices),
            }
        )
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.decode_source_indices_json is not None or args.decode_states_out is not None:
        if args.decode_source_indices_json is None or args.decode_states_out is None:
            raise ValueError("--decode-source-indices-json and --decode-states-out must be used together")
        return run_decode_worker(args)
    plan = build_plan(args)
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    return render_video(args)


if __name__ == "__main__":
    raise SystemExit(main())
