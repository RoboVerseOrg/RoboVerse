#!/usr/bin/env python3
"""Replay a registered task's trajectory with switchable Kujiale backgrounds.

Loads a task from the metasim registry (``--task``), reads its
``traj_filepath``, splits output frames across the requested
``--scenes``, and renders each segment in a fresh subprocess with the
corresponding background scene. The segments are concatenated into a
single video via ffmpeg.

Tasks can opt into per-frame state fixups by defining a
``prepare_replay_state(state)`` classmethod. When present it is applied
to every replayed frame; otherwise frames pass through unchanged.

Example::

    python get_started/replay_multi_scene_render.py \\
        --task box_task.replay \\
        --scenes 0021,0022,0024,0025,0031 \\
        --render-mode raytracing \\
        --out-video out/box_task_switch5.mp4
"""

from __future__ import annotations

import argparse
import os
import pickle
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from loguru import logger as log


def _parse_vec3(text: str) -> tuple[float, float, float]:
    parts = [s.strip() for s in text.split(",") if s.strip()]
    if len(parts) != 3:
        raise ValueError(f"Expected 'x,y,z', got: {text}")
    return float(parts[0]), float(parts[1]), float(parts[2])


def _normalize_scene_name(token: str) -> str:
    s = token.strip().lower()
    if not s:
        raise ValueError("Empty scene token")
    if s.endswith(".usda"):
        s = s[:-5]
    for prefix in ("kujiale_scene_", "kujiale_", "usd_"):
        if s.startswith(prefix):
            s = s[len(prefix) :]
            break
    if not s.isdigit():
        raise ValueError(f"Invalid scene token: {token}")
    if len(s) == 3:
        s = f"0{s}"
    elif len(s) != 4:
        raise ValueError(f"Scene token must be 3 or 4 digits, got: {token}")
    return f"kujiale_scene_{s}"


def _parse_scenes(text: str) -> list[str]:
    out = [_normalize_scene_name(t) for t in text.split(",") if t.strip()]
    if not out:
        raise ValueError("No valid scenes parsed")
    return out


def _to_rgb_frame(frame: np.ndarray, height: int, width: int) -> np.ndarray:
    """Coerce a per-frame camera tensor into a ``(H, W, 3)`` uint-able array.

    Backends differ slightly: mujoco returns ``(H, W, 3)`` after the
    ``[0]`` env index, IsaacSim returns ``(1, H, W, 1|3|4)`` (an extra
    singleton up front and the trailing channel count varies with the
    render mode). Strip leading singletons, then normalise channels.
    """
    while frame.ndim > 3 and frame.shape[0] == 1:
        frame = frame[0]
    if frame.ndim == 2:
        frame = np.stack([frame, frame, frame], axis=-1)
    elif frame.ndim == 3 and frame.shape[-1] == 1:
        frame = np.repeat(frame, 3, axis=-1)
    elif frame.ndim == 3 and frame.shape[-1] == 4:
        frame = frame[..., :3]
    return frame


def _read_traj_length(traj_path: Path, robot_name: str) -> int:
    """Number of frames in the first episode of a v2 trajectory."""
    with traj_path.open("rb") as f:
        data = pickle.load(f)
    return len(data[robot_name][0]["states"])


def _resolve_task_class(task_name: str):
    from metasim.task.registry import get_task_class

    return get_task_class(task_name)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("--task", type=str, required=True, help="Registered task name (e.g. 'box_task.replay').")
    parser.add_argument(
        "--simulator",
        choices=["isaacsim", "mujoco"],
        default="isaacsim",
        help="Simulator backend. ``isaacsim`` enables Kujiale backgrounds and "
        "ray/path-tracing; ``mujoco`` ignores --scenes and renders the bare scenario.",
    )
    parser.add_argument(
        "--scenes",
        type=str,
        default="0021,0022,0024,0025,0031",
        help="Comma-separated Kujiale scene ids (isaacsim only). "
        "Pass 'none' to render without a background (the default for --simulator mujoco).",
    )
    parser.add_argument("--out-video", type=Path, required=True)

    parser.add_argument("--traj-path", type=Path, default=None, help="Override the task's bundled trajectory path.")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument(
        "--render-mode",
        choices=["raytracing", "pathtracing"],
        default="raytracing",
        help="Render quality (isaacsim only).",
    )
    parser.add_argument("--duration-sec", type=float, default=None)
    parser.add_argument("--out-frames", type=int, default=None)
    parser.add_argument("--camera-pos", type=str, default="1.45,-1.00,0.95")
    parser.add_argument("--camera-lookat", type=str, default="0.55,0.0,0.38")
    parser.add_argument("--head-light-intensity", type=float, default=10000.0)
    parser.add_argument("--keep-intermediates", action="store_true")
    parser.add_argument("--tmp-dir", type=Path, default=None)
    parser.add_argument("--python-bin", type=str, default=sys.executable)

    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--_scene", type=str, default=None)
    parser.add_argument("--_out-start", type=int, default=None)
    parser.add_argument("--_out-end", type=int, default=None)
    parser.add_argument("--_index-map", type=Path, default=None)
    parser.add_argument("--_segment-out", type=Path, default=None)

    return parser


def _worker_main(args: argparse.Namespace) -> None:
    if any(v is None for v in (args._scene, args._out_start, args._out_end, args._index_map, args._segment_out)):
        raise ValueError("Worker mode requires --_scene/--_out-start/--_out-end/--_index-map/--_segment-out")

    import imageio.v2 as iio
    import torch

    import metasim.utils.hf_util as hf_util
    import roboverse_pack.scenes  # noqa: F401 — register Kujiale scene cfgs
    from metasim.scenario.cameras import PinholeCameraCfg
    from metasim.scenario.lights import SphereLightCfg
    from metasim.scenario.render import RenderCfg
    from metasim.utils.demo_util import get_traj

    hf_util.check_and_download_recursive = lambda filepaths, n_processes=16: None

    task_cls = _resolve_task_class(args.task)

    camera = PinholeCameraCfg(
        name="camera0",
        pos=_parse_vec3(args.camera_pos),
        look_at=_parse_vec3(args.camera_lookat),
        width=args.width,
        height=args.height,
    )

    scenario_kwargs = dict(
        cameras=[camera],
        lights=[
            SphereLightCfg(
                name="overhead_light",
                intensity=float(args.head_light_intensity),
                color=(1.0, 1.0, 1.0),
                radius=0.15,
                pos=(0.55, 0.0, 1.45),
                is_global=False,
            )
        ],
        simulator=args.simulator,
        renderer=args.simulator,
        headless=True,
        num_envs=1,
    )
    scene = None if args._scene.lower() == "none" else args._scene
    if args.simulator == "isaacsim":
        # Kujiale background + render-quality knob are isaacsim-only.
        if scene is not None:
            scenario_kwargs["scene"] = scene
        scenario_kwargs["render"] = RenderCfg(mode=args.render_mode)
    scenario = task_cls.scenario.update(**scenario_kwargs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = task_cls(scenario, device=device)
    try:
        traj_path = str(args.traj_path) if args.traj_path is not None else env.traj_filepath
        _, _, all_states = get_traj(traj_path, scenario.robots[0], env.handler)
        episode_states = all_states[0]

        prepare = getattr(task_cls, "prepare_replay_state", None)
        frame_to_src = np.load(args._index_map)

        writer = iio.get_writer(args._segment_out, fps=args.fps)
        try:
            log.info(f"scene={args._scene} out=[{args._out_start},{args._out_end}) render={args.render_mode}")
            for out_idx in range(args._out_start, args._out_end):
                src_idx = int(frame_to_src[out_idx])
                state = episode_states[src_idx]
                if prepare is not None:
                    state = prepare(state)
                try:
                    env.handler.set_states([state])
                    env.handler.refresh_render()
                    obs = env.handler.get_states(mode="tensor")
                    rgb = next(iter(obs.cameras.values())).rgb
                    frame = rgb[0].detach().cpu().numpy() if hasattr(rgb[0], "detach") else np.asarray(rgb[0])
                    frame = _to_rgb_frame(frame, args.height, args.width)
                    writer.append_data(np.clip(frame, 0, 255).astype(np.uint8))
                    if (out_idx - args._out_start) % 30 == 0:
                        log.info(f"out_frame={out_idx} src_idx={src_idx}")
                except Exception as e:
                    log.exception(f"worker frame {out_idx} (src {src_idx}) failed: {e}")
                    raise
        finally:
            writer.close()
    finally:
        env.close()

    log.info(f"saved={args._segment_out}")


def _coordinator_main(args: argparse.Namespace, script_path: Path) -> None:
    task_cls = _resolve_task_class(args.task)
    traj_path = (args.traj_path if args.traj_path is not None else Path(task_cls.traj_filepath)).resolve()
    if not traj_path.exists():
        raise FileNotFoundError(f"Trajectory not found: {traj_path}")

    src_total = _read_traj_length(traj_path, task_cls.scenario.robots[0].name)
    if src_total <= 0:
        raise ValueError(f"Trajectory has no frames: {traj_path}")

    if args.out_frames is not None and args.duration_sec is not None:
        raise ValueError("Use either --out-frames or --duration-sec, not both")
    if args.out_frames is not None:
        out_frames = int(args.out_frames)
    elif args.duration_sec is not None:
        out_frames = round(float(args.duration_sec) * float(args.fps))
    else:
        out_frames = src_total
    if out_frames <= 0:
        raise ValueError("Computed output frames must be positive")

    # mujoco can't load Kujiale USDs; treat the request as a single
    # bare-scene segment. ``--scenes none`` is the explicit opt-out.
    if args.simulator != "isaacsim" or args.scenes.strip().lower() == "none":
        scenes: list[str | None] = [None]
    else:
        scenes = _parse_scenes(args.scenes)
    frame_to_src = np.round(np.linspace(0, src_total - 1, out_frames)).astype(np.int32)

    tmp_root = args.tmp_dir.resolve() if args.tmp_dir is not None else Path.cwd() / ".tmp_replay_render"
    tmp_root.mkdir(parents=True, exist_ok=True)
    work_dir = tmp_root / f"render_work_{os.getpid()}"
    suffix = 0
    while work_dir.exists():
        suffix += 1
        work_dir = tmp_root / f"render_work_{os.getpid()}_{suffix}"
    work_dir.mkdir(parents=True, exist_ok=False)

    index_map_path = work_dir / "frame_to_src.npy"
    np.save(index_map_path, frame_to_src)

    bounds = np.linspace(0, out_frames, num=len(scenes) + 1, dtype=int)
    seg_paths: list[Path] = []

    log.info(
        f"task={args.task}, src_total={src_total}, out_frames={out_frames}, "
        f"fps={args.fps}, scenes={scenes}, bounds={bounds.tolist()}"
    )

    for i, scene_name in enumerate(scenes):
        out_start = int(bounds[i])
        out_end = int(bounds[i + 1])
        scene_label = scene_name or "no_scene"
        seg_path = work_dir / f"seg_{i + 1:02d}_{scene_label}_{out_start}_{out_end}.mp4"
        seg_paths.append(seg_path)

        cmd = [
            args.python_bin,
            str(script_path),
            "--_worker",
            "--simulator",
            args.simulator,
            "--task",
            args.task,
            "--traj-path",
            str(traj_path),
            "--_scene",
            scene_name or "none",
            "--_out-start",
            str(out_start),
            "--_out-end",
            str(out_end),
            "--_index-map",
            str(index_map_path),
            "--_segment-out",
            str(seg_path),
            "--fps",
            str(args.fps),
            "--width",
            str(args.width),
            "--height",
            str(args.height),
            "--render-mode",
            args.render_mode,
            # Use ``--flag=value`` form so leading-negative values
            # (e.g. ``-0.15,0.0,0.95``) don't get re-parsed as flags.
            f"--camera-pos={args.camera_pos}",
            f"--camera-lookat={args.camera_lookat}",
            "--head-light-intensity",
            str(args.head_light_intensity),
            "--out-video",
            str(args.out_video),
        ]
        log.info(f"[segment {i + 1}/{len(scenes)}] {scene_name} out[{out_start},{out_end})")
        # IsaacSim's SimulationApp.close() is known to hang on shutdown; the
        # metasim handler watchdog force-exits the worker with code 1 after a
        # timeout. As long as the segment .mp4 was written (i.e. all frames
        # rendered before the force-exit), treat the worker as successful.
        _run_cmd(cmd, success_file=seg_path)

    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        raise RuntimeError("ffmpeg not found in PATH")

    concat_list = work_dir / "concat_list.txt"
    concat_list.write_text("\n".join(f"file '{p.as_posix()}'" for p in seg_paths) + "\n", encoding="utf-8")

    out_video = args.out_video.resolve()
    out_video.parent.mkdir(parents=True, exist_ok=True)

    _run_cmd([
        ffmpeg_bin,
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_list),
        "-c",
        "copy",
        str(out_video),
    ])

    log.info(f"saved={out_video}")
    log.info(f"work_dir={work_dir}")

    if not args.keep_intermediates:
        shutil.rmtree(work_dir, ignore_errors=True)


def _run_cmd(cmd: list[str], success_file: Path | None = None) -> None:
    proc = subprocess.run(cmd, check=False)
    if proc.returncode == 0:
        return
    if success_file is not None and success_file.exists() and success_file.stat().st_size > 0:
        log.warning(
            f"Command exited with code {proc.returncode}, but {success_file} was written "
            f"({success_file.stat().st_size} bytes); treating as success."
        )
        return
    raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}")


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    script_path = Path(__file__).resolve()

    if args._worker:
        _worker_main(args)
    else:
        _coordinator_main(args=args, script_path=script_path)


if __name__ == "__main__":
    main()
