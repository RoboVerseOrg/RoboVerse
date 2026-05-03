from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any


def _option_value(argv: list[str], option: str) -> str | None:
    prefix = f"{option}="
    for index, item in enumerate(argv):
        if item.startswith(prefix):
            return item[len(prefix) :]
        if item == option and index + 1 < len(argv):
            return argv[index + 1]
    return None


def _preconfigure_mujoco_gl(argv: list[str]) -> None:
    sim = _option_value(argv, "--sim")
    renderer = _option_value(argv, "--renderer")
    if "--physics-viewer" in argv:
        return
    if sim == "mujoco" and ("--headless" in argv or (renderer is not None and renderer != "mujoco")):
        os.environ.setdefault("MUJOCO_GL", "egl")


_preconfigure_mujoco_gl(sys.argv[1:])

from roboverse_pack.benchmark import get_benchmark_task_spec
from roboverse_pack.robots.openarm_bimanual_wuji_cfg import LEFT_ARM_DEFAULT_Q, RIGHT_ARM_DEFAULT_Q
from roboverse_pack.teleop.flow import run_native_task_teleop_flow
from roboverse_pack.teleop.runtime import CanonicalTeleopTargets

DEFAULT_BLENDER_RENDER_OUTPUT = Path("outputs") / "bidexbench_cube_reach_blender"
TRAJECTORY_CHOICES = ("hold-initial-pose", "synthetic-reach", "hand-wave", "cube-reach-demo")


def _read_jsonl_packets(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            packet = json.loads(stripped)
            if not isinstance(packet, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object teleop packet")
            yield packet


def _synthetic_packets(*, steps: int, hand_targets: bool) -> Iterator[CanonicalTeleopTargets]:
    total = max(1, int(steps))
    for index in range(total):
        phase = index / max(1, total - 1)
        left_close = 0.15 + 0.70 * phase
        right_close = 0.85 - 0.70 * phase
        kwargs: dict[str, Any] = {}
        if hand_targets:
            kwargs["left_hand_target_q_rad"] = tuple(0.15 * phase for _ in range(20))
            kwargs["right_hand_target_q_rad"] = tuple(0.15 * (1.0 - phase) for _ in range(20))

        yield CanonicalTeleopTargets(
            left_work_pose_cm_xyzw=(-12.0, -70.0, 30.0, 0.0, 0.0, 0.0, 1.0),
            right_work_pose_cm_xyzw=(12.0, -70.0, 30.0, 0.0, 0.0, 0.0, 1.0),
            left_close_ratio=left_close,
            right_close_ratio=right_close,
            transform_profile="script:synthetic",
            **kwargs,
        )


def _hold_initial_pose_packets(*, steps: int) -> Iterator[object]:
    for _ in range(max(1, int(steps))):
        yield object()


def _phase(index: int, total: int) -> float:
    return index / max(1, total - 1)


def _smoothstep(value: float, start: float = 0.0, end: float = 1.0) -> float:
    if end <= start:
        return 1.0 if value >= end else 0.0
    t = max(0.0, min(1.0, (value - start) / (end - start)))
    return t * t * (3.0 - 2.0 * t)


def _lerp(start: float, end: float, amount: float) -> float:
    return start + (end - start) * amount


def _lerp_pose(
    start: tuple[float, float, float, float, float, float, float],
    end: tuple[float, float, float, float, float, float, float],
    amount: float,
) -> tuple[float, float, float, float, float, float, float]:
    return tuple(_lerp(start[index], end[index], amount) for index in range(7))  # type: ignore[return-value]


def _lerp_joint_targets(start: Sequence[float], end: Sequence[float], amount: float) -> tuple[float, ...]:
    if len(start) != len(end):
        raise ValueError(f"joint target length mismatch: {len(start)} != {len(end)}")
    return tuple(_lerp(float(start[index]), float(end[index]), amount) for index in range(len(start)))


def _offset_joint_targets(defaults: Sequence[float], offsets: Sequence[float]) -> tuple[float, ...]:
    if len(defaults) != len(offsets):
        raise ValueError(f"joint target length mismatch: {len(defaults)} != {len(offsets)}")
    return tuple(float(defaults[index]) + float(offsets[index]) for index in range(len(defaults)))


def _wave_hand_targets(*, phase: float, side_offset: float) -> tuple[float, ...]:
    return tuple(
        max(0.02, min(0.24, 0.12 + 0.08 * math.sin(2.0 * math.pi * phase + side_offset + joint * 0.35)))
        for joint in range(20)
    )


def _hand_wave_packets(*, steps: int) -> Iterator[CanonicalTeleopTargets]:
    total = max(1, int(steps))
    for index in range(total):
        phase = _phase(index, total)
        sweep = math.sin(2.0 * math.pi * phase)
        lift = math.sin(4.0 * math.pi * phase)
        left_close = 0.35 + 0.35 * _smoothstep(math.sin(2.0 * math.pi * phase), -1.0, 1.0)
        right_close = 0.35 + 0.35 * _smoothstep(math.sin(2.0 * math.pi * phase + math.pi), -1.0, 1.0)

        yield CanonicalTeleopTargets(
            left_work_pose_cm_xyzw=(-14.0 + 5.0 * sweep, -70.0, 30.0 + 4.0 * lift, 0.0, 0.0, 0.0, 1.0),
            right_work_pose_cm_xyzw=(14.0 + 5.0 * sweep, -70.0, 30.0 - 4.0 * lift, 0.0, 0.0, 0.0, 1.0),
            left_close_ratio=left_close,
            right_close_ratio=right_close,
            left_hand_target_q_rad=_wave_hand_targets(phase=phase, side_offset=0.0),
            right_hand_target_q_rad=_wave_hand_targets(phase=phase, side_offset=math.pi),
            left_arm_target_q_rad=_offset_joint_targets(
                LEFT_ARM_DEFAULT_Q,
                (0.03 * sweep, 0.02 * lift, 0.02 * sweep, 0.02 * lift, 0.01 * lift, -0.03 * sweep, 0.01 * lift),
            ),
            right_arm_target_q_rad=_offset_joint_targets(
                RIGHT_ARM_DEFAULT_Q,
                (0.03 * sweep, -0.02 * lift, -0.02 * sweep, -0.02 * lift, -0.01 * lift, 0.03 * sweep, -0.01 * lift),
            ),
            transform_profile="script:hand-wave",
        )


def _cube_reach_demo_packets(*, steps: int) -> Iterator[CanonicalTeleopTargets]:
    total = max(1, int(steps))
    left_start = (-20.0, -72.0, 34.0, 0.0, 0.0, 0.0, 1.0)
    right_start = (20.0, -72.0, 34.0, 0.0, 0.0, 0.0, 1.0)
    left_near_cube = (-5.0, -58.0, 23.0, 0.0, 0.0, 0.0, 1.0)
    right_near_cube = (5.0, -58.0, 23.0, 0.0, 0.0, 0.0, 1.0)
    left_retract = (-12.0, -66.0, 37.0, 0.0, 0.0, 0.0, 1.0)
    right_retract = (12.0, -66.0, 37.0, 0.0, 0.0, 0.0, 1.0)
    left_near_cube_q = (-0.38, -0.22, 0.02, 1.18, 0.02, -0.03, 0.01)
    right_near_cube_q = (0.38, 0.22, -0.02, 1.18, -0.02, 0.03, -0.01)
    left_retract_q = (-0.42, -0.18, 0.01, 1.22, 0.01, 0.02, 0.00)
    right_retract_q = (0.42, 0.18, -0.01, 1.22, -0.01, -0.02, 0.00)

    for index in range(total):
        phase = _phase(index, total)
        approach = _smoothstep(phase, 0.10, 0.58)
        retract = _smoothstep(phase, 0.72, 1.00)
        left_pose = _lerp_pose(left_start, left_near_cube, approach)
        right_pose = _lerp_pose(right_start, right_near_cube, approach)
        left_pose = _lerp_pose(left_pose, left_retract, retract)
        right_pose = _lerp_pose(right_pose, right_retract, retract)
        left_arm_q = _lerp_joint_targets(LEFT_ARM_DEFAULT_Q, left_near_cube_q, approach)
        right_arm_q = _lerp_joint_targets(RIGHT_ARM_DEFAULT_Q, right_near_cube_q, approach)
        left_arm_q = _lerp_joint_targets(left_arm_q, left_retract_q, retract)
        right_arm_q = _lerp_joint_targets(right_arm_q, right_retract_q, retract)
        close = 0.18 + 0.62 * _smoothstep(phase, 0.48, 0.74) * (1.0 - 0.45 * retract)

        yield CanonicalTeleopTargets(
            left_work_pose_cm_xyzw=left_pose,
            right_work_pose_cm_xyzw=right_pose,
            left_close_ratio=close,
            right_close_ratio=close,
            left_hand_target_q_rad=tuple(close * 0.18 for _ in range(20)),
            right_hand_target_q_rad=tuple(close * 0.18 for _ in range(20)),
            left_arm_target_q_rad=left_arm_q,
            right_arm_target_q_rad=right_arm_q,
            transform_profile="script:cube-reach-demo",
        )


def _effective_trajectory(args: argparse.Namespace) -> str:
    if getattr(args, "hold_initial_pose", False):
        return "hold-initial-pose"
    trajectory = getattr(args, "trajectory", None)
    if trajectory is not None:
        return trajectory
    return "hold-initial-pose"


def _hold_initial_pose_enabled(args: argparse.Namespace) -> bool:
    return _effective_trajectory(args) == "hold-initial-pose"


def _packet_source(args: argparse.Namespace) -> Iterable[object]:
    if args.packet_jsonl is not None:
        return _read_jsonl_packets(Path(args.packet_jsonl).expanduser())
    trajectory = _effective_trajectory(args)
    if trajectory == "hold-initial-pose":
        return _hold_initial_pose_packets(steps=args.steps)
    if trajectory == "synthetic-reach":
        return _synthetic_packets(steps=args.steps, hand_targets=args.hand_targets)
    if trajectory == "hand-wave":
        return _hand_wave_packets(steps=args.steps)
    if trajectory == "cube-reach-demo":
        return _cube_reach_demo_packets(steps=args.steps)
    raise ValueError(f"Unsupported trajectory: {trajectory!r}")


def _packet_source_label(args: argparse.Namespace) -> str:
    if args.packet_jsonl is not None:
        return str(args.packet_jsonl)
    return _effective_trajectory(args)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the migrated BiDexBench cube_reach task with OpenArm + Wuji hands.",
    )
    parser.add_argument("--task", default="benchmark.cube_reach", help="Benchmark task name or alias.")
    parser.add_argument("--robot", default="openarm_bimanual_wuji", help="Robot config name.")
    parser.add_argument("--sim", default="isaacsim", choices=("isaacsim", "mujoco"), help="Native simulator backend.")
    parser.add_argument(
        "--renderer",
        default=None,
        choices=("isaacsim", "mujoco", "blender"),
        help=(
            "Optional render backend. Use --sim mujoco --renderer isaacsim for hybrid rendering, "
            "or --renderer blender for offline Blender/Cycles replay."
        ),
    )
    parser.add_argument("--steps", type=int, default=120, help="Synthetic teleop packet count.")
    parser.add_argument(
        "--packet-jsonl",
        type=Path,
        default=None,
        help="Optional JSONL file of canonical two-hand teleop packets. Overrides --steps.",
    )
    parser.add_argument(
        "--hand-targets",
        action="store_true",
        help="For synthetic-reach, send direct 20-DoF Wuji hand joint targets instead of grip-ratio fallback packets.",
    )
    parser.add_argument(
        "--trajectory",
        choices=TRAJECTORY_CHOICES,
        default=None,
        help="Built-in trajectory pattern. Defaults to hold-initial-pose.",
    )
    parser.add_argument("--headless", dest="headless", action="store_true", default=False, help="Run without a viewer.")
    parser.add_argument(
        "--viewer", dest="headless", action="store_false", help="Open the simulator viewer when supported."
    )
    parser.add_argument(
        "--physics-viewer",
        action="store_true",
        help="For split physics/render runs, also open the physics backend viewer for debugging.",
    )
    parser.add_argument(
        "--hold-initial-pose",
        action="store_true",
        help="Deprecated alias for --trajectory hold-initial-pose.",
    )
    parser.add_argument(
        "--record-states",
        type=Path,
        default=None,
        help="Optional path for saving MetaSim TensorState frames with torch.save.",
    )
    parser.add_argument(
        "--offline-renderer",
        choices=("blender",),
        default=None,
        help="Optional offline renderer to run after physics. First milestone supports blender only.",
    )
    parser.add_argument(
        "--render-output",
        type=Path,
        default=None,
        help="Output directory for offline rendered camera PNG frames.",
    )
    parser.add_argument(
        "--render-samples",
        type=int,
        default=None,
        help="Override Cycles samples for Blender rendering. Defaults to the final-quality setting.",
    )
    parser.add_argument("--render-video-fps", type=int, default=30, help="FPS for per-camera Blender render videos.")
    parser.add_argument(
        "--render-device",
        choices=("AUTO", "CPU", "CUDA", "OPTIX", "HIP", "ONEAPI", "METAL"),
        default="AUTO",
        help="Cycles device for Blender rendering. AUTO selects the fastest available GPU and falls back to CPU.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate task/robot selection without launching simulation."
    )
    return parser


def _configure_rendering_args(args: argparse.Namespace) -> str | None:
    """Normalize live and offline renderer options before launching physics."""
    selected_renderer = args.renderer
    if args.renderer == "blender" and args.offline_renderer == "blender":
        raise ValueError("Use either --renderer blender or --offline-renderer blender, not both")
    if (args.renderer == "blender" or args.offline_renderer == "blender") and args.render_output is None:
        args.render_output = DEFAULT_BLENDER_RENDER_OUTPUT
    return selected_renderer


def _camera_frame_dirs(output_dir: Path) -> list[Path]:
    return sorted(path for path in output_dir.iterdir() if path.is_dir() and list(path.glob("frame_*.png")))


def _write_camera_videos_from_frames(
    output_dir: str | Path,
    *,
    fps: int = 30,
    imageio_module: object | None = None,
) -> list[Path]:
    output_path = Path(output_dir)
    if imageio_module is None:
        import imageio.v2 as imageio_module

    videos: list[Path] = []
    for camera_dir in _camera_frame_dirs(output_path):
        frames = sorted(camera_dir.glob("frame_*.png"))
        if not frames:
            continue
        video_path = camera_dir / f"{camera_dir.name}.mp4"
        with imageio_module.get_writer(video_path, fps=int(fps), macro_block_size=1) as writer:
            for frame_path in frames:
                writer.append_data(imageio_module.imread(frame_path))
        videos.append(video_path)
    return videos


def _render_state_from_session(session: object) -> object:
    render_state = getattr(session, "render_state", None)
    if callable(render_state):
        return render_state()
    render_handler = getattr(session, "render_handler", None)
    if render_handler is not None:
        return render_handler.get_states(mode="tensor")
    handler = getattr(session, "handler", None)
    if handler is not None:
        return handler.get_states(mode="tensor")
    raise ValueError("session does not expose rendered camera state")


def _write_blender_live_frame(output_dir: str | Path, frame_index: int, session: object) -> list[Path]:
    import imageio.v3 as iio

    rendered = _render_state_from_session(session)
    written: list[Path] = []
    for camera_name, camera_state in getattr(rendered, "cameras", {}).items():
        rgb = getattr(camera_state, "rgb", None)
        if rgb is None:
            continue
        try:
            image = rgb[0]
        except Exception:
            image = rgb
        if hasattr(image, "detach"):
            image = image.detach().cpu().numpy()
        camera_dir = Path(output_dir) / camera_name
        camera_dir.mkdir(parents=True, exist_ok=True)
        path = camera_dir / f"frame_{frame_index:06d}.png"
        iio.imwrite(path, image)
        written.append(path)
    return written


def _render_recorded_states_with_blender(args: argparse.Namespace, state_path: Path) -> list[str]:
    if args.render_output is None:
        raise ValueError("--render-output is required when offline Blender rendering is used")

    import torch

    from metasim.sim.blender.offline import BlenderOfflineRenderCfg, render_state_sequence
    from roboverse_pack.tasks.benchmark.base import build_benchmark_scenario

    payload = torch.load(state_path, weights_only=False)
    states = payload["states"]
    task_spec = get_benchmark_task_spec(args.task)
    scenario = build_benchmark_scenario(
        task_spec,
        robot=args.robot,
        simulator=args.sim,
        headless=True,
    )

    outputs = render_state_sequence(
        scenario,
        states,
        BlenderOfflineRenderCfg(
            output_dir=args.render_output,
            samples=args.render_samples,
            device=args.render_device,
        ),
    )
    return [str(path) for path in outputs]


def main() -> None:
    args = _build_parser().parse_args()

    task_spec = get_benchmark_task_spec(args.task)
    task_spec.robot_profile(args.robot)
    selected_renderer = _configure_rendering_args(args)

    if args.dry_run:
        print(
            json.dumps(
                {
                    "task": task_spec.name,
                    "robot": args.robot,
                    "simulator": args.sim,
                    "renderer": selected_renderer,
                    "headless": args.headless,
                    "trajectory": _effective_trajectory(args),
                    "packet_source": _packet_source_label(args),
                    "steps": None if args.packet_jsonl is not None else args.steps,
                    "hand_targets": args.hand_targets,
                    "physics_viewer": args.physics_viewer,
                    "hold_initial_pose": _hold_initial_pose_enabled(args),
                    "record_states": str(args.record_states) if args.record_states is not None else None,
                    "offline_renderer": args.offline_renderer,
                    "render_output": str(args.render_output) if args.render_output is not None else None,
                    "render_samples": args.render_samples,
                    "render_video_fps": args.render_video_fps,
                    "render_device": args.render_device,
                },
                indent=2,
            )
        )
        return

    record_states_path = args.record_states
    if args.offline_renderer == "blender" and record_states_path is None:
        assert args.render_output is not None
        record_states_path = args.render_output / "states.pt"

    live_rendered_frames: list[str] = []
    render_frame_callback = None
    if selected_renderer == "blender":
        assert args.render_output is not None

        def render_frame_callback(frame_index: int, session: object) -> None:
            live_rendered_frames.extend(
                str(path) for path in _write_blender_live_frame(args.render_output, frame_index, session)
            )

    result = run_native_task_teleop_flow(
        task=task_spec.name,
        robot=args.robot,
        simulator=args.sim,
        renderer=selected_renderer,
        packets=_packet_source(args),
        record_states_path=record_states_path,
        render_frame_callback=render_frame_callback,
        headless=args.headless,
        physics_viewer=args.physics_viewer,
        hold_initial_pose=_hold_initial_pose_enabled(args),
        render_samples=args.render_samples if selected_renderer == "blender" else None,
        render_device=args.render_device if selected_renderer == "blender" else None,
    )
    if selected_renderer == "blender":
        assert args.render_output is not None
        result["render_output"] = str(args.render_output)
        result["rendered_frames"] = live_rendered_frames
        result["rendered_videos"] = [
            str(path) for path in _write_camera_videos_from_frames(args.render_output, fps=args.render_video_fps)
        ]
    elif args.offline_renderer == "blender":
        assert record_states_path is not None
        result["offline_renderer"] = "blender"
        result["render_output"] = str(args.render_output)
        result["rendered_frames"] = _render_recorded_states_with_blender(args, record_states_path)
        assert args.render_output is not None
        result["rendered_videos"] = [
            str(path) for path in _write_camera_videos_from_frames(args.render_output, fps=args.render_video_fps)
        ]
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
