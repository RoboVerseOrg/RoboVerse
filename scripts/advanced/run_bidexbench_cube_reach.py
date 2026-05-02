from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Iterable, Iterator
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
from roboverse_pack.teleop.flow import run_native_task_teleop_flow
from roboverse_pack.teleop.runtime import CanonicalTeleopTargets

DEFAULT_BLENDER_RENDER_OUTPUT = Path("outputs") / "bidexbench_cube_reach_blender"


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


def _packet_source(args: argparse.Namespace) -> Iterable[object]:
    if args.packet_jsonl is not None:
        return _read_jsonl_packets(Path(args.packet_jsonl).expanduser())
    return _synthetic_packets(steps=args.steps, hand_targets=args.hand_targets)


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
        help="Send direct 20-DoF Wuji hand joint targets instead of grip-ratio fallback packets.",
    )
    parser.add_argument("--headless", dest="headless", action="store_true", default=False, help="Run without a viewer.")
    parser.add_argument("--viewer", dest="headless", action="store_false", help="Open the simulator viewer when supported.")
    parser.add_argument(
        "--physics-viewer",
        action="store_true",
        help="For split physics/render runs, also open the physics backend viewer for debugging.",
    )
    parser.add_argument(
        "--hold-initial-pose",
        action="store_true",
        help="Ignore synthetic teleop targets and replay the reset joint positions for debugging.",
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
    parser.add_argument("--render-samples", type=int, default=64, help="Cycles samples for offline Blender rendering.")
    parser.add_argument(
        "--render-device",
        choices=("CPU", "CUDA", "OPTIX", "HIP", "ONEAPI", "METAL"),
        default="CPU",
        help="Cycles device for offline Blender rendering.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate task/robot selection without launching simulation.")
    return parser


def _configure_rendering_args(args: argparse.Namespace) -> str | None:
    """Normalize live and offline renderer options before launching physics."""
    selected_renderer = args.renderer
    if args.renderer == "blender":
        args.offline_renderer = "blender"
        selected_renderer = None
    if args.offline_renderer == "blender" and args.render_output is None:
        args.render_output = DEFAULT_BLENDER_RENDER_OUTPUT
    return selected_renderer


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
                    "packet_source": str(args.packet_jsonl) if args.packet_jsonl is not None else "synthetic",
                    "steps": None if args.packet_jsonl is not None else args.steps,
                    "hand_targets": args.hand_targets,
                    "physics_viewer": args.physics_viewer,
                    "hold_initial_pose": args.hold_initial_pose,
                    "record_states": str(args.record_states) if args.record_states is not None else None,
                    "offline_renderer": args.offline_renderer,
                    "render_output": str(args.render_output) if args.render_output is not None else None,
                    "render_samples": args.render_samples,
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

    result = run_native_task_teleop_flow(
        task=task_spec.name,
        robot=args.robot,
        simulator=args.sim,
        renderer=selected_renderer,
        packets=_packet_source(args),
        record_states_path=record_states_path,
        headless=args.headless,
        physics_viewer=args.physics_viewer,
        hold_initial_pose=args.hold_initial_pose,
    )
    if args.offline_renderer == "blender":
        assert record_states_path is not None
        result["offline_renderer"] = "blender"
        result["render_output"] = str(args.render_output)
        result["rendered_frames"] = _render_recorded_states_with_blender(args, record_states_path)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
