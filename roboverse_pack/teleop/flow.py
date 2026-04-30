from __future__ import annotations

"""Native-only teleoperation flow for benchmark tasks."""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from roboverse_pack.benchmark import BenchmarkTaskSpec, get_benchmark_task_spec
from roboverse_pack.teleop.runtime import CanonicalTeleopRuntime, CanonicalTeleopTargets
from roboverse_pack.teleop.transforms import default_avp_transform_profile

ControlMode = Literal["hold", "normalized_action", "joint_target_q"]


@dataclass(frozen=True)
class ControlCommand:
    """Native MetaSim control command emitted by teleop flow."""

    mode: ControlMode
    source: str
    normalized_action: tuple[float, ...] | None = None
    joint_target_q_rad: tuple[float, ...] | None = None
    meta: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.mode == "hold":
            if self.normalized_action is not None or self.joint_target_q_rad is not None:
                raise ValueError("hold command cannot carry action payloads")
            return
        if self.mode == "normalized_action":
            if self.normalized_action is None or self.joint_target_q_rad is not None:
                raise ValueError("normalized_action mode requires only normalized_action")
            return
        if self.mode == "joint_target_q":
            if self.joint_target_q_rad is None or self.normalized_action is not None:
                raise ValueError("joint_target_q mode requires only joint_target_q_rad")
            return
        raise ValueError(f"unsupported control mode: {self.mode}")

    @classmethod
    def from_normalized_action(
        cls,
        action: Sequence[float],
        source: str,
        *,
        meta: dict[str, Any] | None = None,
    ) -> ControlCommand:
        """Build a normalized action command."""
        return cls(mode="normalized_action", source=source, normalized_action=tuple(float(value) for value in action), meta=meta)

    @classmethod
    def from_joint_targets(
        cls,
        joint_target_q_rad: Sequence[float],
        source: str,
        *,
        meta: dict[str, Any] | None = None,
    ) -> ControlCommand:
        """Build a joint target command."""
        return cls(mode="joint_target_q", source=source, joint_target_q_rad=tuple(float(value) for value in joint_target_q_rad), meta=meta)


@dataclass(frozen=True)
class StepResult:
    """Small native session step result."""

    observation: object
    info: dict[str, Any]


def _joint_slice(task_spec: BenchmarkTaskSpec, robot: str, name: str) -> tuple[int, int] | None:
    return task_spec.robot_profile(robot).joint_slices.get(name)


def _baseline_joint_targets(observation: object) -> tuple[float, ...] | None:
    robots = getattr(observation, "robots", None)
    if not isinstance(robots, Mapping) or not robots:
        return None
    robot_state = next(iter(robots.values()))
    if hasattr(robot_state, "joint_pos_rad"):
        return tuple(float(value) for value in robot_state.joint_pos_rad)
    joint_pos = getattr(robot_state, "joint_pos", None)
    if joint_pos is None:
        return None
    try:
        first_env = joint_pos[0]
    except Exception:
        first_env = joint_pos
    if hasattr(first_env, "detach"):
        first_env = first_env.detach().cpu().tolist()
    return tuple(float(value) for value in first_env)


def _control_length(task_spec: BenchmarkTaskSpec, robot: str, baseline_joint_targets: Sequence[float] | None) -> int:
    if baseline_joint_targets is not None:
        return len(tuple(float(value) for value in baseline_joint_targets))
    return task_spec.robot_profile(robot).control_joint_count


def _apply_joint_targets(
    values: list[float],
    joint_slice: tuple[int, int] | None,
    targets: Sequence[float] | None,
    *,
    label: str,
) -> None:
    if targets is None or joint_slice is None:
        return
    start, end = joint_slice
    target_values = [float(value) for value in targets]
    expected = end - start
    if expected != len(target_values):
        raise ValueError(f"{label} length mismatch: expected {expected}, got {len(target_values)}")
    values[start:end] = target_values


def _apply_close_ratio(values: list[float], joint_slice: tuple[int, int] | None, close_ratio: float) -> None:
    if joint_slice is None:
        return
    start, end = joint_slice
    normalized = max(-1.0, min(1.0, 2.0 * float(close_ratio) - 1.0))
    values[start:end] = [normalized] * max(0, end - start)


def _teleop_meta(targets: CanonicalTeleopTargets, *, control_mapping: str) -> dict[str, Any]:
    payload = targets.as_payload()
    payload["control_mapping"] = control_mapping
    return {"teleop": payload}


def teleop_targets_to_control_command(
    task_spec: BenchmarkTaskSpec,
    robot: str,
    targets: CanonicalTeleopTargets,
    baseline_joint_targets: Sequence[float] | None,
) -> ControlCommand:
    """Convert canonical teleop targets into robot joint control."""
    left_hand_slice = _joint_slice(task_spec, robot, "left_hand")
    right_hand_slice = _joint_slice(task_spec, robot, "right_hand")
    control_length = _control_length(task_spec, robot, baseline_joint_targets)
    if control_length <= 0:
        raise ValueError(f"task {task_spec.name} does not expose teleop control joints for robot {robot}")

    has_hand_targets = targets.left_hand_target_q_rad is not None or targets.right_hand_target_q_rad is not None
    if has_hand_targets:
        joint_targets = [0.0] * control_length if baseline_joint_targets is None else [float(value) for value in baseline_joint_targets]
        _apply_joint_targets(joint_targets, left_hand_slice, targets.left_hand_target_q_rad, label="left hand target")
        _apply_joint_targets(joint_targets, right_hand_slice, targets.right_hand_target_q_rad, label="right hand target")
        return ControlCommand.from_joint_targets(
            joint_targets,
            source="teleop",
            meta=_teleop_meta(targets, control_mapping="hand_joint_targets"),
        )

    normalized_action = [0.0] * control_length
    _apply_close_ratio(normalized_action, left_hand_slice, targets.left_close_ratio)
    _apply_close_ratio(normalized_action, right_hand_slice, targets.right_close_ratio)
    return ControlCommand.from_normalized_action(
        normalized_action,
        source="teleop",
        meta=_teleop_meta(targets, control_mapping="grip_fallback"),
    )


def _reference_pose_provider(_hand_key: str) -> tuple[float, float, float, float, float, float, float]:
    return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)


def _decode_targets(teleop_runtime: CanonicalTeleopRuntime, packet: object) -> CanonicalTeleopTargets:
    if isinstance(packet, CanonicalTeleopTargets):
        return packet
    return teleop_runtime.decode_packet(packet)


def _validate_native_request(task_spec: BenchmarkTaskSpec, robot: str, simulator: str) -> None:
    if simulator not in task_spec.supported_simulators:
        supported = ", ".join(task_spec.supported_simulators)
        raise ValueError(f"task {task_spec.name} does not support simulator {simulator!r}; supported: {supported}")
    task_spec.robot_profile(robot)


def _create_native_session(task_spec: BenchmarkTaskSpec, robot: str, simulator: str, *, headless: bool):
    from roboverse_pack.tasks.benchmark.base import NativeBenchmarkSession, build_benchmark_scenario

    scenario = build_benchmark_scenario(task_spec, robot=robot, simulator=simulator, headless=headless)
    return NativeBenchmarkSession(scenario)


def run_native_task_teleop_flow(
    *,
    task: str,
    robot: str | None = None,
    simulator: str,
    packets: Iterable[object] = (),
    session=None,
    teleop_runtime: CanonicalTeleopRuntime | None = None,
    record_run: bool = False,
    headless: bool = True,
) -> dict[str, Any]:
    """Run native-only benchmark teleop packets through one simulation session."""
    if record_run:
        raise NotImplementedError("native benchmark teleop recording is not wired yet")

    task_spec = get_benchmark_task_spec(task)
    selected_robot = task_spec.default_robot if robot is None else str(robot)
    _validate_native_request(task_spec, selected_robot, simulator)

    owns_session = session is None
    if session is None:
        session = _create_native_session(task_spec, selected_robot, simulator, headless=headless)

    try:
        initial_observation, _ = session.reset()
        if teleop_runtime is None:
            teleop_runtime = CanonicalTeleopRuntime(
                reference_pose_provider=_reference_pose_provider,
                transform_profile=default_avp_transform_profile(simulator),
            )

        frame_count = 0
        baseline_joint_targets = _baseline_joint_targets(initial_observation)
        for packet in packets:
            targets = _decode_targets(teleop_runtime, packet)
            control = teleop_targets_to_control_command(task_spec, selected_robot, targets, baseline_joint_targets)
            step_result = session.step(control)
            observation = getattr(step_result, "observation", step_result[0] if isinstance(step_result, tuple) else None)
            baseline_joint_targets = _baseline_joint_targets(observation) or baseline_joint_targets
            frame_count += 1

        return {
            "task": task_spec.name,
            "robot": selected_robot,
            "simulator": simulator,
            "backend": f"{simulator}+native",
            "frame_count": frame_count,
        }
    finally:
        if owns_session or session is not None:
            session.close()


__all__ = [
    "ControlCommand",
    "StepResult",
    "run_native_task_teleop_flow",
    "teleop_targets_to_control_command",
]
