from __future__ import annotations

# ruff: noqa: D103
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from roboverse_pack.benchmark import get_benchmark_task_spec, list_benchmark_task_specs
from roboverse_pack.robots.openarm_bimanual_wuji_cfg import OpenarmBimanualWujiCfg
from roboverse_pack.teleop.flow import run_native_task_teleop_flow
from roboverse_pack.teleop.runtime import CanonicalTeleopTargets


@dataclass(frozen=True)
class _RobotState:
    joint_pos_rad: tuple[float, ...]


@dataclass(frozen=True)
class _Observation:
    robots: dict[str, _RobotState]


class _FakeSession:
    def __init__(self) -> None:
        self.closed = False
        self.controls = []
        self._joint_targets = tuple(float(index) for index in range(54))

    def reset(self):
        return _Observation(robots={"robot": _RobotState(joint_pos_rad=self._joint_targets)}), {"backend": "isaacsim+native"}

    def step(self, control):
        self.controls.append(control)
        if control.mode == "joint_target_q":
            self._joint_targets = tuple(control.joint_target_q_rad or ())
        return SimpleNamespace(
            observation=_Observation(robots={"robot": _RobotState(joint_pos_rad=self._joint_targets)}),
            info={"backend": "isaacsim+native", "controller": control.source},
        )

    def close(self) -> None:
        self.closed = True


def test_benchmark_cube_reach_spec_separates_task_and_openarm_robot_profile() -> None:
    assert "benchmark.cube_reach" in list_benchmark_task_specs()

    spec = get_benchmark_task_spec("cube_reach")
    profile = spec.robot_profile("openarm_bimanual_wuji")

    assert spec.name == "benchmark.cube_reach"
    assert spec.default_robot == "openarm_bimanual_wuji"
    assert "isaacsim" in spec.supported_simulators
    assert spec.scene.objects[0].name == "cube"
    assert spec.scene.objects[0].movable is True
    assert profile.control_joint_count == 54
    assert profile.joint_slices["left_arm"] == (0, 7)
    assert profile.joint_slices["right_arm"] == (7, 14)
    assert profile.joint_slices["left_hand"] == (14, 34)
    assert profile.joint_slices["right_hand"] == (34, 54)


def test_openarm_bimanual_wuji_robot_cfg_uses_roboverse_data_assets() -> None:
    robot = OpenarmBimanualWujiCfg()

    assert robot.name == "openarm_bimanual_wuji"
    assert robot.num_joints == 54
    assert robot.urdf_path.endswith("roboverse_data/robots/openarm/openarm_bimanual_wuji_generated.urdf")
    assert robot.mjcf_path.endswith("roboverse_data/robots/openarm/openarm_wuji.xml")
    assert robot.usd_path.endswith("roboverse_data/robots/openarm/openarm_bimanual_wuji.usd")
    assert Path(robot.urdf_path).is_file()
    assert Path(robot.mjcf_path).is_file()
    assert Path(robot.usd_path).is_file()
    assert len(robot.joint_limits) == 54
    assert list(robot.default_joint_positions)[:7] == [f"openarm_left_joint{i}" for i in range(1, 8)]
    assert list(robot.default_joint_positions)[14:18] == [f"left_finger1_joint{i}" for i in range(1, 5)]


def test_native_teleop_flow_maps_hand_targets_into_joint_control() -> None:
    session = _FakeSession()
    targets = CanonicalTeleopTargets(
        left_work_pose_cm_xyzw=(10.0, 20.0, 30.0, 0.0, 0.0, 0.0, 1.0),
        right_work_pose_cm_xyzw=(40.0, 50.0, 60.0, 0.0, 0.0, 0.0, 1.0),
        left_close_ratio=0.1,
        right_close_ratio=0.9,
        left_hand_target_q_rad=tuple(0.1 for _ in range(20)),
        right_hand_target_q_rad=tuple(0.2 for _ in range(20)),
    )

    result = run_native_task_teleop_flow(
        task="benchmark.cube_reach",
        robot="openarm_bimanual_wuji",
        simulator="isaacsim",
        packets=[targets],
        session=session,
        record_run=False,
    )

    assert result["task"] == "benchmark.cube_reach"
    assert result["robot"] == "openarm_bimanual_wuji"
    assert result["simulator"] == "isaacsim"
    assert result["frame_count"] == 1
    assert session.closed is True

    assert len(session.controls) == 1
    control = session.controls[0]
    assert control.mode == "joint_target_q"
    assert control.source == "teleop"
    assert control.joint_target_q_rad[:14] == pytest.approx(tuple(float(index) for index in range(14)))
    assert control.joint_target_q_rad[14:34] == pytest.approx(tuple(0.1 for _ in range(20)))
    assert control.joint_target_q_rad[34:54] == pytest.approx(tuple(0.2 for _ in range(20)))
    assert control.meta["teleop"]["control_mapping"] == "hand_joint_targets"


def test_native_teleop_flow_uses_grip_fallback_without_retargeted_hand_joints() -> None:
    session = _FakeSession()
    targets = CanonicalTeleopTargets(
        left_work_pose_cm_xyzw=(10.0, 20.0, 30.0, 0.0, 0.0, 0.0, 1.0),
        right_work_pose_cm_xyzw=(40.0, 50.0, 60.0, 0.0, 0.0, 0.0, 1.0),
        left_close_ratio=0.25,
        right_close_ratio=0.75,
    )

    run_native_task_teleop_flow(
        task="benchmark.cube_reach",
        robot="openarm_bimanual_wuji",
        simulator="isaacsim",
        packets=[targets],
        session=session,
        record_run=False,
    )

    control = session.controls[0]
    assert control.mode == "normalized_action"
    assert control.normalized_action[:14] == pytest.approx(tuple(0.0 for _ in range(14)))
    assert control.normalized_action[14:34] == pytest.approx(tuple(-0.5 for _ in range(20)))
    assert control.normalized_action[34:54] == pytest.approx(tuple(0.5 for _ in range(20)))
    assert control.meta["teleop"]["control_mapping"] == "grip_fallback"


def test_native_teleop_flow_rejects_unsupported_simulator_for_task() -> None:
    with pytest.raises(ValueError, match="does not support simulator"):
        run_native_task_teleop_flow(
            task="benchmark.cube_reach",
            robot="openarm_bimanual_wuji",
            simulator="newton",
            packets=[],
            session=_FakeSession(),
            record_run=False,
        )
