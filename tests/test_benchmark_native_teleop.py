from __future__ import annotations

# ruff: noqa: D103
import importlib.util
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from roboverse_pack.benchmark import get_benchmark_task_spec, list_benchmark_task_specs
from roboverse_pack.robots.openarm_bimanual_wuji_cfg import OpenarmBimanualWujiCfg
from roboverse_pack.teleop.flow import run_native_task_teleop_flow
from roboverse_pack.teleop.runtime import CanonicalTeleopTargets

ROOT = Path(__file__).resolve().parents[1]


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
        self.reset_observation = _Observation(robots={"robot": _RobotState(joint_pos_rad=self._joint_targets)})

    def reset(self):
        return self.reset_observation, {"backend": "isaacsim+native"}

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


class _FakeRestorableSession(_FakeSession):
    def __init__(self) -> None:
        super().__init__()
        self.restored_states = []

    def restore_state(self, state):
        self.restored_states.append(state)
        return SimpleNamespace(
            observation=state,
            info={"backend": "isaacsim+native", "controller": "script:initial_pose"},
        )

    def step(self, control):
        raise AssertionError("hold_initial_pose should restore the reset state instead of stepping control")


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


def test_bidexbench_cube_reach_runner_opens_viewer_by_default() -> None:
    script_path = ROOT / "scripts" / "advanced" / "run_bidexbench_cube_reach.py"
    spec = importlib.util.spec_from_file_location("run_bidexbench_cube_reach", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    parser = module._build_parser()

    assert parser.parse_args([]).headless is False
    assert parser.parse_args(["--headless"]).headless is True
    assert parser.parse_args(["--viewer"]).headless is False


def test_bidexbench_cube_reach_runner_accepts_hybrid_renderer() -> None:
    script_path = ROOT / "scripts" / "advanced" / "run_bidexbench_cube_reach.py"
    spec = importlib.util.spec_from_file_location("run_bidexbench_cube_reach", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    parser = module._build_parser()
    args = parser.parse_args(["--sim", "mujoco", "--renderer", "isaacsim"])

    assert args.sim == "mujoco"
    assert args.renderer == "isaacsim"


def test_bidexbench_cube_reach_runner_accepts_physics_viewer_debug_option() -> None:
    script_path = ROOT / "scripts" / "advanced" / "run_bidexbench_cube_reach.py"
    spec = importlib.util.spec_from_file_location("run_bidexbench_cube_reach", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    parser = module._build_parser()

    assert parser.parse_args([]).physics_viewer is False
    assert parser.parse_args(["--physics-viewer"]).physics_viewer is True


def test_bidexbench_cube_reach_runner_accepts_initial_pose_hold_option() -> None:
    script_path = ROOT / "scripts" / "advanced" / "run_bidexbench_cube_reach.py"
    spec = importlib.util.spec_from_file_location("run_bidexbench_cube_reach", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    parser = module._build_parser()

    assert parser.parse_args([]).hold_initial_pose is False
    assert parser.parse_args(["--hold-initial-pose"]).hold_initial_pose is True


def test_bidexbench_cube_reach_runner_preconfigures_mujoco_egl_for_split_render(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script_path = ROOT / "scripts" / "advanced" / "run_bidexbench_cube_reach.py"
    spec = importlib.util.spec_from_file_location("run_bidexbench_cube_reach", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    monkeypatch.delenv("MUJOCO_GL", raising=False)
    module._preconfigure_mujoco_gl(["--sim", "mujoco", "--renderer", "isaacsim"])

    assert os.environ["MUJOCO_GL"] == "egl"


def test_bidexbench_cube_reach_runner_does_not_force_egl_for_physics_viewer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script_path = ROOT / "scripts" / "advanced" / "run_bidexbench_cube_reach.py"
    spec = importlib.util.spec_from_file_location("run_bidexbench_cube_reach", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    monkeypatch.delenv("MUJOCO_GL", raising=False)
    module._preconfigure_mujoco_gl(["--sim", "mujoco", "--renderer", "isaacsim", "--physics-viewer"])

    assert "MUJOCO_GL" not in os.environ


def test_teleop_flow_reports_hybrid_backend_when_renderer_is_requested() -> None:
    result = run_native_task_teleop_flow(
        task="benchmark.cube_reach",
        robot="openarm_bimanual_wuji",
        simulator="mujoco",
        renderer="isaacsim",
        packets=[],
        session=_FakeSession(),
        record_run=False,
    )

    assert result["simulator"] == "mujoco"
    assert result["renderer"] == "isaacsim"
    assert result["backend"] == "mujoco+isaacsim"


def test_native_teleop_flow_can_hold_reset_initial_joint_targets() -> None:
    session = _FakeSession()

    result = run_native_task_teleop_flow(
        task="benchmark.cube_reach",
        robot="openarm_bimanual_wuji",
        simulator="isaacsim",
        packets=[object()],
        session=session,
        record_run=False,
        hold_initial_pose=True,
    )

    assert result["control_mode"] == "initial_pose"
    assert len(session.controls) == 1
    control = session.controls[0]
    assert control.mode == "joint_target_q"
    assert control.source == "script:initial_pose"
    assert control.joint_target_q_rad == pytest.approx(tuple(float(index) for index in range(54)))


def test_initial_pose_hold_uses_session_canonical_joint_order_when_available() -> None:
    session = _FakeSession()
    canonical_initial = tuple(100.0 + float(index) for index in range(54))
    session.current_joint_targets = lambda: canonical_initial

    run_native_task_teleop_flow(
        task="benchmark.cube_reach",
        robot="openarm_bimanual_wuji",
        simulator="isaacsim",
        packets=[object()],
        session=session,
        record_run=False,
        hold_initial_pose=True,
    )

    assert session.controls[0].joint_target_q_rad == pytest.approx(canonical_initial)


def test_initial_pose_hold_restores_reset_state_without_stepping_when_available() -> None:
    session = _FakeRestorableSession()

    result = run_native_task_teleop_flow(
        task="benchmark.cube_reach",
        robot="openarm_bimanual_wuji",
        simulator="isaacsim",
        packets=[object(), object()],
        session=session,
        record_run=False,
        hold_initial_pose=True,
    )

    assert result["control_mode"] == "initial_pose"
    assert result["frame_count"] == 2
    assert session.restored_states == [session.reset_observation, session.reset_observation]
    assert session.controls == []


def test_native_benchmark_session_current_joint_targets_reorders_tensor_state_by_joint_name() -> None:
    import roboverse_pack.tasks.benchmark.base as benchmark_base

    robot = SimpleNamespace(
        name="robot",
        default_joint_positions={"joint_b": 0.0, "joint_a": 0.0},
        joint_limits={"joint_b": (-1.0, 1.0), "joint_a": (-1.0, 1.0)},
    )
    state = SimpleNamespace(
        robots={
            "robot": SimpleNamespace(
                joint_pos=[[1.0, 2.0]],
            )
        }
    )

    class _FakeHandler:
        robots = [robot]

        def get_states(self, mode="tensor"):
            assert mode == "tensor"
            return state

        def get_joint_names(self, _robot_name, sort=True):
            return ["joint_a", "joint_b"] if sort else ["joint_b", "joint_a"]

    session = object.__new__(benchmark_base.NativeBenchmarkSession)
    session.handler = _FakeHandler()

    assert session.current_joint_targets() == pytest.approx((2.0, 1.0))


def test_native_benchmark_session_restore_state_sets_handler_state_and_refreshes_render() -> None:
    import roboverse_pack.tasks.benchmark.base as benchmark_base

    class _FakeHandler:
        def __init__(self) -> None:
            self.set_states_calls = []
            self.refreshed = False

        def set_states(self, state) -> None:
            self.set_states_calls.append(state)

        def refresh_render(self) -> None:
            self.refreshed = True

        def get_states(self, mode="tensor"):
            assert mode == "tensor"
            return "fresh-state"

    handler = _FakeHandler()
    session = object.__new__(benchmark_base.NativeBenchmarkSession)
    session.handler = handler
    session.backend_label = "mujoco+native"

    result = session.restore_state("initial-state")

    assert handler.set_states_calls == ["initial-state"]
    assert handler.refreshed is True
    assert result.observation == "fresh-state"
    assert result.info == {"backend": "mujoco+native", "controller": "script:initial_pose"}


def test_hybrid_benchmark_session_restore_state_sets_physics_and_syncs_render(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import roboverse_pack.tasks.benchmark.base as benchmark_base

    class _FakePhysicsHandler:
        def __init__(self) -> None:
            self.set_states_calls = []

        def set_states(self, state) -> None:
            self.set_states_calls.append(state)

    physics_handler = _FakePhysicsHandler()
    render_handler = object()
    sync_calls = []

    def _fake_sync_render_from_physics(physics, render):
        sync_calls.append((physics, render))
        return "synced-state"

    monkeypatch.setattr(benchmark_base, "_sync_render_from_physics", _fake_sync_render_from_physics)

    session = object.__new__(benchmark_base.HybridBenchmarkSession)
    session.physics_handler = physics_handler
    session.render_handler = render_handler
    session.backend_label = "mujoco+isaacsim"

    result = session.restore_state("initial-state")

    assert physics_handler.set_states_calls == ["initial-state"]
    assert sync_calls == [(physics_handler, render_handler)]
    assert result.observation == "synced-state"
    assert result.info == {"backend": "mujoco+isaacsim", "controller": "script:initial_pose"}


def test_hybrid_teleop_flow_can_open_physics_backend_viewer(monkeypatch: pytest.MonkeyPatch) -> None:
    import roboverse_pack.teleop.flow as teleop_flow

    task_spec = get_benchmark_task_spec("benchmark.cube_reach")
    scenarios = []

    def _fake_build_benchmark_scenario(task_spec, *, robot, simulator, headless, camera_names=None):
        scenario = SimpleNamespace(
            task_spec=task_spec,
            robot=robot,
            simulator=simulator,
            headless=headless,
            camera_names=camera_names,
        )
        scenarios.append(scenario)
        return scenario

    class _FakeHybridBenchmarkSession:
        def __init__(self, physics_scenario, render_scenario) -> None:
            self.physics_scenario = physics_scenario
            self.render_scenario = render_scenario

    monkeypatch.setattr("roboverse_pack.tasks.benchmark.base.build_benchmark_scenario", _fake_build_benchmark_scenario)
    monkeypatch.setattr("roboverse_pack.tasks.benchmark.base.HybridBenchmarkSession", _FakeHybridBenchmarkSession)

    session = teleop_flow._create_native_session(
        task_spec,
        "openarm_bimanual_wuji",
        "mujoco",
        renderer="isaacsim",
        headless=False,
        physics_viewer=True,
    )

    assert session.physics_scenario.headless is False
    assert session.physics_scenario.camera_names == ()
    assert session.render_scenario.headless is False
    assert [scenario.simulator for scenario in scenarios] == ["mujoco", "isaacsim"]


def test_benchmark_session_applies_mujoco_dm_control_patch_before_launch(monkeypatch: pytest.MonkeyPatch) -> None:
    import roboverse_pack.tasks.benchmark.base as benchmark_base

    calls = []
    scenario = SimpleNamespace(simulator="mujoco", headless=True)

    monkeypatch.delenv("MUJOCO_GL", raising=False)
    monkeypatch.setattr(benchmark_base, "_apply_dm_control_struct_compat_patch", lambda: calls.append("patch"))
    monkeypatch.setattr(
        benchmark_base,
        "get_handler",
        lambda scenario: calls.append(f"handler:{scenario.simulator}") or object(),
    )

    benchmark_base._get_handler_with_mujoco_compat(scenario)

    assert calls == ["patch", "handler:mujoco"]
    assert os.environ["MUJOCO_GL"] == "egl"


def test_hybrid_benchmark_session_uses_explicit_split_render_sync(monkeypatch: pytest.MonkeyPatch) -> None:
    import metasim.sim
    import roboverse_pack.tasks.benchmark.base as benchmark_base
    from roboverse_pack.teleop.flow import ControlCommand

    physics_state = SimpleNamespace(cameras={"physics_camera": object()})
    calls = []

    class _ForbiddenHybridHandler:
        def __init__(self, *_args, **_kwargs) -> None:
            raise AssertionError("benchmark split render should not use generic HybridSimHandler")

    class _FakePhysicsHandler:
        def get_states(self, mode="tensor"):
            calls.append(("physics_get_states", mode))
            return physics_state

        def set_dof_targets(self, actions):
            calls.append(("physics_set_dof_targets", actions))

        def simulate(self):
            calls.append(("physics_simulate", None))

        def close(self):
            calls.append(("physics_close", None))

    class _FakeRenderHandler:
        set_states_refreshes = True

        def set_states(self, state):
            calls.append(("render_set_states", state))

        def close(self):
            calls.append(("render_close", None))

    handlers = [_FakePhysicsHandler(), _FakeRenderHandler()]

    monkeypatch.setattr(metasim.sim, "HybridSimHandler", _ForbiddenHybridHandler)
    monkeypatch.setattr(benchmark_base, "_get_handler_with_mujoco_compat", lambda _scenario: handlers.pop(0))
    monkeypatch.setattr(benchmark_base, "control_command_to_metasim_actions", lambda _handler, _control: "actions")

    session = benchmark_base.HybridBenchmarkSession(
        SimpleNamespace(simulator="mujoco"),
        SimpleNamespace(simulator="isaacsim"),
    )

    result = session.step(ControlCommand(mode="hold", source="test"))
    session.close()

    assert result.observation is physics_state
    assert ("physics_set_dof_targets", "actions") in calls
    assert ("physics_simulate", None) in calls
    assert ("render_set_states", physics_state) in calls
    assert calls[-2:] == [("physics_close", None), ("render_close", None)]


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


def test_native_teleop_flow_records_reset_and_step_states(tmp_path: Path) -> None:
    class _RecordingFakeSession(_FakeSession):
        def __init__(self) -> None:
            super().__init__()
            self.observations = []

        def step(self, control):
            result = super().step(control)
            self.observations.append(result.observation)
            return result

    session = _RecordingFakeSession()
    output_path = tmp_path / "states.pt"

    result = run_native_task_teleop_flow(
        task="benchmark.cube_reach",
        robot="openarm_bimanual_wuji",
        simulator="isaacsim",
        packets=[
            CanonicalTeleopTargets(
                left_work_pose_cm_xyzw=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
                right_work_pose_cm_xyzw=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
                left_close_ratio=0.0,
                right_close_ratio=1.0,
            ),
            CanonicalTeleopTargets(
                left_work_pose_cm_xyzw=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
                right_work_pose_cm_xyzw=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
                left_close_ratio=1.0,
                right_close_ratio=0.0,
            ),
        ],
        session=session,
        record_run=False,
        record_states_path=output_path,
    )

    import torch

    payload = torch.load(output_path, weights_only=False)
    assert payload["format"] == "roboverse_benchmark_tensor_states_v1"
    assert payload["task"] == "benchmark.cube_reach"
    assert payload["robot"] == "openarm_bimanual_wuji"
    assert payload["simulator"] == "isaacsim"
    assert payload["renderer"] is None
    assert payload["frame_count"] == 3
    assert payload["states"] == [session.reset_observation, *session.observations]
    assert result["record_states_path"] == str(output_path)


def test_bidexbench_cube_reach_runner_accepts_offline_blender_render_options() -> None:
    script_path = ROOT / "scripts" / "advanced" / "run_bidexbench_cube_reach.py"
    spec = importlib.util.spec_from_file_location("run_bidexbench_cube_reach", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    parser = module._build_parser()
    args = parser.parse_args(
        [
            "--sim",
            "mujoco",
            "--steps",
            "3",
            "--record-states",
            "/tmp/cube_reach_states.pt",
            "--offline-renderer",
            "blender",
            "--render-output",
            "/tmp/cube_reach_blender",
            "--render-samples",
            "8",
            "--render-device",
            "CPU",
        ]
    )

    assert args.record_states == Path("/tmp/cube_reach_states.pt")
    assert args.offline_renderer == "blender"
    assert args.render_output == Path("/tmp/cube_reach_blender")
    assert args.render_samples == 8
    assert args.render_device == "CPU"


def test_bidexbench_cube_reach_runner_calls_blender_offline_renderer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    script_path = ROOT / "scripts" / "advanced" / "run_bidexbench_cube_reach.py"
    spec = importlib.util.spec_from_file_location("run_bidexbench_cube_reach_offline", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    import torch

    calls = []

    class _FakeBlenderOfflineRenderCfg:
        def __init__(self, *, output_dir, samples, device) -> None:
            self.output_dir = output_dir
            self.samples = samples
            self.device = device

    def _fake_render_state_sequence(scenario, states, cfg):
        calls.append((scenario, states, cfg))
        return [Path(cfg.output_dir) / "frame_0000.png"]

    fake_offline_module = SimpleNamespace(
        BlenderOfflineRenderCfg=_FakeBlenderOfflineRenderCfg,
        render_state_sequence=_fake_render_state_sequence,
    )
    monkeypatch.setitem(sys.modules, "metasim.sim.blender.offline", fake_offline_module)

    def _fake_flow(**kwargs):
        kwargs["record_states_path"].parent.mkdir(parents=True, exist_ok=True)
        torch.save({"states": ["reset-state", "step-state"]}, kwargs["record_states_path"])
        return {"frame_count": 1, "record_states_path": str(kwargs["record_states_path"])}

    monkeypatch.setattr(module, "run_native_task_teleop_flow", _fake_flow)

    built_scenarios = []

    def _fake_build_benchmark_scenario(task_spec, *, robot, simulator, headless):
        scenario = SimpleNamespace(task=task_spec.name, robot=robot, simulator=simulator, headless=headless)
        built_scenarios.append(scenario)
        return scenario

    import roboverse_pack.tasks.benchmark.base as benchmark_base

    monkeypatch.setattr(benchmark_base, "build_benchmark_scenario", _fake_build_benchmark_scenario)

    output_dir = tmp_path / "frames"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_bidexbench_cube_reach.py",
            "--steps",
            "1",
            "--offline-renderer",
            "blender",
            "--render-output",
            str(output_dir),
            "--render-samples",
            "8",
            "--render-device",
            "CPU",
        ],
    )

    module.main()

    assert len(calls) == 1
    scenario, states, cfg = calls[0]
    assert built_scenarios == [scenario]
    assert scenario.task == "benchmark.cube_reach"
    assert scenario.robot == "openarm_bimanual_wuji"
    assert scenario.simulator == "isaacsim"
    assert scenario.headless is True
    assert states == ["reset-state", "step-state"]
    assert cfg.output_dir == output_dir
    assert cfg.samples == 8
    assert cfg.device == "CPU"


def test_openarm_bimanual_wuji_usd_imports_in_blender_when_bpy_available() -> None:
    bpy = pytest.importorskip("bpy")

    robot = OpenarmBimanualWujiCfg()
    usd_path = Path(robot.usd_path)
    if not usd_path.is_absolute():
        usd_path = ROOT / usd_path

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    result = bpy.ops.wm.usd_import(filepath=str(usd_path))

    mesh_count = sum(1 for obj in bpy.data.objects if obj.type == "MESH")
    named_links = {obj.name.split(".")[0] for obj in bpy.data.objects}

    assert "FINISHED" in result
    assert mesh_count > 0
    assert "left_palm_link" in named_links
    assert "right_palm_link" in named_links
