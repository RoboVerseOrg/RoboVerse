from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest
import torch

from metasim.integrations.isaaclab.compat.contract import WarnOnce
from metasim.integrations.isaaclab.compat.scene import CompatScene
from metasim.integrations.isaaclab.compat.sensor_registry import SensorRegistry
from metasim.scenario.robot import RobotCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.types import CameraState, RobotState, TensorState


class _FakeHandler:
    def __init__(self, *, num_envs: int, joint_names: list[str], body_names: list[str], env_origins: torch.Tensor):
        self._num_envs = int(num_envs)
        self._joint_names = list(joint_names)
        self._body_names = list(body_names)
        self.scene = SimpleNamespace(env_origins=env_origins)

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    def get_joint_names(self, _obj_name: str, sort: bool = True) -> list[str]:
        out = list(self._joint_names)
        return sorted(out) if sort else out

    def get_body_names(self, _obj_name: str, sort: bool = True) -> list[str]:
        out = list(self._body_names)
        return sorted(out) if sort else out


@pytest.mark.general
def test_compat_camera_sensor_updates_from_handler_camera_state():
    handler = _FakeHandler(
        num_envs=1,
        joint_names=["j0"],
        body_names=["base"],
        env_origins=torch.zeros((1, 3), dtype=torch.float32),
    )
    robot_cfg = RobotCfg(
        name="robot",
        joint_limits={"j0": (-1.0, 1.0)},
        default_joint_positions={".*": 0.0},
    )
    scenario = ScenarioCfg(simulator="mujoco", num_envs=1, robots=[robot_cfg], objects=[], cameras=[], headless=True)
    scene = CompatScene(handler=handler, scenario=scenario, device=torch.device("cpu"))
    scene.add_camera_sensor(name="cam", data_types=["rgb", "depth"], source_camera="cam")

    root_state = torch.zeros((1, 13), dtype=torch.float32)
    root_state[:, 3] = 1.0
    body_state = torch.zeros((1, 1, 13), dtype=torch.float32)
    body_state[:, :, 3] = 1.0
    robot_state = RobotState(
        root_state=root_state,
        body_names=["base"],
        body_state=body_state,
        joint_pos=torch.zeros((1, 1), dtype=torch.float32),
        joint_vel=torch.zeros((1, 1), dtype=torch.float32),
        joint_pos_target=torch.zeros((1, 1), dtype=torch.float32),
        joint_vel_target=torch.zeros((1, 1), dtype=torch.float32),
        joint_effort_target=torch.zeros((1, 1), dtype=torch.float32),
    )

    rgb = torch.zeros((1, 2, 3, 3), dtype=torch.uint8)
    depth = torch.ones((1, 2, 3), dtype=torch.float32)
    camera_state = CameraState(
        rgb=rgb,
        depth=depth,
        pos=torch.zeros((1, 3), dtype=torch.float32),
        quat_world=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
        intrinsics=torch.eye(3, dtype=torch.float32).unsqueeze(0),
    )

    state = TensorState(objects={}, robots={"robot": robot_state}, cameras={"cam": camera_state}, extras={})
    scene.update_from_states(state)

    cam = scene.sensors["cam"]
    assert cam.data is not None
    assert cam.data.image_shape == (2, 3)
    assert set(cam.data.output.keys()) == {"rgb", "depth"}
    assert cam.data.output["rgb"].shape == (1, 2, 3, 3)
    assert cam.data.output["depth"].shape == (1, 2, 3)


@dataclass
class CameraCfg:
    data_types: list[str] = field(default_factory=lambda: ["rgb", "depth"])
    width: int = 4
    height: int = 3


@dataclass
class RayCasterCfg:
    prim_path: str = "/World/Robot"


@dataclass
class ImuCfg:
    prim_path: str = "/World/Robot"


@dataclass
class _SceneCfg:
    cam: CameraCfg = field(default_factory=CameraCfg)
    ray: RayCasterCfg = field(default_factory=RayCasterCfg)
    imu: ImuCfg = field(default_factory=ImuCfg)


@pytest.mark.general
def test_sensor_registry_installs_camera_raycast_and_imu_stubs():
    handler = _FakeHandler(
        num_envs=2,
        joint_names=["j0"],
        body_names=["base"],
        env_origins=torch.zeros((2, 3), dtype=torch.float32),
    )
    robot_cfg = RobotCfg(
        name="robot",
        joint_limits={"j0": (-1.0, 1.0)},
        default_joint_positions={".*": 0.0},
    )
    scenario = ScenarioCfg(simulator="mujoco", num_envs=2, robots=[robot_cfg], objects=[], cameras=[], headless=True)
    scene = CompatScene(handler=handler, scenario=scenario, device=torch.device("cpu"))
    env = SimpleNamespace(scenario=scenario, num_envs=scenario.num_envs, device=torch.device("cpu"), scene=scene)

    cfg = SimpleNamespace(scene=_SceneCfg())
    registry = SensorRegistry(strict=True, warn_once=WarnOnce())
    registry.setup_scene_sensors(env=env, cfg=cfg, plan=None)

    assert set(env.scene.sensors.keys()) == {"cam", "ray", "imu"}
    assert env.scene.sensors["cam"].data is not None
    assert env.scene.sensors["cam"].data.output["rgb"].shape == (2, 3, 4, 3)
    assert env.scene.sensors["cam"].data.output["depth"].shape == (2, 3, 4)

    assert env.scene.sensors["ray"].data.ray_hits_w.shape == (2, 0, 3)
    assert env.scene.sensors["imu"].data.projected_gravity_b.shape == (2, 3)


@pytest.mark.general
def test_contact_sensor_tracks_air_and_contact_time_when_enabled():
    from metasim.integrations.isaaclab.compat.sensors import CompatContactSensor

    sensor = CompatContactSensor(body_names=["b0", "b1"], force_threshold=1.0, track_air_time=True, dt=0.1)

    def _extra(forces: torch.Tensor):
        return SimpleNamespace(
            contact_forces=forces,
            contact_forces_history=forces.unsqueeze(1),
        )

    # step 1: no contact → air time accumulates
    f0 = torch.zeros((1, 2, 3), dtype=torch.float32)
    sensor.update_from_extra(_extra(f0))
    assert sensor.data is not None
    assert float(sensor.data.current_air_time[0, 0].item()) == pytest.approx(0.1)

    # step 2: body0 contacts → capture last air time for body0, start contact time
    f1 = f0.clone()
    f1[0, 0, 0] = 2.0
    sensor.update_from_extra(_extra(f1))
    assert float(sensor.data.last_air_time[0, 0].item()) == pytest.approx(0.1)
    assert float(sensor.data.current_contact_time[0, 0].item()) == pytest.approx(0.1)

    # step 3: body0 lifts off → capture last contact time
    sensor.update_from_extra(_extra(f0))
    assert float(sensor.data.last_contact_time[0, 0].item()) == pytest.approx(0.1)
