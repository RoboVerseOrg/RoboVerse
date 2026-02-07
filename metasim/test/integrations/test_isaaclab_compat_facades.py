from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from metasim.integrations.isaaclab.compat.scene import CompatScene
from metasim.integrations.isaaclab.compat.utils import resolve_scene_entity_cfgs
from metasim.integrations.isaaclab.shim import ensure_isaaclab_shim
from metasim.scenario.robot import RobotCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.types import RobotState, TensorState


class _FakeHandler:
    def __init__(self, *, num_envs: int, joint_names: list[str], body_names: list[str], env_origins: torch.Tensor):
        self._num_envs = int(num_envs)
        self._joint_names = list(joint_names)
        self._body_names = list(body_names)
        self.scene = SimpleNamespace(env_origins=env_origins)
        self._last_set = None

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

    def set_states(self, states, env_ids=None):
        # record calls for assertions
        self._last_set = (states, env_ids)


@pytest.mark.general
def test_compat_articulation_world_frame_and_default_persistence():
    robot_name = "robot1"
    joint_names = ["b", "a"]
    body_names = ["link2", "base", "link1"]
    env_origins = torch.tensor([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=torch.float32)

    handler = _FakeHandler(num_envs=2, joint_names=joint_names, body_names=body_names, env_origins=env_origins)
    robot_cfg = RobotCfg(
        name=robot_name,
        joint_limits={"a": (-1.0, 1.0), "b": (-2.0, 2.0)},
        default_joint_positions={".*": 0.5},
    )
    scenario = ScenarioCfg(simulator="mujoco", num_envs=2, robots=[robot_cfg], objects=[], cameras=[], headless=True)

    scene = CompatScene(handler=handler, scenario=scenario, device=torch.device("cpu"))

    root_state_local = torch.zeros((2, 13), dtype=torch.float32)
    root_state_local[:, 3] = 1.0  # identity quat (w=1)
    body_state_local = torch.zeros((2, 3, 13), dtype=torch.float32)
    body_state_local[:, :, 3] = 1.0

    # Two joints in sorted order: a, b
    joint_pos = torch.tensor([[0.1, -0.2], [0.3, -0.4]], dtype=torch.float32)
    joint_vel = torch.zeros_like(joint_pos)

    rs = RobotState(
        root_state=root_state_local,
        body_names=sorted(body_names),
        body_state=body_state_local,
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        joint_pos_target=torch.zeros_like(joint_pos),
        joint_vel_target=torch.zeros_like(joint_pos),
        joint_effort_target=torch.zeros_like(joint_pos),
    )
    state = TensorState(objects={}, robots={robot_name: rs}, cameras={}, extras={})

    scene.update_from_states(state)

    art = scene[robot_name]
    assert torch.allclose(art.data.root_pos_w, env_origins)
    assert torch.allclose(art.data.body_pos_w[:, 0], env_origins)

    # Mutate default joints (as startup events do) and ensure refresh doesn't overwrite it.
    art.data.default_joint_pos[:] = 1.23
    scene.update_from_states(state)
    assert float(art.data.default_joint_pos[0, 0].item()) == pytest.approx(1.23)


@pytest.mark.general
def test_resolve_scene_entity_cfgs_populates_ids():
    ensure_isaaclab_shim()
    from isaaclab.managers import SceneEntityCfg  # type: ignore

    robot_name = "robot1"
    handler = _FakeHandler(
        num_envs=1,
        joint_names=["j0", "j1"],
        body_names=["base", "link"],
        env_origins=torch.zeros((1, 3), dtype=torch.float32),
    )
    robot_cfg = RobotCfg(
        name=robot_name,
        joint_limits={"j0": (-1.0, 1.0), "j1": (-1.0, 1.0)},
        default_joint_positions={".*": 0.0},
    )
    scenario = ScenarioCfg(simulator="mujoco", num_envs=1, robots=[robot_cfg], objects=[], cameras=[], headless=True)
    scene = CompatScene(handler=handler, scenario=scenario, device=torch.device("cpu"))
    scene.add_contact_sensor(name="contact_forces", body_names=scene[robot_name].body_names)

    cfg = SimpleNamespace(
        rewards=SimpleNamespace(
            joint_limit=SimpleNamespace(
                func=lambda env, asset_cfg: torch.zeros(env.num_envs),
                params={"asset_cfg": SceneEntityCfg(name=robot_name, joint_names=["j0"])},
            ),
            undesired_contacts=SimpleNamespace(
                func=lambda env, sensor_cfg, threshold: torch.zeros(env.num_envs),
                params={"sensor_cfg": SceneEntityCfg(name="contact_forces", body_names=["base"]), "threshold": 1.0},
            ),
        )
    )

    resolve_scene_entity_cfgs(cfg, scene=scene)

    asset_cfg = cfg.rewards.joint_limit.params["asset_cfg"]
    assert getattr(asset_cfg, "joint_ids", None) == [0]

    sensor_cfg = cfg.rewards.undesired_contacts.params["sensor_cfg"]
    assert getattr(sensor_cfg, "body_ids", None) == [0]
