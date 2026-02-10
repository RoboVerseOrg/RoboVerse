from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from metasim.protocol_sim.core.task_alignment import apply_task_initial_state
from metasim.types import RobotState, TensorState


class _DummyRobotCfg:
    def __init__(self, name: str, default_joint_positions: dict[str, float]):
        self.name = name
        self.default_joint_positions = default_joint_positions


class _DummyHandler:
    def __init__(
        self,
        *,
        robot_name: str,
        joint_names: list[str],
        root_state: torch.Tensor,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
        default_joint_positions: dict[str, float],
    ):
        self.robots = [_DummyRobotCfg(robot_name, default_joint_positions)]
        self._joint_names = list(joint_names)
        self.set_states_calls = 0

        self._ts = TensorState(
            objects={},
            robots={
                robot_name: RobotState(
                    root_state=root_state.clone(),
                    body_names=[],
                    body_state=torch.zeros((1, 0, 13), dtype=root_state.dtype),
                    joint_pos=joint_pos.clone(),
                    joint_vel=joint_vel.clone(),
                    joint_pos_target=torch.zeros_like(joint_pos),
                    joint_vel_target=torch.zeros_like(joint_vel),
                    joint_effort_target=torch.zeros_like(joint_vel),
                )
            },
            cameras={},
        )

    def get_joint_names(self, obj_name: str, sort: bool = True):
        assert obj_name in self._ts.robots
        names = list(self._joint_names)
        if sort:
            names.sort()
        return names

    def get_states(self):
        return self._ts

    def set_states(self, ts):
        self._ts = ts
        self.set_states_calls += 1


def _make_env_cfg(robot_name: str, init: dict) -> SimpleNamespace:
    return SimpleNamespace(initial_states=SimpleNamespace(robots={robot_name: init}))


@pytest.mark.general
def test_apply_task_initial_state_applies_joint_overrides_when_pos_missing():
    robot_name = "r1"
    root_initial = torch.tensor(
        [[9.0, 8.0, 7.0, 0.0, 1.0, 0.0, 0.0, 0.3, 0.2, 0.1, 0.4, 0.5, 0.6]],
        dtype=torch.float32,
    )
    handler = _DummyHandler(
        robot_name=robot_name,
        joint_names=["j1", "j0"],
        root_state=root_initial,
        joint_pos=torch.tensor([[9.0, 9.0]], dtype=torch.float32),
        joint_vel=torch.tensor([[1.0, 2.0]], dtype=torch.float32),
        default_joint_positions={"j0": 0.1, "j1": 0.2},
    )
    env_cfg = _make_env_cfg(robot_name, init={"default_joint_pos": {"j1": 1.5}})

    apply_task_initial_state(handler=handler, robot_name=robot_name, env_cfg=env_cfg)

    rs = handler.get_states().robots[robot_name]
    assert torch.allclose(rs.root_state, root_initial)
    assert torch.allclose(rs.joint_pos[0], torch.tensor([0.1, 1.5], dtype=torch.float32))
    assert torch.allclose(rs.joint_vel[0], torch.zeros((2,), dtype=torch.float32))
    assert handler.set_states_calls == 1


@pytest.mark.general
def test_apply_task_initial_state_uses_pos_fallback_when_pos_missing():
    robot_name = "r1"
    handler = _DummyHandler(
        robot_name=robot_name,
        joint_names=["j1", "j0"],
        root_state=torch.zeros((1, 13), dtype=torch.float32),
        joint_pos=torch.tensor([[9.0, 9.0]], dtype=torch.float32),
        joint_vel=torch.tensor([[1.0, 2.0]], dtype=torch.float32),
        default_joint_positions={"j0": 0.1, "j1": 0.2},
    )
    env_cfg = _make_env_cfg(robot_name, init={"default_joint_pos": {"j1": 1.5}})

    apply_task_initial_state(
        handler=handler,
        robot_name=robot_name,
        env_cfg=env_cfg,
        pos_fallback=(0.0, 0.0, 1.0),
    )

    rs = handler.get_states().robots[robot_name]
    assert torch.allclose(rs.root_state[0, 0:3], torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32))
    assert torch.allclose(rs.root_state[0, 3:7], torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32))
    assert torch.allclose(rs.root_state[0, 7:13], torch.zeros((6,), dtype=torch.float32))
    assert torch.allclose(rs.joint_pos[0], torch.tensor([0.1, 1.5], dtype=torch.float32))
    assert torch.allclose(rs.joint_vel[0], torch.zeros((2,), dtype=torch.float32))
    assert handler.set_states_calls == 1
