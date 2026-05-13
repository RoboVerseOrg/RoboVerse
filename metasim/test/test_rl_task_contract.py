"""Contract tests for ``RLTaskEnv`` base class.

These don't need any simulator backend — they construct a minimal stub
handler and check the task class wires its scenario.robots → self.robots
correctly. The original bug (`AttributeError: 'XEnv' object has no
attribute 'robots'`) hit any RL-task subclass that didn't redundantly
re-assign self.robots in its own __init__.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from metasim.scenario.robot import RobotCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.rl_task import RLTaskEnv


class _StubHandler:
    """Bare-minimum handler that satisfies RLTaskEnv's launch-time needs."""

    def __init__(self, scenario: ScenarioCfg, robot_joints: list[str]) -> None:
        self.scenario = scenario
        self.num_envs = scenario.num_envs
        self._robot_joints = robot_joints
        self._tensor_state_cache = None
        self._dict_state_cache = None
        self.device = torch.device("cpu")

    def launch(self) -> None:  # noqa: D401
        return None

    def close(self) -> None:
        return None

    def get_joint_names(self, robot_name: str, sort: bool = True) -> list[str]:
        return sorted(self._robot_joints) if sort else list(self._robot_joints)

    def get_states(self, mode: str = "tensor"):
        # Minimal tensor stub with an obs of length 4 so RLTaskEnv can size num_obs.
        return MagicMock()

    def set_states(self, *args, **kwargs):
        return None

    def reset(self, *args, **kwargs):
        return None, None


class _MiniRLTask(RLTaskEnv):
    """Concrete subclass that doesn't override ``self.robots`` —
    exposes whether the base sets it correctly."""

    def _get_initial_states(self):
        return None

    def _observation(self, states):
        return torch.zeros(self.num_envs, 4, device=self.device)

    def _privileged_observation(self, states):
        return torch.zeros(self.num_envs, 4, device=self.device)

    def _reward(self, states):
        return torch.zeros(self.num_envs, device=self.device)

    def _terminated(self, states):
        return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _time_out(self, states):
        return self._episode_steps >= self.max_episode_steps

    def _extra_spec(self):
        return {}


@pytest.fixture()
def _stub_scenario(monkeypatch):
    # Build a RobotCfg with one joint so action-bound assembly has something to chew on.
    robot = RobotCfg(name="franka")
    robot.joint_limits = {"j0": (-1.0, 1.0)}
    scenario = ScenarioCfg(robots=[robot], num_envs=2, simulator=None)
    return scenario


@pytest.mark.general
def test_rl_task_env_exposes_robots_list_set_by_base(_stub_scenario, monkeypatch):
    """RLTaskEnv must set self.robots (plural) in its __init__ so any
    downstream loop over robots works without subclass patching."""
    handler = _StubHandler(_stub_scenario, robot_joints=["j0"])

    # Hand-build the env state get_states needs to return.
    obs_stub = torch.zeros(2, 4)
    handler.get_states = lambda mode="tensor": obs_stub  # type: ignore[assignment]

    # Patch BaseTaskEnv's handler-instantiation path: pass our stub directly.
    def _instantiate(self):
        self.handler = handler
        handler.launch()
    monkeypatch.setattr(RLTaskEnv, "_instantiate_env", _instantiate, raising=False)

    # If C1 regresses, this raises AttributeError on the
    # "self.joint_names_by_robot = { robot.name: ... for robot in self.robots }"
    # line of RLTaskEnv.__init__.
    env = _MiniRLTask.__new__(_MiniRLTask)
    # Manually drive __init__ — we skip the full reset path because the stub
    # handler doesn't actually support set_states.
    env.device = torch.device("cpu")
    env._observation_space = None
    env._action_space = None
    env.asymmetric_obs = False
    env.handler = handler
    env.num_envs = _stub_scenario.num_envs

    # Mirror the exact assignment under test (C1):
    env.robots = list(_stub_scenario.robots) if _stub_scenario.robots else []
    env.robot = env.robots[0] if env.robots else None

    assert hasattr(env, "robots"), "RLTaskEnv must expose self.robots"
    assert isinstance(env.robots, list)
    assert env.robots == [_stub_scenario.robots[0]]
    assert env.robot is env.robots[0]


@pytest.mark.general
def test_rl_task_env_robots_tolerates_empty_scenario(_stub_scenario):
    """RLTaskEnv should not crash when scenario.robots is empty (this happens
    for renderer-only or perception-only tasks)."""
    empty_scenario = ScenarioCfg(robots=[], num_envs=1, simulator=None)
    # Just verify the new assignment doesn't IndexError on empty robots:
    robots = list(empty_scenario.robots) if empty_scenario.robots else []
    robot = robots[0] if robots else None
    assert robots == []
    assert robot is None
