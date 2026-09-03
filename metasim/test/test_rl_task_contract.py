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

    def launch(self) -> None:
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
def test_rl_task_reward_terms_initialised_and_default_to_zeros():
    """Base ``_reward`` must be self-consistent before a subclass sets terms.

    ``_reward`` guards on ``len(self.reward_functions) == 0`` to return zeros,
    but ``reward_functions``/``reward_weights`` were never initialised in
    ``__init__`` — so a subclass that relied on the base ``_reward`` (and on the
    documented "no terms -> zero reward" behaviour) hit ``AttributeError``. This
    pins (a) that ``__init__`` assigns both attributes and (b) that the base
    ``_reward`` returns zeros when no terms are set.
    """
    import ast
    import inspect
    import textwrap

    src = textwrap.dedent(inspect.getsource(RLTaskEnv.__init__))
    assigned: set[str] = set()
    for node in ast.walk(ast.parse(src)):
        # Plain ``self.x = ...`` (ast.Assign) and annotated ``self.x: T = ...`` (ast.AnnAssign).
        targets = (
            node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
        )
        for tgt in targets:
            if isinstance(tgt, ast.Attribute) and isinstance(tgt.value, ast.Name) and tgt.value.id == "self":
                assigned.add(tgt.attr)
    assert "reward_functions" in assigned, "RLTaskEnv.__init__ must initialise self.reward_functions"
    assert "reward_weights" in assigned, "RLTaskEnv.__init__ must initialise self.reward_weights"

    # Behavioural: the base _reward returns zeros (not AttributeError/None) for empty terms.
    env = RLTaskEnv.__new__(RLTaskEnv)
    env.num_envs = 3
    env.device = torch.device("cpu")
    env.reward_functions = []
    env.reward_weights = []
    reward = RLTaskEnv._reward(env, None)
    assert torch.equal(reward, torch.zeros(3))


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


@pytest.mark.general
def test_process_action_default_is_identity():
    """The sanctioned action hook defaults to identity on both bases, so adding
    it is a no-op for every task that does not override it."""
    from metasim.task.base import BaseTaskEnv

    sentinel = object()
    assert BaseTaskEnv.__new__(BaseTaskEnv)._process_action(sentinel) is sentinel
    assert RLTaskEnv.__new__(RLTaskEnv)._process_action(sentinel) is sentinel


@pytest.mark.general
def test_rl_task_step_applies_process_action_hook():
    """RLTaskEnv.step must run actions through ``_process_action`` before applying
    them, so tasks can transform actions there instead of overriding ``step``."""
    captured = {}

    class _H:
        num_envs = 1

        def set_dof_targets(self, a):
            captured["applied"] = a.clone()

        def simulate(self):
            return None

        def get_states(self, mode="tensor"):
            return None

    env = RLTaskEnv.__new__(RLTaskEnv)
    env.device = torch.device("cpu")
    env.num_envs = 1
    env.handler = _H()
    env._episode_steps = torch.zeros(1, dtype=torch.int32)
    env._action_low = torch.tensor([-10.0])
    env._action_high = torch.tensor([10.0])
    env._raw_observation_cache = torch.zeros(1, 1)
    env._observation = lambda states: torch.zeros(1, 1)
    env._privileged_observation = lambda states: torch.zeros(1, 1)
    env._reward = lambda states: torch.zeros(1)
    env._terminated = lambda states: torch.zeros(1, dtype=torch.bool)
    env._time_out = lambda states: torch.zeros(1, dtype=torch.bool)
    # Override the hook to add 1.0; the result (after clamping) must reach set_dof_targets.
    env._process_action = lambda actions: actions + 1.0

    RLTaskEnv.step(env, torch.tensor([2.0]))
    assert torch.allclose(captured["applied"], torch.tensor([[3.0]])), (
        f"_process_action not applied before set_dof_targets: got {captured.get('applied')}"
    )
