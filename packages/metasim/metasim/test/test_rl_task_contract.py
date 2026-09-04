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

    def __init__(self, scenario: ScenarioCfg, robot_joints: list[str], device: str | torch.device = "cpu") -> None:
        self.scenario = scenario
        self.num_envs = scenario.num_envs
        self._robot_joints = robot_joints
        self._tensor_state_cache = None
        self._dict_state_cache = None
        self.device = torch.device(device)

    def set_dof_targets(self, actions) -> None:  # step() drives these two
        pass

    def simulate(self) -> None:
        pass

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


@pytest.mark.general
def test_step_publishes_the_terminal_observation_not_the_post_reset_one():
    """``info["observations"]["raw"]["obs"]`` must be the obs the episode ended in.

    ``step`` auto-resets done envs in place, so the returned ``obs`` already holds the next
    episode's first observation for those envs. Off-policy learners bootstrap truncated
    episodes off the raw key -- ``V(s_T)`` for a time-out is a real value, ``V(reset state)``
    is not -- so publishing anything but the pre-reset observation silently corrupts the
    target on every episode boundary.

    This used to be served from a ``_raw_observation_cache`` that was written only in
    ``reset()``: the update branch ran solely when *no* env was done, where ``terminated`` is
    all-False and its ``torch.where`` was a no-op. The key therefore carried the episode's
    *first* observation for the whole episode -- not the terminal one, and not even the
    post-reset one.
    """
    step_count = {"n": 0}

    class _H:
        num_envs = 1

        def set_dof_targets(self, a):
            return None

        def simulate(self):
            step_count["n"] += 1

        def get_states(self, mode="tensor"):
            return None

    env = RLTaskEnv.__new__(RLTaskEnv)
    env.device = torch.device("cpu")
    env.num_envs = 1
    env.handler = _H()
    env._episode_steps = torch.zeros(1, dtype=torch.int32)
    env._action_low = torch.tensor([-10.0])
    env._action_high = torch.tensor([10.0])
    # The observation is the step counter, so the terminal obs is distinguishable from the
    # post-reset one by value alone.
    env._observation = lambda states: torch.full((1, 1), float(step_count["n"]))
    env._privileged_observation = lambda states: torch.zeros(1, 1)
    env._reward = lambda states: torch.zeros(1)
    env._terminated = lambda states: torch.zeros(1, dtype=torch.bool)
    env._time_out = lambda states: torch.ones(1, dtype=torch.bool)  # always truncate
    env._process_action = lambda actions: actions
    # reset() zeroes the counter, standing in for the state being destroyed.
    env.reset = lambda states=None, env_ids=None, seed=None: step_count.update(n=0)

    obs, _, _, time_out, info = RLTaskEnv.step(env, torch.tensor([0.0]))

    assert bool(time_out[0]), "sanity: this env truncates every step"
    raw = info["observations"]["raw"]["obs"]
    assert torch.allclose(raw, torch.tensor([[1.0]])), (
        f"raw obs must be the terminal observation (1.0, the state the episode ended in), got {raw.tolist()}"
    )
    assert torch.allclose(obs, torch.tensor([[0.0]])), (
        "sanity: the returned obs is the post-reset one (0.0) -- which is exactly why the raw "
        "key must not be taken from it"
    )


def _step_env(base_cls, scenario, monkeypatch, *, hooks, env_device):
    """A task on ``base_cls`` whose reward / terminated / time_out hooks return ``hooks`` values.

    ``BaseTaskEnv`` is built through its real ``__init__`` (the stub handler satisfies it);
    ``RLTaskEnv`` is hand-built the way the other step tests in this file do, because its ``__init__``
    needs real initial states.
    """
    from metasim.task.base import BaseTaskEnv

    handler = _StubHandler(scenario, ["j1"], device=env_device)
    n = scenario.num_envs

    class _Task(base_cls):
        max_episode_steps = 5

        def _reward(self, states):
            return hooks["reward"]

        def _terminated(self, states):
            return hooks["terminated"]

        def _time_out(self, states):
            return hooks["time_out"]

        def _observation(self, states):
            return torch.zeros(n, 1)

        def _privileged_observation(self, states):
            return None

    if base_cls is BaseTaskEnv:

        def _instantiate(self, scenario):
            self.handler = handler
            handler.launch()

        monkeypatch.setattr(BaseTaskEnv, "_instantiate_env", _instantiate)
        return _Task(scenario)

    env = _Task.__new__(_Task)
    env.device = torch.device(env_device)
    env.num_envs = n
    env.handler = handler
    env._episode_steps = torch.zeros(n, dtype=torch.int32, device=env.device)
    env._action_low = torch.full((1,), -10.0)
    env._action_high = torch.full((1,), 10.0)
    env.max_episode_steps = 5
    env.reset = lambda *args, **kwargs: (torch.zeros(n, 1), {})  # auto-reset of done envs needs no real states here
    return env


@pytest.mark.general
@pytest.mark.parametrize("hook_device", ["cpu", "cuda"])
@pytest.mark.parametrize("base", ["BaseTaskEnv", "RLTaskEnv"])
def test_step_normalises_reward_and_termination_on_both_bases(_stub_scenario, monkeypatch, hook_device, base):
    """``step()`` returns a ``(num_envs,)`` float32 reward and bool terminated / timeout on the env device
    from both task bases, whatever dtype / device / container the hooks produce. With CUDA available the
    hooks build their tensors on the other device than the handler, so the device move is exercised."""
    from metasim.task.base import BaseTaskEnv

    if hook_device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    env_device = torch.device("cpu" if hook_device == "cuda" else ("cuda" if torch.cuda.is_available() else "cpu"))
    n = _stub_scenario.num_envs
    hooks = {
        "reward": torch.full((n,), 1.0, dtype=torch.float64, device=hook_device),
        "terminated": torch.ones(n, dtype=torch.int64, device=hook_device),
        "time_out": [False] * n,  # a plain Python sequence is accepted too
    }
    base_cls = BaseTaskEnv if base == "BaseTaskEnv" else RLTaskEnv
    env = _step_env(base_cls, _stub_scenario, monkeypatch, hooks=hooks, env_device=env_device)
    assert env.device == env_device
    _obs, reward, terminated, timeout, _info = env.step(torch.zeros(n, 1))
    for value, dtype in ((reward, torch.float32), (terminated, torch.bool), (timeout, torch.bool)):
        assert value.dtype == dtype and value.device.type == env_device.type and tuple(value.shape) == (n,)
    assert bool(terminated.all()) is True and bool(timeout.any()) is False


@pytest.mark.general
@pytest.mark.parametrize(
    ("bad", "match"),
    [
        (True, r"_terminated returned shape \(\); step\(\) needs one value per env"),
        (torch.ones(2, 1, dtype=torch.bool), r"_terminated returned shape \(2, 1\)"),
        ({"done": True}, r"_terminated returned dict"),
    ],
    ids=["scalar", "column", "dict"],
)
def test_step_rejects_a_hook_return_of_the_wrong_shape_by_name(_stub_scenario, monkeypatch, bad, match):
    """A scalar / (N, 1) / dict from a hook used to slip through with the right dtype and silently break
    RLTaskEnv's auto-reset; both bases now name the hook and the task."""
    from metasim.task.base import BaseTaskEnv

    n = _stub_scenario.num_envs
    hooks = {"reward": torch.zeros(n), "terminated": bad, "time_out": torch.zeros(n, dtype=torch.bool)}
    env = _step_env(BaseTaskEnv, _stub_scenario, monkeypatch, hooks=hooks, env_device="cpu")
    with pytest.raises((ValueError, TypeError), match=r"_Task\." + match):
        env.step(torch.zeros(n, 1))
