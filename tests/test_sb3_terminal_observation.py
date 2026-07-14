"""Regression: the SB3 VecEnv wrapper must report the *pre-reset* terminal observation.

``RLTaskEnv.step`` auto-resets done envs in place before returning
(``obs[done_indices] = obs_after[done_indices]``), so for a done env the returned obs is
already the first observation of the *next* episode. The SB3 wrapper used to copy that slot
into ``info["terminal_observation"]``. SB3 corrects truncated episodes with
``reward += gamma * V(terminal_observation)``, and ``TimeLimit.truncated`` is computed
correctly, so the bad bootstrap fired on *every* timeout — corrupting the final reward of
every episode of any task that ends only by timeout (locomotion, reach, ...).

These tests drive the real ``RLTaskEnv.step`` (through a stub handler, no simulator) and pin
that the wrapper hands SB3 the observation from the terminal state, not the post-reset one.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import numpy as np
import pytest

_REPO = pathlib.Path(__file__).resolve().parents[1]
_SCRIPTS = ["roboverse_learn/rl/sb3/ppo_train.py", "get_started/rl/0_ppo.py"]

_HORIZON = 3  # episode ends by timeout after 3 steps
_NUM_ENVS = 2


def _load_script(monkeypatch, relpath: str):
    """Import one of the PPO training scripts by path (both parse CLI args at import)."""
    pytest.importorskip("stable_baselines3")
    pytest.importorskip("rich")
    pytest.importorskip("tyro")
    monkeypatch.setattr(sys, "argv", [pathlib.Path(relpath).name])
    name = "_ppo_" + relpath.replace("/", "_").removesuffix(".py")
    spec = importlib.util.spec_from_file_location(name, _REPO / relpath)
    module = importlib.util.module_from_spec(spec)
    # ``@dataclass`` resolution needs the module registered while it executes.
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


class _StubHandler:
    """A one-dimensional "simulator": the state counts up one per step, reset zeroes it."""

    def __init__(self, num_envs: int) -> None:
        import torch

        self.num_envs = num_envs
        self.device = torch.device("cpu")
        self.pos = torch.zeros(num_envs, 1)

    def set_dof_targets(self, actions) -> None:
        pass

    def simulate(self) -> None:
        self.pos += 1.0

    def get_states(self, mode: str = "tensor", env_ids=None):
        return self.pos.clone()

    def set_states(self, states=None, env_ids=None) -> None:
        self.pos[list(env_ids)] = 0.0

    def close(self) -> None:
        pass


def _make_task():
    """A real ``RLTaskEnv`` (so the actual auto-reset path runs) over the stub handler.

    Built with ``__new__`` + explicit field assignment — the same trick MetaSim's own
    ``test_rl_task_contract`` uses — because the normal ``__init__`` needs a live simulator.
    """
    import torch

    from metasim.task.rl_task import RLTaskEnv

    task = RLTaskEnv.__new__(RLTaskEnv)
    task.device = torch.device("cpu")
    task.num_envs = _NUM_ENVS
    task.max_episode_steps = _HORIZON
    task.handler = _StubHandler(_NUM_ENVS)
    task._episode_steps = torch.zeros(_NUM_ENVS, dtype=torch.int32)
    task._action_low = torch.tensor([-10.0])
    task._action_high = torch.tensor([10.0])
    task._initial_states = None
    task._obs_buf = None
    task._priv_obs_buf = None
    task.reward_functions = []
    task.reward_weights = []
    task.num_obs = 1
    task.num_actions = 1
    task._observation = lambda states: states.clone()
    task._privileged_observation = lambda states: states.clone()
    task._terminated = lambda states: torch.zeros(_NUM_ENVS, dtype=torch.bool)
    task.reset(env_ids=list(range(_NUM_ENVS)))
    return task


@pytest.mark.general
@pytest.mark.parametrize("script", _SCRIPTS)
def test_terminal_observation_is_the_pre_reset_observation(monkeypatch, script):
    """On a timeout, ``terminal_observation`` must be the obs of the terminal state."""
    module = _load_script(monkeypatch, script)
    vec = module.VecEnvWrapper(_make_task())

    vec.reset()
    for _ in range(_HORIZON):
        vec.step_async(np.zeros((_NUM_ENVS, 1), dtype=np.float32))
        obs, _rewards, dones, infos = vec.step_wait()

    assert dones.all(), "the stub task must time out after `_HORIZON` steps"
    for env_id in range(_NUM_ENVS):
        assert infos[env_id]["TimeLimit.truncated"] is True
        # The env has already been auto-reset, so the obs SB3 sees for the next rollout step
        # is the new episode's first obs...
        assert obs[env_id] == pytest.approx([0.0])
        # ...but the terminal observation SB3 bootstraps from must be the pre-reset state.
        assert infos[env_id]["terminal_observation"] == pytest.approx([float(_HORIZON)]), (
            "terminal_observation is the post-reset obs — SB3's truncation bootstrap "
            "V(terminal_observation) is evaluated at the wrong state"
        )


@pytest.mark.general
@pytest.mark.parametrize("script", _SCRIPTS)
def test_running_episode_reports_no_terminal_observation(monkeypatch, script):
    """Mid-episode steps must not carry a terminal observation, and obs must stay live."""
    module = _load_script(monkeypatch, script)
    vec = module.VecEnvWrapper(_make_task())

    vec.reset()
    for step in range(1, _HORIZON):
        vec.step_async(np.zeros((_NUM_ENVS, 1), dtype=np.float32))
        obs, _rewards, dones, infos = vec.step_wait()
        assert not dones.any()
        for env_id in range(_NUM_ENVS):
            assert "terminal_observation" not in infos[env_id]
            assert infos[env_id]["TimeLimit.truncated"] is False
            assert obs[env_id] == pytest.approx([float(step)])


@pytest.mark.general
@pytest.mark.parametrize("script", _SCRIPTS)
def test_missing_pre_reset_observation_fails_loudly(monkeypatch, script):
    """A task the wrapper cannot snapshot must raise, not silently report the wrong obs."""
    import torch

    module = _load_script(monkeypatch, script)

    class _OpaqueTask:
        """A task that resets internally without routing obs through ``_observation``."""

        num_envs = _NUM_ENVS
        device = torch.device("cpu")
        observation_space = None
        action_space = None
        raw_obs: dict | None = None

        def __init__(self):
            from gymnasium import spaces

            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        def _observation(self, states):
            return states

        def reset(self, states=None, env_ids=None, seed=None):
            return torch.zeros(_NUM_ENVS, 1), {}

        def step(self, actions):
            done = torch.ones(_NUM_ENVS, dtype=torch.bool)
            info = {"observations": {"raw": {"obs": self.raw_obs}}} if self.raw_obs is not None else {}
            return (
                torch.zeros(_NUM_ENVS, 1),
                torch.zeros(_NUM_ENVS),
                torch.zeros(_NUM_ENVS, dtype=torch.bool),
                done,
                info,
            )

    task = _OpaqueTask()
    vec = module.VecEnvWrapper(task)
    vec.step_async(np.zeros((_NUM_ENVS, 1), dtype=np.float32))
    with pytest.raises(RuntimeError, match="pre-reset observation"):
        vec.step_wait()

    # A task that overrides ``step`` but publishes the pre-reset obs itself (the
    # ``LeggedRobotTask`` contract) is honoured.
    task.raw_obs = torch.full((_NUM_ENVS, 1), 7.0)
    vec.step_async(np.zeros((_NUM_ENVS, 1), dtype=np.float32))
    _obs, _rewards, _dones, infos = vec.step_wait()
    for env_id in range(_NUM_ENVS):
        assert infos[env_id]["terminal_observation"] == pytest.approx([7.0])
