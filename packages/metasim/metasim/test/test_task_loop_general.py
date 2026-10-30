"""The task loop is one sequence of hooks on both bases (``BaseTaskEnv`` and ``RLTaskEnv``).

Backend-free: a recording handler stands in for the simulator, so these pin the order in which
``step`` / ``reset`` call the hooks a task overrides instead of copying the loop.
"""

from __future__ import annotations

import pytest
import torch

from metasim.task.base import BaseTaskEnv
from metasim.task.rl_task import RLTaskEnv


class _Handler:
    num_envs = 2
    device = torch.device("cpu")
    set_states_refreshes = True

    def __init__(self, calls):
        self.calls = calls
        self.scenario = None

    def set_dof_targets(self, actions):
        self.calls.append(("set_dof_targets", actions.clone() if torch.is_tensor(actions) else actions))

    def simulate(self):
        self.calls.append("simulate")

    def get_states(self, env_ids=None, mode="tensor"):
        self.calls.append("get_states")
        return {"step": len(self.calls)}

    def set_states(self, states, env_ids=None):
        self.calls.append(("set_states", list(env_ids)))


def _make(base, calls, *, done=(False, False)):
    class _Task(base):
        def _process_action(self, actions):
            calls.append("process")
            return actions * 2

        def _on_step(self, env_states):
            calls.append("on_step")

        def _on_reset(self, env_ids):
            calls.append(("on_reset", list(env_ids)))

        def _observation(self, env_states):
            calls.append("observation")
            return torch.zeros(2, 1)

        def _privileged_observation(self, env_states):
            return None

        def _reward(self, env_states):
            calls.append("reward")
            return torch.zeros(2)

        def _terminated(self, env_states):
            return torch.tensor(done)

    env = _Task.__new__(_Task)
    env.device = torch.device("cpu")
    env.num_envs = 2
    env.handler = _Handler(calls)
    env._episode_steps = torch.zeros(2, dtype=torch.int32)
    env._initial_states = "initial"
    env.max_episode_steps = 100
    env._prepare_callbacks()
    if base is RLTaskEnv:
        env._action_low, env._action_high = torch.tensor([-1.0]), torch.tensor([1.0])
        env._obs_buf = env._priv_obs_buf = None
    return env


@pytest.mark.general
@pytest.mark.parametrize("base", [BaseTaskEnv, RLTaskEnv])
def test_step_runs_the_same_hook_sequence_on_both_bases(base):
    calls = []
    env = _make(base, calls)
    env.pre_physics_step_callback.append(lambda a: calls.append("pre_cb"))
    env.post_physics_step_callback.append(lambda s: calls.append("post_cb"))
    env.step(torch.full((2, 1), 0.25))
    names = [c if isinstance(c, str) else c[0] for c in calls]
    physics = ["process", "pre_cb", "set_dof_targets", "simulate", "get_states", "post_cb", "on_step"]
    assert names[: len(physics)] == physics
    assert "reward" in names[len(physics) :]
    assert torch.equal(calls[2][1], torch.full((2, 1), 0.5)), "the processed action reaches the handler"
    assert env._episode_steps.tolist() == [1, 1]


@pytest.mark.general
def test_rl_step_runs_dict_callbacks_with_the_env():
    """The humanoid packs' ``{name: (fn, params)}`` callbacks run under RLTaskEnv's step too."""
    calls = []
    env = _make(RLTaskEnv, calls)
    env.pre_physics_step_callback = {"a": (lambda e, gain: calls.append(("pre", e is env, gain)), {"gain": 3})}
    env.post_physics_step_callback = {"b": (lambda e, s, gain: calls.append(("post", e is env, gain)), {"gain": 4})}
    env.step(torch.zeros(2, 1))
    assert ("pre", True, 3) in calls and ("post", True, 4) in calls
    env.post_physics_step_callback = "nope"
    with pytest.raises(TypeError, match="post_physics_step_callback must be"):
        env.step(torch.zeros(2, 1))


@pytest.mark.general
def test_rl_step_clips_in_apply_action_and_a_task_can_replace_it():
    calls = []
    env = _make(RLTaskEnv, calls)
    env.step(torch.full((2, 1), 3.0))  # processed to 6.0, clipped to the joint box
    sent = next(c[1] for c in calls if isinstance(c, tuple) and c[0] == "set_dof_targets")
    assert torch.equal(sent, torch.ones(2, 1))

    written = []
    type(env)._apply_action = lambda self, actions: written.append(actions)
    calls.clear()
    env.step(torch.full((2, 1), 3.0))
    assert written and torch.equal(written[0], torch.full((2, 1), 6.0))
    assert not any(isinstance(c, tuple) and c[0] == "set_dof_targets" for c in calls)


@pytest.mark.general
@pytest.mark.parametrize("base", [BaseTaskEnv, RLTaskEnv])
def test_reset_calls_on_reset_after_the_states_are_written(base):
    calls = []
    env = _make(base, calls)
    env._episode_steps[:] = 7
    env.reset(env_ids=[1])
    write = calls.index(("set_states", [1]))
    assert calls.index(("on_reset", [1])) == write + 1
    assert env._episode_steps.tolist() == [7, 0]


@pytest.mark.general
def test_auto_reset_calls_on_reset_for_the_done_envs_after_on_step():
    calls = []
    env = _make(RLTaskEnv, calls, done=(False, True))
    env.step(torch.zeros(2, 1))
    assert calls.index("on_step") < calls.index(("on_reset", [1]))
    assert env._episode_steps.tolist() == [1, 0]


@pytest.mark.general
def test_checkers_live_in_the_kernel_and_the_old_path_is_the_same_module():
    """Success predicates are kernel vocabulary; the example-pack path stays an alias, not a copy."""
    from metasim.example.example_pack.tasks import checkers as old_pkg
    from metasim.example.example_pack.tasks.checkers import checkers as old_checkers
    from metasim.example.example_pack.tasks.checkers.detectors import RelativeBboxDetector as OldBbox
    from metasim.task import checkers
    from metasim.task.checkers.detectors import RelativeBboxDetector

    assert OldBbox is RelativeBboxDetector
    assert old_checkers is checkers.checkers
    assert old_pkg.DetectedChecker is checkers.DetectedChecker
    for name in checkers.__all__:  # every advertised name exists (``import *`` must not fail)
        assert hasattr(checkers, name), name
