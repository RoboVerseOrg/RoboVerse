"""Contracts that used to hold only by accident (architecture review, 2026-09-03).

* Content packs win over MetaSim's bundled example pack when both define a config name, and a
  shadowed name is reported.
* ``gym.make``-style construction never mutates the task class's shared ``scenario``.
* ``get_states(env_ids=...)`` returns exactly those envs even when the backend ignores ``env_ids``.
* The task-index cache lives in a per-user directory, not the shared temp dir.
"""

from __future__ import annotations

import os
import textwrap

import pytest
import torch

from metasim.task import _static_index
from metasim.utils import package_discovery, setup_util


@pytest.mark.general
def test_defaults_are_searched_last(monkeypatch):
    monkeypatch.setattr(package_discovery, "_entry_point_packages", lambda role: ("content_pack.robots",))
    cands = package_discovery.get_package_candidates("robots", defaults=["metasim.example.example_pack.robots"])
    assert cands.index("content_pack.robots") < cands.index("metasim.example.example_pack.robots")


@pytest.mark.general
def test_shadowed_config_name_is_reported_and_first_wins(tmp_path, monkeypatch):
    for pkg in ("packa", "packb"):
        d = tmp_path / pkg
        d.mkdir()
        (d / "__init__.py").write_text(
            textwrap.dedent(f'''
            class DemoCfg:
                origin = "{pkg}"
            '''),
            encoding="utf-8",
        )
    monkeypatch.syspath_prepend(str(tmp_path))
    setup_util._SHADOW_WARNED.clear()
    from loguru import logger

    messages: list[str] = []
    sink_id = logger.add(lambda m: messages.append(str(m)), level="WARNING")
    try:
        cfg = setup_util._lookup_cfg("DemoCfg", ["packa", "packb"], "Robot")
    finally:
        logger.remove(sink_id)
    assert cfg.origin == "packa"
    assert any("defined in several packages" in m and "packa" in m for m in messages)


@pytest.mark.general
def test_scenario_replace_leaves_original_untouched():
    from metasim.scenario.scenario import ScenarioCfg

    base = ScenarioCfg(num_envs=1)
    other = base.replace(num_envs=8)
    assert base.num_envs == 1 and other.num_envs == 8 and other is not base


@pytest.mark.general
def test_gym_wrapper_does_not_mutate_class_scenario(monkeypatch):
    from metasim.scenario.scenario import ScenarioCfg
    from metasim.task import gym_registration

    class _Env:
        scenario = ScenarioCfg(num_envs=4)

        def __init__(self, scenario, device=None):
            self.scenario = scenario
            self.device = "cpu"
            self.action_space = self.observation_space = None
            self.num_envs = scenario.num_envs

    monkeypatch.setattr(gym_registration, "get_task_class", lambda name: _Env)
    wrapper = gym_registration.GymEnvWrapper.__new__(gym_registration.GymEnvWrapper)
    gym_registration.GymEnvWrapper.__init__(wrapper, "demo")
    assert wrapper.scenario.num_envs == 1
    assert _Env.scenario.num_envs == 4, "gym.make rewrote the task's class-level scenario"


@pytest.mark.general
def test_get_states_slices_when_backend_ignores_env_ids():
    from metasim.sim.base import BaseSimHandler
    from metasim.types import ObjectState, TensorState

    full = TensorState(
        objects={"cube": ObjectState(root_state=torch.arange(4 * 13, dtype=torch.float32).view(4, 13))},
        robots={},
        cameras={},
    )

    class _Handler:
        num_envs = 4
        _get_states = staticmethod(lambda env_ids=None: full)  # ignores env_ids, like six real backends

    sub = BaseSimHandler._enforce_env_subset(_Handler(), full, [1, 3])
    assert sub.objects["cube"].root_state.shape[0] == 2
    assert torch.equal(sub.objects["cube"].root_state, full.objects["cube"].root_state[[1, 3]])

    # A batch that is neither the requested subset nor the full env set is a backend bug, not a slice.
    class _Broken(_Handler):
        num_envs = 5

    with pytest.raises(RuntimeError):
        BaseSimHandler._enforce_env_subset(_Broken(), full, [0, 1])


@pytest.mark.general
def test_index_cache_defaults_to_user_cache_dir(monkeypatch, tmp_path):
    monkeypatch.delenv("METASIM_CACHE_DIR", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert _static_index.default_cache_path() == os.path.join(str(tmp_path), "metasim", "task_index.json")
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    assert _static_index.default_cache_path().startswith(os.path.join(os.path.expanduser("~"), ".cache", "metasim"))


@pytest.mark.general
def test_joint_pos_target_is_derived_from_the_action_cache_for_tensor_and_dict_actions():
    """Backends whose engine does not hold the target (Genesis, Isaac Gym) report it from the last
    ``set_dof_targets`` input: the handler-order slice of a tensor, or the named entries of a dict."""
    from types import SimpleNamespace

    import torch

    from metasim.sim.base import BaseSimHandler

    class _H(BaseSimHandler):
        def _set_states(self, states, env_ids=None):
            pass

        def _set_dof_targets(self, actions):
            pass

        def _get_states(self, env_ids=None):
            return None

        def _simulate(self):
            pass

        def get_joint_names(self, name, sort=True):
            return {"arm": ["a1", "a2"], "hand": ["h1"]}[name]

        _reports_target_from_action_cache = True  # this stub stands in for Genesis / Isaac Gym
        num_envs = 2  # shadow the base properties: no scenario behind this stub
        device = torch.device("cpu")

    h = _H.__new__(_H)
    h.robots = [SimpleNamespace(name="arm"), SimpleNamespace(name="hand")]
    h._actions_cache = None
    assert h._joint_pos_target_from_action_cache("arm") is None, "no action yet"
    h._actions_cache = []  # Genesis / Isaac Gym start with an empty list
    assert h._joint_pos_target_from_action_cache("arm") is None, "an empty list is no action"

    def applied(actions):  # what set_dof_targets records
        h._actions_cache, h._action_stale_envs, h._action_tensor_for = actions, set(), None

    applied(torch.tensor([[0.1, 0.2, 0.9], [0.3, 0.4, 0.8]]))
    assert torch.allclose(h._joint_pos_target_from_action_cache("arm"), torch.tensor([[0.1, 0.2], [0.3, 0.4]]))
    assert torch.allclose(h._joint_pos_target_from_action_cache("hand"), torch.tensor([[0.9], [0.8]]))
    applied([
        {"arm": {"dof_pos_target": {"a2": 2.0, "a1": 1.0}}},
        {"arm": {"dof_pos_target": {"a1": 3.0, "a2": 4.0}}},
    ])
    assert torch.allclose(h._joint_pos_target_from_action_cache("arm"), torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    assert h._joint_pos_target_from_action_cache("hand") is None, "the dict named no target for it"
    buffer = torch.tensor([[0.5, 0.6, 0.7], [0.5, 0.6, 0.7]])
    applied(buffer)
    assert torch.allclose(h._joint_pos_target_from_action_cache("arm"), torch.tensor([[0.5, 0.6], [0.5, 0.6]]))
    buffer.fill_(9.0)  # the caller reuses its action buffer after the first read
    assert torch.allclose(h._joint_pos_target_from_action_cache("arm"), torch.tensor([[0.5, 0.6], [0.5, 0.6]])), (
        "a copy"
    )
    applied(buffer)  # ...and submits the mutated buffer again: it is re-read
    assert torch.allclose(h._joint_pos_target_from_action_cache("arm"), torch.tensor([[9.0, 9.0], [9.0, 9.0]]))
    h.object_dict = {
        "arm": SimpleNamespace(control_type={"a1": "position", "a2": "position"}),
        "hand": SimpleNamespace(control_type={"h1": "effort"}),
    }
    assert h._joint_pos_target_from_action_cache("hand") is None, (
        "an effort-driven robot: a torque is not a position target"
    )
    assert h._joint_pos_target_from_action_cache("arm") is not None

    # after set_states the reset envs report the joint position just written (what MuJoCo / Isaac Sim report)
    from metasim.types import RobotState, TensorState

    def _written(arm_pos, hand_pos):
        root = torch.zeros(arm_pos.shape[0], 13)
        return TensorState(
            objects={},
            robots={
                "arm": RobotState(
                    root_state=root,
                    body_names=None,
                    body_state=None,
                    joint_pos=arm_pos,
                    joint_vel=torch.zeros_like(arm_pos),
                    joint_pos_target=None,
                    joint_vel_target=None,
                    joint_effort_target=None,
                ),
                "hand": RobotState(
                    root_state=root,
                    body_names=None,
                    body_state=None,
                    joint_pos=hand_pos,
                    joint_vel=torch.zeros_like(hand_pos),
                    joint_pos_target=None,
                    joint_vel_target=None,
                    joint_effort_target=None,
                ),
            },
            cameras={},
            extras={},
        )

    h._refresh_action_targets_after_reset(
        _written(torch.tensor([[0.21, 0.22]]), torch.tensor([[0.6]])), torch.tensor([1])
    )
    target = h._joint_pos_target_from_action_cache("arm")
    assert torch.allclose(target, torch.tensor([[9.0, 9.0], [0.21, 0.22]])), "a subset state for env 1, a tensor of ids"
    target[0, 0] = -1.0  # the caller's copy
    assert torch.allclose(h._joint_pos_target_from_action_cache("arm")[0], torch.tensor([9.0, 9.0]))
    full = _written(torch.tensor([[0.11, 0.12], [0.21, 0.22]]), torch.tensor([[0.5], [0.6]]))
    h._refresh_action_targets_after_reset(full, None)  # a full-batch state, a full reset
    assert torch.allclose(h._joint_pos_target_from_action_cache("arm"), torch.tensor([[0.11, 0.12], [0.21, 0.22]]))
    applied(torch.tensor([[0.7, 0.8, 0.9], [0.7, 0.8, 0.9]]))  # set_dof_targets, then a reset with no read between
    h._refresh_action_targets_after_reset(_written(torch.tensor([[0.31, 0.32]]), torch.tensor([[0.4]])), [0])
    assert torch.allclose(h._joint_pos_target_from_action_cache("arm"), torch.tensor([[0.31, 0.32], [0.7, 0.8]]))
    applied([
        {"arm": {"dof_effort_target": {"a1": 1.0, "a2": 1.0}}},
        {"arm": {"dof_effort_target": {"a1": 1.0, "a2": 1.0}}},
    ])
    assert h._joint_pos_target_from_action_cache("arm") is None, "an effort-only dict action: nothing is converted"
    applied(torch.tensor([[0.7, 0.8, 0.9], [0.7, 0.8, 0.9]]))
    partial = [{"robots": {"arm": {"pos": [0.0, 0.0, 0.0], "rot": [1.0, 0.0, 0.0, 0.0], "dof_pos": {"a1": 0.5}}}}]
    h._refresh_action_targets_after_reset(partial, [1])  # a dict reset naming one joint
    assert torch.allclose(h._joint_pos_target_from_action_cache("arm"), torch.tensor([[0.7, 0.8], [0.5, 0.8]])), (
        "the joint the dict left out keeps its target"
    )
    applied([{"arm": {"dof_pos_target": {"a1": 1.0, "a2": 2.0}, "dof_effort_target": {"a1": 0.1, "a2": 0.1}}}] * 2)
    assert torch.allclose(h._joint_pos_target_from_action_cache("arm"), torch.tensor([[1.0, 2.0], [1.0, 2.0]]))
    assert not getattr(h, "_action_input_warned_keys", set()), (
        "reading back an action the backend applied in full does not warn that its effort entry was dropped"
    )
    applied(torch.zeros(3, 3))  # neither one row nor one per env
    with pytest.raises(ValueError, match="3 action rows for 2 envs"):
        h._joint_pos_target_from_action_cache("arm")
