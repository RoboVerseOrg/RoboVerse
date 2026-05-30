"""Regression tests for ``BaseSimHandler.set_states`` key validation.

Background: ``set_states`` used to accept any dict-state key and silently
drop everything it didn't understand. Downstream code that mistakenly
passed ``dof_pos_target`` under a robot dict expected the joints to
follow a target — instead nothing happened and 15 BC experiments
failed silently before the cause was found (see
``bug_set_states_silent_drop_dof_pos_target.md``).

The fix lives on the base handler, so every backend benefits: unknown
keys now log a one-shot warning, and the well-known control-input keys
(``dof_pos_target`` / ``dof_vel_target`` / ``dof_torque``) get a
specific "use set_dof_targets()" hint. Warnings are deduplicated per
``(role, key)`` to avoid spam.
"""

from __future__ import annotations

import pytest
from loguru import logger as _loguru_logger

from metasim.sim.base import BaseSimHandler


@pytest.fixture
def loguru_warnings():
    """Capture loguru WARNING+ records into a list. ``caplog`` only sees
    stdlib logging; the base handler uses loguru, so we attach a sink."""
    records: list[str] = []
    sink_id = _loguru_logger.add(lambda msg: records.append(str(msg)), level="WARNING")
    try:
        yield records
    finally:
        _loguru_logger.remove(sink_id)


class _FakeRobot:
    """Stand-in for RobotCfg with just the fields the warn helpers need."""

    def __init__(self, name: str, joints: list[str] | None = None):
        self.name = name
        self._joints = joints or []


class _StubHandler(BaseSimHandler):
    """Minimal subclass that bypasses ``__init__`` — we only need the
    ``set_states`` / ``set_dof_targets`` boundaries, not a real scenario."""

    def __init__(self, robots: list[_FakeRobot] | None = None):
        self._tensor_state_cache = None
        self._dict_state_cache = None
        self.set_states_calls: list = []
        self.set_dof_targets_calls: list = []
        self.robots = robots or []

    def get_joint_names(self, obj_name: str, sort: bool = True) -> list[str]:
        for robot in self.robots:
            if robot.name == obj_name:
                names = list(robot._joints)
                return sorted(names) if sort else names
        return []

    def _set_states(self, states, env_ids=None):
        self.set_states_calls.append((states, env_ids))

    def _set_dof_targets(self, actions):
        self.set_dof_targets_calls.append(actions)

    def _get_states(self, env_ids=None):
        raise NotImplementedError

    def _simulate(self):
        raise NotImplementedError


def _state(robot_keys: dict | None = None, object_keys: dict | None = None):
    return [
        {
            "objects": {"cube": object_keys or {"pos": [0, 0, 0]}},
            "robots": {"arm": robot_keys or {"pos": [0, 0, 0]}},
        }
    ]


@pytest.mark.general
def test_known_keys_do_not_warn(loguru_warnings):
    handler = _StubHandler()
    states = _state(robot_keys={"pos": [0, 0, 0], "rot": [1, 0, 0, 0], "dof_pos": {"j0": 0.1}})
    handler.set_states(states)
    out = "\n".join(loguru_warnings)
    assert "unknown key" not in out
    assert "control input" not in out
    assert handler.set_states_calls, "_set_states should still be called"


@pytest.mark.general
def test_control_input_key_warns_with_set_dof_targets_hint(loguru_warnings):
    """``dof_pos_target`` is a control input — set_states drops it.
    The warning must steer the caller to ``set_dof_targets`` so the
    silent-no-op bug can't recur."""
    handler = _StubHandler()
    states = _state(robot_keys={"dof_pos_target": {"j0": 0.5}})
    handler.set_states(states)
    out = "\n".join(loguru_warnings)
    assert "dof_pos_target" in out
    assert "set_dof_targets" in out
    assert "control input" in out
    # The dispatch to _set_states must still happen — warning, not exception.
    assert handler.set_states_calls


@pytest.mark.general
def test_unknown_key_warns_with_valid_keys_list(loguru_warnings):
    handler = _StubHandler()
    states = _state(robot_keys={"joint_pos": {"j0": 0.5}})  # typo for dof_pos
    handler.set_states(states)
    out = "\n".join(loguru_warnings)
    assert "joint_pos" in out
    assert "unknown key" in out


@pytest.mark.general
def test_warning_is_deduplicated_per_handler_instance(loguru_warnings):
    """Hot-path set_states must not spam the log every step."""
    handler = _StubHandler()
    states = _state(robot_keys={"dof_pos_target": {"j0": 0.5}})
    handler.set_states(states)
    handler.set_states(states)
    handler.set_states(states)
    out = "\n".join(loguru_warnings)
    assert out.count("dof_pos_target") == 1


@pytest.mark.general
def test_distinct_unknown_keys_each_warn_once(loguru_warnings):
    handler = _StubHandler()
    handler.set_states(_state(robot_keys={"dof_pos_target": {"j0": 0.5}}))
    handler.set_states(_state(robot_keys={"dof_vel_target": {"j0": 0.5}}))
    handler.set_states(_state(robot_keys={"dof_torque": {"j0": 0.5}}))
    out = "\n".join(loguru_warnings)
    assert "dof_pos_target" in out
    assert "dof_vel_target" in out
    assert "dof_torque" in out


@pytest.mark.general
def test_read_only_velocity_keys_warn(loguru_warnings):
    """``vel`` / ``ang_vel`` / ``dof_vel`` are populated by get_states
    but no backend writes them in _set_states — must warn so callers
    don't believe they are initialising momentum."""
    handler = _StubHandler()
    handler.set_states(_state(robot_keys={"vel": [1, 0, 0]}))
    handler.set_states(_state(robot_keys={"ang_vel": [0, 1, 0]}))
    handler.set_states(_state(robot_keys={"dof_vel": {"j0": 0.5}}))
    out = "\n".join(loguru_warnings)
    assert "vel" in out
    assert "no backend currently" in out
    # Each key should warn exactly once (dedupe).
    assert out.count("MuJoCo zeros qvel") == 3


@pytest.mark.general
def test_read_only_velocity_warning_dedupes(loguru_warnings):
    """Hot-path replay (set_states(get_states())) must not spam."""
    handler = _StubHandler()
    state = _state(robot_keys={"vel": [1, 0, 0]})
    for _ in range(5):
        handler.set_states(state)
    out = "\n".join(loguru_warnings)
    assert out.count("vel") >= 1
    assert out.count("MuJoCo zeros qvel") == 1


@pytest.mark.general
def test_object_role_warns_with_object_in_message(loguru_warnings):
    handler = _StubHandler()
    states = _state(object_keys={"dof_pos_target": {"j0": 0.5}})
    handler.set_states(states)
    out = "\n".join(loguru_warnings)
    assert "objects" in out
    assert "dof_pos_target" in out


@pytest.mark.general
def test_tensor_state_is_not_validated():
    """TensorState carries no free-form keys, so we should fast-path
    past the dict validation without raising."""
    handler = _StubHandler()
    # Anything non-list is treated as TensorState by _warn_set_states_keys
    sentinel = object()
    handler.set_states(sentinel)
    assert handler.set_states_calls == [(sentinel, None)]


@pytest.mark.general
def test_set_dof_targets_invalidates_state_cache_unit():
    """Unit-level guard: ``set_dof_targets`` must invalidate both state caches
    so that any backend whose ``get_states`` reads back action-derived fields
    (e.g. MuJoCo ``joint_pos_target = ctrl[reindex]``) cannot return stale data
    between an action and the next ``simulate``."""
    handler = _StubHandler()
    sentinel_tensor = object()
    sentinel_dict = object()
    handler._tensor_state_cache = sentinel_tensor
    handler._dict_state_cache = sentinel_dict

    handler.set_dof_targets([{"franka": {"dof_pos_target": {"j0": 0.1}}}])

    assert handler._tensor_state_cache is None, "tensor cache not invalidated"
    assert handler._dict_state_cache is None, "dict cache not invalidated"


@pytest.mark.general
def test_set_dof_targets_warns_unknown_robot(loguru_warnings):
    handler = _StubHandler(robots=[_FakeRobot("arm_a", ["j0", "j1"])])
    handler.set_dof_targets([{"arm_b": {"dof_pos_target": {"j0": 0.1}}}])
    out = "\n".join(loguru_warnings)
    assert "unknown robot" in out
    assert "arm_b" in out
    assert "arm_a" in out  # known robot listed for hint


@pytest.mark.general
def test_set_dof_targets_warns_unknown_joint(loguru_warnings):
    handler = _StubHandler(robots=[_FakeRobot("arm", ["j0", "j1"])])
    handler.set_dof_targets([{"arm": {"dof_pos_target": {"j99": 0.5}}}])
    out = "\n".join(loguru_warnings)
    assert "unknown joint" in out
    assert "j99" in out


@pytest.mark.general
def test_set_dof_targets_warns_unknown_top_level_key(loguru_warnings):
    handler = _StubHandler(robots=[_FakeRobot("arm", ["j0"])])
    handler.set_dof_targets([{"arm": {"dof_pos_targt": {"j0": 0.5}}}])  # typo
    out = "\n".join(loguru_warnings)
    assert "unknown key" in out
    assert "dof_pos_targt" in out


@pytest.mark.general
def test_set_dof_targets_quiet_on_valid_dict(loguru_warnings):
    handler = _StubHandler(robots=[_FakeRobot("arm", ["j0", "j1"])])
    handler.set_dof_targets([{"arm": {"dof_pos_target": {"j0": 0.5, "j1": 0.1}}}])
    out = "\n".join(loguru_warnings)
    assert "unknown" not in out


@pytest.mark.general
def test_set_dof_targets_warning_dedupes(loguru_warnings):
    handler = _StubHandler(robots=[_FakeRobot("arm", ["j0"])])
    action = [{"arm": {"dof_pos_target": {"j99": 0.5}}}]
    for _ in range(5):
        handler.set_dof_targets(action)
    out = "\n".join(loguru_warnings)
    assert out.count("unknown joint") == 1


@pytest.mark.general
def test_actions_cache_holds_last_set_dof_targets_input():
    """The ``actions_cache`` property is a public contract — every concrete
    handler has its own implementation, and tests assert
    ``handler.actions_cache is not None`` after ``set_dof_targets``.
    The base now owns the contract so it holds for ParallelHandler /
    HybridSimHandler too, where it previously AttributeError'd."""
    handler = _StubHandler(robots=[_FakeRobot("arm", ["j0", "j1"])])
    assert handler.actions_cache is None  # before any action

    action = [{"arm": {"dof_pos_target": {"j0": 0.5, "j1": 0.1}}}]
    handler.set_dof_targets(action)
    assert handler.actions_cache is action


@pytest.mark.general
def test_set_dof_targets_tensor_input_is_not_validated():
    """Tensor actions are indexed, not name-based — they go through the
    fast path. Validation must skip them."""
    import torch as _torch

    handler = _StubHandler(robots=[_FakeRobot("arm", ["j0", "j1"])])
    handler.set_dof_targets(_torch.zeros(1, 2))
    # Just check no crash; lack of warnings is the contract.
    assert handler.set_dof_targets_calls


@pytest.mark.general
def test_set_seed_is_deterministic_on_numpy_and_torch():
    """``BaseSimHandler.set_seed`` is the contract behind ``env.reset(seed=N)``.
    After seeding, ``np.random`` and ``torch.rand`` must produce identical
    sequences across two fresh handlers — otherwise the gym
    reproducibility promise is a lie."""
    import numpy as np
    import torch

    handler_a = _StubHandler()
    handler_b = _StubHandler()

    handler_a.set_seed(42)
    seq_a_np = np.random.rand(8).tolist()
    seq_a_t = torch.rand(8).tolist()

    handler_b.set_seed(42)
    seq_b_np = np.random.rand(8).tolist()
    seq_b_t = torch.rand(8).tolist()

    assert seq_a_np == seq_b_np, "numpy RNG diverged after equal seeds"
    assert seq_a_t == seq_b_t, "torch RNG diverged after equal seeds"


@pytest.mark.general
def test_partial_pose_warns_pos_without_rot(loguru_warnings):
    """Cross-backend divergence: MuJoCo silently fills missing rot with
    identity, Sapien3 raises KeyError. Warn at the boundary so the
    caller knows the input is ambiguous."""
    handler = _StubHandler()
    handler.set_states([{"objects": {}, "robots": {"arm": {"pos": [0, 0, 0]}}}])
    out = "\n".join(loguru_warnings)
    assert "pos" in out and "rot" in out
    assert "cross-backend" in out or "diverge" in out


@pytest.mark.general
def test_partial_pose_warns_rot_without_pos(loguru_warnings):
    handler = _StubHandler()
    handler.set_states([{"objects": {}, "robots": {"arm": {"rot": [1, 0, 0, 0]}}}])
    out = "\n".join(loguru_warnings)
    assert "rot" in out and "pos" in out


@pytest.mark.general
def test_partial_pose_quiet_when_both_present(loguru_warnings):
    handler = _StubHandler()
    handler.set_states([{"objects": {}, "robots": {"arm": {"pos": [0, 0, 0], "rot": [1, 0, 0, 0]}}}])
    out = "\n".join(loguru_warnings)
    assert "cross-backend" not in out and "diverge" not in out


@pytest.mark.general
def test_partial_pose_warning_dedupes_per_entity(loguru_warnings):
    """Hot-path reset() may be called many times — warning must fire
    at most once per (role, entity, missing)."""
    handler = _StubHandler()
    state = [{"objects": {}, "robots": {"arm": {"pos": [0, 0, 0]}}}]
    handler.set_states(state)
    handler.set_states(state)
    handler.set_states(state)
    out = "\n".join(loguru_warnings)
    assert out.count("cross-backend behaviour diverges") <= 1


@pytest.mark.general
def test_set_seed_differs_with_different_seeds():
    """Sanity: distinct seeds must produce distinct sequences. Otherwise
    set_seed is a no-op and the determinism test above is vacuous."""
    import numpy as np

    handler = _StubHandler()

    handler.set_seed(1)
    seq_1 = np.random.rand(4).tolist()
    handler.set_seed(2)
    seq_2 = np.random.rand(4).tolist()

    assert seq_1 != seq_2, "set_seed appears to be a no-op"
