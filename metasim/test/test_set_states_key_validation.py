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


class _StubHandler(BaseSimHandler):
    """Minimal subclass that bypasses ``__init__`` — we only need the
    ``set_states`` boundary, not a real scenario."""

    def __init__(self):
        self._tensor_state_cache = None
        self._dict_state_cache = None
        self.set_states_calls: list = []
        self.set_dof_targets_calls: list = []

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
