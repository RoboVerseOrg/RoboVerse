"""``_extra_spec`` must be callable safely before the handler exists (C2).

The previous ``_instantiate_env`` did
``handler_class(scenario, self.extra_spec)`` — the ``self.extra_spec``
property invokes the subclass's ``_extra_spec``. If the subclass
checked ``self.handler`` (a natural pattern, e.g. to register
handler-specific queries), it hit
``AttributeError: 'XEnv' object has no attribute 'handler'`` because
``self.handler`` was not yet assigned.

After the fix:
- ``self.handler`` is pre-set to ``None`` before ``self.extra_spec``
  is read, so subclasses can safely do ``if self.handler is None: ...``
- ``self.handler`` is then reassigned to the constructed handler
- ``handler.launch()`` calls ``query_type.bind_handler(self)`` to bind
  the queries to the now-live handler

This test directly drives ``_instantiate_env`` with a stub handler
class to verify the timing.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from metasim.task.base import BaseTaskEnv


class _StubHandler:
    def __init__(self, scenario, queries):
        self.scenario = scenario
        self.queries = queries
        self.launched = False

    def launch(self):
        self.launched = True

    def close(self):
        return None


class _SafeTask(BaseTaskEnv):
    """Subclass that DOES inspect ``self.handler`` in ``_extra_spec`` —
    pre-fix this would raise AttributeError."""

    _extra_spec_calls = 0
    _handler_seen_in_extra_spec = "<unset>"

    def _extra_spec(self):
        type(self)._extra_spec_calls += 1
        # The whole point: this must NOT AttributeError.
        type(self)._handler_seen_in_extra_spec = self.handler
        return {"saw_handler": self.handler}


@pytest.fixture(autouse=True)
def _reset_class_state():
    _SafeTask._extra_spec_calls = 0
    _SafeTask._handler_seen_in_extra_spec = "<unset>"


@pytest.mark.general
def test_extra_spec_sees_none_handler_pre_launch(monkeypatch):
    """When ``_extra_spec`` runs, ``self.handler`` is None — not undefined."""
    monkeypatch.setattr(
        "metasim.task.base.get_sim_handler_class",
        lambda sim_type: _StubHandler,
    )

    env = _SafeTask.__new__(_SafeTask)
    env.scenario = SimpleNamespace(simulator="mujoco", robots=[], num_envs=1)
    env._instantiate_env(env.scenario)

    assert _SafeTask._extra_spec_calls == 1
    assert _SafeTask._handler_seen_in_extra_spec is None, (
        "_extra_spec saw something other than None for self.handler at pre-launch time"
    )
    # After _instantiate_env returns, self.handler is the live handler:
    assert isinstance(env.handler, _StubHandler)
    assert env.handler.launched


@pytest.mark.general
def test_extra_spec_queries_passed_to_handler(monkeypatch):
    """The dict returned by _extra_spec is passed as ``queries`` to
    ``handler_class(scenario, queries)``."""
    captured = {}

    class _Capture(_StubHandler):
        def __init__(self, scenario, queries):
            captured["queries"] = queries
            super().__init__(scenario, queries)

    monkeypatch.setattr(
        "metasim.task.base.get_sim_handler_class",
        lambda sim_type: _Capture,
    )

    env = _SafeTask.__new__(_SafeTask)
    env.scenario = SimpleNamespace(simulator="mujoco", robots=[], num_envs=1)
    env._instantiate_env(env.scenario)

    assert captured["queries"] == {"saw_handler": None}


@pytest.mark.general
def test_extra_spec_called_only_once(monkeypatch):
    """Sanity: don't accidentally double-invoke _extra_spec by reading
    the property twice."""
    monkeypatch.setattr(
        "metasim.task.base.get_sim_handler_class",
        lambda sim_type: _StubHandler,
    )
    env = _SafeTask.__new__(_SafeTask)
    env.scenario = SimpleNamespace(simulator="mujoco", robots=[], num_envs=1)
    env._instantiate_env(env.scenario)
    assert _SafeTask._extra_spec_calls == 1
