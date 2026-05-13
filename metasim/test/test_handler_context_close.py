"""HandlerContext must call ``handler.close()`` on every exit path (H2).

Previously the context skipped close() for the isaacsim backend, leaking
Replicator workers + hydra delegates across every successful or failed
``with HandlerContext`` block. The handler itself uses
``_owns_simulation_app`` to decide whether to fully tear down or just
clean up local state, so always-close is safe.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import metasim.sim.sim_context as sim_context_module


class _FakeHandler:
    def __init__(self, scenario):
        self.scenario = scenario
        self.launch_called = False
        self.launch_kwargs = None
        self.close_called = False

    def launch(self, simulation_app=None):
        self.launch_called = True
        self.launch_kwargs = {"simulation_app": simulation_app}

    def close(self):
        self.close_called = True


class _FailingLaunchHandler(_FakeHandler):
    def launch(self, simulation_app=None):
        super().launch(simulation_app=simulation_app)
        raise RuntimeError("simulated launch failure")


def _scenario(sim_name: str):
    return SimpleNamespace(simulator=sim_name, num_envs=1)


@pytest.fixture()
def _stub_dispatch(monkeypatch):
    """Replace get_sim_handler_class with a factory returning _FakeHandler."""

    holder = {"instance": None}

    def _factory(_sim_type):
        def _builder(scenario):
            holder["instance"] = _FakeHandler(scenario)
            return holder["instance"]

        return _builder

    monkeypatch.setattr(sim_context_module, "get_sim_handler_class", _factory)
    return holder


@pytest.mark.general
@pytest.mark.parametrize("simulator", ["mujoco", "isaacsim", "pybullet", "blender"])
def test_handler_context_closes_on_clean_exit(_stub_dispatch, simulator):
    """All backends — including isaacsim — must have close() called on
    normal exit."""
    with sim_context_module.HandlerContext(_scenario(simulator)) as h:
        assert h.launch_called
    assert _stub_dispatch["instance"].close_called, (
        f"close() should have been called for {simulator} on clean exit"
    )


@pytest.mark.general
@pytest.mark.parametrize("simulator", ["mujoco", "isaacsim", "pybullet", "blender"])
def test_handler_context_closes_after_launch_failure(monkeypatch, simulator):
    """When launch() raises, close() must still run as cleanup —
    previously isaacsim got skipped here too."""
    holder = {"instance": None}

    def _factory(_sim_type):
        def _builder(scenario):
            holder["instance"] = _FailingLaunchHandler(scenario)
            return holder["instance"]

        return _builder

    monkeypatch.setattr(sim_context_module, "get_sim_handler_class", _factory)

    with pytest.raises(RuntimeError, match="simulated launch failure"):
        with sim_context_module.HandlerContext(_scenario(simulator)):
            pass  # never reached
    assert holder["instance"].close_called, (
        f"close() should have been called for {simulator} after launch failure"
    )


@pytest.mark.general
def test_handler_context_passes_simulation_app_to_isaacsim_only(_stub_dispatch):
    """Sanity: isaacsim still receives the injected simulation_app handle."""
    fake_app = object()
    with sim_context_module.HandlerContext(_scenario("isaacsim"), simulation_app=fake_app):
        pass
    assert _stub_dispatch["instance"].launch_kwargs == {"simulation_app": fake_app}

    with sim_context_module.HandlerContext(_scenario("mujoco")):
        pass
    # Mujoco's launch was called with no kwargs (the launch() proxy in
    # _FakeHandler just records what it sees — None default).
    assert _stub_dispatch["instance"].launch_kwargs == {"simulation_app": None}
