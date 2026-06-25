"""``BaseSimHandler`` exposes visual-flush coordination (M2).

Before the fix, ``flush_visual_updates`` and ``_defer_all_visual_flushes``
existed only on ``IsaacsimHandler``. Domain-randomization callers
guarded every access with ``hasattr(...)``, and on non-Isaac backends
the DR manager still *assigned* ``handler._defer_all_visual_flushes = True``
during randomization, silently writing an Isaac-private attribute onto
an unrelated handler.

After the fix, ``BaseSimHandler`` declares both:

- ``_defer_all_visual_flushes: bool`` — initialised to ``False`` in
  ``__init__``, so every handler instance has a definite value.
- ``flush_visual_updates(**kwargs)`` — a no-op by default; backends that
  drive an independent renderer (Isaac Sim, hybrid Blender, etc.)
  override it.

These tests use minimal concrete subclasses — they do not import any
real physics backend.
"""

from __future__ import annotations

import inspect

import pytest

from metasim.sim.base import BaseSimHandler


class _ConcreteHandler(BaseSimHandler):
    """Smallest valid subclass: only fills in the abstract methods."""

    def _set_states(self, states, env_ids=None):
        return None

    def _set_dof_targets(self, actions):
        return None

    def _get_states(self, env_ids=None):
        return {}

    def _simulate(self):
        return None

    def launch(self) -> None:  # bypass query-binding side effects
        self.optional_queries = self.optional_queries or {}


def _make_handler() -> _ConcreteHandler:
    """Build a handler without invoking any backend __init__.

    We bypass BaseSimHandler.__init__'s scenario.check_assets() requirement
    by using __new__ + manual minimal field setup. The methods under test
    only need the two attributes declared by the M2 fix.
    """
    h = _ConcreteHandler.__new__(_ConcreteHandler)
    h._tensor_state_cache = None
    h._dict_state_cache = None
    h._defer_all_visual_flushes = False
    return h


@pytest.mark.general
def test_flush_visual_updates_exists_on_base():
    """Method is declared on BaseSimHandler itself, not just isaacsim."""
    assert hasattr(BaseSimHandler, "flush_visual_updates")
    assert callable(BaseSimHandler.flush_visual_updates)


@pytest.mark.general
def test_flush_visual_updates_signature_is_documented():
    """The kwargs DR callers pass must be accepted (no TypeError)."""
    sig = inspect.signature(BaseSimHandler.flush_visual_updates)
    params = sig.parameters
    # Callers pass these by keyword in dr_manager.py and isaacsim_adapter.py.
    assert "wait_for_materials" in params
    assert "settle_passes" in params
    # Both must be keyword-only — they are after ``*`` in the source.
    assert params["wait_for_materials"].kind == inspect.Parameter.KEYWORD_ONLY
    assert params["settle_passes"].kind == inspect.Parameter.KEYWORD_ONLY


@pytest.mark.general
def test_flush_visual_updates_default_is_noop():
    """Default impl returns None and does not raise for any documented kwargs."""
    h = _make_handler()
    assert h.flush_visual_updates() is None
    assert h.flush_visual_updates(wait_for_materials=True) is None
    assert h.flush_visual_updates(settle_passes=5) is None
    assert h.flush_visual_updates(wait_for_materials=True, settle_passes=2) is None


@pytest.mark.general
def test_defer_flag_is_present_on_every_handler():
    """The flag must be a real attribute after construction.

    This protects the dr_manager's ``handler._defer_all_visual_flushes = True``
    line — that line previously *implicitly created* the attribute on
    backends that had never declared it, leaving them with stale state
    after randomization. Now the base reserves it explicitly.
    """
    h = _make_handler()
    assert hasattr(h, "_defer_all_visual_flushes")
    assert h._defer_all_visual_flushes is False


@pytest.mark.general
def test_defer_flag_initialised_by_real_init():
    """``__init__`` itself must set the flag — regression for the dead-code bug.

    The init line previously lived *after* ``return`` inside the
    ``actions_cache`` property, so it never ran and ``BaseSimHandler.__init__``
    left the attribute undefined. ``_make_handler`` above masks that because it
    sets the attribute by hand via ``__new__``; this test goes through the real
    ``__init__`` with a minimal fake scenario so the dead code cannot hide.
    """
    from types import SimpleNamespace

    scenario = SimpleNamespace(
        robots=[],
        cameras=[],
        objects=[],
        lights=[],
        num_envs=1,
        decimation=1,
        headless=True,
        check_assets=lambda: None,
    )
    h = _ConcreteHandler(scenario)
    assert h._defer_all_visual_flushes is False


@pytest.mark.general
def test_defer_flag_round_trips():
    """DR manager toggles the flag around batch randomization; verify the
    write+read+reset path holds."""
    h = _make_handler()
    assert h._defer_all_visual_flushes is False
    h._defer_all_visual_flushes = True
    assert h._defer_all_visual_flushes is True
    # flush during defer is still a safe no-op on the base
    h.flush_visual_updates(wait_for_materials=True, settle_passes=2)
    h._defer_all_visual_flushes = False
    assert h._defer_all_visual_flushes is False


@pytest.mark.general
def test_subclass_can_override_flush():
    """An override should be honoured: this is how IsaacsimHandler hooks in."""

    class _OverridingHandler(_ConcreteHandler):
        def __init__(self):
            self._defer_all_visual_flushes = False
            self.passes_run = 0

        def flush_visual_updates(self, *, wait_for_materials: bool = False, settle_passes: int = 2) -> None:
            if self._defer_all_visual_flushes:
                return
            self.passes_run += max(1, settle_passes)

    h = _OverridingHandler()
    h.flush_visual_updates(settle_passes=3)
    assert h.passes_run == 3

    # While deferred, the override skips work.
    h._defer_all_visual_flushes = True
    h.flush_visual_updates(settle_passes=2)
    assert h.passes_run == 3  # unchanged
