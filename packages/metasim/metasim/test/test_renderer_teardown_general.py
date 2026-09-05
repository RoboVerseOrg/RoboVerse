"""A handler's offscreen renderer is closed by ``close()``, by garbage collection and at exit, never twice."""

from __future__ import annotations

import atexit
import gc

import pytest

from metasim.sim.mujoco import renderer_teardown

pytestmark = pytest.mark.general


class _Renderer:
    def __init__(self):
        self.closed = 0

    def close(self):
        self.closed += 1


class _Owner:
    pass


def test_close_runs_once_and_gc_does_not_close_again(monkeypatch):
    hooks = []
    monkeypatch.setattr(atexit, "register", lambda fn, *a: hooks.append((fn, a)))
    owner, renderer = _Owner(), _Renderer()
    fin = renderer_teardown.attach_renderer_teardown(owner, renderer)
    assert len(hooks) == 1, "one exit hook per owner, registered after the first renderer (before eglTerminate)"
    fin()
    assert renderer.closed == 1
    replacement = _Renderer()
    renderer_teardown.attach_renderer_teardown(owner, replacement)
    dropped = _Renderer()
    renderer_teardown.attach_renderer_teardown(owner, dropped)  # e.g. nulled by a caller without close()
    assert len(hooks) == 1, "replacing the renderer registers no second hook"
    hooks[0][0](*hooks[0][1])
    assert (renderer.closed, replacement.closed, dropped.closed) == (1, 1, 1), "the exit hook closes every live one"
    del owner
    gc.collect()
    assert (renderer.closed, replacement.closed, dropped.closed) == (1, 1, 1), "already closed: nothing runs twice"


def test_a_dropped_owner_closes_its_renderer(monkeypatch):
    monkeypatch.setattr(atexit, "register", lambda fn, *a: None)
    renderer = _Renderer()
    owner = _Owner()
    renderer_teardown.attach_renderer_teardown(owner, renderer)
    del owner
    gc.collect()
    assert renderer.closed == 1


def test_a_replaced_renderer_is_closed_by_the_handler_helper(monkeypatch):
    """``MujocoHandler._new_renderer`` closes the previous renderer and re-attaches the hooks."""
    mujoco = pytest.importorskip("mujoco")
    from metasim.sim.mujoco.mujoco import MujocoHandler

    made = []
    monkeypatch.setattr(mujoco, "Renderer", lambda model, width, height: made.append(_Renderer()) or made[-1])
    monkeypatch.setattr(atexit, "register", lambda fn, *a: None)
    h = MujocoHandler.__new__(MujocoHandler)
    h.renderer = _Renderer()  # assigned directly, without hooks: replacing it must not fail
    h._renderer_finalizer = None
    h._mj_model = object()
    h.viewer = None
    h._new_renderer(640, 480)
    h._new_renderer(256, 256)
    assert [r.closed for r in made] == [1, 0], "the first renderer was closed when replaced"
    h.close()
    assert [r.closed for r in made] == [1, 1] and h.renderer is None
    h.close()
    assert [r.closed for r in made] == [1, 1], "close() is idempotent"


def test_a_closed_renderer_at_exit_is_quiet():
    class _Gone:
        def close(self):
            raise RuntimeError("EGL display already terminated")

    renderer_teardown.close_renderer(_Gone())  # no exception
