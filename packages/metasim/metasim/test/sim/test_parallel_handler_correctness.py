"""Behavioural tests for the ``_worker`` loop in ``ParallelSimWrapper``.

The worker is driven over a ``multiprocessing.Pipe`` and dispatches
commands to a per-env handler. These tests run it in a background
*thread* (not a process) against a stub handler, so they exercise the
real command-dispatch and teardown logic without needing a native
backend.

What they pin down:

- a construction failure surfaces the real exception, not a NameError
  from the ``finally`` block,
- ``refresh_render`` is a routed command, not a parent-side no-op,
- ``get_body_names`` forwards the ``sort`` argument,
- ``close`` is only called when the env was actually constructed.
"""

from __future__ import annotations

import multiprocessing as mp
import queue
import threading

import pytest

from metasim.sim import parallel

# ``_worker`` ends its error path with ``sys.exit(1)`` — correct for a
# subprocess, but in these thread-based tests it surfaces as an
# unhandled-thread-exception warning. The error is asserted on via the
# error queue instead.
pytestmark = pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")


class _StubHandler:
    """Minimal handler: records every dispatched call on the class."""

    calls: list = []

    def __init__(self):
        type(self).calls.append(("__init__", ()))

    def launch(self):
        type(self).calls.append(("launch", ()))

    def render(self):
        type(self).calls.append(("render", ()))

    def refresh_render(self):
        type(self).calls.append(("refresh_render", ()))

    def set_states(self, states):
        type(self).calls.append(("set_states", (states,)))

    def get_states(self, **kwargs):
        type(self).calls.append(("get_states", kwargs))
        return {"stub": "state"}

    def simulate(self):
        type(self).calls.append(("simulate", ()))

    def get_joint_names(self, obj_name, sort):
        type(self).calls.append(("get_joint_names", (obj_name, sort)))
        return ["j0", "j1"]

    def get_body_names(self, obj_name, sort):
        type(self).calls.append(("get_body_names", (obj_name, sort)))
        return ["b0", "b1"]

    def set_dof_targets(self, actions):
        type(self).calls.append(("set_dof_targets", (actions,)))

    @property
    def device(self):
        return "cpu"

    def close(self):
        type(self).calls.append(("close", ()))


def _run_worker_thread(handler_class):
    """Start ``_worker`` in a thread; return (parent_conn, error_queue, thread).

    ``_worker`` closes its ``parent_remote`` argument on entry (the
    process-mode convention of dropping the end it doesn't own), so we
    hand it a throwaway pipe end rather than the channel the test drives.
    """
    parent_conn, child_conn = mp.Pipe()
    _throwaway_a, throwaway_b = mp.Pipe()
    error_queue: queue.Queue = queue.Queue()
    thread = threading.Thread(
        target=parallel._worker,
        args=(0, child_conn, throwaway_b, error_queue, handler_class),
        daemon=True,
    )
    thread.start()
    return parent_conn, error_queue, thread


@pytest.fixture(autouse=True)
def _reset_stub_calls():
    _StubHandler.calls = []


@pytest.mark.general
def test_worker_dispatches_commands_and_closes_cleanly():
    """The worker runs the command loop and closes the env on ``close``."""
    parent, errq, thread = _run_worker_thread(_StubHandler)

    parent.send(("launch", (None,)))
    parent.send(("set_states", ([{"objects": {}}],)))
    parent.send(("simulate", (None,)))
    parent.send(("get_states", ({"mode": "tensor"},)))
    assert parent.recv() == {"stub": "state"}
    parent.send(("get_body_names", ("robot", False)))
    assert parent.recv() == ["b0", "b1"]
    parent.send(("refresh_render", (None,)))
    parent.send(("close", (None,)))
    thread.join(timeout=5)

    assert not thread.is_alive(), "worker thread did not exit on close"
    names = [c[0] for c in _StubHandler.calls]
    assert names == [
        "__init__",
        "launch",
        "set_states",
        "simulate",
        "get_states",
        "get_body_names",
        "refresh_render",
        "close",
    ]
    assert errq.empty()


@pytest.mark.general
def test_worker_forwards_sort_arg_to_get_body_names():
    """``get_body_names`` must reach the handler with the ``sort`` value
    the caller passed, not a dropped/defaulted one.
    """
    parent, _errq, thread = _run_worker_thread(_StubHandler)

    parent.send(("get_body_names", ("robot", False)))
    parent.recv()
    parent.send(("get_joint_names", ("robot", True)))
    parent.recv()
    parent.send(("close", (None,)))
    thread.join(timeout=5)

    by_name = dict((c[0], c[1]) for c in _StubHandler.calls if c[0] in ("get_body_names", "get_joint_names"))
    assert by_name["get_body_names"] == ("robot", False)
    assert by_name["get_joint_names"] == ("robot", True)


@pytest.mark.general
def test_worker_construction_failure_reports_real_error():
    """If ``handler_class()`` raises, the worker queues *that* error —
    not a NameError from a ``finally: env.close()`` on an unbound name.
    """

    class _ExplodingHandler:
        def __init__(self):
            raise RuntimeError("backend init blew up")

    parent, errq, thread = _run_worker_thread(_ExplodingHandler)
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert not errq.empty(), "construction failure should have been queued"
    err_type, err_msg, _tb = errq.get_nowait()
    assert err_type == "RuntimeError"
    assert "backend init blew up" in err_msg
    parent.close()


@pytest.mark.general
def test_worker_does_not_close_unconstructed_env():
    """A construction failure must not trigger a close on a nonexistent env
    (which historically raised NameError and masked the real error).
    """

    class _ExplodingHandler:
        closed = False

        def __init__(self):
            raise RuntimeError("nope")

        def close(self):  # pragma: no cover - must never run
            type(self).closed = True

    parent, errq, thread = _run_worker_thread(_ExplodingHandler)
    thread.join(timeout=5)

    assert _ExplodingHandler.closed is False
    parent.close()


@pytest.mark.general
def test_worker_handler_error_mid_loop_is_queued():
    """An exception raised by a command handler is captured on the error
    queue rather than silently killing the loop.
    """

    class _FailingSimulate(_StubHandler):
        def simulate(self):
            raise ValueError("simulate failed")

    parent, errq, thread = _run_worker_thread(_FailingSimulate)
    parent.send(("simulate", (None,)))
    thread.join(timeout=5)

    assert not errq.empty()
    err_type, err_msg, _tb = errq.get_nowait()
    assert err_type == "ValueError"
    assert "simulate failed" in err_msg
    parent.close()


@pytest.mark.general
def test_parallel_set_states_routes_rows_by_env_id():
    """One rule for every input shape: a full batch is indexed by env id (a permuted ``env_ids`` does
    not transpose anything), one state is broadcast, one-per-env_id is positional, anything else raises.
    """
    from metasim.test.test_parallel_error_handling_general import _FakeRemote, _make_parallel_stub_handler

    remotes = [_FakeRemote() for _ in range(3)]
    h = _make_parallel_stub_handler(remotes, processes=[])  # the shared stub wrapper, real _check_error
    states = [{"objects": {}, "robots": {"r": {"env": i}}} for i in range(3)]

    def sent():
        out = [[m for m in r.sent if m[0] == "set_states"] for r in remotes]
        for r in remotes:
            r.sent.clear()
        return out

    h._set_states(states, env_ids=[2])
    assert sent() == [[], [], [("set_states", ([states[2]],))]]
    h._set_states(states, env_ids=[2, 1, 0])  # permuted full batch: still row k -> env k
    assert sent() == [[("set_states", ([states[i]],))] for i in range(3)]
    h._set_states([states[1]], env_ids=[0, 2])  # broadcast
    assert sent() == [[("set_states", ([states[1]],))], [], [("set_states", ([states[1]],))]]
    h._set_states([states[0], states[2]], env_ids=[0, 2])  # positional, one per env_id
    assert sent() == [[("set_states", ([states[0]],))], [], [("set_states", ([states[2]],))]]
    with pytest.raises(ValueError, match="pass one state"):
        h._set_states([states[0], states[1]], env_ids=[2])
