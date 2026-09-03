"""Regression tests for ``ParallelSimWrapper`` error-surfacing.

Background: ``_check_error()`` used to be called only from ``launch``,
so a worker that died on any other operation (OOM, GPU failure, asset
load error) left the parent either hanging on ``remote.recv()`` or
eventually raising a cryptic ``EOFError`` — the real traceback in
``error_queue`` was never surfaced. Architecture Review Issue 7.

These tests exercise the surfacing logic against a stub
``error_queue`` / ``processes`` / ``remotes`` so they run in the
``general`` suite — no real backend, no GPU.

Forward / backward compat: the new behavior is strictly an information
upgrade — every former-silent failure now raises a ``RuntimeError``
with the real worker traceback. Calls that previously succeeded
continue to succeed unchanged.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import pytest

from metasim.sim.base import BaseSimHandler
from metasim.sim.parallel import ParallelSimWrapper


class _SyncQueue:
    """In-test stand-in for ``mp.Queue`` with the same ``put``/``get``/``empty``
    surface. We use this because ``mp.Queue.put`` is async — the data is
    only visible to ``empty()`` after a background thread ferries it
    through the pipe, which races with the immediate-check in these tests.
    The production code path against a real ``mp.Queue`` is exercised by
    integration tests in ``test_parallel_handler.py``."""

    def __init__(self):
        self._items = deque()

    def put(self, item):
        self._items.append(item)

    def get(self):
        return self._items.popleft()

    def empty(self) -> bool:
        return not self._items


class _StubBaseHandler(BaseSimHandler):
    """Concrete enough to satisfy ``ParallelSimWrapper`` import time;
    never actually instantiated in these tests."""

    def __init__(self, *_a, **_k):
        pass

    def _set_states(self, *_a, **_k):
        pass

    def _set_dof_targets(self, *_a, **_k):
        pass

    def _get_states(self, *_a, **_k):
        return None

    def _simulate(self):
        pass

    def _get_joint_names(self, *_a, **_k):
        return []

    def _get_body_names(self, *_a, **_k):
        return []

    def close(self):
        pass


class _FakeRemote:
    def __init__(self, recv_raises: type[Exception] | None = None, recv_value: Any = None):
        self._recv_raises = recv_raises
        self._recv_value = recv_value
        self.sent: list = []

    def send(self, msg):
        self.sent.append(msg)

    def recv(self):
        if self._recv_raises is not None:
            raise self._recv_raises()
        return self._recv_value


class _FakeProcess:
    def __init__(self, alive: bool = True, exitcode: int | None = None):
        self._alive = alive
        self.exitcode = exitcode

    def is_alive(self) -> bool:
        return self._alive

    def join(self, timeout: float | None = None) -> None:
        pass

    def terminate(self) -> None:
        self._alive = False


def _make_parallel_stub_handler(remotes, processes, error_queue=None) -> Any:
    """Build a ``ParallelHandler`` instance that bypasses ``__init__`` so
    we can plug in fake remotes / processes / error_queue."""
    cls = ParallelSimWrapper(_StubBaseHandler)
    # ParallelHandler.__new__ requires a scenario; bypass it for the stub.
    h = object.__new__(cls)
    h.remotes = remotes
    h.processes = processes
    h.error_queue = error_queue if error_queue is not None else _SyncQueue()
    h.closed = False
    h.waiting = False
    h._num_envs = len(remotes)
    h._tensor_state_cache = None
    h._dict_state_cache = None
    return h


@pytest.mark.general
def test_check_error_surfaces_queue_error_with_traceback():
    err_queue = _SyncQueue()
    err_queue.put(("RuntimeError", "test failure", ["Traceback line 1\n", "Traceback line 2\n"]))
    handler = _make_parallel_stub_handler(
        remotes=[_FakeRemote()],
        processes=[_FakeProcess(alive=True)],
        error_queue=err_queue,
    )

    with pytest.raises(RuntimeError, match=r"Parallel worker error \(RuntimeError\): test failure"):
        handler._check_error()


@pytest.mark.general
def test_check_error_surfaces_dead_worker_without_queue_entry():
    """OOM-kill / SIGKILL leaves error_queue empty but the process is dead.
    Without this detection the parent would hang on the next recv()."""
    handler = _make_parallel_stub_handler(
        remotes=[_FakeRemote(), _FakeRemote()],
        processes=[_FakeProcess(alive=True), _FakeProcess(alive=False, exitcode=-9)],
    )

    with pytest.raises(RuntimeError, match=r"died without reporting an error"):
        handler._check_error()


@pytest.mark.general
def test_check_error_quiet_when_workers_alive_and_queue_empty():
    """Hot path — must be cheap and silent in the normal case."""
    handler = _make_parallel_stub_handler(
        remotes=[_FakeRemote()],
        processes=[_FakeProcess(alive=True)],
    )
    # Should return without raising or logging.
    handler._check_error()


@pytest.mark.general
def test_check_error_no_op_after_close():
    handler = _make_parallel_stub_handler(
        remotes=[_FakeRemote()],
        processes=[_FakeProcess(alive=False, exitcode=0)],
    )
    handler.closed = True
    # Even with a dead worker, post-close calls must not raise.
    handler._check_error()


@pytest.mark.general
def test_recv_or_surface_translates_eof_into_runtime_error():
    """EOFError on recv = pipe closed = worker died. The user should see
    a clear RuntimeError, not a cryptic IPC exception."""
    err_queue = _SyncQueue()
    err_queue.put(("OSError", "GPU lost", ["worker traceback\n"]))
    handler = _make_parallel_stub_handler(
        remotes=[_FakeRemote(recv_raises=EOFError)],
        processes=[_FakeProcess(alive=False, exitcode=1)],
        error_queue=err_queue,
    )

    # _recv_or_surface should drain the error_queue and re-raise the real error.
    with pytest.raises(RuntimeError, match=r"Parallel worker error \(OSError\): GPU lost"):
        handler._recv_or_surface(0)


@pytest.mark.general
def test_recv_or_surface_raises_clear_message_when_queue_empty():
    """Worker killed externally with no traceback in queue — still must
    surface a clear, actionable RuntimeError instead of EOFError."""
    handler = _make_parallel_stub_handler(
        remotes=[_FakeRemote(recv_raises=BrokenPipeError)],
        processes=[_FakeProcess(alive=False, exitcode=-9)],
    )

    with pytest.raises(RuntimeError, match=r"died without reporting"):
        handler._recv_or_surface(0)


@pytest.mark.general
def test_set_seed_forwards_to_every_worker():
    """Reproducibility contract for parallel rollouts: ``set_seed(N)`` on
    the parent must seed each worker process. Without per-worker
    forwarding, ``env.reset(seed=42)`` only seeds the parent's RNG and
    the actual per-env rollouts diverge despite the seed."""
    remotes = [_FakeRemote(), _FakeRemote(), _FakeRemote()]
    handler = _make_parallel_stub_handler(
        remotes=remotes,
        processes=[_FakeProcess(alive=True), _FakeProcess(alive=True), _FakeProcess(alive=True)],
    )

    handler.set_seed(123)

    # Each worker's pipe should have received a ("set_seed", (123,)) command.
    for i, remote in enumerate(remotes):
        assert remote.sent, f"worker {i} received no command"
        cmd, payload = remote.sent[-1]
        assert cmd == "set_seed", f"worker {i} got {cmd!r}, expected 'set_seed'"
        assert payload == (123,), f"worker {i} got payload {payload!r}, expected (123,)"


@pytest.mark.general
def test_launch_surfaces_worker_error_instead_of_eof():
    """A worker that dies inside ``launch`` used to surface as a bare ``EOFError`` from the
    handshake ``recv``; the real traceback in ``error_queue`` must be what the caller sees."""
    err_queue = _SyncQueue()
    err_queue.put(("FileNotFoundError", "missing.usd", ["worker traceback\n"]))
    handler = _make_parallel_stub_handler(
        remotes=[_FakeRemote(recv_raises=EOFError)],
        processes=[_FakeProcess(alive=False, exitcode=1)],
        error_queue=err_queue,
    )
    with pytest.raises(RuntimeError, match=r"Parallel worker error \(FileNotFoundError\): missing\.usd"):
        handler.launch()


@pytest.mark.general
def test_close_tolerates_dead_worker():
    """``close`` must not raise ``BrokenPipeError`` on a worker that already died."""

    class _DeadRemote(_FakeRemote):
        def send(self, msg):
            raise BrokenPipeError

    handler = _make_parallel_stub_handler(
        remotes=[_DeadRemote()],
        processes=[_FakeProcess(alive=False, exitcode=1)],
    )
    handler.close()
    assert handler.closed


class _BoomHandler(_StubBaseHandler):
    """Handler whose construction fails, as an asset-loading error inside a worker would."""

    def __init__(self, *_a, **_k):
        raise RuntimeError("asset missing: boom.urdf")


@pytest.mark.general
def test_constructor_reports_worker_construction_error():
    """End-to-end with real worker processes: a handler that raises in ``__init__`` used to
    surface as ``EOFError`` from the constructor handshake, with the traceback lost."""
    from metasim.scenario.scenario import ScenarioCfg

    scenario = ScenarioCfg(num_envs=2)
    with pytest.raises(RuntimeError, match=r"asset missing: boom\.urdf") as excinfo:
        ParallelSimWrapper(_BoomHandler)(scenario)
    assert "EOFError" not in str(excinfo.value)
