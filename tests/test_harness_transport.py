"""Transport tests: typed serialize round-trip, in-proc handle, live ws loopback, per-connection
isolation, broken-socket cleanup, and a real cross-process (subprocess) round-trip."""

from __future__ import annotations

import contextlib
import socket
import subprocess
import sys
import threading
import time

import pytest
import torch

from roboverse_learn.eval.harness.embodiment import infer_embodiment
from roboverse_learn.eval.harness.obs import ObsBatch
from roboverse_learn.eval.harness.spec import derive_action_spec, derive_obs_spec


class _Act:
    def __init__(self, is_ee=False):
        self.is_ee = is_ee


class _Robot:
    def __init__(self, name, joints, ee_joints=()):
        self.name = name
        self.joint_limits = {j: (0.0, 1.0) for j in joints}
        self.actuators = {j: _Act(is_ee=(j in ee_joints)) for j in joints}
        self.gripper_open_q = None
        self.gripper_joint_name = None
        self.ee_body_name = f"{name}_hand"


def _franka():
    arm = [f"panda_joint{i}" for i in range(1, 8)]
    grip = ["panda_finger_joint1", "panda_finger_joint2"]
    return _Robot("franka", arm + grip, ee_joints=grip)


def _specs():
    emb = infer_embodiment([_franka()])
    return derive_obs_spec(emb, include_ee_pose=True), derive_action_spec(emb, chunk_len=2)


def _obs(ospec, b=2):
    t = {}
    for f in ospec.fields:
        t[f.key] = torch.zeros(b, *f.shape, dtype=torch.uint8 if f.dtype == "uint8" else torch.float32)
    return ObsBatch(ospec, torch.arange(b), t)


@pytest.mark.general
def test_serialize_roundtrip():
    pytest.importorskip("msgpack")
    pytest.importorskip("msgpack_numpy")
    from roboverse_learn.eval.harness.transport import serialize as S

    ospec, aspec = _specs()
    # spec round-trip preserves fields/dtype/chunk_len
    assert S.obs_spec_from_wire(S.obs_spec_to_wire(ospec)).keys() == ospec.keys()
    a2 = S.action_spec_from_wire(S.action_spec_to_wire(aspec))
    assert a2.chunk_len == 2 and a2.keys() == aspec.keys()
    # obs carrier round-trip through msgpack bytes preserves tensors
    obs = _obs(ospec)
    obs.tensors["arm.joint_pos"] = torch.arange(2 * 7, dtype=torch.float32).reshape(2, 7)
    back = S.obs_from_wire(S.decode(S.encode(S.obs_to_wire(obs))))
    assert torch.allclose(back.tensors["arm.joint_pos"], obs.tensors["arm.joint_pos"])
    assert back.spec.keys() == ospec.keys()


@pytest.mark.general
def test_inprocess_handle_is_policy():
    from roboverse_learn.eval.harness.adapters import ZeroActionPolicy
    from roboverse_learn.eval.harness.transport.base import InProcessTransport, PolicyHandle

    ospec, aspec = _specs()
    handle = PolicyHandle(InProcessTransport(ZeroActionPolicy()))
    handle.bind(ospec, aspec)
    handle.reset(torch.arange(2))
    action = handle.act(_obs(ospec))
    assert set(action.tensors) == set(aspec.keys())
    assert torch.count_nonzero(action.tensors["arm.joint_pos"]) == 0


def _serve_in_thread(**kwargs) -> int:
    """Start serve_policy on a background thread; return the bound port."""
    from roboverse_learn.eval.harness.transport.ws import serve_policy

    port_box: dict[str, int] = {}
    ready = threading.Event()

    def _ready(p):
        port_box["p"] = p
        ready.set()

    th = threading.Thread(
        target=serve_policy,
        kwargs={"host": "localhost", "port": 0, "ready": _ready, **kwargs},
        daemon=True,
    )
    th.start()
    assert ready.wait(15), "ws server did not start"
    return port_box["p"]


@pytest.mark.general
def test_ws_loopback_zero_policy():
    pytest.importorskip("websockets")
    pytest.importorskip("msgpack")
    pytest.importorskip("msgpack_numpy")
    from roboverse_learn.eval.harness.adapters import ZeroActionPolicy
    from roboverse_learn.eval.harness.transport.base import PolicyHandle
    from roboverse_learn.eval.harness.transport.ws import WsPolicyTransport

    port = _serve_in_thread(policy=ZeroActionPolicy())
    handle = PolicyHandle(WsPolicyTransport(f"ws://localhost:{port}"))
    try:
        assert handle.describe().name == "zero"  # PolicyCard survived the hello handshake
        ospec, aspec = _specs()
        handle.bind(ospec, aspec)
        handle.reset(torch.arange(2))
        obs = _obs(ospec)
        obs.tensors["arm.joint_pos"] = torch.randn(2, 7)
        action = handle.act(obs)  # obs serialized -> remote act -> action serialized back
        assert set(action.tensors) == set(aspec.keys())
        assert torch.count_nonzero(action.tensors["arm.joint_pos"]) == 0  # zero policy over the wire
        assert list(action.env_ids) == [0, 1]
    finally:
        handle.close()


@pytest.mark.general
def test_ws_shared_policy_rejects_second_client():
    # BLOCKER regression: serve_policy(policy=<instance>) shared ONE policy across connections, so
    # a second sim process re-bound the first one's spec and got actions shaped for the wrong
    # embodiment. The second connection must be refused with an actionable error.
    pytest.importorskip("websockets")
    pytest.importorskip("msgpack_numpy")
    from roboverse_learn.eval.harness.adapters import ZeroActionPolicy
    from roboverse_learn.eval.harness.transport.ws import WsPolicyTransport, WsProtocolError

    port = _serve_in_thread(policy=ZeroActionPolicy())
    first = WsPolicyTransport(f"ws://localhost:{port}")
    first.connect()
    try:
        second = WsPolicyTransport(f"ws://localhost:{port}", timeout=10)
        with pytest.raises(WsProtocolError, match="single shared policy instance"):
            second.connect()
        second.close()
        # the first client is untouched
        ospec, aspec = _specs()
        first.bind(ospec, aspec)
        assert set(first.infer(_obs(ospec)).tensors) == set(aspec.keys())
    finally:
        first.close()


@pytest.mark.general
def test_ws_factory_gives_each_client_its_own_policy():
    # the isolated multi-client mode: factory= builds a fresh instance per connection, so two sim
    # processes can share one server without re-binding each other's spec.
    pytest.importorskip("websockets")
    pytest.importorskip("msgpack_numpy")
    from roboverse_learn.eval.harness.adapters import ZeroActionPolicy
    from roboverse_learn.eval.harness.transport.ws import WsPolicyTransport

    made: list[int] = []

    def _factory():
        p = ZeroActionPolicy()
        made.append(id(p))
        return p

    port = _serve_in_thread(factory=_factory)
    a = WsPolicyTransport(f"ws://localhost:{port}")
    b = WsPolicyTransport(f"ws://localhost:{port}")
    a.connect()
    b.connect()
    try:
        ospec, aspec = _specs()
        a.bind(ospec, aspec)
        b.bind(ospec, aspec)
        assert set(a.infer(_obs(ospec)).tensors) == set(aspec.keys())
        assert set(b.infer(_obs(ospec)).tensors) == set(aspec.keys())
        assert len(set(made)) == 2, "both connections shared one policy instance"
    finally:
        a.close()
        b.close()


@pytest.mark.general
def test_ws_serve_policy_requires_exactly_one_source():
    from roboverse_learn.eval.harness.transport.ws import serve_policy

    with pytest.raises(ValueError, match="exactly one"):
        serve_policy()
    with pytest.raises(ValueError, match="exactly one"):
        serve_policy(object(), factory=object)


@pytest.mark.general
def test_ws_timeout_closes_the_socket():
    # BLOCKER regression: on timeout/ConnectionClosed the transport set self._ws = None WITHOUT
    # closing it, and close() then skipped the close — every WsProtocolError leaked an open TCP
    # connection and a pending recv task.
    pytest.importorskip("websockets")
    pytest.importorskip("msgpack_numpy")
    from roboverse_learn.eval.harness.transport.ws import WsPolicyTransport, WsProtocolError

    class _SlowPolicy:
        def describe(self):
            from roboverse_learn.eval.harness.policy import PolicyCard
            from roboverse_learn.eval.harness.spec import ActionSpec, ObsSpec

            return PolicyCard("slow", ObsSpec(()), ActionSpec((), chunk_len=1))

        def bind(self, obs_spec, action_spec):
            pass

        def reset(self, env_ids):
            time.sleep(5)  # longer than the client's timeout

        def act(self, obs):
            raise AssertionError("not reached")

    port = _serve_in_thread(policy=_SlowPolicy())
    t = WsPolicyTransport(f"ws://localhost:{port}", timeout=0.5)
    t.connect()
    ws = t._ws  # keep a handle on the live socket the transport is about to abandon
    with pytest.raises(WsProtocolError, match="timed out"):
        t.reset(torch.arange(1))
    assert t._ws is None  # transport marked broken...
    assert ws.state.name in ("CLOSED", "CLOSING"), ws.state  # ...and the socket was actually closed
    with pytest.raises(WsProtocolError, match="closed/broken"):
        t.reset(torch.arange(1))
    t.close()  # must not raise even though the socket is already gone


@pytest.mark.general
def test_ws_cross_process_roundtrip():
    # the isolation claim, for real: the policy runs in a SEPARATE OS process (the serve CLI), the
    # client here talks to it over a socket. (The other ws tests run the server in a thread.)
    pytest.importorskip("websockets")
    pytest.importorskip("msgpack_numpy")
    from roboverse_learn.eval.harness.transport.base import PolicyHandle
    from roboverse_learn.eval.harness.transport.ws import WsPolicyTransport

    with contextlib.closing(socket.socket()) as s:  # ask the OS for a free port, then hand it over
        s.bind(("localhost", 0))
        port = s.getsockname()[1]
    proc = subprocess.Popen(
        [sys.executable, "-m", "roboverse_learn.eval.harness.transport.serve", "--policy", "zero", "--port", str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        handle = None
        for _ in range(100):  # wait for the server process to bind
            if proc.poll() is not None:
                raise AssertionError(f"serve process died: {proc.stdout.read()[-2000:]}")
            try:
                handle = PolicyHandle(WsPolicyTransport(f"ws://localhost:{port}", timeout=10))
                break
            except Exception:
                time.sleep(0.2)
        assert handle is not None, "could not connect to the serve subprocess"
        try:
            assert handle.describe().name == "zero"
            ospec, aspec = _specs()
            handle.bind(ospec, aspec)
            handle.reset(torch.arange(2))
            action = handle.act(_obs(ospec))
            assert set(action.tensors) == set(aspec.keys())
            assert torch.count_nonzero(action.tensors["arm.joint_pos"]) == 0
        finally:
            handle.close()
    finally:
        proc.terminate()
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=10)
