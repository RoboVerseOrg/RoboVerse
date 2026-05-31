"""Dep-free tests for the RoboTwin closed-loop eval harness policy layer.

The env rollout needs a RoboTwin checkout + sim, but the policy abstraction
(``ReplayPolicy`` emitting recorded waypoints, the ``--policy`` builder, and the
DP-without-checkpoint guard) is pure Python and is what the harness's correctness
hinges on. These lock it without a sim: a replay policy must emit each recorded
command once in order then stop, and an unconfigured DP policy must fail loudly,
never silently no-op.
"""

from __future__ import annotations

import importlib.util
import os

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_EVAL = os.path.join(_HERE, "..", "tools", "robotwin_integration", "eval_robotwin_policy.py")


def _load_eval():
    spec = importlib.util.spec_from_file_location("rt_eval", _EVAL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.general
def test_replay_policy_emits_each_waypoint_once_then_stops():
    ev = _load_eval()
    bridge = {"seed": 4, "vectors": [np.arange(14) + i for i in range(3)]}
    pol = ev.ReplayPolicy(bridge)
    assert pol.seed == 4
    out = [pol.predict({}) for _ in range(5)]
    # 3 recorded waypoints emitted in order, then None (episode should end).
    assert np.allclose(out[0], bridge["vectors"][0])
    assert np.allclose(out[2], bridge["vectors"][2])
    assert out[3] is None and out[4] is None


@pytest.mark.general
def test_replay_policy_reset_rewinds():
    ev = _load_eval()
    pol = ev.ReplayPolicy({"seed": 0, "vectors": [np.zeros(14), np.ones(14)]})
    pol.predict({})
    pol.reset()
    assert np.allclose(pol.predict({}), np.zeros(14))  # back to the first waypoint


@pytest.mark.general
def test_dp_policy_without_server_fails_loud():
    """Selecting --policy dp without a --server must raise, not silently no-op."""
    ev = _load_eval()

    class A:
        policy, bridge, server, camera = "dp", None, None, "head_camera"

    with pytest.raises(SystemExit):
        ev._build_policy(A())  # dp needs --server


@pytest.mark.general
def test_dp_joint_qpos_prefers_achieved_then_command():
    """The DP client reads achieved real_vector, falling back to the command vector."""
    ev = _load_eval()
    import numpy as np

    real = np.arange(14, dtype=float)
    obs_real = {"joint_action": {"real_vector": real, "vector": np.zeros(14)}}
    assert np.allclose(ev.DPPolicy._robotwin_joint_qpos(obs_real), real)
    obs_cmd = {"joint_action": {"vector": real}}
    assert np.allclose(ev.DPPolicy._robotwin_joint_qpos(obs_cmd), real)


@pytest.mark.general
def test_dp_client_chunk_caching(monkeypatch):
    """DPPolicy emits one action per step from a cached chunk, re-querying when empty."""
    ev = _load_eval()
    import numpy as np

    # Build a DPPolicy without opening a socket (skip __init__), wire a fake RPC.
    pol = ev.DPPolicy.__new__(ev.DPPolicy)
    pol.camera = "head_camera"
    pol._cache = []
    calls = {"n": 0}

    def fake_rpc(msg):
        calls["n"] += 1
        # return a fresh 3-step chunk each predict, tagged by call index
        base = calls["n"] * 10
        return {"action_chunk": [np.full(14, base + i, dtype=float) for i in range(3)]}

    pol._rpc = fake_rpc
    obs = {"observation": {"head_camera": {"rgb": np.zeros((4, 4, 3), np.uint8)}}, "robot_joint_state": np.zeros(14)}

    a0 = pol.predict(obs)
    a1 = pol.predict(obs)
    a2 = pol.predict(obs)  # exhausts the first chunk (3 actions, 1 rpc call)
    assert calls["n"] == 1
    assert a0[0] == 10 and a1[0] == 11 and a2[0] == 12
    a3 = pol.predict(obs)  # cache empty -> a second rpc call
    assert calls["n"] == 2
    assert a3[0] == 20


@pytest.mark.general
def test_dp_client_raises_on_server_error(monkeypatch):
    """A server-side predict error must propagate, not be silently swallowed."""
    ev = _load_eval()
    import numpy as np

    pol = ev.DPPolicy.__new__(ev.DPPolicy)
    pol.camera = "head_camera"
    pol._cache = []
    pol._rpc = lambda msg: {"error": "boom", "traceback": "..."}
    obs = {"observation": {"head_camera": {"rgb": np.zeros((4, 4, 3), np.uint8)}}, "robot_joint_state": np.zeros(14)}
    with pytest.raises(RuntimeError, match="dp server predict failed"):
        pol.predict(obs)


@pytest.mark.general
def test_policy_builder_validates_required_args():
    ev = _load_eval()

    class A:
        policy, bridge, server, camera = "replay", None, None, "head_camera"

    with pytest.raises(SystemExit):
        ev._build_policy(A())  # replay needs --bridge
