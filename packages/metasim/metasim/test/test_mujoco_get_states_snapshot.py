"""Regression tests for the snapshot-based joint reads in ``MujocoHandler``.

Background: ``_get_states`` used to call
``self.physics.data.joint(name).qpos.item()`` for every joint, every
call. Under tight ``set_states → simulate → get_states`` loops, that
NamedIndexer path produced alternating-arm corruption — single-frame
~150° spikes on otherwise-smooth trajectories, 271/300 rollouts
affected (see ``bug_get_states_arm_qpos_alternating_corruption.md``).

The fix snapshots ``data.qpos`` / ``data.qvel`` / ``data.ctrl`` /
``data.actuator_force`` once at the top of the call and reads each
joint via ``model.joint(name).qposadr`` indexed into the snapshot.

These tests exercise the snapshot helpers directly so they can run
anywhere — no real MuJoCo runtime needed.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest


def _make_handler():
    """Return a bare ``MujocoHandler`` instance — bypasses __init__ since
    the helpers only need ``self.physics.model.joint(name).qposadr``."""
    from metasim.sim.mujoco.mujoco import MujocoHandler

    handler = MujocoHandler.__new__(MujocoHandler)
    handler.physics = SimpleNamespace(
        model=SimpleNamespace(joint=lambda name: _fake_joint(name)),
        data=SimpleNamespace(),
    )
    return handler


def _fake_joint(name: str):
    """Map joint name → fake (qposadr, dofadr). The mapping is fixed
    per name so tests can assert reads go to the right index."""
    table = {
        "robot/L0": (0, 0),
        "robot/L1": (1, 1),
        "robot/L2": (2, 2),
        "robot/R0": (3, 3),
        "robot/R1": (4, 4),
        "robot/R2": (5, 5),
    }
    qadr, dadr = table[name]
    return SimpleNamespace(qposadr=np.array([qadr]), dofadr=np.array([dadr]))


@pytest.mark.general
def test_read_joint_qpos_uses_model_qposadr_into_snapshot():
    handler = _make_handler()
    snapshot = np.array([10.0, 11.0, 12.0, 20.0, 21.0, 22.0])
    assert handler._read_joint_qpos("robot/L0", snapshot) == 10.0
    assert handler._read_joint_qpos("robot/R2", snapshot) == 22.0


@pytest.mark.general
def test_read_joint_qvel_uses_model_dofadr_into_snapshot():
    handler = _make_handler()
    snapshot = np.array([0.5, 0.6, 0.7, 1.5, 1.6, 1.7])
    assert handler._read_joint_qvel("robot/L1", snapshot) == 0.6
    assert handler._read_joint_qvel("robot/R0", snapshot) == 1.5


@pytest.mark.general
def test_snapshot_isolates_caller_from_subsequent_mutations():
    """The whole point of the snapshot: once captured, a downstream
    mutation of ``data.qpos`` must not change values already read."""
    handler = _make_handler()
    qpos_live = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    qpos_snapshot = np.array(qpos_live, copy=True)

    # Simulate the kind of overwrite a substep might do
    qpos_live[:] = 999.0

    # Snapshot still has the captured values
    assert handler._read_joint_qpos("robot/L0", qpos_snapshot) == 1.0
    assert handler._read_joint_qpos("robot/R2", qpos_snapshot) == 6.0


@pytest.mark.general
def test_snapshot_helpers_return_python_floats_not_views():
    """Returning Python ``float`` matters — torch.tensor([...]) of floats
    has no shared-memory link back to the snapshot, so the downstream
    tensor is immune to any further mutation of MjData."""
    handler = _make_handler()
    snapshot = np.array([7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
    value = handler._read_joint_qpos("robot/L1", snapshot)
    assert type(value) is float
