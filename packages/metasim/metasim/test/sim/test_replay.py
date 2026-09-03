"""Record → replay contracts every backend must meet (see ``metasim.utils.replay``).

L0 — the same actions from the same initial state reproduce the recording on the same backend.
L1 — any recorded state written back with ``set_states`` reads back unchanged (positions *and*
velocities) and one step from it reproduces the recorded next state. Until L1 held, no mid-episode
state was a usable checkpoint: MuJoCo's ``_set_states`` zeroed every velocity and SuperDex ignored
root velocities, so "restore and continue" silently diverged from the original run.

Runs on the shared init-pose handler (Franka + primitives + a scaled URDF object + an articulated
box) with 1 env; multi-env handlers are exercised through the same code path with env 0 anchoring.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from loguru import logger as log

from metasim.utils.replay import record, verify_action_replay, verify_state_replay

STEPS = 120
SETTLE_STEPS = 30
# L1 one-step tolerance per backend. The TensorState round-trip is exact (1e-6) everywhere; what
# differs is engine state that no public API exposes. SuperDex (position-based dynamics) rebuilds
# its previous-position buffer from the written velocity, which leaves a ~1e-3 rad/s residual on
# the first step after a restore (measured 1.3e-3 on the Franka). Widen only where measured; a new
# backend gets the strict default until someone characterises its residual.
L1_STEP_TOL = {"superdex": 2e-3}
DEFAULT_L1_STEP_TOL = 1e-4
HOME = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04])
ORDER = [
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_joint7",
    "panda_finger_joint1",
    "panda_finger_joint2",
]


def _actions(handler, steps: int, seed: int = 0) -> list[torch.Tensor]:
    """Smooth sinusoidal joint targets around the home pose, in the handler's sorted joint order."""
    names = handler.get_joint_names("franka", sort=True)
    rng = np.random.default_rng(seed)
    phase = rng.uniform(0, 2 * np.pi, len(ORDER))
    amp = np.r_[np.full(7, 0.4), 0.1, 0.1]
    col = {n: i for i, n in enumerate(names)}
    out = []
    for t in range(steps):
        target = HOME + amp * np.sin(0.05 * t + phase)
        a = torch.zeros(handler.num_envs, len(names))
        for j, n in enumerate(ORDER):
            a[:, col[n]] = float(target[j])
        out.append(a)
    return out


@pytest.fixture
def trajectory(handler):
    """A recording from a settled rest state; shared by the L0 and L1 tests of one handler."""
    handler.set_states(handler.get_states(mode="tensor"))
    for _ in range(SETTLE_STEPS):
        handler.simulate()
    init = handler.get_states(mode="tensor")
    return record(handler, init, _actions(handler, STEPS))


@pytest.mark.sim("mujoco", "superdex", "newton", "sapien3", "pybullet", "isaacsim", "isaacgym")
def test_l0_action_replay_reproduces_recording(handler, trajectory):
    report = verify_action_replay(handler, trajectory, tol=1e-4)
    log.info(f"[{handler.scenario.simulator}] {report}")
    assert report.passed, str(report)


@pytest.mark.sim("mujoco", "superdex", "newton", "sapien3", "pybullet", "isaacsim", "isaacgym")
def test_l1_state_roundtrip_and_one_step(handler, trajectory):
    tol_step = L1_STEP_TOL.get(handler.scenario.simulator, DEFAULT_L1_STEP_TOL)
    roundtrip, one_step = verify_state_replay(handler, trajectory, every=10, tol_roundtrip=1e-6, tol_step=tol_step)
    log.info(f"[{handler.scenario.simulator}] {roundtrip}\n{one_step}")
    assert roundtrip.passed, str(roundtrip)
    assert one_step.passed, str(one_step)


@pytest.mark.sim("mujoco", "superdex", "newton", "sapien3", "pybullet", "isaacsim", "isaacgym")
def test_set_states_restores_velocities(handler):
    """A state carrying non-zero velocities must come back with those velocities, not zeros."""
    handler.set_states(handler.get_states(mode="tensor"))
    for a in _actions(handler, 40):
        handler.set_dof_targets(a)
        handler.simulate()
    moving = handler.get_states(mode="tensor")
    robot = moving.robots["franka"]
    assert robot.joint_vel is not None and float(robot.joint_vel.abs().max()) > 1e-3, "robot did not move"
    handler.set_states(moving)
    back = handler.get_states(mode="tensor").robots["franka"]
    assert torch.allclose(back.joint_vel.cpu(), robot.joint_vel.cpu(), atol=1e-6), (
        f"joint velocities not restored: max |Δ| = {(back.joint_vel.cpu() - robot.joint_vel.cpu()).abs().max():.3e}"
    )
