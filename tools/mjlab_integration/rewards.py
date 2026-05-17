"""Ports of mjlab reward formulas, applied to recorded rollouts.

We currently mirror the dm-control-style cartpole reward used by mjlab's
`cartpole_smooth_reward` so the report can show a side-by-side reward
curve between raw mujoco and MetaSim/MuJoCo rollouts driven under
identical control.

Source: mjlab/tasks/cartpole/cartpole_env_cfg.py:cartpole_smooth_reward.
Verified against mjlab's constants (`_GAUSSIAN_SCALE`,
`_QUADRATIC_SCALE` derived from value_at_margin=0.1).

Adding more rewards (velocity tracking for go1/g1, etc.) follows the
same pattern — compute the reward signal from RolloutResult.qpos /
qvel / ctrl and a few task-specific knobs.
"""

from __future__ import annotations

import math

import numpy as np

from .raw_rollout import RolloutResult

_GAUSSIAN_SCALE = math.sqrt(-2 * math.log(0.1))
_QUADRATIC_SCALE = math.sqrt(1 - 0.1)


def _gaussian_tolerance(x: np.ndarray, margin: float) -> np.ndarray:
    if margin == 0:
        return (x == 0).astype(np.float64)
    scaled = x / margin * _GAUSSIAN_SCALE
    return np.exp(-0.5 * scaled**2)


def _quadratic_tolerance(x: np.ndarray, margin: float) -> np.ndarray:
    if margin == 0:
        return (x == 0).astype(np.float64)
    scaled = x / margin * _QUADRATIC_SCALE
    return np.clip(1 - scaled**2, 0.0, None)


def cartpole_reward(rollout: RolloutResult, *, slider_jnt: str = "slider", hinge_jnt: str = "hinge_1") -> np.ndarray:
    """Reproduce mjlab's `cartpole_smooth_reward` over a rollout.

    Returns reward per control step (T,). The first qpos entry is the
    initial state, so reward at step k uses qpos[k+1] / qvel[k+1] /
    ctrl[k] — the post-step state for the action taken at step k.
    """
    if rollout.jnt_qposadr is None or rollout.jnt_dofadr is None:
        raise ValueError("RolloutResult must carry jnt_qposadr / jnt_dofadr for reward computation")

    short = [n.split("/")[-1] for n in rollout.joint_names]
    slider_idx = short.index(slider_jnt)
    hinge_idx = short.index(hinge_jnt)

    sl_qposadr = rollout.jnt_qposadr[slider_idx]
    hn_qposadr = rollout.jnt_qposadr[hinge_idx]
    hn_dofadr = rollout.jnt_dofadr[hinge_idx]

    # Use the post-step state (qpos[1:]) against the action that produced it (ctrl).
    cart_pos = rollout.qpos[1:, sl_qposadr]
    hinge_angle = rollout.qpos[1:, hn_qposadr]
    hinge_vel = rollout.qvel[1:, hn_dofadr]
    ctrl = rollout.ctrl[:, 0]  # cartpole has a single actuator

    pole_cos = np.cos(hinge_angle)
    upright = (pole_cos + 1) / 2
    centered = (1 + _gaussian_tolerance(cart_pos, margin=2.0)) / 2
    small_control = (4 + _quadratic_tolerance(ctrl, margin=1.0)) / 5
    small_velocity = (1 + _gaussian_tolerance(hinge_vel, margin=5.0)) / 2

    return upright * centered * small_control * small_velocity
