"""Deterministic scripted policies that drive non-trivial reward/success.

These are not learned — they exist so the replay-parity test exercises real
reaching / grasping / lifting reward and (for Lift) genuine task success,
rather than the all-zeros signal a passive policy produces.
"""

from __future__ import annotations

import numpy as np


def lift_grasp_policy():
    """Closed-loop servo for robosuite ``Lift``: reach → descend → grasp → lift."""

    def policy(t: int, obs: dict) -> np.ndarray:
        eef = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
        cube = np.asarray(obs["cube_pos"], dtype=np.float64)
        delta = cube - eef
        horiz = np.linalg.norm(delta[:2])
        a = np.zeros(7)
        above_cube = horiz < 0.015
        # gripper: open while approaching, close once over the cube
        grasp = above_cube and delta[2] > -0.02
        a[6] = 1.0 if (grasp or t > 45) else -1.0
        if not above_cube:
            a[:2] = np.clip(delta[:2] * 20.0, -1, 1)
            a[2] = np.clip(delta[2] * 20.0, -1, 1) * 0.3  # hover, don't dive
        elif t < 45:
            a[:2] = np.clip(delta[:2] * 20.0, -1, 1)
            a[2] = np.clip(delta[2] * 20.0, -1, 1)  # descend onto cube
        else:
            a[2] = 1.0  # lift
        return a

    return policy


def descend_close_policy(action_dim: int = 7):
    """Open-loop press-down-and-close — works for any gripper task as a probe."""

    def policy(t: int, obs: dict | None = None) -> np.ndarray:
        a = np.zeros(action_dim)
        a[2] = -1.0 if t < 30 else 1.0
        if action_dim >= 7:
            a[6] = 1.0 if t >= 25 else -1.0
        a[0] = 0.1 * np.sin(2 * np.pi * t / 30)
        return a

    return policy
