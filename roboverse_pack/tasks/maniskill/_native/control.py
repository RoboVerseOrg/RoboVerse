"""Vendored ManiSkill PD joint-position controllers (byte-faithful math).

These mirror ``mani_skill.agents.controllers.pd_joint_pos`` so the shipped MetaSim tasks consume the
exact same action contract as native ManiSkill (a normalized ``Box(-1, 1)``) and turn it into the
absolute joint-position drive targets the sapien3 handler applies. Kept dependency-free (numpy only)
so the package is clone-deletable.
"""

from __future__ import annotations

import numpy as np


def clip_and_scale(action: np.ndarray, low: float, high: float) -> np.ndarray:
    """ManiSkill ``gym_utils.clip_and_scale_action`` (float32, matches the torch path)."""
    a = np.clip(np.float32(action), np.float32(-1), np.float32(1))
    return (np.float32(0.5) * np.float32(high + low) + np.float32(0.5) * np.float32(high - low) * a).astype(np.float32)


class PandaPDJointDeltaPos:
    """ManiSkill ``pd_joint_delta_pos`` for the Panda: 7 arm-delta dims + 1 mimic-gripper dim.

    ``compute_targets(current_qpos, action)`` returns the 9 absolute joint-position targets, ordered
    ``[arm1..arm7, finger1, finger2]`` (the panda URDF active-joint order).
    """

    ARM_DOF = 7
    ARM_DELTA = (-0.1, 0.1)  # ManiSkill PDJointPosControllerConfig lower/upper (arm)
    GRIPPER_RANGE = (-0.01, 0.04)  # gripper mimic absolute range

    action_dim = 8

    def compute_targets(self, current_qpos: np.ndarray, action: np.ndarray) -> np.ndarray:
        q = np.asarray(current_qpos, dtype=np.float32).ravel()
        action = np.asarray(action, dtype=np.float32).ravel()
        target = q.copy()
        # Arm: delta added to the qpos at action time (use_delta=True, use_target=False).
        target[: self.ARM_DOF] = q[: self.ARM_DOF] + clip_and_scale(action[: self.ARM_DOF], *self.ARM_DELTA)
        # Gripper: absolute (use_delta=False); the second finger mimics the first (×1 + 0).
        grip = clip_and_scale(action[self.ARM_DOF], *self.GRIPPER_RANGE)
        target[self.ARM_DOF :] = grip
        return target
