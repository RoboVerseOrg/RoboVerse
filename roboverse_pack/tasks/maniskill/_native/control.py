# Copyright (c) ManiSkill contributors
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from ManiSkill (https://github.com/haosulab/ManiSkill).
# Changes: reimplemented in numpy (no upstream file copied) so MetaSim's sapien3 handler reproduces the
#   `pd_joint_delta_pos` action contract of `mani_skill.agents.controllers.pd_joint_pos` — same normalized
#   `Box(-1, 1)` action space, same float32 clip-and-scale, same arm-delta + absolute mimic-gripper target math.
# Full license: roboverse_pack/tasks/simpler_env/_native/control/LICENSE
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


class PDJointDeltaPos:
    """ManiSkill ``pd_joint_delta_pos`` controller, generalized over the robot's action layout.

    ``compute_targets(current_qpos, action)`` returns the absolute joint-position targets:

    * the first ``arm_dof`` dims are arm deltas (clip→scale ``arm_range`` → added to current qpos),
    * if ``gripper`` is True, one trailing dim is the absolute mimic-gripper target
      (clip→scale ``gripper_range``) written to every remaining DOF (the panda's two fingers).

    Covers Panda (``arm_dof=7, gripper=True`` → 8-dim) and PandaStick / arm-only robots
    (``arm_dof=7, gripper=False`` → 7-dim), matching ManiSkill's per-robot action contract 1:1.
    """

    def __init__(self, arm_dof: int = 7, arm_range=(-0.1, 0.1), gripper: bool = True, gripper_range=(-0.01, 0.04)):
        self.arm_dof = arm_dof
        self.arm_range = arm_range
        self.gripper = gripper
        self.gripper_range = gripper_range

    @property
    def action_dim(self) -> int:
        return self.arm_dof + (1 if self.gripper else 0)

    def compute_targets(self, current_qpos: np.ndarray, action: np.ndarray) -> np.ndarray:
        q = np.asarray(current_qpos, dtype=np.float32).ravel()
        action = np.asarray(action, dtype=np.float32).ravel()
        target = q.copy()
        # Arm: delta added to the qpos at action time (use_delta=True, use_target=False).
        target[: self.arm_dof] = q[: self.arm_dof] + clip_and_scale(action[: self.arm_dof], *self.arm_range)
        if self.gripper:
            # Gripper: absolute (use_delta=False); every remaining DOF mimics the control dim (×1 + 0).
            grip = clip_and_scale(action[self.arm_dof], *self.gripper_range)
            target[self.arm_dof :] = grip
        return target


class PandaPDJointDeltaPos(PDJointDeltaPos):
    """Panda default: 7 arm-delta dims + 1 mimic-gripper dim (8-dim)."""

    def __init__(self):
        super().__init__(arm_dof=7, arm_range=(-0.1, 0.1), gripper=True, gripper_range=(-0.01, 0.04))
