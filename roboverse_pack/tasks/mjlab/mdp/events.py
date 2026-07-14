# Copyright (c) mjlab contributors
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from mjlab (https://github.com/mujocolab/mjlab).
# Changes: event terms ported from mjlab's `envs/mdp/events` to RoboVerse's reset-event convention (`func(env,
#   env_ids, **params)`), mutating state through MetaSim's handler instead of mjlab's scene entities.
# Full license: roboverse_pack/tasks/mjlab/LICENSE
"""Event term functions ported from mjlab.

Signature for reset-mode events:
    ``func(env, env_ids: torch.Tensor, **params) -> None``  — mutates state in place

Reset events run after ``ManagerBasedRVEnv._reset_idx`` has restored
initial states; they're the standard mechanism for per-episode noise
injection (joint angle/velocity randomization, base velocity push, etc.).
"""

from __future__ import annotations

import torch

from .scene_entity import SceneEntityCfg, resolve_joint_ids


def reset_joints_by_offset(
    env,
    env_ids: torch.Tensor,
    position_range: tuple[float, float],
    velocity_range: tuple[float, float],
    asset_cfg: SceneEntityCfg,
) -> None:
    """Add uniform noise to joint positions and velocities for the reset envs.

    Reads the current state (which the parent ``_reset_idx`` just restored
    to initial), perturbs the selected joint subset, and writes back via
    ``handler.set_states``. This matches mjlab's
    ``mdp.reset_joints_by_offset`` exactly.

    Args:
        env: The env.
        env_ids: Indices of envs to perturb. Shape ``(K,)``.
        position_range: ``(low, high)`` uniform offset for joint positions.
        velocity_range: ``(low, high)`` uniform offset for joint velocities.
        asset_cfg: Selects the entity + joint subset.
    """
    if env_ids.numel() == 0:
        return

    states = env.handler.get_states(mode="tensor")
    state = states.robots[asset_cfg.name]
    joint_ids = resolve_joint_ids(env, asset_cfg)
    n_envs_reset = env_ids.numel()
    n_joints_sel = joint_ids.numel()

    pos_lo, pos_hi = position_range
    vel_lo, vel_hi = velocity_range
    pos_offsets = torch.empty((n_envs_reset, n_joints_sel), device=env.device).uniform_(pos_lo, pos_hi)
    vel_offsets = torch.empty((n_envs_reset, n_joints_sel), device=env.device).uniform_(vel_lo, vel_hi)

    # In-place edit of the cached TensorState — handler will copy on set_states.
    state.joint_pos[env_ids[:, None], joint_ids[None, :]] += pos_offsets
    state.joint_vel[env_ids[:, None], joint_ids[None, :]] += vel_offsets

    env.handler.set_states(states, env_ids=env_ids.tolist())
