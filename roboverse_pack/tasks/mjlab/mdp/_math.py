# Copyright (c) mjlab contributors
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from mjlab (https://github.com/mujocolab/mjlab).
# Changes: quaternion / frame-transform helpers reimplemented in torch against RoboVerse's `TensorState` layout;
#   they reproduce the body-frame quantities (base lin/ang velocity, projected gravity, incl. mjlab's normalized
#   gravity vector) that mjlab's `Entity` computes internally.
# Full license: roboverse_pack/tasks/mjlab/LICENSE
"""Quaternion / frame-transform helpers used by velocity MDP rewards.

Mjlab's reward functions expect body-frame quantities (e.g.
``asset.data.root_link_lin_vel_b``). RoboVerse's ``TensorState.root_state``
stores world-frame linear/angular velocity at ``[:, 7:10]`` / ``[:, 10:13]``
(quaternion at ``[:, 3:7]`` in xyzw order); these helpers do the
world-frame → body-frame transform that mjlab's Entity does internally.

For scene-MJCF tasks where the asset isn't a RoboVerse RobotCfg, fall
back to reading from ``physics.data.qpos`` / ``qvel`` directly (MuJoCo
free-joint convention: qpos[3:7] is wxyz, qvel[0:3] is world-frame
linear vel, qvel[3:6] is body-frame angular vel — note the body-frame
default for ang vel, unlike lin vel).
"""

from __future__ import annotations

import numpy as np
import torch

# mjlab uses the *normalized* gravity direction (entity.py: gravity_vec_w =
# [0, 0, -1.0]) for projected_gravity_b — NOT the 9.81 m/s² magnitude. The
# `upright` reward and projected_gravity obs both consume this, so matching
# mjlab's unit vector is required for 1:1 (a 9.81× scale silently saturates
# the upright reward to ~0 on any tilt).
GRAVITY_W: tuple[float, float, float] = (0.0, 0.0, -1.0)


def quat_apply_inverse_xyzw(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate ``v`` from world frame into body frame using the inverse of ``q``.

    Args:
        q: ``(..., 4)`` quaternion in **xyzw** order.
        v: ``(..., 3)`` world-frame vector.

    Returns:
        ``(..., 3)`` body-frame vector.
    """
    x, y, z, w = q.unbind(dim=-1)
    # Inverse rotation = transpose of R(q). Equivalently rotate by conjugate.
    # Standard quaternion vector rotation: v_b = q^-1 * v * q.
    # For unit q, q^-1 = (-x, -y, -z, w); applying gives:
    qvec = torch.stack([-x, -y, -z], dim=-1)  # (..., 3)
    t = 2.0 * torch.cross(qvec, v, dim=-1)
    return v + w.unsqueeze(-1) * t + torch.cross(qvec, t, dim=-1)


def quat_apply_xyzw(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate ``v`` from body frame into world frame using ``q`` (xyzw)."""
    x, y, z, w = q.unbind(dim=-1)
    qvec = torch.stack([x, y, z], dim=-1)
    t = 2.0 * torch.cross(qvec, v, dim=-1)
    return v + w.unsqueeze(-1) * t + torch.cross(qvec, t, dim=-1)


def get_base_state(env, env_states, asset_name: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(root_quat_xyzw, lin_vel_w, ang_vel_w)`` of the asset's base.

    Reads from ``env_states.robots[name].root_state`` (RoboVerse layout:
    pos(3)+quat_xyzw(4)+lin_vel(3)+ang_vel(3)) when the asset is a
    registered robot; falls back to ``physics.data.qpos[3:7] (wxyz)`` +
    ``qvel[:6]`` for scene-MJCF assets.

    Note: MuJoCo qvel for a free joint stores **body-frame** angular vel
    in qvel[3:6], not world. We rotate it to world here so caller
    consistently gets world-frame vels.
    """
    if asset_name in env_states.robots:
        rs = env_states.robots[asset_name].root_state
        # MetaSim TensorState.root_state stores the quaternion in **wxyz**
        # order (same as the MuJoCo free-joint qpos convention used in the
        # scene-MJCF fallback below). The body-frame helpers here expect
        # xyzw, so reorder. (Reading it as xyzw directly silently applies a
        # 180°-about-x rotation — identity wxyz [1,0,0,0] becomes xyzw
        # [1,0,0,0] = 180° about x — which flipped projected_gravity to
        # [0,0,+1] and corrupted base_lin/ang_vel on every Newton run.)
        q_wxyz = rs[:, 3:7]
        quat_xyzw = torch.cat([q_wxyz[:, 1:4], q_wxyz[:, 0:1]], dim=-1)
        lin_w = rs[:, 7:10]
        ang_w = rs[:, 10:13]
        return quat_xyzw, lin_w, ang_w
    # Scene-MJCF fallback (num_envs=1 for now).
    qpos = np.asarray(env.handler.physics.data.qpos, dtype=np.float32)
    qvel = np.asarray(env.handler.physics.data.qvel, dtype=np.float32)
    # MuJoCo free joint: qpos[3:7] = quat in wxyz; convert to xyzw.
    w, x, y, z = float(qpos[3]), float(qpos[4]), float(qpos[5]), float(qpos[6])
    quat_xyzw = torch.tensor([[x, y, z, w]], device=env.device, dtype=torch.float32)
    lin_w = torch.tensor([[float(qvel[0]), float(qvel[1]), float(qvel[2])]], device=env.device, dtype=torch.float32)
    # MuJoCo free joint: qvel[3:6] = body-frame angular vel; rotate to world.
    ang_b = torch.tensor([[float(qvel[3]), float(qvel[4]), float(qvel[5])]], device=env.device, dtype=torch.float32)
    ang_w = quat_apply_xyzw(quat_xyzw, ang_b)
    # Broadcast to num_envs (scene-MJCF is single env but keep generic).
    return (
        quat_xyzw.expand(env.num_envs, -1),
        lin_w.expand(env.num_envs, -1),
        ang_w.expand(env.num_envs, -1),
    )


def base_lin_vel_b(env, env_states, asset_name: str) -> torch.Tensor:
    """Body-frame linear velocity of the asset's base. Shape ``(num_envs, 3)``."""
    quat, lin_w, _ = get_base_state(env, env_states, asset_name)
    return quat_apply_inverse_xyzw(quat, lin_w)


def base_ang_vel_b(env, env_states, asset_name: str) -> torch.Tensor:
    """Body-frame angular velocity of the asset's base. Shape ``(num_envs, 3)``."""
    quat, _, ang_w = get_base_state(env, env_states, asset_name)
    return quat_apply_inverse_xyzw(quat, ang_w)


def projected_gravity_b(env, env_states, asset_name: str) -> torch.Tensor:
    """Gravity vector in body frame. Shape ``(num_envs, 3)``."""
    quat, _, _ = get_base_state(env, env_states, asset_name)
    gravity_w = torch.tensor(GRAVITY_W, device=env.device, dtype=torch.float32).expand(env.num_envs, -1)
    return quat_apply_inverse_xyzw(quat, gravity_w)
