# Copyright (c) mjlab contributors
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from mjlab (https://github.com/mujocolab/mjlab).
# Changes: termination terms ported from mjlab to RoboVerse's `func(env, env_states, **params) -> BoolTensor`
#   convention, reading `TensorState` / MuJoCo qpos instead of mjlab's scene entities.
# Full license: roboverse_pack/tasks/mjlab/LICENSE
"""Termination term functions ported from mjlab.

Signature: ``func(env: ManagerBasedRVEnv, env_states: TensorState, **params) -> torch.BoolTensor``
Returns shape ``(num_envs,)``. The manager loop routes terms with
``time_out=True`` into ``reset_time_outs`` (truncation), the rest into
``reset_terminated`` (failure); the union triggers env reset.
"""

from __future__ import annotations

import torch

from .scene_entity import SceneEntityCfg


def time_out(env, env_states) -> torch.Tensor:
    """Episode reached ``max_episode_steps``. Wire with ``time_out=True``."""
    return env._episode_steps >= env.max_episode_steps


def bad_orientation(env, env_states, asset_cfg: SceneEntityCfg, limit_angle: float) -> torch.Tensor:
    """Robot's up-axis tilted more than ``limit_angle`` radians from vertical.

    Reads the root quaternion of the asset and compares its body-frame z-axis
    with world-up. Falls back to ``physics.data.qpos[3:7]`` (MuJoCo
    free-joint convention: wxyz) when the asset isn't a registered robot.
    """
    if asset_cfg.name in env_states.robots:
        state = env_states.robots[asset_cfg.name]
        # root_state layout in RoboVerse: pos(3) + quat_wxyz(4) + lin_vel(3) + ang_vel(3).
        # All handlers (mujoco/mjx/newton/sapien) store quat wxyz at root_state[3:7].
        quat = state.root_state[:, 3:7]
        w, x, y, z = quat.unbind(dim=-1)
    else:
        # Scene-MJCF fallback. MuJoCo free joint stores quat as wxyz at qpos[3:7].
        import numpy as np

        q = np.asarray(env.handler.physics.data.qpos[3:7], dtype=np.float32)
        w_, x_, y_, z_ = float(q[0]), float(q[1]), float(q[2]), float(q[3])
        # Broadcast over num_envs (single-env scene-MJCF path).
        w = torch.full((env.num_envs,), w_, device=env.device)
        x = torch.full((env.num_envs,), x_, device=env.device)
        y = torch.full((env.num_envs,), y_, device=env.device)
        z = torch.full((env.num_envs,), z_, device=env.device)
    # For unit quaternion (x, y, z, w), body-z·world-z = 1 - 2(x^2 + y^2).
    cos_tilt = 1.0 - 2.0 * (x * x + y * y)
    tilt = torch.acos(torch.clamp(cos_tilt, -1.0, 1.0))
    return tilt > limit_angle
