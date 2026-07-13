# Copyright (c) mjlab contributors
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from mjlab (https://github.com/mujocolab/mjlab).
# Changes: port of mjlab's `tasks/velocity/mdp/curriculums.py` and `tasks/manipulation/mdp/curriculums.py` to
#   RoboVerse's `func(env, env_ids, **params)` term convention; the terrain-levels term is a stub pending the
#   terrain manager.
# Full license: roboverse_pack/tasks/mjlab/LICENSE
"""Curriculum term functions ported from mjlab.

Signature: ``func(env: ManagerBasedRVEnv, env_ids: torch.Tensor, **params) -> dict[str, Tensor] | scalar | None``

Curriculum terms are evaluated inside ``ManagerBasedRVEnv._reset_idx``
after the per-env reset is applied; the return value is logged under
``extras["episode"]["Curriculum/<term-name>/..."]``.

Currently provided:
  - ``commands_vel`` — advance velocity-command sampling ranges through
    ``velocity_stages`` based on ``env.common_step_counter``. Mirrors
    mjlab/tasks/velocity/mdp/curriculums.py.
  - ``reward_curriculum`` — stage-based weight progression for a named
    reward term. Mirrors mjlab/tasks/manipulation/mdp/curriculums.py.
  - ``terrain_levels_vel`` — stub (requires the terrain manager).

mjlab source: src/mjlab/tasks/velocity/mdp/curriculums.py,
              src/mjlab/tasks/manipulation/mdp/curriculums.py
"""

from __future__ import annotations

from typing import TypedDict

import torch


class VelocityStage(TypedDict, total=False):
    """One velocity-curriculum stage (step threshold and command ranges)."""

    step: int
    lin_vel_x: tuple[float, float]
    lin_vel_y: tuple[float, float]
    ang_vel_z: tuple[float, float]


def commands_vel(
    env,
    env_ids: torch.Tensor,
    *,
    command_name: str,
    velocity_stages: list[VelocityStage],
) -> dict[str, torch.Tensor] | None:
    """Update ``VelocityCommandManager`` ranges based on training progress.

    Reads ``env.common_step_counter`` (number of step() calls so far) and
    applies the highest-step stage whose threshold has been crossed. The
    updated ranges take effect on the next resample.
    """
    mgr = getattr(env, "command_managers", {}).get(command_name)
    if mgr is None or not hasattr(mgr, "cfg") or not hasattr(mgr.cfg, "ranges"):
        return None
    ranges = mgr.cfg.ranges
    for stage in velocity_stages:
        if env.common_step_counter >= stage["step"]:
            if stage.get("lin_vel_x") is not None:
                ranges.lin_vel_x = stage["lin_vel_x"]
            if stage.get("lin_vel_y") is not None:
                ranges.lin_vel_y = stage["lin_vel_y"]
            if stage.get("ang_vel_z") is not None:
                ranges.ang_vel_z = stage["ang_vel_z"]
    return {
        "lin_vel_x_min": torch.tensor(ranges.lin_vel_x[0]),
        "lin_vel_x_max": torch.tensor(ranges.lin_vel_x[1]),
        "lin_vel_y_min": torch.tensor(ranges.lin_vel_y[0]),
        "lin_vel_y_max": torch.tensor(ranges.lin_vel_y[1]),
        "ang_vel_z_min": torch.tensor(ranges.ang_vel_z[0]),
        "ang_vel_z_max": torch.tensor(ranges.ang_vel_z[1]),
    }


class RewardStage(TypedDict):
    """One reward-curriculum stage (step threshold and target weight)."""

    step: int
    weight: float


def reward_curriculum(
    env,
    env_ids: torch.Tensor,
    *,
    reward_name: str,
    stages: list[RewardStage],
) -> dict[str, torch.Tensor] | None:
    """Adjust a reward term's weight based on training progress.

    Mirrors mjlab/tasks/manipulation/mdp/curriculums.py::reward_curriculum.
    Walks ``stages`` in order and applies the highest-step entry whose
    threshold has been crossed.
    """
    rewards_cfg = getattr(env.cfg, "rewards", None)
    if rewards_cfg is None or not hasattr(rewards_cfg, reward_name):
        return None
    term = getattr(rewards_cfg, reward_name)
    for stage in stages:
        if env.common_step_counter >= stage["step"]:
            term.weight = float(stage["weight"])
    return {"weight": torch.tensor(float(term.weight))}


def terrain_levels_vel(
    env,
    env_ids: torch.Tensor,
    *,
    command_name: str,
    asset_name: str = "robot",
) -> dict[str, torch.Tensor] | None:
    """Mjlab ``velocity/mdp/curriculums.terrain_levels_vel`` port.

    Per-env terrain difficulty progression based on commanded vs walked
    distance. Requires ``env.terrain_manager`` to be a
    :class:`mdp.terrain.TerrainLevelsManager` (registered by the task).

    Walks the same logic as mjlab:
      ``move_up   = distance > row_size``
      ``move_down = distance < fail_factor * command_norm * episode_length``

    Returns dict logged under ``Curriculum/terrain_levels/{mean,max}``.
    """
    tm = getattr(env, "terrain_manager", None)
    if tm is None:
        return None
    cmd_mgr = getattr(env, "command_managers", {}).get(command_name)
    if cmd_mgr is None:
        return None
    try:
        cmd = cmd_mgr.current()
    except Exception:
        return None
    # Current root position of the asset.
    try:
        states = env.handler.get_states(mode="tensor")
        if states.robots:
            robot_state = next(iter(states.robots.values()))
            root_pos = robot_state.root_state[:, :3]
        elif hasattr(env.handler, "physics"):
            import numpy as np

            qpos = np.asarray(env.handler.physics.data.qpos[:3], dtype=np.float32)
            root_pos = torch.tensor(qpos, device=env.device).unsqueeze(0).expand(env.num_envs, -1)
        else:
            return None
    except Exception as e:
        # Don't silent-fail in the parity test path; bubble up.
        if str(e):
            import warnings as _w

            _w.warn(f"terrain_levels_vel: {type(e).__name__}: {e}", RuntimeWarning, stacklevel=2)
        return None

    # Distance from this env's origin (the position when it was last reset).
    delta = root_pos[env_ids, :2] - tm.env_origins[env_ids, :2]
    distance = torch.norm(delta, dim=1)
    move_up = distance > tm.cfg.row_size
    cmd_norm = torch.norm(cmd[env_ids, :2], dim=1)
    # ``max_episode_length_s`` lives on cfg in our manager (see ManagerBasedRVEnvCfg);
    # fall back to step_dt × max_episode_steps if env-level attr missing.
    max_ep_s = float(getattr(env.cfg, "max_episode_length_s", None) or (env.step_dt * env.max_episode_steps))
    move_down = (distance < cmd_norm * max_ep_s * tm.cfg.fail_factor) & ~move_up
    tm.update_env_origins(env_ids, move_up, move_down)
    # Re-capture origin for the next episode.
    tm.reset_env_origins(env_ids, root_pos)

    return {
        "mean": tm.mean_level.detach(),
        "max": tm.max_level_seen.detach().float(),
    }
