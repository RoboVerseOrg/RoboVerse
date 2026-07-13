# Copyright (c) The BeyondMimic authors (HybridRobotics/whole_body_tracking)
# SPDX-License-Identifier: MIT
#
# Adapted from BeyondMimic / whole_body_tracking (https://github.com/HybridRobotics/whole_body_tracking).
# Changes: upstream's event terms ported from Isaac Lab manager terms to MetaSim's TensorState-based callbacks.
# Full license: roboverse_pack/tasks/beyondmimic/LICENSE.beyondmimic

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch

from metasim.types import TensorState
from metasim.utils.math import sample_uniform
from roboverse_pack.tasks.beyondmimic.metasim.configs.cfg_randomizers import randomize_prop_by_op

if TYPE_CHECKING:
    from roboverse_pack.tasks.beyondmimic.metasim.envs.base_legged_robot import LeggedRobotTask


def randomize_joint_default_pos(  # startup
    env: LeggedRobotTask,
    env_ids: torch.Tensor | None = None,
    joint_ids: torch.Tensor | None = None,
    pos_distribution_params: tuple[float, float] | None = None,
    operation: Literal["add", "scale", "abs"] = "abs",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """Randomize the joint default positions which may be different from URDF due to calibration errors."""
    # save nominal value for export
    env.default_dof_pos_nominal = torch.clone(env.default_dof_pos_sorted[0])

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scenario.num_envs, device=env.device)

    # resolve joint indices
    if joint_ids is None:
        joint_ids = torch.arange(len(env.sorted_joint_names), device=env.device)

    if pos_distribution_params is not None:
        # pos = env.default_dof_pos.unsqueeze(0).repeat(env.scenario.num_envs, 1).to(env.device).clone()  # FIXME
        pos = env.default_dof_pos_sorted.to(env.device).clone()  # [n_envs, n_dofs]
        pos = randomize_prop_by_op(
            pos, pos_distribution_params, env_ids, joint_ids, operation=operation, distribution=distribution
        )[env_ids][:, joint_ids]

        if env_ids != slice(None) and joint_ids != slice(None):
            env_ids = env_ids[:, None]
        env.default_dof_pos_sorted[env_ids, joint_ids] = pos


def push_by_setting_velocity(
    env: LeggedRobotTask,
    env_states: TensorState,
    interval_range_s: tuple | int,
    velocity_range: dict[str, tuple[float, float]],
):
    """Randomly set robot's root velocity to simulate a push.

    The function takes a dictionary of velocity ranges for each axis and rotation. The keys of the dictionary are "x", "y", "z", "roll", "pitch", and "yaw". The values are tuples of the form (min, max).
    """
    if not hasattr(env, "push_interval"):
        env.push_interval = (
            sample_uniform(
                interval_range_s[0],
                interval_range_s[1],
                (env.num_envs, 1),
                device=env.device,
            ).flatten()
            / env.step_dt  # convert seconds to simulation steps
        ).to(torch.int)
    # TODO different from how interval events are triggered in Isaac Lab, consider adapting this
    push_env_ids = (
        torch.logical_and(env._episode_steps % env.push_interval == 0, env._episode_steps > 0)
        .nonzero(as_tuple=False)
        .flatten()
    )
    if len(push_env_ids) == 0:
        return
    ranges = torch.tensor(
        [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]], device=env.device
    )
    env_states.robots[env.name].root_state[push_env_ids, 7:13] += sample_uniform(
        ranges[:, 0], ranges[:, 1], (len(push_env_ids), 6), device=env.device
    )  # add random velocity to root's linear and angular velocities

    env.handler.set_states(env_states, push_env_ids.tolist())
