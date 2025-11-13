from __future__ import annotations
from typing import Sequence
import torch
from roboverse_pack.tasks.unitree_rl.base.types import EnvTypes


def lin_vel_cmd_levels(
    env: EnvTypes,
    env_ids: list[int] | torch.Tensor,
    reward_term_name: str = "track_lin_vel_xy",
) -> torch.Tensor:
    if env.common_step_counter % env.max_episode_steps == 0:
        command_term = env.commands_manager
        ranges = command_term.ranges
        limit_ranges = command_term.limit_ranges

        reward_term_scales = env.reward_scales[reward_term_name][0] / env.step_dt
        reward = (
            torch.mean(env.episode_rewards[reward_term_name][env_ids])
            / env.cfg.episode_length_s
        )

        if reward > reward_term_scales * 0.8:
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            ranges.lin_vel_x = torch.clamp(
                torch.tensor(ranges.lin_vel_x, device=env.device) + delta_command,
                limit_ranges.lin_vel_x[0],
                limit_ranges.lin_vel_x[1],
            ).tolist()
            ranges.lin_vel_y = torch.clamp(
                torch.tensor(ranges.lin_vel_y, device=env.device) + delta_command,
                limit_ranges.lin_vel_y[0],
                limit_ranges.lin_vel_y[1],
            ).tolist()

    return torch.tensor(env.commands_manager.ranges.lin_vel_x[1], device=env.device)


def terrain_levels_vel(env: EnvTypes, env_ids: Sequence[int]) -> torch.Tensor:
    """Curriculum based on the distance the robot walked when commanded to move at a desired velocity.

    This term is used to increase the difficulty of the terrain when the robot walks far enough and decrease the
    difficulty when the robot walks less than half of the distance required by the commanded velocity.

    .. note::
        It is only possible to use this term with the terrain type ``generator``. For further information
        on different terrain types, check the :class:`isaaclab.terrains.TerrainImporter` class.

    Returns:
        The mean terrain level for the given environment ids.
    """
    # extract the used quantities (to enable type-hinting)
    env_states = env.get_states()
    base = env_states.robots[env.name]
    terrain = env.handler.terrain
    command = env.commands_manager.value
    # compute the distance the robot walked
    distance = torch.norm(
        base.root_state[env_ids, :2] - env.handler.scene.env_origins[env_ids, :2], dim=1
    )
    # robots that walked far enough progress to harder terrains
    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
    # robots that walked less than half of their required distance go to simpler terrains
    move_down = (
        distance
        < torch.norm(command[env_ids, :2], dim=1)
        * (env.max_episode_steps * env.step_dt)
        * 0.5
    )
    move_down *= ~move_up
    # update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down)
    # return the mean terrain level
    return torch.mean(terrain.terrain_levels.float())
