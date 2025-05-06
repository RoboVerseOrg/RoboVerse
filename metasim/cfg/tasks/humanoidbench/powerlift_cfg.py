"""Powerlift task for humanoid robots.

TODO: to be implemented
"""

from __future__ import annotations

import torch

from metasim.cfg.checkers import _PowerliftChecker
from metasim.cfg.objects import RigidObjCfg
from metasim.constants import PhysicStateType
from metasim.types import EnvState
from metasim.utils import configclass
from metasim.utils.humanoid_reward_util import tolerance

from .base_cfg import HumanoidBaseReward, HumanoidTaskCfg, StableReward


class PowerliftReward(HumanoidBaseReward):
    """Reward function for the powerlift task."""

    def __init__(self, robot_name="h1"):
        """Initialize the powerlift reward."""
        super().__init__(robot_name)

    def __call__(self, states: list[EnvState]) -> torch.FloatTensor:
        """Compute the powerlift reward."""
        rewards = torch.zeros(len(states))

        for i, state in enumerate(states):
            # Retrieve dumbbell and target location
            dumbbell_pos = state["metasim_body_dumbbell/dumbbell"]["pos"]

            reward_height = tolerance(dumbbell_pos[2], bounds=(1.9, 2.1), margin=2)

            # The robot's stability reward
            stable = StableReward(robot_name=self.robot_name)(states)

            # Calculate the total reward
            reward = 0.2 * stable + 0.8 * reward_height

            rewards[i] = reward

        return rewards


@configclass
class PowerliftCfg(HumanoidTaskCfg):
    """Powerlift task for humanoid robots."""

    episode_length = 1000
    objects = [
        RigidObjCfg(
            name="powerlift",
            mjcf_path="roboverse_data/assets/humanoidbench/powerlift/dumbbell/mjcf/dumbbell.xml",
            physics=PhysicStateType.GEOM,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/humanoidbench/powerlift/v2/initial_state_v2.json"
    checker = _PowerliftChecker()
    reward_weights = [1.0]
    reward_functions = [PowerliftReward]
