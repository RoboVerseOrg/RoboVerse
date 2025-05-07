"""Package task for humanoid robots."""

from __future__ import annotations

import torch

from metasim.cfg.checkers import _PackageChecker
from metasim.cfg.objects import RigidObjCfg
from metasim.constants import PhysicStateType
from metasim.types import EnvState
from metasim.utils import configclass
from metasim.utils.humanoid_robot_util import left_hand_position, right_hand_position

from .base_cfg import HumanoidBaseReward, HumanoidTaskCfg, StableReward


class PackageReward(HumanoidBaseReward):
    """Reward function for the package task."""

    def __init__(self, robot_name="h1"):
        """Initialize the package reward."""
        super().__init__(robot_name)

    def __call__(self, states: list[EnvState]) -> torch.FloatTensor:
        """Compute the package reward."""
        rewards = torch.zeros(len(states))

        for i, state in enumerate(states):
            # Retrieve package and target location
            package_pos = state["metasim_body_package/package"]["pos"]
            target_pos = state["metasim_body_target/target"]["pos"]

            # Calculate the distance between the package and the target
            package_target_dist = torch.norm(package_pos - target_pos)

            # Calculate package height reward: min(1, z_package)
            height_package = torch.min(torch.tensor(1.0), package_pos[2])

            # Determine success: The distance between the package and the target is less than 0.1.
            success = (package_target_dist < 0.1).float()

            # Get robot left and right hand positions
            left_hand_pos = left_hand_position(state)
            right_hand_pos = right_hand_position(state)

            # Calculate the distance between the hands and the package
            d_hand = torch.norm(left_hand_pos - package_pos) + torch.norm(right_hand_pos - package_pos)

            # The robot's stability reward
            stable = StableReward(robot_name=self.robot_name)(states)

            # Calculate the total reward
            reward = (-3.0 * package_target_dist) - (0.1 * d_hand) + stable + height_package + (1000.0 * success)

            rewards[i] = reward

        return rewards


@configclass
class PackageCfg(HumanoidTaskCfg):
    """Package task for humanoid robots."""

    episode_length = 1000
    objects = [
        RigidObjCfg(
            name="package",
            mjcf_path="roboverse_data/assets/humanoidbench/package/package/mjcf/package.xml",
            physics=PhysicStateType.GEOM,
        ),
        RigidObjCfg(
            name="target",
            mjcf_path="roboverse_data/assets/humanoidbench/package/target/mjcf/target.xml",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/humanoidbench/package/v2/initial_state_v2.json"
    checker = _PackageChecker()
    reward_weights = [1.0]
    reward_functions = [PackageReward]
