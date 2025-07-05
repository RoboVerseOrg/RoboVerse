"""Walking on slide task for humanoid robots."""

from __future__ import annotations

import torch

from metasim.cfg.checkers import _SlideChecker
from metasim.cfg.objects import RigidObjCfg
from metasim.constants import PhysicStateType
from metasim.types import EnvState
from metasim.utils import configclass, humanoid_reward_util
from metasim.utils.humanoid_robot_util import (
    actuator_forces,
    head_height,
    left_foot_height,
    right_foot_height,
    robot_velocity,
    torso_upright,
)

from .base_cfg import (
    HumanoidBaseReward,
    HumanoidTaskCfg,
)


class SlideReward(HumanoidBaseReward):
    """Reward function for the slide task."""

    def __init__(self, robot_name="h1"):
        """Initialize the slide reward."""
        super().__init__(robot_name)

    def __call__(self, states: list[EnvState]) -> torch.FloatTensor:
        """Compute the slide reward."""
        results = []
        for state in states:
            # Reward for controlled movement
            small_control = humanoid_reward_util.tolerance(
                actuator_forces(state, self._robot_name),
                margin=10,
                value_at_margin=0,
                sigmoid="quadratic",
            ).mean()
            small_control = (4 + small_control) / 5

            # Reward for sliding motion - encourage horizontal velocity
            root_x_speed = humanoid_reward_util.tolerance(
                robot_velocity(state, self._robot_name)[0],  # x-axis velocity
                bounds=(1.0, float("inf")),  # Adjust velocity threshold as needed
                margin=1.0,
                value_at_margin=0,
                sigmoid="linear",
            )

            # Reward for maintaining upright posture
            upright = humanoid_reward_util.tolerance(
                torso_upright(state),
                bounds=(0.5, 1.0),
                sigmoid="linear",
                margin=1.9,
                value_at_margin=0,
            )

            # Left foot vertical distance
            head_z = head_height(state)
            left_foot_z = left_foot_height(state)
            right_foot_z = right_foot_height(state)
            vertical_foot_left = humanoid_reward_util.tolerance(
                head_z - left_foot_z,
                bounds=(1.2, float("inf")),
                margin=0.45,
            )
            vertical_foot_right = humanoid_reward_util.tolerance(
                head_z - right_foot_z,
                bounds=(1.2, float("inf")),
                margin=0.45,
            )

            # Combine rewards
            reward = small_control * root_x_speed * upright * vertical_foot_left * vertical_foot_right

            # log.info(f"root_x_speed: {root_x_speed}, upright: {upright}, vertical_foot_left: {vertical_foot_left}, vertical_foot_right: {vertical_foot_right}")
            results.append(reward)
        return torch.tensor(results)


@configclass
class SlideCfg(HumanoidTaskCfg):
    """Slide task for humanoid robots."""

    episode_length = 1000
    objects = [
        RigidObjCfg(
            name="slide",
            mjcf_path="roboverse_data/assets/humanoidbench/slide/floor/mjcf/floor.xml",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/humanoidbench/slide/v2/initial_state_v2.json"
    checker = _SlideChecker()
    reward_weights = [1.0]
    reward_functions = [SlideReward]
