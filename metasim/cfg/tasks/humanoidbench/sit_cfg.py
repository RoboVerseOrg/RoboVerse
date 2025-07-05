"""Sit task for humanoid robots."""

from __future__ import annotations

import torch

from metasim.cfg.checkers import _SitChecker
from metasim.cfg.objects import RigidObjCfg
from metasim.constants import PhysicStateType
from metasim.types import EnvState
from metasim.utils import configclass, humanoid_reward_util
from metasim.utils.humanoid_robot_util import (
    actuator_forces,
    head_height,
    robot_position,
    robot_velocity,
    torso_upright,
)

from .base_cfg import HumanoidBaseReward, HumanoidTaskCfg


class SitReward(HumanoidBaseReward):
    """Reward function for the sit task."""

    def __init__(self, robot_name="h1"):
        """Initialize the sit reward."""
        super().__init__(robot_name)

    def __call__(self, states: list[EnvState]) -> torch.FloatTensor:
        """Compute the sit reward."""
        results = []
        for state in states:
            # Get robot position (center of mass)
            robot_pos = robot_position(state, self.robot_name)

            # Get chair position (assuming the chair object is named "sit" based on the config)
            chair_pos = state["metasim_body_chair/chair"]["pos"]

            # Get head height for posture calculation
            head_z = head_height(state)

            # Get IMU position (assuming it's at the torso)
            imu_z = state[f"metasim_site_{self.robot_name}/imu"]["pos"][2]

            # Get velocity for stillness calculation
            velocity = robot_velocity(state, self.robot_name)
            vx, vy = velocity[0], velocity[1]

            # Calculate sitting components using tolerance function
            # sittingx = tol(xrobot−xchair, (−0.19, 0.19), 0.2)
            sittingx = humanoid_reward_util.tolerance(robot_pos[0] - chair_pos[0], bounds=(-0.19, 0.19), margin=0.2)

            # sittingy = tol(yrobot−ychair, (0, 0), 0.1)
            sittingy = humanoid_reward_util.tolerance(robot_pos[1] - chair_pos[1], bounds=(0, 0), margin=0.1)

            # sittingz = tol(zrobot, (0.68, 0.72), 0.2)
            sittingz = humanoid_reward_util.tolerance(robot_pos[2], bounds=(0.68, 0.72), margin=0.2)

            # posture = tol(zhead−zIMU, (0.35, 0.45), 0.3)
            posture = humanoid_reward_util.tolerance(head_z - imu_z, bounds=(0.35, 0.45), margin=0.3)

            # stillx = tol(vx, (0, 0), 2)
            stillx = humanoid_reward_util.tolerance(vx, bounds=(0, 0), margin=2)

            # stilly = tol(vy, (0, 0), 2)
            stilly = humanoid_reward_util.tolerance(vy, bounds=(0, 0), margin=2)

            # Get upright component
            upright = humanoid_reward_util.tolerance(
                torso_upright(state),
                bounds=(0.9, 1.0),
                sigmoid="linear",
                margin=0.2,
                value_at_margin=0,
            )

            # Small control
            small_control = humanoid_reward_util.tolerance(
                actuator_forces(state, self.robot_name),
                margin=10,
                value_at_margin=0,
                sigmoid="quadratic",
            ).mean()
            small_control = (4 + small_control) / 5

            # Calculate combined reward
            # R(s,a) = ((0.5·sittingz + 0.5·sittingx×sittingy)×upright×posture)×e×mean(stillx,stilly)
            sitting_position = 0.5 * sittingz + 0.5 * sittingx * sittingy

            # Fixed: Use a simple average instead of torch.mean() on scalar values
            stillness = (stillx + stilly) / 2.0

            reward = sitting_position * upright * posture * small_control * stillness

            results.append(reward)

        return torch.tensor(results)


@configclass
class SitCfg(HumanoidTaskCfg):
    """Sit task for humanoid robots."""

    episode_length = 1000
    objects = [
        RigidObjCfg(
            name="sit",
            mjcf_path="roboverse_data/assets/humanoidbench/sit/chair/mjcf/chair.xml",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/humanoidbench/sit/v2/initial_state_v2.json"
    checker = _SitChecker()
    reward_weights = [1.0]
    reward_functions = [SitReward]
