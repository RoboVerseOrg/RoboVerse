"""Crawl task for humanoid robots."""

from __future__ import annotations

import torch

from metasim.cfg.checkers import _CrawlChecker
from metasim.cfg.objects import RigidObjCfg
from metasim.constants import PhysicStateType
from metasim.types import EnvState
from metasim.utils import configclass, humanoid_reward_util
from metasim.utils.humanoid_robot_util import (
    actuator_forces,
    robot_position,
    robot_rotation,
    robot_velocity,
)

from .base_cfg import HumanoidBaseReward, HumanoidTaskCfg


class CrawlReward(HumanoidBaseReward):
    """Reward function for the crawl task."""

    def __init__(self, robot_name="h1"):
        """Initialize the crawl reward."""
        super().__init__(robot_name)

        # Define the expected quaternion for crawling, normalized
        quat_raw = torch.tensor([0.75, 0, 0.65, 0])
        self.quat_crawl = quat_raw / torch.norm(quat_raw)

    def __call__(self, states: list[EnvState]) -> torch.FloatTensor:
        """Compute the crawl reward."""
        results = []
        for state in states:
            # Get robot position and IMU data
            robot_pos = robot_position(state, self.robot_name)
            imu_pos = robot_position(state, self.robot_name)

            # Get pelvis quaternion for orientation calculation
            pelvis_quat = robot_rotation(state, self.robot_name)

            # Get velocity for speed calculation
            velocity = robot_velocity(state, self.robot_name)
            vx = velocity[0]

            # heightcrawl = height((0.6,1),1)
            # Use robot height (z position) with tolerance between 0.6 and 1.0
            height_crawl = humanoid_reward_util.tolerance(robot_pos[2], bounds=(0.6, 1.0), margin=1.0)

            # heightIMU = tol(zIMU,(0.6,1),1)
            height_imu = humanoid_reward_util.tolerance(imu_pos[2], bounds=(0.6, 1.0), margin=1.0)

            # orientation = tol(‖quatpelvis−quatcrawl‖,(0,0),1)
            # Calculate quaternion distance and use tolerance function
            quat_diff = torch.norm(pelvis_quat.clone().detach() - self.quat_crawl)
            orientation = humanoid_reward_util.tolerance(quat_diff, bounds=(0, 0), margin=1.0)

            # tunnel = tol(yIMU,(−1,1),0)
            # Reward when IMU's y-coordinate is within the tunnel
            tunnel = humanoid_reward_util.tolerance(imu_pos[1], bounds=(-1, 1), margin=0.0)

            # speed = tol(vx,(1,+∞),1)
            # Reward for forward velocity greater than 1
            speed = humanoid_reward_util.tolerance(vx, bounds=(1, float("inf")), margin=1.0)

            # Calculate the small control factor (e) similar to sit task
            small_control = humanoid_reward_util.tolerance(
                actuator_forces(state, self.robot_name),
                margin=10,
                value_at_margin=0,
                sigmoid="quadratic",
            ).mean()
            small_control = (4 + small_control) / 5

            # R = tunnel × (0.1·e + 0.25·min(heightcrawl,heightIMU) + 0.25·orientation + 0.4·speed)
            min_height = min(height_crawl, height_imu)
            reward = tunnel * (0.1 * small_control + 0.25 * min_height + 0.25 * orientation + 0.4 * speed)

            results.append(reward)

        return torch.tensor(results)


@configclass
class CrawlCfg(HumanoidTaskCfg):
    """Crawl task for humanoid robots."""

    episode_length = 1000
    objects = [
        RigidObjCfg(
            name="tunnel",
            mjcf_path="roboverse_data/assets/humanoidbench/crawl/tunnel/mjcf/tunnel.xml",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/humanoidbench/crawl/v2/initial_state_v2.json"
    checker = _CrawlChecker()
    reward_weights = [1.0]
    reward_functions = [CrawlReward]
