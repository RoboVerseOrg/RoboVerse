"""Base class for humanoid tasks."""

from __future__ import annotations

import logging

import numpy as np
import torch
from loguru import logger as log
from rich.logging import RichHandler

from metasim.cfg.tasks.base_task_cfg import BaseRLTaskCfg
from metasim.constants import BenchmarkType, TaskType
from metasim.types import EnvState
from metasim.utils import configclass, humanoid_reward_util
from metasim.utils.humanoid_robot_util import (
    actuator_forces_tensor,
    neck_height_tensor,
    robot_velocity_tensor,
    torso_upright_tensor,
)

logging.addLevelName(5, "TRACE")
log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])

########################################################
## Constants adapted from humanoid_bench/tasks/basic_locomotion_envs.py
########################################################

# Height of head above which stand reward is 1.
H1_STAND_HEAD_HEIGHT = 1.65
H1_STAND_NECK_HEIGHT = 1.41
H1_CRAWL_HEAD_HEIGHT = 0.8
G1_STAND_HEAD_HEIGHT = 1.28
G1_STAND_NECK_HEIGHT = 1.0
G1_CRAWL_HEAD_HEIGHT = 0.6


@configclass
class HumanoidTaskCfg(BaseRLTaskCfg):
    """Base class for humanoid tasks."""

    decimation: int = 1
    source_benchmark = BenchmarkType.HUMANOIDBENCH
    task_type = TaskType.LOCOMOTION
    episode_length = 800  # TODO: may change
    objects = []
    reward_weights = [1.0]

    @staticmethod
    def humanoid_obs_flatten_func(envstates: list[EnvState]) -> torch.Tensor:
        """Observation function for humanoid tasks.

        Args:
            envstates (list[EnvState]): List of environment states to process.

        Returns:
            torch.Tensor: Flattened observations for all environments.
        """
        env_obs = []
        results_state = []
        for _, object_state in sorted(envstates.objects.items()):
            results_state.append(object_state.root_state)
        for _, robot_state in sorted(envstates.robots.items()):
            results_state.append(robot_state.root_state)
            results_state.append(robot_state.joint_pos)
            results_state.append(robot_state.joint_vel)
        return torch.cat(results_state, dim=1)


class HumanoidBaseReward:
    """Base class for humanoid rewards."""

    def __init__(self, robot_name="h1"):
        """Initialize the humanoid reward."""
        self.robot_name = robot_name
        if robot_name == "h1" or robot_name == "h1_simple_hand" or robot_name == "h1_hand":
            self._stand_height = H1_STAND_HEAD_HEIGHT
            self._stand_neck_height = H1_STAND_NECK_HEIGHT
            self._crawl_height = H1_CRAWL_HEAD_HEIGHT
        elif robot_name == "g1":
            self._stand_height = G1_STAND_HEAD_HEIGHT
            self._stand_neck_height = G1_STAND_NECK_HEIGHT
            self._crawl_height = G1_CRAWL_HEAD_HEIGHT
        else:
            raise ValueError(f"Unknown robot {robot_name}")


class StableReward(HumanoidBaseReward):
    """Base class for locomotion rewards."""

    _move_speed = None
    htarget_low = np.array([-1.0, -1.0, 0.8])
    htarget_high = np.array([1000.0, 1.0, 2.0])
    success_bar = None

    def __init__(self, robot_name="h1"):
        """Initialize the locomotion reward."""
        super().__init__(robot_name)

    def __call__(self, states: list[EnvState]) -> torch.FloatTensor:
        """Compute the locomotion reward."""
        ret_rewards = []
        standing = humanoid_reward_util.tolerance_tensor(
            neck_height_tensor(states, self.robot_name),
            bounds=(self._stand_neck_height, float("inf")),
            margin=self._stand_neck_height / 4,
        )
        upright = humanoid_reward_util.tolerance_tensor(
            torso_upright_tensor(states, self.robot_name),
            bounds=(0.9, float("inf")),
            margin=1.9,
            value_at_margin=0,
            sigmoid="linear",
        )
        stand_reward = standing * upright
        small_control = humanoid_reward_util.tolerance_tensor(
            actuator_forces_tensor(states, self.robot_name),
            margin=10,
            value_at_margin=0,
            sigmoid="quadratic",
        ).mean()
        small_control = (4 + small_control) / 5
        ret_rewards = small_control * stand_reward
        # ret_rewards.append()
        # for state in states:
        #     standing = humanoid_reward_util.tolerance(
        #         neck_height(state, self.robot_name),
        #         bounds=(self._stand_neck_height, float("inf")),
        #         margin=self._stand_neck_height / 4,
        #     )
        #     upright = humanoid_reward_util.tolerance(
        #         torso_upright(state, self.robot_name),
        #         bounds=(0.9, float("inf")),
        #         sigmoid="linear",
        #         margin=1.9,
        #         value_at_margin=0,
        #     )
        #     stand_reward = standing * upright
        #     small_control = humanoid_reward_util.tolerance(
        #         actuator_forces(state, self.robot_name),
        #         margin=10,
        #         value_at_margin=0,
        #         sigmoid="quadratic",
        #     ).mean()
        #     small_control = (4 + small_control) / 5
        #     ret_rewards.append(small_control * stand_reward)

        # ret_rewards = torch.tensor(ret_rewards)
        return ret_rewards


class BaseLocomotionReward(HumanoidBaseReward):
    """Base class for locomotion rewards."""

    _move_speed = None
    htarget_low = np.array([-1.0, -1.0, 0.8])
    htarget_high = np.array([1000.0, 1.0, 2.0])
    success_bar = None

    def __init__(self, robot_name="h1"):
        """Initialize the locomotion reward."""
        super().__init__(robot_name)

    def __call__(self, states: list[EnvState]) -> torch.FloatTensor:
        """Compute the locomotion reward."""
        moving_reward = []
        stable_rewards = StableReward(self.robot_name)(states)
        if self._move_speed == 0:
            horizontal_velocity = robot_velocity_tensor(states, self.robot_name)[[0, 1]]
            dont_move = humanoid_reward_util.tolerance_tensor(horizontal_velocity, margin=2).mean()
            moving_reward = dont_move
        else:
            com_velocity = robot_velocity_tensor(states, self.robot_name)[0]
            move = humanoid_reward_util.tolerance_tensor(
                com_velocity,
                bounds=(self._move_speed, float("inf")),
                margin=self._move_speed,
                value_at_margin=0,
                sigmoid="linear",
            )
            move = (5 * move + 1) / 6
            moving_reward = move
        # for state in states:
        #     if self._move_speed == 0:
        #         horizontal_velocity = robot_velocity(state, self.robot_name)[[0, 1]]
        #         dont_move = humanoid_reward_util.tolerance(horizontal_velocity, margin=2).mean()
        #         moving_reward.append(dont_move)
        #     else:
        #         com_velocity = robot_velocity(state, self.robot_name)[0]
        #         move = humanoid_reward_util.tolerance(
        #             com_velocity,
        #             bounds=(self._move_speed, float("inf")),
        #             margin=self._move_speed,
        #             value_at_margin=0,
        #             sigmoid="linear",
        #         )
        #         move = (5 * move + 1) / 6
        #         moving_reward.append(move)
        # moving_reward = torch.tensor(moving_reward)
        return stable_rewards * moving_reward
