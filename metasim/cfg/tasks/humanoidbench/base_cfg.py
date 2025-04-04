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
    actuator_forces,
    center_of_mass_velocity,
    head_height,
    torso_upright,
)

logging.addLevelName(5, "TRACE")
log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])

########################################################
## Constants adapted from humanoid_bench/tasks/basic_locomotion_envs.py
########################################################

# Height of head above which stand reward is 1.
_H1_STAND_HEIGHT = 1.65
_H1_CRAWL_HEIGHT = 0.8
_G1_STAND_HEIGHT = 1.28
_G1_CRAWL_HEIGHT = 0.6


@configclass
class HumanoidTaskCfg(BaseRLTaskCfg):
    """Base class for humanoid tasks."""

    decimation: int = 10
    source_benchmark = BenchmarkType.HUMANOIDBENCH
    task_type = TaskType.LOCOMOTION
    episode_length = 800  # TODO: may change
    objects = []
    reward_weights = [1.0]

    @staticmethod
    def humanoid_obs_flatten_func(envstates: list[EnvState]) -> torch.Tensor:
        """Observation function for humanoid tasks."""
        env_obs = []
        for envstate in envstates:
            flattened = []
            for obj_name, obj_state in envstate.items():
                if obj_name == "h1" or obj_name == "g1" or obj_name == "h1_simple_hand" or obj_name == "h1_hand":
                    for key in ["pos", "rot", "vel", "ang_vel"]:
                        if key in obj_state and obj_state[key] is not None:
                            flattened.append(np.array(obj_state[key]).flatten())
                    if "dof_pos" in obj_state:
                        for key, value in obj_state["dof_pos"].items():
                            if isinstance(value, (np.ndarray, float)):
                                flattened.append(np.array(value).flatten())
                    if "dof_vel" in obj_state:
                        for key, value in obj_state["dof_vel"].items():
                            if isinstance(value, (np.ndarray, float)):
                                flattened.append(np.array(value).flatten())
                elif not obj_name.startswith("metasim"):
                    # flatten pos, rot, vel, ang_vel for other objects. (state-based observation)
                    for key in ["pos", "rot", "vel", "ang_vel"]:
                        if key in obj_state and obj_state[key] is not None:
                            flattened.append(obj_state[key].numpy())
            env_obs.append(np.concatenate([arr for arr in flattened]))

        env_obs = np.stack(env_obs)
        return env_obs


class StableReward:
    """Base class for locomotion rewards."""

    _move_speed = None
    htarget_low = np.array([-1.0, -1.0, 0.8])
    htarget_high = np.array([1000.0, 1.0, 2.0])
    success_bar = None

    def __init__(self, robot_name="h1"):
        """Initialize the locomotion reward."""
        self.robot_name = robot_name
        if robot_name == "h1" or robot_name == "h1_simple_hand" or robot_name == "h1_hand":
            self._stand_height = _H1_STAND_HEIGHT
            self._crawl_height = _H1_CRAWL_HEIGHT
        elif robot_name == "g1":
            self._stand_height = _G1_STAND_HEIGHT
            self._crawl_height = _G1_CRAWL_HEIGHT
        else:
            raise ValueError(f"Unknown robot {robot_name}")

    def __call__(self, states: list[EnvState]) -> torch.FloatTensor:
        """Compute the locomotion reward."""
        ret_rewards = []
        for state in states:
            standing = humanoid_reward_util.tolerance(
                head_height(state),
                bounds=(self._stand_height, float("inf")),
                margin=self._stand_height / 4,
            )
            upright = humanoid_reward_util.tolerance(
                torso_upright(state),
                bounds=(0.9, float("inf")),
                sigmoid="linear",
                margin=1.9,
                value_at_margin=0,
            )
            stand_reward = standing * upright
            small_control = humanoid_reward_util.tolerance(
                actuator_forces(state, self.robot_name),
                margin=10,
                value_at_margin=0,
                sigmoid="quadratic",
            ).mean()
            small_control = (4 + small_control) / 5
            ret_rewards.append(small_control * stand_reward)

        ret_rewards = torch.tensor(ret_rewards)
        return ret_rewards


class BaseLocomotionReward:
    """Base class for locomotion rewards."""

    _move_speed = None
    htarget_low = np.array([-1.0, -1.0, 0.8])
    htarget_high = np.array([1000.0, 1.0, 2.0])
    success_bar = None

    def __init__(self, robot_name="h1"):
        """Initialize the locomotion reward."""
        self.robot_name = robot_name
        if robot_name == "h1" or robot_name == "h1_simple_hand" or robot_name == "h1_hand":
            self._stand_height = _H1_STAND_HEIGHT
            self._crawl_height = _H1_CRAWL_HEIGHT
        elif robot_name == "g1":
            self._stand_height = _G1_STAND_HEIGHT
            self._crawl_height = _G1_CRAWL_HEIGHT
        else:
            raise ValueError(f"Unknown robot {robot_name}")

    def __call__(self, states: list[EnvState]) -> torch.FloatTensor:
        """Compute the locomotion reward."""
        moving_reward = []
        stable_rewards = StableReward(self.robot_name)(states)
        for state in states:
            if self._move_speed == 0:
                horizontal_velocity = center_of_mass_velocity(state)[[0, 1]]
                dont_move = humanoid_reward_util.tolerance(horizontal_velocity, margin=2).mean()
                moving_reward.append(dont_move)
            else:
                com_velocity = center_of_mass_velocity(state)[0]
                move = humanoid_reward_util.tolerance(
                    com_velocity,
                    bounds=(self._move_speed, float("inf")),
                    margin=self._move_speed,
                    value_at_margin=0,
                    sigmoid="linear",
                )
                move = (5 * move + 1) / 6
                moving_reward.append(move)
        moving_reward = torch.tensor(moving_reward)
        return stable_rewards * moving_reward
