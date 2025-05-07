"""Sub-module containing the base task configuration."""

from __future__ import annotations

from dataclasses import MISSING

import gymnasium as gym
import torch

from metasim.cfg.checkers import BaseChecker
from metasim.cfg.objects import BaseObjCfg
from metasim.cfg.simulator_params import SimParamCfg
from metasim.constants import BenchmarkType, TaskType
from metasim.types import EnvState
from metasim.utils import configclass


@configclass
class BaseTaskCfg:
    """Base task configuration.

    Attributes:
        decimation: The decimation factor for the task.
        episode_length: The length of the episode.
        objects: The list of object configurations.
        traj_filepath: The file path to the trajectory.
        source_benchmark: The source benchmark.
        task_type: The type of the task.
        checker: The checker for the task.
        can_tabletop: Whether the task can be tabletop.
        reward_functions: The list of reward functions.
        reward_weights: The list of reward weights.
        sim_params: The simulation params.
    """

    decimation: int = 3
    episode_length: int = MISSING
    objects: list[BaseObjCfg] = MISSING
    traj_filepath: str = MISSING
    source_benchmark: BenchmarkType = MISSING
    task_type: TaskType = MISSING
    checker: BaseChecker = BaseChecker()
    can_tabletop: bool = False
    reward_functions: list[callable[[list[EnvState], str | None], torch.FloatTensor]] = MISSING
    reward_weights: list[float] = MISSING
    sim_params: SimParamCfg = SimParamCfg()
    control_type: str = None # TODO provide a basic control type
    action_scale: float = 1.0


@configclass
class BaseRLTaskCfg(BaseTaskCfg):
    """Base RL task configuration."""

    # action_space: gym.spaces.Space, can be inferenced from robot (joint_limits)
    observation_space: gym.spaces.Space = MISSING
    observation_function: callable[[list[EnvState]], torch.FloatTensor] = MISSING  # [dummy_obs]

    reward_range: tuple[float, float] = (-float("inf"), float("inf"))
