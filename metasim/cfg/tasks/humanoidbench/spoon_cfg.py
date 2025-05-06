"""Spoon in cup task for humanoid robots.

TODO: to be implemented
"""

from __future__ import annotations

import torch

from metasim.cfg.checkers import _SpoonChecker
from metasim.cfg.objects import RigidObjCfg
from metasim.constants import PhysicStateType
from metasim.types import EnvState
from metasim.utils import configclass

from .base_cfg import HumanoidBaseReward, HumanoidTaskCfg


class SpoonReward(HumanoidBaseReward):
    """Reward function for the spoon task."""

    def __init__(self, robot_name="h1"):
        """Initialize the spoon reward."""
        super().__init__(robot_name)

    def __call__(self, states: list[EnvState]) -> torch.FloatTensor:
        """Compute the spoon reward."""
        return torch.zeros(len(states))


@configclass
class SpoonCfg(HumanoidTaskCfg):
    """Spoon in cup task for humanoid robots."""

    episode_length = 1000
    objects = [
        RigidObjCfg(
            name="spoon",
            mjcf_path="roboverse_data/assets/humanoidbench/spoon/spoon/mjcf/spoon.xml",
            physics=PhysicStateType.GEOM,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/humanoidbench/spoon/v2/initial_state_v2.json"
    checker = _SpoonChecker()
    reward_weights = [1.0]
    reward_functions = [SpoonReward]
