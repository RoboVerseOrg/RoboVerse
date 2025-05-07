"""Highbar task for humanoid robots.

TODO: to be implemented
"""

from __future__ import annotations

import torch

from metasim.cfg.checkers import _HighbarChecker
from metasim.cfg.objects import RigidObjCfg
from metasim.constants import PhysicStateType
from metasim.types import EnvState
from metasim.utils import configclass

from .base_cfg import HumanoidBaseReward, HumanoidTaskCfg


class HighbarReward(HumanoidBaseReward):
    """Reward function for the highbar task."""

    def __init__(self, robot_name="h1"):
        """Initialize the highbar reward."""
        super().__init__(robot_name)

    def __call__(self, states: list[EnvState]) -> torch.FloatTensor:
        """Compute the highbar reward."""
        return torch.zeros(len(states))


@configclass
class HighbarCfg(HumanoidTaskCfg):
    """Highbar task for humanoid robots."""

    episode_length = 1000
    objects = [
        RigidObjCfg(
            name="highbar",
            mjcf_path="roboverse_data/assets/humanoidbench/highbar/highbar/mjcf/highbar.xml",
            physics=PhysicStateType.GEOM,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/humanoidbench/highbar/v2/initial_state_v2.json"
    checker = _HighbarChecker()
    reward_weights = [1.0]
    reward_functions = [HighbarReward]
