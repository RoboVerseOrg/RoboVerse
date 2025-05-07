"""Stand task for humanoid robots."""

from metasim.cfg.checkers import _StandChecker
from metasim.utils import configclass

from .base_cfg import BaseLocomotionReward, HumanoidTaskCfg


class StandReward(BaseLocomotionReward):
    """Reward function for the stand task."""

    _move_speed = 0
    success_bar = 800


@configclass
class StandCfg(HumanoidTaskCfg):
    """Stand task for humanoid robots."""

    episode_length = 1000
    # traj_filepath = "roboverse_data/trajs/humanoidbench/stand/v2/h1_v2.pkl"
    traj_filepath = "roboverse_data/trajs/humanoidbench/stand/v2/initial_state_v2.json"
    checker = _StandChecker()
    reward_weights = [1.0]
    reward_functions = [StandReward]
