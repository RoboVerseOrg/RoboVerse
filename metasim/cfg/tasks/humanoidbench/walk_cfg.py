"""Walking task for humanoid robots."""

from metasim.cfg.checkers import _WalkChecker
from metasim.utils import configclass

from .base_cfg import BaseLocomotionReward, HumanoidTaskCfg


class WalkReward(BaseLocomotionReward):
    """Reward function for the walk task."""

    _move_speed = 1
    success_bar = 700


@configclass
class WalkCfg(HumanoidTaskCfg):
    """Walking task for humanoid robots."""

    episode_length = 1000
    # traj_filepath = "roboverse_data/trajs/humanoidbench/walk/v2/h1_v2.pkl"
    traj_filepath = "roboverse_data/trajs/humanoidbench/walk/v2/initial_state_v2.json"
    checker = _WalkChecker()
    reward_functions = [WalkReward]
    reward_weights = [1.0]
