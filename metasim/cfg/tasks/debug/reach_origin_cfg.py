from __future__ import annotations

import torch

from metasim.cfg.tasks.base_task_cfg import BaseTaskCfg
from metasim.constants import BenchmarkType, TaskType
from metasim.utils import configclass


def negative_distance(states, robot_name: str | None = None) -> torch.Tensor:
    ee_pos = torch.stack([state["metasim_body_panda_hand"]["pos"] for state in states])
    distances = torch.norm(ee_pos, dim=1)
    return -distances  # Negative distance as reward


@configclass
class ReachOriginCfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.DEBUG
    task_type = TaskType.TABLETOP_MANIPULATION
    can_tabletop = True
    episode_length = 100
    objects = []
    traj_filepath = "roboverse_data/trajs/debug/reach_origin/franka_v2.json"
    reward_functions = [negative_distance]
    reward_weights = [1.0]
    ## TODO: add a empty checker to suppress warning
