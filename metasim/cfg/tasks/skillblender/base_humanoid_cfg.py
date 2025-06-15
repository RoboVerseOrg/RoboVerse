"""Base class for legged-gym style humanoid tasks."""

from __future__ import annotations

import torch

from metasim.cfg.tasks.skillblender.base_legged_cfg import BaseLeggedTaskCfg
from metasim.utils import configclass


@configclass
class HumanoidExtraCfg:
    """
    An Extension of cfg.

    Attributes:
    delay: delay in seconds
    freq: frequency for controlling sample waypoint
    resample_on_env_reset: resample waypoints on env reset
    """

    delay = 0.0  # delay in seconds
    freq = 10  # frequency for controlling sample waypoint
    resample_on_env_reset: bool = True


@configclass
class BaseHumanoidCfg(BaseLeggedTaskCfg):
    """
    An Extension of BaseLeggedTaskCfg for humanoid tasks.

    elbow_indices: indices of the elbows joints
    contact_indices: indices of the contact joints
    """

    task_name: str = "skillblender_humanoid_task"
    humand: HumanoidExtraCfg = HumanoidExtraCfg()

    elbow_indices: torch.Tensor | None = None
    contact_indices: torch.Tensor | None = None
    traj_filepath: str = "roboverse_data/trajs/skillblender/initial_state_v2.json"

    class human:
        delay = 0.0  # delay in seconds
        freq = 10
        resample_on_env_reset = True
        filename = "CMU.npy"
        rng = None
