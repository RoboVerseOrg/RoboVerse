"""Base class for legged-gym style humanoid tasks."""

from __future__ import annotations

from dataclasses import MISSING

import torch

from metasim.cfg.tasks.skillblender.base_legged_cfg import BaseLeggedTaskCfg, LeggedRobotCfgPPO
from metasim.utils import configclass


@configclass
class Algorithm(LeggedRobotCfgPPO.Algorithm):
    pass


# LeggedRobotCfgPPO.
@configclass
class BaseHumanoidCfgPPO(LeggedRobotCfgPPO):
    algorithm: Algorithm = Algorithm(
        entropy_coef=0.001, learning_rate=1e-5, num_learning_epochs=2, gamma=0.994, lam=0.9
    )


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
    human: HumanoidExtraCfg = HumanoidExtraCfg()
    elbow_indices: torch.Tensor = MISSING
    knee_indices: torch.Tensor = MISSING
    wrist_indices: torch.Tensor = MISSING
    torso_indices: torch.Tensor = MISSING
    contact_indices: torch.Tensor = MISSING

    init_states = [
        {
            "objects": {},
            "robots": {
                "h1_wrist": {
                    "pos": torch.tensor([0.0, 0.0, 1.0]),
                    "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                    "dof_pos": {
                        "left_hip_yaw": 0.0,
                        "left_hip_roll": 0.0,
                        "left_hip_pitch": -0.4,
                        "left_knee": 0.8,
                        "left_ankle": -0.4,
                        "right_hip_yaw": 0.0,
                        "right_hip_roll": 0.0,
                        "right_hip_pitch": -0.4,
                        "right_knee": 0.8,
                        "right_ankle": -0.4,
                        "torso": 0.0,
                        "left_shoulder_pitch": 0.0,
                        "left_shoulder_roll": 0.0,
                        "left_shoulder_yaw": 0.0,
                        "left_elbow": 0.0,
                        "right_shoulder_pitch": 0.0,
                        "right_shoulder_roll": 0.0,
                        "right_shoulder_yaw": 0.0,
                        "right_elbow": 0.0,
                    },
                },
                "g1": {
                    "pos": torch.tensor([0.0, 0.0, 0.735]),
                    "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                    "dof_pos": {
                        "left_hip_pitch": -0.4,
                        "left_hip_roll": 0,
                        "left_hip_yaw": 0.0,
                        "left_knee": 0.8,
                        "left_ankle_pitch": -0.4,
                        "left_ankle_roll": 0,
                        "right_hip_pitch": -0.4,
                        "right_hip_roll": 0,
                        "right_hip_yaw": 0.0,
                        "right_knee": 0.8,
                        "right_ankle_pitch": -0.4,
                        "right_ankle_roll": 0,
                        "waist_yaw": 0.0,
                        "left_shoulder_pitch": 0.0,
                        "left_shoulder_roll": 0.0,
                        "left_shoulder_yaw": 0.0,
                        "left_elbow": 0.0,
                        "right_shoulder_pitch": 0.0,
                        "right_shoulder_roll": 0.0,
                        "right_shoulder_yaw": 0.0,
                        "right_elbow": 0.0,
                    },
                },
            },
        }
    ]

    def __post_init__(self):
        self.num_actions = self.robots[0].num_joints
