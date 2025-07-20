# h2o_task_cfg.py
from __future__ import annotations
import torch
from metasim.cfg.tasks.base_task_cfg import BaseRLTaskCfg
from metasim.utils import configclass
@configclass
class H2OTaskCfg(BaseRLTaskCfg):
    
    # initial pose
    init_states =  [
            {
                "objects": {},
                "robots": {
                    "humanoid": {
                        "pos": torch.tensor([0.0, 0.0, 1.0]),
                        "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),  # w, x, y, z
                        "lin_vel": torch.tensor([0.0, 0.0, 0.0]),
                        "ang_vel": torch.tensor([0.0, 0.0, 0.0]),
                        "max_linvel": 0.5,
                        "max_angvel": 0.5,
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
                },
            }
        ]
