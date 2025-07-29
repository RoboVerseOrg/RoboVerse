# h2o_task_cfg.py
from __future__ import annotations

import torch

from metasim.cfg.tasks.base_task_cfg import BaseRLTaskCfg, SimParamCfg
from metasim.utils import configclass


@configclass
class EvalCfg(BaseRLTaskCfg):
    """Configuration for the HDC evaluation task. See project page https://humanoid-diffusion-controller.github.io/."""

    decimation = 4
    sim_params = SimParamCfg(
        # core
        dt=0.005,
        substeps=1,
        num_threads=4,
        solver_type=1,
        num_position_iterations=4,
        num_velocity_iterations=0,
        contact_offset=0.02,
        rest_offset=0.0,
        bounce_threshold_velocity=0.2,
        max_depenetration_velocity=10.0,
        default_buffer_size_multiplier=10,
    )
    objects = []
    traj_filepath = None
    num_observations = 138
    num_privileged_obs = 215
    num_actions = 19
    max_episode_length = 1000
    # initial pose
    init_states = [
        {
            "objects": {},
            "robots": {
                "h1_verse": {
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
    command_ranges = None
    command_dim = None
    use_vision = False
