# h2o_task_cfg.py
from __future__ import annotations
import torch
from metasim.cfg.tasks.base_task_cfg import BaseRLTaskCfg, SimParamCfg
from metasim.utils import configclass
@configclass
class HDCTaskCfg(BaseRLTaskCfg):

    decimation=4
    sim_params = SimParamCfg(
        # core
        dt=0.005,                 # [s] sim step
        substeps=1,               # internal sub-steps
        gravity=[0.0, 0.0, -9.81],
        up_axis=1,                # 0: Y-up, 1: Z-up

        # physx-related options in flat form
        num_threads=4,
        solver_type=1,                   # 0 = PGS, 1 = TGS
        num_position_iterations=4,
        num_velocity_iterations=0,
        contact_offset=0.02,             # [m]
        rest_offset=0.0,                 # [m]
        bounce_threshold_velocity=0.2,   # [m/s]
        max_depenetration_velocity=10.0,
        max_gpu_contact_pairs=16_777_216,
        default_buffer_size_multiplier=10,
        contact_collection=2,            # 0: never, 1: last, 2: all sub-steps
    )
    # dt :  0.005
    # substeps : 1
    # gravity : [0., 0. ,-9.81]  # [m/s^2]
    # up_axis : 1  # 0 is y, 1 is z

    # physx:
    #     num_threads : 4
    #     solver_type : 1  # 0: pgs, 1: tgs
    #     num_position_iterations : 4
    #     num_velocity_iterations : 0
    #     contact_offset : 0.02  # [m]
    #     rest_offset : 0.0   # [m]
    #     bounce_threshold_velocity : 0.2 #0.5 [m/s]
    #     max_depenetration_velocity : 10
    #     max_gpu_contact_pairs : 16777216 #  -> needed for 8000 envs and more
    #     default_buffer_size_multiplier : 10
    #     contact_collection : 2 # 0: never, 1: last sub-step, 2: all sub-steps (default:2)

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
