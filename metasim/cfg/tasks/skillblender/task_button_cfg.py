"""ButtonPress config in SkillBench in Skillblender"""

from __future__ import annotations

from typing import Callable

import torch

from metasim.cfg.objects import PrimitiveCubeCfg
from metasim.cfg.simulator_params import SimParamCfg
from metasim.cfg.tasks.base_task_cfg import BaseRLTaskCfg
from metasim.cfg.tasks.skillblender.base_humanoid_cfg import BaseHumanoidCfg, BaseHumanoidCfgPPO
from metasim.types import EnvState
from metasim.utils import configclass


# define new reward function
def reward_wrist_button_distance(env_states: EnvState, robot_name: str, cfg: BaseRLTaskCfg):
    wrist_pos = env_states.robots[robot_name].body_state[:, cfg.wrist_indices, :7]  # [num_envs, 2, 7], two hands
    wrist_pos = wrist_pos[:, 0, :3]  # [num_envs, 3], left hand, position only
    button_goal_pos = env_states.robots[robot_name].extra["button_goal_pos"][:, :3]  # [num_envs, 3]
    wrist_button_diff = wrist_pos - button_goal_pos  # [num_envs, 3]
    wrist_button_error = torch.mean(torch.abs(wrist_button_diff), dim=1)
    return torch.exp(-4 * wrist_button_error), wrist_button_error


def reward_right_arm_default(env_states: EnvState, robot_name: str, cfg: BaseRLTaskCfg):
    """
    Calculates the reward for keeping right arm joint positions close to default positions.
    """
    joint_diff = env_states.robots[robot_name].joint_pos - cfg.default_joint_pd_target
    right_arm_diff = joint_diff[:, cfg.right_shoulder_pitch_index]  # start from right shoulder pitch
    right_arm_error = torch.mean(torch.abs(right_arm_diff), dim=1)
    return torch.exp(-4 * right_arm_error), right_arm_error


@configclass
class TaskButtonCfgPPO(BaseHumanoidCfgPPO):
    @configclass
    class Policy(BaseHumanoidCfgPPO.Policy):
        """Network config class for PPO."""

        num_dofs = 19
        frame_stack = 1
        command_dim = 3
        skill_dict = {
            "h1_wrist_walking": {
                "experiment_name": "h1_wrist_walking",
                "load_run": "2025_0101_093233",
                "checkpoint": -1,
                "low_high": (-2, 2),
            },
            "h1_wrist_reaching": {
                "experiment_name": "h1_wrist_reaching",
                "load_run": "2025_0621_134216",
                "checkpoint": -1,
                "low_high": (-1, 1),
            },
        }

    @configclass
    class Runner(BaseHumanoidCfgPPO.Runner):
        max_iterations = 15001  # 3001  # number of policy updates
        policy_class_name = "ActorCriticHierarchical"

        # logging
        save_interval = 500
        experiment_name = "task_button"

    policy = Policy()
    runner = Runner()


@configclass
class TaskButtonCfg(BaseHumanoidCfg):
    """Cfg class for Skillbench:ButtonPress."""

    task_name = "task_button"
    sim_params = SimParamCfg(
        dt=0.001,
        contact_offset=0.01,
        substeps=1,
        num_position_iterations=4,
        num_velocity_iterations=0,
        bounce_threshold_velocity=0.1,
        replace_cylinder_with_capsule=False,
        friction_offset_threshold=0.04,
        num_threads=10,
    )

    ppo_cfg = TaskButtonCfgPPO()

    command_ranges = BaseHumanoidCfg.CommandRanges(
        lin_vel_x=[-0, 0], lin_vel_y=[-0, 0], ang_vel_yaw=[-0, 0], heading=[-0, 0]
    )

    num_actions = 19
    frame_stack = 1
    c_frame_stack = 3
    command_dim = 3
    num_single_obs = 3 * num_actions + 6 + command_dim  # see `obs_buf = torch.cat(...)` for details
    num_observations = int(frame_stack * num_single_obs)
    single_num_privileged_obs = 3 * num_actions + 18 + 9
    num_privileged_obs = int(c_frame_stack * single_num_privileged_obs)

    num_privileged_obs = int(c_frame_stack * single_num_privileged_obs)
    commands = BaseHumanoidCfg.CommandsConfig(num_commands=4, resampling_time=8.0)

    reward_functions: list[Callable] = [reward_wrist_button_distance, reward_right_arm_default]

    reward_weights: dict[str, float] = {"wrist_button_distance": 5, "right_arm_default": 0.5}

    objects = [
        PrimitiveCubeCfg(
            name="wall",
            size=[0.05, 2.0, 3.0],
            color=[1.0, 1.0, 1.0],
            fix_base_link=True,
            enabled_gravity=True,
        ),
    ]
    env_spacing = 5.0
    init_states = [
        {
            "objects": {
                "wall": {
                    "pos": torch.tensor([1.5, 0.0, 1.5]),
                    "rot": torch.tensor([
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                    ]),
                },
            },
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
            },
        }
    ]

    def __post_init__(self):
        super().__post_init__()
        self.command_ranges.button_pos_y = [-0.5, 0.5]
        self.command_ranges.button_pos_z = [-0.5, 0.5]
        self.button_ori_z = 1.0
        self.right_arm_joint_names = [
            "right_elbow",
            "right_shoulder_pitch",
            "right_shoulder_roll",
            "right_shoulder_yaw",
        ]
