"""Base class for humanoid tasks."""

from __future__ import annotations

import logging
from typing import Literal

import torch
from loguru import logger as log
from rich.logging import RichHandler

from metasim.cfg.objects import RigidObjCfg
from metasim.cfg.sensors.contact import ContactForceSensorCfg
from metasim.cfg.tasks.base_task_cfg import BaseRLTaskCfg, SimParamCfg
from metasim.constants import BenchmarkType, PhysicStateType, TaskType
from metasim.types import EnvState
from metasim.utils import configclass, math
from metasim.utils.bidex_reward_util import compute_hand_reward

logging.addLevelName(5, "TRACE")
log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])

########################################################
## Constants adapted from DexterousHands/bidexhands/tasks/shadow_hand_over.py
########################################################


@configclass
class ShadowHandOverCfg(BaseRLTaskCfg):
    """class for bidex shadow hand over tasks."""

    source_benchmark = BenchmarkType.BIDEX
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 75  # TODO: may change
    traj_filepath = "roboverse_data/trajs/bidex/ShadowHandOver/v2/initial_state_v2.json"
    device = "cuda:0"
    num_envs = None
    obs_shape = 398  # 398-dimensional observation space
    objects_type = ["cube"]
    objects_urdf_path = ["roboverse_data/assets/bidex/objects/cube_multicolor.urdf"]
    objects = [
        RigidObjCfg(
            name=objects_type[0],
            scale=(1, 1, 1),
            physics=PhysicStateType.RIGIDBODY,
            urdf_path=objects_urdf_path[0],
        ),
    ]
    robots = ["shadow_hand_right", "shadow_hand_left"]
    sim_params = SimParamCfg(
        dt=1.0 / 60.0,
        contact_offset=0.002,
        num_velocity_iterations=0,
        bounce_threshold_velocity=0.2,
        num_threads=4,
        use_gpu_pipeline=True,
        use_gpu=True,
        substeps=2,
    )

    init_goal_pos = torch.tensor(
        [0.01, -0.635, 0.54], dtype=torch.float32, device=device
    )  # Initial goal position, shape (3,)
    init_goal_rot = torch.tensor(
        [1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=device
    )  # Initial goal rotation, shape (4,)
    goal_pos = None  # Placeholder for goal position, to be set later, shape (num_envs, 3)
    goal_rot = None  # Placeholder for goal rotation, to be set later, shape (num_envs, 4)
    r_fingertips = ["robot0_ffdistal", "robot0_mfdistal", "robot0_rfdistal", "robot0_lfdistal", "robot0_thdistal"]
    l_fingertips = ["robot1_ffdistal", "robot1_mfdistal", "robot1_rfdistal", "robot1_lfdistal", "robot1_thdistal"]
    sensors = []
    for name in r_fingertips:
        sensors.append(ContactForceSensorCfg(base_link=("shadow_hand_right", name), source_link=None, name=name))
    for name in l_fingertips:
        sensors.append(ContactForceSensorCfg(base_link=("shadow_hand_left", name), source_link=None, name=name))
    r_fingertips_idx = None
    l_fingertips_idx = None
    vel_obs_scale: float = 0.2  # Scale for velocity observations
    force_torque_obs_scale: float = 10.0  # Scale for force and torque observations
    shadow_hand_dof_lower_limits: torch.Tensor = torch.tensor(
        [
            -0.489,
            -0.698,
            -0.349,
            0,
            0,
            0,
            -0.349,
            0,
            0,
            0,
            -0.349,
            0,
            0,
            0,
            0,
            -0.349,
            0,
            0,
            0,
            -1.047,
            0,
            -0.209,
            -0.524,
            -1.571,
        ],
        dtype=torch.float32,
        device=device,
    )
    shadow_hand_dof_upper_limits: torch.Tensor = torch.tensor(
        [
            0.14,
            0.489,
            0.349,
            1.571,
            1.571,
            1.571,
            0.349,
            1.571,
            1.571,
            1.571,
            0.349,
            1.571,
            1.571,
            1.571,
            0.785,
            0.349,
            1.571,
            1.571,
            1.571,
            1.047,
            1.222,
            0.209,
            0.524,
            0,
        ],
        dtype=torch.float32,
        device=device,
    )
    sim: Literal["isaaclab", "isaacgym", "genesis", "pyrep", "pybullet", "sapien", "sapien3", "mujoco", "blender"] = (
        "isaacgym"
    )
    dist_reward_scale = 50
    success_tolerance = 0.1
    reach_goal_bonus = 250
    fall_penalty = 0.0
    init_states = [
        {
            "objects": {
                "cube": {
                    "pos": torch.tensor([0.01, -0.385, 0.54]),
                    "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                },
            },
            "robots": {
                "shadow_hand_right": {
                    "pos": torch.tensor([0.0, 0.0, 0.5]),
                    "rot": torch.tensor([0.0, 0.0, -0.707, 0.707]),
                    "dof_pos": {
                        "robot0_WRJ1": 0.0,
                        "robot0_WRJ0": 0.0,
                        "robot0_FFJ3": 0.0,
                        "robot0_FFJ2": 0.0,
                        "robot0_FFJ1": 0.0,
                        "robot0_FFJ0": 0.0,
                        "robot0_MFJ3": 0.0,
                        "robot0_MFJ2": 0.0,
                        "robot0_MFJ1": 0.0,
                        "robot0_MFJ0": 0.0,
                        "robot0_RFJ3": 0.0,
                        "robot0_RFJ2": 0.0,
                        "robot0_RFJ1": 0.0,
                        "robot0_RFJ0": 0.0,
                        "robot0_LFJ4": 0.0,
                        "robot0_LFJ3": 0.0,
                        "robot0_LFJ2": 0.0,
                        "robot0_LFJ1": 0.0,
                        "robot0_LFJ0": 0.0,
                        "robot0_THJ4": 0.0,
                        "robot0_THJ3": 0.0,
                        "robot0_THJ2": 0.0,
                        "robot0_THJ1": 0.0,
                        "robot0_THJ0": 0.0,
                    },
                },
                "shadow_hand_left": {
                    "pos": torch.tensor([0.0, -1.0, 0.5]),
                    "rot": torch.tensor([-0.707, 0.707, 0.0, 0.0]),
                    "dof_pos": {
                        "robot1_WRJ1": 0.0,
                        "robot1_WRJ0": 0.0,
                        "robot1_FFJ3": 0.0,
                        "robot1_FFJ2": 0.0,
                        "robot1_FFJ1": 0.0,
                        "robot1_FFJ0": 0.0,
                        "robot1_MFJ3": 0.0,
                        "robot1_MFJ2": 0.0,
                        "robot1_MFJ1": 0.0,
                        "robot1_MFJ0": 0.0,
                        "robot1_RFJ3": 0.0,
                        "robot1_RFJ2": 0.0,
                        "robot1_RFJ1": 0.0,
                        "robot1_RFJ0": 0.0,
                        "robot1_LFJ4": 0.0,
                        "robot1_LFJ3": 0.0,
                        "robot1_LFJ2": 0.0,
                        "robot1_LFJ1": 0.0,
                        "robot1_LFJ0": 0.0,
                        "robot1_THJ4": 0.0,
                        "robot1_THJ3": 0.0,
                        "robot1_THJ2": 0.0,
                        "robot1_THJ1": 0.0,
                        "robot1_THJ0": 0.0,
                    },
                },
            },
        }
    ]

    def observation_fn(self, envstates: list[EnvState], actions: torch.Tensor, device=None) -> torch.Tensor:
        """Observation function for shadow hand over tasks.

        Args:
            envstates (list[EnvState]): List of environment states to process.
            actions (torch.Tensor): Actions taken by the agents in the environment, shape (num_envs, num_actions).

        Returns:
            Compute the observations of all environment. The observation is composed of three parts:
            the state values of the left and right hands, and the information of objects and target.
            The state values of the left and right hands were the same for each task, including hand
            joint and finger positions, velocity, and force information. The detail 422-dimensional
            observational space as shown in below:

            Index       Description
            0 - 23	    right shadow hand dof position
            24 - 47	    right shadow hand dof velocity
            48 - 71	    right shadow hand dof force
            72 - 136	right shadow hand fingertip pose, linear velocity, angle velocity (5 x 13)
            137 - 166	right shadow hand fingertip force, torque (5 x 6)
            167 - 186	right shadow hand actions
            187 - 210	left shadow hand dof position
            211 - 234	left shadow hand dof velocity
            235 - 258	left shadow hand dof force
            259 - 323	left shadow hand fingertip pose, linear velocity, angle velocity (5 x 13)
            324 - 353	left shadow hand fingertip force, torque (5 x 6)
            354 - 373	left shadow hand actions
            374 - 380	object pose
            381 - 383	object linear velocity
            383 - 386	object angle velocity
            387 - 393	goal pose
            394 - 397	goal rot - object rot
        """
        if device is None:
            device = self.device
        num_envs = envstates.robots["shadow_hand_right"].root_state.shape[0]
        if self.num_envs is None:
            self.num_envs = num_envs
        if self.goal_pos is None:
            self.goal_pos = torch.tensor(self.init_goal_pos, dtype=torch.float32).view(1, -1).repeat(num_envs, 1)
        if self.goal_rot is None:
            self.goal_rot = torch.tensor(self.init_goal_rot, dtype=torch.float32).view(1, -1).repeat(num_envs, 1)
        obs = torch.zeros((num_envs, 398), dtype=torch.float32, device=device)
        obs[:, :24] = math.scale_transform(
            envstates.robots["shadow_hand_right"].joint_pos,
            self.shadow_hand_dof_lower_limits,
            self.shadow_hand_dof_upper_limits,
        )
        obs[:, 24:48] = envstates.robots["shadow_hand_right"].joint_vel * self.vel_obs_scale
        obs[:, 48:72] = envstates.robots["shadow_hand_right"].joint_force * self.force_torque_obs_scale
        if self.r_fingertips_idx is None:
            self.r_fingertips_idx = [
                envstates.robots["shadow_hand_right"].body_names.index(name) for name in self.r_fingertips
            ]
        obs[:, 72:137] = (
            envstates.robots["shadow_hand_right"].body_state[:, self.r_fingertips_idx, :].view(num_envs, -1)
        )
        t = 137
        for name in self.r_fingertips:
            # shape: (num_envs, 3) + (num_envs, 3) => (num_envs, 6)
            force = envstates.sensors[name].force  # (num_envs, 3)
            torque = envstates.sensors[name].torque  # (num_envs, 3)
            obs[:, t : t + 6] = torch.cat([force, torque], dim=1) * self.force_torque_obs_scale  # (num_envs, 6)
            t += 6
        obs[:, 167:187] = actions[:, :20]  # actions for right hand
        obs[:, 187:211] = math.scale_transform(
            envstates.robots["shadow_hand_left"].joint_pos,
            self.shadow_hand_dof_lower_limits,
            self.shadow_hand_dof_upper_limits,
        )
        obs[:, 211:235] = envstates.robots["shadow_hand_left"].joint_vel * self.vel_obs_scale
        obs[:, 235:259] = envstates.robots["shadow_hand_left"].joint_force * self.force_torque_obs_scale
        if self.l_fingertips_idx is None:
            self.l_fingertips_idx = [
                envstates.robots["shadow_hand_left"].body_names.index(name) for name in self.l_fingertips
            ]
        obs[:, 259:324] = (
            envstates.robots["shadow_hand_left"].body_state[:, self.l_fingertips_idx, :].view(num_envs, -1)
        )
        t = 324
        for name in self.l_fingertips:
            # shape: (num_envs, 3) + (num_envs, 3) => (num_envs, 6)
            force = envstates.sensors[name].force
            torque = envstates.sensors[name].torque
            obs[:, t : t + 6] = torch.cat([force, torque], dim=1) * self.force_torque_obs_scale  # (num_envs, 6)
            t += 6
        obs[:, 354:374] = actions[:, :20]  # actions for left hand
        obs[:, 374:387] = envstates.objects["cube"].root_state
        obs[:384:387] *= self.vel_obs_scale  # object angvel
        obs[:, 387:394] = torch.cat([self.goal_pos, self.goal_rot], dim=1)  # goal position and rotation (num_envs, 7)
        obs[:, 394:] = math.quat_mul(
            envstates.objects["cube"].root_state[:, 3:7], math.quat_inv(self.goal_rot)
        )  # goal rotation - object rotation
        return obs

    def reward_fn(
        self,
        envstates: list[EnvState],
        actions,
        reset_buf,
        reset_goal_buf,
        episode_length_buf,
        success_buf,
    ) -> torch.Tensor:
        """Compute the reward of all environment. The core function is compute_hand_reward.

        Args:
            envstates (list[EnvState]): States of the environment
            actions (tensor): Actions of agents in the all environment, shape (num_envs, num_actions)
            episode_length_buf (torch.Tensor): The episode length buffer, shape (num_envs,)
            reset_buf (torch.Tensor): The reset buffer of all environments at this time, shape (num_envs,)
            reset_goal_buf (torch.Tensor): The reset goal buffer of all environments at this time, shape (num_envs,)
            success_buf (torch.Tensor): The success buffer of all environments at this time, shape (num_envs,)

        Returns:
            reward (torch.Tensor): The reward of all environments at this time, shape (num_envs,)
            reset_buf (torch.Tensor): The reset buffer of all environments at this time, shape (num_envs,)
            reset_goal_buf (torch.Tensor): The reset goal buffer of all environments at this time, shape (num_envs,)
            success_buf (torch.Tensor): The success buffer of all environments at this time, shape (num_envs,)
        """
        (
            reward,
            reset_buf,
            reset_goal_buf,
            success_buf,
        ) = compute_hand_reward(
            reset_buf=reset_buf,
            reset_goal_buf=reset_goal_buf,
            episode_length_buf=episode_length_buf,
            success_buf=success_buf,
            max_episode_length=self.episode_length,
            object_pos=envstates.objects["cube"].root_state[:, :3],
            object_rot=envstates.objects["cube"].root_state[:, 3:7],
            target_pos=self.goal_pos,
            target_rot=self.goal_rot,
            dist_reward_scale=self.dist_reward_scale,
            actions=actions,
            success_tolerance=self.success_tolerance,
            reach_goal_bonus=self.reach_goal_bonus,
            fall_penalty=self.fall_penalty,
            ignore_z_rot=False,  # Todo : set to True if the object is a pen or similar object that does not require z-rotation alignment
        )
        return reward, reset_buf, reset_goal_buf, success_buf

    def goal_reset_fn(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Reset the goal position and rotation for the environment.

        Args:
            env_ids (torch.Tensor): The reset goal buffer of all environments at this time, shape (num_envs_to_reset,).
        """
        self.goal_pos[env_ids] = self.init_goal_pos.clone()
        self.goal_rot[env_ids] = self.init_goal_rot.clone()  # Todo: add randomization to goal pos and rot
