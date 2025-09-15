from __future__ import annotations

from functools import partial

import torch
import math
from metasim.scenario.scenario import ScenarioCfg
from metasim.utils import configclass
from metasim.utils.humanoid_robot_util import (
    contact_forces_tensor,
    dof_pos_tensor,
    dof_vel_tensor,
    ref_dof_pos_tensor,
)

# from roboverse_learn.rl.unitree_rl.configs.base_humanoid import BaseHumanoidCfg
from roboverse_learn.rl.unitree_rl.configs.base_legged import BaseLeggedTaskCfg, LeggedRobotCfgPPO
from roboverse_learn.rl.unitree_rl.envs.base_humanoid import Humanoid
from roboverse_learn.rl.unitree_rl.helper.utils import find_unique_candidate



@configclass
class HumanoidWalkingCfgPPO(LeggedRobotCfgPPO):
    seed: int = 0

    algorithm = LeggedRobotCfgPPO.Algorithm(
        entropy_coef=0.001, learning_rate=1e-5, num_learning_epochs=2, gamma=0.994, lam=0.9
    )
    runner = LeggedRobotCfgPPO.Runner(
        num_steps_per_env=60, max_iterations=15001, save_interval=100, experiment_name="humanoid_walking"
    )


@configclass
class HumanoidWalkingCfg(BaseLeggedTaskCfg):
    """Configuration for the walking task."""

    task_name = "humanoid_walking"
    env_spacing: float = 1.0
    max_episode_length_s: int = 24
    control = BaseLeggedTaskCfg.ControlCfg(action_scale=0.25, action_offset=True, torque_limit_scale=0.85)

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
                "g1_dex3": {
                    "pos": torch.tensor([0.0, 0.0, 0.76]),
                    "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                    "dof_pos": {
                        # Hips & legs
                        "left_hip_pitch_joint": -0.4,
                        "left_hip_roll_joint": 0.0,
                        "left_hip_yaw_joint": 0.0,
                        "left_knee_joint": 0.8,
                        "left_ankle_pitch_joint": -0.4,
                        "left_ankle_roll_joint": 0.0,
                        "right_hip_pitch_joint": -0.4,
                        "right_hip_roll_joint": 0.0,
                        "right_hip_yaw_joint": 0.0,
                        "right_knee_joint": 0.8,
                        "right_ankle_pitch_joint": -0.4,
                        "right_ankle_roll_joint": 0.0,
                        # Waist
                        "waist_yaw_joint": 0.0,
                        "waist_roll_joint": 0.0,
                        "waist_pitch_joint": 0.0,
                        # Shoulders & arms
                        "left_shoulder_pitch_joint": 0.0,
                        "left_shoulder_roll_joint": 0.0,
                        "left_shoulder_yaw_joint": 0.0,
                        "left_elbow_joint": 0.0,
                        "left_wrist_roll_joint": 0.0,
                        "left_wrist_pitch_joint": 0.0,
                        "left_wrist_yaw_joint": 0.0,
                        "right_shoulder_pitch_joint": 0.0,
                        "right_shoulder_roll_joint": 0.0,
                        "right_shoulder_yaw_joint": 0.0,
                        "right_elbow_joint": 0.0,
                        "right_wrist_roll_joint": 0.0,
                        "right_wrist_pitch_joint": 0.0,
                        "right_wrist_yaw_joint": 0.0,
                        # Hands
                        "left_hand_thumb_0_joint": 0.0,
                        "left_hand_thumb_1_joint": 0.0,
                        "left_hand_thumb_2_joint": 0.0,
                        "left_hand_middle_0_joint": 0.0,
                        "left_hand_middle_1_joint": 0.0,
                        "left_hand_index_0_joint": 0.0,
                        "left_hand_index_1_joint": 0.0,
                        "right_hand_thumb_0_joint": 0.0,
                        "right_hand_thumb_1_joint": 0.0,
                        "right_hand_thumb_2_joint": 0.0,
                        "right_hand_middle_0_joint": 0.0,
                        "right_hand_middle_1_joint": 0.0,
                        "right_hand_index_0_joint": 0.0,
                        "right_hand_index_1_joint": 0.0,
                    },
                },
            },
        }
    ]

    ppo_cfg = HumanoidWalkingCfgPPO()

    frame_stack = 1
    c_frame_stack = 3

    reward_cfg = BaseLeggedTaskCfg.RewardCfg(base_height_target=0.80, tracking_sigma=1 / 0.2, max_contact_force=700)

    reward_weights: dict[str, float] = {
        "termination": -0.0,
        "ang_vel_xy": -0.2,  # Increased penalty for xy angular velocity to reduce waist waving
        "base_height": 0.2,
        "lin_vel_z": -2.0,
        # "feet_swing_height": -20.0,
        "feet_air_time": 0.0,
        "orientation_sq": -1.0,
        "base_height_sq": -5.0,
        "collision": -1.0,
        "feet_stumble": -0.0,
        "stand_still": -0.0,
        "joint_pos": 1.6,
        "feet_clearance": 2.0,
        "feet_contact_number": 2.4,
        # gait
        "foot_slip": -0.05,
        "feet_distance": 0.3,
        "knee_distance": 0.2,
        # contact
        "feet_contact_forces": -0.05,
        "contact": 0.18,
        # vel tracking
        "tracking_lin_vel": 6.0,
        "tracking_ang_vel": 3.0,
        "vel_mismatch_exp": 0.5,
        "low_speed": 0.2,
        "track_vel_hard": 1.0,
        # base pos
        "default_joint_pos": 1.0,
        "contact_no_vel": -0.2,
        "upper_body_pos": 1.0,  # Increased to keep upper body more stable
        "orientation": 1.0,
        "base_acc": 0.4,  # Increased penalty for base acceleration to reduce jerkiness
        "waist_joint_stability": 10.0,  # New reward for waist joint stability
        "hip_upright_axis": 6.0,  # NEW: Reward for keeping hip axis pointing upward
        # energy
        "action_smoothness": -0.08,  # Increased penalty for jerky actions
        "torques": -1e-5,
        "dof_vel": -1e-3,
        "dof_acc": -2e-6,  # Increased penalty for joint accelerations
        "torque_limits": -0.001,
        "hip_pos": -1.0,
        "dof_pos_limits": -5,
        # optional
        "action_rate": -0.015,  # Increased penalty for action rate changes
    }

    def __post_init__(self):
        super().__post_init__()
        self.num_single_obs: int = self.commands.commands_dim + 9 + 3 * self.num_actions + 2
        self.num_observations: int = int(self.frame_stack * self.num_single_obs)
        self.single_num_privileged_obs: int = self.commands.commands_dim + 12 + 4 * self.num_actions + 14
        self.num_privileged_obs = int(self.c_frame_stack * self.single_num_privileged_obs)


class HumanoidWalkingTask(Humanoid):
    """
    Wrapper for walking

    # TODO implement push robot
    """

    def __init__(self, task_cfg, scenario: ScenarioCfg):
        self.decimation = scenario.decimation
        self._init_from_cfg(task_cfg)
        super().__init__(scenario)
        self._prepare_ref_indices()

    def _init_from_cfg(self, task_cfg):
        self.cfg = task_cfg
        self.num_obs = self.cfg.num_observations
        self.num_actions = self.cfg.num_actions
        self.num_privileged_obs = self.cfg.num_privileged_obs
        self.max_episode_length = math.ceil(self.cfg.max_episode_length_s / self.cfg.sim_params.dt)
        from metasim.utils.dict import class_to_dict
        self.train_cfg = class_to_dict(self.cfg.ppo_cfg)

    def _init_buffers(self):
        super()._init_buffers()
        self.noise_scale_vec = self._get_noise_scale_vec()

    def _prepare_ref_indices(self):
        """get joint indices for reference pos computation."""
        joint_names = self.handler.get_joint_names(self.robot.name)
        find_func = partial(find_unique_candidate, data_base=joint_names)

        def name_extend_func(x):
            return [x, f"{x}_joint"]

        self.left_hip_pitch_joint_idx = find_func(candidates=name_extend_func("left_hip_pitch"))
        self.left_knee_joint_idx = find_func(candidates=name_extend_func("left_knee"))
        self.right_hip_pitch_joint_idx = find_func(candidates=name_extend_func("right_hip_pitch"))
        self.right_knee_joint_idx = find_func(candidates=name_extend_func("right_knee"))
        self.left_ankle_joint_idx = find_func(
            candidates=name_extend_func("left_ankle") + name_extend_func("left_ankle_pitch")
        )
        self.right_ankle_joint_idx = find_func(
            candidates=name_extend_func("right_ankle") + name_extend_func("right_ankle_pitch")
        )

    def _compute_ref_state(self):
        """compute reference target position for walking task."""
        phase = self._get_phase()
        sin_pos = torch.sin(2 * torch.pi * phase)
        sin_pos_l = sin_pos.clone()
        sin_pos_r = sin_pos.clone()
        self.ref_dof_pos = torch.zeros(
            self.num_envs, self.handler.robot_num_dof, device=self.device, requires_grad=False
        )
        # Scale gait amplitude by command magnitude so zero command => no gait
        # Combine linear speed and a fraction of yaw command to allow in-place turning
        lin_speed = torch.norm(self.commands[:, :2], dim=1)
        yaw_speed = torch.abs(self.commands[:, 2]) if self.commands.shape[1] > 2 else 0.0
        speed_factor = torch.clamp(lin_speed + 0.5 * yaw_speed, 0.0, 1.0).unsqueeze(1)
        scale_1 = self.cfg.reward_cfg.target_joint_pos_scale
        scale_2 = 2 * scale_1
        sin_pos_l[sin_pos_l > 0] = 0
        self.ref_dof_pos[:, self.left_hip_pitch_joint_idx] = sin_pos_l * scale_1  # left_hip_pitch_joint
        self.ref_dof_pos[:, self.left_knee_joint_idx] = sin_pos_l * scale_2  # left_knee_joint
        self.ref_dof_pos[:, self.left_ankle_joint_idx] = sin_pos_l * scale_1  # left_ankle_joint
        sin_pos_r[sin_pos_r < 0] = 0
        self.ref_dof_pos[:, self.right_hip_pitch_joint_idx] = sin_pos_r * scale_1  # right_hip_pitch_joint
        self.ref_dof_pos[:, self.right_knee_joint_idx] = sin_pos_r * scale_2  # right_knee_joint
        self.ref_dof_pos[:, self.right_ankle_joint_idx] = sin_pos_r * scale_1  # right_ankle_joint
        # Double support phase
        self.ref_dof_pos[torch.abs(sin_pos) < 0.1] = 0
        self.ref_dof_pos = 2 * self.ref_dof_pos
        # Apply speed-dependent amplitude
        self.ref_dof_pos *= speed_factor

    def _parse_ref_pos(self, envstate):
        envstate.robots[self.robot.name].extra["ref_dof_pos"] = self.ref_dof_pos

    def _parse_state_for_reward(self, envstate):
        """
        Parse all the states to prepare for reward computation, legged_robot level reward computation.
        """

        super()._parse_state_for_reward(envstate)
        self._compute_ref_state()
        self._parse_ref_pos(envstate)

    def _get_noise_scale_vec(self) -> torch.Tensor:
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        # Observation layout (single frame):
        # 0:3 commands, 3:6 base_ang_vel, 6:9 base_euler_xyz, 9:12 projected_gravity,
        # 12:12+A q, 12+A:12+2A dq, 12+2A:12+3A actions, +1 sin, +1 cos
        noise_vec[0:3] = 0.0  # commands (no noise)
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.cfg.normalization.obs_scales.ang_vel
        noise_vec[6:9] = 0.0  # base_euler_xyz (keep clean)
        noise_vec[9:12] = noise_scales.gravity * noise_level
        start = 12
        A = self.num_actions
        noise_vec[start : start + A] = noise_scales.dof_pos * noise_level * self.cfg.normalization.obs_scales.dof_pos
        noise_vec[start + A : start + 2 * A] = (
            noise_scales.dof_vel * noise_level * self.cfg.normalization.obs_scales.dof_vel
        )
        noise_vec[start + 2 * A : start + 3 * A] = 0.0  # previous actions (actor already outputs noisy actions)
        noise_vec[start + 3 * A : start + 3 * A + 2] = 0.0  # sin/cos phase

        return noise_vec

    def compute_observations(self, envstates):
        """compute observation and privileged observation."""

        phase = self._get_phase()

        sin_phase = torch.sin(2 * torch.pi * phase).unsqueeze(1)
        cos_phase = torch.cos(2 * torch.pi * phase).unsqueeze(1)

        stance_mask = self._get_gait_phase()
        contact_mask = contact_forces_tensor(envstates, self.robot.name)[:, self.feet_indices, 2] > 5

        q = (
            dof_pos_tensor(envstates, self.robot.name) - self.cfg.default_joint_pd_target
        ) * self.cfg.normalization.obs_scales.dof_pos
        dq = dof_vel_tensor(envstates, self.robot.name) * self.cfg.normalization.obs_scales.dof_vel
        diff = dof_pos_tensor(envstates, self.robot.name) - ref_dof_pos_tensor(envstates, self.robot.name)

        self.privileged_obs_buf = torch.cat(
            (
                self.commands[:, :3] * self.commands_scale,  # 3
                self.base_lin_vel * self.cfg.normalization.obs_scales.lin_vel,  # 3
                self.base_ang_vel * self.cfg.normalization.obs_scales.ang_vel,  # 3
                self.base_euler_xyz * self.cfg.normalization.obs_scales.quat,  # 3
                self.projected_gravity,  # 3
                q,  # |A|
                dq,  # |A|
                self.actions,  # |A|
                diff,  # |A|
                self.rand_push_force[:, :3],  # 3
                self.rand_push_torque,  # 3
                self.env_frictions,  # 1
                self.body_mass / 30.0,  # 1
                stance_mask,  # 2
                contact_mask,  # 2
                sin_phase,  # 1
                cos_phase,  # 1
            ),
            dim=-1,
        )

        obs_buf = torch.cat(
            (
                self.commands[:, :3] * self.commands_scale,  # 3
                self.base_ang_vel * self.cfg.normalization.obs_scales.ang_vel,  # 3
                self.base_euler_xyz * self.cfg.normalization.obs_scales.quat,  # 3
                self.projected_gravity,  # 3
                q,  # |A|
                dq,  # |A|
                self.actions,
                sin_phase,
                cos_phase,
            ),
            dim=-1,
        )

        obs_now = obs_buf.clone()
        self.obs_history.append(obs_now)
        self.critic_history.append(self.privileged_obs_buf)
        obs_buf_all = torch.stack([self.obs_history[i] for i in range(self.obs_history.maxlen)], dim=1)
        self.obs_buf = obs_buf_all.reshape(self.num_envs, -1)
        self.privileged_obs_buf = torch.cat([self.critic_history[i] for i in range(self.cfg.c_frame_stack)], dim=1)

        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec
