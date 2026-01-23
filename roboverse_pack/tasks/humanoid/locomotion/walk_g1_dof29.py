from __future__ import annotations

import copy
import math

import torch

from metasim.queries import ContactForces
from metasim.scenario.lights import DomeLightCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.scenario.simulator_params import SimParamCfg
from metasim.task.registry import register_task
from metasim.types import TensorState
from metasim.utils import configclass
from metasim.utils.math import quat_rotate_inverse
from roboverse_pack.callback_funcs.humanoid import (
    reset_funcs,
    reward_funcs,
    step_funcs,
    termination_funcs,
)
from roboverse_pack.tasks.humanoid.base import LeggedRobotTask
from roboverse_pack.tasks.humanoid.cfg_base import BaseEnvCfg
from roboverse_pack.utils.humanoid_utils import get_indices_from_substring


@configclass
class WalkG1Dof29EnvCfg(BaseEnvCfg):
    """Environment configuration for humanoid walking task."""

    obs_len_history = 1
    priv_obs_len_history = 1
    episode_length_s = 20.0

    control = BaseEnvCfg.Control(action_scale=0.5, soft_joint_pos_limit_factor=0.9)
    observed_joint_names = ["waist.*", ".*_hip.*", ".*_knee.*", ".*_ankle.*"]

    @configclass
    class RewardsScales:
        """Reward weights for gait, posture, and energy usage."""

        termination_penalty = (-200.0, {}, reward_funcs.termination_penalty)
        track_lin_vel_xy = (
            1.0,
            {"std": math.sqrt(0.25)},
            reward_funcs.track_lin_vel_xy_yaw_frame,
        )
        track_ang_vel_z = (
            1.0,
            {"std": math.sqrt(0.25)},
            reward_funcs.track_ang_vel_z_world,
        )
        lin_vel_z = -0.2
        ang_vel_xy = -0.05
        flat_orientation = -1.0
        action_rate = -0.005
        dof_acc_l2 = (
            -1.0e-7,
            {"joint_names": (".*_hip_.*", ".*_knee_joint")},
            reward_funcs.joint_acc,
        )
        dof_torques_l2 = (
            -2.0e-6,
            {"joint_names": (".*_hip_.*", ".*_knee_joint")},
            reward_funcs.joint_torques_l2,
        )
        feet_air_time = (
            0.75,
            {"threshold": 0.4, "body_names": ".*_ankle_roll_link"},
            reward_funcs.feet_air_time_positive_biped,
        )
        feet_slide = (-0.1, {"body_names": ".*_ankle_roll_link"})
        dof_pos_limits = (
            -1.0,
            {"joint_names": (".*_ankle_pitch_joint", ".*_ankle_roll_joint")},
            reward_funcs.joint_pos_limits,
        )
        joint_deviation_hip = (
            -0.1,
            {"joint_names": (".*_hip_yaw_joint", ".*_hip_roll_joint")},
            reward_funcs.joint_deviation_l1,
        )
        joint_deviation_arms = (
            -0.1,
            {
                "joint_names": (
                    ".*_shoulder_pitch_joint",
                    ".*_shoulder_roll_joint",
                    ".*_shoulder_yaw_joint",
                    ".*_elbow_joint",
                    ".*_wrist_.*_joint",
                )
            },
            reward_funcs.joint_deviation_l1,
        )
        joint_deviation_torso = (
            -0.2,
            {"joint_names": "waist_.*_joint"},
            reward_funcs.joint_deviation_l1,
        )

    rewards = BaseEnvCfg.Rewards(
        only_positive_rewards=False,
        scales=RewardsScales(),
    )

    commands = BaseEnvCfg.Commands(
        value=None,
        resample=step_funcs.resample_commands,
        heading_command=True,
        resampling_time=1.0e9,
        rel_standing_envs=0.02,
        ranges=BaseEnvCfg.Commands.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-0.5, 0.5),
            ang_vel_yaw=(-1.0, 1.0),
            heading=(-3.14, 3.14),
        ),
        limit_ranges=BaseEnvCfg.Commands.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-0.5, 0.5),
            ang_vel_yaw=(-1.0, 1.0),
            heading=(-3.14, 3.14),
        ),
    )

    curriculum = BaseEnvCfg.Curriculum(enabled=False, funcs={})

    callbacks_query = {"contact_forces": ContactForces(history_length=3)}
    callbacks_setup = {}
    callbacks_reset = {
        "random_root_state": (
            reset_funcs.random_root_state_terrain_aware,
            {
                "pose_range": [
                    [-0.5, -0.5, 0.0, 0.0, 0.0, -3.14],  # x, y, z_offset, roll, pitch, yaw
                    [0.5, 0.5, 0.0, 0.0, 0.0, 3.14],
                ],
                "velocity_range": [[0] * 6, [0] * 6],
                # base_height_offset is None by default, uses robot's default z position
            },
        ),
        "reset_joints_by_scale": (
            reset_funcs.reset_joints_by_scale,
            {"position_range": (1.0, 1.0), "velocity_range": (0.0, 0.0)},
        ),
    }
    callbacks_post_step = {}
    callbacks_terminate = {
        "time_out": termination_funcs.time_out,
        "base_contact": (
            termination_funcs.root_height_below_minimum,
            {"minimum_height": 0.2},
        ),
    }
    initial_states = BaseEnvCfg.InitialStates()
    initial_states.robots = {
        **BaseEnvCfg.InitialStates.robots,
        "g1_dof29": {
            "pos": [0.0, 0.0, 0.78],
            "default_joint_pos": {
                ".*_hip_pitch_joint": -0.1,
                ".*_knee_joint": 0.3,
                ".*_ankle_pitch_joint": -0.2,
                ".*_wrist_.*_joint": 0.0,
            },
        },
    }


@register_task(
    "unitree_rl.walk_g1_dof29",
    "g1.walk_g1_dof29",
    "walk_g1_dof29",
)
class WalkG1Dof29Task(LeggedRobotTask):
    """Registered humanoid locomotion task."""

    env_cfg_cls = WalkG1Dof29EnvCfg
    task_name = "walk_g1_dof29"

    scenario = ScenarioCfg(
        robots=["g1_dof29"],
        objects=[],
        cameras=[],
        num_envs=128,
        simulator="isaacgym",
        headless=True,
        env_spacing=2.5,
        decimation=1,
        sim_params=SimParamCfg(
            dt=0.005,
            substeps=1,
            num_threads=10,
            solver_type=1,
            num_position_iterations=4,
            num_velocity_iterations=0,
            contact_offset=0.01,
            rest_offset=0.0,
            bounce_threshold_velocity=0.5,
            max_depenetration_velocity=1.0,
            default_buffer_size_multiplier=5,
            replace_cylinder_with_capsule=True,
            friction_correlation_distance=0.025,
            friction_offset_threshold=0.04,
            njmax=210,
            nconmax=64,
            newton_use_mujoco_contacts=True,
            newton_solver_iterations=100,
            newton_ls_iterations=10,
            newton_solver="newton",
            newton_integrator="implicit",
            newton_cone="pyramidal",
            newton_impratio=1.0,
            newton_ls_parallel=True,
        ),
        lights=[
            DomeLightCfg(
                intensity=800.0,
                color=(0.85, 0.9, 1.0),
            )
        ],
    )

    def __init__(
        self,
        scenario: ScenarioCfg | None = None,
        device: str | torch.device | None = None,
        env_cfg: WalkG1Dof29EnvCfg | None = None,
    ) -> None:
        scenario_copy = copy.deepcopy(scenario or type(self).scenario)
        scenario_copy.__post_init__()

        if env_cfg is None:
            env_cfg = type(self).env_cfg_cls()

        if device is None:
            device = "cpu" if scenario_copy.simulator == "mujoco" else ("cuda" if torch.cuda.is_available() else "cpu")

        super().__init__(scenario=scenario_copy, config=env_cfg, device=device)

    def _pre_physics_step(self, actions: torch.Tensor) -> torch.Tensor:
        actions = super()._pre_physics_step(actions)
        if hasattr(self, "action_mask") and self.action_mask is not None:
            actions = actions * self.action_mask
        return actions

    def _init_buffers(self):
        self.obs_joint_indices = get_indices_from_substring(self.cfg.observed_joint_names, self.sorted_joint_names)
        if self.obs_joint_indices.numel() == 0:
            self.obs_joint_indices = torch.arange(self.num_actions, device=self.device)
        else:
            self.obs_joint_indices = self.obs_joint_indices.to(self.device)
        self.obs_joint_indices = self.obs_joint_indices.long()
        self.num_obs_joints = int(self.obs_joint_indices.numel())

        self.action_mask = torch.zeros(self.num_actions, dtype=torch.float, device=self.device)
        self.action_mask[self.obs_joint_indices] = 1.0

        # base_ang_vel + projected_gravity + commands + dof pos/vel/actions
        self.num_obs = 9 + self.num_obs_joints * 3
        self.num_priv_obs = self.num_obs + 3

        self.obs_clip_limit = 100.0
        self.obs_scale = torch.ones(size=(self.num_obs,), dtype=torch.float, device=self.device)
        self.priv_obs_scale = torch.ones(size=(self.num_priv_obs,), dtype=torch.float, device=self.device)
        self.obs_noise = torch.zeros(size=(self.num_obs,), dtype=torch.float, device=self.device)

        self.obs_noise[0:3] = 0.1  # base_lin_vel
        self.obs_noise[3:6] = 0.2  # base_ang_vel
        self.obs_noise[6:9] = 0.05  # projected_gravity
        # commands are noiseless
        pos_start = 12
        self.obs_noise[pos_start : pos_start + self.num_obs_joints] = 0.01
        vel_start = pos_start + self.num_obs_joints
        self.obs_noise[vel_start : vel_start + self.num_obs_joints] = 1.5
        return super()._init_buffers()

    def _compute_task_observations(self, env_states: TensorState):
        robot_state = env_states.robots[self.robot.name]
        base_quat = robot_state.root_state[:, 3:7]
        base_lin_vel = quat_rotate_inverse(base_quat, robot_state.root_state[:, 7:10])
        base_ang_vel = quat_rotate_inverse(base_quat, robot_state.root_state[:, 10:13])
        projected_gravity = quat_rotate_inverse(base_quat, self.gravity_vec)

        q = robot_state.joint_pos[:, self.obs_joint_indices] - self.default_dof_pos[self.obs_joint_indices]
        dq = robot_state.joint_vel[:, self.obs_joint_indices] - self.default_dof_vel[self.obs_joint_indices]
        actions = self.actions[:, self.obs_joint_indices]

        obs_buf = torch.cat(
            (
                self.commands_manager.value,  # 3
                base_ang_vel,  # 3
                projected_gravity,  # 3
                q,  # |A|
                dq,  # |A|
                actions,  # |A|
                # gait
            ),
            dim=-1,
        )

        priv_obs_buf = torch.cat(
            (
                self.commands_manager.value,  # 3
                base_lin_vel,  # 3
                base_ang_vel,  # 3
                projected_gravity,  # 3
                q,  # |A|
                dq,  # |A|
                actions,  # |A|
                # gait
            ),
            dim=-1,
        )

        obs_buf += (2 * torch.rand_like(obs_buf) - 1) * self.obs_noise

        # clip observations -> scale observations
        obs_buf = obs_buf.clip(-self.obs_clip_limit, self.obs_clip_limit) * self.obs_scale
        priv_obs_buf = priv_obs_buf.clip(-self.obs_clip_limit, self.obs_clip_limit) * self.priv_obs_scale

        return obs_buf, priv_obs_buf
