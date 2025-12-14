from __future__ import annotations

import copy
import math

import torch

import roboverse_pack.utils.curriculum_utils as curr_funs
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
from roboverse_pack.queries.lidar import LidarPointCloud
from roboverse_pack.randomization.humanoid import (
    MassRandomizer,
    MaterialRandomizer,
)
from roboverse_pack.tasks.humanoid.base import LeggedRobotTask
from roboverse_pack.tasks.humanoid.cfg_base import BaseEnvCfg


@configclass
class WalkG1Dof29EnvCfg(BaseEnvCfg):
    """Environment configuration for humanoid walking task."""

    obs_len_history = 5
    priv_obs_len_history = 5
    episode_length_s = 20.0

    control = BaseEnvCfg.Control(action_scale=0.25, soft_joint_pos_limit_factor=0.9)

    @configclass
    class RewardsScales:
        """Reward weights for gait, posture, and energy usage."""

        track_lin_vel_xy = (1.0, {"std": math.sqrt(0.25)})
        track_ang_vel_z = (0.5, {"std": math.sqrt(0.25)})
        is_alive = 0.15
        lin_vel_z = -2.0
        ang_vel_xy = -0.05
        joint_vel = -0.001
        joint_acc = -2.5e-7
        action_rate = -0.05
        joint_pos_limits = -5.0
        energy = -2e-5
        joint_deviation_arms = (
            -0.1,
            {"joint_names": (".*_shoulder_.*_joint", ".*_elbow_joint", ".*_wrist_.*")},
            reward_funcs.joint_deviation_l1,
        )
        joint_deviation_waists = (
            -1.0,
            {"joint_names": "waist.*"},
            reward_funcs.joint_deviation_l1,
        )
        joint_deviation_legs = (
            -1.0,
            {"joint_names": (".*_hip_roll_joint", ".*_hip_yaw_joint")},
            reward_funcs.joint_deviation_l1,
        )
        flat_orientation = -5.0
        base_height = (-10.0, {"target_height": 0.78})
        feet_gait = (
            0.5,
            {
                "period": 0.8,
                "offset": [0.0, 0.5],
                "threshold": 0.55,
                "body_names": (".*ankle_roll.*"),
            },
        )
        feet_slide = (-0.2, {"body_names": (".*ankle_roll.*")})
        feet_clearance = (
            1.0,
            {
                "std": math.sqrt(0.05),
                "tanh_mult": 2.0,
                "target_height": 0.1,
                "body_names": (".*ankle_roll.*"),
            },
        )
        undesired_contacts = (-1.0, {"threshold": 1, "body_names": ("(?!.*ankle.*).*")})

    rewards = BaseEnvCfg.Rewards(
        only_positive_rewards=False,
        scales=RewardsScales(),
    )

    commands = BaseEnvCfg.Commands(
        value=None,
        resample=step_funcs.resample_commands,
        heading_command=False,
        rel_standing_envs=0.02,
        ranges=BaseEnvCfg.Commands.Ranges(lin_vel_x=(-0.1, 0.1), lin_vel_y=(-0.1, 0.1), ang_vel_yaw=(-0.1, 0.1)),
        limit_ranges=BaseEnvCfg.Commands.Ranges(lin_vel_x=(-0.5, 1.0), lin_vel_y=(-0.3, 0.3), ang_vel_yaw=(-0.2, 0.2)),
    )

    curriculum = BaseEnvCfg.Curriculum(
        enabled=True,
        funcs={
            "lin_vel_cmd_levels": curr_funs.lin_vel_cmd_levels,
            #  "terrain_levels": curr_funs.terrain_levels_vel
        },
    )

    callbacks_query = {
        "contact_forces": ContactForces(history_length=3),
        "lidar_point_cloud": LidarPointCloud(enabled=False),
    }
    callbacks_setup = {
        "material_randomizer": MaterialRandomizer(
            obj_name="g1_dof29",
            static_friction_range=(0.3, 1.0),
            dynamic_friction_range=(0.3, 1.0),
            restitution_range=(0.0, 0.0),
            num_buckets=64,
        ),
        "mass_randomizer": MassRandomizer(
            obj_name="g1_dof29",
            body_names="torso_link",
            mass_distribution_params=(-1.0, 3.0),
            operation="add",
        ),
    }
    callbacks_reset = {
        "random_root_state": (
            reset_funcs.random_root_state_terrain_aware,
            {
                "pose_range": [
                    [-0.5, -0.5, 0.0, 0, 0, -3.14],  # x, y, z_offset, roll, pitch, yaw
                    [0.5, 0.5, 0.05, 0, 0, 3.14],  # z_offset can vary slightly
                ],
                "velocity_range": [[0] * 6, [0] * 6],
                # base_height_offset is None by default, uses robot's default z position (0.8m from cfg_base.py)
            },
        ),
        "reset_joints_by_scale": (
            reset_funcs.reset_joints_by_scale,
            {"position_range": (1.0, 1.0), "velocity_range": (-1.0, 1.0)},
        ),
    }
    callbacks_post_step = {
        "push_robot": (
            step_funcs.push_by_setting_velocity,
            {
                "interval_range_s": (5.0, 5.0),
                "velocity_range": [[-0.5, -0.5, 0.0], [0.5, 0.5, 0.0]],
            },
        )
    }
    callbacks_terminate = {
        "time_out": termination_funcs.time_out,
        "base_height": (
            termination_funcs.root_height_below_minimum,
            {"minimum_height": 0.2},
        ),
        "bad_orientation": (termination_funcs.bad_orientation, {"limit_angle": 0.8}),
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

    def _init_buffers(self):
        # commands + base_ang_vel + projected_gravity + dof pos/vel/prev actions
        self.num_obs = 3 + 3 + 3 + self.num_actions * 3
        # commands + base_lin_vel + base_ang_vel + projected_gravity + dof pos/vel/prev actions
        self.num_priv_obs = 3 + 3 + 3 + 3 + self.num_actions * 3
        # Rewrite SOME Hyfer-Parameters
        self.obs_clip_limit = 100.0
        self.obs_scale = torch.ones(size=(self.num_obs,), dtype=torch.float, device=self.device)
        self.priv_obs_scale = torch.ones(size=(self.num_priv_obs,), dtype=torch.float, device=self.device)
        self.obs_noise = torch.zeros(size=(self.num_obs,), dtype=torch.float, device=self.device)

        ##################### for observation scale #####################
        self.obs_scale[3:6] = 0.2  # angular velocity
        self.obs_scale[9 + self.num_actions : 9 + 2 * self.num_actions] = 0.05  # joint velocity

        ##################### for priviliged observation scale #####################
        self.priv_obs_scale[6:9] = 0.2  # angular velocity
        self.priv_obs_scale[12 + self.num_actions : 12 + 2 * self.num_actions] = 0.05  # joint velocity

        ################### for noise vector ####################
        # [0:3] -> commands
        self.obs_noise[3:6] = 0.2  # [3:6] -> base_ang_vel
        self.obs_noise[6:9] = 0.05  # projected_gravity
        self.obs_noise[9 : 9 + self.num_actions] = 0.01
        self.obs_noise[9 + self.num_actions : 9 + 2 * self.num_actions] = 1.5  # joint velocities
        return super()._init_buffers()

    def _compute_task_observations(self, env_states: TensorState):
        robot_state = env_states.robots[self.robot.name]
        base_quat = robot_state.root_state[:, 3:7]
        base_lin_vel = quat_rotate_inverse(base_quat, robot_state.root_state[:, 7:10])
        base_ang_vel = quat_rotate_inverse(base_quat, robot_state.root_state[:, 10:13])
        projected_gravity = quat_rotate_inverse(base_quat, self.gravity_vec)

        q = env_states.robots[self.name].joint_pos - self.default_dof_pos
        dq = env_states.robots[self.name].joint_vel - self.default_dof_vel

        # gait = self._gait_phase()

        obs_buf = torch.cat(
            (
                self.commands_manager.value,  # 3
                base_ang_vel,  # 3
                projected_gravity,  # 3
                q,  # |A|
                dq,  # |A|
                self.actions,  # |A|
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
                self.actions,  # |A|
                # gait
            ),
            dim=-1,
        )

        obs_buf += (2 * torch.rand_like(obs_buf) - 1) * self.obs_noise

        # clip observations -> scale observations
        obs_buf = obs_buf.clip(-self.obs_clip_limit, self.obs_clip_limit) * self.obs_scale
        priv_obs_buf = priv_obs_buf.clip(-self.obs_clip_limit, self.obs_clip_limit) * self.priv_obs_scale

        return obs_buf, priv_obs_buf
