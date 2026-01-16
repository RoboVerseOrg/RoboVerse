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
from metasim.utils.math import euler_xyz_from_quat, quat_rotate_inverse
from roboverse_pack.callback_funcs.humanoid import (
    reset_funcs,
    reward_funcs,
    step_funcs,
    termination_funcs,
)
from roboverse_pack.randomization.humanoid import (
    MassRandomizer,
    MaterialRandomizer,
)
from roboverse_pack.tasks.humanoid.base import LeggedRobotTask
from roboverse_pack.tasks.humanoid.cfg_base import BaseEnvCfg
from roboverse_pack.utils.curriculum_utils import lin_vel_cmd_levels
from roboverse_pack.utils.humanoid_utils import Indexer


@configclass
class WalkAgibotA2Dof12EnvCfg(BaseEnvCfg):
    """Configuration for the 12-DOF A2 walking task."""

    episode_length_s = 20.0
    obs_len_history = 1
    priv_obs_len_history = 1

    control = BaseEnvCfg.Control(
        action_scale=0.5,
        action_clip=100,
        soft_joint_pos_limit_factor=0.95,
        decimation=5,
    )

    @configclass
    class RewardsScales:
        """Reward weights for gait stability and efficiency."""

        track_lin_vel_xy = (2.0, {"std": math.sqrt(0.25)})
        track_ang_vel_z = (1.0, {"std": math.sqrt(0.25)})
        lin_vel_z = -2.0
        ang_vel_xy = -0.05
        flat_orientation = -1.0
        base_height = (-100.0, {"target_height": 0.98})
        joint_acc = -1e-7
        joint_vel = -0.001
        action_rate = -0.2
        joint_pos_limits = -5.0
        joint_effort_limits = (-1e-5, {"soft_limit_factor": 0.95})
        is_alive = 0.3
        joint_deviation_legs = (
            -1.0,
            {"joint_names": (".*_hip_roll.*", ".*_hip_yaw.*", ".*_toe_roll.*")},
            reward_funcs.joint_deviation_l1,
        )
        feet_slide = (-0.2, {"body_names": (".*toe_roll.*")})
        feet_clearance = (
            1.0,
            {
                "std": math.sqrt(0.05),
                "tanh_mult": 2.0,
                "target_height": 0.18,
                "body_names": (".*toe_roll.*"),
            },
        )
        feet_gait = (
            0.18,
            {
                "period": 1.0,
                "offset": [0.0, 0.5],
                "threshold": 0.55,
                "body_names": (".*toe_roll.*"),
            },
        )
        # energy = -1e-5
        ########################

    rewards = BaseEnvCfg.Rewards(scales=RewardsScales(), only_positive_rewards=True)

    commands = BaseEnvCfg.Commands(
        value=None,
        resample=step_funcs.resample_commands,
        heading_command=True,
        resampling_time=10.0,
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

    curriculum = BaseEnvCfg.Curriculum(
        enabled=False,
        funcs={
            "lin_vel_cmd_levels": lin_vel_cmd_levels,
        },
    )

    callbacks_query = {"contact_forces": ContactForces(history_length=3)}
    callbacks_setup = {
        "material_randomizer": MaterialRandomizer(
            obj_name="agibot_a2_dof12",
            static_friction_range=(0.1, 1.25),
            dynamic_friction_range=(0.1, 1.25),
            restitution_range=(0.0, 0.0),
            num_buckets=64,
        ),
        "mass_randomizer": MassRandomizer(
            obj_name="agibot_a2_dof12",
            body_names="base_link",
            mass_distribution_params=(-1.0, 3.0),
            operation="add",
        ),
    }
    callbacks_reset = {
        "random_root_state": (
            reset_funcs.random_root_state,
            {
                "pose_range": [
                    [0.0, 0.0, 0, 0, 0, 0],
                    [0.0, 0.0, 0, 0, 0, 0],
                ],
                "velocity_range": [[-0.5] * 6, [0.5] * 6],
            },
        ),
        "reset_joints_by_scale": (
            reset_funcs.reset_joints_by_scale,
            {"position_range": (0.5, 1.5), "velocity_range": (1.0, 1.0)},
        ),
    }
    callbacks_post_step = {
        "push_robot": (
            step_funcs.push_by_setting_velocity,
            {
                "interval_range_s": (5.0, 5.0),
                "velocity_range": [[-1.5, -1.5, 0.0], [1.5, 1.5, 0.0]],
            },
        )
    }
    callbacks_terminate = {
        "time_out": termination_funcs.time_out,
        "undesired_contact": (
            termination_funcs.undesired_contact,
            {
                # TODO(zhangyi): add more undesired contact names here
                "contact_names": [
                    ".*base_link.*",
                ],
                "limit_range": 1.0,
            },
        ),
        "bad_orientation": (termination_funcs.bad_orientation, {"limit_angle": 0.8}),
        "root_height_below_minimum": (termination_funcs.root_height_below_minimum, {"minimum_height": 0.7}),
    }


@register_task(
    "agibot_a2.walk_agibot_a2_dof12",
    "walk_agibot_a2_dof12",
)
class WalkAgibotA2Dof12Task(LeggedRobotTask):
    """Registered task wrapper with scenario defaults and cfg hooks."""

    env_cfg_cls = WalkAgibotA2Dof12EnvCfg
    task_name = "walk_agibot_a2_dof12"

    scenario = ScenarioCfg(
        robots=["agibot_a2_dof12"],
        objects=[],
        cameras=[],
        num_envs=128,
        simulator="isaacgym",
        headless=True,
        env_spacing=2.5,
        decimation=1,
        sim_params=SimParamCfg(
            dt=0.002,
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
        env_cfg: WalkAgibotA2Dof12EnvCfg | None = None,
    ) -> None:
        scenario_copy = copy.deepcopy(scenario or type(self).scenario)
        scenario_copy.__post_init__()

        if env_cfg is None:
            env_cfg = type(self).env_cfg_cls()

        if device is None:
            device = "cpu" if scenario_copy.simulator == "mujoco" else ("cuda" if torch.cuda.is_available() else "cpu")

        super().__init__(scenario=scenario_copy, config=env_cfg, device=device)

    def _init_buffers(self):
        # ---------- obs slice ----------
        indexer = Indexer()

        s_sin = indexer.take(1)
        s_cos = indexer.take(1)
        s_cmd = indexer.take(3)  # [lin_x, lin_y, yaw]
        s_cmd_lin = slice(s_cmd.start, s_cmd.start + 2)  # only use the first two dimensions for scaling
        s_dof_pos = indexer.take(self.num_actions)
        s_dof_vel = indexer.take(self.num_actions)
        s_prev_act = indexer.take(self.num_actions)
        s_base_ang = indexer.take(3)
        s_base_euler = indexer.take(3)

        self.num_obs_single = indexer.i  # should be 47

        s_base_lin = indexer.take(3)

        self.num_priv_obs_single = indexer.i  # should be 50

        # ---------- init buffer ----------
        self.obs_clip_limit = 100.0
        self.obs_scale = torch.ones(self.num_obs_single, dtype=torch.float, device=self.device)
        self.priv_obs_scale = torch.ones(self.num_priv_obs_single, dtype=torch.float, device=self.device)
        self.obs_noise = torch.zeros(self.num_obs_single, dtype=torch.float, device=self.device)

        ####### for observation scale #######
        self.obs_scale[s_cmd_lin] = 2.0
        self.obs_scale[s_dof_vel] = 0.05

        ####### for priviliged observation scale #######
        self.priv_obs_scale[s_cmd_lin] = 2.0
        self.priv_obs_scale[s_dof_vel] = 0.05
        self.priv_obs_scale[s_base_lin] = 2.0

        ####### for observation noise #######
        self.obs_noise[s_dof_pos] = 0.02
        self.obs_noise[s_dof_vel] = 1.5
        self.obs_noise[s_base_ang] = 0.2
        self.obs_noise[s_base_euler] = 0.05

        return super()._init_buffers()

    def gait_phase(self, period: float = 0.8) -> torch.Tensor:
        """Compute gait phase based on episode length buffer."""
        global_phase = (self._episode_steps * self.step_dt) % period / period

        phase = torch.zeros(self.num_envs, 2, device=self.device)
        phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
        phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
        return phase

    def _compute_task_observations(self, env_states: TensorState):
        robot_state = env_states.robots[self.robot.name]
        base_quat = robot_state.root_state[:, 3:7]
        base_lin_vel = quat_rotate_inverse(base_quat, robot_state.root_state[:, 7:10])
        base_ang_vel = quat_rotate_inverse(base_quat, robot_state.root_state[:, 10:13])
        roll, pitch, yaw = euler_xyz_from_quat(base_quat)
        base_euler_xyz = torch.stack([roll, pitch, yaw], dim=-1)

        gait_phase = self.gait_phase()

        q = env_states.robots[self.robot.name].joint_pos - self.default_dof_pos
        dq = env_states.robots[self.robot.name].joint_vel - self.default_dof_vel
        prev_act = self.actions

        obs_buf = torch.cat(
            (
                gait_phase,  # 2
                self.commands_manager.value,  # 3
                q,  # num_actions
                dq,  # num_actions
                prev_act,  # num_actions
                base_ang_vel,  # 3
                base_euler_xyz,  # 3
            ),
            dim=-1,
        )

        priv_obs_buf = torch.cat(
            (
                gait_phase,  # 2
                self.commands_manager.value,  # 3
                q,  # num_actions
                dq,  # num_actions
                prev_act,  # num_actions
                base_ang_vel,  # 3
                base_euler_xyz,  # 3
                base_lin_vel,  # 3
            ),
            dim=-1,
        )

        obs_buf += (2 * torch.rand_like(obs_buf) - 1) * self.obs_noise

        # clip observations -> scale observations
        obs_buf = obs_buf.clip(-self.obs_clip_limit, self.obs_clip_limit) * self.obs_scale
        priv_obs_buf = priv_obs_buf.clip(-self.obs_clip_limit, self.obs_clip_limit) * self.priv_obs_scale

        return obs_buf, priv_obs_buf
