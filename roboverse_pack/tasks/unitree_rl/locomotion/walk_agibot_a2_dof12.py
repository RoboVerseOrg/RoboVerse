from __future__ import annotations

import copy

import torch

from metasim.scenario.lights import DomeLightCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.scenario.simulator_params import SimParamCfg
from metasim.task.registry import register_task
from metasim.types import TensorState
from metasim.utils.math import euler_xyz_from_quat, quat_rotate_inverse
from roboverse_learn.rl.unitree_rl.configs.locomotion.walk_agibot_a2_dof12 import (
    WalkAgibotA2Dof12EnvCfg,
    WalkAgibotA2Dof12RslRlTrainCfg,
)
from roboverse_pack.tasks.unitree_rl.base import LeggedRobotTask


@register_task(
    "unitree_rl.walk_agibot_a2_dof12",
    "agibot_a2.walk_agibot_a2_dof12",
    "walk_agibot_a2_dof12",
)
class WalkAgibotA2Dof12Task(LeggedRobotTask):
    """Registered task wrapper with scenario defaults and cfg hooks."""

    env_cfg_cls = WalkAgibotA2Dof12EnvCfg
    train_cfg_cls = WalkAgibotA2Dof12RslRlTrainCfg
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
        i = 0

        def take(n: int) -> slice:
            nonlocal i
            s = slice(i, i + n)
            i += n
            return s

        s_sin = take(1)
        s_cos = take(1)
        s_cmd = take(3)  # [lin_x, lin_y, yaw]
        s_cmd_lin = slice(s_cmd.start, s_cmd.start + 2)  # only use the first two dimensions for scaling
        s_dof_pos = take(self.num_actions)
        s_dof_vel = take(self.num_actions)
        s_prev_act = take(self.num_actions)
        s_base_ang = take(3)
        s_base_euler = take(3)

        self.num_obs_single = i  # should be 47

        s_base_lin = take(3)

        self.num_priv_obs_single = i  # should be 50

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
