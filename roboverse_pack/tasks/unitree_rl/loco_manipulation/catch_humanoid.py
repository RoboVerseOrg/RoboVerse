import torch

from metasim.types import TensorState
from metasim.utils.math import quat_rotate_inverse
from roboverse_pack.tasks.unitree_rl.base import LeggedRobotTask


class CatchHumanoidTask(LeggedRobotTask):
    """Humanoid locomotion-manipulation task for catching an object."""

    def _init_buffers(self):
        self.noise_scale_vec = self._get_noise_scale_vec()
        return super()._init_buffers()

    def _get_noise_scale_vec(self):
        noise_vec = torch.zeros(size=(47,), dtype=torch.float, device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.ang_vel * noise_level * self.cfg.normalization.obs_scales.ang_vel
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:9] = 0.0  # commands
        noise_vec[9 : 9 + self.num_actions] = (
            noise_scales.dof_pos * noise_level * self.cfg.normalization.obs_scales.dof_pos
        )
        noise_vec[9 + self.num_actions : 9 + 2 * self.num_actions] = (
            noise_scales.dof_vel * noise_level * self.cfg.normalization.obs_scales.dof_vel
        )
        noise_vec[9 + 2 * self.num_actions : 9 + 3 * self.num_actions] = 0.0  # previous actions
        noise_vec[9 + 3 * self.num_actions : 9 + 3 * self.num_actions + 2] = 0.0  # sin/cos phase

        return noise_vec

    def _compute_task_observations(self, env_states: TensorState):
        robot_state = env_states.robots[self.name]
        base_quat = robot_state.root_state[:, 3:7]
        base_lin_vel = quat_rotate_inverse(base_quat, robot_state.root_state[:, 7:10])
        base_ang_vel = quat_rotate_inverse(base_quat, robot_state.root_state[:, 10:13])
        projected_gravity = quat_rotate_inverse(base_quat, self.gravity_vec)

        q = (env_states.robots[self.name].joint_pos - self.default_dof_pos) * self.cfg.normalization.obs_scales.dof_pos
        dq = env_states.robots[self.name].joint_vel * self.cfg.normalization.obs_scales.dof_vel

        obs_buf = torch.cat(
            (
                base_ang_vel * self.cfg.normalization.obs_scales.ang_vel,  # 3
                projected_gravity,  # 3
                self.commands[:, :3] * self.commands_scale,  # 3
                q,  # num_actions
                dq,  # num_actions
                self.actions,  # num_actions
                # self.history_buffer['actions'][-1]  # num_actions
            ),
            dim=-1,
        )

        # add noise if needed
        if self.add_noise:
            obs_buf += (2 * torch.rand_like(obs_buf) - 1) * self.noise_scale_vec

        priv_obs_buf = torch.cat(
            (
                base_lin_vel * self.cfg.normalization.obs_scales.lin_vel,
                base_ang_vel * self.cfg.normalization.obs_scales.ang_vel,
                projected_gravity,
                self.commands[:, :3] * self.commands_scale,
                q,  # num_actions
                dq,  # num_actions
                self.actions,
            ),
            dim=-1,
        )

        return obs_buf, priv_obs_buf
