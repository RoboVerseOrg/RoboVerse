import torch

from metasim.constants import TaskType
from metasim.utils import configclass
from metasim.utils.math import normalize, quat_mul

from ..base_task_cfg import BaseTaskCfg
from .task_env.ant_isaacgymenv import AntIsaacGymEnv


@configclass
class AntIsaacGymCfg(BaseTaskCfg):
    episode_length = 100
    objects = []
    traj_filepath = None
    task_type = TaskType.LOCOMOTION
    task_env = AntIsaacGymEnv

    dt = 0.0166
    initial_potential =  torch.tensor([-1000./dt], device=device, dtype=torch.float, requires_grad=False)
    potential = None

    def reward_fn(
        self,
        states,
        up_weight=0.1,
        heading_weight=0.5,
        actions_cost_scale=0.005,
        energy_cost_scale=0.05,
        joints_at_limit_cost_scale=0.1,
        termination_height=0.31,
        death_cost=-2.0,
        dof_velocity_scale=0.2,
        contact_force_scale=0.1,
    ):
        # Handle both multi-env (IsaacGym) and single-env (Mujoco) formats
        rewards = []
        for env_state in states:
            # Get ant states
            ant_state = env_state["ant"]

            pos = ant_state["pos"]
            rot = ant_state["rot"]
            vel = ant_state["vel"]
            ang_vel = ant_state["ang_vel"]

            joint_pos = torch.tensor([v for v in ant_state["dof_pos"].values()])
            joint_vel = torch.tensor([v for v in ant_state["dof_vel"].values()])
            if len(rot) == 4:
                up_z = 1.0 - 2.0 * (rot[1] ** 2 + rot[2] ** 2)
            else:
                up_z = rot[2]

            up_reward = torch.zeros(1, dtype=torch.float32)
            if up_z > 0.93:
                up_reward = torch.tensor([up_weight], dtype=torch.float32)

            vel_x = vel[0]
            heading_reward = heading_weight * vel_x / 0.8 if vel_x < 0.8 else heading_weight

            actions_cost = torch.sum(joint_vel**2)

            electricity_cost = torch.sum(torch.abs(joint_vel * joint_pos))

            joint_names = list(ant_state["dof_pos"].keys())
            joint_limits = {
                "hip_1": (-0.6981, 0.6981),
                "ankle_1": (0.5236, 1.7453),
                "hip_2": (-0.6981, 0.6981),
                "ankle_2": (-1.7453, -0.5236),
                "hip_3": (-0.6981, 0.6981),
                "ankle_3": (-1.7453, -0.5236),
                "hip_4": (-0.6981, 0.6981),
                "ankle_4": (0.5236, 1.7453),
            }

            joints_at_limit = torch.tensor([float(abs(v) > 0.99) for v in ant_state["dof_pos"].values()])
            dof_at_limit_cost = torch.sum(joints_at_limit)

            alive_reward = torch.tensor([0.5], dtype=torch.float32)

            height = pos[2]
            total_reward = (
                alive_reward
                + up_reward
                + heading_reward
                - actions_cost_scale * actions_cost
                - energy_cost_scale * electricity_cost
                - joints_at_limit_cost_scale * dof_at_limit_cost
            )

            if height < termination_height:
                total_reward = torch.tensor([death_cost], dtype=torch.float32)

            rewards.append(total_reward)

        return torch.cat(rewards)

    observation_space = {}

    @torch.jit.script
    def _compute_heading_and_up(
        torso_rotation, inv_start_rot, to_target, vec0, vec1, up_idx
    ):
        # type: (Tensor, Tensor, Tensor, Tensor, Tensor, int) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]
        num_envs = torso_rotation.shape[0]
        target_dirs = normalize(to_target)

        torso_quat = quat_mul(torso_rotation, inv_start_rot)
        up_vec = get_basis_vector(torso_quat, vec1).view(num_envs, 3)
        heading_vec = get_basis_vector(torso_quat, vec0).view(num_envs, 3)
        up_proj = up_vec[:, up_idx]
        heading_proj = torch.bmm(heading_vec.view(
            num_envs, 1, 3), target_dirs.view(num_envs, 3, 1)).view(num_envs)

        return torso_quat, up_proj, heading_proj, up_vec, heading_vec
