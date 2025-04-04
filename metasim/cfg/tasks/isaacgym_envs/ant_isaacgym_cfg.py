import torch

from metasim.constants import TaskType
from metasim.utils import configclass

from ..base_task_cfg import BaseTaskCfg


@configclass
class AntIsaacGymCfg(BaseTaskCfg):
    episode_length = 100
    objects = []
    traj_filepath = None
    task_type = TaskType.LOCOMOTION

    def reward_fn(self, states):
        # Reward constants
        up_weight = 0.1
        heading_weight = 0.5
        actions_cost_scale = 0.005
        energy_cost_scale = 0.001
        joints_at_limit_cost_scale = 0.1
        termination_height = 0.3
        death_cost = -1.0

        # Handle both multi-env (IsaacGym) and single-env (Mujoco) formats
        rewards = []
        for env_state in states:
            # Get ant states
            ant_state = env_state["ant"]

            # Extract positions, rotations and velocities
            pos = ant_state["pos"]
            rot = ant_state["rot"]  # quaternion [w, x, y, z]
            vel = ant_state["vel"]
            ang_vel = ant_state["ang_vel"]

            # Joint states
            joint_pos = torch.tensor([v for v in ant_state["dof_pos"].values()])
            joint_vel = torch.tensor([v for v in ant_state["dof_vel"].values()])

            # 1. Calculate up alignment reward (z-axis alignment with world up)
            # Convert quaternion to get body's up vector
            # For a quaternion [w, x, y, z], the up vector transform is:
            # up_x = 2 * (x*z + w*y)
            # up_y = 2 * (y*z - w*x)
            # up_z = 1 - 2 * (x*x + y*y)
            if len(rot) == 4:  # quaternion
                up_z = 1.0 - 2.0 * (rot[1] ** 2 + rot[2] ** 2)
            else:  # rotation matrix or other format
                # Assuming the third row of rotation matrix represents up vector
                up_z = rot[2]

            up_reward = torch.zeros(1, dtype=torch.float32)
            if up_z > 0.93:  # ant is upright
                up_reward = torch.tensor([up_weight], dtype=torch.float32)

            # 2. Heading reward (moving forward)
            # Use x-velocity as forward motion metric
            vel_x = vel[0]
            heading_reward = heading_weight * torch.clamp(vel_x / 0.8, 0.0, 1.0)

            # 3. Action costs
            # Since we don't have the actions here, use joint velocities as proxy
            actions_cost = torch.sum(joint_vel**2)

            # 4. Electricity costs (joint velocities * positions)
            electricity_cost = torch.sum(torch.abs(joint_vel * joint_pos))

            # 5. Joint limit penalties
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

            joints_at_limit = torch.tensor([
                float(
                    abs(ant_state["dof_pos"][j] - joint_limits[j][0]) < 0.01
                    or abs(ant_state["dof_pos"][j] - joint_limits[j][1]) < 0.01
                )
                for j in joint_names
            ])
            dof_at_limit_cost = torch.sum(joints_at_limit)

            # 6. Alive reward (constant reward for being alive)
            alive_reward = torch.tensor([0.5], dtype=torch.float32)

            # Height termination
            height = pos[2]
            total_reward = (
                alive_reward
                + up_reward
                + heading_reward
                - actions_cost_scale * actions_cost
                - energy_cost_scale * electricity_cost
                - joints_at_limit_cost_scale * dof_at_limit_cost
            )

            # Death penalty
            if height < termination_height:
                total_reward = torch.tensor([death_cost], dtype=torch.float32)

            rewards.append(total_reward)

        # Sum individual rewards for each environment
        return torch.cat(rewards)
