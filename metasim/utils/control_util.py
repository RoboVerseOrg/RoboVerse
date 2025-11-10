import torch

from metasim.task.base import BaseTaskEnv


def get_action(task_env: BaseTaskEnv, action: torch.Tensor) -> torch.Tensor:
    """Compute effort from actions using PD control."""
    # Scale the actions
    action_scaled = task_env.action_scale * action

    if task_env.manual_pd_on:  # manual PD control
        # Get current joint positions and velocities
        env_states = task_env.handler.get_states()
        sorted_dof_pos = env_states.robots[task_env.robot.name].joint_pos
        sorted_dof_vel = env_states.robots[task_env.robot.name].joint_vel

        # Compute PD control effort
        default_pos = task_env.default_dof_pos
        if isinstance(default_pos, dict):
            default_pos = torch.tensor(
                [default_pos[name] for name in task_env.sorted_joint_names], dtype=torch.float32, device=task_env.device
            )
        elif not isinstance(default_pos, torch.Tensor):
            default_pos = torch.tensor(default_pos, dtype=torch.float32, device=task_env.device)
        default_pos = default_pos.to(task_env.device)
        if default_pos.dim() == 1:
            default_pos = default_pos.unsqueeze(0).repeat(task_env.num_envs, 1)
        if task_env.action_offset:
            effort = (
                task_env.p_gains * (action_scaled + default_pos - sorted_dof_pos) - task_env.d_gains * sorted_dof_vel
            )
        else:
            effort = task_env.p_gains * (action_scaled - sorted_dof_pos) - task_env.d_gains * sorted_dof_vel

        # Apply torque limits
        effort = torch.clip(effort, -task_env.torque_limits, task_env.torque_limits)
        send_action = effort

    else:  # direct control
        send_action = action_scaled
    return send_action.to(torch.float32)
