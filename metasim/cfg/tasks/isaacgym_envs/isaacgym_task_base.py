from __future__ import annotations

import torch


class IsaacGymTaskBase:
    def reset_robot_state(self, handler, env_ids: list[int] | None = None):
        if not hasattr(handler, "gym") or not hasattr(handler, "sim"):
            return

        if env_ids is None:
            env_ids = list(range(handler.num_envs))

        if len(env_ids) == 0:
            return

        robot = handler.scenario.robots[0]

        if hasattr(handler, "_root_states") and hasattr(handler, "_dof_states"):
            env_ids_int32 = torch.tensor(env_ids, dtype=torch.int32, device=handler.device)

            num_objects = len(handler.scenario.objects)
            robot_actor_idx = num_objects

            if hasattr(robot, "default_position") and hasattr(robot, "default_orientation"):
                root_states = handler._root_states.view(handler.num_envs, -1, 13)

                for env_id in env_ids:
                    root_states[env_id, robot_actor_idx, :3] = torch.tensor(
                        robot.default_position, device=handler.device, dtype=torch.float32
                    )

                    quat = robot.default_orientation
                    root_states[env_id, robot_actor_idx, 3:7] = torch.tensor(
                        [quat[1], quat[2], quat[3], quat[0]], device=handler.device, dtype=torch.float32
                    )

                    root_states[env_id, robot_actor_idx, 7:13] = 0.0

                try:
                    from isaacgym import gymtorch

                    handler.gym.set_actor_root_state_tensor_indexed(
                        handler.sim,
                        gymtorch.unwrap_tensor(handler._root_states),
                        gymtorch.unwrap_tensor(env_ids_int32),
                        len(env_ids),
                    )
                except ImportError:
                    pass

            if hasattr(robot, "default_joint_positions"):
                dof_states = handler._dof_states.view(handler.num_envs, -1, 2)

                obj_num_dof = sum(getattr(obj, "num_dof", 0) for obj in handler.scenario.objects)
                robot_dof_start = obj_num_dof
                robot_dof_end = robot_dof_start + robot.num_joints

                joint_names = list(robot.default_joint_positions.keys())
                default_positions = torch.tensor(
                    [robot.default_joint_positions[name] for name in joint_names],
                    device=handler.device,
                    dtype=torch.float32,
                )

                for env_id in env_ids:
                    dof_states[env_id, robot_dof_start:robot_dof_end, 0] = default_positions
                    dof_states[env_id, robot_dof_start:robot_dof_end, 1] = 0.0

                try:
                    from isaacgym import gymtorch

                    handler.gym.set_dof_state_tensor_indexed(
                        handler.sim,
                        gymtorch.unwrap_tensor(handler._dof_states),
                        gymtorch.unwrap_tensor(env_ids_int32),
                        len(env_ids),
                    )
                except ImportError:
                    pass
