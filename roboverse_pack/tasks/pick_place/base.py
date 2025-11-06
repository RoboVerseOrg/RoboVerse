"""Pick and place task base classes."""

from __future__ import annotations

import torch
from loguru import logger as log

from metasim.constants import PhysicStateType
from metasim.scenario.objects import PrimitiveCubeCfg, RigidObjCfg
from metasim.scenario.scenario import ScenarioCfg, SimParamCfg
from metasim.task.registry import register_task
from metasim.task.rl_task import RLTaskEnv
from metasim.utils.math import matrix_from_quat

# Default configuration as a global dictionary
DEFAULT_CONFIG = {
    "action_scale": 0.04,
    "reward_config": {
        "scales": {
            "gripper_approach": 2.0,
            "gripper_close": 0.4,
            "robot_target_qpos": 0.1,
            "tracking_approach": 4.0,
            "tracking_progress": 150.0,
            "rotation_tracking": 2.0,
        }
    },
    # Trajectory tracking settings
    "trajectory_tracking": {
        "num_waypoints": 5,
        "reach_threshold": 0.05,
        "grasp_check_distance": 0.02,
        "enable_rotation_tracking": False,
        "rotation_error_threshold": 0.2,
    },
    # Randomization settings
    "randomization": {
        "box_pos_range": 0.1,
        "robot_pos_noise": 0.0,
        "joint_noise_range": 0.05,
    },
}


class PickPlaceBase(RLTaskEnv):
    """Abstract base class for pick and place tasks.

    Reward shaping and task design adapted from DeepMind's Mujoco Playground
    (Apache 2.0 License), re-implemented for RoboVerse.
    """

    def __init__(self, scenario, device=None):
        self.robot_name = self.scenario.robots[0].name
        self._last_action = None
        self._action_scale = DEFAULT_CONFIG.get("action_scale", 0.04)
        self.num_envs = scenario.num_envs

        self._pre_init_trajectory_tracking(scenario, device)
        self._complete_trajectory_tracking_init(device)
        super().__init__(scenario, device)

        self.reward_functions = [
            self._reward_gripper_approach,
            self._reward_gripper_close,
            self._reward_robot_target_qpos,
            self._reward_trajectory_tracking,
            self._reward_rotation_tracking,
        ]
        self.reward_weights = [
            DEFAULT_CONFIG["reward_config"]["scales"]["gripper_approach"],
            DEFAULT_CONFIG["reward_config"]["scales"]["gripper_close"],
            DEFAULT_CONFIG["reward_config"]["scales"]["robot_target_qpos"],
            1.0,
            1.0,  # rotation_tracking weight is already applied inside the function
        ]

    def _get_initial_states(self) -> list[dict] | None:
        """Get initial states for all environments. Must be implemented by subclasses."""
        pass

    def _pre_init_trajectory_tracking(self, scenario, device):
        """Pre-initialize trajectory tracking (before super().__init__())."""
        traj_config = DEFAULT_CONFIG["trajectory_tracking"]

        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self._traj_device = torch.device(device)
        self._traj_num_envs = scenario.num_envs

        self.num_waypoints = traj_config["num_waypoints"]
        self.reach_threshold = traj_config["reach_threshold"]
        self.grasp_check_distance = traj_config["grasp_check_distance"]
        self.enable_rotation_tracking = traj_config.get("enable_rotation_tracking", False)
        self.rotation_error_threshold = traj_config.get("rotation_error_threshold", 0.1)

        self.current_waypoint_idx = torch.zeros(self._traj_num_envs, dtype=torch.long, device=self._traj_device)
        self.waypoints_reached = torch.zeros(
            self._traj_num_envs, self.num_waypoints, dtype=torch.bool, device=self._traj_device
        )
        self.prev_distance_to_waypoint = torch.zeros(self._traj_num_envs, device=self._traj_device)

        self.object_grasped = torch.zeros(self._traj_num_envs, dtype=torch.bool, device=self._traj_device)

        self.w_tracking_approach = DEFAULT_CONFIG["reward_config"]["scales"]["tracking_approach"]
        self.w_tracking_progress = DEFAULT_CONFIG["reward_config"]["scales"]["tracking_progress"]
        self.w_rotation_tracking = DEFAULT_CONFIG["reward_config"]["scales"]["rotation_tracking"]

    def _complete_trajectory_tracking_init(self, device):
        """Complete trajectory tracking initialization (after super().__init__())."""
        initial_states_list = self._get_initial_states()
        if initial_states_list is None or len(initial_states_list) == 0:
            raise ValueError("No initial states found")

        first_env_state = initial_states_list[0]
        waypoint_positions = []
        waypoint_rotations = []

        for i in range(self.num_waypoints):
            marker_name = f"traj_marker_{i}"
            if marker_name in first_env_state["objects"]:
                pos = first_env_state["objects"][marker_name]["pos"]
                rot = first_env_state["objects"][marker_name]["rot"]
                waypoint_positions.append(pos)
                waypoint_rotations.append(rot)
            else:
                raise ValueError(f"Marker {marker_name} not found in initial states")

        self.waypoint_positions = torch.stack(waypoint_positions).to(device)
        self.waypoint_rotations = torch.stack(waypoint_rotations).to(device)

    def _prepare_states(self, states, env_ids):
        """Preprocess initial states, randomizing positions within specified ranges.

        Only handles generic objects (object, markers) and robot state.
        Specific objects (wall, window, cup, table) should be handled by subclasses if needed.
        """
        from copy import deepcopy

        states = deepcopy(states)

        rand_config = DEFAULT_CONFIG["randomization"]

        initial_states_list = self._get_initial_states()
        box_center = initial_states_list[0]["objects"]["object"]["pos"]
        if not isinstance(box_center, torch.Tensor):
            box_center = torch.tensor(box_center, device=self.device)
        else:
            box_center = box_center.to(self.device)

        box_pos_range_val = rand_config["box_pos_range"]
        box_pos_range = torch.tensor(
            [
                [box_center[0] - box_pos_range_val, box_center[1] - box_pos_range_val, box_center[2]],
                [box_center[0] + box_pos_range_val, box_center[1] + box_pos_range_val, box_center[2]],
            ],
            device=self.device,
        )

        box_pos = (
            torch.rand(self.num_envs, 3, device=self.device) * (box_pos_range[1] - box_pos_range[0]) + box_pos_range[0]
        )
        box_quat = states.objects["object"].root_state[:, 3:7].clone()
        zero_vel = torch.zeros(self.num_envs, 3, device=self.device)
        zero_ang_vel = torch.zeros(self.num_envs, 3, device=self.device)
        states.objects["object"].root_state = torch.cat([box_pos, box_quat, zero_vel, zero_ang_vel], dim=-1)

        # Handle trajectory markers
        for i in range(self.num_waypoints):
            marker_name = f"traj_marker_{i}"
            if marker_name in states.objects:
                marker_pos = self.waypoint_positions[i].unsqueeze(0).expand(self.num_envs, -1)
                marker_quat = self.waypoint_rotations[i].unsqueeze(0).expand(self.num_envs, -1)
                states.objects[marker_name].root_state = torch.cat(
                    [marker_pos, marker_quat, zero_vel, zero_ang_vel], dim=-1
                )

        # Handle robot state
        robot_pos = states.robots[self.robot_name].root_state[:, 0:3].clone()
        robot_pos_noise_val = rand_config["robot_pos_noise"]
        robot_pos_noise = (torch.rand(self.num_envs, 3, device=self.device) - 0.5) * robot_pos_noise_val
        robot_pos_new = robot_pos + robot_pos_noise
        robot_quat = states.robots[self.robot_name].root_state[:, 3:7].clone()
        robot_vel = states.robots[self.robot_name].root_state[:, 7:].clone()
        states.robots[self.robot_name].root_state = torch.cat([robot_pos_new, robot_quat, robot_vel], dim=-1)

        robot_joint_pos = states.robots[self.robot_name].joint_pos.clone()
        joint_noise_range = rand_config["joint_noise_range"]
        joint_noise = (torch.rand_like(robot_joint_pos, device=self.device) - 0.5) * 2 * joint_noise_range
        robot_joint_pos_new = robot_joint_pos + joint_noise
        robot_joint_pos_new[:, 0] = torch.clamp(robot_joint_pos_new[:, 0], 0.0, 0.04)
        robot_joint_pos_new[:, 1] = torch.clamp(robot_joint_pos_new[:, 1], 0.0, 0.04)
        robot_joint_pos_new[:, 2:] = torch.clamp(robot_joint_pos_new[:, 2:], -2.8973, 2.8973)
        states.robots[self.robot_name].joint_pos = robot_joint_pos_new

        return states

    def reset(self, env_ids=None):
        """Reset environment and last actions."""
        if env_ids is None or self._last_action is None:
            self._last_action = self._initial_states.robots[self.robot_name].joint_pos[:, :]
        else:
            self._last_action[env_ids] = self._initial_states.robots[self.robot_name].joint_pos[env_ids, :]

        if env_ids is None:
            env_ids_tensor = torch.arange(self.num_envs, device=self.device)
        else:
            env_ids_tensor = (
                torch.tensor(env_ids, device=self.device) if not isinstance(env_ids, torch.Tensor) else env_ids
            )

        self.current_waypoint_idx[env_ids_tensor] = 0
        self.waypoints_reached[env_ids_tensor] = False
        self.prev_distance_to_waypoint[env_ids_tensor] = 0.0
        self.object_grasped[env_ids_tensor] = False

        obs, info = super().reset(env_ids=env_ids)

        states = self.handler.get_states()

        if env_ids is None:
            env_ids_list = list(range(self.num_envs))
        else:
            env_ids_list = env_ids if isinstance(env_ids, list) else list(env_ids)

        ee_pos, _ = self._get_ee_state(states)
        target_pos = self.waypoint_positions[0].unsqueeze(0).expand(len(env_ids_list), -1)
        self.prev_distance_to_waypoint[env_ids_list] = torch.norm(ee_pos[env_ids_list] - target_pos, dim=-1)

        return obs, info

    def step(self, actions):
        """Step with delta control."""
        current_states = self.handler.get_states(mode="tensor")
        box_pos = current_states.objects["object"].root_state[:, 0:3]
        gripper_pos, _ = self._get_ee_state(current_states)
        gripper_box_dist = torch.norm(box_pos - gripper_pos, dim=-1)

        delta_actions = actions * self._action_scale
        new_actions = self._last_action + delta_actions
        real_actions = torch.maximum(torch.minimum(new_actions, self._action_high), self._action_low)

        distance_threshold = 0.02
        gripper_values = torch.where(
            gripper_box_dist > distance_threshold,
            torch.tensor(0.04, device=self.device, dtype=real_actions.dtype),
            torch.tensor(0.0, device=self.device, dtype=real_actions.dtype),
        )

        real_actions[:, 0] = gripper_values
        real_actions[:, 1] = gripper_values

        obs, reward, terminated, time_out, info = super().step(real_actions)
        self._last_action = real_actions

        updated_states = self.handler.get_states(mode="tensor")
        updated_box_pos = updated_states.objects["object"].root_state[:, 0:3]
        updated_gripper_pos, _ = self._get_ee_state(updated_states)

        gripper_joint_pos = updated_states.robots[self.robot_name].joint_pos[:, :2]
        gripper_closed = gripper_joint_pos.mean(dim=-1) < 0.02
        gripper_box_dist = torch.norm(updated_gripper_pos - updated_box_pos, dim=-1)
        is_grasping = gripper_closed & (gripper_box_dist < self.grasp_check_distance)

        old_grasped = self.object_grasped.clone()
        self.object_grasped = is_grasping

        newly_grasped = is_grasping & (~old_grasped)
        newly_released = (~is_grasping) & old_grasped

        if newly_grasped.any() and newly_grasped[0]:
            log.info(
                f"[Env 0] Object grasped! Gripper distance: {gripper_box_dist[0].item():.4f}m, "
                f"Gripper joint: {gripper_joint_pos[0].mean().item():.4f}"
            )

        if newly_released.any() and newly_released[0]:
            log.info(f"[Env 0] Object released! Gripper distance: {gripper_box_dist[0].item():.4f}m")

        step_count = getattr(self, "_debug_step_count", 0)
        self._debug_step_count = step_count + 1

        if (step_count % 100 == 0) or terminated[0] or time_out[0]:
            num_reached = self.waypoints_reached[0].sum().item()
            current_idx = self.current_waypoint_idx[0].item()
            target_pos_final = self.waypoint_positions[current_idx]
            distance = torch.norm(updated_gripper_pos[0] - target_pos_final, dim=-1).item()
            grasped = self.object_grasped[0].item()

            status = "Episode End" if (terminated[0] or time_out[0]) else f"Step {step_count}"
            log.info(
                f"[{status} - Env 0] Progress: {num_reached}/{self.num_waypoints} waypoints | "
                f"Current: #{current_idx} | Grasped: {grasped} | "
                f"Distance to target: {distance:.4f}m (threshold: {self.reach_threshold}m)"
            )

            if step_count % 100 == 0:
                log.debug(f"  Target pos: {target_pos_final.cpu().numpy()}")
                log.debug(f"  EE pos: {updated_gripper_pos[0].cpu().numpy()}")

        return obs, reward, terminated, time_out, info

    def _reward_gripper_approach(self, env_states) -> torch.Tensor:
        """Reward for gripper approaching the box."""
        box_pos = env_states.objects["object"].root_state[:, 0:3]
        gripper_pos, _ = self._get_ee_state(env_states)
        gripper_box_dist = torch.norm(box_pos - gripper_pos, dim=-1)

        approach_reward_far = 1 - torch.tanh(gripper_box_dist)
        approach_reward_near = 1 - torch.tanh(gripper_box_dist * 10)
        return approach_reward_far + approach_reward_near

    def _reward_gripper_close(self, env_states) -> torch.Tensor:
        """Reward for closing gripper when close to box."""
        box_pos = env_states.objects["object"].root_state[:, 0:3]
        gripper_pos, _ = self._get_ee_state(env_states)
        gripper_box_dist = torch.norm(box_pos - gripper_pos, dim=-1)

        close_bonus = (gripper_box_dist < 0.02).float()
        return close_bonus

    def _reward_robot_target_qpos(self, env_states) -> torch.Tensor:
        """Reward for robot staying close to target joint positions."""
        robot_joint_pos = env_states.robots[self.robot_name].joint_pos[:, 2:]
        target_joint_pos = self._initial_states.robots[self.robot_name].joint_pos[:, 2:]

        joint_error = torch.norm(robot_joint_pos - target_joint_pos, dim=-1)
        return 1 - torch.tanh(joint_error)

    def _reward_trajectory_tracking(self, env_states) -> torch.Tensor:
        """Reward for tracking waypoints (only when object is grasped)."""
        ee_pos, _ = self._get_ee_state(env_states)
        grasped_mask = self.object_grasped
        tracking_reward = torch.zeros(self.num_envs, device=self.device)

        if grasped_mask.any():
            target_pos = self.waypoint_positions[self.current_waypoint_idx]
            distance = torch.norm(ee_pos - target_pos, dim=-1)

            # Distance-based reward (far + near) / 2
            distance_reward_far = 1 - torch.tanh(1.0 * distance)
            distance_reward_near = 1 - torch.tanh(10.0 * distance)
            approach_reward = (distance_reward_far + distance_reward_near) / 2.0
            approach_reward = approach_reward * self.w_tracking_approach * grasped_mask.float()

            # Check distance condition
            distance_reached = (distance < self.reach_threshold) & grasped_mask

            # Check rotation condition if rotation tracking is enabled
            rotation_reached = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
            rot_err = None
            if self.enable_rotation_tracking:
                box_quat = env_states.objects["object"].root_state[:, 3:7]
                box_mat = matrix_from_quat(box_quat).reshape(self.num_envs, 9)
                target_quat = self.waypoint_rotations[self.current_waypoint_idx]
                target_mat = matrix_from_quat(target_quat).reshape(self.num_envs, 9)
                rot_err = torch.norm(target_mat[:, :6] - box_mat[:, :6], dim=-1)
                rotation_reached = rot_err < self.rotation_error_threshold

            # Both distance and rotation must be satisfied to consider as reached
            reached = distance_reached & rotation_reached
            newly_reached = reached & (
                ~self.waypoints_reached[torch.arange(self.num_envs, device=self.device), self.current_waypoint_idx]
            )
            progress_reward = newly_reached.float() * self.w_tracking_progress

            if newly_reached.any():
                if newly_reached[0]:
                    wp_idx = self.current_waypoint_idx[0].item()
                    rot_info = ""
                    if self.enable_rotation_tracking and rot_err is not None:
                        rot_info = f", Rotation error: {rot_err[0].item():.4f} < {self.rotation_error_threshold}"
                    log.info(
                        f"[Env 0] Reached waypoint #{wp_idx}! Distance: {distance[0].item():.4f}m < {self.reach_threshold}m{rot_info}"
                    )

                self.waypoints_reached[newly_reached, self.current_waypoint_idx[newly_reached]] = True

                # Advance to next waypoint if not at the last one
                can_advance = newly_reached & (self.current_waypoint_idx < self.num_waypoints - 1)

                if can_advance.any() and can_advance[0]:
                    old_idx = self.current_waypoint_idx[0].item()
                    new_idx = old_idx + 1
                    log.info(f"   -> Advancing to waypoint #{new_idx}")

                self.current_waypoint_idx[can_advance] += 1

                if can_advance.any():
                    new_target_pos = self.waypoint_positions[self.current_waypoint_idx[can_advance]]
                    self.prev_distance_to_waypoint[can_advance] = torch.norm(
                        ee_pos[can_advance] - new_target_pos, dim=-1
                    )

            maintain_reward = torch.zeros(self.num_envs, device=self.device)
            all_reached = self.waypoints_reached.all(dim=1)
            completed_mask = all_reached & grasped_mask

            if completed_mask.any():
                last_target_pos = self.waypoint_positions[-1].unsqueeze(0).expand(self.num_envs, -1)
                distance_to_last = torch.norm(ee_pos - last_target_pos, dim=-1)

                maintain_reward[completed_mask] = torch.where(
                    distance_to_last[completed_mask] < self.reach_threshold,
                    torch.full((completed_mask.sum(),), 5, device=self.device),
                    (1 - torch.tanh(1.0 * distance_to_last[completed_mask])) * self.w_tracking_approach,
                )

            tracking_reward = torch.where(all_reached, maintain_reward, approach_reward + progress_reward)

        return tracking_reward

    def _reward_rotation_tracking(self, env_states) -> torch.Tensor:
        """Reward for tracking target rotation at waypoints when object is grasped."""
        if not self.enable_rotation_tracking:
            return torch.zeros(self.num_envs, device=self.device)

        grasped_mask = self.object_grasped
        rotation_reward = torch.zeros(self.num_envs, device=self.device)

        if grasped_mask.any():
            box_quat = env_states.objects["object"].root_state[:, 3:7]
            box_mat = matrix_from_quat(box_quat).reshape(self.num_envs, 9)

            target_quat = self.waypoint_rotations[self.current_waypoint_idx]
            target_mat = matrix_from_quat(target_quat).reshape(self.num_envs, 9)

            rot_err = torch.norm(target_mat[:, :6] - box_mat[:, :6], dim=-1)
            rotation_reward = (1 - torch.tanh(rot_err)) * grasped_mask.float() * self.w_rotation_tracking

        return rotation_reward

    def _observation(self, env_states) -> torch.Tensor:
        """Get observation using RoboVerse tensor state."""
        box_pos = env_states.objects["object"].root_state[:, 0:3]  # [num_envs, 3]
        box_quat = env_states.objects["object"].root_state[:, 3:7]  # [num_envs, 4]

        gripper_pos, gripper_quat = self._get_ee_state(env_states)  # (B, 3), (B, 4)
        gripper_mat = matrix_from_quat(gripper_quat).view(self.num_envs, -1)  # (B, 9)
        robot_joint_pos = env_states.robots[self.robot_name].joint_pos  # [num_envs, num_joints]
        robot_joint_vel = env_states.robots[self.robot_name].joint_vel  # [num_envs, num_joints]

        # Convert quaternion to rotation matrix for box
        box_mat = matrix_from_quat(box_quat)  # [num_envs, 3, 3]
        box_mat_flat = box_mat.view(self.num_envs, -1)  # [num_envs, 9]

        # Ensure gripper_mat has correct shape [num_envs, 9]
        if gripper_mat.dim() == 3:
            gripper_mat = gripper_mat.view(self.num_envs, -1)  # Reshape to [num_envs, 9]

        box_to_gripper = box_pos - gripper_pos  # [num_envs, 3]

        target_pos = self.waypoint_positions[self.current_waypoint_idx]
        target_to_gripper = target_pos - gripper_pos
        num_reached = self.waypoints_reached.sum(dim=1, keepdim=True).float() / self.num_waypoints

        # Convert target quaternion to rotation matrix
        target_quat = self.waypoint_rotations[self.current_waypoint_idx]
        target_mat = matrix_from_quat(target_quat).reshape(self.num_envs, 9)  # [num_envs, 9]

        obs_list = [
            robot_joint_pos,
            robot_joint_vel,
            gripper_pos,
            gripper_mat[:, 3:],
            box_mat_flat[:, 3:],
            target_mat[:, 3:],
            box_to_gripper,
            target_to_gripper,
            num_reached,
        ]

        obs = torch.cat(obs_list, dim=-1)  # [num_envs, obs_dim]

        return obs

    def _get_ee_state(self, states):
        """Return EE state using site queries.

        Returns:
            ee_pos_world: (B, 3) gripper position from site
            ee_mat_world: (B, 9) gripper rotation matrix from site
        """
        robot_config = self.robot
        rs = states.robots[robot_config.name]
        device = (rs.joint_pos if isinstance(rs.joint_pos, torch.Tensor) else torch.tensor(rs.joint_pos)).device

        body_state = (
            rs.body_state
            if isinstance(rs.body_state, torch.Tensor)
            else torch.tensor(rs.body_state, device=device).float()
        )

        # Use panda_hand directly for more accurate EE position
        hand_body_index = rs.body_names.index("panda_hand")
        hand_pos = body_state[:, hand_body_index, 0:3]  # (B, 3)
        hand_quat = body_state[:, hand_body_index, 3:7]  # (B, 4) wxyz

        # Add offset from panda_hand to actual gripper center
        from metasim.utils.math import quat_apply

        offset_local = torch.tensor([0.0, 0.0, 0.1034], device=device, dtype=hand_pos.dtype)  # (3,)
        offset_world = quat_apply(hand_quat, offset_local.expand(hand_pos.shape[0], -1))  # (B, 3)

        ee_pos_world = hand_pos + offset_world  # (B, 3)
        ee_quat_world = hand_quat  # (B, 4) wxyz

        return ee_pos_world, ee_quat_world


@register_task("pick_place.base", "pick_place_base")
class PickPlaceTable(PickPlaceBase):
    """Concrete implementation of pick and place base task with table."""

    scenario = ScenarioCfg(
        objects=[
            PrimitiveCubeCfg(
                name="object",
                size=(0.04, 0.04, 0.04),
                mass=0.02,
                physics=PhysicStateType.RIGIDBODY,
                color=(1.0, 0.0, 0.0),
            ),
            PrimitiveCubeCfg(
                name="table",
                size=(0.2, 0.3, 0.4),
                mass=10.0,
                physics=PhysicStateType.RIGIDBODY,
                color=(0.8, 0.6, 0.4),
                fix_base_link=True,
            ),
            # Visualization: Trajectory waypoints (5 spheres showing trajectory path)
            RigidObjCfg(
                name="traj_marker_0",
                urdf_path="roboverse_pack/tasks/pick_place/marker/marker.urdf",
                mjcf_path="roboverse_pack/tasks/pick_place/marker/marker.xml",
                usd_path="roboverse_pack/tasks/pick_place/marker/marker.usd",
                scale=0.2,
                physics=PhysicStateType.XFORM,
                enabled_gravity=False,
                collision_enabled=False,
                fix_base_link=True,
            ),
            RigidObjCfg(
                name="traj_marker_1",
                urdf_path="roboverse_pack/tasks/pick_place/marker/marker.urdf",
                mjcf_path="roboverse_pack/tasks/pick_place/marker/marker.xml",
                usd_path="roboverse_pack/tasks/pick_place/marker/marker.usd",
                scale=0.2,
                physics=PhysicStateType.XFORM,
                enabled_gravity=False,
                collision_enabled=False,
                fix_base_link=True,
            ),
            RigidObjCfg(
                name="traj_marker_2",
                urdf_path="roboverse_pack/tasks/pick_place/marker/marker.urdf",
                mjcf_path="roboverse_pack/tasks/pick_place/marker/marker.xml",
                usd_path="roboverse_pack/tasks/pick_place/marker/marker.usd",
                scale=0.2,
                physics=PhysicStateType.XFORM,
                enabled_gravity=False,
                collision_enabled=False,
                fix_base_link=True,
            ),
            RigidObjCfg(
                name="traj_marker_3",
                urdf_path="roboverse_pack/tasks/pick_place/marker/marker.urdf",
                mjcf_path="roboverse_pack/tasks/pick_place/marker/marker.xml",
                usd_path="roboverse_pack/tasks/pick_place/marker/marker.usd",
                scale=0.2,
                physics=PhysicStateType.XFORM,
                enabled_gravity=False,
                collision_enabled=False,
                fix_base_link=True,
            ),
            RigidObjCfg(
                name="traj_marker_4",
                urdf_path="roboverse_pack/tasks/pick_place/marker/marker.urdf",
                mjcf_path="roboverse_pack/tasks/pick_place/marker/marker.xml",
                usd_path="roboverse_pack/tasks/pick_place/marker/marker.usd",
                scale=0.2,
                physics=PhysicStateType.XFORM,
                enabled_gravity=False,
                collision_enabled=False,
                fix_base_link=True,
            ),
        ],
        robots=["franka"],
        sim_params=SimParamCfg(
            dt=0.005,
        ),
        decimation=4,
    )
    max_episode_steps = 200

    def _get_initial_states(self) -> list[dict] | None:
        """Get initial states for all environments."""
        init = [
            {
                "objects": {
                    "object": {
                        "pos": torch.tensor([0.654277, -0.345737, 0.020000]),
                        "rot": torch.tensor([0.706448, -0.031607, 0.706347, 0.031698]),
                    },
                    "table": {
                        "pos": torch.tensor([0.499529, 0.253941, 0.200000]),
                        "rot": torch.tensor([0.999067, -0.000006, 0.000009, 0.043198]),
                    },
                    # Trajectory waypoints (world coordinates)
                    "traj_marker_0": {
                        "pos": torch.tensor([0.610000, -0.280000, 0.150000]),
                        "rot": torch.tensor([1.000000, 0.000000, 0.000000, 0.000000]),
                    },
                    "traj_marker_1": {
                        "pos": torch.tensor([0.600000, -0.190000, 0.220000]),
                        "rot": torch.tensor([1.000000, 0.000000, 0.000000, 0.000000]),
                    },
                    "traj_marker_2": {
                        "pos": torch.tensor([0.560000, -0.110000, 0.360000]),
                        "rot": torch.tensor([0.998750, 0.000000, 0.049979, -0.000000]),
                    },
                    "traj_marker_3": {
                        "pos": torch.tensor([0.530000, 0.010000, 0.470000]),
                        "rot": torch.tensor([1.000000, 0.000000, 0.000000, 0.000000]),
                    },
                    "traj_marker_4": {
                        "pos": torch.tensor([0.510000, 0.130000, 0.460000]),
                        "rot": torch.tensor([0.984726, 0.000000, 0.174108, -0.000000]),
                    },
                },
                "robots": {
                    "franka": {
                        "pos": torch.tensor([-0.025, -0.160, 0.018054]),
                        "rot": torch.tensor([1.000000, 0.000000, 0.000000, 0.000000]),
                        "dof_pos": {
                            "panda_finger_joint1": 0.04,
                            "panda_finger_joint2": 0.04,
                            "panda_joint1": 0.0,
                            "panda_joint2": -0.785398,
                            "panda_joint3": 0.0,
                            "panda_joint4": -2.356194,
                            "panda_joint5": 0.0,
                            "panda_joint6": 1.570796,
                            "panda_joint7": 0.785398,
                        },
                    },
                },
            }
            for _ in range(self.num_envs)
        ]

        return init
