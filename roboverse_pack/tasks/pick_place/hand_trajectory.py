"""Base class for pick and place tasks with hand trajectory tracking."""

from __future__ import annotations

import torch
from loguru import logger as log

from metasim.constants import PhysicStateType
from metasim.scenario.objects import PrimitiveCubeCfg, RigidObjCfg
from metasim.scenario.scenario import ScenarioCfg, SimParamCfg
from metasim.task.registry import register_task
from metasim.task.rl_task import RLTaskEnv
from metasim.utils.math import matrix_from_quat


class TrajectoryTrackingTaskBase(RLTaskEnv):
    """Base class for pick and place tasks with trajectory tracking.

    Subclasses should override:
    - DEFAULT_CONFIG: Task-specific configuration
    - scenario: ScenarioCfg with objects and robots
    - _get_initial_states(): Initial state positions
    - obstacle_name: Name of the static obstacle object (e.g., "wall", "table", "window")
    """

    DEFAULT_CONFIG = {
        "action_scale": 0.03,
        "reward_config": {
            "scales": {
                "hand_approach": 2.0,
                "hand_close": 0.4,
                "robot_target_qpos": 0.1,
                "tracking_approach": 4.0,
                "tracking_progress": 150.0,
            }
        },
        "trajectory_tracking": {
            "num_waypoints": 5,
            "reach_threshold": 0.10,
            "grasp_check_distance": 0.04,
        },
        "randomization": {
            "box_pos_range": 0.05,
            "robot_pos_noise": 0.0,
            "joint_noise_range": 0.05,
        },
    }

    # Name of the static obstacle object - subclasses should override
    obstacle_name = "table"

    scenario = ScenarioCfg(
        objects=[
            PrimitiveCubeCfg(
                name="object",
                size=(0.04, 0.04, 0.06),
                mass=0.02,
                physics=PhysicStateType.RIGIDBODY,
                color=(0.2, 0.2, 0.7),
            ),
            PrimitiveCubeCfg(
                name="wall",
                size=(0.8, 0.1, 0.3),
                mass=1000.0,
                physics=PhysicStateType.RIGIDBODY,
                color=(0.7, 0.7, 0.7),
            ),
            RigidObjCfg(
                name="table",
                scale=(1, 1, 1),
                physics=PhysicStateType.RIGIDBODY,
                enabled_gravity=False,
                usd_path="roboverse_data/assets/EmbodiedGenData/demo_assets/table/usd/table.usd",
                urdf_path="roboverse_data/assets/EmbodiedGenData/demo_assets/table/result/table.urdf",
                mjcf_path="roboverse_data/assets/EmbodiedGenData/demo_assets/table/mjcf/table.xml",
            ),
            # Visualization: Trajectory waypoints (5 spheres showing trajectory path)
            RigidObjCfg(
                name="traj_marker_0",
                urdf_path="roboverse_pack/tasks/pick_place/marker/marker.urdf",
                mjcf_path="roboverse_pack/tasks/pick_place/marker/marker.xml",
                usd_path="roboverse_pack/tasks/pick_place/marker/marker/marker.usd",
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
                usd_path="roboverse_pack/tasks/pick_place/marker/marker/marker.usd",
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
                usd_path="roboverse_pack/tasks/pick_place/marker/marker/marker.usd",
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
                usd_path="roboverse_pack/tasks/pick_place/marker/marker/marker.usd",
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
                usd_path="roboverse_pack/tasks/pick_place/marker/marker/marker.usd",
                scale=0.2,
                physics=PhysicStateType.XFORM,
                enabled_gravity=False,
                collision_enabled=False,
                fix_base_link=True,
            ),
        ],
        robots=["vega"],
        sim_params=SimParamCfg(
            dt=0.005,
        ),
        decimation=4,
    )
    max_episode_steps = 300

    def __init__(self, scenario, device=None):
        self.robot_name = self.scenario.robots[0].name
        self._last_action = None
        self._action_scale = self.DEFAULT_CONFIG.get("action_scale", 0.04)
        self.num_envs = scenario.num_envs

        self._pre_init_trajectory_tracking(scenario, device)
        self._complete_trajectory_tracking_init(device)
        super().__init__(scenario, device)

        # Initialize finger joint indices for left hand
        # Left hand finger joints: L_th_j0, L_th_j1, L_th_j2, L_ff_j1, L_ff_j2, L_mf_j1, L_mf_j2, L_rf_j1, L_rf_j2, L_lf_j1, L_lf_j2
        joint_names = self.handler.get_joint_names(self.robot_name, sort=True)
        self.left_hand_finger_joint_names = [
            "L_th_j0",
            "L_th_j1",
            "L_th_j2",
            "L_ff_j1",
            "L_ff_j2",
            "L_mf_j1",
            "L_mf_j2",
            "L_rf_j1",
            "L_rf_j2",
            "L_lf_j1",
            "L_lf_j2",
        ]
        self.left_hand_finger_joint_indices = [
            joint_names.index(name) for name in self.left_hand_finger_joint_names if name in joint_names
        ]
        # Get finger open/close positions from robot config (left hand finger joints)
        robot_config = self.robot
        self.hand_open_q = torch.tensor(robot_config.gripper_open_q, device=self.device)  # (11,) finger open positions
        self.hand_close_q = torch.tensor(
            robot_config.gripper_close_q, device=self.device
        )  # (11,) finger close positions

        self.reward_functions = [
            self._reward_hand_approach,
            self._reward_hand_close,
            self._reward_robot_target_qpos,
            self._reward_trajectory_tracking,
        ]
        self.reward_weights = [
            self.DEFAULT_CONFIG["reward_config"]["scales"]["hand_approach"],
            self.DEFAULT_CONFIG["reward_config"]["scales"]["hand_close"],
            self.DEFAULT_CONFIG["reward_config"]["scales"]["robot_target_qpos"],
            1.0,
        ]

    def _pre_init_trajectory_tracking(self, scenario, device):
        """Pre-initialize trajectory tracking (before super().__init__())."""
        traj_config = self.DEFAULT_CONFIG["trajectory_tracking"]

        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self._traj_device = torch.device(device)
        self._traj_num_envs = scenario.num_envs

        self.num_waypoints = traj_config["num_waypoints"]
        self.reach_threshold = traj_config["reach_threshold"]
        self.grasp_check_distance = traj_config["grasp_check_distance"]

        self.current_waypoint_idx = torch.zeros(self._traj_num_envs, dtype=torch.long, device=self._traj_device)
        self.waypoints_reached = torch.zeros(
            self._traj_num_envs, self.num_waypoints, dtype=torch.bool, device=self._traj_device
        )
        self.prev_distance_to_waypoint = torch.zeros(self._traj_num_envs, device=self._traj_device)

        self.object_grasped = torch.zeros(self._traj_num_envs, dtype=torch.bool, device=self._traj_device)

        self.w_tracking_approach = self.DEFAULT_CONFIG["reward_config"]["scales"]["tracking_approach"]
        self.w_tracking_progress = self.DEFAULT_CONFIG["reward_config"]["scales"]["tracking_progress"]

    def _complete_trajectory_tracking_init(self, device):
        """Complete trajectory tracking initialization (after super().__init__())."""
        initial_states_list = self._get_initial_states()
        if initial_states_list is None or len(initial_states_list) == 0:
            raise ValueError("No initial states found")

        first_env_state = initial_states_list[0]
        waypoint_positions = []

        for i in range(self.num_waypoints):
            marker_name = f"traj_marker_{i}"
            if marker_name in first_env_state["objects"]:
                pos = first_env_state["objects"][marker_name]["pos"]
                waypoint_positions.append(pos)
            else:
                raise ValueError(f"Marker {marker_name} not found in initial states")

        self.waypoint_positions = torch.stack(waypoint_positions).to(device)

    def _prepare_states(self, states, env_ids):
        """Preprocess initial states, randomizing positions within specified ranges."""
        from copy import deepcopy

        states = deepcopy(states)

        rand_config = self.DEFAULT_CONFIG["randomization"]

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

        # Keep obstacle in place (no randomization)
        obstacle_pos = states.objects[self.obstacle_name].root_state[:, 0:3].clone()
        obstacle_quat = states.objects[self.obstacle_name].root_state[:, 3:7].clone()
        states.objects[self.obstacle_name].root_state = torch.cat(
            [obstacle_pos, obstacle_quat, zero_vel, zero_ang_vel], dim=-1
        )

        marker_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).expand(self.num_envs, -1)

        for i in range(self.num_waypoints):
            marker_name = f"traj_marker_{i}"
            marker_pos = self.waypoint_positions[i].unsqueeze(0).expand(self.num_envs, -1)
            states.objects[marker_name].root_state = torch.cat(
                [marker_pos, marker_quat, zero_vel, zero_ang_vel], dim=-1
            )

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

        # Use hand base position for distance calculation
        hand_pos = self._get_hand_position(states)  # (B, 3)
        target_pos = self.waypoint_positions[0].unsqueeze(0).expand(len(env_ids_list), -1)
        self.prev_distance_to_waypoint[env_ids_list] = torch.norm(hand_pos[env_ids_list] - target_pos, dim=-1)

        return obs, info

    def step(self, actions):
        """Step with delta control."""
        current_states = self.handler.get_states(mode="tensor")
        box_pos = current_states.objects["object"].root_state[:, 0:3]  # (B, 3)

        # Get hand base position for distance calculation
        hand_pos = self._get_hand_position(current_states)  # (B, 3)
        # Calculate distance from hand to box
        hand_box_dist = torch.norm(hand_pos - box_pos, dim=-1)  # (B,)

        delta_actions = actions * self._action_scale
        new_actions = self._last_action + delta_actions
        real_actions = torch.maximum(torch.minimum(new_actions, self._action_high), self._action_low)

        # Auto-control finger joints based on distance to object
        # If far from object: open hand (finger open positions)
        # If close to object: close hand (finger close positions)
        distance_threshold = 0.03
        finger_targets_open = self.hand_open_q.unsqueeze(0).expand(self.num_envs, -1)  # (B, 11)
        finger_targets_close = self.hand_close_q.unsqueeze(0).expand(self.num_envs, -1)  # (B, 11)

        # Determine target finger positions based on distance
        finger_targets = torch.where(
            hand_box_dist.unsqueeze(-1) > distance_threshold,
            finger_targets_open,
            finger_targets_close,
        )  # (B, 11)

        # Set finger joint targets in actions
        for i, joint_idx in enumerate(self.left_hand_finger_joint_indices):
            if i < finger_targets.shape[1]:
                real_actions[:, joint_idx] = finger_targets[:, i]

        obs, reward, terminated, time_out, info = super().step(real_actions)
        self._last_action = real_actions

        updated_states = self.handler.get_states(mode="tensor")
        updated_box_pos = updated_states.objects["object"].root_state[:, 0:3]  # (B, 3)

        # Get updated hand position
        updated_hand_pos = self._get_hand_position(updated_states)  # (B, 3)
        updated_hand_box_dist = torch.norm(updated_hand_pos - updated_box_pos, dim=-1)  # (B,)

        # Check if fingers are closed (check finger joint positions)
        finger_joint_pos = updated_states.robots[self.robot_name].joint_pos[
            :, self.left_hand_finger_joint_indices
        ]  # (B, 11)
        denom = self.hand_close_q.unsqueeze(0) - self.hand_open_q.unsqueeze(0) + 1e-6
        finger_close_ratios = ((finger_joint_pos - self.hand_open_q.unsqueeze(0)) / denom).clamp(0.0, 1.0)  # (B, 11)
        hand_closed = finger_close_ratios.mean(dim=-1) > 0.5  # Hand is considered closed if average ratio > 0.7

        # Check if hand is close to object (using hand position instead of finger tips)
        hand_close_to_object = updated_hand_box_dist < self.grasp_check_distance  # (B,)

        # Object is grasped if hand is closed AND hand is close to object
        is_grasping = hand_closed & hand_close_to_object

        old_grasped = self.object_grasped.clone()
        self.object_grasped = is_grasping

        newly_grasped = is_grasping & (~old_grasped)
        newly_released = (~is_grasping) & old_grasped

        if newly_grasped.any() and newly_grasped[0]:
            log.info(
                f"[Env 0] Object grasped! Hand-box distance: {updated_hand_box_dist[0].item():.4f}m, "
                f"Hand closed ratio: {finger_close_ratios[0].mean().item():.4f}"
            )

        if newly_released.any() and newly_released[0]:
            log.info(f"[Env 0] Object released! Hand-box distance: {updated_hand_box_dist[0].item():.4f}m")

        step_count = getattr(self, "_debug_step_count", 0)
        self._debug_step_count = step_count + 1

        if (step_count % 100 == 0) or terminated[0] or time_out[0]:
            num_reached = self.waypoints_reached[0].sum().item()
            current_idx = self.current_waypoint_idx[0].item()
            target_pos_final = self.waypoint_positions[current_idx]
            # Use hand position for distance calculation
            hand_pos_single = updated_hand_pos[0]  # (3,)
            distance = torch.norm(hand_pos_single - target_pos_final, dim=-1).item()
            grasped = self.object_grasped[0].item()

            status = "Episode End" if (terminated[0] or time_out[0]) else f"Step {step_count}"
            log.info(
                f"[{status} - Env 0] Progress: {num_reached}/{self.num_waypoints} waypoints | "
                f"Current: #{current_idx} | Grasped: {grasped} | "
                f"Distance to target: {distance:.4f}m (threshold: {self.reach_threshold}m)"
            )

            if step_count % 100 == 0:
                log.debug(f"  Target pos: {target_pos_final.cpu().numpy()}")
                log.debug(f"  Hand pos: {hand_pos_single.cpu().numpy()}")

                # Calculate object to waypoint relative displacement (env0)
                box_pos_env0 = updated_box_pos[0]  # (3,)
                object_to_waypoint = box_pos_env0 - target_pos_final  # (3,)
                object_to_waypoint_dist = torch.norm(object_to_waypoint, dim=-1).item()

                # Print the information
                log.info(
                    f"[Step {step_count} - Env 0] Object to waypoint displacement: "
                    f"({object_to_waypoint[0].item():.4f}, {object_to_waypoint[1].item():.4f}, {object_to_waypoint[2].item():.4f}) m, "
                    f"distance: {object_to_waypoint_dist:.4f}m"
                )

        return obs, reward, terminated, time_out, info

    def _get_hand_position(self, states):
        """Get position of the palm center using fingertip offsets.

        The hand position is approximated as the midpoint between the thumb tip and
        the little finger tip. Since some tip links are removed during URDF import,
        we reconstruct their world positions from the last articulated segment using
        fixed offsets derived from the URDF (`L_th_l2` and `L_lf_l2`).

        Returns:
            hand_pos: (B, 3) tensor with position of hand base
        """
        rs = states.robots[self.robot.name]
        device = (rs.joint_pos if isinstance(rs.joint_pos, torch.Tensor) else torch.tensor(rs.joint_pos)).device

        body_state = (
            rs.body_state
            if isinstance(rs.body_state, torch.Tensor)
            else torch.tensor(rs.body_state, device=device).float()
        )

        name_to_index = {name: idx for idx, name in enumerate(rs.body_names)}
        required_links = ["L_th_l2", "L_lf_l2"]
        missing_links = [link for link in required_links if link not in name_to_index]
        if missing_links:
            raise ValueError(f"Required finger links missing in body_names: {missing_links}")

        thumb_link_index = name_to_index["L_th_l2"]
        pinky_link_index = name_to_index["L_lf_l2"]

        thumb_link_pos = body_state[:, thumb_link_index, 0:3]  # (B, 3)
        thumb_link_quat = body_state[:, thumb_link_index, 3:7]  # (B, 4)
        pinky_link_pos = body_state[:, pinky_link_index, 0:3]  # (B, 3)
        pinky_link_quat = body_state[:, pinky_link_index, 3:7]  # (B, 4)

        # Offsets from URDF (expressed in local link frames)
        thumb_tip_offset = torch.tensor([-0.0230, 0.0151, 0.0018], device=device).unsqueeze(0)  # (1, 3)
        pinky_tip_offset = torch.tensor([-0.0182, 0.0, -0.0306], device=device).unsqueeze(0)  # (1, 3)

        thumb_rot = matrix_from_quat(thumb_link_quat)  # (B, 3, 3)
        pinky_rot = matrix_from_quat(pinky_link_quat)  # (B, 3, 3)

        thumb_tip_world = thumb_link_pos + torch.bmm(
            thumb_rot, thumb_tip_offset.expand(thumb_link_pos.shape[0], -1).unsqueeze(-1)
        ).squeeze(-1)
        pinky_tip_world = pinky_link_pos + torch.bmm(
            pinky_rot, pinky_tip_offset.expand(pinky_link_pos.shape[0], -1).unsqueeze(-1)
        ).squeeze(-1)

        return 0.5 * (thumb_tip_world + pinky_tip_world)

    def _get_finger_tips_positions(self, states):
        """Get positions of all five finger tips.

        Returns:
            finger_tips_pos: (B, 5, 3) tensor with positions of [thumb, index, middle, ring, little] fingers
        """
        robot_config = self.robot
        rs = states.robots[robot_config.name]
        device = (rs.joint_pos if isinstance(rs.joint_pos, torch.Tensor) else torch.tensor(rs.joint_pos)).device

        body_state = (
            rs.body_state
            if isinstance(rs.body_state, torch.Tensor)
            else torch.tensor(rs.body_state, device=device).float()
        )

        # Left hand finger tip link names (using last link of each finger since tip links may be collapsed)
        # Order: thumb, index, middle, ring, little
        # Use l2 links as they are the last movable links before tip (tip links may be fixed and collapsed)
        finger_tip_link_names = ["L_th_l2", "L_ff_l2", "L_mf_l2", "L_rf_l2", "L_lf_l2"]
        finger_tips_pos = []

        for link_name in finger_tip_link_names:
            try:
                link_body_index = rs.body_names.index(link_name)
                link_pos = body_state[:, link_body_index, 0:3]  # (B, 3)
                finger_tips_pos.append(link_pos)
            except ValueError:
                # Fallback: use hand base position if link not found
                try:
                    hand_base_index = rs.body_names.index("L_hand_base")
                    hand_pos = body_state[:, hand_base_index, 0:3]  # (B, 3)
                    finger_tips_pos.append(hand_pos)
                except ValueError:
                    # Final fallback: use L_arm_l7 (last arm link)
                    arm_end_index = rs.body_names.index("L_arm_l7")
                    arm_end_pos = body_state[:, arm_end_index, 0:3]  # (B, 3)
                    finger_tips_pos.append(arm_end_pos)

        # Stack to (B, 5, 3)
        finger_tips_pos = torch.stack(finger_tips_pos, dim=1)
        return finger_tips_pos

    def _reward_hand_approach(self, env_states) -> torch.Tensor:
        """Reward for hand approaching the box."""
        box_pos = env_states.objects["object"].root_state[:, 0:3]  # (B, 3)
        hand_pos = self._get_hand_position(env_states)  # (B, 3)

        # Calculate distance from hand to box
        hand_box_dist = torch.norm(hand_pos - box_pos, dim=-1)  # (B,)

        # Calculate reward based on distance
        approach_reward_far = 1 - torch.tanh(hand_box_dist)  # (B,)
        approach_reward_near = 1 - 2 * torch.tanh(hand_box_dist * 10)  # (B,)

        return approach_reward_far + approach_reward_near  # (B,)

    def _reward_hand_close(self, env_states) -> torch.Tensor:
        """Reward for hand being close to box."""
        box_pos = env_states.objects["object"].root_state[:, 0:3]  # (B, 3)
        hand_pos = self._get_hand_position(env_states)  # (B, 3)

        # Calculate distance from hand to box
        hand_box_dist = torch.norm(hand_pos - box_pos, dim=-1)  # (B,)

        # Check if hand is close to box (within threshold)
        close_threshold = 0.08
        hand_close_bonus = (hand_box_dist < close_threshold).float()  # (B,)

        return hand_close_bonus  # (B,)

    def _reward_robot_target_qpos(self, env_states) -> torch.Tensor:
        """Reward for robot staying close to target joint positions."""
        robot_joint_pos = env_states.robots[self.robot_name].joint_pos[:, 2:]
        target_joint_pos = self._initial_states.robots[self.robot_name].joint_pos[:, 2:]

        joint_error = torch.norm(robot_joint_pos - target_joint_pos, dim=-1)
        return 1 - torch.tanh(joint_error)

    def _reward_trajectory_tracking(self, env_states) -> torch.Tensor:
        """Reward for tracking waypoints (only when object is grasped)."""
        # Use hand base position for distance calculation
        hand_pos = self._get_hand_position(env_states)  # (B, 3)
        grasped_mask = self.object_grasped
        tracking_reward = torch.zeros(self.num_envs, device=self.device)

        if grasped_mask.any():
            target_pos = self.waypoint_positions[self.current_waypoint_idx]
            distance = torch.norm(hand_pos - target_pos, dim=-1)

            approach_reward = (1 - torch.tanh(1.0 * distance)) * self.w_tracking_approach
            approach_reward = approach_reward * grasped_mask.float()

            reached = (distance < self.reach_threshold) & grasped_mask
            newly_reached = reached & (
                ~self.waypoints_reached[torch.arange(self.num_envs, device=self.device), self.current_waypoint_idx]
            )
            progress_reward = newly_reached.float() * self.w_tracking_progress

            if newly_reached.any():
                if newly_reached[0]:
                    wp_idx = self.current_waypoint_idx[0].item()
                    log.info(
                        f"[Env 0] Reached waypoint #{wp_idx}! Distance: {distance[0].item():.4f}m < {self.reach_threshold}m"
                    )

                self.waypoints_reached[newly_reached, self.current_waypoint_idx[newly_reached]] = True

                can_advance = newly_reached & (self.current_waypoint_idx < self.num_waypoints - 1)

                if can_advance.any() and can_advance[0]:
                    old_idx = self.current_waypoint_idx[0].item()
                    new_idx = old_idx + 1
                    log.info(f"   -> Advancing to waypoint #{new_idx}")

                self.current_waypoint_idx[can_advance] += 1

                if can_advance.any():
                    new_target_pos = self.waypoint_positions[self.current_waypoint_idx[can_advance]]
                    self.prev_distance_to_waypoint[can_advance] = torch.norm(
                        hand_pos[can_advance] - new_target_pos, dim=-1
                    )

            maintain_reward = torch.zeros(self.num_envs, device=self.device)
            all_reached = self.waypoints_reached.all(dim=1)
            completed_mask = all_reached & grasped_mask

            if completed_mask.any():
                last_target_pos = self.waypoint_positions[-1].unsqueeze(0).expand(self.num_envs, -1)
                distance_to_last = torch.norm(hand_pos - last_target_pos, dim=-1)

                maintain_reward[completed_mask] = torch.where(
                    distance_to_last[completed_mask] < self.reach_threshold,
                    torch.full((completed_mask.sum(),), 5, device=self.device),
                    (1 - torch.tanh(1.0 * distance_to_last[completed_mask])) * self.w_tracking_approach,
                )

            tracking_reward = torch.where(all_reached, maintain_reward, approach_reward + progress_reward)

        return tracking_reward

    def _observation(self, env_states) -> torch.Tensor:
        """Get observation using RoboVerse tensor state."""
        box_pos = env_states.objects["object"].root_state[:, 0:3]  # [num_envs, 3]
        box_quat = env_states.objects["object"].root_state[:, 3:7]  # [num_envs, 4]

        # Use hand base position for distance calculations
        hand_pos = self._get_hand_position(env_states)  # (B, 3)
        # Also get finger tips for observation (but not for distance calculations)
        finger_tips_pos = self._get_finger_tips_positions(env_states)  # (B, 5, 3)

        robot_joint_pos = env_states.robots[self.robot_name].joint_pos  # [num_envs, num_joints]
        robot_joint_vel = env_states.robots[self.robot_name].joint_vel  # [num_envs, num_joints]

        # Convert quaternion to rotation matrix for box
        box_mat = matrix_from_quat(box_quat)  # [num_envs, 3, 3]
        box_mat_flat = box_mat.view(self.num_envs, -1)  # [num_envs, 9]

        box_to_hand = box_pos - hand_pos  # [num_envs, 3]

        target_pos = self.waypoint_positions[self.current_waypoint_idx]
        target_to_hand = target_pos - hand_pos
        distance_to_target = torch.norm(target_to_hand, dim=-1, keepdim=True)

        waypoint_onehot = torch.nn.functional.one_hot(self.current_waypoint_idx, num_classes=self.num_waypoints).float()

        num_reached = self.waypoints_reached.sum(dim=1, keepdim=True).float() / self.num_waypoints
        grasped_flag = self.object_grasped.float().unsqueeze(-1)

        obs_list = [
            robot_joint_pos,
            robot_joint_vel,
            hand_pos,  # Hand base position
            finger_tips_pos.view(self.num_envs, -1),  # All finger tips flattened (B, 15)
            box_mat_flat[:, 3:],
            box_to_hand,
            target_pos,
            target_to_hand,
            distance_to_target,
            waypoint_onehot,
            num_reached,
            grasped_flag,
        ]

        obs = torch.cat(obs_list, dim=-1)  # [num_envs, obs_dim]

        return obs

    def _get_initial_states(self) -> list[dict] | None:
        """Get initial states for all environments."""
        # Using saved poses from keyboard control (saved at: 2025-11-12 06:36:58)
        wall_name = next(
            (obj.name for obj in self.scenario.objects if obj.name in ("wall", "nwall")),
            "wall",
        )

        init = [
            {
                "objects": {
                    "object": {
                        "pos": torch.tensor([0.434350, 0.036057, 0.816744]),
                        "rot": torch.tensor([0.999990, -0.000028, 0.001505, -0.004311]),
                    },
                    wall_name: {
                        "pos": torch.tensor([0.632921, -0.217400, 0.946513]),
                        "rot": torch.tensor([0.999490, -0.000045, 0.001448, -0.031900]),
                    },
                    "table": {
                        "pos": torch.tensor([0.680000, -0.200000, 0.399963]),
                        "rot": torch.tensor([1.000000, -0.000000, -0.000000, 0.000000]),
                    },
                    # Trajectory waypoints (world coordinates)
                    "traj_marker_0": {
                        "pos": torch.tensor([0.460000, -0.00000, 1.020000]),
                        "rot": torch.tensor([1.000000, 0.000000, 0.000000, 0.000000]),
                    },
                    "traj_marker_1": {
                        "pos": torch.tensor([0.400000, -0.00000, 1.220000]),
                        "rot": torch.tensor([1.000000, 0.000000, 0.000000, 0.000000]),
                    },
                    "traj_marker_2": {
                        "pos": torch.tensor([0.340000, -0.00000, 1.360000]),
                        "rot": torch.tensor([0.998750, 0.000000, 0.049979, 0.000000]),
                    },
                    "traj_marker_3": {
                        "pos": torch.tensor([0.430000, -0.00000, 1.220000]),
                        "rot": torch.tensor([1.000000, 0.000000, 0.000000, 0.000000]),
                    },
                    "traj_marker_4": {
                        "pos": torch.tensor([0.030000, 0.000000, 1.080000]),
                        "rot": torch.tensor([0.984726, 0.000000, 0.174108, 0.000000]),
                    },
                },
                "robots": {
                    "vega": {
                        "pos": torch.tensor([-0.230727, -0.190042, 0.000081]),
                        "rot": torch.tensor([1.000101, 0.000000, -0.000000, -0.000000]),
                        "dof_pos": {
                            # Base wheels
                            "B_wheel_j1": -16.534304,
                            "B_wheel_j2": 0.882773,
                            "R_wheel_j1": 40.744644,
                            "R_wheel_j2": 4.372887,
                            "L_wheel_j1": 2.951701,
                            "L_wheel_j2": 9.724807,
                            # Torso
                            "torso_j1": 0.165709,
                            "torso_j2": -0.000001,
                            "torso_j3": -0.083552,
                            # Left arm
                            "L_arm_j1": 0.014374,
                            "L_arm_j2": 0.047725,
                            "L_arm_j3": 0.599123,
                            "L_arm_j4": -0.369980,
                            "L_arm_j5": -2.883693,
                            "L_arm_j6": -1.396676,
                            "L_arm_j7": -1.380400,
                            # Right arm
                            "R_arm_j1": 0.120635,
                            "R_arm_j2": -0.000509,
                            "R_arm_j3": -0.031708,
                            "R_arm_j4": 0.168648,
                            "R_arm_j5": 0.008231,
                            "R_arm_j6": 0.209306,
                            "R_arm_j7": -0.144059,
                            # Left hand
                            "L_th_j0": -0.018750,
                            "L_th_j1": 0.140380,
                            "L_th_j2": 0.391281,
                            "L_ff_j1": 0.289102,
                            "L_ff_j2": 0.054875,
                            "L_mf_j1": -0.111047,
                            "L_mf_j2": -0.116898,
                            "L_rf_j1": 0.284011,
                            "L_rf_j2": 0.067212,
                            "L_lf_j1": -0.110856,
                            "L_lf_j2": -0.184257,
                            # Right hand
                            "R_th_j0": 0.717207,
                            "R_th_j1": 0.114617,
                            "R_th_j2": 0.156082,
                            "R_ff_j1": 0.198593,
                            "R_ff_j2": 0.223362,
                            "R_mf_j1": -0.071861,
                            "R_mf_j2": -0.079096,
                            "R_rf_j1": -0.031696,
                            "R_rf_j2": -0.034683,
                            "R_lf_j1": 0.222540,
                            "R_lf_j2": 0.251838,
                        },
                    },
                },
            }
            for _ in range(self.num_envs)
        ]

        return init


@register_task("pick_place.hand_trajectory", "pick_place_hand_trajectory")
class PickPlaceHandTrajectory(TrajectoryTrackingTaskBase):
    """Pick up a box with the robot hand (using finger tips) over a table.

    This is a concrete implementation of the hand trajectory tracking task.
    """

    def _get_initial_states(self) -> list[dict] | None:
        """Get initial states for all environments."""
        # Using saved poses from keyboard control (saved at: 2025-11-12 06:36:58)
        wall_name = next(
            (obj.name for obj in self.scenario.objects if obj.name in ("wall", "nwall")),
            "wall",
        )

        init = [
            {
                "objects": {
                    "object": {
                        "pos": torch.tensor([0.434350, 0.016057, 0.816744]),
                        "rot": torch.tensor([0.999990, -0.000028, 0.001505, -0.004311]),
                    },
                    wall_name: {
                        "pos": torch.tensor([0.632921, -0.217400, 0.946513]),
                        "rot": torch.tensor([0.999490, -0.000045, 0.001448, -0.031900]),
                    },
                    "table": {
                        "pos": torch.tensor([0.680000, -0.200000, 0.399963]),
                        "rot": torch.tensor([1.000000, -0.000000, -0.000000, 0.000000]),
                    },
                    # Trajectory waypoints (world coordinates)
                    "traj_marker_0": {
                        "pos": torch.tensor([0.40000, -0.460000, 1.020000]),
                        "rot": torch.tensor([1.000000, 0.000000, 0.000000, 0.000000]),
                    },
                    "traj_marker_1": {
                        "pos": torch.tensor([0.400000, -0.320000, 1.220000]),
                        "rot": torch.tensor([1.000000, 0.000000, 0.000000, 0.000000]),
                    },
                    "traj_marker_2": {
                        "pos": torch.tensor([0.40000, -0.190000, 1.360000]),
                        "rot": torch.tensor([0.998750, 0.000000, 0.049979, 0.000000]),
                    },
                    "traj_marker_3": {
                        "pos": torch.tensor([0.40000, -0.070000, 1.220000]),
                        "rot": torch.tensor([1.000000, 0.000000, 0.000000, 0.000000]),
                    },
                    "traj_marker_4": {
                        "pos": torch.tensor([0.40000, 0.000000, 1.080000]),
                        "rot": torch.tensor([0.984726, 0.000000, 0.174108, 0.000000]),
                    },
                },
                "robots": {
                    "vega": {
                        "pos": torch.tensor([-0.230727, -0.190042, 0.000081]),
                        "rot": torch.tensor([1.000101, 0.000000, -0.000000, -0.000000]),
                        "dof_pos": {
                            # Base wheels
                            "B_wheel_j1": 0.0,
                            "B_wheel_j2": 0.0,
                            "R_wheel_j1": 0.0,
                            "R_wheel_j2": 0.0,
                            "L_wheel_j1": 0.0,
                            "L_wheel_j2": 0.0,
                            # Torso - upright
                            "torso_j1": 0.2,
                            "torso_j2": 0.2,
                            "torso_j3": 0.0,
                            # Head - forward looking
                            # "head_j1": 0.0,
                            # "head_j2": 0.0,
                            # "head_j3": 0.0,
                            # Left arm - neutral pose
                            "L_arm_j1": 0.0,
                            "L_arm_j2": 0.0,
                            "L_arm_j3": 0.0,
                            "L_arm_j4": 0.0,
                            "L_arm_j5": 0.0,
                            "L_arm_j6": 0.0,
                            "L_arm_j7": 0.0,
                            # Right arm - neutral pose
                            "R_arm_j1": 0.0,
                            "R_arm_j2": 0.0,
                            "R_arm_j3": 0.0,
                            "R_arm_j4": 0.0,
                            "R_arm_j5": 0.0,
                            "R_arm_j6": 0.0,
                            "R_arm_j7": 0.0,
                            # Left hand - open
                            "L_th_j0": 0.0,
                            "L_th_j1": 0.0,
                            "L_th_j2": 0.0,
                            "L_ff_j1": 0.0,
                            "L_ff_j2": 0.0,
                            "L_mf_j1": 0.0,
                            "L_mf_j2": 0.0,
                            "L_rf_j1": 0.0,
                            "L_rf_j2": 0.0,
                            "L_lf_j1": 0.0,
                            "L_lf_j2": 0.0,
                            # Right hand - open
                            "R_th_j0": 0.0,
                            "R_th_j1": 0.0,
                            "R_th_j2": 0.0,
                            "R_ff_j1": 0.0,
                            "R_ff_j2": 0.0,
                            "R_mf_j1": 0.0,
                            "R_mf_j2": 0.0,
                            "R_rf_j1": 0.0,
                            "R_rf_j2": 0.0,
                            "R_lf_j1": 0.0,
                            "R_lf_j2": 0.0,
                        },
                    },
                },
            }
            for _ in range(self.num_envs)
        ]

        return init
