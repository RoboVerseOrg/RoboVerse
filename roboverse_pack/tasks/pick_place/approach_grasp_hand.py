"""Simple Approach and Grasp task with hand control.

This task focuses on learning to approach the object and grasp it with hand.
Simple hand control: close when near the object.
"""

from __future__ import annotations

from copy import deepcopy

import torch
from loguru import logger as log

from metasim.constants import PhysicStateType
from metasim.scenario.objects import PrimitiveCubeCfg, RigidObjCfg
from metasim.scenario.scenario import ScenarioCfg, SimParamCfg
from metasim.task.registry import register_task
from roboverse_pack.tasks.pick_place.hand_trajectory import TrajectoryTrackingTaskBase


@register_task("approachgrasphand", "approachgrasphand")
class ApproachGraspHand(TrajectoryTrackingTaskBase):
    """Simple Approach and Grasp task with hand control.

    This task focuses on:
    - Approaching the object
    - Grasping the object with simple hand control (close when near)

    Success condition: Object is grasped (reward given when entering grasp state).
    Episode terminates if object is released.
    """

    GRASP_DISTANCE_THRESHOLD = 0.04  # Distance threshold for grasp check and hand closing
    GRASP_HISTORY_WINDOW = 5  # Number of frames to check for stable grasp

    # Shoulder lift parameters
    SHOULDER_INITIAL_POS = 0.0  # Default joint position for L_arm_j1
    SHOULDER_LIFT_OFFSET = 0.3  # Amount to lift shoulder when grasped (positive = lift up)
    SHOULDER_LIFT_KP = 0.2  # Proportional gain for shoulder lift control
    SHOULDER_MAX_DELTA = 0.1  # Maximum change per step

    DEFAULT_CONFIG_SIMPLE = deepcopy(TrajectoryTrackingTaskBase.DEFAULT_CONFIG)
    DEFAULT_CONFIG_SIMPLE["reward_config"]["scales"].update({
        "hand_approach": 0.5,
        "grasp_reward": 4.0,
    })
    DEFAULT_CONFIG_SIMPLE["grasp_config"] = {
        "grasp_check_distance": GRASP_DISTANCE_THRESHOLD,
        "hand_close_distance": GRASP_DISTANCE_THRESHOLD,
    }

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
    max_episode_steps = 200

    def __init__(self, scenario, device=None):
        # Placeholders needed during super().__init__ (reset may be called there)
        self.object_grasped = None
        self.stable_grasp = None  # Stable grasp state (5 frames stable)
        self._grasp_notified = None
        self._distance_history = None  # History buffer for stable grasp check
        self.shoulder_joint_name = "L_arm_j1"
        self.shoulder_joint_index = None

        super().__init__(scenario, device)

        # Override reward functions for this task
        self.reward_functions = [
            self._reward_hand_approach,
            self._reward_grasp,
        ]
        self.reward_weights = [
            self.DEFAULT_CONFIG_SIMPLE["reward_config"]["scales"]["hand_approach"],
            self.DEFAULT_CONFIG_SIMPLE["reward_config"]["scales"]["grasp_reward"],
        ]

        # Get config values
        grasp_config = self.DEFAULT_CONFIG_SIMPLE["grasp_config"]
        self.grasp_check_distance = grasp_config["grasp_check_distance"]
        self.hand_close_distance = grasp_config["hand_close_distance"]

        # Initialize tracking buffers
        self.object_grasped = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.stable_grasp = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._grasp_notified = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._distance_history = torch.full(
            (self.num_envs, self.GRASP_HISTORY_WINDOW),
            float("inf"),
            device=self.device,
        )

        # Find shoulder joint index
        joint_names = self.handler.get_joint_names(self.robot_name, sort=True)
        if self.shoulder_joint_name in joint_names:
            self.shoulder_joint_index = joint_names.index(self.shoulder_joint_name)
        else:
            log.warning(f"Joint {self.shoulder_joint_name} not found, shoulder lift disabled")

    def reset(self, env_ids=None):
        """Reset environment and tracking variables."""
        obs, info = super().reset(env_ids=env_ids)

        if env_ids is None:
            env_ids_tensor = torch.arange(self.num_envs, device=self.device)
        else:
            env_ids_tensor = (
                torch.tensor(env_ids, device=self.device) if not isinstance(env_ids, torch.Tensor) else env_ids
            )

        # Reset grasp tracking
        self.object_grasped[env_ids_tensor] = False
        self.stable_grasp[env_ids_tensor] = False
        if self._grasp_notified is None or self._grasp_notified.shape[0] != self.num_envs:
            self._grasp_notified = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self.stable_grasp = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self._distance_history = torch.full(
                (self.num_envs, self.GRASP_HISTORY_WINDOW),
                float("inf"),
                device=self.device,
            )
        else:
            self._grasp_notified[env_ids_tensor] = False
            self.stable_grasp[env_ids_tensor] = False
            self._distance_history[env_ids_tensor] = float("inf")

        return obs, info

    def step(self, actions):
        """Step with delta control and simple hand control."""
        current_states = self.handler.get_states(mode="tensor")
        box_pos = current_states.objects["object"].root_state[:, 0:3]
        hand_pos = self._get_hand_position(current_states)
        hand_box_dist = torch.norm(hand_pos - box_pos, dim=-1)

        # Apply delta control
        delta_actions = actions * self._action_scale
        new_actions = self._last_action + delta_actions
        real_actions = torch.clamp(new_actions, self._action_low, self._action_high)

        # Simple hand control: close when near object
        real_actions = self._apply_simple_hand_control(real_actions, hand_box_dist)

        # Apply shoulder lift control if stable grasp
        if self.stable_grasp is not None and self.stable_grasp.any() and self.shoulder_joint_index is not None:
            real_actions = self._apply_shoulder_lift_control(real_actions, current_states)

        # Bypass TrajectoryTrackingTaskBase.step to avoid its hand control logic
        # Call RLTaskEnv.step directly
        obs, reward, terminated, time_out, info = super(TrajectoryTrackingTaskBase, self).step(real_actions)
        self._last_action = real_actions

        # Update grasp state after step (for next step's comparison)
        updated_states = self.handler.get_states(mode="tensor")
        old_grasped = self.object_grasped.clone()
        self.object_grasped = self._compute_grasp_state(updated_states)

        newly_grasped = self.object_grasped & (~old_grasped)
        newly_released = (~self.object_grasped) & old_grasped

        if newly_grasped.any() and newly_grasped[0]:
            log.info(f"[Env 0] Object grasped! Distance: {hand_box_dist[0].item():.4f}m")
            self._grasp_notified[newly_grasped] = True

        if newly_released.any() and newly_released[0]:
            log.info(f"[Env 0] Object released! Distance: {hand_box_dist[0].item():.4f}m")
            self._grasp_notified[newly_released] = False

        # Terminate episode if object is released
        terminated = terminated | newly_released

        info["grasp_success"] = self.object_grasped
        info["stage"] = torch.full((self.num_envs,), 1, dtype=torch.long, device=self.device)

        return obs, reward, terminated, time_out, info

    def _apply_simple_hand_control(self, actions, hand_box_dist):
        """Simple hand control: close when near object."""
        # Close hand when close to object
        hand_close = hand_box_dist < self.hand_close_distance
        finger_targets_open = self.hand_open_q.unsqueeze(0).expand(self.num_envs, -1)  # (B, 11)
        finger_targets_close = self.hand_close_q.unsqueeze(0).expand(self.num_envs, -1)  # (B, 11)

        # Determine target finger positions based on distance
        finger_targets = torch.where(
            hand_close.unsqueeze(-1),
            finger_targets_close,
            finger_targets_open,
        )  # (B, 11)

        # Set finger joint targets in actions
        for i, joint_idx in enumerate(self.left_hand_finger_joint_indices):
            if i < finger_targets.shape[1]:
                actions[:, joint_idx] = finger_targets[:, i]

        return actions

    def _apply_shoulder_lift_control(self, actions, states):
        """Apply shoulder lift control when object is stably grasped."""
        joint_pos = states.robots[self.robot_name].joint_pos
        shoulder_idx = self.shoulder_joint_index

        # Target position: initial position - lift offset (negative offset lifts up for L_arm_j1)
        target_lift = self.SHOULDER_INITIAL_POS - self.SHOULDER_LIFT_OFFSET
        joint_error = target_lift - joint_pos[:, shoulder_idx]

        # Apply proportional control with max delta limit
        desired = joint_pos[:, shoulder_idx] + self.SHOULDER_LIFT_KP * joint_error
        delta = torch.clamp(
            desired - joint_pos[:, shoulder_idx],
            -self.SHOULDER_MAX_DELTA,
            self.SHOULDER_MAX_DELTA,
        )
        shoulder_value = torch.clamp(
            joint_pos[:, shoulder_idx] + delta,
            self._action_low[shoulder_idx],
            self._action_high[shoulder_idx],
        )

        # Apply lift control only to environments where object is stably grasped
        actions[self.stable_grasp, shoulder_idx] = shoulder_value[self.stable_grasp]

        # Maintain current finger angles when lifting (preserve grasp state)
        # This is better than forcing hand_close_q as it maintains the successful grasp angle
        for i, joint_idx in enumerate(self.left_hand_finger_joint_indices):
            if i < len(self.left_hand_finger_joint_indices):
                # Keep current finger joint positions (maintain grasp angle)
                actions[self.stable_grasp, joint_idx] = joint_pos[self.stable_grasp, joint_idx]

        return actions

    def _compute_grasp_state(self, states):
        """Compute if object is grasped (requires 5 stable frames based on distance only)."""
        box_pos = states.objects["object"].root_state[:, 0:3]
        hand_pos = self._get_hand_position(states)
        hand_box_dist = torch.norm(hand_pos - box_pos, dim=-1)

        # Update rolling distance history
        if self._distance_history is None or self._distance_history.shape[0] != self.num_envs:
            self._distance_history = torch.full(
                (self.num_envs, self.GRASP_HISTORY_WINDOW),
                float("inf"),
                device=self.device,
            )
        self._distance_history = torch.roll(self._distance_history, shifts=-1, dims=1)
        self._distance_history[:, -1] = hand_box_dist

        # Object is grasped if distance has been stable (close) for 5 frames
        self.stable_grasp = (self._distance_history < self.grasp_check_distance).all(dim=1)
        is_grasping = self.stable_grasp

        return is_grasping

    def _reward_hand_approach(self, env_states) -> torch.Tensor:
        """Reward for hand approaching the box."""
        box_pos = env_states.objects["object"].root_state[:, 0:3]
        hand_pos = self._get_hand_position(env_states)
        hand_box_dist = torch.norm(box_pos - hand_pos, dim=-1)

        approach_reward_far = 1 - torch.tanh(hand_box_dist)
        approach_reward_near = 1 - torch.tanh(hand_box_dist * 10)
        return approach_reward_far + approach_reward_near

    def _reward_grasp(self, env_states) -> torch.Tensor:
        """Reward for entering grasp state (only when newly grasped)."""
        # Compute current grasp state
        current_grasped = self._compute_grasp_state(env_states)

        # Compare with previous grasp state to find newly grasped
        # self.object_grasped contains the grasp state from previous step
        newly_grasped = current_grasped & (~self.object_grasped)

        # Only give reward when entering grasp state, not continuously
        return newly_grasped.float()

    def _get_initial_states(self) -> list[dict] | None:
        """Get initial states for all environments."""
        init = [
            {
                "objects": {
                    "object": {
                        "pos": torch.tensor([0.434350, 0.016057, 0.816744]),
                        "rot": torch.tensor([0.999990, -0.000028, 0.001505, -0.004311]),
                    },
                    "wall": {
                        "pos": torch.tensor([0.632921, -0.217400, 0.946513]),
                        "rot": torch.tensor([0.999490, -0.000045, 0.001448, -0.031900]),
                    },
                    "table": {
                        "pos": torch.tensor([0.680000, -0.200000, 0.399963]),
                        "rot": torch.tensor([1.000000, -0.000000, -0.000000, 0.000000]),
                    },
                    "traj_marker_0": {
                        "pos": torch.tensor([0.40000, -0.460000, 1.020000]),
                        "rot": torch.tensor([1.000000, 0.000000, 0.000000, 0.000000]),
                    },
                    "traj_marker_1": {
                        "pos": torch.tensor([0.400000, -0.320000, 1.220000]),
                        "rot": torch.tensor([1.000000, 0.000000, 0.000000, 0.000000]),
                    },
                    "traj_marker_2": {
                        "pos": torch.tensor([0.40000, -0.190000, 1.220000]),
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