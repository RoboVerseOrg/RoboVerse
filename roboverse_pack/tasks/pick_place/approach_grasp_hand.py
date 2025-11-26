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


@register_task("approach_grasp_hand", "approachgrasphand")
class ApproachGraspHand(TrajectoryTrackingTaskBase):
    """Simple Approach and Grasp task with hand control.

    This task focuses on:
    - Approaching the object
    - Grasping the object with simple hand control (close when near)

    Success condition: Object is grasped (reward given when entering grasp state).
    Episode terminates if object is released.
    """

    GRASP_DISTANCE_THRESHOLD = 0.04  # Distance threshold for grasp check and hand closing
    GRASP_HISTORY_WINDOW = 10  # Number of frames to check for stable grasp

    # Shoulder lift parameters (L_arm_j1)
    SHOULDER_LIFT_OFFSET = -1.0  # Lift shoulder joint down (negative = lift up for L_arm_j1)
    SHOULDER_LIFT_KP = 0.2  # Proportional gain for shoulder lift control
    SHOULDER_MAX_DELTA = 0.3  # Maximum change per step
    HAND_MARKER_NAME = "hand_debug_marker"

    DEFAULT_CONFIG_SIMPLE = deepcopy(TrajectoryTrackingTaskBase.DEFAULT_CONFIG)
    DEFAULT_CONFIG_SIMPLE["reward_config"]["scales"].update({
        "hand_approach": 0.5,
        "grasp_reward": 20.0,
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
                fix_base_link=True,
            ),
            RigidObjCfg(
                name="table",
                scale=(1, 1, 1),
                physics=PhysicStateType.RIGIDBODY,
                enabled_gravity=False,
                usd_path="roboverse_data/assets/EmbodiedGenData/demo_assets/table/usd/table.usd",
                urdf_path="roboverse_data/assets/EmbodiedGenData/demo_assets/table/result/table.urdf",
                mjcf_path="roboverse_data/assets/EmbodiedGenData/demo_assets/table/mjcf/table.xml",
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
            RigidObjCfg(
                name=HAND_MARKER_NAME,
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
        robots=["vega"],
        sim_params=SimParamCfg(
            dt=0.005,
        ),
        decimation=4,
    )
    max_episode_steps = 250

    def __init__(self, scenario, device=None):
        # Placeholders needed during super().__init__ (reset may be called there)
        self.object_grasped = None
        self.stable_grasp = None  # Stable grasp state (5 frames stable)
        self._grasp_notified = None
        self._distance_history = None  # History buffer for stable grasp check
        self._initial_box_height = None
        self.shoulder_joint_name = "L_arm_j1"
        self.shoulder_joint_index = None
        self.initial_joint_pos = None

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

        # Cached tensors for hand marker visualization
        self._hand_marker_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        self._hand_marker_zero = torch.zeros((self.num_envs, 3), device=self.device)

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

        # Lazily allocate tracking buffers (super().__init__ calls reset before post-init finishes)
        if self.object_grasped is None or self.object_grasped.shape[0] != self.num_envs:
            self.object_grasped = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if self.stable_grasp is None or self.stable_grasp.shape[0] != self.num_envs:
            self.stable_grasp = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if self._grasp_notified is None or self._grasp_notified.shape[0] != self.num_envs:
            self._grasp_notified = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if (
            self._distance_history is None
            or self._distance_history.shape[0] != self.num_envs
            or self._distance_history.shape[1] != self.GRASP_HISTORY_WINDOW
        ):
            self._distance_history = torch.full(
                (self.num_envs, self.GRASP_HISTORY_WINDOW),
                float("inf"),
                device=self.device,
            )

        # Reset grasp tracking
        self.object_grasped[env_ids_tensor] = False
        self.stable_grasp[env_ids_tensor] = False
        self._grasp_notified[env_ids_tensor] = False
        self._distance_history[env_ids_tensor] = float("inf")

        # Store initial joint positions if not already stored
        if self.initial_joint_pos is None:
            states = self.handler.get_states(mode="tensor")
            self.initial_joint_pos = states.robots[self.robot_name].joint_pos.clone()

        return obs, info

    def step(self, actions):
        """Step with delta control and simple hand control."""
        current_states = self.handler.get_states(mode="tensor")
        # Apply delta control
        delta_actions = actions * self._action_scale
        new_actions = self._last_action + delta_actions
        real_actions = torch.clamp(new_actions, self._action_low, self._action_high)

        # Simple hand control: keep hand open (debug mode)
        real_actions, hand_pos, hand_box_dist = self._apply_simple_hand_control(real_actions, current_states)
        self._update_hand_marker(current_states, hand_pos)

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

    def _apply_simple_hand_control(self, actions, states):
        """Simple hand control: close when near object."""
        box_pos = states.objects["object"].root_state[:, 0:3]
        hand_pos = self._get_hand_position(states)
        hand_box_dist = torch.norm(hand_pos - box_pos, dim=-1)

        # Close hand when close to object, otherwise keep open
        hand_close = hand_box_dist < self.hand_close_distance
        finger_targets_open = self.hand_open_q.unsqueeze(0).expand(self.num_envs, -1)  # (B, 11)
        finger_targets_close = self.hand_close_q.unsqueeze(0).expand(self.num_envs, -1)  # (B, 11)
        finger_targets = torch.where(
            hand_close.unsqueeze(-1),
            finger_targets_close,
            finger_targets_open,
        )

        for i, joint_idx in enumerate(self.left_hand_finger_joint_indices):
            if i < finger_targets.shape[1]:
                actions[:, joint_idx] = finger_targets[:, i]

        return actions, hand_pos, hand_box_dist

    def _apply_shoulder_lift_control(self, actions, current_states):
        """Apply shoulder lift control when stable grasp."""
        if self.initial_joint_pos is None:
            self.initial_joint_pos = current_states.robots[self.robot_name].joint_pos.clone()

        joint_pos = current_states.robots[self.robot_name].joint_pos
        shoulder_idx = self.shoulder_joint_index

        # Target position: initial position + lift offset (negative offset lifts up for L_arm_j1)
        target_lift = self.initial_joint_pos[:, shoulder_idx] + self.SHOULDER_LIFT_OFFSET
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

        # Apply lift control only to environments where stable grasp
        actions[self.stable_grasp, shoulder_idx] = shoulder_value[self.stable_grasp]

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

    def _update_hand_marker(self, states, hand_pos):
        """Visualize hand position using a debug marker."""
        if self.HAND_MARKER_NAME not in states.objects:
            return

        marker_state = states.objects[self.HAND_MARKER_NAME].root_state
        marker_state[:, 0:3] = hand_pos
        marker_state[:, 3:7] = self._hand_marker_quat
        marker_state[:, 7:10] = self._hand_marker_zero
        marker_state[:, 10:13] = self._hand_marker_zero

        self.handler.set_states(states)

    def _reward_hand_approach(self, env_states) -> torch.Tensor:
        """Reward for hand approaching the box."""
        box_pos = env_states.objects["object"].root_state[:, 0:3]
        hand_pos = self._get_hand_position(env_states)
        hand_box_dist = torch.norm(box_pos - hand_pos, dim=-1)

        approach_reward_far = 1 - torch.tanh(hand_box_dist)
        clipped_dist = hand_box_dist.clamp(0.03, 0.10)
        approach_reward_near = 50.0 * (0.10 - clipped_dist)
        return approach_reward_far + approach_reward_near

    def _reward_grasp(self, env_states) -> torch.Tensor:
        """Reward only when stable grasp."""
        if self.stable_grasp is None:
            return torch.zeros(self.num_envs, device=self.device)
        return self.stable_grasp.float()



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
                    self.HAND_MARKER_NAME: {
                        "pos": torch.tensor([0.0, 0.0, 0.0]),
                        "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
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
                            "L_arm_j5": 1.5,
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
                            # Left hand - open (match gripper_open_q)
                            "L_th_j0": 1.505,  # max 1.605 - 0.1
                            "L_th_j1": 0.0834,  # max 0.1834 - 0.1
                            "L_th_j2": 0.1731,  # max 0.2731 - 0.1
                            "L_ff_j1": 0.30,  # max 0.40 - 0.1
                            "L_ff_j2": 0.35,  # max 0.45 - 0.1
                            "L_mf_j1": 0.30,
                            "L_mf_j2": 0.35,
                            "L_rf_j1": 0.30,
                            "L_rf_j2": 0.35,
                            "L_lf_j1": 0.30,
                            "L_lf_j2": 0.35,
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
