"""Stage 1: Approach and Grasp & Lift task.

This task focuses on learning to approach the object, grasp it, and lift it.
This is the first stage of the two-stage training pipeline.
"""

from __future__ import annotations

from copy import deepcopy

import torch
from loguru import logger as log

from metasim.constants import PhysicStateType
from metasim.scenario.objects import PrimitiveCubeCfg, RigidObjCfg
from metasim.scenario.scenario import ScenarioCfg, SimParamCfg
from metasim.task.registry import register_task
from roboverse_pack.tasks.pick_place.base import TrajectoryTrackingTaskBase


@register_task("pick_place.approach_grasp", "pick_place_approach_grasp")
class PickPlaceApproachGrasp(TrajectoryTrackingTaskBase):
    """Stage 1: Approach and Grasp with Stability Test task.

    This task focuses on:
    - Approaching the object
    - Grasping the object
    - Stability test: returning to initial pose while maintaining grasp

    Success condition: Object is grasped and remains stable during stability test.
    """

    DISTANCE_THRESHOLD = 0.05

    DEFAULT_CONFIG = deepcopy(TrajectoryTrackingTaskBase.DEFAULT_CONFIG)
    DEFAULT_CONFIG["reward_config"]["scales"].update(
        {
            "gripper_approach": 0.5,
            "stability_reward": 5.0,
        }
    )
    DEFAULT_CONFIG["grasp_stability"] = {
        "grasp_check_distance": DISTANCE_THRESHOLD,
        "stability_test": {
            "enabled": True,
            "stability_distance_threshold": DISTANCE_THRESHOLD,
            "return_to_init_after_grasp": True,
        },
    }

    def __init__(self, scenario, device=None):
        # Placeholders needed during super().__init__ (reset may be called there)
        self.in_stability_test = None
        self.initial_joint_pos = None
        self._non_hand_joint_indices = None
        self.shoulder_joint_name = "L_arm_j1"
        self.shoulder_joint_index = None
        self.shoulder_lift_offset = 3.0
        self.shoulder_max_delta = 0.3
        self.grasp_history_window = 10
        self._distance_history = None
        self._stability_notified = None

        super().__init__(scenario, device)

        # Stage 1 specific: Only use gripper close reward
        self.reward_functions = [
            self._reward_gripper_approach,
            self._reward_stability_active,
        ]
        self.reward_weights = [
            self.DEFAULT_CONFIG["reward_config"]["scales"]["gripper_approach"],
            self.DEFAULT_CONFIG["reward_config"]["scales"]["stability_reward"],
        ]
        
        # Get config values
        self.grasp_check_distance = self.DEFAULT_CONFIG["grasp_stability"]["grasp_check_distance"]
        
        # Stability test config
        stability_config = self.DEFAULT_CONFIG["grasp_stability"]["stability_test"]
        self.stability_test_enabled = stability_config["enabled"]
        self.stability_distance_threshold = stability_config["stability_distance_threshold"]
        self.return_to_init_after_grasp = stability_config["return_to_init_after_grasp"]
        
        # Track stability state buffers
        self.in_stability_test = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.initial_joint_pos = None
        self.shoulder_joint_index = self._find_joint_index(self.shoulder_joint_name)
        self._stability_notified = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def reset(self, env_ids=None):
        """Reset environment and tracking variables."""
        self._ensure_stability_buffers()
        obs, info = super().reset(env_ids=env_ids)
        
        if env_ids is None:
            env_ids_tensor = torch.arange(self.num_envs, device=self.device)
        else:
            env_ids_tensor = (
                torch.tensor(env_ids, device=self.device) if not isinstance(env_ids, torch.Tensor) else env_ids
            )
        
        # Reset stability test flag
        self.in_stability_test[env_ids_tensor] = False
        if self._stability_notified is None or self._stability_notified.shape[0] != self.num_envs:
            self._stability_notified = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        else:
            self._stability_notified[env_ids_tensor] = False
        
        # Store initial joint positions if not already stored
        if self.initial_joint_pos is None:
            states = self.handler.get_states(mode="tensor")
            self.initial_joint_pos = states.robots[self.robot_name].joint_pos.clone()
        self._distance_history = torch.full(
            (self.num_envs, self.grasp_history_window),
            float("inf"),
            device=self.device,
        )
        
        return obs, info

    def step(self, actions):
        """Step with delta control and stage 1 rewards."""
        self._ensure_stability_buffers()

        current_states = self.handler.get_states(mode="tensor")
        box_pos = current_states.objects["object"].root_state[:, 0:3]
        hand_pos = self._get_hand_position(current_states)
        hand_box_dist = torch.norm(hand_pos - box_pos, dim=-1)

        real_actions = self._apply_action_strategy(actions, current_states, hand_box_dist)

        # Bypass TrajectoryTrackingTaskBase.step (which contains waypoint logic we don't need)
        obs, reward, terminated, time_out, info = super(TrajectoryTrackingTaskBase, self).step(real_actions)
        self._last_action = real_actions

        updated_states = self.handler.get_states(mode="tensor")
        old_grasped = self.object_grasped.clone()

        (
            is_grasping,
            newly_grasped,
            newly_released,
            updated_hand_box_dist,
        ) = self._compute_grasp_state(updated_states, old_grasped)

        self.object_grasped = is_grasping
        self._update_stability_flags(newly_grasped, is_grasping, updated_hand_box_dist)
        self._maybe_log_release(newly_released, updated_hand_box_dist)

        info["grasp_success"] = self.object_grasped
        info["in_stability_test"] = self.in_stability_test
        info["stage"] = torch.full((self.num_envs,), 1, dtype=torch.long, device=self.device)

        self._maybe_notify_stable()

        return obs, reward, terminated, time_out, info

    def _apply_action_strategy(self, actions, current_states, hand_box_dist):
        if self.auto_grip_ratio is None:
            self.auto_grip_ratio = torch.zeros(self.num_envs, device=self.device)

        delta_actions = actions * self._action_scale
        new_actions = self._last_action + delta_actions
        real_actions = torch.clamp(new_actions, self._action_low, self._action_high)

        if self._should_use_stability_control() and self.shoulder_joint_index is not None:
            real_actions = self._apply_stability_control(real_actions, current_states)

        if not (self.stability_test_enabled and self.in_stability_test.all()):
            self._apply_auto_grip(real_actions, hand_box_dist)

        return real_actions

    def _should_use_stability_control(self):
        return (
            self.stability_test_enabled
            and self.in_stability_test is not None
            and self.in_stability_test.any()
        )

    def _apply_stability_control(self, real_actions, current_states):
        if self.initial_joint_pos is None:
            self.initial_joint_pos = current_states.robots[self.robot_name].joint_pos.clone()

        joint_pos = current_states.robots[self.robot_name].joint_pos
        shoulder_idx = self.shoulder_joint_index
        target_lift = self.initial_joint_pos[:, shoulder_idx] - self.shoulder_lift_offset
        joint_error = target_lift - joint_pos[:, shoulder_idx]

        kp = 0.2
        desired = joint_pos[:, shoulder_idx] + kp * joint_error
        delta = torch.clamp(desired - joint_pos[:, shoulder_idx], -self.shoulder_max_delta, self.shoulder_max_delta)
        shoulder_value = torch.clamp(
            joint_pos[:, shoulder_idx] + delta,
            self._action_low[shoulder_idx],
            self._action_high[shoulder_idx],
        )

        real_actions[self.in_stability_test, shoulder_idx] = shoulder_value[self.in_stability_test]

        for i, joint_idx in enumerate(self.left_hand_finger_joint_indices):
            if i < self.gripper_close_q.shape[0]:
                real_actions[self.in_stability_test, joint_idx] = self.gripper_close_q[i]

        return real_actions

    def _apply_auto_grip(self, real_actions, hand_box_dist):
        close_mask = hand_box_dist <= self.grasp_check_distance
        delta = torch.where(close_mask, torch.full_like(hand_box_dist, 0.1), torch.full_like(hand_box_dist, -0.1))
        self.auto_grip_ratio = torch.clamp(self.auto_grip_ratio + delta, 0.0, 1.0)

        finger_targets = self.gripper_open_q + self.auto_grip_ratio.unsqueeze(-1) * (
            self.gripper_close_q - self.gripper_open_q
        )
        not_in_test = (
            ~self.in_stability_test
            if self.stability_test_enabled
            else torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        )
        for i, joint_idx in enumerate(self.left_hand_finger_joint_indices):
            if i < finger_targets.shape[1]:
                real_actions[not_in_test, joint_idx] = finger_targets[not_in_test, i]

    def _compute_grasp_state(self, states, old_grasped):
        box_pos = states.objects["object"].root_state[:, 0:3]
        hand_pos = self._get_hand_position(states)
        hand_box_dist = torch.norm(hand_pos - box_pos, dim=-1)

        # Update rolling distance history
        if self._distance_history is None or self._distance_history.shape[0] != self.num_envs:
            self._distance_history = torch.full(
                (self.num_envs, self.grasp_history_window),
                float("inf"),
                device=self.device,
            )
        self._distance_history = torch.roll(self._distance_history, shifts=-1, dims=1)
        self._distance_history[:, -1] = hand_box_dist

        stable_grasp = (self._distance_history < self.grasp_check_distance).all(dim=1)
        is_grasping = stable_grasp
        newly_grasped = is_grasping & (~old_grasped)
        newly_released = (~is_grasping) & old_grasped
        return is_grasping, newly_grasped, newly_released, hand_box_dist

    def _update_stability_flags(self, newly_grasped, is_grasping, hand_box_dist):
        if (
            newly_grasped.any()
            and self.stability_test_enabled
            and self.return_to_init_after_grasp
            and self.in_stability_test is not None
        ):
            self.in_stability_test[newly_grasped] = True

        if self.stability_test_enabled and self.in_stability_test is not None and self.in_stability_test.any():
            lost_grasp = self.in_stability_test & (~is_grasping | (hand_box_dist >= self.stability_distance_threshold))
            if lost_grasp.any():
                self.in_stability_test[lost_grasp] = False
                if self._stability_notified is not None:
                    self._stability_notified[lost_grasp] = False

    def _maybe_log_release(self, newly_released, hand_box_dist):
        if newly_released.any() and newly_released[0]:
            log.info(f"[Env 0] Object released! Hand-box distance: {hand_box_dist[0].item():.4f}m")

    def _maybe_notify_stable(self):
        if self._stability_notified is None or self.in_stability_test is None:
            return
        newly_stable = self.in_stability_test & (~self._stability_notified)
        if newly_stable.any():
            env_list = ", ".join(str(i.item()) for i in torch.nonzero(newly_stable, as_tuple=False).flatten())
            # log.info(f"[Stability] Stable grasp confirmed in env(s): {env_list}")
            self._stability_notified[newly_stable] = True

    def _reward_stability_active(self, env_states) -> torch.Tensor:
        """Reward for staying in stability test (per-step)"""
        self._ensure_stability_buffers()
        if not self.stability_test_enabled or self.in_stability_test is None:
            return torch.zeros(self.num_envs, device=self.device)
        return self.in_stability_test.float()

    def _reward_grasp_maintain(self, env_states) -> torch.Tensor:
        """Continuous reward for maintaining grasp (per step)."""
        # Check if object is grasped
        box_pos = env_states.objects["object"].root_state[:, 0:3]  # (B, 3)
        hand_pos = self._get_hand_position(env_states)  # (B, 3)
        hand_box_dist = torch.norm(hand_pos - box_pos, dim=-1)  # (B,)
        
        # Check if fingers are closed
        finger_joint_pos = env_states.robots[self.robot_name].joint_pos[
            :, self.left_hand_finger_joint_indices
        ]  # (B, 11)
        denom = self.gripper_close_q.unsqueeze(0) - self.gripper_open_q.unsqueeze(0) + 1e-6
        finger_close_ratios = ((finger_joint_pos - self.gripper_open_q.unsqueeze(0)) / denom).clamp(0.0, 1.0)
        hand_closed = finger_close_ratios.mean(dim=-1) > 0.05
        
        # Check if hand is close to object
        hand_close_to_object = hand_box_dist < self.grasp_check_distance
        
        # Object is grasped if hand is closed AND hand is close to object
        is_grasping = hand_closed & hand_close_to_object
        
        # Return 1.0 if grasping, 0.0 otherwise (will be scaled by weight)
        return is_grasping.float()

    def _ensure_stability_buffers(self):
        """Ensure tensors used for stability logic exist on the correct device."""
        if (
            self.in_stability_test is None
            or self.in_stability_test.shape[0] != self.num_envs
            or self.in_stability_test.device != self.device
        ):
            self.in_stability_test = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if (
            self._stability_notified is None
            or self._stability_notified.shape[0] != self.num_envs
            or self._stability_notified.device != self.device
        ):
            self._stability_notified = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _find_joint_index(self, joint_name: str) -> int | None:
        joint_names = self.handler.get_joint_names(self.robot_name, sort=True)
        if joint_name in joint_names:
            return joint_names.index(joint_name)
        return None

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

