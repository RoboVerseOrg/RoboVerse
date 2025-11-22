"""Stage 2: Trajectory Tracking task.

This task focuses on learning to track waypoints while holding the object.
This is the second stage of the two-stage training pipeline.
It can be initialized from a Stage 1 checkpoint.
"""

from __future__ import annotations

import os
import torch
from loguru import logger as log

from metasim.constants import PhysicStateType
from metasim.scenario.objects import PrimitiveCubeCfg, RigidObjCfg
from metasim.scenario.scenario import ScenarioCfg, SimParamCfg
from metasim.task.registry import register_task
from roboverse_pack.tasks.pick_place.base import TrajectoryTrackingTaskBase


@register_task("pick_place.track", "pick_place_track")
class PickPlaceTrack(TrajectoryTrackingTaskBase):
    """Stage 2: Trajectory Tracking task.

    This task focuses on:
    - Tracking waypoints while holding the object
    - Maintaining grasp during movement

    Assumes object is already grasped (can be initialized from Stage 1 checkpoint).
    """

    DEFAULT_CONFIG = {
        "action_scale": 0.03,
        "reward_config": {
            "scales": {
                "tracking_approach": 4.0,
                "tracking_progress": 150.0,
                "grasp_maintain": 1.0,  # Reward for maintaining grasp
            }
        },
        "trajectory_tracking": {
            "num_waypoints": 5,
            "reach_threshold": 0.10,
            "grasp_check_distance": 0.12,
        },
        "stage1_checkpoint": None,  # Path to Stage 1 checkpoint for initialization
        "randomization": {
            "box_pos_range": 0.05,
            "robot_pos_noise": 0.0,
            "joint_noise_range": 0.05,
        },
    }

    def __init__(self, scenario, device=None):
        """Initialize Stage 2 task.
        
        Args:
            scenario: Scenario configuration
            device: Device to run on
        """
        # Override parent's reward config
        self.DEFAULT_CONFIG["reward_config"]["scales"].update({
            "tracking_approach": 4.0,
            "tracking_progress": 150.0,
            "grasp_maintain": 1.0,
        })
        
        super().__init__(scenario, device)
        
        # Stage 2 specific: Only use tracking rewards
        self.reward_functions = [
            self._reward_trajectory_tracking,
            self._reward_grasp_maintain,
        ]
        self.reward_weights = [
            1.0,  # tracking reward weight is handled internally
            self.DEFAULT_CONFIG["reward_config"]["scales"]["grasp_maintain"],
        ]
        
        # Check if we should start with object already grasped
        # This can be controlled via environment variable or config
        import os
        self.start_with_grasp = os.getenv("STAGE2_START_WITH_GRASP", "true").lower() == "true"

    def reset(self, env_ids=None):
        """Reset environment.
        
        If initialized from Stage 1 checkpoint, start with object already grasped.
        """
        obs, info = super().reset(env_ids=env_ids)
        
        if self.start_with_grasp:
            # Initialize with object already grasped (for Stage 2 training)
            if env_ids is None:
                env_ids_tensor = torch.arange(self.num_envs, device=self.device)
            else:
                env_ids_tensor = (
                    torch.tensor(env_ids, device=self.device) if not isinstance(env_ids, torch.Tensor) else env_ids
                )
            
            # Set object as grasped initially
            states = self.handler.get_states(mode="tensor")
            hand_pos = self._get_hand_position(states)  # (B, 3)
            box_pos = states.objects["object"].root_state[:, 0:3]  # (B, 3)
            hand_box_dist = torch.norm(hand_pos - box_pos, dim=-1)  # (B,)
            
            # Check if hand is close enough to consider grasped
            close_enough = hand_box_dist < self.grasp_check_distance
            
            # Set finger joints to closed position for environments that are close enough
            finger_joint_pos = states.robots[self.robot_name].joint_pos[
                :, self.left_hand_finger_joint_indices
            ]  # (B, 11)
            denom = self.gripper_close_q.unsqueeze(0) - self.gripper_open_q.unsqueeze(0) + 1e-6
            finger_close_ratios = ((finger_joint_pos - self.gripper_open_q.unsqueeze(0)) / denom).clamp(0.0, 1.0)
            hand_closed = finger_close_ratios.mean(dim=-1) > 0.5
            
            # Object is grasped if hand is closed AND close to object
            is_grasping = hand_closed & close_enough
            self.object_grasped[env_ids_tensor] = is_grasping[env_ids_tensor]
            
            if is_grasping.any() and is_grasping[0]:
                log.info(
                    f"[Env 0] Stage 2: Starting with object grasped. "
                    f"Hand-box distance: {hand_box_dist[0].item():.4f}m"
                )
        
        return obs, info

    def step(self, actions):
        """Step with delta control and stage 2 rewards."""
        current_states = self.handler.get_states(mode="tensor")
        box_pos = current_states.objects["object"].root_state[:, 0:3]  # (B, 3)

        # Get hand base position for distance calculation
        hand_pos = self._get_hand_position(current_states)  # (B, 3)
        # Calculate distance from hand to box
        hand_box_dist = torch.norm(hand_pos - box_pos, dim=-1)  # (B,)

        if self.auto_grip_ratio is None:
            self.auto_grip_ratio = torch.zeros(self.num_envs, device=self.device)

        delta_actions = actions * self._action_scale
        new_actions = self._last_action + delta_actions
        real_actions = torch.maximum(torch.minimum(new_actions, self._action_high), self._action_low)

        # Auto-control finger joints: maintain grasp if already grasping, otherwise try to grasp
        finger_joint_pos = current_states.robots[self.robot_name].joint_pos[
            :, self.left_hand_finger_joint_indices
        ]  # (B, 11)
        denom = self.gripper_close_q.unsqueeze(0) - self.gripper_open_q.unsqueeze(0) + 1e-6
        finger_close_ratios = ((finger_joint_pos - self.gripper_open_q.unsqueeze(0)) / denom).clamp(0.0, 1.0)
        hand_closed = finger_close_ratios.mean(dim=-1) > 0.5
        hand_close_to_object = hand_box_dist < self.grasp_check_distance
        is_grasping = hand_closed & hand_close_to_object

        # If already grasping, maintain grasp (keep fingers closed)
        # If not grasping, try to grasp (auto-close when close to object)
        distance_threshold = 0.03
        maintain_grasp_mask = is_grasping  # Already grasping, maintain it
        try_grasp_mask = (~is_grasping) & (hand_box_dist <= distance_threshold)  # Not grasping but close, try to grasp
        open_hand_mask = (~is_grasping) & (hand_box_dist > distance_threshold)  # Far from object, open hand

        # Update auto_grip_ratio
        delta = torch.zeros_like(hand_box_dist)
        delta[maintain_grasp_mask] = 0.0  # Maintain current grip
        delta[try_grasp_mask] = 0.1  # Close hand
        delta[open_hand_mask] = -0.1  # Open hand
        self.auto_grip_ratio = torch.clamp(self.auto_grip_ratio + delta, 0.0, 1.0)

        grip_lerp = self.auto_grip_ratio.unsqueeze(-1)
        finger_targets = self.gripper_open_q + grip_lerp * (self.gripper_close_q - self.gripper_open_q)

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

        # Check if fingers are closed
        finger_joint_pos = updated_states.robots[self.robot_name].joint_pos[
            :, self.left_hand_finger_joint_indices
        ]  # (B, 11)
        denom = self.gripper_close_q.unsqueeze(0) - self.gripper_open_q.unsqueeze(0) + 1e-6
        finger_close_ratios = ((finger_joint_pos - self.gripper_open_q.unsqueeze(0)) / denom).clamp(0.0, 1.0)  # (B, 11)
        hand_closed = finger_close_ratios.mean(dim=-1) > 0.5

        # Check if hand is close to object
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

        # Add stage info
        info["grasp_success"] = is_grasping
        info["stage"] = torch.full((self.num_envs,), 2, dtype=torch.long, device=self.device)  # Stage 2

        return obs, reward, terminated, time_out, info

    def _reward_grasp_maintain(self, env_states) -> torch.Tensor:
        """Reward for maintaining grasp during tracking."""
        # Reward is proportional to how well the grasp is maintained
        is_grasping = self.object_grasped.float()  # (B,)
        return is_grasping  # Simple binary reward: 1 if grasping, 0 if not

    def _get_initial_states(self) -> list[dict] | None:
        """Get initial states for all environments."""
        # Same as base class, but object starts at a higher position if starting with grasp
        init = [
            {
                "objects": {
                    "object": {
                        # Start object at a higher position if starting with grasp
                        "pos": torch.tensor([0.434350, 0.016057, 0.916744 if self.start_with_grasp else 0.816744]),
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
                            # Left hand - open initially (will be closed if starting with grasp)
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


def load_stage1_checkpoint_for_stage2(
    checkpoint_path: str,
    actor,
    obs_normalizer,
    device: torch.device,
    strict: bool = False,
) -> dict:
    """Load Stage 1 checkpoint for Stage 2 initialization.
    
    Args:
        checkpoint_path: Path to Stage 1 checkpoint file
        actor: Actor network to load weights into
        obs_normalizer: Observation normalizer to load weights into
        device: Device to load checkpoint on
        strict: Whether to strictly match state dict keys
        
    Returns:
        Dictionary with loaded checkpoint info
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Stage 1 checkpoint not found: {checkpoint_path}")
    
    log.info(f"Loading Stage 1 checkpoint from {checkpoint_path} for Stage 2 initialization")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Load actor weights (may have different observation dimensions, so use strict=False)
    try:
        actor.load_state_dict(checkpoint["actor_state_dict"], strict=strict)
        log.info("Successfully loaded actor weights from Stage 1 checkpoint")
    except Exception as e:
        log.warning(f"Could not load actor weights: {e}")
        if strict:
            raise
    
    # Load observation normalizer (may have different dimensions)
    try:
        if "obs_normalizer_state" in checkpoint:
            obs_normalizer.load_state_dict(checkpoint["obs_normalizer_state"], strict=strict)
            log.info("Successfully loaded observation normalizer from Stage 1 checkpoint")
    except Exception as e:
        log.warning(f"Could not load observation normalizer: {e}")
        if strict:
            raise
    
    return checkpoint

