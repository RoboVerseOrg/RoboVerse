"""AllegroHand object reorientation task.

This task tests the AllegroHand's ability to manipulate objects (cube, egg, or pen)
to match target orientations using in-hand manipulation.
"""

from typing import List

import numpy as np
import torch

from metasim.cfg.objects import RigidObjCfg
from metasim.constants import PhysicStateType, TaskType
from metasim.utils import configclass
from metasim.utils.math import quat_inv, quat_mul

from ..base_task_cfg import BaseTaskCfg


@configclass
class AllegroHandCfg(BaseTaskCfg):
    """Configuration for AllegroHand object reorientation task."""

    episode_length = 600
    traj_filepath = None
    task_type = TaskType.TABLETOP_MANIPULATION

    # Object type: "block", "egg", or "pen"
    object_type = "block"

    # Reward parameters
    dist_reward_scale = -10.0
    rot_reward_scale = 1.0
    action_penalty_scale = -0.0002
    reach_goal_bonus = 250.0
    success_tolerance = 0.1  # radians
    fall_dist = 0.24
    fall_penalty = 0.0
    rot_eps = 0.1
    av_factor = 0.1
    max_consecutive_successes = 0

    # Control parameters
    use_relative_control = False
    actions_moving_average = 1.0
    dof_speed_scale = 20.0

    # Observation parameters
    obs_type = "full_no_vel"  # Options: "full_no_vel", "full", "full_state"

    # Reset parameters
    reset_position_noise = 0.01
    reset_rotation_noise = 0.0
    reset_dof_pos_noise = 0.2
    reset_dof_vel_noise = 0.0

    # Domain randomization
    force_scale = 0.0
    force_prob_range = (0.001, 0.1)
    force_decay = 0.99
    force_decay_interval = 0.08

    objects: List[RigidObjCfg] = None

    def __post_init__(self):
        """Initialize objects after dataclass initialization."""
        super().__post_init__()
        self._prev_actions = None
        self._goal_rotations = None

        # Initialize objects if not already set
        if self.objects is None:
            self.objects = [
                RigidObjCfg(
                    name="block",
                    usd_path="roboverse_data/assets/isaacgymenvs/block_allegrohand_multicolor/usd/cube_multicolor_allegro.usd",
                    mjcf_path="roboverse_data/assets/isaacgymenvs/block_allegrohand_multicolor/urdf/cube_multicolor_allegro.urdf",
                    urdf_path="roboverse_data/assets/isaacgymenvs/block_allegrohand_multicolor/urdf/cube_multicolor_allegro.urdf",
                    default_position=(0.0, -0.20000000298023224, 0.6600000023841858),  # From IsaacGym reference
                    default_orientation=(1.0, 0.0, 0.0, 0.0),  # w, x, y, z
                ),
                RigidObjCfg(
                    name="goal",
                    usd_path="roboverse_data/assets/isaacgymenvs/block_allegrohand_multicolor/usd/cube_multicolor_allegro.usd",
                    mjcf_path="roboverse_data/assets/isaacgymenvs/block_allegrohand_multicolor/urdf/cube_multicolor_allegro.urdf",
                    urdf_path="roboverse_data/assets/isaacgymenvs/block_allegrohand_multicolor/urdf/cube_multicolor_allegro.urdf",
                    default_position=(-0.20000000298023224, -0.25999999046325684, 0.6399999856948853),  # From IsaacGym reference
                    default_orientation=(1.0, 0.0, 0.0, 0.0),  # w, x, y, z
                    physics=PhysicStateType.XFORM,  # Goal doesn't need physics
                ),
            ]

    observation_space = {
        "full_no_vel": 50,
        "full": 72,
        "full_state": 88
    }

    randomize = {
        "robot": {
            "allegro_hand": {
                "joint_qpos": {
                    "type": "uniform",
                    "low": -0.2,
                    "high": 0.2,
                }
            }
        },
        "object": {
            "block": {
                "position": {
                    "x": [0.0, 0.0],  # Keep x fixed
                    "y": [-0.21, -0.19],  # Small variation around -0.2
                    "z": [0.56, 0.56],  # Keep z fixed
                },
                "orientation": {
                    "x": [-1.0, 1.0],
                    "y": [-1.0, 1.0],
                    "z": [-1.0, 1.0],
                    "w": [-1.0, 1.0],
                }
            },
            "goal": {
                "orientation": {
                    "x": [-1.0, 1.0],
                    "y": [-1.0, 1.0],
                    "z": [-1.0, 1.0],
                    "w": [-1.0, 1.0],
                }
            }
        }
    }

    def get_observation(self, states):
        """Get observations from states."""
        observations = []

        for i, env_state in enumerate(states):
            robot_state = env_state["robots"]["allegro_hand"]
            block_state = env_state["objects"]["block"]
            goal_state = env_state["objects"]["goal"]

            # Get hand joint positions
            if "joint_qpos" in robot_state:
                hand_pos = torch.tensor(robot_state["joint_qpos"], dtype=torch.float32)
            elif "dof_pos" in robot_state:
                hand_pos = torch.tensor([v for v in robot_state["dof_pos"].values()], dtype=torch.float32)
            else:
                hand_pos = torch.zeros(16, dtype=torch.float32)

            # Get hand base position (for relative calculations)
            hand_base_pos = torch.tensor(robot_state.get("pos", [0.0, 0.0, 0.5]), dtype=torch.float32)

            # Get object pose relative to hand
            object_pos = torch.tensor(block_state["pos"], dtype=torch.float32)
            object_rot = torch.tensor(block_state["rot"], dtype=torch.float32)

            # Get target pose relative to hand
            target_pos = torch.tensor(goal_state["pos"], dtype=torch.float32)
            target_rot = torch.tensor(goal_state["rot"], dtype=torch.float32)

            # Normalize quaternions to ensure they are valid
            object_rot_norm = torch.norm(object_rot, p=2, dim=-1, keepdim=True)
            if object_rot_norm > 0:
                object_rot = object_rot / object_rot_norm
            else:
                object_rot = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)

            target_rot_norm = torch.norm(target_rot, p=2, dim=-1, keepdim=True)
            if target_rot_norm > 0:
                target_rot = target_rot / target_rot_norm
            else:
                target_rot = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)

            # Compute relative rotation between object and goal
            quat_diff = quat_mul(object_rot, quat_inv(target_rot))
            # Normalize the result
            quat_diff_norm = torch.norm(quat_diff, p=2, dim=-1, keepdim=True)
            if quat_diff_norm > 0:
                quat_diff = quat_diff / quat_diff_norm
            else:
                quat_diff = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)

            # Build observation based on obs_type
            if self.obs_type == "full_no_vel":
                obs = torch.cat([
                    hand_pos,  # 16
                    object_pos,  # 3
                    object_rot,  # 4
                    target_pos,  # 3
                    target_rot,  # 4
                    quat_diff,  # 4
                    self._get_prev_actions(i) if hasattr(self, '_prev_actions') and self._prev_actions is not None else torch.zeros(16, dtype=torch.float32),  # Previous actions
                ])  # Total: 50
            elif self.obs_type == "full":
                if "joint_qvel" in robot_state:
                    hand_vel = torch.tensor(robot_state["joint_qvel"], dtype=torch.float32)
                elif "dof_vel" in robot_state:
                    hand_vel = torch.tensor([v for v in robot_state["dof_vel"].values()], dtype=torch.float32)
                else:
                    hand_vel = torch.zeros(16, dtype=torch.float32)
                object_vel = torch.tensor(block_state.get("lin_vel", block_state.get("vel", [0.0, 0.0, 0.0])), dtype=torch.float32)
                object_ang_vel = torch.tensor(block_state.get("ang_vel", [0.0, 0.0, 0.0]), dtype=torch.float32)

                obs = torch.cat([
                    hand_pos,  # 16
                    hand_vel * 0.2,  # 16 (scaled)
                    object_pos,  # 3
                    object_rot,  # 4
                    object_vel,  # 3
                    object_ang_vel * 0.2,  # 3 (scaled)
                    target_pos,  # 3
                    target_rot,  # 4
                    quat_diff,  # 4
                    self._get_prev_actions(i) if hasattr(self, '_prev_actions') and self._prev_actions is not None else torch.zeros(16, dtype=torch.float32),  # Previous actions
                ])  # Total: 72
            elif self.obs_type == "full_state":
                if "joint_qvel" in robot_state:
                    hand_vel = torch.tensor(robot_state["joint_qvel"])
                elif "dof_vel" in robot_state:
                    hand_vel = torch.tensor([v for v in robot_state["dof_vel"].values()])
                else:
                    hand_vel = torch.zeros(16)
                object_vel = torch.tensor(block_state.get("lin_vel", block_state.get("vel", [0.0, 0.0, 0.0])))
                object_ang_vel = torch.tensor(block_state.get("ang_vel", [0.0, 0.0, 0.0]))

                obs = torch.cat([
                    hand_pos,  # 16
                    hand_vel * 0.2,  # 16 (scaled)
                    torch.zeros(16),  # Force/torque placeholder
                    object_pos,  # 3
                    object_rot,  # 4
                    object_vel,  # 3
                    object_ang_vel * 0.2,  # 3 (scaled)
                    target_pos,  # 3
                    target_rot,  # 4
                    quat_diff,  # 4
                    self._get_prev_actions(i) if hasattr(self, '_prev_actions') and self._prev_actions is not None else torch.zeros(16, dtype=torch.float32),  # Previous actions
                ])  # Total: 88

            # Check for NaN values and replace with zeros if found
            if torch.isnan(obs).any():
                print(f"Warning: NaN detected in observation. Replacing with zeros.")
                obs = torch.nan_to_num(obs, nan=0.0)

            observations.append(obs)

        return torch.stack(observations) if observations else torch.zeros((0, 50))

    def reward_fn(self, states, actions):
        """Compute reward."""
        # Debug print states type
        if not hasattr(self, '_debug_printed_reward'):
            print(f"Reward fn - States type: {type(states)}")
            print(f"Reward fn - States class name: {states.__class__.__name__ if hasattr(states, '__class__') else 'No class'}")
            if hasattr(states, '__dict__'):
                print(f"Reward fn - States attributes: {list(states.__dict__.keys())}")
            self._debug_printed_reward = True

        # Update previous actions for next observation
        if hasattr(self, '_prev_actions'):
            num_envs = len(actions)
            if self._prev_actions is None or self._prev_actions.shape[0] != num_envs:
                self._prev_actions = torch.zeros((num_envs, 16), dtype=torch.float32)

            for i, act in enumerate(actions):
                if isinstance(act, dict):
                    robot_name = list(act.keys())[0]
                    if "dof_pos_target" in act[robot_name]:
                        action_values = list(act[robot_name]["dof_pos_target"].values())
                        self._prev_actions[i] = torch.tensor(action_values, dtype=torch.float32)
        # Handle TensorState object from IsaacGym
        if hasattr(states, '__class__') and states.__class__.__name__ == 'TensorState':
            # Get states from TensorState
            # Debug print object keys
            if not hasattr(self, '_debug_printed_objects'):
                print(f"Objects in states: {list(states.objects.keys())}")
                for obj_name, obj_state in states.objects.items():
                    print(f"Object '{obj_name}' has attributes: {dir(obj_state)}")
                    if hasattr(obj_state, 'root_state'):
                        print(f"Object '{obj_name}' root_state shape: {obj_state.root_state.shape}")
                        print(f"Object '{obj_name}' root_state sample: {obj_state.root_state[0]}")
                self._debug_printed_objects = True

            object_state = states.objects.get("block")
            goal_state = states.objects.get("goal")

            if object_state is None or goal_state is None:
                print(f"ERROR: Missing objects - block: {object_state is not None}, goal: {goal_state is not None}")
                return torch.zeros(len(actions), device=actions[0][list(actions[0].keys())[0]]["dof_pos_target"][list(actions[0][list(actions[0].keys())[0]]["dof_pos_target"].keys())[0]].device if isinstance(actions[0], dict) else torch.device('cpu'))

            # Get positions and rotations as tensors
            object_pos = object_state.root_state[:, :3]  # Shape: (num_envs, 3)
            object_rot = object_state.root_state[:, 3:7]  # Shape: (num_envs, 4)
            goal_pos = goal_state.root_state[:, :3]
            goal_rot = goal_state.root_state[:, 3:7]

            # Process actions to get tensor
            num_envs = object_pos.shape[0]
            action_tensor = torch.zeros((num_envs, 16), device=object_pos.device)

            for i, act in enumerate(actions):
                if isinstance(act, dict):
                    robot_name = list(act.keys())[0]  # Get robot name
                    if "dof_pos_target" in act[robot_name]:
                        action_values = list(act[robot_name]["dof_pos_target"].values())
                        action_tensor[i] = torch.tensor(action_values, device=object_pos.device)

            # Compute rewards in batch
            # Distance reward (position)
            pos_dist = torch.norm(object_pos - goal_pos, p=2, dim=1)
            dist_reward = self.dist_reward_scale * pos_dist

            # Check for NaN in positions
            if torch.isnan(object_pos).any() or torch.isnan(goal_pos).any():
                print(f"NaN in positions - object_pos: {object_pos[0]}, goal_pos: {goal_pos[0]}")

            # Ensure quaternions are normalized before computing differences
            object_rot = torch.nn.functional.normalize(object_rot, p=2, dim=1)
            goal_rot = torch.nn.functional.normalize(goal_rot, p=2, dim=1)

            # Check for NaN in rotations
            if torch.isnan(object_rot).any() or torch.isnan(goal_rot).any():
                print(f"NaN in rotations - object_rot: {object_rot[0]}, goal_rot: {goal_rot[0]}")

            # Rotation reward - compute quaternion distance
            # Using dot product method for quaternion distance
            quat_dot = torch.abs(torch.sum(object_rot * goal_rot, dim=1))
            quat_dot = torch.clamp(quat_dot, min=0.0, max=1.0)  # Ensure within valid range

            # Handle edge case where quat_dot is exactly 1.0
            rot_dist = torch.where(
                quat_dot >= 1.0,
                torch.zeros_like(quat_dot),
                2.0 * torch.acos(quat_dot)
            )

            # Avoid division by very small numbers
            rot_reward = self.rot_reward_scale / (torch.abs(rot_dist) + self.rot_eps)

            # Action penalty
            action_penalty = torch.sum(action_tensor ** 2, dim=1) * self.action_penalty_scale

            # Total reward
            rewards = dist_reward + rot_reward + action_penalty

            # Success bonus (based on rotation tolerance)
            success_mask = torch.abs(rot_dist) <= self.success_tolerance
            rewards[success_mask] += self.reach_goal_bonus

            # Fall penalty
            fall_mask = pos_dist >= self.fall_dist
            rewards[fall_mask] += self.fall_penalty

            return rewards

        else:
            # Handle list format (for other simulators)
            rewards = []
            for i, env_state in enumerate(states):
                object_state = env_state["objects"]["block"]
                goal_state = env_state["objects"]["goal"]

                # Get action values
                if isinstance(actions[i], dict):
                    robot_name = list(actions[i].keys())[0]  # Get robot name
                    if "dof_pos_target" in actions[i][robot_name]:
                        action = torch.tensor(list(actions[i][robot_name]["dof_pos_target"].values()))
                    else:
                        action = torch.zeros(16)
                else:
                    action = torch.zeros(16)

                object_pos = torch.tensor(object_state["pos"])
                object_rot = torch.tensor(object_state["rot"])

                goal_pos = torch.tensor(goal_state["pos"])
                goal_rot = torch.tensor(goal_state["rot"])

                # Distance reward (position)
                pos_dist = torch.norm(object_pos - goal_pos, p=2)
                dist_reward = self.dist_reward_scale * pos_dist

                # Rotation reward
                # Normalize quaternions
                object_rot = torch.nn.functional.normalize(object_rot, p=2, dim=0)
                goal_rot = torch.nn.functional.normalize(goal_rot, p=2, dim=0)

                # Compute quaternion distance using dot product
                quat_dot = torch.abs(torch.dot(object_rot, goal_rot))
                quat_dot = torch.clamp(quat_dot, max=1.0)
                rot_dist = 2.0 * torch.acos(quat_dot)

                rot_reward = self.rot_reward_scale / (torch.abs(rot_dist) + self.rot_eps)

                # Action penalty
                action_penalty = torch.sum(action ** 2) * self.action_penalty_scale

                # Total reward
                reward = dist_reward + rot_reward + action_penalty

                # Success bonus (based on rotation tolerance)
                if torch.abs(rot_dist) <= self.success_tolerance:
                    reward += self.reach_goal_bonus

                # Fall penalty
                if pos_dist >= self.fall_dist:
                    reward += self.fall_penalty

                rewards.append(reward.item())

            return torch.tensor(rewards) if rewards else torch.tensor([0.0])

    def termination_fn(self, states):
        """Check if episode should terminate."""
        # Handle TensorState object from IsaacGym
        if hasattr(states, '__class__') and states.__class__.__name__ == 'TensorState':
            robot_state = states.robots["allegro_hand"]
            block_state = states.objects["block"]

            # Get positions
            robot_pos = robot_state.root_state[:, :3]  # Shape: (num_envs, 3)
            block_pos = block_state.root_state[:, :3]  # Shape: (num_envs, 3)

            # Check for NaN values
            has_nan = torch.isnan(robot_pos).any(dim=1) | torch.isnan(block_pos).any(dim=1)

            # Terminate if object falls too far or if NaN detected
            goal_dist = torch.norm(block_pos - robot_pos, p=2, dim=1)
            terminations = (goal_dist >= self.fall_dist) | has_nan

            return terminations

        else:
            # Handle list format (for other simulators)
            terminations = []
            for env_state in states:
                robot_state = env_state["robots"]["allegro_hand"]
                block_state = env_state["objects"]["block"]
                robot_pos = torch.tensor(robot_state["pos"])
                block_pos = torch.tensor(block_state["pos"])

                # Terminate if object falls too far
                goal_dist = torch.norm(block_pos - robot_pos, p=2)
                terminate = goal_dist >= self.fall_dist

                terminations.append(terminate.item() if isinstance(terminate, torch.Tensor) else terminate)

            return torch.tensor(terminations) if terminations else torch.tensor([False])

    def build_scene(self, config=None):
        """Build the scene - this is called by the simulator."""
        # Initialize any task-specific scene parameters
        self._prev_actions = None
        self._success_count = 0

    def reset(self, env_ids=None):
        """Reset task-specific state."""
        # Randomize goal orientation
        if env_ids is None:
            num_envs = 1
        else:
            num_envs = len(env_ids)

        # Generate random goal orientations (normalized quaternions)
        rand_floats = torch.rand((num_envs, 4)) * 2.0 - 1.0  # [-1, 1]
        new_rot = torch.nn.functional.normalize(rand_floats, p=2, dim=1)

        # Ensure w component is positive for consistency
        mask = new_rot[:, 0] < 0
        new_rot[mask] = -new_rot[mask]

        # This will be used by the simulator to set goal orientations
        self._goal_rotations = new_rot

        # Reset previous actions
        if self._prev_actions is None:
            self._prev_actions = torch.zeros((num_envs, 16), dtype=torch.float32)
        else:
            if env_ids is not None:
                self._prev_actions[env_ids] = 0.0

    def post_reset(self):
        """Called after reset."""
        pass

    def _get_prev_actions(self, env_idx):
        """Get previous actions for a specific environment."""
        if hasattr(self, '_prev_actions') and self._prev_actions is not None:
            if env_idx < len(self._prev_actions):
                return self._prev_actions[env_idx]
        return torch.zeros(16, dtype=torch.float32)
