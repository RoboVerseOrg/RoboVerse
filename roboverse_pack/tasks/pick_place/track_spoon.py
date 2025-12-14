"""Stage 3: Track task for trajectory tracking with spoon object.

This task inherits from PickPlaceBase and implements track functionality
with spoon-specific mesh configurations and saved poses from object_layout.py.
"""

from __future__ import annotations

import os
import pickle
from copy import deepcopy

import numpy as np
import torch
from loguru import logger as log

from metasim.constants import PhysicStateType
from metasim.scenario.objects import RigidObjCfg
from metasim.scenario.scenario import ScenarioCfg, SimParamCfg
from metasim.task.registry import register_task
from roboverse_pack.tasks.pick_place.base import DEFAULT_CONFIG, PickPlaceBase


def load_states_from_pkl(pkl_path: str):
    """Load state list from pkl file."""
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"State file not found: {pkl_path}")

    with open(pkl_path, "rb") as f:
        states_list = pickle.load(f)

    log.info(f"Loaded {len(states_list)} states from {pkl_path}")
    return states_list


def convert_state_dict_to_initial_state(state_dict: dict, device: torch.device, robot_name: str = "franka") -> dict:
    """Convert state dict to initial state format."""
    initial_state = {
        "objects": {},
        "robots": {},
    }

    if "objects" in state_dict and "robots" in state_dict:
        for obj_name, obj_state in state_dict["objects"].items():
            pos = obj_state.get("pos")
            rot = obj_state.get("rot")

            if isinstance(pos, (list, tuple, np.ndarray)):
                pos = torch.tensor(pos, device=device, dtype=torch.float32)
            elif isinstance(pos, torch.Tensor):
                pos = pos.to(device).float()

            if isinstance(rot, (list, tuple, np.ndarray)):
                rot = torch.tensor(rot, device=device, dtype=torch.float32)
            elif isinstance(rot, torch.Tensor):
                rot = rot.to(device).float()

            initial_state["objects"][obj_name] = {
                "pos": pos,
                "rot": rot,
            }

            if "dof_pos" in obj_state:
                initial_state["objects"][obj_name]["dof_pos"] = obj_state["dof_pos"]

        for robot_name_key, robot_state in state_dict["robots"].items():
            pos = robot_state.get("pos")
            rot = robot_state.get("rot")

            if isinstance(pos, (list, tuple, np.ndarray)):
                pos = torch.tensor(pos, device=device, dtype=torch.float32)
            elif isinstance(pos, torch.Tensor):
                pos = pos.to(device).float()

            if isinstance(rot, (list, tuple, np.ndarray)):
                rot = torch.tensor(rot, device=device, dtype=torch.float32)
            elif isinstance(rot, torch.Tensor):
                rot = rot.to(device).float()

            initial_state["robots"][robot_name_key] = {
                "pos": pos,
                "rot": rot,
            }

            if "dof_pos" in robot_state:
                initial_state["robots"][robot_name_key]["dof_pos"] = robot_state["dof_pos"]
    else:
        # Flat format: convert to nested
        for name, entity_state in state_dict.items():
            if name in ["objects", "robots"]:
                continue

            pos = entity_state.get("pos")
            rot = entity_state.get("rot")

            if isinstance(pos, (list, tuple, np.ndarray)):
                pos = torch.tensor(pos, device=device, dtype=torch.float32)
            elif isinstance(pos, torch.Tensor):
                pos = pos.to(device).float()
            elif isinstance(pos, np.ndarray):
                pos = torch.from_numpy(pos).to(device).float()

            if isinstance(rot, (list, tuple, np.ndarray)):
                rot = torch.tensor(rot, device=device, dtype=torch.float32)
            elif isinstance(rot, torch.Tensor):
                rot = rot.to(device).float()
            elif isinstance(rot, np.ndarray):
                rot = torch.from_numpy(rot).to(device).float()

            entity_entry = {
                "pos": pos,
                "rot": rot,
            }

            if "dof_pos" in entity_state:
                entity_entry["dof_pos"] = entity_state["dof_pos"]

            if name == robot_name:
                initial_state["robots"][name] = entity_entry
            else:
                initial_state["objects"][name] = entity_entry

    return initial_state


DEFAULT_CONFIG_TRACK = deepcopy(DEFAULT_CONFIG)
DEFAULT_CONFIG_TRACK["reward_config"]["scales"].update({
    "tracking_approach": 4.0,
    "tracking_progress": 150.0,
    "rotation_tracking": 2.0,
})
# Remove unused rewards
DEFAULT_CONFIG_TRACK["reward_config"]["scales"].pop("gripper_approach", None)
DEFAULT_CONFIG_TRACK["reward_config"]["scales"].pop("gripper_close", None)
# Disable randomization for exact state reproduction
DEFAULT_CONFIG_TRACK["randomization"]["box_pos_range"] = 0.0
DEFAULT_CONFIG_TRACK["randomization"]["robot_pos_noise"] = 0.0
DEFAULT_CONFIG_TRACK["randomization"]["joint_noise_range"] = 0.0
# Increase reach threshold for spoon (more lenient for higher waypoints)
DEFAULT_CONFIG_TRACK["trajectory_tracking"]["reach_threshold"] = 0.08  # Increased from 0.05 to 0.08m


@register_task("pick_place.track_spoon", "pick_place_track_spoon")
class PickPlaceTrackSpoon(PickPlaceBase):
    """Trajectory tracking task for spoon object.

    This task inherits from PickPlaceTrack and customizes:
    - Scenario: Uses spoon mesh, table mesh, and basket from EmbodiedGenData
    - Initial states: Loads poses from saved_poses_20251206_spoon_basket.py and forces gripper closed
    """

    scenario = ScenarioCfg(
        objects=[
            # Use actual spoon mesh from EmbodiedGenData (matches approach_grasp.py)
            RigidObjCfg(
                name="object",
                scale=(1, 1, 1),
                physics=PhysicStateType.RIGIDBODY,
                usd_path="roboverse_data/assets/EmbodiedGenData/new_assets/spoon/1/usd/2f1c3077a8d954e58fc0bf75cf35e849.usd",
                urdf_path="roboverse_data/assets/EmbodiedGenData/new_assets/spoon/1/2f1c3077a8d954e58fc0bf75cf35e849.urdf",
                mjcf_path="roboverse_data/assets/EmbodiedGenData/new_assets/spoon/1/mjcf/2f1c3077a8d954e58fc0bf75cf35e849.xml",
            ),
            # Use actual table mesh from EmbodiedGenData (matches approach_grasp.py)
            RigidObjCfg(
                name="table",
                scale=(1, 1, 1),
                physics=PhysicStateType.RIGIDBODY,
                usd_path="roboverse_data/assets/EmbodiedGenData/demo_assets/table/usd/table.usd",
                urdf_path="roboverse_data/assets/EmbodiedGenData/demo_assets/table/result/table.urdf",
                mjcf_path="roboverse_data/assets/EmbodiedGenData/demo_assets/table/mjcf/table.xml",
                fix_base_link=True,
            ),
            # Basket for visualization (matches approach_grasp.py)
            RigidObjCfg(
                name="basket",
                scale=(1, 1, 1),
                physics=PhysicStateType.RIGIDBODY,
                usd_path="roboverse_data/assets/EmbodiedGenData/new_assets/basket/1/usd/663158968e3f5900af1f6e7cecef24c7.usd",
                urdf_path="roboverse_data/assets/EmbodiedGenData/new_assets/basket/1/663158968e3f5900af1f6e7cecef24c7.urdf",
                mjcf_path="roboverse_data/assets/EmbodiedGenData/new_assets/basket/1/mjcf/663158968e3f5900af1f6e7cecef24c7.xml",
            ),
            # Trajectory waypoint markers
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
    max_episode_steps = 300  # Increased to allow more time to reach all waypoints

    def __init__(self, scenario, device=None, state_file_path=None):
        # Use state_file_path from config if provided, otherwise use default
        if state_file_path is not None:
            self.state_file_path = state_file_path
        else:
            # Default state file path for spoon task
            self.state_file_path = (
                "eval_states/pick_place.approach_grasp_simple_franka_lift_states_154states_20251207_171537.pkl"
            )
        self._loaded_states = None

        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._device = device

        self.object_grasped = None

        super().__init__(scenario, device)

        # Override reach_threshold with more lenient value for spoon task
        self.reach_threshold = DEFAULT_CONFIG_TRACK["trajectory_tracking"]["reach_threshold"]

        self.object_grasped = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.reward_functions = [
            self._reward_trajectory_tracking,
            self._reward_rotation_tracking,
        ]
        self.reward_weights = [
            1.0,
            1.0,  # rotation_tracking weight is already applied inside the function
        ]

    def _prepare_states(self, states, env_ids):
        """Override to disable randomization for track task."""
        return states

    def _get_initial_states(self) -> list[dict] | None:
        """Load initial states from pkl file with spoon-specific enhancements."""
        if self._loaded_states is not None:
            return self._loaded_states

        states_list = load_states_from_pkl(self.state_file_path)

        device = getattr(self, "_device", None) or getattr(self, "device", None)
        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        initial_states = []
        robot_name = "franka"
        for state_dict in states_list:
            initial_state = convert_state_dict_to_initial_state(state_dict, device, robot_name=robot_name)
            initial_states.append(initial_state)

        if len(initial_states) < self.num_envs:
            k = self.num_envs // len(initial_states)
            remainder = self.num_envs % len(initial_states)
            initial_states = initial_states * k + initial_states[:remainder]

        initial_states = initial_states[: self.num_envs]

        # Load trajectory marker positions from saved poses file (spoon-specific enhancement)
        saved_poses_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
            "get_started",
            "output",
            "saved_poses_20251206_spoon_basket.py",
        )

        saved_traj_markers = None
        saved_table = None
        saved_basket = None

        if os.path.exists(saved_poses_path):
            try:
                import importlib.util

                spec = importlib.util.spec_from_file_location("saved_poses", saved_poses_path)
                saved_poses_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(saved_poses_module)
                saved_poses = saved_poses_module.poses

                # Extract trajectory markers
                saved_traj_markers = {}
                for i in range(self.num_waypoints):
                    marker_name = f"traj_marker_{i}"
                    if marker_name in saved_poses["objects"]:
                        saved_traj_markers[marker_name] = saved_poses["objects"][marker_name]

                # Extract table and basket if present
                if "table" in saved_poses["objects"]:
                    saved_table = saved_poses["objects"]["table"]
                if "basket" in saved_poses["objects"]:
                    saved_basket = saved_poses["objects"]["basket"]

                log.info(f"Loaded trajectory markers from {saved_poses_path}")
            except Exception as e:
                log.warning(f"Failed to load saved poses from {saved_poses_path}: {e}")

        # Default waypoint positions (fallback if saved poses not available)
        default_positions = [
            torch.tensor([0.610000, -0.280000, 0.150000], device=device),
            torch.tensor([0.600000, -0.190000, 0.220000], device=device),
            torch.tensor([0.560000, -0.110000, 0.360000], device=device),
            torch.tensor([0.530000, 0.010000, 0.470000], device=device),
            torch.tensor([0.510000, 0.130000, 0.460000], device=device),
        ]
        default_rotations = [
            torch.tensor([1.000000, 0.000000, 0.000000, 0.000000], device=device),
            torch.tensor([1.000000, 0.000000, 0.000000, 0.000000], device=device),
            torch.tensor([0.998750, 0.000000, 0.049979, -0.000000], device=device),
            torch.tensor([1.000000, 0.000000, 0.000000, 0.000000], device=device),
            torch.tensor([0.984726, 0.000000, 0.174108, -0.000000], device=device),
        ]

        for env_idx, initial_state in enumerate(initial_states):
            if "objects" not in initial_state:
                initial_state["objects"] = {}

            # Force gripper to be fully closed for proper grasping
            if "robots" in initial_state and robot_name in initial_state["robots"]:
                robot_state = initial_state["robots"][robot_name]
                if "dof_pos" in robot_state:
                    dof_pos = robot_state["dof_pos"]
                    # Set finger joints to 0.0 (fully closed)
                    if "panda_finger_joint1" in dof_pos:
                        dof_pos["panda_finger_joint1"] = 0.0
                    if "panda_finger_joint2" in dof_pos:
                        dof_pos["panda_finger_joint2"] = 0.0
                    log.debug(
                        f"[Env {env_idx}] Forced gripper closed: finger1={dof_pos.get('panda_finger_joint1', 'N/A')}, finger2={dof_pos.get('panda_finger_joint2', 'N/A')}"
                    )

            # Update table position from saved poses if available
            if saved_table is not None and "table" in initial_state["objects"]:
                initial_state["objects"]["table"]["pos"] = (
                    saved_table["pos"].to(device)
                    if isinstance(saved_table["pos"], torch.Tensor)
                    else torch.tensor(saved_table["pos"], device=device)
                )
                initial_state["objects"]["table"]["rot"] = (
                    saved_table["rot"].to(device)
                    if isinstance(saved_table["rot"], torch.Tensor)
                    else torch.tensor(saved_table["rot"], device=device)
                )

            # Add basket from saved poses if available
            if saved_basket is not None:
                basket_pos = (
                    saved_basket["pos"].to(device)
                    if isinstance(saved_basket["pos"], torch.Tensor)
                    else torch.tensor(saved_basket["pos"], device=device)
                )
                basket_rot = (
                    saved_basket["rot"].to(device)
                    if isinstance(saved_basket["rot"], torch.Tensor)
                    else torch.tensor(saved_basket["rot"], device=device)
                )
                initial_state["objects"]["basket"] = {
                    "pos": basket_pos,
                    "rot": basket_rot,
                }

            # Use trajectory markers from saved poses if available, otherwise use defaults
            for i in range(self.num_waypoints):
                marker_name = f"traj_marker_{i}"
                if marker_name not in initial_state["objects"]:
                    if saved_traj_markers is not None and marker_name in saved_traj_markers:
                        # Use saved marker position
                        marker_data = saved_traj_markers[marker_name]
                        marker_pos = (
                            marker_data["pos"].to(device)
                            if isinstance(marker_data["pos"], torch.Tensor)
                            else torch.tensor(marker_data["pos"], device=device)
                        )
                        marker_rot = (
                            marker_data["rot"].to(device)
                            if isinstance(marker_data["rot"], torch.Tensor)
                            else torch.tensor(marker_data["rot"], device=device)
                        )
                        initial_state["objects"][marker_name] = {
                            "pos": marker_pos,
                            "rot": marker_rot,
                        }
                    elif i < len(default_positions):
                        # Use default position
                        initial_state["objects"][marker_name] = {
                            "pos": default_positions[i].clone(),
                            "rot": default_rotations[i].clone(),
                        }

        self._loaded_states = initial_states
        log.info(f"Loaded {len(initial_states)} initial states from {self.state_file_path}")
        return initial_states

    def reset(self, env_ids=None):
        """Reset environment, keeping object grasped."""
        if env_ids is None or self._last_action is None:
            self._last_action = self._initial_states.robots[self.robot_name].joint_pos[:, :]
            # Force gripper to be closed in _last_action (spoon-specific enhancement)
            self._last_action[:, 0] = 0.0  # panda_finger_joint1
            self._last_action[:, 1] = 0.0  # panda_finger_joint2
        else:
            self._last_action[env_ids] = self._initial_states.robots[self.robot_name].joint_pos[env_ids, :]
            # Force gripper to be closed in _last_action for reset envs (spoon-specific enhancement)
            self._last_action[env_ids, 0] = 0.0  # panda_finger_joint1
            self._last_action[env_ids, 1] = 0.0  # panda_finger_joint2

        if env_ids is None:
            env_ids_tensor = torch.arange(self.num_envs, device=self.device)
            env_ids_list = list(range(self.num_envs))
        else:
            env_ids_tensor = (
                torch.tensor(env_ids, device=self.device) if not isinstance(env_ids, torch.Tensor) else env_ids
            )
            env_ids_list = env_ids if isinstance(env_ids, list) else list(env_ids)

        self.current_waypoint_idx[env_ids_tensor] = 0
        self.waypoints_reached[env_ids_tensor] = False
        self.prev_distance_to_waypoint[env_ids_tensor] = 0.0

        self.object_grasped[env_ids_tensor] = True

        obs, info = super(PickPlaceBase, self).reset(env_ids=env_ids)

        states = self.handler.get_states()
        if env_ids is None:
            env_ids_list = list(range(self.num_envs))
        else:
            env_ids_list = env_ids if isinstance(env_ids, list) else list(env_ids)

        ee_pos, _ = self._get_ee_state(states)
        target_pos = self.waypoint_positions[0].unsqueeze(0).expand(len(env_ids_list), -1)
        self.prev_distance_to_waypoint[env_ids_list] = torch.norm(ee_pos[env_ids_list] - target_pos, dim=-1)

        info["grasp_success"] = self.object_grasped
        info["stage"] = torch.full((self.num_envs,), 3, dtype=torch.long, device=self.device)

        return obs, info

    def step(self, actions):
        """Step with delta control, keeping gripper closed."""
        current_joint_pos = self.handler.get_states().robots[self.robot_name].joint_pos
        delta_actions = actions * self._action_scale
        new_actions = current_joint_pos + delta_actions

        real_actions = torch.clamp(new_actions, self._action_low, self._action_high)

        gripper_value_closed = torch.tensor(0.0, device=self.device, dtype=real_actions.dtype)
        real_actions[:, 0] = gripper_value_closed
        real_actions[:, 1] = gripper_value_closed

        obs, reward, terminated, time_out, info = super(PickPlaceBase, self).step(real_actions)
        self._last_action = real_actions

        # Enhanced grasp detection: check actual gripper joint positions
        updated_states = self.handler.get_states(mode="tensor")
        box_pos = updated_states.objects["object"].root_state[:, 0:3]
        gripper_pos, _ = self._get_ee_state(updated_states)
        gripper_box_dist = torch.norm(gripper_pos - box_pos, dim=-1)

        gripper_joint_pos = updated_states.robots[self.robot_name].joint_pos[:, :2]
        gripper_actually_closed = gripper_joint_pos.mean(dim=-1) < 0.02

        is_grasping = (gripper_box_dist < self.grasp_check_distance) & gripper_actually_closed
        self.object_grasped = is_grasping
        newly_released = ~is_grasping

        if newly_released.any() and newly_released[0]:
            env_idx = 0
            log.warning(
                f"[Env {env_idx}] Object released during tracking! "
                f"Distance: {gripper_box_dist[env_idx].item():.4f}m, "
                f"Gripper joint pos: {gripper_joint_pos[env_idx].cpu().numpy()}, "
                f"Gripper closed: {gripper_actually_closed[env_idx].item()}"
            )

        terminated = terminated | newly_released

        info["grasp_success"] = self.object_grasped
        info["stage"] = torch.full((self.num_envs,), 3, dtype=torch.long, device=self.device)

        return obs, reward, terminated, time_out, info

    def _reward_trajectory_tracking(self, env_states) -> torch.Tensor:
        """Override reward calculation with increased progress reward for later waypoints.

        This addresses the issue where agent gets stuck after waypoint 2 by:
        1. Increasing progress reward multiplier (from 0.1 to 0.3) for better guidance
        2. Adding adaptive scaling for later waypoints to encourage progress
        """
        ee_pos, _ = self._get_ee_state(env_states)
        grasped_mask = self.object_grasped
        tracking_reward = torch.zeros(self.num_envs, device=self.device)

        if grasped_mask.any():
            target_pos = self.waypoint_positions[self.current_waypoint_idx]
            distance = torch.norm(ee_pos - target_pos, dim=-1)

            # Progress reward with higher multiplier for better guidance
            not_already_reached = ~self.waypoints_reached[
                torch.arange(self.num_envs, device=self.device), self.current_waypoint_idx
            ]
            distance_reduction = self.prev_distance_to_waypoint - distance

            # Adaptive progress reward: higher for later waypoints (2, 3, 4) to encourage progress
            # Waypoints 0-1: 0.2 (20%), Waypoints 2+: 0.4 (40%) to overcome larger distances
            # Create multiplier tensor matching num_envs shape
            progress_multiplier = torch.where(
                self.current_waypoint_idx >= 2,
                torch.full((self.num_envs,), 0.4, device=self.device),
                torch.full((self.num_envs,), 0.2, device=self.device),
            )

            progress_reward_component = (
                torch.clamp(distance_reduction * self.w_tracking_progress * progress_multiplier, min=0.0)
                * not_already_reached.float()
                * grasped_mask.float()
            )

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
                from metasim.utils.math import matrix_from_quat

                box_quat = env_states.objects["object"].root_state[:, 3:7]
                box_mat = matrix_from_quat(box_quat).reshape(self.num_envs, 9)
                target_quat = self.waypoint_rotations[self.current_waypoint_idx]
                target_mat = matrix_from_quat(target_quat).reshape(self.num_envs, 9)
                rot_err = torch.norm(target_mat[:, :6] - box_mat[:, :6], dim=-1)
                rotation_reached = rot_err < self.rotation_error_threshold

            # Both distance and rotation must be satisfied to consider as reached
            reached = distance_reached & rotation_reached
            newly_reached = reached & not_already_reached
            # Waypoint reached bonus (one-time large reward when reaching a waypoint)
            waypoint_reached_bonus = newly_reached.float() * self.w_tracking_progress

            # Update prev_distance for next step
            self.prev_distance_to_waypoint[~newly_reached] = distance[~newly_reached]

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

            # Combine all reward components
            tracking_reward = torch.where(
                all_reached, maintain_reward, approach_reward + progress_reward_component + waypoint_reached_bonus
            )

        return tracking_reward
