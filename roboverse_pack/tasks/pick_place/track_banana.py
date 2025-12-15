"""Stage 3: Track task for trajectory tracking.

Trains trajectory tracking from saved grasp states.
Object is already grasped, only needs to learn trajectory following.
"""

from __future__ import annotations

import os
import pickle
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import yaml
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


@register_task("pick_place.track_banana", "pick_place_track_banana")
class PickPlaceTrackBanana(PickPlaceBase):
    """Trajectory tracking task from grasp states.

    Assumes object is already grasped, only learns trajectory following.
    Initial states loaded from pkl file.
    """

    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="table",
                scale=(1, 1, 1),
                physics=PhysicStateType.RIGIDBODY,
                usd_path="roboverse_data/assets/EmbodiedGenData/demo_assets/table/usd/table.usd",
                urdf_path="roboverse_data/assets/EmbodiedGenData/demo_assets/table/result/table.urdf",
                mjcf_path="roboverse_data/assets/EmbodiedGenData/demo_assets/table/mjcf/table.xml",
            ),
            RigidObjCfg(
                name="lamp",
                scale=(1, 1, 1),
                physics=PhysicStateType.RIGIDBODY,
                usd_path="roboverse_data/assets/EmbodiedGenData/new_assets/lighting_fixtures/1/usd/0a4489b1a2875c82a580f8b62d346e08.usd",
                urdf_path="roboverse_data/assets/EmbodiedGenData/new_assets/lighting_fixtures/1/0a4489b1a2875c82a580f8b62d346e08.urdf",
                mjcf_path="roboverse_data/assets/EmbodiedGenData/new_assets/lighting_fixtures/1/mjcf/0a4489b1a2875c82a580f8b62d346e08.xml",
            ),
            RigidObjCfg(
                name="basket",
                scale=(1, 1, 1),
                physics=PhysicStateType.RIGIDBODY,
                usd_path="roboverse_data/assets/EmbodiedGenData/new_assets/basket/1/usd/663158968e3f5900af1f6e7cecef24c7.usd",
                urdf_path="roboverse_data/assets/EmbodiedGenData/new_assets/basket/1/663158968e3f5900af1f6e7cecef24c7.urdf",
                mjcf_path="roboverse_data/assets/EmbodiedGenData/new_assets/basket/1/mjcf/663158968e3f5900af1f6e7cecef24c7.xml",
            ),
            RigidObjCfg(
                name="bowl",
                scale=(1, 1, 1),
                physics=PhysicStateType.RIGIDBODY,
                usd_path="roboverse_data/assets/EmbodiedGenData/new_assets/bowl/1/usd/0f296af3df66565c9e1a7c2bc7b35d72.usd",
                urdf_path="roboverse_data/assets/EmbodiedGenData/new_assets/bowl/1/0f296af3df66565c9e1a7c2bc7b35d72.urdf",
                mjcf_path="roboverse_data/assets/EmbodiedGenData/new_assets/bowl/1/mjcf/0f296af3df66565c9e1a7c2bc7b35d72.xml",
            ),
            RigidObjCfg(
                name="object",
                scale=(1, 1, 1),
                physics=PhysicStateType.RIGIDBODY,
                usd_path="roboverse_data/assets/EmbodiedGenData/demo_assets/banana/usd/banana.usd",
                urdf_path="roboverse_data/assets/EmbodiedGenData/demo_assets/banana/result/banana.urdf",
                mjcf_path="roboverse_data/assets/EmbodiedGenData/demo_assets/banana/mjcf/banana.xml",
            ),
            RigidObjCfg(
                name="screw_driver",
                scale=(1.5, 1.5, 1.5),
                physics=PhysicStateType.RIGIDBODY,
                usd_path="roboverse_data/assets/EmbodiedGenData/new_assets/screwdriver/1/usd/ae51f060e3455e9f84a4fec81cc9284b.usd",
                urdf_path="roboverse_data/assets/EmbodiedGenData/new_assets/screwdriver/1/ae51f060e3455e9f84a4fec81cc9284b.urdf",
                mjcf_path="roboverse_data/assets/EmbodiedGenData/new_assets/screwdriver/1/mjcf/ae51f060e3455e9f84a4fec81cc9284b.xml",
            ),
            RigidObjCfg(
                name="spoon",
                scale=(1, 1, 1),
                physics=PhysicStateType.RIGIDBODY,
                usd_path="roboverse_data/assets/EmbodiedGenData/new_assets/spoon/1/usd/2f1c3077a8d954e58fc0bf75cf35e849.usd",
                urdf_path="roboverse_data/assets/EmbodiedGenData/new_assets/spoon/1/2f1c3077a8d954e58fc0bf75cf35e849.urdf",
                mjcf_path="roboverse_data/assets/EmbodiedGenData/new_assets/spoon/1/mjcf/2f1c3077a8d954e58fc0bf75cf35e849.xml",
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

    def __init__(self, scenario, device=None):
        # Start from current file and walk upward to find RoboVerse root
        root = Path(__file__).resolve()
        while root != root.parent:
            if (root / "roboverse_learn").exists():
                roboverseroot = root
                break
            root = root.parent
        else:
            raise RuntimeError("Could not locate RoboVerse root directory")

        # Now construct full path to the YAML
        config_path = roboverseroot / "roboverse_learn" / "rl" / "fast_td3" / "configs" / "track_banana.yaml"

        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        self.state_file_path = cfg["state_file_path"]
        self._loaded_states = None

        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._device = device

        self.object_grasped = None

        super().__init__(scenario, device)

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
        """Load initial states from pkl file."""
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

        # Default waypoint positions
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

            for i in range(self.num_waypoints):
                marker_name = f"traj_marker_{i}"
                if marker_name not in initial_state["objects"]:
                    if i < len(default_positions):
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
        else:
            self._last_action[env_ids] = self._initial_states.robots[self.robot_name].joint_pos[env_ids, :]

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
        current_joint_pos = self.handler.get_states(mode="tensor").robots[self.robot_name].joint_pos
        delta_actions = actions * self._action_scale
        new_actions = current_joint_pos + delta_actions
        real_actions = torch.clamp(new_actions, self._action_low, self._action_high)

        # delta_actions = actions * self._action_scale
        # new_actions = self._last_action + delta_actions

        # delta_actions = actions * self._action_scale
        # new_actions = self._last_action + delta_actions

        gripper_value_closed = torch.tensor(0.0, device=self.device, dtype=real_actions.dtype)
        real_actions[:, 0] = gripper_value_closed
        real_actions[:, 1] = gripper_value_closed

        obs, reward, terminated, time_out, info = super(PickPlaceBase, self).step(real_actions)
        self._last_action = real_actions

        updated_states = self.handler.get_states(mode="tensor")
        box_pos = updated_states.objects["object"].root_state[:, 0:3]
        gripper_pos, _ = self._get_ee_state(updated_states)
        gripper_box_dist = torch.norm(gripper_pos - box_pos, dim=-1)
        is_grasping = gripper_box_dist < self.grasp_check_distance

        self.object_grasped = is_grasping
        newly_released = ~is_grasping

        if newly_released.any() and newly_released[0]:
            log.warning(f"[Env 0] Object released during tracking! Distance: {gripper_box_dist[0].item():.4f}m")

        terminated = terminated | newly_released

        info["grasp_success"] = self.object_grasped
        info["stage"] = torch.full((self.num_envs,), 3, dtype=torch.long, device=self.device)

        return obs, reward, terminated, time_out, info
