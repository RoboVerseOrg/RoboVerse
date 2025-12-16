"""Stage 3: Track task for trajectory tracking.

Trains trajectory tracking from saved grasp states.
Object is already grasped, only needs to learn trajectory following.
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
from metasim.utils.math import matrix_from_quat
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
    "rotation_tracking": 0.0,
})
# 移除不需要的奖励
DEFAULT_CONFIG_TRACK["reward_config"]["scales"].pop("gripper_approach", None)
DEFAULT_CONFIG_TRACK["reward_config"]["scales"].pop("gripper_close", None)
# Disable randomization for exact state reproduction
DEFAULT_CONFIG_TRACK["randomization"]["box_pos_range"] = 0.0
DEFAULT_CONFIG_TRACK["randomization"]["robot_pos_noise"] = 0.0
DEFAULT_CONFIG_TRACK["randomization"]["joint_noise_range"] = 0.0
DEFAULT_CONFIG_TRACK["trajectory_tracking"]["num_waypoints"] = 3


@register_task("pick_place.track_knife", "pick_place_track_knife")
class PickPlaceTrackKnife(PickPlaceBase):
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
                usd_path="roboverse_data/EmbodiedGenData/all_asset/table/usd/table.usd",
                urdf_path="roboverse_data/EmbodiedGenData/all_asset/table/table.urdf",
                mjcf_path="roboverse_data/EmbodiedGenData/all_asset/table/mjcf/table.xml",
                fix_base_link=True,
            ),
            RigidObjCfg(
                name="bowl",
                scale=(1, 1, 1),
                physics=PhysicStateType.RIGIDBODY,
                usd_path="roboverse_data/EmbodiedGenData/all_asset/bowl/usd/bowl.usd",
                urdf_path="roboverse_data/EmbodiedGenData/all_asset/bowl/bowl.urdf",
                mjcf_path="roboverse_data/EmbodiedGenData/all_asset/bowl/mjcf/bowl.xml",
            ),
            RigidObjCfg(
                name="object",
                scale=(1, 1, 1),
                enabled_gravity=False,
                physics=PhysicStateType.RIGIDBODY,
                usd_path="roboverse_data/EmbodiedGenData/all_asset/ceramic_teapot/usd/ceramic_teapot.usd",
                urdf_path="roboverse_data/EmbodiedGenData/all_asset/ceramic_teapot/ceramic_teapot.urdf",
                mjcf_path="roboverse_data/EmbodiedGenData/all_asset/ceramic_teapot/mjcf/ceramic_teapot.xml",
            ),
            RigidObjCfg(
                name="plate",
                scale=(1, 1, 1),
                physics=PhysicStateType.RIGIDBODY,
                usd_path="roboverse_data/EmbodiedGenData/all_asset/plate/usd/plate.usd",
                urdf_path="roboverse_data/EmbodiedGenData/all_asset/plate/plate.urdf",
                mjcf_path="roboverse_data/EmbodiedGenData/all_asset/plate/mjcf/plate.xml",
            ),
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
                urdf_path="roboverse_pack/tasks/pick_place/marker/axis_marker.urdf",
                mjcf_path="roboverse_pack/tasks/pick_place/marker/axis_marker.xml",
                usd_path="roboverse_pack/tasks/pick_place/marker/axis_marker.usd",
                scale=0.2,
                physics=PhysicStateType.XFORM,
                enabled_gravity=False,
                collision_enabled=False,
                fix_base_link=True,
            ),
            RigidObjCfg(
                name="traj_marker_4",
                urdf_path="roboverse_pack/tasks/pick_place/marker/axis_marker.urdf",
                mjcf_path="roboverse_pack/tasks/pick_place/marker/axis_marker.xml",
                usd_path="roboverse_pack/tasks/pick_place/marker/axis_marker.usd",
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
        self.state_file_path = "/usr1/home/s125mdg56_03/RoboVerse/eval_states/pick_place.approach_grasp_knife_franka_lift_states_198states_20251216_215959.pkl"
        self._loaded_states = None
        self._action_scale = 0.04

        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._device = device

        self.object_grasped = None

        super().__init__(scenario, device)
        self.local_offset = torch.tensor([-0.01538026, 0.1282216, -0.00245847]).to(device)

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
        # import ipdb; ipdb.set_trace()
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
            torch.tensor([-0.025798, -0.496286, 1.093025], device=device),
            torch.tensor([-0.045492, -0.285306, 1.081898], device=device),
            torch.tensor([-0.030328, -0.130204, 1.082140], device=device),
            torch.tensor([-0.015164, 0.064898, 1.052381], device=device),
            torch.tensor([0.130000, 0.160000, 1.082622], device=device),
        ]
        default_rotations = [
            torch.tensor([-0.835365, -0.002910, -0.008611, 0.549620], device=device),
            torch.tensor([-0.317816, -0.002321, 0.001691, 0.948148], device=device),
            torch.tensor([-0.489972, -0.004560, 0.003323, 0.871720], device=device),
            torch.tensor([-0.644740, -0.006638, 0.004836, 0.764358], device=device),
            torch.tensor([-0.776629, -0.008479, 0.006178, 0.629871], device=device),
        ]

        for env_idx, initial_state in enumerate(initial_states):
            if "objects" not in initial_state:
                initial_state["objects"] = {}
                # import ipdb; ipdb.set_trace()
            for i in range(self.num_waypoints):
                # import ipdb; ipdb.set_trace()
                marker_name = f"traj_marker_{i}"
                # if marker_name not in initial_state["objects"]:
                # import ipdb; ipdb.set_trace()
                if i < len(default_positions):
                    initial_state["objects"][marker_name] = {
                        "pos": default_positions[i].clone(),
                        "rot": default_rotations[i].clone(),
                    }

        self._loaded_states = initial_states
        # import ipdb; ipdb.set_trace()
        log.info(f"Loaded {len(initial_states)} initial states from {self.state_file_path}")
        return initial_states

    def step(self, actions):
        """Step with delta control, keeping gripper closed."""
        # print(actions[0])
        delta_actions = actions * self._action_scale
        current_actions = self.handler.get_states().robots[self.robot_name].joint_pos
        new_actions = current_actions + delta_actions
        # print(self._action_low, self._action_high)
        # print(self._action_scale)
        real_actions = torch.clamp(new_actions, self._action_low, self._action_high)

        gripper_value_closed = torch.tensor(0.0, device=self.device, dtype=real_actions.dtype)
        real_actions[:, 0] = gripper_value_closed
        real_actions[:, 1] = gripper_value_closed

        obs, reward, terminated, time_out, info = super(PickPlaceBase, self).step(real_actions)
        # obs, reward, terminated, time_out, info = super().step(real_actions)
        self._last_action = real_actions.clone()

        updated_states = self.handler.get_states(mode="tensor")

        if self.local_offset is not None:
            box_pos = self.get_geometric_center(updated_states)
        else:
            box_pos = updated_states.objects["object"].root_state[:, 0:3]
        gripper_pos, _ = self._get_ee_state(updated_states)

        gripper_box_dist = torch.norm(gripper_pos - box_pos, dim=-1)
        # print(gripper_box_dist)
        # print(self.local_offset)
        is_grasping = gripper_box_dist < self.grasp_check_distance

        self.object_grasped = is_grasping
        newly_released = ~is_grasping

        if newly_released.any() and newly_released[0]:
            log.warning(f"[Env 0] Object released during tracking! Distance: {gripper_box_dist[0].item():.4f}m")

        terminated = terminated | newly_released

        info["grasp_success"] = self.object_grasped
        info["stage"] = torch.full((self.num_envs,), 3, dtype=torch.long, device=self.device)

        return obs, reward, terminated, time_out, info

    def _reward_gripper_close(self, env_states) -> torch.Tensor:
        """Reward for closing gripper when close to box."""
        if self.local_offset is not None:
            box_pos = self.get_geometric_center(env_states)
        else:
            box_pos = env_states.objects["object"].root_state[:, 0:3]

        gripper_pos, _ = self._get_ee_state(env_states)
        gripper_box_dist = torch.norm(box_pos - gripper_pos, dim=-1)

        close_bonus = (gripper_box_dist < 0.02).float()
        return close_bonus

    def _reward_gripper_approach(self, env_states) -> torch.Tensor:
        """Reward for gripper approaching the box."""
        if self.local_offset is not None:
            box_pos = self.get_geometric_center(env_states)
        else:
            box_pos = env_states.objects["object"].root_state[:, 0:3]
        gripper_pos, _ = self._get_ee_state(env_states)
        gripper_box_dist = torch.norm(box_pos - gripper_pos, dim=-1)

        approach_reward_far = 1 - torch.tanh(gripper_box_dist)
        approach_reward_near = 1 - torch.tanh(gripper_box_dist * 10)
        return approach_reward_far + approach_reward_near

    def get_geometric_center(self, current_states):
        """Get the geometric center of the object in the world frame."""
        root_pos = current_states.objects["object"].root_state[:, 0:3]
        root_rot = current_states.objects["object"].root_state[:, 3:7]
        # local_offset = self.local_offset.to(self.device)
        w, x, y, z = root_rot[:, 0], root_rot[:, 1], root_rot[:, 2], root_rot[:, 3]

        q_vec = torch.stack([x, y, z], dim=1)
        # w = w.unsqueeze(1)

        v = self.local_offset.unsqueeze(0)

        t = 2.0 * torch.cross(q_vec, v, dim=1)

        final_vec = v + w.view(-1, 1) * t + torch.cross(q_vec, t, dim=-1)

        return root_pos + final_vec

    def _observation(self, env_states) -> torch.Tensor:
        """Get observation using RoboVerse tensor state."""
        # if self.local_offset is not None:
        #     box_pos = self.get_geometric_center(env_states)  # [num_envs, 3]
        # else:
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
