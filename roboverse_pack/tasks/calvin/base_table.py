from __future__ import annotations

import xml.etree.ElementTree as ET

import gymnasium as gym
import torch

from metasim.scenario.objects import ArticulationObjCfg
from metasim.scenario.robot import BaseActuatorCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.base import BaseTaskEnv
from metasim.task.registry import register_task
from metasim.types import Termination
from metasim.utils.demo_util import get_traj
from metasim.utils.ik_solver import setup_ik_solver
from metasim.utils.tensor_util import array_to_tensor
from roboverse_pack.robots.franka_with_gripper_extension_cfg import FrankaWithGripperExtensionCfg

all_joint_names = {
    "franka": [
        "panda_finger_joint1",
        "panda_finger_joint2",
        "panda_joint1",
        "panda_joint2",
        "panda_joint3",
        "panda_joint4",
        "panda_joint5",
        "panda_joint6",
        "panda_joint7",
    ],
    "table": ["base__button", "base__switch", "base__slide", "base__drawer"],
}


@register_task("calvin.base_table")
class BaseCalvinTableTask(BaseTaskEnv):
    scenario = ScenarioCfg(
        robots=[
            FrankaWithGripperExtensionCfg(
                name="franka",
                default_position=[-0.34, -0.46, 0.24],
                default_orientation=[1, 0, 0, 0],
                actuators={
                    "panda_joint1": BaseActuatorCfg(velocity_limit=2.175, torque_limit=87, stiffness=280, damping=10),
                    "panda_joint2": BaseActuatorCfg(velocity_limit=2.175, torque_limit=87, stiffness=280, damping=10),
                    "panda_joint3": BaseActuatorCfg(velocity_limit=2.175, torque_limit=87, stiffness=280, damping=10),
                    "panda_joint4": BaseActuatorCfg(velocity_limit=2.175, torque_limit=87, stiffness=280, damping=10),
                    "panda_joint5": BaseActuatorCfg(velocity_limit=2.61, torque_limit=12.0, stiffness=200, damping=5),
                    "panda_joint6": BaseActuatorCfg(velocity_limit=2.61, torque_limit=12.0, stiffness=200, damping=5),
                    "panda_joint7": BaseActuatorCfg(velocity_limit=2.61, torque_limit=12.0, stiffness=200, damping=5),
                    "panda_finger_joint1": BaseActuatorCfg(
                        velocity_limit=0.2, torque_limit=20.0, is_ee=True, stiffness=30000, damping=1000
                    ),
                    "panda_finger_joint2": BaseActuatorCfg(
                        velocity_limit=0.2, torque_limit=20.0, is_ee=True, stiffness=30000, damping=1000
                    ),
                },
                default_joint_positions={
                    "panda_joint1": -1.21779206,
                    "panda_joint2": 1.03987646,
                    "panda_joint3": 2.11978261,
                    "panda_joint4": -2.34205014,
                    "panda_joint5": -0.87015947,
                    "panda_joint6": 1.64119353,
                    "panda_joint7": 0.55344866,
                    "panda_finger_joint1": 0.04,
                    "panda_finger_joint2": 0.04,
                },
                control_type="joint_position",
                fix_base_link=True,
                urdf_path="roboverse_data/robots/franka_calvin/panda_longer_finger.urdf",
                # usd_path=None,
                # mjcf_path=None,
                # mjx_mjcf_path=None,
            )
        ],
        decimation=8,
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self._is_initialized=False
        self.ik_solver = setup_ik_solver(self.scenario.robots[0], solver="pyroki", use_seed=False)
        self._articulated_object_joints = {}
        for obj_cfg in self.scenario.objects:
            if isinstance(obj_cfg, ArticulationObjCfg):
                joint_names = self._get_joint_names_from_urdf(obj_cfg.urdf_path)
                self._articulated_object_joints[obj_cfg.name] = joint_names

    def _action_space(self):
        if self.scenario.robots[0].control_type == "joint_position":
            return gym.spaces.Box(low=-1.0, high=1.0, shape=(9,), dtype=float)
        elif self.scenario.robots[0].control_type == "ee_pose":
            return gym.spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=float)
        else:
            raise NotImplementedError

    @staticmethod
    def _get_joint_names_from_urdf(urdf_path: str):
        try:
            tree = ET.parse(urdf_path)
            root = tree.getroot()
            joint_names = []
            for joint in root.findall("joint"):
                if joint.get("type") != "fixed":
                    joint_names.append(joint.get("name"))
            return joint_names
        except (ET.ParseError, FileNotFoundError):
            return []

    def _get_initial_states(self):

        init_states, all_actions, all_states = get_traj(self.traj_filepath, self.scenario.robots[0], self.handler)

        # self.done_states = all_states[:][-1]
        self.global_init_states = init_states

        self.global_done_states = [traj[-1] for traj in all_states]

        """Return per-env initial states (override in subclasses)."""
        return init_states

    def _is_state_equal(self, s1, s2, atol=1e-4):

        if set(s1.keys()) != set(s2.keys()):
            return False

        def compare_entity(e1, e2):

            if not torch.allclose(e1["pos"].cpu(), e2["pos"].cpu(), atol=atol):
                return False

            if not torch.allclose(e1["rot"].cpu(), e2["rot"].cpu(), atol=atol):
                return False

            if "dof_pos" in e1:
                dof1 = e1["dof_pos"]
                dof2 = e2.get("dof_pos", {})

                if len(dof1) != len(dof2):
                    return False

                for joint_name, val1 in dof1.items():
                    if joint_name not in dof2:
                        return False

                    val2 = dof2[joint_name]

                    if isinstance(val1, torch.Tensor):
                        val1 = val1.item()
                    if isinstance(val2, torch.Tensor):
                        val2 = val2.item()

                    if abs(val1 - val2) > atol:
                        return False

            return True

        objs1 = s1.get("objects", {})
        objs2 = s2.get("objects", {})
        if set(objs1.keys()) != set(objs2.keys()):
            return False

        for name in objs1:
            if not compare_entity(objs1[name], objs2[name]):
                return False

        robots1 = s1.get("robots", {})
        robots2 = s2.get("robots", {})
        if set(robots1.keys()) != set(robots2.keys()):
            return False

        for name in robots1:
            if not compare_entity(robots1[name], robots2[name]):
                return False

        return True

    def reset(self, states, env_ids=None):

        if env_ids is None:
            env_ids = list(range(self.num_envs))

        if hasattr(env_ids, "cpu"):
            env_ids = env_ids.cpu().numpy()

        if not hasattr(self, "done_states") or self.done_states is None:
            self.done_states = [None] * self.num_envs

        incoming_done_states = []

        for s in states:
            found_idx = -1
            for i, global_s in enumerate(self.global_init_states):
                # import ipdb; ipdb.set_trace()
                if self._is_state_equal(s, global_s):
                    found_idx = i
                    break

            if found_idx != -1:
                incoming_done_states.append(self.global_done_states[found_idx])
            else:
                # print("Warning: State not found in global cache!")
                incoming_done_states.append(None)

        for i, env_id in enumerate(env_ids):
            self.done_states[env_id] = incoming_done_states[i]

        return super().reset(states=states, env_ids=env_ids)

    def step(self, action):
        try:
            if isinstance(action, list) and action and isinstance(action[0], dict):
                robot_name = self.scenario.robots[0].name
                extracted_actions = [env_act[robot_name] for env_act in action]
                import torch

                if isinstance(extracted_actions[0], torch.Tensor):
                    action = torch.stack(extracted_actions)
                else:
                    action = torch.tensor(extracted_actions, device=self.device)
            elif isinstance(action, dict):
                robot_name = self.scenario.robots[0].name
                if robot_name in action:
                    action = action[robot_name]

        except Exception as e:
            pass

        if self.scenario.robots[0].control_type == "joint_position":
            assert action.shape[-1] == 9, f"Expected action shape (9,), got {action.shape}"

            obs, reward, success, time_out, extras = super().step(action)

            return obs, reward, success, time_out, extras

        elif self.scenario.robots[0].control_type == "ee_pose":
            action = array_to_tensor(action, device=self.device).float()

            curr_state = self.handler.get_states(mode="tensor")
            curr_robot_q = curr_state.robots["franka"].joint_pos

            eff_pos = action[:, :3]
            eff_orn = action[:, 3:7]
            gripper_width = action[:, 7]

            q_solution, ik_succ = self.ik_solver.solve_ik_batch(eff_pos, eff_orn, curr_robot_q)

            actions = self.ik_solver.compose_joint_action(
                q_solution=q_solution,
                gripper_widths=gripper_width,
                current_q=curr_robot_q,
                return_dict=False,
            )

            obs, reward, success, time_out, extras = super().step(actions)

            return obs, reward, success, time_out, extras

        else:
            raise NotImplementedError

    def _terminated(self, env_states) -> Termination:

        success = self.check_state_tolerance(env_states)

        return success

    def check_state_tolerance(self, current_state, pos_tol=0.05, rot_tol=0.1, joint_tol=0.1, verbose=False):

        num_envs = self.num_envs
        device = self.device

        all_success = torch.ones(num_envs, dtype=torch.bool, device=device)

        ref_target_dict = self.done_states[0]

        if "objects" in ref_target_dict:
            for obj_name in ref_target_dict["objects"].keys():
                obj_state = current_state.objects[obj_name].root_state
                curr_pos = obj_state[:, :3]  # (num_envs, 3)
                curr_rot = obj_state[:, 3:7]  # (num_envs, 4)
                target_pos_list = [s["objects"][obj_name]["pos"] for s in self.done_states]
                target_rot_list = [s["objects"][obj_name]["rot"] for s in self.done_states]
                target_pos = torch.stack(target_pos_list).to(device)  # (num_envs, 3)
                target_rot = torch.stack(target_rot_list).to(device)  # (num_envs, 4)

                pos_err = torch.norm(curr_pos - target_pos, dim=1)  # (num_envs,)
                quat_dot = torch.sum(curr_rot * target_rot, dim=1).abs()
                quat_dot = torch.clamp(quat_dot, 0.0, 1.0)
                rot_err = 2.0 * torch.acos(quat_dot)  # (num_envs,)

                obj_success = (pos_err < pos_tol) & (rot_err < rot_tol)
                all_success = all_success & obj_success

                if verbose:
                    failed_indices = torch.nonzero(~obj_success).flatten()
                    if len(failed_indices) > 0:
                        idx = failed_indices[0]
                        # print(f"[Env {idx}] Obj '{obj_name}' Fail: PosErr={pos_err[idx]:.3f}, RotErr={rot_err[idx]:.3f}")

        franka_joint_order = [
            "panda_finger_joint1",
            "panda_finger_joint2",
            "panda_joint1",
            "panda_joint2",
            "panda_joint3",
            "panda_joint4",
            "panda_joint5",
            "panda_joint6",
            "panda_joint7",
        ]

        if "robots" in ref_target_dict:
            for robot_name in ref_target_dict["robots"].keys():
                curr_joints = current_state.robots[robot_name].joint_pos
                target_joints_list = []
                joints_to_check_indices = []
                target_vals_batch = []
                ref_dof_dict = ref_target_dict["robots"][robot_name]["dof_pos"]

                for col_idx, joint_name in enumerate(franka_joint_order):
                    if joint_name in ref_dof_dict:
                        joints_to_check_indices.append(col_idx)

                if not joints_to_check_indices:
                    continue

                for env_i in range(num_envs):
                    env_target_dict = self.done_states[env_i]["robots"][robot_name]["dof_pos"]
                    row_vals = [env_target_dict[franka_joint_order[idx]] for idx in joints_to_check_indices]
                    target_vals_batch.append(row_vals)

                target_joints = torch.tensor(target_vals_batch, device=device, dtype=torch.float32)  # (num_envs, k)
                curr_joints_subset = curr_joints[:, joints_to_check_indices]  # (num_envs, k)
                joint_diff = torch.abs(curr_joints_subset - target_joints)
                max_joint_err, _ = torch.max(joint_diff, dim=1)  # (num_envs,)
                robot_success = max_joint_err < joint_tol
                all_success = all_success & robot_success

                if verbose:
                    failed_indices = torch.nonzero(~robot_success).flatten()
                    if len(failed_indices) > 0:
                        idx = failed_indices[0]
                        # print(f"[Env {idx}] Robot '{robot_name}' Fail: MaxJointErr={max_joint_err[idx]:.3f}")

        return all_success
