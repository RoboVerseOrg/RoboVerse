"""Stage 1: Simple Approach and Grasp task for cutting_tools object.

This task inherits from PickPlaceApproachGraspSimple and customizes it for the cutting_tools object
with specific mesh configurations and saved poses from object_layout.py.
"""

from __future__ import annotations

import importlib.util
import os

import torch
from loguru import logger as log

from metasim.constants import PhysicStateType
from metasim.scenario.objects import RigidObjCfg
from metasim.scenario.scenario import ScenarioCfg, SimParamCfg
from metasim.task.registry import register_task
from metasim.utils.math import quat_apply
from roboverse_pack.tasks.pick_place.approach_grasp import PickPlaceApproachGraspSimple
from roboverse_pack.tasks.pick_place.base import PickPlaceBase


@register_task("pick_place.approach_grasp_simple_cuttingtool", "pick_place_approach_grasp_simple_cuttingtool")
class PickPlaceApproachGraspSimpleCuttingTool(PickPlaceApproachGraspSimple):
    """Simple Approach and Grasp task for cutting_tools object.

    This task inherits from PickPlaceApproachGraspSimple and customizes:
    - Scenario: Uses cutting_tools mesh, table mesh, and basket from EmbodiedGenData
    - Initial states: Loads poses from saved_poses_20251210_cuttingtool.py
    """

    scenario = ScenarioCfg(
        objects=[
            # Use actual cutting_tools mesh from EmbodiedGenData (matches object_layout.py)
            RigidObjCfg(
                name="object",
                scale=(1, 1, 1),
                physics=PhysicStateType.RIGIDBODY,
                usd_path="roboverse_data/assets/EmbodiedGenData/new_assets/cutting_tools/1/usd/c5810e7c2c785fe3940372b205090bad.usd",
                urdf_path="roboverse_data/assets/EmbodiedGenData/new_assets/cutting_tools/1/c5810e7c2c785fe3940372b205090bad.urdf",
                mjcf_path="roboverse_data/assets/EmbodiedGenData/new_assets/cutting_tools/1/mjcf/c5810e7c2c785fe3940372b205090bad.xml",
            ),
            # Use actual table mesh from EmbodiedGenData (matches object_layout.py)
            RigidObjCfg(
                name="table",
                scale=(1, 1, 1),
                physics=PhysicStateType.RIGIDBODY,
                usd_path="roboverse_data/assets/EmbodiedGenData/demo_assets/table/usd/table.usd",
                urdf_path="roboverse_data/assets/EmbodiedGenData/demo_assets/table/result/table.urdf",
                mjcf_path="roboverse_data/assets/EmbodiedGenData/demo_assets/table/mjcf/table.xml",
                fix_base_link=True,
            ),
            # Basket for visualization (matches object_layout.py)
            RigidObjCfg(
                name="basket",
                scale=(1, 1, 1),
                physics=PhysicStateType.RIGIDBODY,
                usd_path="roboverse_data/assets/EmbodiedGenData/new_assets/basket/1/usd/663158968e3f5900af1f6e7cecef24c7.usd",
                urdf_path="roboverse_data/assets/EmbodiedGenData/new_assets/basket/1/663158968e3f5900af1f6e7cecef24c7.urdf",
                mjcf_path="roboverse_data/assets/EmbodiedGenData/new_assets/basket/1/mjcf/663158968e3f5900af1f6e7cecef24c7.xml",
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
        super().__init__(scenario, device)

        # Grasp point offset for cuttingtool: point to the handle center instead of object center
        # Cutting tools typically have handles at one end. The offset is in the object's local frame.
        # Adjust these values based on your cuttingtool mesh:
        # - Negative x typically points toward the handle end (assuming tool extends along x-axis)
        # - You may need to adjust based on the actual mesh orientation
        # Default: [-0.08, 0.0, 0.0] means 8cm along negative x-axis (toward handle)
        # If your tool is oriented differently, adjust the offset accordingly
        self.grasp_point_offset_local = torch.tensor([-0.08, 0.0, 0.0])  # [x, y, z] in object frame

    def _get_grasp_point(self, states):
        """Get the grasp point for the cuttingtool (handle center) instead of object center.

        Args:
            states: Environment states

        Returns:
            grasp_point: (B, 3) grasp point in world coordinates (handle center)
        """
        # Get object center position and rotation
        box_pos = states.objects["object"].root_state[:, 0:3]  # (B, 3) - center of object
        box_quat = states.objects["object"].root_state[:, 3:7]  # (B, 4) wxyz - object orientation

        # Move offset to correct device and expand to batch size
        offset_local = (
            self.grasp_point_offset_local.to(box_pos.device).unsqueeze(0).expand(box_pos.shape[0], -1)
        )  # (B, 3)

        # Transform grasp point offset from object local frame to world frame
        # quat_apply rotates a vector by a quaternion
        # This accounts for the object's rotation in the world
        offset_world = quat_apply(box_quat, offset_local)  # (B, 3)

        # Add offset to object center to get grasp point (handle position)
        grasp_point = box_pos + offset_world  # (B, 3)

        return grasp_point

    def step(self, actions):
        """Step with delta control and simple gripper control, using handle center as grasp point."""
        current_states = self.handler.get_states(mode="tensor")

        # Use grasp point (handle center) instead of object center
        grasp_point = self._get_grasp_point(current_states)  # (B, 3)
        gripper_pos, _ = self._get_ee_state(current_states)

        # Calculate 3D Euclidean distance between gripper and grasp point (handle)
        gripper_box_dist = torch.norm(gripper_pos - grasp_point, dim=-1)

        # Apply delta control
        delta_actions = actions * self._action_scale
        new_actions = self._last_action + delta_actions
        real_actions = torch.clamp(new_actions, self._action_low, self._action_high)

        # Simple gripper control: close when near grasp point (ONLY 3D distance, NO z-axis condition)
        real_actions = self._apply_simple_gripper_control(real_actions, gripper_box_dist, dist_z=None)

        # Apply joint2 lift control if grasped
        if self.object_grasped is not None and self.object_grasped.any() and self.joint2_index is not None:
            real_actions = self._apply_joint2_lift_control(real_actions, current_states)

        # Bypass PickPlaceBase.step to avoid its gripper control logic
        obs, reward, terminated, time_out, info = super(PickPlaceBase, self).step(real_actions)
        self._last_action = real_actions

        # Update grasp state after step
        updated_states = self.handler.get_states(mode="tensor")
        old_grasped = self.object_grasped.clone()
        self.object_grasped = self._compute_grasp_state(updated_states)

        newly_grasped = self.object_grasped & (~old_grasped)
        newly_released = (~self.object_grasped) & old_grasped

        if newly_grasped.any() and newly_grasped[0]:
            log.info(f"[Env 0] Object grasped! Distance: {gripper_box_dist[0].item():.4f}m")

        if newly_released.any() and newly_released[0]:
            log.info(f"[Env 0] Object released! Distance: {gripper_box_dist[0].item():.4f}m")

        # Terminate episode if object is released
        terminated = terminated | newly_released

        # Track lift state: check if joint2 has been lifted significantly
        lift_active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if self.joint2_index is not None and self.initial_joint_pos is not None:
            current_joint2 = updated_states.robots[self.robot_name].joint_pos[:, self.joint2_index]
            initial_joint2 = self.initial_joint_pos[:, self.joint2_index]
            # Lift is active if joint2 has moved up significantly (more than 0.1 radians)
            lift_active = (current_joint2 - initial_joint2) > 0.1

        info["grasp_success"] = self.object_grasped
        info["lift_active"] = lift_active
        info["stage"] = torch.full((self.num_envs,), 1, dtype=torch.long, device=self.device)

        return obs, reward, terminated, time_out, info

    def _compute_grasp_state(self, states):
        """Compute if object is grasped, using handle center as grasp point."""
        # Use grasp point (handle center) instead of object center
        grasp_point = self._get_grasp_point(states)  # (B, 3)
        gripper_pos, _ = self._get_ee_state(states)
        gripper_box_dist = torch.norm(gripper_pos - grasp_point, dim=-1)

        # Update rolling distance history
        if self._distance_history is None or self._distance_history.shape[0] != self.num_envs:
            self._distance_history = torch.full(
                (self.num_envs, self.GRASP_HISTORY_WINDOW),
                float("inf"),
                device=self.device,
            )
        self._distance_history = torch.roll(self._distance_history, shifts=-1, dims=1)
        self._distance_history[:, -1] = gripper_box_dist

        # Object is grasped if distance has been stable (close) for 5 frames
        stable_grasp = (self._distance_history < self.grasp_check_distance).all(dim=1)
        is_grasping = stable_grasp

        return is_grasping

    def _get_initial_states(self) -> list[dict] | None:
        """Get initial states for all environments.

        Uses saved poses from object_layout.py. Loads cutting_tools, table, basket, and trajectory markers
        from saved_poses_20251210_cuttingtool.py.
        """
        # Add path to saved poses
        saved_poses_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
            "get_started",
            "output",
            "saved_poses_20251210_cuttingtool.py",
        )
        if os.path.exists(saved_poses_path):
            # Load saved poses dynamically
            spec = importlib.util.spec_from_file_location("saved_poses", saved_poses_path)
            saved_poses_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(saved_poses_module)
            saved_poses = saved_poses_module.poses
        else:
            # Fallback to default poses if saved file not found
            log.warning(f"Saved poses file not found at {saved_poses_path}, using default poses")
            saved_poses = None

        if saved_poses is not None:
            # Use saved poses from object_layout.py
            init = []
            for _ in range(self.num_envs):
                env_state = {
                    "objects": {
                        # Cutting_tools as the object to pick
                        "object": saved_poses["objects"]["cutting_tools"],
                        "table": saved_poses["objects"]["table"],
                        # Basket for visualization (if present in saved poses)
                        "basket": saved_poses["objects"].get(
                            "basket", saved_poses["objects"]["table"]
                        ),  # Fallback to table if basket not found
                        # Include trajectory markers if present
                        "traj_marker_0": saved_poses["objects"]["traj_marker_0"],
                        "traj_marker_1": saved_poses["objects"]["traj_marker_1"],
                        "traj_marker_2": saved_poses["objects"]["traj_marker_2"],
                        "traj_marker_3": saved_poses["objects"]["traj_marker_3"],
                        "traj_marker_4": saved_poses["objects"]["traj_marker_4"],
                    },
                    "robots": {
                        "franka": saved_poses["robots"]["franka"],
                    },
                }
                init.append(env_state)
        else:
            # Default poses (fallback)
            init = [
                {
                    "objects": {
                        "object": {
                            "pos": torch.tensor([0.654277, -0.345737, 0.020000]),
                            "rot": torch.tensor([0.706448, -0.031607, 0.706347, 0.031698]),
                        },
                        "table": {
                            "pos": torch.tensor([0.499529, 0.253941, 0.200000]),
                            "rot": torch.tensor([0.999067, -0.000006, 0.000009, 0.043198]),
                        },
                        # Trajectory waypoints (world coordinates)
                        "traj_marker_0": {
                            "pos": torch.tensor([0.610000, -0.280000, 0.150000]),
                            "rot": torch.tensor([1.000000, 0.000000, 0.000000, 0.000000]),
                        },
                        "traj_marker_1": {
                            "pos": torch.tensor([0.600000, -0.190000, 0.220000]),
                            "rot": torch.tensor([1.000000, 0.000000, 0.000000, 0.000000]),
                        },
                        "traj_marker_2": {
                            "pos": torch.tensor([0.560000, -0.110000, 0.360000]),
                            "rot": torch.tensor([0.998750, 0.000000, 0.049979, -0.000000]),
                        },
                        "traj_marker_3": {
                            "pos": torch.tensor([0.530000, 0.010000, 0.470000]),
                            "rot": torch.tensor([1.000000, 0.000000, 0.000000, 0.000000]),
                        },
                        "traj_marker_4": {
                            "pos": torch.tensor([0.510000, 0.130000, 0.460000]),
                            "rot": torch.tensor([0.984726, 0.000000, 0.174108, -0.000000]),
                        },
                    },
                    "robots": {
                        "franka": {
                            "pos": torch.tensor([-0.025, -0.160, 0.018054]),
                            "rot": torch.tensor([1.000000, 0.000000, 0.000000, 0.000000]),
                            "dof_pos": {
                                "panda_finger_joint1": 0.04,
                                "panda_finger_joint2": 0.04,
                                "panda_joint1": 0.0,
                                "panda_joint2": -0.785398,
                                "panda_joint3": 0.0,
                                "panda_joint4": -2.356194,
                                "panda_joint5": 0.0,
                                "panda_joint6": 1.570796,
                                "panda_joint7": 0.785398,
                            },
                        },
                    },
                }
                for _ in range(self.num_envs)
            ]

        return init
