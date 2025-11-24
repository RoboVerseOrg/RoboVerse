"""Stage 1: Simple Approach and Grasp task with gripper control.

This task focuses on learning to approach the object, grasp it with gripper, and lift it.
Simple gripper control: close when near the object.
"""

from __future__ import annotations

from copy import deepcopy

import torch
from loguru import logger as log

from metasim.constants import PhysicStateType
from metasim.scenario.objects import PrimitiveCubeCfg, RigidObjCfg
from metasim.scenario.scenario import ScenarioCfg, SimParamCfg
from metasim.task.registry import register_task
from roboverse_pack.tasks.pick_place.base import DEFAULT_CONFIG, PickPlaceBase


@register_task("pick_place.approach_grasp_simple", "pick_place_approach_grasp_simple")
class PickPlaceApproachGraspSimple(PickPlaceBase):
    """Simple Approach and Grasp task with gripper control.

    This task focuses on:
    - Approaching the object
    - Grasping the object with simple gripper control (close when near)

    Success condition: Object is grasped (reward given when entering grasp state).
    Episode terminates if object is released.
    """

    GRASP_DISTANCE_THRESHOLD = 0.02  # Distance threshold for both grasp check and gripper closing
    GRASP_HISTORY_WINDOW = 5  # Number of frames to check for stable grasp

    # Joint2 lift parameters (for franka: panda_joint2)
    JOINT2_LIFT_OFFSET = 0.5  # Amount to lift joint2 when grasped (positive = lift up)
    JOINT2_LIFT_KP = 0.2  # Proportional gain for joint2 lift control
    JOINT2_LIFT_MAX_DELTA = 0.3  # Maximum change per step

    DEFAULT_CONFIG_SIMPLE = deepcopy(DEFAULT_CONFIG)
    DEFAULT_CONFIG_SIMPLE["reward_config"]["scales"].update({
        "gripper_approach": 0.5,
        "grasp_reward": 4.0,
    })
    DEFAULT_CONFIG_SIMPLE["grasp_config"] = {
        "grasp_check_distance": GRASP_DISTANCE_THRESHOLD,
        "gripper_close_distance": GRASP_DISTANCE_THRESHOLD,
    }

    scenario = ScenarioCfg(
        objects=[
            PrimitiveCubeCfg(
                name="object",
                size=(0.04, 0.04, 0.04),
                mass=0.02,
                physics=PhysicStateType.RIGIDBODY,
                color=(1.0, 0.0, 0.0),
            ),
            PrimitiveCubeCfg(
                name="table",
                size=(0.2, 0.3, 0.4),
                mass=10.0,
                physics=PhysicStateType.RIGIDBODY,
                color=(0.8, 0.6, 0.4),
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
        ],
        robots=["franka"],
        sim_params=SimParamCfg(
            dt=0.005,
        ),
        decimation=4,
    )
    max_episode_steps = 200

    def __init__(self, scenario, device=None):
        # Placeholders needed during super().__init__ (reset may be called there)
        self.object_grasped = None
        self.gripper_joint_indices = [0, 1]  # panda_finger_joint1, panda_finger_joint2
        self._grasp_notified = None
        self._distance_history = None  # History buffer for stable grasp check
        self.joint2_name = "panda_joint2"
        self.joint2_index = None
        self.initial_joint_pos = None

        super().__init__(scenario, device)

        # Override reward functions for this task
        self.reward_functions = [
            self._reward_gripper_approach,
            self._reward_grasp,
        ]
        self.reward_weights = [
            self.DEFAULT_CONFIG_SIMPLE["reward_config"]["scales"]["gripper_approach"],
            self.DEFAULT_CONFIG_SIMPLE["reward_config"]["scales"]["grasp_reward"],
        ]

        # Get config values
        grasp_config = self.DEFAULT_CONFIG_SIMPLE["grasp_config"]
        self.grasp_check_distance = grasp_config["grasp_check_distance"]
        self.gripper_close_distance = grasp_config["gripper_close_distance"]

        # Initialize tracking buffers
        self.object_grasped = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._grasp_notified = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._distance_history = torch.full(
            (self.num_envs, self.GRASP_HISTORY_WINDOW),
            float("inf"),
            device=self.device,
        )

        # Find joint2 index
        joint_names = self.handler.get_joint_names(self.robot_name, sort=True)
        if self.joint2_name in joint_names:
            self.joint2_index = joint_names.index(self.joint2_name)
        else:
            log.warning(f"Joint {self.joint2_name} not found, joint2 lift disabled")

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
        if self._grasp_notified is None or self._grasp_notified.shape[0] != self.num_envs:
            self._grasp_notified = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self._distance_history = torch.full(
                (self.num_envs, self.GRASP_HISTORY_WINDOW),
                float("inf"),
                device=self.device,
            )
        else:
            self._grasp_notified[env_ids_tensor] = False
            self._distance_history[env_ids_tensor] = float("inf")

        # Store initial joint positions if not already stored
        if self.initial_joint_pos is None:
            states = self.handler.get_states(mode="tensor")
            self.initial_joint_pos = states.robots[self.robot_name].joint_pos.clone()

        return obs, info

    def step(self, actions):
        """Step with delta control and simple gripper control."""
        current_states = self.handler.get_states(mode="tensor")
        box_pos = current_states.objects["object"].root_state[:, 0:3]
        gripper_pos, _ = self._get_ee_state(current_states)
        gripper_box_dist = torch.norm(gripper_pos - box_pos, dim=-1)

        # Apply delta control
        delta_actions = actions * self._action_scale
        new_actions = self._last_action + delta_actions
        real_actions = torch.clamp(new_actions, self._action_low, self._action_high)

        # Simple gripper control: close when near object
        real_actions = self._apply_simple_gripper_control(real_actions, gripper_box_dist)

        # Apply joint2 lift control if grasped
        if self.object_grasped is not None and self.object_grasped.any() and self.joint2_index is not None:
            real_actions = self._apply_joint2_lift_control(real_actions, current_states)

        # Bypass PickPlaceBase.step to avoid its gripper control logic
        # Call RLTaskEnv.step directly
        # Note: reward functions will be called inside super().step()
        # and they will compute newly_grasped by comparing current state with self.object_grasped
        obs, reward, terminated, time_out, info = super(PickPlaceBase, self).step(real_actions)
        self._last_action = real_actions

        # Update grasp state after step (for next step's comparison)
        updated_states = self.handler.get_states(mode="tensor")
        old_grasped = self.object_grasped.clone()
        self.object_grasped = self._compute_grasp_state(updated_states)

        newly_grasped = self.object_grasped & (~old_grasped)
        newly_released = (~self.object_grasped) & old_grasped

        if newly_grasped.any() and newly_grasped[0]:
            log.info(f"[Env 0] Object grasped! Distance: {gripper_box_dist[0].item():.4f}m")
            self._grasp_notified[newly_grasped] = True

        if newly_released.any() and newly_released[0]:
            log.info(f"[Env 0] Object released! Distance: {gripper_box_dist[0].item():.4f}m")
            self._grasp_notified[newly_released] = False

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

    def _apply_simple_gripper_control(self, actions, gripper_box_dist):
        """Simple gripper control: close when near object."""
        # Close gripper when close to object
        gripper_close = gripper_box_dist < self.gripper_close_distance
        gripper_value_close = torch.tensor(0.0, device=self.device, dtype=actions.dtype)  # Closed
        gripper_value_open = torch.tensor(0.04, device=self.device, dtype=actions.dtype)  # Open

        # Set gripper joints
        for gripper_idx in self.gripper_joint_indices:
            actions[:, gripper_idx] = torch.where(
                gripper_close,
                gripper_value_close,
                gripper_value_open,
            )

        return actions

    def _apply_joint2_lift_control(self, actions, current_states):
        """Apply joint2 lift control when object is grasped."""
        if self.initial_joint_pos is None:
            self.initial_joint_pos = current_states.robots[self.robot_name].joint_pos.clone()

        joint_pos = current_states.robots[self.robot_name].joint_pos
        joint2_idx = self.joint2_index

        # Target position: initial position + lift offset (positive offset lifts up)
        target_lift = self.initial_joint_pos[:, joint2_idx] + self.JOINT2_LIFT_OFFSET
        joint_error = target_lift - joint_pos[:, joint2_idx]

        # Apply proportional control with max delta limit
        desired = joint_pos[:, joint2_idx] + self.JOINT2_LIFT_KP * joint_error
        delta = torch.clamp(
            desired - joint_pos[:, joint2_idx],
            -self.JOINT2_LIFT_MAX_DELTA,
            self.JOINT2_LIFT_MAX_DELTA,
        )
        joint2_value = torch.clamp(
            joint_pos[:, joint2_idx] + delta,
            self._action_low[joint2_idx],
            self._action_high[joint2_idx],
        )

        # Apply lift control only to environments where object is grasped
        actions[self.object_grasped, joint2_idx] = joint2_value[self.object_grasped]

        return actions

    def _compute_grasp_state(self, states):
        """Compute if object is grasped (requires 5 stable frames based on distance only)."""
        box_pos = states.objects["object"].root_state[:, 0:3]
        gripper_pos, _ = self._get_ee_state(states)
        gripper_box_dist = torch.norm(gripper_pos - box_pos, dim=-1)

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

    def _reward_gripper_approach(self, env_states) -> torch.Tensor:
        """Reward for gripper approaching the box."""
        box_pos = env_states.objects["object"].root_state[:, 0:3]
        gripper_pos, _ = self._get_ee_state(env_states)
        gripper_box_dist = torch.norm(box_pos - gripper_pos, dim=-1)

        approach_reward_far = 1 - torch.tanh(gripper_box_dist)
        approach_reward_near = 1 - torch.tanh(gripper_box_dist * 10)
        return approach_reward_far + approach_reward_near

    def _reward_grasp(self, env_states) -> torch.Tensor:
        """Reward for maintaining grasp state (continuous reward while grasped)."""
        # Use cached grasp state (computed in step method)
        return self.object_grasped.float()

    def _get_initial_states(self) -> list[dict] | None:
        """Get initial states for all environments."""
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
