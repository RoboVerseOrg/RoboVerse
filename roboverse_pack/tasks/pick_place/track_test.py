"""Trajectory tracking task that assumes the object is already grasped.

The hand maintains a stable grasp while following the predefined waypoint trajectory.
The hand always uses the saved grasp configuration (closed fingers).
"""

from __future__ import annotations

import torch
from loguru import logger as log

from metasim.constants import PhysicStateType
from metasim.scenario.objects import PrimitiveCubeCfg, RigidObjCfg
from metasim.scenario.scenario import ScenarioCfg, SimParamCfg
from metasim.task.registry import register_task
from metasim.utils.math import matrix_from_quat
from roboverse_pack.tasks.pick_place.hand_trajectory import TrajectoryTrackingTaskBase


@register_task("track_grasp_hand_relative", "trackgrasphandrelative")
class TrackGraspHandRelative(TrajectoryTrackingTaskBase):
    """Trajectory tracking task that assumes the object is already grasped.

    The hand maintains a stable grasp while following the predefined waypoint trajectory.
    The hand always uses the saved grasp configuration (closed fingers).
    """

    HAND_MARKER_NAME = "hand_debug_marker"

    # Saved finger positions from vega robot (L hand fingers) - saved_poses_20251126_101518.py
    SAVED_FINGER_POSITIONS = {
        "L_ff_j1": -0.551686,
        "L_ff_j2": -0.621416,
        "L_lf_j1": 0.023751,
        "L_lf_j2": 0.027291,
        "L_mf_j1": -0.551840,
        "L_mf_j2": -0.625254,
        "L_rf_j1": 0.014122,
        "L_rf_j2": 0.015942,
        "L_th_j0": 0.693886,
        "L_th_j1": 0.149071,
        "L_th_j2": 0.201562,
    }

    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="object",
                scale=(0.3, 0.3, 0.3),
                physics=PhysicStateType.RIGIDBODY,
                usd_path="/home/balen/murphy/isaaclab_rv/2/RoboVerse/roboverse_pack/whale_doll/whale_doll.usd",
                urdf_path="roboverse_pack/whale_doll/whale_doll.urdf",
                default_position=(0.104252, -0.076198, 0.846706),
                default_orientation=(0.454115, 0.132146, 0.231502, -0.850132),
            ),
            PrimitiveCubeCfg(
                name="wall",
                size=(0.8, 0.1, 0.2),
                mass=1000.0,
                physics=PhysicStateType.RIGIDBODY,
                color=(0.7, 0.7, 0.7),
                fix_base_link=True,
                default_position=(0.532921, -0.217400, 0.946513),
                default_orientation=(1, 0.0, 0.0, 0.0),
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
                default_position=(0.560000, -0.250000, 0.399868),
                default_orientation=(1.000000, -0.000000, -0.000000, 0.000000),
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
    max_episode_steps = 150

    def __init__(self, scenario, device=None):
        # Placeholder needed during super().__init__ (reset may be called there)
        self.saved_finger_targets = None

        super().__init__(scenario, device)

        # Override reward functions for this task
        self.reward_functions = [
            self._reward_trajectory_tracking,
        ]
        self.reward_weights = [
            1.0,
        ]

        # Initialize saved finger positions tensor
        self.saved_finger_targets = None

    def reset(self, env_ids=None):
        """Reset environment and initialize finger positions."""
        obs, info = super().reset(env_ids=env_ids)

        # Initialize saved finger positions if not already done
        if self.saved_finger_targets is None and hasattr(self, 'left_hand_finger_joint_names'):
            num_fingers = len(self.left_hand_finger_joint_indices)
            self.saved_finger_targets = torch.zeros((self.num_envs, num_fingers), device=self.device)
            for i, joint_name in enumerate(self.left_hand_finger_joint_names):
                if i < num_fingers and joint_name in self.SAVED_FINGER_POSITIONS:
                    self.saved_finger_targets[:, i] = self.SAVED_FINGER_POSITIONS[joint_name]

        return obs, info

    def step(self, actions):
        """Step with delta control and hand control (always use saved finger positions)."""
        current_states = self.handler.get_states(mode="tensor")
        
        # Apply delta control
        delta_actions = actions * self._action_scale
        new_actions = self._last_action + delta_actions
        real_actions = torch.clamp(new_actions, self._action_low, self._action_high)

        # Always use saved finger positions (task assumes object is already grasped)
        if self.saved_finger_targets is not None:
            for i, joint_idx in enumerate(self.left_hand_finger_joint_indices):
                if i < self.saved_finger_targets.shape[1]:
                    real_actions[:, joint_idx] = self.saved_finger_targets[:, i]

        # Bypass TrajectoryTrackingTaskBase.step to avoid its hand control logic
        obs, reward, terminated, time_out, info = super(TrajectoryTrackingTaskBase, self).step(real_actions)
        self._last_action = real_actions

        # Check if all waypoints are reached - terminate immediately if so
        all_waypoints_reached = self.waypoints_reached.all(dim=1)
        if all_waypoints_reached.any():
            if all_waypoints_reached[0]:
                log.info("[Env 0] All waypoints reached! Terminating episode.")
            terminated = terminated | all_waypoints_reached

        info["stage"] = torch.full((self.num_envs,), 1, dtype=torch.long, device=self.device)
        info["all_waypoints_reached"] = all_waypoints_reached

        return obs, reward, terminated, time_out, info



    def _reward_trajectory_tracking(self, env_states) -> torch.Tensor:
        """Reward for tracking waypoints."""
        # Use object position for distance calculation (object is already grasped)
        object_pos = env_states.objects["object"].root_state[:, 0:3]  # (B, 3)
        tracking_reward = torch.zeros(self.num_envs, device=self.device)

        target_pos = self.waypoint_positions[self.current_waypoint_idx]
        distance = torch.norm(object_pos - target_pos, dim=-1)

        approach_reward = (1 - torch.tanh(1.0 * distance)) * self.w_tracking_approach

        reached = distance < self.reach_threshold
        newly_reached = reached & (
            ~self.waypoints_reached[torch.arange(self.num_envs, device=self.device), self.current_waypoint_idx]
        )
        progress_reward = newly_reached.float() * self.w_tracking_progress

        if newly_reached.any():
            if newly_reached[0]:
                wp_idx = self.current_waypoint_idx[0].item()
                log.info(
                    f"[Env 0] Reached waypoint #{wp_idx}! Distance: {distance[0].item():.4f}m < {self.reach_threshold}m"
                )

            self.waypoints_reached[newly_reached, self.current_waypoint_idx[newly_reached]] = True

            can_advance = newly_reached & (self.current_waypoint_idx < self.num_waypoints - 1)

            if can_advance.any() and can_advance[0]:
                old_idx = self.current_waypoint_idx[0].item()
                new_idx = old_idx + 1
                log.info(f"   -> Advancing to waypoint #{new_idx}")

            self.current_waypoint_idx[can_advance] += 1

            if can_advance.any():
                new_target_pos = self.waypoint_positions[self.current_waypoint_idx[can_advance]]
                self.prev_distance_to_waypoint[can_advance] = torch.norm(
                    object_pos[can_advance] - new_target_pos, dim=-1
                )

        maintain_reward = torch.zeros(self.num_envs, device=self.device)
        all_reached = self.waypoints_reached.all(dim=1)

        if all_reached.any():
            last_target_pos = self.waypoint_positions[-1].unsqueeze(0).expand(self.num_envs, -1)
            distance_to_last = torch.norm(object_pos - last_target_pos, dim=-1)

            maintain_reward[all_reached] = torch.where(
                distance_to_last[all_reached] < self.reach_threshold,
                torch.full((all_reached.sum(),), 5, device=self.device),
                (1 - torch.tanh(1.0 * distance_to_last[all_reached])) * self.w_tracking_approach,
            )

        tracking_reward = torch.where(all_reached, maintain_reward, approach_reward + progress_reward)

        return tracking_reward

    def _get_hand_position(self, states):
        """Get palm center position."""
        rs = states.robots[self.robot_name]
        body_state = rs.body_state
        if body_state is None:
            raise ValueError("Robot body_state is required to compute hand pose.")
        if not torch.is_tensor(body_state):
            body_state = torch.as_tensor(body_state, device=self.device, dtype=torch.float32)
        else:
            body_state = body_state.to(self.device)

        name_to_index = {name: idx for idx, name in enumerate(rs.body_names)}
        if "L_arm_l7" not in name_to_index:
            raise ValueError("Required link 'L_arm_l7' missing in body_names.")
        link_index = name_to_index["L_arm_l7"]

        link_pos = body_state[:, link_index, 0:3]
        link_quat = body_state[:, link_index, 3:7]

        hand_offset_local = torch.tensor(
            [0.25864, -0.035, -0.03513],
            device=link_pos.device,
            dtype=link_pos.dtype,
        )
        hand_offset = (
            hand_offset_local.view(1, 3, 1)
            .repeat(link_pos.shape[0], 1, 1)
        )
        link_rot = matrix_from_quat(link_quat)
        hand_pos = link_pos + torch.bmm(link_rot, hand_offset).squeeze(-1)

        return hand_pos


    def _get_initial_states(self) -> list[dict] | None:
        """Get initial states for all environments."""
        init = [
            {
                "objects": {
                    "object": {
                        "pos": torch.tensor([0.104252, -0.076198, 0.846706]),
                        "rot": torch.tensor([0.454115, 0.132146, 0.231502, -0.850132]),
                    },
                    "wall": {
                        "pos": torch.tensor([0.532921, -0.217400, 0.946513]),
                        "rot": torch.tensor([0.999490, -0.000045, 0.001448, -0.031900]),
                    },
                    "table": {
                        "pos": torch.tensor([0.560000, -0.250000, 0.399868]),
                        "rot": torch.tensor([1.000000, -0.000000, -0.000000, 0.000000]),
                    },
                    # Trajectory waypoints (world coordinates)
                    "traj_marker_0": {
                        "pos": torch.tensor([0.300000, -0.460000, 1.020000]),
                        "rot": torch.tensor([1.000000, 0.000000, 0.000000, 0.000000]),
                    },
                    "traj_marker_1": {
                        "pos": torch.tensor([0.300000, -0.320000, 1.220000]),
                        "rot": torch.tensor([1.000000, 0.000000, 0.000000, 0.000000]),
                    },
                    "traj_marker_2": {
                        "pos": torch.tensor([0.300000, -0.190000, 1.220000]),
                        "rot": torch.tensor([0.998750, 0.000000, 0.049979, 0.000000]),
                    },
                    "traj_marker_3": {
                        "pos": torch.tensor([0.300000, -0.070000, 1.220000]),
                        "rot": torch.tensor([1.000000, 0.000000, 0.000000, 0.000000]),
                    },
                    "traj_marker_4": {
                        "pos": torch.tensor([0.300000, 0.000000, 1.080000]),
                        "rot": torch.tensor([0.984726, 0.000000, 0.174108, 0.000000]),
                    },
                    self.HAND_MARKER_NAME: {
                        "pos": torch.tensor([0.0, 0.0, 0.0]),
                        "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                    },
                },
                "robots": {
                    "vega": {
                        "pos": torch.tensor([-0.353636, -0.230209, 0.000511]),
                        "rot": torch.tensor([1.000201, 0.000000, -0.000000, -0.000000]),
                        "dof_pos": {
                            # From saved_poses_20251126_101518.py
                            "B_wheel_j1": 3.098256,
                            "B_wheel_j2": 11.410545,
                            "L_arm_j1": 0.132269,
                            "L_arm_j2": -0.005679,
                            "L_arm_j3": -0.039935,
                            "L_arm_j4": -0.007740,
                            "L_arm_j5": 0.792566,
                            "L_arm_j6": -0.068200,
                            "L_arm_j7": 0.003688,
                            "L_ff_j1": -0.551686,
                            "L_ff_j2": -0.621416,
                            "L_lf_j1": 0.023751,
                            "L_lf_j2": 0.027291,
                            "L_mf_j1": -0.551840,
                            "L_mf_j2": -0.625254,
                            "L_rf_j1": 0.014122,
                            "L_rf_j2": 0.015942,
                            "L_th_j0": 0.693886,
                            "L_th_j1": 0.149071,
                            "L_th_j2": 0.201562,
                            "L_wheel_j1": -0.050129,
                            "L_wheel_j2": -2.361024,
                            "R_arm_j1": -0.012039,
                            "R_arm_j2": 0.000147,
                            "R_arm_j3": 0.002277,
                            "R_arm_j4": 0.015811,
                            "R_arm_j5": -0.001019,
                            "R_arm_j6": -0.003106,
                            "R_arm_j7": 0.000569,
                            "R_ff_j1": 0.006052,
                            "R_ff_j2": 0.006850,
                            "R_lf_j1": 0.008481,
                            "R_lf_j2": 0.009723,
                            "R_mf_j1": 0.005918,
                            "R_mf_j2": 0.006720,
                            "R_rf_j1": 0.007117,
                            "R_rf_j2": 0.008026,
                            "R_th_j0": 0.264549,
                            "R_th_j1": 0.078753,
                            "R_th_j2": 0.106434,
                            "R_wheel_j1": 0.081973,
                            "R_wheel_j2": 5.777865,
                            "torso_j1": 0.153865,
                            "torso_j2": 0.016780,
                            "torso_j3": -0.021553,
                        },
                    },
                },
            }
            for _ in range(self.num_envs)
        ]

        return init
