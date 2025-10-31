"""Pour water task with rotation tracking."""

from __future__ import annotations

import torch

from metasim.constants import PhysicStateType
from metasim.scenario.objects import PrimitiveCubeCfg, PrimitiveCylinderCfg, RigidObjCfg
from metasim.scenario.scenario import ScenarioCfg, SimParamCfg
from metasim.task.registry import register_task

from .base import PickPlaceBase

DEFAULT_CONFIG = {
    "action_scale": 0.04,
    "reward_config": {
        "scales": {
            "gripper_approach": 2.0,
            "gripper_close": 0.4,
            "robot_target_qpos": 0.1,
            "tracking_approach": 3.0,
            "tracking_progress": 300.0,
            "rotation_tracking": 2.0,
        }
    },
    "trajectory_tracking": {
        "num_waypoints": 5,
        "reach_threshold": 0.10,
        "grasp_check_distance": 0.02,
        "enable_rotation_tracking": True,
    },
    "randomization": {
        "box_pos_range": 0.015,
        "robot_pos_noise": 0.1,
        "joint_noise_range": 0.05,
    },
}


@register_task("pick_place.pour_water", "pick_place_pour_water")
class PickPlacePourWater(PickPlaceBase):
    """Pour water task with rotation tracking enabled."""

    scenario = ScenarioCfg(
        objects=[
            PrimitiveCubeCfg(
                name="table",
                size=(0.2, 0.3, 0.4),
                mass=10.0,
                physics=PhysicStateType.RIGIDBODY,
                color=(0.8, 0.6, 0.4),
                fix_base_link=True,
            ),
            PrimitiveCubeCfg(
                name="object",
                size=(0.05, 0.05, 0.1),
                mass=0.02,
                physics=PhysicStateType.RIGIDBODY,
                color=(1.0, 0.0, 0.0),
            ),
            PrimitiveCylinderCfg(
                name="cup",
                radius=0.04,
                height=0.06,
                physics=PhysicStateType.RIGIDBODY,
                color=(0.0, 0.0, 1.0),
            ),
            RigidObjCfg(
                name="traj_marker_0",
                urdf_path="roboverse_pack/tasks/pick_place/marker/axis_marker.urdf",
                mjcf_path="roboverse_pack/tasks/pick_place/marker/axis_marker.xml",
                usd_path="roboverse_pack/tasks/pick_place/marker/axis_marker.usd",
                scale=1.0,
                physics=PhysicStateType.XFORM,
                enabled_gravity=False,
                collision_enabled=False,
                fix_base_link=True,
            ),
            RigidObjCfg(
                name="traj_marker_1",
                urdf_path="roboverse_pack/tasks/pick_place/marker/axis_marker.urdf",
                mjcf_path="roboverse_pack/tasks/pick_place/marker/axis_marker.xml",
                usd_path="roboverse_pack/tasks/pick_place/marker/axis_marker.usd",
                scale=1.0,
                physics=PhysicStateType.XFORM,
                enabled_gravity=False,
                collision_enabled=False,
                fix_base_link=True,
            ),
            RigidObjCfg(
                name="traj_marker_2",
                urdf_path="roboverse_pack/tasks/pick_place/marker/axis_marker.urdf",
                mjcf_path="roboverse_pack/tasks/pick_place/marker/axis_marker.xml",
                usd_path="roboverse_pack/tasks/pick_place/marker/axis_marker.usd",
                scale=1.0,
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
                scale=1.0,
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
                scale=1.0,
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
        global DEFAULT_CONFIG
        old_config = self._save_and_override_config()
        super().__init__(scenario, device)
        self._restore_config(old_config)

    def _save_and_override_config(self):
        """Save current base config and override with pour_water config."""
        from .base import DEFAULT_CONFIG as BASE_CONFIG

        old_config = BASE_CONFIG.copy()
        BASE_CONFIG.update(DEFAULT_CONFIG)
        return old_config

    def _restore_config(self, old_config):
        """Restore original base config."""
        from .base import DEFAULT_CONFIG as BASE_CONFIG

        BASE_CONFIG.clear()
        BASE_CONFIG.update(old_config)

    def _prepare_states(self, states, env_ids):
        """Override to use cup and table."""
        from copy import deepcopy

        states = deepcopy(states)
        rand_config = DEFAULT_CONFIG["randomization"]

        initial_states_list = self._get_initial_states()
        box_center = initial_states_list[0]["objects"]["object"]["pos"]
        if not isinstance(box_center, torch.Tensor):
            box_center = torch.tensor(box_center, device=self.device)
        else:
            box_center = box_center.to(self.device)

        box_pos_range_val = rand_config["box_pos_range"]
        box_pos_range = torch.tensor(
            [
                [box_center[0] - box_pos_range_val, box_center[1] - box_pos_range_val, box_center[2]],
                [box_center[0] + box_pos_range_val, box_center[1] + box_pos_range_val, box_center[2]],
            ],
            device=self.device,
        )

        box_pos = (
            torch.rand(self.num_envs, 3, device=self.device) * (box_pos_range[1] - box_pos_range[0]) + box_pos_range[0]
        )
        box_quat = states.objects["object"].root_state[:, 3:7].clone()
        zero_vel = torch.zeros(self.num_envs, 3, device=self.device)
        zero_ang_vel = torch.zeros(self.num_envs, 3, device=self.device)
        states.objects["object"].root_state = torch.cat([box_pos, box_quat, zero_vel, zero_ang_vel], dim=-1)

        table_pos = states.objects["table"].root_state[:, 0:3].clone()
        table_quat = states.objects["table"].root_state[:, 3:7].clone()
        states.objects["table"].root_state = torch.cat([table_pos, table_quat, zero_vel, zero_ang_vel], dim=-1)

        cup_pos = states.objects["cup"].root_state[:, 0:3].clone()
        cup_quat = states.objects["cup"].root_state[:, 3:7].clone()
        states.objects["cup"].root_state = torch.cat([cup_pos, cup_quat, zero_vel, zero_ang_vel], dim=-1)

        for i in range(self.num_waypoints):
            marker_name = f"traj_marker_{i}"
            marker_pos = self.waypoint_positions[i].unsqueeze(0).expand(self.num_envs, -1)
            marker_quat = self.waypoint_rotations[i].unsqueeze(0).expand(self.num_envs, -1)
            states.objects[marker_name].root_state = torch.cat(
                [marker_pos, marker_quat, zero_vel, zero_ang_vel], dim=-1
            )

        robot_pos = states.robots[self.robot_name].root_state[:, 0:3].clone()
        robot_pos_noise_val = rand_config["robot_pos_noise"]
        robot_pos_noise = (torch.rand(self.num_envs, 3, device=self.device) - 0.5) * robot_pos_noise_val
        robot_pos_new = robot_pos + robot_pos_noise
        robot_quat = states.robots[self.robot_name].root_state[:, 3:7].clone()
        robot_vel = states.robots[self.robot_name].root_state[:, 7:].clone()
        states.robots[self.robot_name].root_state = torch.cat([robot_pos_new, robot_quat, robot_vel], dim=-1)

        robot_joint_pos = states.robots[self.robot_name].joint_pos.clone()
        joint_noise_range = rand_config["joint_noise_range"]
        joint_noise = (torch.rand_like(robot_joint_pos, device=self.device) - 0.5) * 2 * joint_noise_range
        robot_joint_pos_new = robot_joint_pos + joint_noise
        robot_joint_pos_new[:, 0] = torch.clamp(robot_joint_pos_new[:, 0], 0.0, 0.04)
        robot_joint_pos_new[:, 1] = torch.clamp(robot_joint_pos_new[:, 1], 0.0, 0.04)
        robot_joint_pos_new[:, 2:] = torch.clamp(robot_joint_pos_new[:, 2:], -2.8973, 2.8973)
        states.robots[self.robot_name].joint_pos = robot_joint_pos_new

        return states

    def _get_initial_states(self) -> list[dict] | None:
        """Get initial states for pour water task."""
        init = [
            {
                "objects": {
                    "table": {
                        "pos": torch.tensor([1.078600, -0.331668, 0.409999]),
                        "rot": torch.tensor([0.999999, 0.000076, 0.000050, -0.001621]),
                    },
                    "object": {
                        "pos": torch.tensor([1.090786, -0.691668, 0.869998]),
                        "rot": torch.tensor([0.999999, 0.000076, 0.000050, -0.001621]),
                    },
                    "cup": {
                        "pos": torch.tensor([1.100000, -0.080000, 0.870000]),
                        "rot": torch.tensor([1.000000, 0.000000, 0.000000, 0.000000]),
                    },
                    "traj_marker_0": {
                        "pos": torch.tensor([1.099999, -0.580000, 1.009999]),
                        "rot": torch.tensor([1.000000, 0.000000, 0.000000, 0.000000]),
                    },
                    "traj_marker_1": {
                        "pos": torch.tensor([1.109999, -0.450000, 1.069999]),
                        "rot": torch.tensor([0.997793, -0.059422, -0.024480, 0.016688]),
                    },
                    "traj_marker_2": {
                        "pos": torch.tensor([1.139999, -0.300000, 1.169999]),
                        "rot": torch.tensor([0.978341, -0.204819, -0.022266, 0.020064]),
                    },
                    "traj_marker_3": {
                        "pos": torch.tensor([1.129999, -0.190000, 1.129999]),
                        "rot": torch.tensor([-0.881654, 0.469909, 0.018128, -0.039276]),
                    },
                    "traj_marker_4": {
                        "pos": torch.tensor([1.119999, -0.160000, 1.009999]),
                        "rot": torch.tensor([0.884543, -0.461212, 0.069681, 0.003568]),
                    },
                },
                "robots": {
                    "franka": {
                        "pos": torch.tensor([0.820000, -0.380000, 0.830000]),
                        "rot": torch.tensor([1.000000, 0.000000, 0.000000, 0.000000]),
                        "dof_pos": {
                            "panda_finger_joint1": 0.040000,
                            "panda_finger_joint2": 0.040000,
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
