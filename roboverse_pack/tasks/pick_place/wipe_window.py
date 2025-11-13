"""Pick up a box with the robot gripper."""

from __future__ import annotations

import torch

from metasim.constants import PhysicStateType
from metasim.scenario.objects import PrimitiveCubeCfg, RigidObjCfg
from metasim.scenario.scenario import ScenarioCfg, SimParamCfg
from metasim.task.registry import register_task

from .base import PickPlaceBase

# Default configuration as a global dictionary
DEFAULT_CONFIG = {
    "action_scale": 0.04,
    "reward_config": {
        "scales": {
            "gripper_approach": 2.0,
            "gripper_close": 0.4,
            "robot_target_qpos": 0.1,
            "tracking_approach": 4.0,
            "tracking_progress": 150.0,
        }
    },
    # Trajectory tracking settings
    "trajectory_tracking": {
        "num_waypoints": 6,
        "reach_threshold": 0.10,
        "grasp_check_distance": 0.02,
    },
    # Randomization settings
    "randomization": {
        "box_pos_range": 0.1,
        "robot_pos_noise": 0.0,
        "joint_noise_range": 0.05,
    },
}


@register_task("pick_place.wipe_window", "wipe_window")
class WipeWindow(PickPlaceBase):
    """Wipe window in Z-pattern with a wiper tool.

    Reward shaping and task design adapted from DeepMind's Mujoco Playground
    (Apache 2.0 License), re-implemented for RoboVerse.
    """

    scenario = ScenarioCfg(
        objects=[
            PrimitiveCubeCfg(
                name="object",
                size=(0.02, 0.15, 0.02),
                mass=0.02,
                physics=PhysicStateType.RIGIDBODY,
                color=(0.2, 0.6, 1.0),
            ),
            PrimitiveCubeCfg(
                name="window",
                size=(0.01, 0.6, 0.6),
                mass=100.0,
                physics=PhysicStateType.RIGIDBODY,
                color=(0.7, 0.9, 1.0),
                fix_base_link=True,
            ),
            # Visualization: Trajectory waypoints (6 spheres for Z-pattern)
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
                name="traj_marker_5",
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

    def _get_initial_states(self) -> list[dict] | None:
        """Get initial states for all environments."""
        init = [
            {
                "objects": {
                    "object": {
                        "pos": torch.tensor([0.300028, -0.200003, 0.010001]),
                        "rot": torch.tensor([1.000000, 0.000007, 0.000183, -0.000043]),
                    },
                    "window": {
                        "pos": torch.tensor([0.570000, -0.030000, 0.300000]),
                        "rot": torch.tensor([1.000000, 0.000000, 0.000000, 0.000000]),
                    },
                    # Trajectory waypoints (world coordinates) - Z-pattern
                    "traj_marker_0": {
                        "pos": torch.tensor([0.560000, -0.150000, 0.500000]),
                        "rot": torch.tensor([1.000000, 0.000000, 0.000000, 0.000000]),
                    },
                    "traj_marker_1": {
                        "pos": torch.tensor([0.550000, 0.150000, 0.500000]),
                        "rot": torch.tensor([1.000000, 0.000000, 0.000000, 0.000000]),
                    },
                    "traj_marker_2": {
                        "pos": torch.tensor([0.550000, 0.150000, 0.300000]),
                        "rot": torch.tensor([1.000000, 0.000000, 0.000000, 0.000000]),
                    },
                    "traj_marker_3": {
                        "pos": torch.tensor([0.550000, -0.150000, 0.300000]),
                        "rot": torch.tensor([1.000000, 0.000000, 0.000000, 0.000000]),
                    },
                    "traj_marker_4": {
                        "pos": torch.tensor([0.560000, -0.170000, 0.100000]),
                        "rot": torch.tensor([1.000000, 0.000000, 0.000000, 0.000000]),
                    },
                    "traj_marker_5": {
                        "pos": torch.tensor([0.550000, 0.140000, 0.100000]),
                        "rot": torch.tensor([1.000000, 0.000000, 0.000000, 0.000000]),
                    },
                },
                "robots": {
                    "franka": {
                        "pos": torch.tensor([-0.000006, -0.020000, -0.000004]),
                        "rot": torch.tensor([1.000007, -0.000000, -0.000000, 0.000000]),
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
