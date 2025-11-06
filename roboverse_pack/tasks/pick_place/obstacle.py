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
            "tracking_approach": 3.0,
            "tracking_progress": 300.0,
        }
    },
    # Trajectory tracking settings
    "trajectory_tracking": {
        "num_waypoints": 5,
        "reach_threshold": 0.10,
        "grasp_check_distance": 0.02,
    },
    # Randomization settings
    "randomization": {
        "box_pos_range": 0.015,
        "robot_pos_noise": 0.1,
        "joint_noise_range": 0.05,
    },
}


@register_task("pick_place.obstacle", "pick_place_obstacle")
class PickPlaceObstacle(PickPlaceBase):
    """Pick up a cube and navigate over a wall obstacle.

    Reward shaping and task design adapted from DeepMind's Mujoco Playground
    (Apache 2.0 License), re-implemented for RoboVerse.
    """

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
                name="wall",
                size=(0.8, 0.05, 0.3),
                mass=100.0,
                physics=PhysicStateType.RIGIDBODY,
                color=(0.7, 0.7, 0.7),
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

    def _get_initial_states(self) -> list[dict] | None:
        """Get initial states for all environments."""
        init = [
            {
                "objects": {
                    "object": {
                        "pos": torch.tensor([0.460786, -0.331668, 0.019999]),
                        "rot": torch.tensor([0.999999, 0.000076, 0.000050, -0.001621]),
                    },
                    "wall": {
                        "pos": torch.tensor([0.610000, -0.050000, 0.150000]),
                        "rot": torch.tensor([1.000000, 0.000000, 0.000000, 0.000000]),
                    },
                    # Trajectory waypoints (world coordinates)
                    "traj_marker_0": {
                        "pos": torch.tensor([0.460000, -0.270000, 0.200000]),
                        "rot": torch.tensor([1.000000, 0.000000, 0.000000, 0.000000]),
                    },
                    "traj_marker_1": {
                        "pos": torch.tensor([0.470000, -0.210000, 0.310000]),
                        "rot": torch.tensor([0.912042, -0.307311, -0.257331, 0.086707]),
                    },
                    "traj_marker_2": {
                        "pos": torch.tensor([0.500000, -0.050000, 0.410000]),
                        "rot": torch.tensor([1.000000, 0.000000, 0.000000, 0.000000]),
                    },
                    "traj_marker_3": {
                        "pos": torch.tensor([0.500000, 0.090000, 0.340000]),
                        "rot": torch.tensor([1.000000, 0.000000, 0.000000, 0.000000]),
                    },
                    "traj_marker_4": {
                        "pos": torch.tensor([0.500000, 0.150000, 0.240000]),
                        "rot": torch.tensor([1.000000, 0.000000, 0.000000, 0.000000]),
                    },
                },
                "robots": {
                    "franka": {
                        "pos": torch.tensor([0.148651, -0.206552, 0.014102]),
                        "rot": torch.tensor([0.976835, 0.029517, 0.020012, 0.213930]),
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
