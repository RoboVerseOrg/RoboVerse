"""Pour water task with rotation tracking."""

from __future__ import annotations

import torch

from metasim.constants import PhysicStateType
from metasim.scenario.objects import PrimitiveCubeCfg, RigidObjCfg
from metasim.scenario.scenario import ScenarioCfg, SimParamCfg
from metasim.task.registry import register_task

from .base import DEFAULT_CONFIG as BASE_DEFAULT_CONFIG
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
            "rotation_tracking": 1.0,
        }
    },
    "trajectory_tracking": {
        "num_waypoints": 5,
        "reach_threshold": 0.20,
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
            RigidObjCfg(
                name="table",
                scale=(1, 1, 1),
                physics=PhysicStateType.RIGIDBODY,
                usd_path="roboverse_data/assets/EmbodiedGenData/demo_assets/table/usd/table.usd",
                urdf_path="roboverse_data/assets/EmbodiedGenData/demo_assets/table/result/table.urdf",
                mjcf_path="roboverse_data/assets/EmbodiedGenData/demo_assets/table/mjcf/table.xml",
                fix_base_link=True,
            ),
            RigidObjCfg(
                name="mug",
                scale=(1, 1, 1),
                physics=PhysicStateType.RIGIDBODY,
                usd_path="roboverse_data/assets/EmbodiedGenData/demo_assets/mug/usd/mug.usd",
                urdf_path="roboverse_data/assets/EmbodiedGenData/demo_assets/mug/result/mug.urdf",
                mjcf_path="roboverse_data/assets/EmbodiedGenData/demo_assets/mug/mjcf/mug.xml",
            ),
            RigidObjCfg(
                name="vase",
                scale=(1, 1, 1),
                physics=PhysicStateType.RIGIDBODY,
                usd_path="roboverse_data/assets/EmbodiedGenData/demo_assets/vase/usd/vase.usd",
                urdf_path="roboverse_data/assets/EmbodiedGenData/demo_assets/vase/result/vase.urdf",
                mjcf_path="roboverse_data/assets/EmbodiedGenData/demo_assets/vase/mjcf/vase.xml",
            ),
            PrimitiveCubeCfg(
                name="object",
                size=(0.03, 0.03, 0.1),
                mass=0.01,
                physics=PhysicStateType.RIGIDBODY,
                color=(1.0, 0.0, 0.0),
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
        """Initialize with custom config override."""
        old_config = BASE_DEFAULT_CONFIG.copy()
        BASE_DEFAULT_CONFIG.update(DEFAULT_CONFIG)
        super().__init__(scenario, device)
        BASE_DEFAULT_CONFIG.clear()
        BASE_DEFAULT_CONFIG.update(old_config)

    def _get_initial_states(self) -> list[dict] | None:
        """Get initial states for pour water task."""
        init = [
            {
                "objects": {
                    "table": {
                        "pos": torch.tensor([0.850000, -0.200000, 0.390000]),
                        "rot": torch.tensor([1.000000, 0.000000, 0.000000, 0.000000]),
                    },
                    "mug": {
                        "pos": torch.tensor([0.820184, 0.189953, 0.855219]),
                        "rot": torch.tensor([-0.004200, -0.000021, -0.000200, -0.999990]),
                    },
                    "vase": {
                        "pos": torch.tensor([1.110714, -0.024288, 0.930882]),
                        "rot": torch.tensor([0.999996, 0.000027, -0.000295, 0.002596]),
                    },
                    "object": {
                        "pos": torch.tensor([1.088784, -0.703373, 0.881773]),
                        "rot": torch.tensor([0.999256, 0.000226, 0.038482, -0.002340]),
                    },
                    "traj_marker_0": {
                        "pos": torch.tensor([1.119999, -0.610000, 1.069999]),
                        "rot": torch.tensor([1.000000, 0.000000, 0.000000, 0.000000]),
                    },
                    "traj_marker_1": {
                        "pos": torch.tensor([1.109999, -0.450000, 1.199999]),
                        "rot": torch.tensor([0.997793, -0.059422, -0.024480, 0.016688]),
                    },
                    "traj_marker_2": {
                        "pos": torch.tensor([1.139999, -0.300000, 1.299999]),
                        "rot": torch.tensor([0.978341, -0.204819, -0.022266, 0.020064]),
                    },
                    "traj_marker_3": {
                        "pos": torch.tensor([1.129999, -0.190000, 1.259999]),
                        "rot": torch.tensor([-0.904037, 0.425257, 0.016142, -0.040133]),
                    },
                    "traj_marker_4": {
                        "pos": torch.tensor([1.119999, -0.160000, 1.179999]),
                        "rot": torch.tensor([0.884543, -0.461212, 0.069681, 0.003568]),
                    },
                },
                "robots": {
                    "franka": {
                        "pos": torch.tensor([0.770510, -0.469331, 0.777111]),
                        "rot": torch.tensor([1.000271, -0.000000, -0.000000, 0.000000]),
                        "dof_pos": {
                            "panda_finger_joint1": 0.000020,
                            "panda_finger_joint2": 0.000001,
                            "panda_joint1": -0.000719,
                            "panda_joint2": -0.003293,
                            "panda_joint3": 0.000657,
                            "panda_joint4": -0.069795,
                            "panda_joint5": -0.000057,
                            "panda_joint6": -0.000359,
                            "panda_joint7": 0.000016,
                        },
                    },
                },
            }
            for _ in range(self.num_envs)
        ]

        return init
