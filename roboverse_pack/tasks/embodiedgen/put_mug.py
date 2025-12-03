"""Put banana on table task using EmbodiedGen assets."""

from __future__ import annotations

import torch

from metasim.constants import PhysicStateType
from metasim.scenario.objects import PrimitiveCubeCfg, RigidObjCfg
from metasim.scenario.scenario import ScenarioCfg, SimParamCfg
from metasim.task.registry import register_task

from .base import EmbodiedGenBaseTask


@register_task("embodiedgen.put_banana", "put_banana")
class PutBananaTask(EmbodiedGenBaseTask):
    """Put the banana into the mug.

    The robot needs to pick up the banana and place it inside the mug.
    The scene contains multiple objects on the table to make it more realistic and challenging.
    """

    HAND_MARKER_NAME = "hand_debug_marker"
    max_episode_steps = 25000

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
                size=(0.8, 0.1, 0.3),
                mass=1000.0,
                physics=PhysicStateType.RIGIDBODY,
                color=(0.7, 0.7, 0.7),
                # fix_base_link=True,
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
                scale=0.1,
                physics=PhysicStateType.XFORM,
                enabled_gravity=False,
                collision_enabled=False,
                fix_base_link=True,
                default_position=(0.300000, -0.460000, 1.020000),
                default_orientation=(1.000000, 0.000000, 0.000000, 0.000000),
            ),
            RigidObjCfg(
                name="traj_marker_1",
                urdf_path="roboverse_pack/tasks/pick_place/marker/marker.urdf",
                mjcf_path="roboverse_pack/tasks/pick_place/marker/marker.xml",
                usd_path="roboverse_pack/tasks/pick_place/marker/marker.usd",
                scale=0.1,
                physics=PhysicStateType.XFORM,
                enabled_gravity=False,
                collision_enabled=False,
                fix_base_link=True,
                default_position=(0.300000, -0.320000, 1.220000),
                default_orientation=(1.000000, 0.000000, 0.000000, 0.000000),
            ),
            RigidObjCfg(
                name="traj_marker_2",
                urdf_path="roboverse_pack/tasks/pick_place/marker/marker.urdf",
                mjcf_path="roboverse_pack/tasks/pick_place/marker/marker.xml",
                usd_path="roboverse_pack/tasks/pick_place/marker/marker.usd",
                scale=0.1,
                physics=PhysicStateType.XFORM,
                enabled_gravity=False,
                collision_enabled=False,
                fix_base_link=True,
                default_position=(0.300000, -0.190000, 1.220000),
                default_orientation=(0.998750, 0.000000, 0.049979, 0.000000),
            ),
            RigidObjCfg(
                name="traj_marker_3",
                urdf_path="roboverse_pack/tasks/pick_place/marker/marker.urdf",
                mjcf_path="roboverse_pack/tasks/pick_place/marker/marker.xml",
                usd_path="roboverse_pack/tasks/pick_place/marker/marker.usd",
                scale=0.1,
                physics=PhysicStateType.XFORM,
                enabled_gravity=False,
                collision_enabled=False,
                fix_base_link=True,
                default_position=(0.300000, -0.070000, 1.220000),
                default_orientation=(1.000000, 0.000000, 0.000000, 0.000000),
            ),
            RigidObjCfg(
                name="traj_marker_4",
                urdf_path="roboverse_pack/tasks/pick_place/marker/marker.urdf",
                mjcf_path="roboverse_pack/tasks/pick_place/marker/marker.xml",
                usd_path="roboverse_pack/tasks/pick_place/marker/marker.usd",
                scale=0.1,
                physics=PhysicStateType.XFORM,
                enabled_gravity=False,
                collision_enabled=False,
                fix_base_link=True,
                default_position=(0.300000, 0.000000, 1.080000),
                default_orientation=(0.984726, 0.000000, 0.174108, 0.000000),
            ),

        ],
        robots=["vega"],
        sim_params=SimParamCfg(
            dt=0.005,
        ),
        decimation=4,
    )


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
                        "pos": torch.tensor([0.200000, -0.460000, 1.020000]),
                        "rot": torch.tensor([1.000000, 0.000000, 0.000000, 0.000000]),
                    },
                    "traj_marker_1": {
                        "pos": torch.tensor([0.500000, -0.320000, 1.220000]),
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
