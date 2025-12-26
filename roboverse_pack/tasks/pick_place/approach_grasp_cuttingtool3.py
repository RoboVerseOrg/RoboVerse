"""Stage 1: Simple Approach and Grasp task for cutting tool object.

This task inherits from PickPlaceApproachGraspSimple and customizes it for the cutting tool object
with specific mesh configurations and saved poses from object_layout.py.
"""

from __future__ import annotations

import torch

from metasim.constants import PhysicStateType
from metasim.scenario.objects import RigidObjCfg
from metasim.scenario.scenario import ScenarioCfg, SimParamCfg
from metasim.task.registry import register_task
from roboverse_pack.tasks.pick_place.approach_grasp import PickPlaceApproachGraspSimple


@register_task(
    "pick_place.approach_grasp_simple_cuttingtool3",
    "pick_place.approach_grasp_simple_cuttingtool3",
    "pick_place.approach_grasp_cuttingtool3",
)
class PickPlaceApproachGraspSimpleCuttingTool3(PickPlaceApproachGraspSimple):
    """Simple Approach and Grasp task for cutting tool object.

    This task inherits from PickPlaceApproachGraspSimple and customizes:
    - Scenario: Uses cutting tool mesh, table mesh, and basket from EmbodiedGenData
    - Initial states: Loads poses from saved_poses_20251206_cuttingtool3_basket.py
    """

    scenario = ScenarioCfg(
        objects=[
            # EmbodiedGen Assets - Put Banana in Mug Scene
            RigidObjCfg(
                name="table",
                scale=(1, 1, 1),
                physics=PhysicStateType.RIGIDBODY,
                usd_path="roboverse_data/assets/EmbodiedGenData/demo_assets/table/usd/table.usd",
                urdf_path="roboverse_data/assets/EmbodiedGenData/demo_assets/table/result/table.urdf",
                mjcf_path="roboverse_data/assets/EmbodiedGenData/demo_assets/table/mjcf/table.xml",
            ),
            RigidObjCfg(
                name="basket",
                scale=(1, 1, 1),
                physics=PhysicStateType.RIGIDBODY,
                usd_path="roboverse_data/assets/EmbodiedGenData/new_assets/basket/1/usd/663158968e3f5900af1f6e7cecef24c7.usd",
                urdf_path="roboverse_data/assets/EmbodiedGenData/new_assets/basket/1/663158968e3f5900af1f6e7cecef24c7.urdf",
                mjcf_path="roboverse_data/assets/EmbodiedGenData/new_assets/basket/1/mjcf/663158968e3f5900af1f6e7cecef24c7.xml",
            ),
            RigidObjCfg(
                name="object",
                scale=(1, 1, 1),
                physics=PhysicStateType.RIGIDBODY,
                usd_path="roboverse_data/assets/EmbodiedGenData/new_assets/cutting_tools/1/usd/c5810e7c2c785fe3940372b205090bad.usd",
                urdf_path="roboverse_data/assets/EmbodiedGenData/new_assets/cutting_tools/1/c5810e7c2c785fe3940372b205090bad.urdf",
                mjcf_path="roboverse_data/assets/EmbodiedGenData/new_assets/cutting_tools/1/mjcf/c5810e7c2c785fe3940372b205090bad.xml",
            ),
            RigidObjCfg(
                name="spoon",
                scale=(1, 1, 1),
                physics=PhysicStateType.RIGIDBODY,
                usd_path="roboverse_data/assets/EmbodiedGenData/new_assets/spoon/1/usd/2f1c3077a8d954e58fc0bf75cf35e849.usd",
                urdf_path="roboverse_data/assets/EmbodiedGenData/new_assets/spoon/1/2f1c3077a8d954e58fc0bf75cf35e849.urdf",
                mjcf_path="roboverse_data/assets/EmbodiedGenData/new_assets/spoon/1/mjcf/2f1c3077a8d954e58fc0bf75cf35e849.xml",
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
                name="book",
                scale=(1, 1, 1),
                physics=PhysicStateType.RIGIDBODY,
                usd_path="roboverse_data/assets/EmbodiedGenData/demo_assets/book/usd/book.usd",
                urdf_path="roboverse_data/assets/EmbodiedGenData/demo_assets/book/result/book.urdf",
                mjcf_path="roboverse_data/assets/EmbodiedGenData/demo_assets/book/mjcf/book.xml",
            ),
            RigidObjCfg(
                name="remote_control",
                scale=(1, 1, 1),
                physics=PhysicStateType.RIGIDBODY,
                usd_path="roboverse_data/assets/EmbodiedGenData/demo_assets/remote_control/usd/remote_control.usd",
                urdf_path="roboverse_data/assets/EmbodiedGenData/demo_assets/remote_control/result/remote_control.urdf",
                mjcf_path="roboverse_data/assets/EmbodiedGenData/demo_assets/remote_control/mjcf/remote_control.xml",
            ),
            # Trajectory markers
            RigidObjCfg(
                name="traj_marker_0",
                urdf_path="roboverse_pack/tasks/pick_place/marker/marker.urdf",
                mjcf_path="roboverse_pack/tasks/pick_place/marker/marker.xml",
                usd_path="roboverse_pack/tasks/pick_place/marker/marker.usd",
                scale=0.2,
                physics=PhysicStateType.XFORM,
                enabled_gravity=False,
                collision_enabled=False,
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
        """Hardcoded initial states from saved_poses_20251206_banana_basket.py."""
        saved_poses = {
            "objects": {
                "table": {
                    "pos": torch.tensor([0.400000, -0.200000, 0.400000]),
                    "rot": torch.tensor([1.000000, 0.000000, 0.000000, 0.000000]),
                },
                "basket": {
                    "pos": torch.tensor([0.330000, 0.160000, 0.825000]),
                    "rot": torch.tensor([1.000000, 0.000000, 0.000000, 0.000000]),
                },
                "object": {
                    "pos": torch.tensor([0.320000, -0.760000, 0.820000]),
                    "rot": torch.tensor([0.930507, 0.000000, -0.000000, 0.366273]),
                },
                "spoon": {
                    "pos": torch.tensor([0.580000, -0.690000, 0.850000]),
                    "rot": torch.tensor([0.961352, -0.120799, 0.030845, 0.245473]),
                },
                "mug": {
                    "pos": torch.tensor([0.570000, 0.040000, 0.863000]),
                    "rot": torch.tensor([1.000000, 0.000000, 0.000000, 0.000000]),
                },
                "book": {
                    "pos": torch.tensor([0.550000, -0.280000, 0.820000]),
                    "rot": torch.tensor([1.000000, 0.000000, 0.000000, 0.000000]),
                },
                "remote_control": {
                    "pos": torch.tensor([0.170000, -0.190000, 0.811000]),
                    "rot": torch.tensor([0.921060, 0.000000, -0.000000, 0.389418]),
                },
                "traj_marker_0": {
                    "pos": torch.tensor([0.230000, -0.680000, 0.830000]),
                    "rot": torch.tensor([1.000000, 0.000000, 0.000000, 0.000000]),
                },
                "traj_marker_1": {
                    "pos": torch.tensor([0.240000, -0.620000, 0.870000]),
                    "rot": torch.tensor([1.000000, 0.000000, 0.000000, 0.000000]),
                },
                "traj_marker_2": {
                    "pos": torch.tensor([0.290000, -0.560000, 0.920000]),
                    "rot": torch.tensor([1.000000, 0.000000, 0.000000, 0.000000]),
                },
                "traj_marker_3": {
                    "pos": torch.tensor([0.340000, -0.460000, 0.930000]),
                    "rot": torch.tensor([0.601833, 0.798621, 0.000000, -0.000000]),
                },
                "traj_marker_4": {
                    "pos": torch.tensor([0.550000, -0.290000, 0.900000]),
                    "rot": torch.tensor([1.000000, 0.000000, 0.000000, 0.000000]),
                },
            },
            "robots": {
                "franka": {
                    "pos": torch.tensor([0.800000, -0.800000, 0.780000]),
                    "rot": torch.tensor([0.640796, 0.019186, 0.016023, 0.767304]),
                    "dof_pos": {
                        "panda_finger_joint1": 0.040000,
                        "panda_finger_joint2": 0.040000,
                        "panda_joint1": 0.000000,
                        "panda_joint2": -0.785398,
                        "panda_joint3": 0.000000,
                        "panda_joint4": -2.356194,
                        "panda_joint5": 0.000000,
                        "panda_joint6": 1.570796,
                        "panda_joint7": 0.785398,
                    },
                },
            },
        }

        init = []
        for _ in range(self.num_envs):
            env_state = {
                "objects": {
                    "object": saved_poses["objects"]["object"],
                    "table": saved_poses["objects"]["table"],
                    "lamp": saved_poses["objects"]["lamp"],
                    "basket": saved_poses["objects"]["basket"],
                    "bowl": saved_poses["objects"]["bowl"],
                    "cutting_tools": saved_poses["objects"]["cutting_tools"],
                    "spoon": saved_poses["objects"]["spoon"],
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

        return init
