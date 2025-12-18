"""Stage 1: Simple Approach and Grasp task for banana object.

This task inherits from PickPlaceApproachGraspSimple and customizes it for the banana object
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
from roboverse_pack.tasks.pick_place.approach_grasp import PickPlaceApproachGraspSimple


@register_task("pick_place.approach_grasp_simple_banana2", "pick_place.approach_grasp_simple_banana2", "pick_place.approach_grasp_banana2")
class PickPlaceApproachGraspSimpleBanana2(PickPlaceApproachGraspSimple):
    """Simple Approach and Grasp task for banana object.

    This task inherits from PickPlaceApproachGraspSimple and customizes:
    - Scenario: Uses banana mesh, table mesh, and basket from EmbodiedGenData
    - Initial states: Loads poses from saved_poses_20251206_banana_basket.py
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
                name="bowl",
                scale=(1, 1, 1),
                physics=PhysicStateType.RIGIDBODY,
                usd_path="roboverse_data/assets/EmbodiedGenData/new_assets/bowl/1/usd/0f296af3df66565c9e1a7c2bc7b35d72.usd",
                urdf_path="roboverse_data/assets/EmbodiedGenData/new_assets/bowl/1/0f296af3df66565c9e1a7c2bc7b35d72.urdf",
                mjcf_path="roboverse_data/assets/EmbodiedGenData/new_assets/bowl/1/mjcf/0f296af3df66565c9e1a7c2bc7b35d72.xml",
            ),
            RigidObjCfg(
                name="cutting_tools",
                scale=(1, 1, 1),
                physics=PhysicStateType.RIGIDBODY,
                usd_path="roboverse_data/assets/EmbodiedGenData/new_assets/cutting_tools/1/usd/c5810e7c2c785fe3940372b205090bad.usd",
                urdf_path="roboverse_data/assets/EmbodiedGenData/new_assets/cutting_tools/1/c5810e7c2c785fe3940372b205090bad.urdf",
                mjcf_path="roboverse_data/assets/EmbodiedGenData/new_assets/cutting_tools/1/mjcf/c5810e7c2c785fe3940372b205090bad.xml",
            ),
            RigidObjCfg(
                name="screw_driver",
                scale=(1, 1, 1),
                physics=PhysicStateType.RIGIDBODY,
                usd_path="roboverse_data/assets/EmbodiedGenData/new_assets/screwdriver/1/usd/ae51f060e3455e9f84a4fec81cc9284b.usd",
                urdf_path="roboverse_data/assets/EmbodiedGenData/new_assets/screwdriver/1/ae51f060e3455e9f84a4fec81cc9284b.urdf",
                mjcf_path="roboverse_data/assets/EmbodiedGenData/new_assets/screwdriver/1/mjcf/ae51f060e3455e9f84a4fec81cc9284b.xml",
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
                name="object",
                scale=(1, 1, 1),
                physics=PhysicStateType.RIGIDBODY,
                usd_path="roboverse_data/assets/EmbodiedGenData/demo_assets/banana/usd/banana.usd",
                urdf_path="roboverse_data/assets/EmbodiedGenData/demo_assets/banana/result/banana.urdf",
                mjcf_path="roboverse_data/assets/EmbodiedGenData/demo_assets/banana/mjcf/banana.xml",
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
        """Get initial states for all environments.

        Uses saved poses from object_layout.py. Loads banana, table, basket, and trajectory markers
        from saved_poses_20251206_banana_basket.py.
        """
        # Add path to saved poses
        saved_poses_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
            "get_started",
            "output",
            "saved_poses_20251206_banana_basket.py",
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
                        # Banana as the object to pick
                        "object": saved_poses["objects"].get("banana", saved_poses["objects"].get("object")),
                        "table": saved_poses["objects"]["table"],
                        "lamp": saved_poses["objects"].get("lamp"),
                        "basket": saved_poses["objects"].get("basket"),
                        "bowl": saved_poses["objects"].get("bowl"),
                        "cutting_tools": saved_poses["objects"].get("cutting_tools"),
                        "spoon": saved_poses["objects"].get("spoon"),
                        # Include trajectory markers if present
                        "traj_marker_0": saved_poses["objects"].get("traj_marker_0"),
                        "traj_marker_1": saved_poses["objects"].get("traj_marker_1"),
                        "traj_marker_2": saved_poses["objects"].get("traj_marker_2"),
                        "traj_marker_3": saved_poses["objects"].get("traj_marker_3"),
                        "traj_marker_4": saved_poses["objects"].get("traj_marker_4"),
                    },
                    "robots": {
                        "franka": saved_poses["robots"]["franka"],
                    },
                }
                # Remove None values
                env_state["objects"] = {k: v for k, v in env_state["objects"].items() if v is not None}
                init.append(env_state)
        else:
            # Default poses (fallback) - using poses from original approach_grasp_banana.py
            init = [
                {
                    "objects": {
                        "table": {
                            "pos": torch.tensor([0.400000, -0.200000, 0.400000]),
                            "rot": torch.tensor([1.000000, 0.000000, 0.000000, 0.000000]),
                        },
                        "bowl": {
                            "pos": torch.tensor([0.600000, 0.310000, 0.863000]),
                            "rot": torch.tensor([1.000000, 0.000000, 0.000000, 0.000000]),
                        },
                        "cutting_tools": {
                            "pos": torch.tensor([0.340000, 0.250000, 0.820000]),
                            "rot": torch.tensor([0.930507, 0.000000, -0.000000, 0.366273]),
                        },
                        "screw_driver": {
                            "pos": torch.tensor([0.190000, 0.100000, 0.811000]),
                            "rot": torch.tensor([-0.868588, -0.274057, -0.052298, 0.409518]),
                        },
                        "spoon": {
                            "pos": torch.tensor([0.350000, 0.270000, 0.860000]),
                            "rot": torch.tensor([0.952675, -0.005654, 0.187418, 0.239269]),
                        },
                        "object": {
                            "pos": torch.tensor([0.570000, -0.200000, 0.795000]),
                            "rot": torch.tensor([0.644470, -0.113948, 0.110352, 0.747993]),
                        },
                        "mug": {
                            "pos": torch.tensor([0.310000, 0.070000, 0.863000]),
                            "rot": torch.tensor([1.000000, 0.000000, 0.000000, 0.000000]),
                        },
                        "book": {
                            "pos": torch.tensor([0.210000, -0.480000, 0.820000]),
                            "rot": torch.tensor([1.000000, 0.000000, 0.000000, 0.000000]),
                        },
                        "traj_marker_0": {
                            "pos": torch.tensor([0.580000, -0.220000, 0.850000]),
                            "rot": torch.tensor([1.000000, 0.000000, 0.000000, 0.000000]),
                        },
                        "traj_marker_1": {
                            "pos": torch.tensor([0.540000, -0.230000, 0.950000]),
                            "rot": torch.tensor([1.000000, 0.000000, 0.000000, 0.000000]),
                        },
                        "traj_marker_2": {
                            "pos": torch.tensor([0.440000, -0.310000, 0.970000]),
                            "rot": torch.tensor([1.000000, 0.000000, 0.000000, 0.000000]),
                        },
                        "traj_marker_3": {
                            "pos": torch.tensor([0.310000, -0.380000, 0.950000]),
                            "rot": torch.tensor([0.601833, 0.798621, 0.000000, -0.000000]),
                        },
                        "traj_marker_4": {
                            "pos": torch.tensor([0.240000, -0.480000, 0.900000]),
                            "rot": torch.tensor([1.000000, 0.000000, 0.000000, 0.000000]),
                        },
                    },
                    "robots": {
                        "franka": {
                            "pos": torch.tensor([0.800000, -0.800000, 0.780000]),
                            "rot": torch.tensor([1.000000, 0.000000, 0.000000, 0.000000]),
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
                for _ in range(self.num_envs)
            ]

        return init
