"""Stage 1: Simple Approach and Grasp task for screwdriver object.

This task inherits from PickPlaceApproachGraspSimple and customizes it for the screwdriver object
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


@register_task("pick_place.approach_grasp_simple_screwdriver", "pick_place_approach_grasp_simple_screwdriver")
class PickPlaceApproachGraspSimpleScrewDriver(PickPlaceApproachGraspSimple):
    """Simple Approach and Grasp task for screwdriver object.

    This task inherits from PickPlaceApproachGraspSimple and customizes:
    - Scenario: Uses screwdriver mesh, table mesh, and basket from EmbodiedGenData
    - Initial states: Loads poses from saved_poses_20251206_screwdriver_basket.py
    """

    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="table",
                scale=(1, 1, 1),
                physics=PhysicStateType.RIGIDBODY,
                usd_path="roboverse_data/assets/EmbodiedGenData/demo_assets/table/usd/table.usd",
                urdf_path="roboverse_data/assets/EmbodiedGenData/demo_assets/table/result/table.urdf",
                mjcf_path="roboverse_data/assets/EmbodiedGenData/demo_assets/table/mjcf/table.xml",
            ),
            RigidObjCfg(
                name="lamp",
                scale=(1, 1, 1),
                physics=PhysicStateType.RIGIDBODY,
                usd_path="roboverse_data/assets/EmbodiedGenData/new_assets/lighting_fixtures/1/usd/0a4489b1a2875c82a580f8b62d346e08.usd",
                urdf_path="roboverse_data/assets/EmbodiedGenData/new_assets/lighting_fixtures/1/0a4489b1a2875c82a580f8b62d346e08.urdf",
                mjcf_path="roboverse_data/assets/EmbodiedGenData/new_assets/lighting_fixtures/1/mjcf/0a4489b1a2875c82a580f8b62d346e08.xml",
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
            # Use actual screwdriver mesh from EmbodiedGenData (matches object_layout.py)
            RigidObjCfg(
                name="object",
                scale=(1.5, 1.5, 1.5),
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
        """Get initial states for all environments.

        Uses saved poses from object_layout.py. Loads screwdriver, table, basket, and trajectory markers
        from saved_poses_20251206_screwdriver_basket.py.
        """
        # Add path to saved poses
        saved_poses_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
            "get_started",
            "output",
            "saved_poses_20251206_screwdriver_basket.py",
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
                        # Screwdriver as the object to pick
                        "object": saved_poses["objects"].get(
                            "screwdriver",
                            saved_poses["objects"].get("screw_driver", saved_poses["objects"].get("object")),
                        ),
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
            # Default poses (fallback) - using poses from original approach_grasp_screwdriver.py
            init = [
                {
                    "objects": {
                        "table": {
                            "pos": torch.tensor([0.400000, -0.200000, 0.400000]),
                            "rot": torch.tensor([1.000000, 0.000000, 0.000000, 0.000000]),
                        },
                        "lamp": {
                            "pos": torch.tensor([0.680000, 0.310000, 1.050000]),
                            "rot": torch.tensor([1.000000, 0.000000, 0.000000, 0.000000]),
                        },
                        "basket": {
                            "pos": torch.tensor([0.240000, -0.440000, 0.925000]),
                            "rot": torch.tensor([1.000000, 0.000000, 0.000000, 0.000000]),
                        },
                        "bowl": {
                            "pos": torch.tensor([0.620000, -0.080000, 0.863000]),
                            "rot": torch.tensor([1.000000, 0.000000, 0.000000, 0.000000]),
                        },
                        "cutting_tools": {
                            "pos": torch.tensor([0.230000, 0.090000, 0.820000]),
                            "rot": torch.tensor([0.930507, 0.000000, -0.000000, 0.366273]),
                        },
                        "object": {
                            "pos": torch.tensor([0.590000, -0.360000, 0.811000]),
                            "rot": torch.tensor([-0.824810, -0.390085, -0.023230, 0.408623]),
                        },
                        "spoon": {
                            "pos": torch.tensor([0.470000, -0.710000, 0.830000]),
                            "rot": torch.tensor([0.928125, -0.061844, -0.058501, -0.362397]),
                        },
                        "traj_marker_0": {
                            "pos": torch.tensor([0.600000, -0.330000, 0.840000]),
                            "rot": torch.tensor([1.000000, 0.000000, 0.000000, 0.000000]),
                        },
                        "traj_marker_1": {
                            "pos": torch.tensor([0.560000, -0.400000, 1.010000]),
                            "rot": torch.tensor([1.000000, 0.000000, 0.000000, 0.000000]),
                        },
                        "traj_marker_2": {
                            "pos": torch.tensor([0.530000, -0.400000, 1.140000]),
                            "rot": torch.tensor([0.962425, -0.271547, 0.000000, 0.000000]),
                        },
                        "traj_marker_3": {
                            "pos": torch.tensor([0.430000, -0.340000, 1.160000]),
                            "rot": torch.tensor([0.601833, 0.798621, 0.000000, -0.000000]),
                        },
                        "traj_marker_4": {
                            "pos": torch.tensor([0.260000, -0.400000, 1.170000]),
                            "rot": torch.tensor([1.000000, 0.000000, 0.000000, 0.000000]),
                        },
                    },
                    "robots": {
                        "franka": {
                            "pos": torch.tensor([0.800000, -0.800000, 0.780000]),
                            "rot": torch.tensor([0.581682, -0.000000, -0.000001, 0.813414]),
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
