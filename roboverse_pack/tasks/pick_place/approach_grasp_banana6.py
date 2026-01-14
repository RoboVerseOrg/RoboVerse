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


@register_task(
    "pick_place.approach_grasp_simple_banana6",
    "pick_place.approach_grasp_simple_banana6",
    "pick_place.approach_grasp_banana6",
)
class PickPlaceApproachGraspSimpleBanana6(PickPlaceApproachGraspSimple):
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
                name="object",
                scale=(1, 1, 1),
                physics=PhysicStateType.RIGIDBODY,
                usd_path="roboverse_data/assets/EmbodiedGenData/demo_assets/banana/usd/banana.usd",
                urdf_path="roboverse_data/assets/EmbodiedGenData/demo_assets/banana/result/banana.urdf",
                mjcf_path="roboverse_data/assets/EmbodiedGenData/demo_assets/banana/mjcf/banana.xml",
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
                        "object": {
                            "pos": torch.tensor([0.220000, 0.090000, 0.825000]),
                            "rot": torch.tensor([1.000000, 0.000000, 0.000000, 0.000000]),
                        },
                        "traj_marker_0": {
                            "pos": torch.tensor([0.250000, -0.570000, 0.850000]),
                            "rot": torch.tensor([1.000000, 0.000000, 0.000000, 0.000000]),
                        },
                        "traj_marker_1": {
                            "pos": torch.tensor([0.310000, -0.460000, 0.900000]),
                            "rot": torch.tensor([1.000000, 0.000000, 0.000000, 0.000000]),
                        },
                        "traj_marker_2": {
                            "pos": torch.tensor([0.340000, -0.330000, 0.960000]),
                            "rot": torch.tensor([1.000000, 0.000000, 0.000000, 0.000000]),
                        },
                        "traj_marker_3": {
                            "pos": torch.tensor([0.420000, -0.150000, 0.920000]),
                            "rot": torch.tensor([0.601833, 0.798621, 0.000000, -0.000000]),
                        },
                        "traj_marker_4": {
                            "pos": torch.tensor([0.420000, 0.020000, 0.870000]),
                            "rot": torch.tensor([1.000000, 0.000000, 0.000000, 0.000000]),
                        },
                    },
                    "robots": {
                        "franka": {
                            "pos": torch.tensor([0.810000, -0.330000, 0.820000]),
                            "rot": torch.tensor([0.215409, -0.102827, 0.002411, 0.971092]),
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
