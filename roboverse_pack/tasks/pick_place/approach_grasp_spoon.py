"""Stage 1: Simple Approach and Grasp task for spoon object.

This task inherits from PickPlaceApproachGraspSimple and customizes it for the spoon object
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


@register_task("pick_place.approach_grasp_simple_spoon", "pick_place_approach_grasp_simple_spoon")
class PickPlaceApproachGraspSimpleSpoon(PickPlaceApproachGraspSimple):
    """Simple Approach and Grasp task for spoon object.

    This task inherits from PickPlaceApproachGraspSimple and customizes:
    - Scenario: Uses spoon mesh, table mesh, and basket from EmbodiedGenData
    - Initial states: Loads poses from saved_poses_20251206_spoon_basket.py
    """

    scenario = ScenarioCfg(
        objects=[
            # Use actual spoon mesh from EmbodiedGenData (matches object_layout.py)
            RigidObjCfg(
                name="object",
                scale=(1, 1, 1),
                physics=PhysicStateType.RIGIDBODY,
                usd_path="roboverse_data/assets/EmbodiedGenData/new_assets/spoon/1/usd/2f1c3077a8d954e58fc0bf75cf35e849.usd",
                urdf_path="roboverse_data/assets/EmbodiedGenData/new_assets/spoon/1/2f1c3077a8d954e58fc0bf75cf35e849.urdf",
                mjcf_path="roboverse_data/assets/EmbodiedGenData/new_assets/spoon/1/mjcf/2f1c3077a8d954e58fc0bf75cf35e849.xml",
            ),
            # Use actual table mesh from EmbodiedGenData (matches object_layout.py)
            RigidObjCfg(
                name="table",
                scale=(1, 1, 1),
                physics=PhysicStateType.RIGIDBODY,
                usd_path="roboverse_data/assets/EmbodiedGenData/demo_assets/table/usd/table.usd",
                urdf_path="roboverse_data/assets/EmbodiedGenData/demo_assets/table/result/table.urdf",
                mjcf_path="roboverse_data/assets/EmbodiedGenData/demo_assets/table/mjcf/table.xml",
                fix_base_link=True,
            ),
            # Basket for visualization (matches object_layout.py)
            RigidObjCfg(
                name="basket",
                scale=(1, 1, 1),
                physics=PhysicStateType.RIGIDBODY,
                usd_path="roboverse_data/assets/EmbodiedGenData/new_assets/basket/1/usd/663158968e3f5900af1f6e7cecef24c7.usd",
                urdf_path="roboverse_data/assets/EmbodiedGenData/new_assets/basket/1/663158968e3f5900af1f6e7cecef24c7.urdf",
                mjcf_path="roboverse_data/assets/EmbodiedGenData/new_assets/basket/1/mjcf/663158968e3f5900af1f6e7cecef24c7.xml",
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

        Uses saved poses from object_layout.py. Loads spoon, table, basket, and trajectory markers
        from saved_poses_20251206_spoon_basket.py.
        """
        # Add path to saved poses
        saved_poses_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
            "get_started",
            "output",
            "saved_poses_20251206_spoon_basket.py",
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
                        # Spoon as the object to pick
                        "object": saved_poses["objects"]["spoon"],
                        "table": saved_poses["objects"]["table"],
                        # Basket for visualization (if present in saved poses)
                        "basket": saved_poses["objects"].get(
                            "basket", saved_poses["objects"]["table"]
                        ),  # Fallback to table if basket not found
                        # Include trajectory markers if present
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
        else:
            # Default poses (fallback)
            init = [
                {
                    "objects": {
                        "object": {
                            "pos": torch.tensor([0.654277, -0.345737, 0.020000]),
                            "rot": torch.tensor([0.706448, -0.031607, 0.706347, 0.031698]),
                        },
                        "table": {
                            "pos": torch.tensor([0.499529, 0.253941, 0.200000]),
                            "rot": torch.tensor([0.999067, -0.000006, 0.000009, 0.043198]),
                        },
                        # Trajectory waypoints (world coordinates)
                        "traj_marker_0": {
                            "pos": torch.tensor([0.610000, -0.280000, 0.150000]),
                            "rot": torch.tensor([1.000000, 0.000000, 0.000000, 0.000000]),
                        },
                        "traj_marker_1": {
                            "pos": torch.tensor([0.600000, -0.190000, 0.220000]),
                            "rot": torch.tensor([1.000000, 0.000000, 0.000000, 0.000000]),
                        },
                        "traj_marker_2": {
                            "pos": torch.tensor([0.560000, -0.110000, 0.360000]),
                            "rot": torch.tensor([0.998750, 0.000000, 0.049979, -0.000000]),
                        },
                        "traj_marker_3": {
                            "pos": torch.tensor([0.530000, 0.010000, 0.470000]),
                            "rot": torch.tensor([1.000000, 0.000000, 0.000000, 0.000000]),
                        },
                        "traj_marker_4": {
                            "pos": torch.tensor([0.510000, 0.130000, 0.460000]),
                            "rot": torch.tensor([0.984726, 0.000000, 0.174108, -0.000000]),
                        },
                    },
                    "robots": {
                        "franka": {
                            "pos": torch.tensor([-0.025, -0.160, 0.018054]),
                            "rot": torch.tensor([1.000000, 0.000000, 0.000000, 0.000000]),
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
