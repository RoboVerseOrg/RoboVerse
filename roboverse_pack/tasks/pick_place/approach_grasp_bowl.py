"""Stage 1: Simple Approach and Grasp task for bowl object (ground layout).

This task:
- Uses EmbodiedGenData/assets/* assets (not demo_assets/new_assets)
- Uses a hardcoded initial state
- Targets the bowl as the grasp object (mapped to name="object" to match PickPlaceBase)
- Uses hardcoded trajectory markers (traj_marker_0..4) required by PickPlaceBase
"""

from __future__ import annotations

import torch

from metasim.constants import PhysicStateType
from metasim.scenario.objects import RigidObjCfg
from metasim.scenario.scenario import ScenarioCfg, SimParamCfg
from metasim.task.registry import register_task
from roboverse_pack.tasks.pick_place.approach_grasp import PickPlaceApproachGraspSimple


@register_task("pick_place.approach_grasp_simple_bowl", "pick_place_approach_grasp_simple_bowl")
class PickPlaceApproachGraspSimpleBowl(PickPlaceApproachGraspSimple):
    """Approach+grasp task where the target object is a bowl (ground layout)."""

    scenario = ScenarioCfg(
        objects=[
            # Target: bowl, mapped to name="object"
            RigidObjCfg(
                name="object",
                scale=(1, 1, 1),
                physics=PhysicStateType.RIGIDBODY,
                usd_path="roboverse_data/assets/EmbodiedGenData/assets/bowl/0f296af3df66565c9e1a7c2bc7b35d72/usd/0f296af3df66565c9e1a7c2bc7b35d72.usd",
                urdf_path="roboverse_data/assets/EmbodiedGenData/assets/bowl/0f296af3df66565c9e1a7c2bc7b35d72/0f296af3df66565c9e1a7c2bc7b35d72.urdf",
                mjcf_path="roboverse_data/assets/EmbodiedGenData/assets/bowl/0f296af3df66565c9e1a7c2bc7b35d72/mjcf/0f296af3df66565c9e1a7c2bc7b35d72.xml",
            ),
            # Context objects (match saved_poses_20251214_224520.py names)
            RigidObjCfg(
                name="basket",
                scale=(1, 1, 1),
                physics=PhysicStateType.RIGIDBODY,
                usd_path="roboverse_data/assets/EmbodiedGenData/assets/basket/663158968e3f5900af1f6e7cecef24c7/usd/663158968e3f5900af1f6e7cecef24c7.usd",
                urdf_path="roboverse_data/assets/EmbodiedGenData/assets/basket/663158968e3f5900af1f6e7cecef24c7/663158968e3f5900af1f6e7cecef24c7.urdf",
                mjcf_path="roboverse_data/assets/EmbodiedGenData/assets/basket/663158968e3f5900af1f6e7cecef24c7/mjcf/663158968e3f5900af1f6e7cecef24c7.xml",
            ),
            RigidObjCfg(
                name="cutting_tools",
                scale=(1, 1, 1),
                physics=PhysicStateType.RIGIDBODY,
                usd_path="roboverse_data/assets/EmbodiedGenData/assets/cutting_tools/c5810e7c2c785fe3940372b205090bad/usd/c5810e7c2c785fe3940372b205090bad.usd",
                urdf_path="roboverse_data/assets/EmbodiedGenData/assets/cutting_tools/c5810e7c2c785fe3940372b205090bad/c5810e7c2c785fe3940372b205090bad.urdf",
                mjcf_path="roboverse_data/assets/EmbodiedGenData/assets/cutting_tools/c5810e7c2c785fe3940372b205090bad/mjcf/c5810e7c2c785fe3940372b205090bad.xml",
            ),
            RigidObjCfg(
                name="lighting_fixtures",
                scale=(1, 1, 1),
                physics=PhysicStateType.RIGIDBODY,
                usd_path="roboverse_data/assets/EmbodiedGenData/assets/lighting_fixtures/03f09dca16db5598a67f0715cf3fb157/usd/03f09dca16db5598a67f0715cf3fb157.usd",
                urdf_path="roboverse_data/assets/EmbodiedGenData/assets/lighting_fixtures/03f09dca16db5598a67f0715cf3fb157/03f09dca16db5598a67f0715cf3fb157.urdf",
                mjcf_path="roboverse_data/assets/EmbodiedGenData/assets/lighting_fixtures/03f09dca16db5598a67f0715cf3fb157/mjcf/03f09dca16db5598a67f0715cf3fb157.xml",
            ),
            RigidObjCfg(
                name="screwdriver",
                scale=(1.5, 1.5, 1.5),
                physics=PhysicStateType.RIGIDBODY,
                usd_path="roboverse_data/assets/EmbodiedGenData/assets/screwdriver/ae51f060e3455e9f84a4fec81cc9284b/usd/ae51f060e3455e9f84a4fec81cc9284b.usd",
                urdf_path="roboverse_data/assets/EmbodiedGenData/assets/screwdriver/ae51f060e3455e9f84a4fec81cc9284b/ae51f060e3455e9f84a4fec81cc9284b.urdf",
                mjcf_path="roboverse_data/assets/EmbodiedGenData/assets/screwdriver/ae51f060e3455e9f84a4fec81cc9284b/mjcf/ae51f060e3455e9f84a4fec81cc9284b.xml",
            ),
            RigidObjCfg(
                name="spoon",
                scale=(1, 1, 1),
                physics=PhysicStateType.RIGIDBODY,
                usd_path="roboverse_data/assets/EmbodiedGenData/assets/spoon/2f1c3077a8d954e58fc0bf75cf35e849/usd/2f1c3077a8d954e58fc0bf75cf35e849.usd",
                urdf_path="roboverse_data/assets/EmbodiedGenData/assets/spoon/2f1c3077a8d954e58fc0bf75cf35e849/2f1c3077a8d954e58fc0bf75cf35e849.urdf",
                mjcf_path="roboverse_data/assets/EmbodiedGenData/assets/spoon/2f1c3077a8d954e58fc0bf75cf35e849/mjcf/2f1c3077a8d954e58fc0bf75cf35e849.xml",
            ),
            # Required: trajectory markers (PickPlaceBase requires them in initial state)
            *[
                RigidObjCfg(
                    name=f"traj_marker_{i}",
                    urdf_path="roboverse_pack/tasks/pick_place/marker/marker.urdf",
                    mjcf_path="roboverse_pack/tasks/pick_place/marker/marker.xml",
                    usd_path="roboverse_pack/tasks/pick_place/marker/marker.usd",
                    scale=0.2,
                    physics=PhysicStateType.XFORM,
                    enabled_gravity=False,
                    collision_enabled=False,
                    fix_base_link=True,
                )
                for i in range(5)
            ],
        ],
        robots=["franka"],
        sim_params=SimParamCfg(dt=0.005),
        decimation=4,
    )

    max_episode_steps = 200

    def _get_initial_states(self) -> list[dict] | None:
        # Hardcoded initial state (objects + robot + traj_marker_0..4)
        init = [
            {
                "objects": {
                    # target bowl -> name "object"
                    "object": {
                        "pos": torch.tensor([1.060000, -0.380000, 0.130000]),
                        "rot": torch.tensor([0.998750, 0.000000, 0.049979, -0.000000]),
                    },
                    "basket": {
                        "pos": torch.tensor([0.550000, -0.470000, 0.200000]),
                        "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                    },
                    "cutting_tools": {
                        "pos": torch.tensor([1.140000, -0.180000, 0.040000]),
                        "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                    },
                    "lighting_fixtures": {
                        "pos": torch.tensor([0.970000, 0.070000, 0.210000]),
                        "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                    },
                    "screwdriver": {
                        "pos": torch.tensor([1.220000, -0.500000, 0.100000]),
                        "rot": torch.tensor([0.947354, 0.023689, 0.319209, 0.007982]),
                    },
                    "spoon": {
                        "pos": torch.tensor([1.390000, -0.280000, 0.020000]),
                        "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                    },
                    "traj_marker_0": {
                        "pos": torch.tensor([0.990000, -0.380000, 0.230000]),
                        "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                    },
                    "traj_marker_1": {
                        "pos": torch.tensor([0.930000, -0.380000, 0.322500]),
                        "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                    },
                    "traj_marker_2": {
                        "pos": torch.tensor([0.790000, -0.380000, 0.385000]),
                        "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                    },
                    "traj_marker_3": {
                        "pos": torch.tensor([0.690000, -0.380000, 0.377500]),
                        "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                    },
                    "traj_marker_4": {
                        "pos": torch.tensor([0.580000, -0.380000, 0.362500]),
                        "rot": torch.tensor([0.999687, -0.024997, 0.000000, 0.000000]),
                    },
                },
                "robots": {
                    "franka": {
                        "pos": torch.tensor([0.910000, -0.790000, 0.030000]),
                        "rot": torch.tensor([-0.666275, -0.000000, 0.000000, -0.745703]),
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
                    }
                },
            }
            for _ in range(self.num_envs)
        ]

        return init
