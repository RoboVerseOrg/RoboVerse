"""Put mug on table task using EmbodiedGen assets."""

from __future__ import annotations

import torch

from metasim.constants import PhysicStateType
from metasim.example.example_pack.tasks.checkers import DetectedChecker, RelativeBboxDetector
from metasim.scenario.objects import RigidObjCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.registry import register_task

from .base import EmbodiedGenBaseTask


@register_task("embodiedgen.put_mug", "put_mug")
class PutMugTask(EmbodiedGenBaseTask):
    """Put the mug on the table at a target location.

    The robot needs to pick up the mug and place it at a designated target area on the table.
    This task is slightly different from put_banana as the mug has a handle and different shape.
    """

    max_episode_steps = 250

    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="table",
                scale=(1, 1, 1),
                physics=PhysicStateType.GEOM,
                usd_path="roboverse_data/assets/EmbodiedGenData/demo_assets/table/usd/table.usd",
                urdf_path="roboverse_data/assets/EmbodiedGenData/demo_assets/table/result/table.urdf",
                mjcf_path="roboverse_data/assets/EmbodiedGenData/demo_assets/table/mjcf/table.mjcf",
                fix_base_link=True,
            ),
            RigidObjCfg(
                name="mug",
                scale=(1, 1, 1),
                physics=PhysicStateType.RIGIDBODY,
                usd_path="roboverse_data/assets/EmbodiedGenData/demo_assets/mug/usd/mug.usd",
                urdf_path="roboverse_data/assets/EmbodiedGenData/demo_assets/mug/result/mug.urdf",
                mjcf_path="roboverse_data/assets/EmbodiedGenData/demo_assets/mug/mjcf/mug.mjcf",
            ),
            # Target location - using vase as a visual marker
            RigidObjCfg(
                name="target",
                scale=(0.3, 0.3, 0.05),  # Flat marker
                physics=PhysicStateType.XFORM,  # Non-physical, just visual
                usd_path="roboverse_data/assets/EmbodiedGenData/demo_assets/vase/usd/vase.usd",
                urdf_path="roboverse_data/assets/EmbodiedGenData/demo_assets/vase/result/vase.urdf",
                mjcf_path="roboverse_data/assets/EmbodiedGenData/demo_assets/vase/mjcf/vase.mjcf",
            ),
        ],
        robots=["franka"],
    )

    # Checker: mug should be placed at the target location on the table
    checker = DetectedChecker(
        obj_name="mug",
        detector=RelativeBboxDetector(
            base_obj_name="target",
            relative_pos=[0.0, 0.0, 0.0],  # Center of target
            relative_quat=[1.0, 0.0, 0.0, 0.0],
            checker_lower=[-0.1, -0.1, -0.06],  # Larger tolerance due to mug handle
            checker_upper=[0.1, 0.1, 0.06],
            ignore_base_ori=True,  # Don't care about mug orientation
        ),
    )

    def _get_initial_states(self) -> list[dict] | None:
        """Define initial states manually without trajectory file."""
        # Create initial states for each environment
        init_states = [
            {
                "objects": {
                    "table": {
                        "pos": torch.tensor([0.4, -0.2, 0.4]),
                        "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                    },
                    "mug": {
                        "pos": torch.tensor([0.68, -0.34, 0.863]),  # Starting position on table
                        "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                    },
                    "target": {
                        "pos": torch.tensor([0.30, 0.05, 0.82]),  # Target location (opposite side)
                        "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                    },
                },
                "robots": {
                    "franka": {
                        "pos": torch.tensor([0.8, -0.8, 0.78]),
                        "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                        "dof_pos": {
                            "panda_joint1": 0.0,
                            "panda_joint2": -0.785398,
                            "panda_joint3": 0.0,
                            "panda_joint4": -2.356194,
                            "panda_joint5": 0.0,
                            "panda_joint6": 1.570796,
                            "panda_joint7": 0.785398,
                            "panda_finger_joint1": 0.04,
                            "panda_finger_joint2": 0.04,
                        },
                    },
                },
            }
            for _ in range(self.num_envs)
        ]

        return init_states
