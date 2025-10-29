"""Put banana on table task using EmbodiedGen assets."""

from __future__ import annotations

import torch

from metasim.constants import PhysicStateType
from metasim.example.example_pack.tasks.checkers import DetectedChecker, RelativeBboxDetector
from metasim.scenario.objects import RigidObjCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.registry import register_task

from .base import EmbodiedGenBaseTask


@register_task("embodiedgen.put_banana", "put_banana")
class PutBananaTask(EmbodiedGenBaseTask):
    """Put the banana into the mug.

    The robot needs to pick up the banana and place it inside the mug.
    The scene contains multiple objects on the table to make it more realistic and challenging.
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
            ),
            RigidObjCfg(
                name="banana",
                scale=(1, 1, 1),
                physics=PhysicStateType.RIGIDBODY,
                usd_path="roboverse_data/assets/EmbodiedGenData/demo_assets/banana/usd/banana.usd",
                urdf_path="roboverse_data/assets/EmbodiedGenData/demo_assets/banana/result/banana.urdf",
                mjcf_path="roboverse_data/assets/EmbodiedGenData/demo_assets/banana/mjcf/banana.mjcf",
            ),
            RigidObjCfg(
                name="mug",
                scale=(1, 1, 1),
                physics=PhysicStateType.RIGIDBODY,
                usd_path="roboverse_data/assets/EmbodiedGenData/demo_assets/mug/usd/mug.usd",
                urdf_path="roboverse_data/assets/EmbodiedGenData/demo_assets/mug/result/mug.urdf",
                mjcf_path="roboverse_data/assets/EmbodiedGenData/demo_assets/mug/mjcf/mug.mjcf",
            ),
            RigidObjCfg(
                name="book",
                scale=(1, 1, 1),
                physics=PhysicStateType.RIGIDBODY,
                usd_path="roboverse_data/assets/EmbodiedGenData/demo_assets/book/usd/book.usd",
                urdf_path="roboverse_data/assets/EmbodiedGenData/demo_assets/book/result/book.urdf",
                mjcf_path="roboverse_data/assets/EmbodiedGenData/demo_assets/book/mjcf/book.mjcf",
            ),
            # RigidObjCfg(
            #     name="lamp",
            #     scale=(1, 1, 1),
            #     physics=PhysicStateType.RIGIDBODY,
            #     usd_path="roboverse_data/assets/EmbodiedGenData/demo_assets/lamp/usd/lamp.usd",
            #     urdf_path="roboverse_data/assets/EmbodiedGenData/demo_assets/lamp/result/lamp.urdf",
            #     mjcf_path="roboverse_data/assets/EmbodiedGenData/demo_assets/lamp/mjcf/lamp.mjcf",
            # ),
            RigidObjCfg(
                name="remote_control",
                scale=(1, 1, 1),
                physics=PhysicStateType.RIGIDBODY,
                usd_path="roboverse_data/assets/EmbodiedGenData/demo_assets/remote_control/usd/remote_control.usd",
                urdf_path="roboverse_data/assets/EmbodiedGenData/demo_assets/remote_control/result/remote_control.urdf",
                mjcf_path="roboverse_data/assets/EmbodiedGenData/demo_assets/remote_control/mjcf/remote_control.mjcf",
            ),
            RigidObjCfg(
                name="rubiks_cube",
                scale=(1, 1, 1),
                physics=PhysicStateType.RIGIDBODY,
                usd_path="roboverse_data/assets/EmbodiedGenData/demo_assets/rubik's_cube/usd/rubik's_cube.usd",
                urdf_path="roboverse_data/assets/EmbodiedGenData/demo_assets/rubik's_cube/result/rubik's_cube.urdf",
                mjcf_path="roboverse_data/assets/EmbodiedGenData/demo_assets/rubik's_cube/mjcf/rubik's_cube.mjcf",
            ),
            RigidObjCfg(
                name="vase",
                scale=(1, 1, 1),
                physics=PhysicStateType.RIGIDBODY,
                usd_path="roboverse_data/assets/EmbodiedGenData/demo_assets/vase/usd/vase.usd",
                urdf_path="roboverse_data/assets/EmbodiedGenData/demo_assets/vase/result/vase.urdf",
                mjcf_path="roboverse_data/assets/EmbodiedGenData/demo_assets/vase/mjcf/vase.mjcf",
            ),
        ],
        robots=["franka"],
    )

    # Checker: banana should be placed inside the mug
    checker = DetectedChecker(
        obj_name="banana",
        detector=RelativeBboxDetector(
            base_obj_name="mug",
            relative_pos=[0.0, 0.0, 0.05],  # Slightly above mug bottom, inside the mug
            relative_quat=[1.0, 0.0, 0.0, 0.0],
            checker_lower=[-0.04, -0.04, -0.03],  # Tight tolerance to ensure it's inside
            checker_upper=[0.04, 0.04, 0.06],  # Allow some height variation
            ignore_base_ori=True,  # Don't care about banana orientation
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
                    "banana": {
                        "pos": torch.tensor([0.28, -0.58, 0.825]),  # Starting position on table (left)
                        "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                    },
                    "mug": {
                        "pos": torch.tensor([0.48, -0.54, 0.863]),  # Target: mug on table (right)
                        "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                    },
                    "book": {
                        "pos": torch.tensor([0.3, -0.28, 0.82]),  # Book on table
                        "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                    },
                    # "lamp": {
                    #     "pos": torch.tensor([0.68, 0.10, 1.05]),  # Lamp on table
                    #     "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                    # },
                    "remote_control": {
                        "pos": torch.tensor([0.68, -0.54, 0.811]),  # Remote on table
                        "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                    },
                    "rubiks_cube": {
                        "pos": torch.tensor([0.68, -0.34, 0.83]),  # Rubik's cube on table
                        "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                    },
                    "vase": {
                        "pos": torch.tensor([0.30, 0.05, 0.95]),  # Vase on table
                        "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                    },
                },
                "robots": {
                    "franka": {
                        "pos": torch.tensor([0.890000, -0.250001, 0.780000]),
                        "rot": torch.tensor([-0.029191, -0.024987, 0.000730, -0.999261]),
                        "dof_pos": {
                            "panda_finger_joint1": 0.040000,
                            "panda_finger_joint2": 0.040000,
                            "panda_joint1": -0.000000,
                            "panda_joint2": -0.785398,
                            "panda_joint3": -0.000000,
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

        return init_states
