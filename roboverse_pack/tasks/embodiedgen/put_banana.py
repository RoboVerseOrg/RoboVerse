"""Put banana on table task using EmbodiedGen assets."""

from __future__ import annotations

import torch

from metasim.constants import PhysicStateType
from metasim.scenario.lights import DiskLightCfg
from metasim.scenario.objects import RigidObjCfg
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
            ),
        ],
        robots=["vega"],
        lights=[
            DiskLightCfg(
                name="overhead_light",
                intensity=10000.0,
                color=(1.0, 1.0, 1.0),
                radius=1.0,
                pos=(0.0, 0.0, 2.0),  # 原点上方 2m
                rot=(0.7071, 0.0, 0.0, 0.7071),  # 45° 向下照射
            ),
        ],
        sim_params=SimParamCfg(
            dt=0.005,
        ),
        decimation=4,
        scene="kujiale_scene_0008",
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
                },
                "robots": {
                    "vega": {
                        "pos": torch.tensor([-0.353636, -0.230209, 0.00511]),
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
