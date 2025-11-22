"""This script is used to test the static scene."""

from __future__ import annotations

from typing import Literal

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

import os

import rootutils
import torch
import tyro
from loguru import logger as log
from rich.logging import RichHandler

rootutils.setup_root(__file__, pythonpath=True)
log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])


from metasim.constants import PhysicStateType
from metasim.scenario.cameras import PinholeCameraCfg
from metasim.scenario.objects import (
    ArticulationObjCfg,
    PrimitiveCubeCfg,
    PrimitiveSphereCfg,
    RigidObjCfg,
)
from metasim.scenario.scenario import ScenarioCfg
from metasim.utils import configclass
from metasim.utils.obs_utils import ObsSaver
from metasim.utils.setup_util import get_handler

if __name__ == "__main__":

    @configclass
    class Args:
        """Arguments for the static scene."""

        robot: str = "vega"

        ## Handlers
        sim: Literal[
            "isaacsim",
            "isaacgym",
            "genesis",
            "pybullet",
            "sapien2",
            "sapien3",
            "mujoco",
        ] = "isaacgym"

        ## Others
        num_envs: int = 1
        headless: bool = False

        def __post_init__(self):
            """Post-initialization configuration."""
            log.info(f"Args: {self}")

    args = tyro.cli(Args)

    # initialize scenario
    scenario = ScenarioCfg(
        robots=[args.robot],
        simulator=args.sim,
        headless=args.headless,
        num_envs=args.num_envs,
    )

    # add cameras
    scenario.cameras = [PinholeCameraCfg(width=1024, height=1024, pos=(1.5, -1.5, 1.5), look_at=(0.0, 0.0, 0.0))]

    # add objects
    scenario.objects = [
        PrimitiveCubeCfg(
            name="cube",
            size=(0.1, 0.1, 0.1),
            color=[1.0, 0.0, 0.0],
            physics=PhysicStateType.RIGIDBODY,
        ),
        PrimitiveSphereCfg(
            name="sphere",
            radius=0.1,
            color=[0.0, 0.0, 1.0],
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="bbq_sauce",
            scale=(2, 2, 2),
            physics=PhysicStateType.RIGIDBODY,
            usd_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/bbq_sauce/usd/bbq_sauce.usd",
            urdf_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/bbq_sauce/urdf/bbq_sauce.urdf",
            mjcf_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/bbq_sauce/mjcf/bbq_sauce.xml",
        ),
    ]

    log.info(f"Using simulator: {args.sim}")
    handler = get_handler(scenario)

    init_states = [
        {
            "objects": {
                "cube": {
                    "pos": torch.tensor([0.6, -0.2, 0.05]),
                    "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                },
                "sphere": {
                    "pos": torch.tensor([0.4, -0.2, 0.05]),
                    "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                },
                "bbq_sauce": {
                    "pos": torch.tensor([0.7, -0.3, 0.14]),
                    "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                },

            },
            "robots": {
                "vega": {
                    "pos": torch.tensor([0.0, 0.0, 0.0]),
                    "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                    "dof_pos": {
                        # Base wheels - neutral
                        "B_wheel_j1": 0.0,
                        "B_wheel_j2": 0.0,
                        "R_wheel_j1": 0.0,
                        "R_wheel_j2": 0.0,
                        "L_wheel_j1": 0.0,
                        "L_wheel_j2": 0.0,
                        # Torso - upright
                        "torso_j1": 0.0,
                        "torso_j2": 0.0,
                        "torso_j3": 0.0,
                        # # Head - forward looking
                        # "head_j1": 0.0,
                        # "head_j2": 0.0,
                        # "head_j3": 0.0,
                        # Left arm - neutral pose
                        "L_arm_j1": 0.0,
                        "L_arm_j2": 0.0,
                        "L_arm_j3": 0.0,
                        "L_arm_j4": 0.0,
                        "L_arm_j5": 0.0,
                        "L_arm_j6": 0.0,
                        "L_arm_j7": 0.0,
                        # Right arm - neutral pose
                        "R_arm_j1": 0.0,
                        "R_arm_j2": 0.0,
                        "R_arm_j3": 0.0,
                        "R_arm_j4": 0.0,
                        "R_arm_j5": 0.0,
                        "R_arm_j6": 0.0,
                        "R_arm_j7": 0.0,
                        # Left hand - open
                        "L_th_j0": 0.0,
                        "L_th_j1": 0.0,
                        "L_th_j2": 0.0,
                        "L_ff_j1": 0.0,
                        "L_ff_j2": 0.0,
                        "L_mf_j1": 0.0,
                        "L_mf_j2": 0.0,
                        "L_rf_j1": 0.0,
                        "L_rf_j2": 0.0,
                        "L_lf_j1": 0.0,
                        "L_lf_j2": 0.0,
                        # Right hand - open
                        "R_th_j0": 0.0,
                        "R_th_j1": 0.0,
                        "R_th_j2": 0.0,
                        "R_ff_j1": 0.0,
                        "R_ff_j2": 0.0,
                        "R_mf_j1": 0.0,
                        "R_mf_j2": 0.0,
                        "R_rf_j1": 0.0,
                        "R_rf_j2": 0.0,
                        "R_lf_j1": 0.0,
                        "R_lf_j2": 0.0,
                    },
                },
            },
        }
    ]
    handler.set_states(init_states * scenario.num_envs)
    os.makedirs("get_started/output", exist_ok=True)

    obs = handler.get_states(mode="tensor")
    ## Main loop
    obs_saver = ObsSaver(video_path=f"get_started/output/1_move_robot_{args.sim}.mp4")
    obs_saver.add(obs)

    step = 0
    robot = scenario.robots[0]
    for _ in range(10000):
        log.debug(f"Step {step}")
        actions = [
            {
                robot.name: {
                    "dof_pos_target": {
                        joint_name: (
                            torch.rand(1).item()*1.0
                            * (robot.joint_limits[joint_name][1] - robot.joint_limits[joint_name][0])
                            + robot.joint_limits[joint_name][0]*1.0
                        )
                        for joint_name in robot.joint_limits.keys()
                    }
                }
            }
            for _ in range(scenario.num_envs)
        ]
        handler.set_dof_targets(actions)
        handler.simulate()
        obs = handler.get_states(mode="tensor")
        obs_saver.add(obs)
        step += 1

    obs_saver.save()
