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


from metasim.constants import PhysicStateType, SimType
from metasim.scenario.cameras import PinholeCameraCfg
from metasim.scenario.objects import ArticulationObjCfg, PrimitiveCubeCfg, PrimitiveSphereCfg, RigidObjCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.utils import configclass
from metasim.utils.obs_utils import ObsSaver
from metasim.utils.setup_util import get_sim_handler_class

if __name__ == "__main__":

    @configclass
    class Args:
        """Arguments for the static scene."""

        robot: str = "franka"

        ## Handlers
        sim: Literal["isaacsim", "isaacgym", "genesis", "pybullet", "sapien2", "sapien3", "mujoco"] = "mujoco"

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
        RigidObjCfg(
            name="handle",
            mjcf_path="roboverse_data/assets/playground/open_cabinet/mjcf/handle.xml",
            physics=PhysicStateType.RIGIDBODY,
            fix_base_link=True,
        ),
        # Target position for the handle
        PrimitiveSphereCfg(
            name="target",
            mass=0.01,
            radius=0.02,
            physics=PhysicStateType.XFORM,
            color=(0.7, 1.0, 0.7),  # Green color for target
        ),
        # Barrier to avoid collision
        RigidObjCfg(
            name="barrier",
            mjcf_path="roboverse_data/assets/playground/open_cabinet/mjcf/barrier.xml",
            physics=PhysicStateType.RIGIDBODY,
            fix_base_link=True,
        ),
    ]

    log.info(f"Using simulator: {args.sim}")
    env_class = get_sim_handler_class(SimType(args.sim))
    env = env_class(scenario)

    init_states = [
        {
            "objects": {
                "cube": {
                    "pos": torch.tensor([0.3, -0.2, 0.05]),
                    "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                },
                "sphere": {
                    "pos": torch.tensor([0.4, -0.6, 0.05]),
                    "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                },
                "bbq_sauce": {
                    "pos": torch.tensor([0.7, -0.3, 0.14]),
                    "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                },
                "box_base": {
                    "pos": torch.tensor([0.5, 0.2, 0.1]),
                    "rot": torch.tensor([0.0, 0.7071, 0.0, 0.7071]),
                    "dof_pos": {"box_joint": 0.0},
                },
            },
            "robots": {
                "franka": {
                    "pos": torch.tensor([0.0, 0.0, 0.0]),
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
    ]
    env.launch()
    env.set_states(init_states * scenario.num_envs)
    os.makedirs("get_started/output", exist_ok=True)

    obs = env.get_states(mode="tensor")
    ## Main loop
    obs_saver = ObsSaver(video_path=f"get_started/output/1_move_robot_{args.sim}.mp4")
    obs_saver.add(obs)

    step = 0
    robot = scenario.robots[0]
    for _ in range(100):
        log.debug(f"Step {step}")
        actions = [
            {
                robot.name: {
                    "dof_pos_target": {
                        joint_name: (
                            torch.rand(1).item()
                            * (robot.joint_limits[joint_name][1] - robot.joint_limits[joint_name][0])
                            + robot.joint_limits[joint_name][0]
                        )
                        for joint_name in robot.joint_limits.keys()
                    }
                }
            }
            for _ in range(scenario.num_envs)
        ]
        env.set_dof_targets(actions)
        env.simulate()
        obs = env.get_states(mode="tensor")
        obs_saver.add(obs)
        step += 1

    obs_saver.save()
