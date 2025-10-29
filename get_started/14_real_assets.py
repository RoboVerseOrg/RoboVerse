from __future__ import annotations

from typing import Literal

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass
import rootutils
import torch
import tyro
from loguru import logger as log
from rich.logging import RichHandler

rootutils.setup_root(__file__, pythonpath=True)
log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])
import os

from huggingface_hub import snapshot_download

from metasim.constants import PhysicStateType, SimType
from metasim.scenario.cameras import PinholeCameraCfg
from metasim.scenario.objects import RigidObjCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.utils import configclass
from metasim.utils.obs_utils import ObsSaver
from metasim.utils.setup_util import get_sim_handler_class


@configclass
class RealAssetCfg:
    """Arguments for the static scene."""

    robot: str = "franka"
    sim: Literal[
        "isaacsim",
        "isaacgym",
        "genesis",
        "pybullet",
        "sapien3",
        "mujoco",
    ] = "isaacsim"
    num_envs: int = 1
    headless: bool = True

    def __post_init__(self):
        log.info(f"RealAssetCfg: {self}")


if __name__ == "__main__":
    args = tyro.cli(RealAssetCfg)

    # download EmbodiedGen assets from huggingface dataset
    data_dir = "roboverse_data/assets/EmbodiedGenData"
    snapshot_download(
        repo_id="HorizonRobotics/EmbodiedGenData",
        repo_type="dataset",
        local_dir=data_dir,
        allow_patterns="demo_assets/*",
        local_dir_use_symlinks=False,
    )

    # initialize scenario
    scenario = ScenarioCfg(
        robots=[args.robot],
        headless=args.headless,
        num_envs=args.num_envs,
        simulator=args.sim,
    )

    # add cameras
    scenario.cameras = [
        PinholeCameraCfg(
            name="camera",
            width=1024,
            height=1024,
            pos=(2, -1, 1.5),
            look_at=(0.0, 0.0, 0.0),
        )
    ]

    # add objects
    scenario.objects = [
        RigidObjCfg(
            name="table",
            scale=(1, 1, 1),
            physics=PhysicStateType.RIGIDBODY,
            fix_base_link=True,
            usd_path=f"{data_dir}/demo_assets/table/usd/table.usd",
            urdf_path=f"{data_dir}/demo_assets/table/result/table.urdf",
            mjcf_path=f"{data_dir}/demo_assets/table/mjcf/table.mjcf",
        ),
        RigidObjCfg(
            name="banana",
            scale=(1, 1, 1),
            fix_base_link=True,
            physics=PhysicStateType.GEOM,
            usd_path=f"{data_dir}/demo_assets/banana/usd/banana.usd",
            urdf_path=f"{data_dir}/demo_assets/banana/result/banana.urdf",
            mjcf_path=f"{data_dir}/demo_assets/banana/mjcf/banana.mjcf",
        ),
        RigidObjCfg(
            name="book",
            scale=(1, 1, 1),
            physics=PhysicStateType.RIGIDBODY,
            usd_path=f"{data_dir}/demo_assets/book/usd/book.usd",
            urdf_path=f"{data_dir}/demo_assets/book/result/book.urdf",
            mjcf_path=f"{data_dir}/demo_assets/book/mjcf/book.mjcf",
        ),
        RigidObjCfg(
            name="lamp",
            scale=(1, 1, 1),
            physics=PhysicStateType.RIGIDBODY,
            usd_path=f"{data_dir}/demo_assets/lamp/usd/lamp.usd",
            urdf_path=f"{data_dir}/demo_assets/lamp/result/lamp.urdf",
            mjcf_path=f"{data_dir}/demo_assets/lamp/mjcf/lamp.mjcf",
        ),
        RigidObjCfg(
            name="mug",
            scale=(1, 1, 1),
            physics=PhysicStateType.RIGIDBODY,
            usd_path=f"{data_dir}/demo_assets/mug/usd/mug.usd",
            urdf_path=f"{data_dir}/demo_assets/mug/result/mug.urdf",
            mjcf_path=f"{data_dir}/demo_assets/mug/mjcf/mug.mjcf",
        ),
        RigidObjCfg(
            name="remote_control",
            scale=(1, 1, 1),
            physics=PhysicStateType.RIGIDBODY,
            usd_path=f"{data_dir}/demo_assets/remote_control/usd/remote_control.usd",
            urdf_path=f"{data_dir}/demo_assets/remote_control/result/remote_control.urdf",
            mjcf_path=f"{data_dir}/demo_assets/remote_control/mjcf/remote_control.mjcf",
        ),
        RigidObjCfg(
            name="rubiks_cube",
            scale=(1, 1, 1),
            physics=PhysicStateType.RIGIDBODY,
            usd_path=f"{data_dir}/demo_assets/rubik's_cube/usd/rubik's_cube.usd",
            urdf_path=f"{data_dir}/demo_assets/rubik's_cube/result/rubik's_cube.urdf",
            mjcf_path=f"{data_dir}/demo_assets/rubik's_cube/mjcf/rubik's_cube.mjcf",
        ),
        RigidObjCfg(
            name="vase",
            scale=(1, 1, 1),
            physics=PhysicStateType.RIGIDBODY,
            usd_path=f"{data_dir}/demo_assets/vase/usd/vase.usd",
            urdf_path=f"{data_dir}/demo_assets/vase/result/vase.urdf",
            mjcf_path=f"{data_dir}/demo_assets/vase/mjcf/vase.mjcf",
        ),
    ]

    # set initial states
    init_states = [
        {
            "objects": {
                "table": {
                    "pos": torch.tensor([0.4, -0.2, 0.4]),
                    "rot": torch.tensor([1, 0, 0, 0]),
                },
                "banana": {
                    "pos": torch.tensor([0.28, -0.58, 0.825]),
                    "rot": torch.tensor([1, 0, 0, 0]),
                },
                "book": {
                    "pos": torch.tensor([0.3, -0.28, 0.82]),
                    "rot": torch.tensor([1, 0, 0, 0]),
                },
                "lamp": {
                    "pos": torch.tensor([0.68, 0.10, 1.05]),
                    "rot": torch.tensor([1, 0, 0, 0]),
                },
                "mug": {
                    "pos": torch.tensor([0.68, -0.34, 0.863]),
                    "rot": torch.tensor([1, 0, 0, 0]),
                },
                "remote_control": {
                    "pos": torch.tensor([0.68, -0.54, 0.811]),
                    "rot": torch.tensor([1, 0, 0, 0]),
                },
                "rubiks_cube": {
                    "pos": torch.tensor([0.48, -0.54, 0.83]),
                    "rot": torch.tensor([1, 0, 0, 0]),
                },
                "vase": {
                    "pos": torch.tensor([0.30, 0.05, 0.95]),
                    "rot": torch.tensor([1, 0, 0, 0]),
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
    ]

    log.info(f"Using simulator: {args.sim}")
    env_class = get_sim_handler_class(SimType(args.sim))
    handler = env_class(scenario)
    handler.launch()
    handler.set_states(init_states * scenario.num_envs)
    os.makedirs("get_started/output", exist_ok=True)

    obs = handler.get_states(mode="tensor")
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
        handler.set_dof_targets(actions)
        handler.simulate()
        obs = handler.get_states(mode="tensor")
        obs_saver.add(obs)
        step += 1

    obs_saver.save()
