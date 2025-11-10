from __future__ import annotations

import argparse
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

import imageio
from huggingface_hub import snapshot_download
from tqdm import tqdm

from generation.enums import AssetType, SimAssetMapper
from generation.load_asset import load_embodiedgen_asset, load_embodiedgen_layout_pose
from metasim.constants import SimType
from metasim.scenario.cameras import PinholeCameraCfg
from metasim.scenario.robot import RobotCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.utils import configclass
from metasim.utils.obs_utils import ObsSaver
from metasim.utils.setup_util import get_sim_handler_class


@configclass
class LayoutInitCfg:
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
    args = tyro.cli(LayoutInitCfg)

    # Download EmbodiedGen layout scene from huggingface dataset
    # You can download more scenes from example_layouts folder under https://huggingface.co/datasets/HorizonRobotics/EmbodiedGenData
    data_dir = "roboverse_data/assets/EmbodiedGenData"
    snapshot_download(
        repo_id="HorizonRobotics/EmbodiedGenData",
        repo_type="dataset",
        local_dir=data_dir,
        allow_patterns="example_layouts/task_0002/*",
    )

    layout_file = "roboverse_data/assets/EmbodiedGenData/example_layouts/task_0002/layout.json"

    # initialize scenario
    scenario = ScenarioCfg(
        robots=[args.robot],
        headless=args.headless,
        num_envs=args.num_envs,
        simulator=args.sim,
        decimation=2,
    )
    if args.sim == "mujoco":
        scenario.decimation *= 10

    simulation_app = None
    if args.sim == "isaacsim":
        # Initialize IsaacLab app, use for asset conversion and isaac simulate.
        from isaaclab.app import AppLauncher

        parser = argparse.ArgumentParser()
        AppLauncher.add_app_launcher_args(parser)
        isaac_args = parser.parse_args([])
        isaac_args.enable_cameras = True
        isaac_args.headless = args.headless
        app_launcher = AppLauncher(isaac_args)
        simulation_app = app_launcher.app

    scenario.cameras = [
        PinholeCameraCfg(
            name="camera",
            width=1024,
            height=1024,
            pos=(2, -1, 1.5),
            look_at=(0.0, 0.0, 0.3),
        )
    ]
    scenario.objects = load_embodiedgen_asset(
        layout_file,
        target_type=SimAssetMapper[args.sim],
        source_type=AssetType.MESH,
        simulation_app=simulation_app,
        exit_close=False,
    )
    if args.sim == "genesis":
        for obj_cfg in scenario.objects:
            # obj_cfg.genesis_read_mjcf = True
            obj_cfg.file_type: dict[str, str] = {**RobotCfg.file_type, "isaacgym": "mjcf"}

    scenario.init_states = load_embodiedgen_layout_pose(layout_file, z_offset=0.1)
    scenario.init_states[0]["robots"]["franka"] = scenario.init_states[0]["robots"].pop("default")
    scenario.init_states[0]["robots"]["franka"]["dof_pos"] = {
        "panda_joint1": 0.0,
        "panda_joint2": -0.785398,
        "panda_joint3": 0.0,
        "panda_joint4": -2.356194,
        "panda_joint5": 0.0,
        "panda_joint6": 1.570796,
        "panda_joint7": 0.785398,
        "panda_finger_joint1": 0.04,
        "panda_finger_joint2": 0.04,
    }
    log.info(f"Using simulator: {args.sim}")
    env_class = get_sim_handler_class(SimType(args.sim))
    handler = env_class(scenario)

    # Asset usd convert need simulation_app, simulation_app can not init once,
    # therefore we pass it to handler.launch()
    if simulation_app is not None and isaac_args is not None:
        handler.launch(simulation_app, isaac_args)
    else:
        handler.launch()
    handler.set_states(scenario.init_states * scenario.num_envs)
    os.makedirs("get_started/output", exist_ok=True)

    # First frame image
    save_path = f"get_started/output/16_embodiedgen_layout_{args.sim}.png"
    log.info(f"Saving image to {save_path}")
    obs = handler.get_states(mode="dict")[0]
    imageio.imwrite(save_path, obs["cameras"]["camera"]["rgb"])

    # Video
    obs_saver = ObsSaver(video_path=f"get_started/output/16_embodiedgen_layout_dynamic_{args.sim}.mp4")
    total_step = 100
    robot = scenario.robots[0]
    for idx in tqdm(range(total_step)):
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

    obs_saver.save()
    if hasattr(handler, "simulation_app"):
        handler.close()
        handler.simulation_app.close()
