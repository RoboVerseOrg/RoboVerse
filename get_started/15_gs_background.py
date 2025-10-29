from __future__ import annotations

from typing import Literal

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass
import numpy as np
import rootutils
import torch
import tyro
from loguru import logger as log
from rich.logging import RichHandler

rootutils.setup_root(__file__, pythonpath=True)
log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])
import os

import cv2
from huggingface_hub import snapshot_download

from metasim.constants import PhysicStateType, SimType
from metasim.scenario.cameras import PinholeCameraCfg
from metasim.scenario.objects import RigidObjCfg
from metasim.scenario.scenario import GSSceneCfg, ScenarioCfg
from metasim.utils import configclass
from metasim.utils.setup_util import get_sim_handler_class


def depth_to_colormap(depth, inv_depth=True, depth_range=(0.1, 3.0), colormap=cv2.COLORMAP_TURBO):
    depth = np.squeeze(depth)

    dmin, dmax = depth_range
    valid = depth > 0

    if inv_depth:
        depth_clip = np.clip(depth, dmin, dmax, where=valid, out=np.copy(depth))
        inv = np.zeros_like(depth_clip, dtype=np.float32)
        inv[valid] = 1.0 / depth_clip[valid]
        nmin, nmax = 1.0 / dmax, 1.0 / dmin
        norm = (inv - nmin) / (nmax - nmin)
    else:
        depth_clip = np.clip(depth, dmin, dmax, where=valid, out=np.copy(depth))
        norm = (depth_clip - dmin) / (dmax - dmin)

    norm = np.nan_to_num(norm, nan=0.0, posinf=0.0, neginf=0.0)
    img_u8 = (np.clip(norm, 0.0, 1.0) * 255.0).astype(np.uint8)  # 2D, uint8
    color_bgr = cv2.applyColorMap(img_u8, colormap)  # (H,W,3) BGR
    return color_bgr[:, :, ::-1]  # RGB


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
    ] = "mujoco"
    num_envs: int = 1  # only support single env for now
    assert num_envs == 1, "Only support single env for now"

    scene_id: int = 16  # 0-16
    assert scene_id in range(17), "Only support scene_id 0-16 for now"

    headless: bool = True
    with_gs_background: bool = True

    def __post_init__(self):
        log.info(f"RealAssetCfg: {self}")


if __name__ == "__main__":
    args = tyro.cli(RealAssetCfg)

    # download EmbodiedGen assets from huggingface dataset
    data_dir = "roboverse_data/assets/EmbodiedGenData"

    snapshot_download(
        repo_id="xinjjj/scene3d-bg",
        repo_type="dataset",
        local_dir=data_dir,
        allow_patterns=f"bg_scenes/scene_{args.scene_id:03d}/*.ply",
        local_dir_use_symlinks=False,
    )

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
        gs_scene=GSSceneCfg(
            with_gs_background=args.with_gs_background,
            gs_background_path=f"{data_dir}/bg_scenes/scene_{args.scene_id:03d}/gs_model.ply",
            gs_background_pose_tum=(0, 0, 0, 0, 1, 0, 0),  # format: (x, y, z, qx, qy, qz, qw)
        ),
    )

    # add cameras
    scenario.cameras = [
        PinholeCameraCfg(
            name="camera",
            width=1024,
            height=1024,
            pos=(2, -1, 1.5),
            look_at=(0.0, 0.0, 0.0),
            data_types=["rgb", "depth", "instance_seg"],  # Enable instance segmentation for GS blending
        )
    ]

    # add objects
    scenario.objects = [
        RigidObjCfg(
            name="table",
            scale=(1, 1, 1),
            physics=PhysicStateType.RIGIDBODY,
            usd_path=f"{data_dir}/demo_assets/table/usd/table.usd",
            urdf_path=f"{data_dir}/demo_assets/table/result/table.urdf",
            mjcf_path=f"{data_dir}/demo_assets/table/mjcf/table.mjcf",
        ),
        RigidObjCfg(
            name="banana",
            scale=(1, 1, 1),
            physics=PhysicStateType.RIGIDBODY,
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
    env = env_class(scenario)
    env.launch()
    env.set_states(init_states)
    obs = env.get_states(mode="dict")[0]  # get states as a dictionary
    # obs_tensor = env.get_states(mode="tensor")  # get states as a tensor

    os.makedirs("get_started/output", exist_ok=True)

    # save rgb image
    save_path = f"get_started/output/15_gs_background_{args.sim}.jpg"
    log.info(f"Saving image to {save_path}")
    rgb = obs["cameras"]["camera"]["rgb"]
    if isinstance(rgb, torch.Tensor):
        rgb = rgb.detach().cpu().numpy()
    rgb = rgb.squeeze()
    rgb = rgb[:, :, :3]
    cv2.imwrite(save_path, rgb[:, :, ::-1].copy(), [int(cv2.IMWRITE_JPEG_QUALITY), 95])  # RGB -> BGR

    # save depth image
    save_path = f"get_started/output/15_gs_background_{args.sim}_depth.jpg"
    log.info(f"Saving depth image to {save_path}")
    depth = obs["cameras"]["camera"]["depth"]
    if isinstance(depth, torch.Tensor):
        depth = depth.detach().cpu().numpy()
    depth = depth.squeeze()
    depth_color = depth_to_colormap(depth, inv_depth=True, depth_range=(1.0, 5.0))
    cv2.imwrite(save_path, depth_color[:, :, ::-1].copy(), [int(cv2.IMWRITE_JPEG_QUALITY), 95])  # RGB -> BGR
