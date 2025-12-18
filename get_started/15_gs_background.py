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
from tqdm import tqdm

from metasim.constants import PhysicStateType, SimType
from metasim.scenario.cameras import PinholeCameraCfg
from metasim.scenario.lights import DistantLightCfg, DomeLightCfg, SphereLightCfg
from metasim.scenario.objects import RigidObjCfg
from metasim.scenario.robot import RobotCfg
from metasim.scenario.scenario import GSSceneCfg, ScenarioCfg
from metasim.types import CameraState
from metasim.utils import configclass
from metasim.utils.obs_utils import ObsSaver
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
    ] = "isaacsim"
    num_envs: int = 1  # only support single env for now
    assert num_envs == 1, "Only support single env for now"

    scene_id: int = 16  # 0-16
    assert scene_id in range(17), "Only support scene_id 0-16 for now"

    headless: bool = True
    with_gs_background: bool = True
    num_views: int = 60
    circle_around: bool = False

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
        decimation=2,
        gs_scene=GSSceneCfg(
            with_gs_background=args.with_gs_background,
            gs_background_path=f"{data_dir}/bg_scenes/scene_{args.scene_id:03d}/gs_model.ply",
            gs_background_pose_tum=(0, 0, 0, 0, 1, 0, 0),  # format: (x, y, z, qx, qy, qz, qw)
        ),
    )
    if args.sim == "mujoco":
        scenario.decimation *= 10

    # add lights
    scenario.lights = [
        # Main overhead light (Key Light)
        SphereLightCfg(
            pos=(0.0, 0.0, 2.0),
            radius=0.5,
            intensity=4000.0,
            color=(1.0, 0.99, 0.96),
        ),
        # Fill lights from 4 directions to soften shadows
        DistantLightCfg(
            intensity=600.0,
            color=(1.0, 1.0, 1.0),
            azimuth=0.0,
            polar=45.0,
        ),
        DistantLightCfg(
            intensity=600.0,
            color=(1.0, 1.0, 1.0),
            azimuth=90.0,
            polar=45.0,
        ),
        DistantLightCfg(
            intensity=600.0,
            color=(1.0, 1.0, 1.0),
            azimuth=180.0,
            polar=45.0,
        ),
        DistantLightCfg(
            intensity=600.0,
            color=(1.0, 1.0, 1.0),
            azimuth=270.0,
            polar=45.0,
        ),
        # Ambient light
        DomeLightCfg(
            intensity=300.0,
            color=(0.8, 0.8, 0.9),
        ),
    ]

    # add cameras
    total_num_views = args.num_views
    num_cameras = 1  # Only create 1 physical camera to save resources
    cameras = []
    radius = 1.0
    height = 1.3
    look_at = (0.0, 0.0, 0.8)  # Look at table surface center
    center = (0.0, 0.0, 0.0)  # Camera center of rotation

    fovy_deg = 75.0
    image_hw = (512, 512)  # (H, W)
    horizontal_aperture = 20.955
    vertical_aperture = horizontal_aperture * image_hw[0] / image_hw[1]
    focal_length = vertical_aperture / (2 * np.tan(np.deg2rad(fovy_deg) / 2))

    # Create a single camera instance
    cameras.append(
        PinholeCameraCfg(
            name="camera_0",
            width=image_hw[1],
            height=image_hw[0],
            focal_length=focal_length,
            horizontal_aperture=horizontal_aperture,
            pos=(center[0] + radius, center[1], height),  # Initial position
            look_at=look_at,
            data_types=[
                "rgb",
                "depth",
                "instance_seg",
                "instance_id_seg",
            ],  # Enable instance segmentation for GS blending
        )
    )
    scenario.cameras = cameras

    # add objects
    scenario.objects = [
        RigidObjCfg(
            name="table",
            scale=(1, 1, 1),
            physics=PhysicStateType.RIGIDBODY,
            fix_base_link=True,
            usd_path=f"{data_dir}/demo_assets/table/usd/table.usd",
            urdf_path=f"{data_dir}/demo_assets/table/result/table.urdf",
            mjcf_path=f"{data_dir}/demo_assets/table/mjcf/table.xml",
            file_type={**RobotCfg.file_type, "isaacgym": "mjcf", "genesis": "mjcf"},
            # You need set pose for fix_base_link object to update usd stage for isaac 5.0.
            default_position=(0.0, 0.0, 0.4),
            default_orientation=(1.0, 0.0, 0.0, 0.0),
        ),
        RigidObjCfg(
            name="banana",
            scale=(1, 1, 1),
            physics=PhysicStateType.RIGIDBODY,
            usd_path=f"{data_dir}/demo_assets/banana/usd/banana.usd",
            urdf_path=f"{data_dir}/demo_assets/banana/result/banana.urdf",
            mjcf_path=f"{data_dir}/demo_assets/banana/mjcf/banana.xml",
            file_type={**RobotCfg.file_type, "isaacgym": "mjcf", "genesis": "mjcf"},
        ),
        RigidObjCfg(
            name="book",
            scale=(1, 1, 1),
            physics=PhysicStateType.RIGIDBODY,
            usd_path=f"{data_dir}/demo_assets/book/usd/book.usd",
            urdf_path=f"{data_dir}/demo_assets/book/result/book.urdf",
            mjcf_path=f"{data_dir}/demo_assets/book/mjcf/book.xml",
            file_type={**RobotCfg.file_type, "isaacgym": "mjcf", "genesis": "mjcf"},
        ),
        RigidObjCfg(
            name="lamp",
            scale=(1, 1, 1),
            physics=PhysicStateType.RIGIDBODY,
            usd_path=f"{data_dir}/demo_assets/lamp/usd/lamp.usd",
            urdf_path=f"{data_dir}/demo_assets/lamp/result/lamp.urdf",
            mjcf_path=f"{data_dir}/demo_assets/lamp/mjcf/lamp.xml",
            file_type={**RobotCfg.file_type, "isaacgym": "mjcf", "genesis": "mjcf"},
        ),
        RigidObjCfg(
            name="mug",
            scale=(1, 1, 1),
            physics=PhysicStateType.RIGIDBODY,
            usd_path=f"{data_dir}/demo_assets/mug/usd/mug.usd",
            urdf_path=f"{data_dir}/demo_assets/mug/result/mug.urdf",
            mjcf_path=f"{data_dir}/demo_assets/mug/mjcf/mug.xml",
            file_type={**RobotCfg.file_type, "isaacgym": "mjcf", "genesis": "mjcf"},
        ),
        RigidObjCfg(
            name="remote_control",
            scale=(1, 1, 1),
            physics=PhysicStateType.RIGIDBODY,
            usd_path=f"{data_dir}/demo_assets/remote_control/usd/remote_control.usd",
            urdf_path=f"{data_dir}/demo_assets/remote_control/result/remote_control.urdf",
            mjcf_path=f"{data_dir}/demo_assets/remote_control/mjcf/remote_control.xml",
            file_type={**RobotCfg.file_type, "isaacgym": "mjcf", "genesis": "mjcf"},
        ),
        RigidObjCfg(
            name="rubiks_cube",
            scale=(1, 1, 1),
            physics=PhysicStateType.RIGIDBODY,
            usd_path=f"{data_dir}/demo_assets/rubik's_cube/usd/rubik's_cube.usd",
            urdf_path=f"{data_dir}/demo_assets/rubik's_cube/result/rubik's_cube.urdf",
            mjcf_path=f"{data_dir}/demo_assets/rubik's_cube/mjcf/rubik's_cube.xml",
            file_type={**RobotCfg.file_type, "isaacgym": "mjcf", "genesis": "mjcf"},
        ),
        RigidObjCfg(
            name="vase",
            scale=(1, 1, 1),
            physics=PhysicStateType.RIGIDBODY,
            usd_path=f"{data_dir}/demo_assets/vase/usd/vase.usd",
            urdf_path=f"{data_dir}/demo_assets/vase/result/vase.urdf",
            mjcf_path=f"{data_dir}/demo_assets/vase/mjcf/vase.xml",
            file_type={**RobotCfg.file_type, "isaacgym": "mjcf", "genesis": "mjcf"},
        ),
    ]
    # set initial states
    z_offset = 0.4  # Increase offset to see objects drop
    init_states = [
        {
            "objects": {
                "table": {
                    "pos": torch.tensor([0.0, 0.0, 0.4]),
                    "rot": torch.tensor([1, 0, 0, 0]),
                },
                "banana": {
                    "pos": torch.tensor([-0.12, -0.38, 0.825 + z_offset]),
                    "rot": torch.tensor([1, 0, 0, 0]),
                },
                "book": {
                    "pos": torch.tensor([-0.1, -0.08, 0.82 + z_offset]),
                    "rot": torch.tensor([1, 0, 0, 0]),
                },
                "lamp": {
                    "pos": torch.tensor([0.28, 0.30, 1.05 + z_offset]),
                    "rot": torch.tensor([1, 0, 0, 0]),
                },
                "mug": {
                    "pos": torch.tensor([0.28, -0.14, 0.863 + z_offset]),
                    "rot": torch.tensor([1, 0, 0, 0]),
                },
                "remote_control": {
                    "pos": torch.tensor([0.28, -0.34, 0.811 + z_offset]),
                    "rot": torch.tensor([1, 0, 0, 0]),
                },
                "rubiks_cube": {
                    "pos": torch.tensor([0.08, -0.34, 0.83 + z_offset]),
                    "rot": torch.tensor([1, 0, 0, 0]),
                },
                "vase": {
                    "pos": torch.tensor([-0.1, 0.25, 0.95 + z_offset]),
                    "rot": torch.tensor([1, 0, 0, 0]),
                },
            },
            "robots": {
                "franka": {
                    "pos": torch.tensor([0.4, -0.6, 0.78]),
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
    env.set_states(init_states * scenario.num_envs)
    obs = env.get_states(mode="dict")[0]  # get states as a dictionary

    os.makedirs("get_started/output", exist_ok=True)

    # save rgb image
    save_path = f"get_started/output/15_gs_background_{args.sim}.jpg"
    log.info(f"Saving image to {save_path}")
    rgb = obs["cameras"]["camera_0"]["rgb"]
    if isinstance(rgb, torch.Tensor):
        rgb = rgb.detach().cpu().numpy()
    rgb = rgb.squeeze()
    rgb = rgb[:, :, :3]
    cv2.imwrite(save_path, rgb[:, :, ::-1].copy(), [int(cv2.IMWRITE_JPEG_QUALITY), 95])  # RGB -> BGR

    # save depth image
    save_path = f"get_started/output/15_gs_background_{args.sim}_depth.jpg"
    log.info(f"Saving depth image to {save_path}")
    depth = obs["cameras"]["camera_0"]["depth"]
    if isinstance(depth, torch.Tensor):
        depth = depth.detach().cpu().numpy()
    depth = depth.squeeze()
    depth_color = depth_to_colormap(depth, inv_depth=True, depth_range=(1.0, 5.0))
    cv2.imwrite(save_path, depth_color[:, :, ::-1].copy(), [int(cv2.IMWRITE_JPEG_QUALITY), 95])  # RGB -> BGR

    # Video with rotating camera
    obs_saver = ObsSaver(video_path=f"get_started/output/15_gs_background_360_{args.sim}.mp4")

    total_step = total_num_views
    robot = scenario.robots[0] if scenario.robots else None

    for idx in tqdm(range(total_step)):
        # Update camera pose dynamically
        current_view_idx = idx
        angle = 0
        if args.circle_around:
            angle = 2 * np.pi * current_view_idx / total_num_views

        pos_x = center[0] + radius * np.cos(angle)
        pos_y = center[1] + radius * np.sin(angle)

        # Update the single camera's position
        if hasattr(env, "cameras") and len(env.cameras) > 0:
            env.cameras[0].pos = (pos_x, pos_y, height)
            # Look at remains constant

        # Force update camera pose in the simulator
        if hasattr(env, "_update_camera_pose"):
            env._update_camera_pose()

        # Optional: Add random robot actions if robot exists
        if robot is not None:
            # Retrieve default pose from init_states for smoother motion
            default_dof_pos = init_states[0]["robots"]["franka"]["dof_pos"]

            actions = [
                {
                    robot.name: {
                        "dof_pos_target": {
                            joint_name: (
                                max(
                                    robot.joint_limits[joint_name][0],
                                    min(
                                        robot.joint_limits[joint_name][1],
                                        default_dof_pos.get(joint_name, 0.0)
                                        + (torch.rand(1).item() * 2 - 1) * 0.1,  # +/- 0.1 rad range
                                    ),
                                )
                            )
                            for joint_name in robot.joint_limits.keys()
                        }
                    }
                }
                for _ in range(scenario.num_envs)
            ]
            env.set_dof_targets(actions)

        # Simulate and get observations
        env.simulate()
        obs = env.get_states(mode="tensor")

        # Fix RGB tensor dimensions if needed (handle 5D -> 4D conversion)
        # ObsSaver expects (num_envs, H, W, C) but some simulators return (num_envs, 1, H, W, C)

        current_cam_name = "camera_0"
        if current_cam_name in obs.cameras:
            cam_state = obs.cameras[current_cam_name]
            if cam_state.rgb is not None and cam_state.rgb.dim() == 5:
                # Shape: (num_envs, 1, H, W, C) -> squeeze to (num_envs, H, W, C)
                cam_state = CameraState(rgb=cam_state.rgb.squeeze(1), depth=cam_state.depth)

            # Replace cameras dict with only the current camera to create a rotating effect
            obs.cameras = {"orbit_camera": cam_state}

            # Visualize depth and append to RGB image
            if cam_state.depth is not None:
                depth_vis = depth_to_colormap(
                    cam_state.depth.cpu().numpy(), inv_depth=True, depth_range=(1.0, 5.0)
                )  # (H, W, 3) RGB

                depth_tensor = torch.from_numpy(depth_vis.copy()).float()  # (H, W, 3)
                if cam_state.rgb is not None:
                    # Resize depth to match rgb if needed (should be same)
                    # Concatenate vertically: RGB above, Depth below
                    # RGB shape: (num_envs, H, W, C) or (H, W, C)
                    rgb_tensor = cam_state.rgb
                    if rgb_tensor.dim() == 4:
                        rgb_tensor = rgb_tensor.squeeze(0)  # (H, W, C)

                    # Handle RGBA case by dropping alpha channel if present
                    if rgb_tensor.shape[-1] == 4:
                        rgb_tensor = rgb_tensor[..., :3]

                    # Ensure depth_tensor is on same device
                    depth_tensor = depth_tensor.to(rgb_tensor.device)

                    combined_img = torch.cat([rgb_tensor, depth_tensor], dim=0)  # Vertical concat (2H, W, C)

                    # Update obs with combined image
                    # We need to wrap it back to (1, H, W, C) if that was the original shape
                    if cam_state.rgb.dim() == 4:
                        combined_img = combined_img.unsqueeze(0)

                    # Create new camera state
                    new_cam_state = CameraState(rgb=combined_img, depth=cam_state.depth)
                    obs.cameras = {"orbit_camera": new_cam_state}

        obs_saver.add(obs)

    obs_saver.save(fps=15)  # Set video FPS to 15
    log.info(f"Video saved to get_started/output/15_gs_background_{args.sim}.mp4")

    if hasattr(env, "simulation_app"):
        env.close()
        env.simulation_app.close()
