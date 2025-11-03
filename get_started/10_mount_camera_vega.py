"""This script is used to test mounting camera on Vega robot's head (left eye position)."""

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
from metasim.randomization import SceneRandomizer
from metasim.randomization.presets.scene_presets import ScenePresets
from metasim.randomization.scene_randomizer import SceneMaterialPoolCfg
from metasim.scenario.cameras import PinholeCameraCfg
from metasim.scenario.lights import DiskLightCfg, SphereLightCfg
from metasim.scenario.objects import PrimitiveCubeCfg, PrimitiveSphereCfg, RigidObjCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.utils import configclass
from metasim.utils.setup_util import get_handler
from huggingface_hub import snapshot_download
import imageio.v2 as iio
import numpy as np
from numpy.typing import NDArray
from metasim.utils.state import TensorState

if __name__ == "__main__":

    @configclass
    class Args:
        """Arguments for mounting camera on Vega robot."""

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
        ] = "isaacsim"

        ## Others
        num_envs: int = 1
        headless: bool = True

        def __post_init__(self):
            """Post-initialization configuration."""
            log.info(f"Args: {self}")

    args = tyro.cli(Args)

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
        simulator=args.sim,
        headless=args.headless,
        num_envs=args.num_envs,
    )

    from scipy.spatial.transform import Rotation as R


    quat_xyzw = R.from_euler("xyz", [0, 0, 0], degrees=True).as_quat()
    quat = (quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2])  # convert to wxyz
    # Use torso_l3 as mount_link since fixed joints may be merged in USD
    translation_from_torso_l3 = (0.01742, 0.0302, 0.50528)  # Calculated offset from torso_l3 to head_l3 + original offset

    # add cameras
    scenario.cameras = [
        PinholeCameraCfg(
            name="camera_third_person",
            width=1920,
            height=1080,
            pos=(3, -3, 2),
            look_at=(0.0, 0.0, 0.0),
        ),
        PinholeCameraCfg(
            name="camera_first_person",
            width=1920,
            height=1080,
            pos=(1.5, -1.5, 1.5),
            look_at=(0.0, 0.0, 0.0),
            mount_to=args.robot,
            mount_link="torso_l3",  # Use torso_l3 since fixed joints may be merged in USD
            mount_pos=translation_from_torso_l3,
            mount_quat=quat,
        ),
    ]

    # add lights - indoor lighting for 7m x 7m x 5m room
    # 1 central DiskLight + 4 corner SphereLight
    scenario.lights = [
        DiskLightCfg(
            name="ceiling_main",
            intensity=20000.0,
            color=(1.0, 1.0, 1.0),
            radius=1.2,
            pos=(0.0, 0.0, 4.5),
            rot=(0.7071, 0.0, 0.0, 0.7071),  # Point downward
        ),
        SphereLightCfg(
            name="ceiling_ne",
            intensity=7000.0,
            color=(1.0, 1.0, 1.0),
            radius=0.6,
            pos=(2.5, 2.5, 4.0),
        ),
        SphereLightCfg(
            name="ceiling_nw",
            intensity=7000.0,
            color=(1.0, 1.0, 1.0),
            radius=0.6,
            pos=(-2.5, 2.5, 4.0),
        ),
        SphereLightCfg(
            name="ceiling_sw",
            intensity=7000.0,
            color=(1.0, 1.0, 1.0),
            radius=0.6,
            pos=(-2.5, -2.5, 4.0),
        ),
        SphereLightCfg(
            name="ceiling_se",
            intensity=7000.0,
            color=(1.0, 1.0, 1.0),
            radius=0.6,
            pos=(2.5, -2.5, 4.0),
        ),
    ]

    # add objects - real assets from EmbodiedGen
    scenario.objects = [
        RigidObjCfg(
            name="table",
            scale=(1, 1, 1),
            physics=PhysicStateType.RIGIDBODY,
            enabled_gravity=False,
            fix_base_link=True,
            usd_path=f"{data_dir}/demo_assets/table/usd/table.usd",
            urdf_path=f"{data_dir}/demo_assets/table/result/table.urdf",
            mjcf_path=f"{data_dir}/demo_assets/table/mjcf/table.xml",
        ),
        RigidObjCfg(
            name="banana",
            scale=(1, 1, 1),
            enabled_gravity=False,
            physics=PhysicStateType.RIGIDBODY,
            usd_path=f"{data_dir}/demo_assets/banana/usd/banana.usd",
            urdf_path=f"{data_dir}/demo_assets/banana/result/banana.urdf",
            mjcf_path=f"{data_dir}/demo_assets/banana/mjcf/banana.xml",
        ),
        RigidObjCfg(
            name="book",
            scale=(1, 1, 1),
            physics=PhysicStateType.RIGIDBODY,
            enabled_gravity=False,
            usd_path=f"{data_dir}/demo_assets/book/usd/book.usd",
            urdf_path=f"{data_dir}/demo_assets/book/result/book.urdf",
            mjcf_path=f"{data_dir}/demo_assets/book/mjcf/book.xml",
        ),
        RigidObjCfg(
            name="lamp",
            scale=(1, 1, 1),
            physics=PhysicStateType.RIGIDBODY,
            enabled_gravity=False,
            usd_path=f"{data_dir}/demo_assets/lamp/usd/lamp.usd",
            urdf_path=f"{data_dir}/demo_assets/lamp/result/lamp.urdf",
            mjcf_path=f"{data_dir}/demo_assets/lamp/mjcf/lamp.xml",
        ),
        RigidObjCfg(
            name="mug",
            scale=(1, 1, 1),
            physics=PhysicStateType.RIGIDBODY,
            enabled_gravity=False,
            usd_path=f"{data_dir}/demo_assets/mug/usd/mug.usd",
            urdf_path=f"{data_dir}/demo_assets/mug/result/mug.urdf",
            mjcf_path=f"{data_dir}/demo_assets/mug/mjcf/mug.xml",
        ),
        RigidObjCfg(
            name="remote_control",
            scale=(1, 1, 1),
            physics=PhysicStateType.RIGIDBODY,
            enabled_gravity=False,
            usd_path=f"{data_dir}/demo_assets/remote_control/usd/remote_control.usd",
            urdf_path=f"{data_dir}/demo_assets/remote_control/result/remote_control.urdf",
            mjcf_path=f"{data_dir}/demo_assets/remote_control/mjcf/remote_control.xml",
        ),
        RigidObjCfg(
            name="rubiks_cube",
            scale=(1, 1, 1),
            physics=PhysicStateType.RIGIDBODY,
            enabled_gravity=False,
            usd_path=f"{data_dir}/demo_assets/rubik's_cube/usd/rubik's_cube.usd",
            urdf_path=f"{data_dir}/demo_assets/rubik's_cube/result/rubik's_cube.urdf",
            mjcf_path=f"{data_dir}/demo_assets/rubik's_cube/mjcf/rubik's_cube.xml",
        ),
        RigidObjCfg(
            name="vase",
            scale=(1, 1, 1),
            physics=PhysicStateType.RIGIDBODY,
            enabled_gravity=False,
            usd_path=f"{data_dir}/demo_assets/vase/usd/vase.usd",
            urdf_path=f"{data_dir}/demo_assets/vase/result/vase.urdf",
            mjcf_path=f"{data_dir}/demo_assets/vase/mjcf/vase.xml",
        ),
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

    # Setup room scene (fixed, no randomization)
    log.info("Setting up empty room scene...")
    scene_cfg = ScenePresets.empty_room(
        room_size=8.0,
        wall_height=5.0,
        wall_thickness=0.1,
    )
    # Use fixed materials (no randomization)
    # Wood floor
    scene_cfg.floor_materials = SceneMaterialPoolCfg(
        material_paths=["roboverse_data/materials/arnold/Wood/Oak_Planks.mdl"],
        selection_strategy="sequential",
    )
    # Stone walls
    scene_cfg.wall_materials = SceneMaterialPoolCfg(
        material_paths=["roboverse_data/materials/arnold/Masonry/Brick_Wall_Brown.mdl"],
        selection_strategy="sequential",
    )
    scene_cfg.ceiling_materials = SceneMaterialPoolCfg(
        material_paths=["roboverse_data/materials/arnold/Architecture/Ceiling_Tiles.mdl"],
        selection_strategy="sequential",
    )
    
    scene_rand = SceneRandomizer(scene_cfg, seed=42)
    scene_rand.bind_handler(handler)
    scene_rand()  # Apply scene once
    log.info("  Room: 4m x 4m x 3m (enclosed, empty)")

    # Set initial states - all objects translated to x=0.6 area
    z_offset = 0.0
    x_offset = 1.0  # Shift all objects to x=0.6 area (original was around x=0.4)
    init_states = [
        {
            "objects": {
                "table": {
                    "pos": torch.tensor([0.4 + x_offset, -0.2, 0.4]),
                    "rot": torch.tensor([1, 0, 0, 0]),
                },
                "banana": {
                    "pos": torch.tensor([0.28 + x_offset, -0.58, 0.825 + z_offset]),
                    "rot": torch.tensor([1, 0, 0, 0]),
                },
                "book": {
                    "pos": torch.tensor([0.3 + x_offset, -0.28, 0.82 + z_offset]),
                    "rot": torch.tensor([1, 0, 0, 0]),
                },
                "lamp": {
                    "pos": torch.tensor([0.68 + x_offset, 0.10, 1.05 + z_offset]),
                    "rot": torch.tensor([1, 0, 0, 0]),
                },
                "mug": {
                    "pos": torch.tensor([0.68 + x_offset, -0.34, 0.863 + z_offset]),
                    "rot": torch.tensor([1, 0, 0, 0]),
                },
                "remote_control": {
                    "pos": torch.tensor([0.68 + x_offset, -0.54, 0.811 + z_offset]),
                    "rot": torch.tensor([1, 0, 0, 0]),
                },
                "rubiks_cube": {
                    "pos": torch.tensor([0.48 + x_offset, -0.54, 0.83 + z_offset]),
                    "rot": torch.tensor([1, 0, 0, 0]),
                },
                "vase": {
                    "pos": torch.tensor([0.30 + x_offset, 0.05, 0.95 + z_offset]),
                    "rot": torch.tensor([1, 0, 0, 0]),
                },
                "cube": {
                    "pos": torch.tensor([0.3, -0.2, 0.05]),
                    "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                },
                "sphere": {
                    "pos": torch.tensor([0.3 , 0.2, 0.05]),
                    "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                },
                "bbq_sauce": {
                    "pos": torch.tensor([0.2 , -0.3, 0.14]),
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
                        # Head - forward looking
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

    # Custom ObsSaver class for single camera
    class SingleCameraObsSaver:
        """Save observations from a single camera to video."""
        
        def __init__(self, video_path: str, camera_name: str):
            """Initialize the SingleCameraObsSaver."""
            self.video_path = video_path
            self.camera_name = camera_name
            self.images: list[NDArray] = []
            
        def add(self, state: TensorState):
            """Add the observation from specified camera to the list."""
            if self.video_path is None:
                return
                
            try:
                if self.camera_name not in state.cameras:
                    log.warning(f"Camera {self.camera_name} not found in state")
                    return
                    
                rgb_data = state.cameras[self.camera_name].rgb  # (N, H, W, 3)
                # Take first environment if multiple environments
                if len(rgb_data.shape) == 4:
                    rgb_data = rgb_data[0]  # (H, W, 3)
                
                # Convert to numpy array
                if isinstance(rgb_data, torch.Tensor):
                    image = rgb_data.cpu().numpy()
                else:
                    image = np.array(rgb_data)
                
                # Normalize to [0, 255] if needed
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = image.astype(np.uint8)
                    
                self.images.append(image)
            except Exception as e:
                log.error(f"Error adding observation from camera {self.camera_name}: {e}")
                
        def save(self):
            """Save the video."""
            if self.video_path is not None and self.images:
                log.info(f"Saving video of {len(self.images)} frames from {self.camera_name} to {self.video_path}")
                os.makedirs(os.path.dirname(self.video_path), exist_ok=True)
                iio.mimsave(self.video_path, self.images, fps=30)

    obs = handler.get_states(mode="tensor")
    ## Main loop
    # Create two separate ObsSaver instances for each camera
    obs_saver_third = SingleCameraObsSaver(
        video_path=f"get_started/output/10_mount_camera_vega_third_person_{args.sim}.mp4",
        camera_name="camera_third_person"
    )
    obs_saver_first = SingleCameraObsSaver(
        video_path=f"get_started/output/10_mount_camera_vega_first_person_{args.sim}.mp4",
        camera_name="camera_first_person"
    )
    obs_saver_third.add(obs)
    obs_saver_first.add(obs)

    step = 0
    robot = scenario.robots[0]
    for _ in range(100):
        log.debug(f"Step {step}")
        actions = [
            {
                robot.name: {
                    "dof_pos_target": {
                        joint_name: (
                            torch.rand(1).item() *0.8
                            * (robot.joint_limits[joint_name][1] - robot.joint_limits[joint_name][0])
                            + robot.joint_limits[joint_name][0] * 0.8
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
        obs_saver_third.add(obs)
        obs_saver_first.add(obs)
        step += 1

    obs_saver_third.save()
    obs_saver_first.save()

