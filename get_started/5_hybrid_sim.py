"""Run a hybrid simulation with separate physics and rendering backends."""

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
from metasim.sim import HybridSimHandler
from metasim.utils import configclass
from metasim.utils.obs_utils import ObsSaver
from metasim.utils.setup_util import get_handler

if __name__ == "__main__":

    @configclass
    class Args:
        """Arguments for the hybrid-sim demo."""

        robot: str = "franka"

        ## Handlers
        sim: Literal["isaacsim", "isaacgym", "genesis", "pybullet", "sapien2", "sapien3", "mujoco", "superdex"] = "mujoco"
        renderer: (
            Literal[
                "isaacsim",
                "isaacgym",
                "genesis",
                "pybullet",
                "sapien2",
                "sapien3",
                "mujoco",
                "superdex",
                "blender",
            ]
            | None
        ) = "isaacsim"

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
    # NB: camera deliberately off the (1,-1,1) body-diagonal of the cube so the
    # default-oriented PrimitiveCubeCfg presents a 3-face cube silhouette
    # rather than a hexagonal corner-on view.
    scenario.cameras = [PinholeCameraCfg(width=1024, height=1024, pos=(1.2, -1.6, 1.4), look_at=(0.0, 0.0, 0.0))]

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
        ArticulationObjCfg(
            name="box_base",
            fix_base_link=True,
            usd_path="roboverse_data/assets/rlbench/close_box/box_base/usd/box_base.usd",
            urdf_path="roboverse_data/assets/rlbench/close_box/box_base/urdf/box_base_unique.urdf",
            mjcf_path="roboverse_data/assets/rlbench/close_box/box_base/mjcf/box_base_unique.mjcf",
        ),
    ]

    # Align Blender visual loading with mujoco's MJCF kinematics so link
    # frames match exactly (avoids the USD-baked-rotation gripper-flip class
    # of bug when physics ≠ Blender). Also gives us MJCF-driven textures.
    # Also enable photoreal HDRI lighting by default — RoboVerse bundles a
    # 4-image sample set and asset_paths.get_hdri_dir picks the first that
    # exists locally (bundled, third_party, or downloaded).
    if args.renderer == "blender":
        for robot_cfg in scenario.robots:
            robot_cfg.file_type = {**robot_cfg.file_type, "blender": "mjcf"}
        for obj_cfg in scenario.objects:
            if hasattr(obj_cfg, "file_type") and getattr(obj_cfg, "mjcf_path", None):
                obj_cfg.file_type = {**obj_cfg.file_type, "blender": "mjcf"}
        from roboverse_pack.blender.asset_paths import get_hdri_dir as _get_hdri_dir

        hdri_dir = _get_hdri_dir(download=False)
        if hdri_dir is not None:
            scenario.render.hdri_path = str(hdri_dir)

    if args.renderer is None:
        log.info(f"Using simulator: {args.sim}")
        handler = get_handler(scenario)
    else:
        log.info(f"Using simulator: {args.sim}, render: {args.renderer}")
        handler_physics = get_handler(scenario)
        scenario.update(simulator=args.renderer)
        handler_renderer = get_handler(scenario)
        handler = HybridSimHandler(scenario, handler_physics, handler_renderer)

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
    handler.set_states(init_states * scenario.num_envs)
    obs = handler.get_states(mode="tensor")
    os.makedirs("get_started/output", exist_ok=True)

    ## Main loop
    obs_saver = ObsSaver(video_path=f"get_started/output/5_hybrid_sim_{args.sim}_render_{args.renderer}.mp4")
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

    # close handler for stability
    handler.close()
