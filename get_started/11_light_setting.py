"""This script is used to test the static scene."""

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

import imageio

from metasim.constants import PhysicStateType, SimType
from metasim.utils import configclass
from metasim.utils.setup_util import get_sim_handler_class
from scenario_cfg.cameras import PinholeCameraCfg
from scenario_cfg.lights import DiskLightCfg, DistantLightCfg, DomeLightCfg
from scenario_cfg.objects import ArticulationObjCfg, PrimitiveCubeCfg, PrimitiveSphereCfg, RigidObjCfg
from scenario_cfg.render import RenderCfg
from scenario_cfg.scenario import ScenarioCfg

if __name__ == "__main__":

    @configclass
    class Args:
        """Arguments for the static scene."""

        robot: str = "franka"

        ## Handlers
        sim: Literal["isaacsim"] = "isaacsim"

        ## Others
        num_envs: int = 1
        headless: bool = False
        designed_lighting: bool = False
        lighting_preset: str = "natural"  # natural, studio, outdoor, indoor

        def __post_init__(self):
            """Post-initialization configuration."""
            log.info(f"Args: {self}")

    args = tyro.cli(Args)

    # initialize scenario
    scenario = ScenarioCfg(robots=[args.robot], headless=args.headless, num_envs=args.num_envs, simulator=args.sim)

    # Configure lighting based on preset
    if args.designed_lighting:
        log.info(f"Using designed lighting configuration with preset: {args.lighting_preset}")

        if args.lighting_preset == "natural":
            # Natural outdoor lighting - combines dome light with directional sun
            scenario.lights = [
                # Sky dome light - provides soft ambient lighting from all directions
                DomeLightCfg(
                    intensity=800.0,  # Moderate ambient lighting
                    color=(0.85, 0.9, 1.0),  # Slightly blue sky color
                ),
                # Sun light - main directional light
                DistantLightCfg(
                    intensity=1200.0,  # Strong sunlight
                    polar=35.0,  # Sun at 35° elevation (natural angle)
                    azimuth=60.0,  # From the northeast
                    color=(1.0, 0.98, 0.95),  # Slightly warm sunlight
                ),
                # Soft area light for subtle fill
                DiskLightCfg(
                    intensity=300.0,
                    radius=1.5,  # Large disk for soft light
                    pos=(2.0, -2.0, 4.0),  # Side fill light
                    rot=(0.7071, 0.7071, 0.0, 0.0),  # Angled towards scene
                    color=(0.95, 0.95, 1.0),
                ),
            ]

        elif args.lighting_preset == "studio":
            # Studio – simplified with fewer but stronger lights
            scenario.lights = [
                # Main key light (stronger intensity to compensate for fewer lights)
                DiskLightCfg(
                    intensity=4000.0,  # Higher intensity since we have fewer lights
                    radius=1.8,  # Larger for softer, more even coverage
                    pos=(3.0, 2.0, 3.2),
                    rot=(0.8660, 0.0, 0.5, 0.0),
                    color=(1.0, 0.98, 0.96),
                ),
                # Fill light (increased to balance the key light)
                DiskLightCfg(
                    intensity=1800.0,  # Increased to provide better fill
                    radius=1.8,  # Larger for soft, diffuse fill
                    pos=(-2.2, 1.2, 2.6),
                    rot=(0.7071, 0.0, -0.7071, 0.0),
                    color=(0.96, 0.97, 1.0),
                ),
                # Ambient light (higher to compensate for fewer lights)
                DomeLightCfg(
                    intensity=300.0,  # Increased ambient for overall brightness
                    color=(0.95, 0.95, 0.95),
                ),
            ]
        else:
            log.warning(f"Unknown lighting preset: {args.lighting_preset}, using natural preset")
            args.lighting_preset = "natural"

        if args.sim in ["isaacsim"]:
            scenario.render = RenderCfg(mode="pathtracing")  # Best quality for complex lighting
            log.info("Using pathtracing render mode for optimal lighting quality")

    else:
        log.info("Using default lighting configuration")

    # add cameras
    scenario.cameras = [
        PinholeCameraCfg(
            width=1024,
            height=1024,
            pos=[4.0, 4.0, 3.0],
            look_at=[0.0, 0.0, 1.0],
        )
    ]

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
            usd_path="get_started/example_assets/bbq_sauce/usd/bbq_sauce.usd",
            urdf_path="get_started/example_assets/bbq_sauce/urdf/bbq_sauce.urdf",
            mjcf_path="get_started/example_assets/bbq_sauce/mjcf/bbq_sauce.xml",
        ),
        ArticulationObjCfg(
            name="box_base",
            fix_base_link=True,
            usd_path="get_started/example_assets/box_base/usd/box_base.usd",
            urdf_path="get_started/example_assets/box_base/urdf/box_base_unique.urdf",
            mjcf_path="get_started/example_assets/box_base/mjcf/box_base_unique.mjcf",
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
    env.set_states(init_states)
    # while True:
    #     env.simulate()
    obs = env.get_states(mode="dict")
    os.makedirs("get_started/output", exist_ok=True)
    if args.designed_lighting:
        save_path = f"get_started/output/11_light_setting_{args.sim}_{args.lighting_preset}.png"
    else:
        save_path = f"get_started/output/11_light_setting_{args.sim}_default.png"
    log.info(f"Saving image to {save_path}")
    imageio.imwrite(save_path, next(iter(obs.cameras.values())).rgb[0].cpu().numpy())
