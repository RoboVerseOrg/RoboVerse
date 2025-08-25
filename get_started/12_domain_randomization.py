"""Domain Randomization Example for MetaSim."""

from __future__ import annotations

import rootutils

rootutils.setup_root(__file__, pythonpath=True)

from typing import Literal

import torch
import tyro
from loguru import logger as log
from rich.logging import RichHandler

from metasim.constants import PhysicStateType, SimType
from metasim.scenario.objects import ArticulationObjCfg, PrimitiveCubeCfg, PrimitiveSphereCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.utils import configclass
from metasim.utils.setup_util import get_sim_handler_class
from roboverse_pack.randomization.friction_randomizer import FrictionRandomCfg, FrictionRandomizer
from roboverse_pack.randomization.mass_randomizer import MassRandomCfg, MassRandomizer

log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])


def run_domain_randomization(args):
    """Demonstrate domain randomization with specified simulator."""
    log.info(f"=== {args.simulator.upper()} Domain Randomization Demo ===")

    # Create scenario and update simulator
    scenario = ScenarioCfg(
        robots=["franka"],
        num_envs=args.num_envs,  # Multiple environments for parallel testing
        simulator=args.simulator,  # Will be overridden
        headless=args.headless,  # Will be overridden
    )

    # Add objects (same as 0_static_scene.py)
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
        ArticulationObjCfg(
            name="box_base",
            fix_base_link=True,
            usd_path="metasim/example/example_assets/box_base/usd/box_base.usd",
            urdf_path="metasim/example/example_assets/box_base/urdf/box_base_unique.urdf",
            mjcf_path="metasim/example/example_assets/box_base/mjcf/box_base_unique.mjcf",
        ),
    ]

    # Get handler class based on simulator
    env_class = get_sim_handler_class(SimType(args.simulator))
    env = env_class(scenario)

    env.launch()
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
    ] * scenario.num_envs

    env.set_states(init_states)

    # initialize randomizers
    cube_mass_config = MassRandomCfg(
        obj_name="cube",
        range=(0.3, 0.7),
        operation="abs",
        distribution="uniform",
    )
    cube_mass_randomizer = MassRandomizer(cube_mass_config)
    cube_mass_randomizer.bind_handler(env)

    franka_friction_config = FrictionRandomCfg(
        obj_name="franka",
        range=(0.5, 1.5),
        operation="add",
        distribution="gaussian",
    )
    franka_friction_randomizer = FrictionRandomizer(franka_friction_config)
    franka_friction_randomizer.bind_handler(env)

    franka_mass_config = MassRandomCfg(
        obj_name="franka",
        range=(0.2, 0.4),
        operation="abs",
        distribution="log_uniform",
    )
    franka_mass_randomizer = MassRandomizer(franka_mass_config)
    franka_mass_randomizer.bind_handler(env)

    # Get cube mass using randomizer
    cube_mass = cube_mass_randomizer.get_body_mass("cube")
    robot_friction = franka_friction_randomizer.get_body_friction("franka")
    robot_mass = franka_mass_randomizer.get_body_mass("franka")

    # Store initial values for comparison
    initial_values = {
        "cube_mass": cube_mass.clone(),
        "franka_mass": robot_mass.clone(),
        "franka_friction": robot_friction.clone(),
    }

    # run randomization for cube mass
    cube_mass_randomizer()
    randomized_cube_mass = cube_mass_randomizer.get_body_mass("cube")
    log.info("================================================")
    log.info("randomizing cube mass by uniform distribution")
    log.info(f"  Before: {initial_values['cube_mass'].cpu().numpy().round(3)} kg")
    log.info(f"  After:  {randomized_cube_mass.cpu().numpy().round(3)} kg")

    # run randomization for franka friction
    franka_friction_randomizer()
    randomized_robot_friction = franka_friction_randomizer.get_body_friction("franka")
    log.info("================================================")
    log.info("randomizing franka friction by gaussian distribution")
    log.info(f"  Before: {initial_values['franka_friction'][0].cpu().numpy().round(3)}")
    log.info(f"  After:  {randomized_robot_friction[0].cpu().numpy().round(3)}")

    # run randomization for franka mass
    franka_mass_randomizer()
    randomized_sphere_mass = franka_mass_randomizer.get_body_mass("franka")
    log.info("================================================")
    log.info("randomizing franka mass by log_uniform distribution")
    log.info(f"  Before: {initial_values['franka_mass'].cpu().numpy().round(3)} kg")
    log.info(f"  After:  {randomized_sphere_mass.cpu().numpy().round(3)} kg")

    # Run simulation for a few steps
    for _ in range(50):
        env.simulate()

    env.close()


def main():
    @configclass
    class Args:
        """Arguments for the static scene."""

        robot: str = "franka"

        ## Handlers
        simulator: Literal["isaacsim"] = "isaacsim"

        ## Others
        num_envs: int = 1
        headless: bool = False

        def __post_init__(self):
            """Post-initialization configuration."""
            log.info(f"Args: {self}")

    args = tyro.cli(Args)
    """Main function to run the domain randomization demo."""
    log.info("Starting Domain Randomization Demo")
    # Run IsaacSim demo
    run_domain_randomization(args)
    log.info("\nRandomization demo completed. Check the logs above for detailed results.")


if __name__ == "__main__":
    main()
