"""Refactored Domain Randomization Example for MetaSim using MassRandomizer."""

from __future__ import annotations
from metasim.utils import configclass
import tyro
from typing import Literal

import torch
from loguru import logger as log
from metasim.constants import PhysicStateType, SimType
from metasim.scenario.objects import ArticulationObjCfg, PrimitiveCubeCfg, PrimitiveSphereCfg, RigidObjCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.utils.setup_util import get_sim_handler_class
from rich.logging import RichHandler
from metasim.sim.queries.mass_randomizer import MassRandomizer
from roboverse_pack.randomization.randomization import MassRandomCfg, RandomizationCfg

log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])


def run_domain_randomization_refactored(args):
    """Demonstrate refactored domain randomization with specified simulator."""
    log.info(f"=== {args.simulator.upper()} Refactored Domain Randomization Demo ===")

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

    # Create mass randomizers for different objects
    cube_randomizer = MassRandomizer(obj_name="cube")
    sphere_randomizer = MassRandomizer(obj_name="sphere")
    robot_randomizer = MassRandomizer(obj_name="franka")

    # Create optional queries dictionary
    optional_queries = {
        "cube_randomizer": cube_randomizer,
        "sphere_randomizer": sphere_randomizer,
        "robot_randomizer": robot_randomizer,
    }

    env = env_class(scenario, optional_queries=optional_queries)

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
    log.info("Initial states set successfully")

    # Get initial mass and friction values
    log.info("\n" + "=" * 60)
    log.info("INITIAL VALUES (Before Randomization)")
    log.info("=" * 60)

    # Get robot body masses
    robot_masses = env.get_body_mass("franka")
    log.info(f"Robot body masses (shape: {robot_masses.shape}):")
    log.info(f"  Values: {robot_masses[0].cpu().numpy().round(3)}")

    # Get cube mass
    cube_mass = env.get_body_mass("cube")
    log.info(f"Cube mass: {cube_mass.cpu().numpy().round(3)} kg")

    # Get sphere mass
    sphere_mass = env.get_body_mass("sphere")
    log.info(f"Sphere mass: {sphere_mass.cpu().numpy().round(3)} kg")

    # Get friction values
    robot_friction = env.get_body_friction("franka")
    log.info(f"Robot friction (shape: {robot_friction.shape}):")
    log.info(f"  Values: {robot_friction[0].cpu().numpy().round(3)}")

    cube_friction = env.get_body_friction("cube")
    log.info(f"Cube friction: {cube_friction.cpu().numpy().round(3)}")

    sphere_friction = env.get_body_friction("sphere")
    log.info(f"Sphere friction: {sphere_friction.cpu().numpy().round(3)}")

    # Store initial values for comparison
    initial_values = {
        "cube_mass": cube_mass.clone(),
        "sphere_mass": sphere_mass.clone(),
        "robot_friction": robot_friction.clone(),
        "cube_friction": cube_friction.clone(),
        "sphere_friction": sphere_friction.clone(),
    }

    # Set specific mass values
    log.info("\n" + "=" * 60)
    log.info("SETTING SPECIFIC VALUES")
    log.info("=" * 60)

    # Set cube mass to 1.0 kg for all environments
    new_cube_mass = torch.ones([scenario.num_envs, 1], device=env.device)
    env.set_body_mass("cube", new_cube_mass)

    # Verify the change
    updated_cube_mass = env.get_body_mass("cube")
    log.info(f"Cube mass set to: {updated_cube_mass.cpu().numpy().round(3)} kg")

    # Set specific friction for robot's first body
    robot_body_names = env._get_body_names("franka")
    if robot_body_names:
        first_body = robot_body_names[0]
        new_friction = torch.tensor([0.8] * scenario.num_envs, device=env.device)
        env.set_body_friction("franka", new_friction, body_name=first_body)
        # Verify the change
        updated_friction = env.get_body_friction("franka", body_name=first_body)
        log.info(f"{first_body} friction set to: {updated_friction.cpu().numpy().round(3)}")

    # Apply domain randomization using the new system
    log.info("\n" + "=" * 60)
    log.info("REFACTORED DOMAIN RANDOMIZATION")
    log.info("=" * 60)

    # Create randomization configurations
    cube_mass_config = MassRandomCfg(
        enabled=True, obj_name="cube", range=(0.3, 0.7), operation="abs", distribution="uniform"
    )

    sphere_mass_config = MassRandomCfg(
        enabled=True, obj_name="sphere", range=(0.2, 0.4), operation="abs", distribution="gaussian"
    )

    robot_friction_config = MassRandomCfg(
        enabled=True, obj_name="franka", range=(0.5, 1.5), operation="scale", distribution="log_uniform"
    )

    # Apply randomization using the randomizers
    log.info("Randomizing cube mass (uniform, 0.3-0.7 kg)...")
    cube_randomizer(cube_mass_config)
    randomized_cube_mass = env.get_body_mass("cube")
    log.info(f"  Before: {initial_values['cube_mass'].cpu().numpy().round(3)} kg")
    log.info(f"  After:  {randomized_cube_mass.cpu().numpy().round(3)} kg")

    log.info("Randomizing sphere mass (gaussian, 0.2-0.4 kg)...")
    sphere_randomizer(sphere_mass_config)
    randomized_sphere_mass = env.get_body_mass("sphere")
    log.info(f"  Before: {initial_values['sphere_mass'].cpu().numpy().round(3)} kg")
    log.info(f"  After:  {randomized_sphere_mass.cpu().numpy().round(3)} kg")

    log.info("Randomizing robot friction (log-uniform, 0.5-1.5x scale)...")
    robot_randomizer(robot_friction_config)
    randomized_robot_friction = env.get_body_friction("franka")
    log.info(f"  Before: {initial_values['robot_friction'][0].cpu().numpy().round(3)}")
    log.info(f"  After:  {randomized_robot_friction[0].cpu().numpy().round(3)}")

    # Summary table
    log.info("\n" + "=" * 80)
    log.info("RANDOMIZATION SUMMARY")
    log.info("=" * 80)
    log.info(f"{'Object':<15} {'Property':<10} {'Before':<20} {'After':<20} {'Change':<10}")
    log.info("-" * 80)

    # Format arrays for display
    cube_before_str = f"{initial_values['cube_mass'].cpu().numpy().round(3)}"
    cube_after_str = f"{randomized_cube_mass.cpu().numpy().round(3)}"
    sphere_before_str = f"{initial_values['sphere_mass'].cpu().numpy().round(3)}"
    sphere_after_str = f"{randomized_sphere_mass.cpu().numpy().round(3)}"
    robot_before_str = f"{initial_values['robot_friction'][0].cpu().numpy().round(3)}"
    robot_after_str = f"{randomized_robot_friction[0].cpu().numpy().round(3)}"

    log.info(f"{'Cube':<15} {'Mass':<10} {cube_before_str:<20} {cube_after_str:<20} {'Uniform':<10}")
    log.info(f"{'Sphere':<15} {'Mass':<10} {sphere_before_str:<20} {sphere_after_str:<20} {'Gaussian':<10}")
    log.info(f"{'Robot':<15} {'Friction':<10} {robot_before_str:<20} {robot_after_str:<20} {'Log-Uniform':<10}")

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
    """Main function to run the refactored domain randomization demo."""
    log.info("Starting Refactored Domain Randomization Demo")
    # Run IsaacSim demo
    run_domain_randomization_refactored(args)
    log.info("\nDemo completed! Check the logs above for detailed results.")


if __name__ == "__main__":
    main()
