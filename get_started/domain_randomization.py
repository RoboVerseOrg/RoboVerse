"""Domain Randomization Demo for MetaSim."""

from __future__ import annotations
from metasim.utils import configclass
import tyro
from typing import Literal

import torch
from loguru import logger as log
from metasim.constants import PhysicStateType, SimType
from metasim.scenario.cameras import PinholeCameraCfg
from metasim.scenario.objects import ArticulationObjCfg, PrimitiveCubeCfg, PrimitiveSphereCfg, RigidObjCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.utils.setup_util import get_sim_handler_class
from rich.logging import RichHandler

log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])


def randomize_body_mass(
    handler,
    obj_name: str,
    mass_range: tuple[float, float],
    body_name: str | None = None,
    env_ids: list[int] | None = None,
    operation: str = "scale",
    distribution: str = "uniform",
    device: str = "cpu",
) -> None:
    """External helper: randomize mass using handler's get/set APIs."""
    if env_ids is None:
        env_ids = list(range(handler.num_envs))

    current_mass = handler.get_body_mass(obj_name, body_name, env_ids)

    if distribution == "uniform":
        rand_values = torch.rand(len(env_ids), device=device) * (mass_range[1] - mass_range[0]) + mass_range[0]
    elif distribution == "log_uniform":
        log_min, log_max = torch.log(torch.tensor(mass_range[0])), torch.log(torch.tensor(mass_range[1]))
        rand_values = torch.exp(torch.rand(len(env_ids), device=device) * (log_max - log_min) + log_min)
    elif distribution == "gaussian":
        mean = (mass_range[0] + mass_range[1]) / 2
        std = (mass_range[1] - mass_range[0]) / 6
        rand_values = torch.normal(mean, std, size=(len(env_ids),), device=device)
        rand_values = torch.clamp(rand_values, mass_range[0], mass_range[1])
    else:
        raise ValueError(f"Unsupported distribution: {distribution}")

    if operation == "add":
        new_mass = current_mass + (
            rand_values.unsqueeze(-1) if body_name is None and current_mass.ndim == 2 else rand_values
        )
    elif operation == "scale":
        new_mass = current_mass * (
            rand_values.unsqueeze(-1) if body_name is None and current_mass.ndim == 2 else rand_values
        )
    elif operation == "abs":
        new_mass = rand_values.unsqueeze(-1) if body_name is None and current_mass.ndim == 2 else rand_values
    else:
        raise ValueError(f"Unsupported operation: {operation}")

    handler.set_body_mass(obj_name, new_mass, body_name, env_ids)


def randomize_body_friction(
    handler,
    obj_name: str,
    friction_range: tuple[float, float],
    body_name: str | None = None,
    env_ids: list[int] | None = None,
    operation: str = "scale",
    distribution: str = "uniform",
    device: str = "cpu",
) -> None:
    """External helper: randomize friction using handler's get/set APIs."""
    if env_ids is None:
        env_ids = list(range(handler.num_envs))

    current_friction = handler.get_body_friction(obj_name, body_name, env_ids)

    if distribution == "uniform":
        rand_values = (
            torch.rand(len(env_ids), device=device) * (friction_range[1] - friction_range[0]) + friction_range[0]
        )
    elif distribution == "log_uniform":
        log_min, log_max = torch.log(torch.tensor(friction_range[0])), torch.log(torch.tensor(friction_range[1]))
        rand_values = torch.exp(torch.rand(len(env_ids), device=device) * (log_max - log_min) + log_min)
    elif distribution == "gaussian":
        mean = (friction_range[0] + friction_range[1]) / 2
        std = (friction_range[1] - friction_range[0]) / 6
        rand_values = torch.normal(mean, std, size=(len(env_ids),), device=device)
        rand_values = torch.clamp(rand_values, friction_range[0], friction_range[1])
    else:
        raise ValueError(f"Unsupported distribution: {distribution}")

    if operation == "add":
        new_friction = current_friction + (
            rand_values.unsqueeze(-1) if body_name is None and current_friction.ndim == 2 else rand_values
        )
    elif operation == "scale":
        new_friction = current_friction * (
            rand_values.unsqueeze(-1) if body_name is None and current_friction.ndim == 2 else rand_values
        )
    elif operation == "abs":
        new_friction = rand_values.unsqueeze(-1) if body_name is None and current_friction.ndim == 2 else rand_values
    else:
        raise ValueError(f"Unsupported operation: {operation}")

    handler.set_body_friction(obj_name, new_friction, body_name, env_ids)


def demo_domain_randomization(args):
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
        RigidObjCfg(
            name="bbq_sauce",
            scale=(2, 2, 2),
            physics=PhysicStateType.RIGIDBODY,
            usd_path="metasim/example/example_assets/bbq_sauce/usd/bbq_sauce.usd",
            urdf_path="metasim/example/example_assets/bbq_sauce/urdf/bbq_sauce.urdf",
            mjcf_path="metasim/example/example_assets/bbq_sauce/mjcf/bbq_sauce.xml",
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

    # Get bbq_sauce mass
    bbq_mass = env.get_body_mass("bbq_sauce")
    log.info(f"BBQ Sauce mass: {bbq_mass.cpu().numpy().round(3)} kg")

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
        "bbq_mass": bbq_mass.clone(),
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

    # Apply domain randomization
    log.info("\n" + "=" * 60)
    log.info("DOMAIN RANDOMIZATION")
    log.info("=" * 60)

    # Randomize cube mass with uniform distribution
    log.info("Randomizing cube mass (uniform, 0.3-0.7 kg)...")
    randomize_body_mass(env, "cube", mass_range=(0.3, 0.7), operation="abs", distribution="uniform")
    randomized_cube_mass = env.get_body_mass("cube")
    log.info(f"  Before: {initial_values['cube_mass'].cpu().numpy().round(3)} kg")
    log.info(f"  After:  {randomized_cube_mass.cpu().numpy().round(3)} kg")

    # Randomize robot friction with log-uniform distribution
    log.info("Randomizing robot friction (log-uniform, 0.5-1.5x scale)...")
    randomize_body_friction(env, "franka", friction_range=(0.5, 1.5), operation="scale", distribution="log_uniform")
    randomized_robot_friction = env.get_body_friction("franka")
    log.info(f"  Before: {initial_values['robot_friction'][0].cpu().numpy().round(3)}")
    log.info(f"  After:  {randomized_robot_friction[0].cpu().numpy().round(3)}")

    # Randomize sphere mass with gaussian distribution
    log.info("Randomizing sphere mass (gaussian, 0.2-0.4 kg)...")
    randomize_body_mass(env, "sphere", mass_range=(0.2, 0.4), operation="abs", distribution="gaussian")
    randomized_sphere_mass = env.get_body_mass("sphere")
    log.info(f"  Before: {initial_values['sphere_mass'].cpu().numpy().round(3)} kg")
    log.info(f"  After:  {randomized_sphere_mass.cpu().numpy().round(3)} kg")

    # Randomize bbq_sauce mass with scale operation
    log.info("Randomizing BBQ sauce mass (uniform, 0.8-1.2x scale)...")
    randomize_body_mass(env, "bbq_sauce", mass_range=(0.8, 1.2), operation="scale", distribution="uniform")
    randomized_bbq_mass = env.get_body_mass("bbq_sauce")
    log.info(f"  Before: {initial_values['bbq_mass'].cpu().numpy().round(3)} kg")
    log.info(f"  After:  {randomized_bbq_mass.cpu().numpy().round(3)} kg")

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
    bbq_before_str = f"{initial_values['bbq_mass'].cpu().numpy().round(3)}"
    bbq_after_str = f"{randomized_bbq_mass.cpu().numpy().round(3)}"
    robot_before_str = f"{initial_values['robot_friction'][0].cpu().numpy().round(3)}"
    robot_after_str = f"{randomized_robot_friction[0].cpu().numpy().round(3)}"

    log.info(f"{'Cube':<15} {'Mass':<10} {cube_before_str:<20} {cube_after_str:<20} {'Uniform':<10}")
    log.info(f"{'Sphere':<15} {'Mass':<10} {sphere_before_str:<20} {sphere_after_str:<20} {'Gaussian':<10}")
    log.info(f"{'BBQ Sauce':<15} {'Mass':<10} {bbq_before_str:<20} {bbq_after_str:<20} {'Scale':<10}")
    log.info(f"{'Robot':<15} {'Friction':<10} {robot_before_str:<20} {robot_after_str:<20} {'Log-Uniform':<10}")

    # Run simulation for a few steps to see the effects
    for step in range(50):
        env.simulate()
        if step % 10 == 0:
            log.info(f"  Step {step:2d}/50")

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
    demo_domain_randomization(args)
    log.info("\nDemo completed! Check the logs above for detailed results.")


if __name__ == "__main__":
    main()
