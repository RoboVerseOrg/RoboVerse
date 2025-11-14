"""Test object randomizer functionality."""

from __future__ import annotations

import pytest
import rootutils
import torch
from loguru import logger as log

rootutils.setup_root(__file__, pythonpath=True)
from metasim.randomization.object_randomizer import (
    ObjectRandomCfg,
    ObjectRandomizer,
    PhysicsRandomCfg,
    PoseRandomCfg,
)
from metasim.test.randomization.conftest import get_shared_scenario
from metasim.utils.math import euler_xyz_from_quat


def get_object_from_randomizer(randomizer):
    """Helper function to get object instance from randomizer."""
    obj_name = randomizer.cfg.obj_name
    if obj_name in randomizer.handler.scene.articulations:
        return randomizer.handler.scene.articulations[obj_name]
    elif obj_name in randomizer.handler.scene.rigid_objects:
        return randomizer.handler.scene.rigid_objects[obj_name]
    else:
        raise ValueError(f"Object {obj_name} not found in the scene")


def object_physics(handler, distribution="uniform"):
    """Test object physics properties (mass, friction, restitution) randomization."""

    # Create object randomizer with physics randomization
    cfg = ObjectRandomCfg(
        obj_name="cube",
        physics=PhysicsRandomCfg(
            enabled=True,
            mass_range=(0.1, 1.0),
            friction_range=(0.1, 1.0),
            restitution_range=(0.1, 1.0),
            distribution=distribution,
            operation="abs",
        ),
    )

    randomizer = ObjectRandomizer(cfg, seed=789)
    randomizer.bind_handler(handler)

    # Get current mass before randomization
    current_mass = randomizer.get_mass("cube").cpu().numpy()
    current_friction = randomizer.get_friction("cube").cpu().numpy()
    current_restitution = randomizer.get_restitution("cube").cpu().numpy()
    # Apply randomization
    randomizer()
    # Get new mass after randomization
    new_mass = randomizer.get_mass("cube").cpu().numpy()
    new_friction = randomizer.get_friction("cube").cpu().numpy()
    new_restitution = randomizer.get_restitution("cube").cpu().numpy()

    assert (current_mass != new_mass).all() and (new_mass >= 0.1).all() and (new_mass <= 1.0).all(), (
        "Mass should have changed after randomization"
    )
    assert (current_friction != new_friction).all() and (new_friction >= 0.1).all() and (new_friction <= 1.0).all(), (
        "Friction should have changed after randomization"
    )
    assert (
        (current_restitution != new_restitution).all()
        and (new_restitution >= 0.1).all()
        and (new_restitution <= 1.0).all()
    ), "Restitution should have changed after randomization"

    # We don't enforce strict inequality since randomization could theoretically produce same value
    log.info(f"Object physics randomization (Type: {distribution}) test passed")


def object_pose(handler, distribution="uniform"):
    """Test object pose (position and rotation) randomization."""

    if distribution not in ["uniform", "gaussian"]:
        log.warning("Pose randomization only supports uniform and gaussian distributions")
        return

    # Create object randomizer with pose randomization
    cfg = ObjectRandomCfg(
        obj_name="cube",
        pose=PoseRandomCfg(
            enabled=True,
            position_range=((0.1, 10), (0.1, 10), (0.05, 0.05)),  # Don't change z
            rotation_range=(10, 60),
            rotation_axes=(False, False, True),  # Only rotate around z-axis
            distribution=distribution,
            operation="abs",
            keep_on_ground=False,
        ),
    )

    randomizer = ObjectRandomizer(cfg, seed=789)
    randomizer.bind_handler(handler)

    # Get current pose before randomization
    pos_before, rot_before = randomizer.get_pose("cube")
    # Apply randomization
    randomizer()

    # Get new pose after randomization
    pos_after, rot_after = randomizer.get_pose("cube")

    assert not (pos_before == pos_after).all(), "Position should have changed after randomization"
    assert (torch.abs(pos_before - pos_after)[:, :2] > 0.1).all() and (
        torch.abs(pos_before - pos_after)[:, :2] <= 10
    ).all(), "X and Y position should have changed, Z should remain the same"
    log.info(f"{rot_before} vs {rot_after}")
    assert not (rot_before == rot_after).all(), "Rotation should have changed after randomization"

    # r_before, p_before, y_before = euler_xyz_from_quat(rot_before)
    r_after, p_after, y_after = euler_xyz_from_quat(rot_after)

    # assert (
    #     (torch.abs(r_after - r_before) < 0.1)
    #     | ((torch.abs(r_after - r_before) - 2 * torch.pi).abs() < 0.1)
    # ).all(), (
    #     "X rotation should remain the same due to rotation_axes=(False, False, True)"
    # )
    # assert (
    #     (torch.abs(p_after - p_before) < 0.1)
    #     | ((torch.abs(p_after - p_before) - 2 * torch.pi).abs() < 0.1)
    # ).all(), "X and Y rotation should remain the same due to rotation_axes=(False, False, True)"
    assert ((torch.abs(y_after) >= 10 / 180 * 3.14159) & (torch.abs(y_after) <= 60 / 180 * 3.14159)).all(), (
        f"Z rotation should have changed, whereas {torch.abs(y_after)} not in range."
    )

    # Position or rotation should have changed (with high probability)
    log.info(f"Object pose randomization (Type: {distribution}) test passed")


def object_operation_types(handler, distribution="uniform"):
    """Test different operation types for object randomization."""
    # Test scale operation
    cfg_scale = ObjectRandomCfg(
        obj_name="cube",
        physics=PhysicsRandomCfg(
            enabled=True,
            mass_range=(1.1, 1.1),
            distribution=distribution,
            operation="scale",
        ),
    )
    randomizer_scale = ObjectRandomizer(cfg_scale, seed=789)
    randomizer_scale.bind_handler(handler)
    before_mass = randomizer_scale.get_mass("cube").cpu().clone()
    randomizer_scale()
    after_mass = randomizer_scale.get_mass("cube").cpu().clone()
    assert ((after_mass / before_mass - 1.1).abs() <= 1e-3).all(), "Scale operation failed"

    # Test add operation
    cfg_add = ObjectRandomCfg(
        obj_name="cube",
        physics=PhysicsRandomCfg(
            enabled=True,
            mass_range=(0.5, 0.5),
            distribution=distribution,
            operation="add",
        ),
    )
    randomizer_add = ObjectRandomizer(cfg_add, seed=789)
    randomizer_add.bind_handler(handler)
    before_mass = randomizer_add.get_mass("cube").cpu().clone()
    randomizer_add()
    after_mass = randomizer_add.get_mass("cube").cpu().clone()
    assert ((after_mass - before_mass - 0.5).abs() <= 1e-3).all(), "Add operation failed"

    # Test abs operation
    cfg_abs = ObjectRandomCfg(
        obj_name="cube",
        physics=PhysicsRandomCfg(
            enabled=True,
            mass_range=(1.0, 1.0),
            distribution=distribution,
            operation="abs",
        ),
    )
    randomizer_abs = ObjectRandomizer(cfg_abs, seed=789)
    randomizer_abs.bind_handler(handler)
    before_mass = randomizer_abs.get_mass("cube").cpu().clone()
    randomizer_abs()
    after_mass = randomizer_abs.get_mass("cube").cpu().clone()
    assert ((after_mass - 1.0).abs() <= 1e-3).all(), "Abs operation failed"

    log.info(f"Object operation types (Type: {distribution}) test passed")


def object_seed(handler, distribution="uniform"):
    """Test that object randomization is reproducible with same seed."""
    # Create object randomizer
    cfg = ObjectRandomCfg(
        obj_name="cube",
        physics=PhysicsRandomCfg(
            enabled=True,
            mass_range=(0.5, 2.0),
            distribution=distribution,
            operation="scale",
        ),
    )

    # Test reproducibility
    randomizer = ObjectRandomizer(cfg, seed=789)
    randomizer.bind_handler(handler)

    # Store RNG internal state by generating some values
    randomizer.set_seed(42)
    val1 = randomizer._rng.random()

    randomizer.set_seed(42)
    val2 = randomizer._rng.random()

    assert val1 == val2, "Same seed should produce same random values"
    log.info("Object seed reproducibility test passed")


def object_envid(handler, distribution="uniform"):
    """Test that randomization affects only specified env_ids (mass only)."""
    # Choose a subset of envs to randomize
    num_envs = handler.num_envs
    if num_envs < 2:
        log.info("Must have at least 2 environments to test env_id scoping.")
        return

    # Use even env indices as target set when possible, else [0]
    random_env_ids = [0]
    static_env_ids = [i for i in range(1, num_envs)]

    # Configure randomizer to only randomize mass on target envs
    cfg = ObjectRandomCfg(
        obj_name="cube",
        env_ids=random_env_ids,
        physics=PhysicsRandomCfg(
            enabled=True,
            mass_range=(0.1, 1.0),
            distribution=distribution,
            operation="abs",
        ),
    )

    randomizer = ObjectRandomizer(cfg, seed=789)
    randomizer.bind_handler(handler)

    # Mass before randomization (all envs)
    mass_before = randomizer.get_mass("cube").cpu().numpy()

    before_mass = get_object_from_randomizer(randomizer).root_physx_view.get_masses().cpu().clone()
    # Apply randomization limited to target_env_ids
    randomizer()
    # Mass after randomization (all envs)
    after_mass = get_object_from_randomizer(randomizer).root_physx_view.get_masses().cpu().clone()

    assert ((after_mass - before_mass)[random_env_ids].abs() > 1e-3).all(), (
        "Mass should have changed for target env_ids"
    )
    assert ((after_mass - before_mass)[static_env_ids].abs() <= 1e-3).all(), (
        "Mass should remain the same for non-target env_ids"
    )

    log.info(f"Object env_id scoping (Type: {distribution}) test passed")


TEST_FUNCTIONS = [
    object_physics,
    object_pose,
    object_operation_types,
    object_seed,
    # object_multiple_objects,
    object_envid,
]


def _process_run_handler(scenario):
    """Process function for standalone mode - creates its own handler."""
    from metasim.utils.setup_util import get_handler

    handler = get_handler(scenario)
    distributions = ["uniform", "log_uniform", "gaussian"]
    for dist in distributions:
        for test_func in TEST_FUNCTIONS:
            test_func(handler, distribution=dist)
    handler.close()


def run_test(sim="isaacsim", num_envs=2):
    """Standalone test function for direct execution."""
    import multiprocessing as mp

    log.info(f"Running object randomizer test in standalone mode with {sim} and {num_envs}")

    if sim not in ["isaacsim"]:
        log.warning(f"Skipping: Only testing IsaacSim here, got {sim}")
        return

    scenario = get_shared_scenario(sim, num_envs)

    ctx = mp.get_context("spawn")
    p = ctx.Process(target=_process_run_handler, args=(scenario,))
    p.start()
    p.join(timeout=60)

    assert p.exitcode == 0, f"IsaacSim process exited abnormally: {p.exitcode}"
    log.info("IsaacSim headless test finished successfully.")


@pytest.mark.usefixtures("shared_handler")
def test_object_randomizer_with_shared_handler(shared_handler):
    """Run object randomizer tests using the child-process handler via proxy."""

    log.info("Running object randomizer tests with shared handler (proxy)")

    proxy = shared_handler  # HandlerProxy

    distributions = ["uniform", "log_uniform", "gaussian"]
    # Run all object test functions with different distributions
    for dist in distributions:
        for test_func in TEST_FUNCTIONS:
            proxy.run_test(test_func, distribution=dist)

    log.info("All object randomizer tests completed with shared handler (proxy)")


if __name__ == "__main__":
    # Direct execution for quick testing - uses standalone mode
    import sys

    sim = "isaacsim" if len(sys.argv) < 2 else sys.argv[1]
    num_envs = 4 if len(sys.argv) < 3 else int(sys.argv[2])
    run_test(sim, num_envs)
