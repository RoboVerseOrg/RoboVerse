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
from metasim.utils.math import euler_xyz_from_quat


def object_physics(handler, distribution="uniform", common_range=(0.1, 1.0)):
    """Test object physics properties (mass, friction, restitution) randomization."""

    # Create object randomizer with physics randomization
    cfg = ObjectRandomCfg(
        obj_name="cube",
        physics=PhysicsRandomCfg(
            enabled=True,
            mass_range=common_range,
            friction_range=common_range,
            restitution_range=common_range,
            distribution=distribution,
            operation="abs",
        ),
    )

    randomizer = ObjectRandomizer(cfg, seed=789)
    randomizer.bind_handler(handler)

    # Get object instance
    obj_inst = _get_object(randomizer)

    # Get current mass before randomization
    current_mass = obj_inst.root_physx_view.get_masses().clone().cpu()
    current_materials = obj_inst.root_physx_view.get_material_properties().clone().cpu()
    current_friction = current_materials[..., 0]  # static friction
    current_restitution = current_materials[..., 2]

    # Apply randomization
    randomizer()

    # Get new mass after randomization
    new_mass = obj_inst.root_physx_view.get_masses().clone().cpu()
    new_materials = obj_inst.root_physx_view.get_material_properties().clone().cpu()
    new_friction = new_materials[..., 0]
    new_restitution = new_materials[..., 2]
    assert (
        (current_mass != new_mass).all() and (new_mass >= common_range[0]).all() and (new_mass <= common_range[1]).all()
    ), "Mass should have changed after randomization"
    assert (
        (current_friction != new_friction).all()
        and (new_friction >= common_range[0]).all()
        and (new_friction <= common_range[1]).all()
    ), "Friction should have changed after randomization"
    assert (
        (current_restitution != new_restitution).all()
        and (new_restitution >= common_range[0]).all()
        and (new_restitution <= common_range[1]).all()
    ), "Restitution should have changed after randomization"

    # We don't enforce strict inequality since randomization could theoretically produce same value
    log.info(f"Object physics randomization (Type: {distribution}) test passed")


def object_pose(handler, distribution="uniform", common_range=(0.1, 1.0)):
    """Test object pose (position and rotation) randomization."""

    if distribution not in ["uniform", "gaussian"]:
        log.warning("Pose randomization only supports uniform and gaussian distributions")
        return

    # Create object randomizer with pose randomization
    cfg = ObjectRandomCfg(
        obj_name="cube",
        pose=PoseRandomCfg(
            enabled=True,
            position_range=(common_range, common_range, (0.05, 0.05)),  # Don't change z
            rotation_range=common_range,
            rotation_axes=(False, False, True),  # Only rotate around z-axis
            distribution=distribution,
            operation="add",
            keep_on_ground=False,
        ),
    )

    randomizer = ObjectRandomizer(cfg, seed=789)
    randomizer.bind_handler(handler)

    # Get object instance
    obj_inst = _get_object(randomizer)

    # Get current pose before randomization
    root_state_before = obj_inst.data.root_state_w
    pos_before = root_state_before[:, 0:3].cpu()
    rot_before = root_state_before[:, 3:7].cpu()
    r_before, p_before, y_before = euler_xyz_from_quat(rot_before)

    # Apply randomization
    randomizer()

    # Get new pose after randomization
    root_state_after = obj_inst.data.root_state_w
    pos_after = root_state_after[:, 0:3].cpu()
    rot_after = root_state_after[:, 3:7].cpu()

    assert not (pos_before == pos_after).all(), "Position should have changed after randomization"
    assert (torch.abs(pos_before - pos_after)[:, :2] > common_range[0]).all() and (
        torch.abs(pos_before - pos_after)[:, :2] <= common_range[1]
    ).all(), "X and Y position should have changed, Z should remain the same"
    # log.info(f"{rot_before} vs {rot_after}")
    assert not (rot_before == rot_after).all(), "Rotation should have changed after randomization"

    # r_before, p_before, y_before = euler_xyz_from_quat(rot_before)
    r_after, p_after, y_after = euler_xyz_from_quat(rot_after)

    assert (
        (torch.abs(r_after - r_before) < 1e-3) | ((torch.abs(r_after - r_before) - 2 * torch.pi).abs() < 1e-3)
    ).all(), "X rotation should remain the same due to rotation_axes=(False, False, True)"
    assert (
        (torch.abs(p_after - p_before) < 1e-3) | ((torch.abs(p_after - p_before) - 2 * torch.pi).abs() < 1e-3)
    ).all(), "X and Y rotation should remain the same due to rotation_axes=(False, False, True)"
    y_ranges = (common_range[0] / 180 * 3.14159, common_range[1] / 180 * 3.14159)
    y_diff = torch.abs(y_after - y_before)
    assert (
        ((y_diff >= y_ranges[0]) | ((y_diff - 2 * torch.pi).abs() >= y_ranges[0]))
        & ((y_diff <= y_ranges[1]) | ((y_diff - 2 * torch.pi).abs() <= y_ranges[1]))
    ).all(), f"Z rotation should have changed, whereas {torch.abs(y_after - y_before)} not in range."

    # Position or rotation should have changed (with high probability)
    log.info(f"Object pose randomization (Type: {distribution}) test passed")


def object_operation_types(handler, distribution="uniform", common_range=(0.1, 1.0)):
    """Test different operation types for object randomization."""
    common_range = (common_range[1], common_range[1])  # Use fixed value for easier testing
    # Test scale operation
    cfg_scale = ObjectRandomCfg(
        obj_name="cube",
        physics=PhysicsRandomCfg(
            enabled=True,
            mass_range=common_range,
            distribution=distribution,
            operation="scale",
        ),
    )
    randomizer_scale = ObjectRandomizer(cfg_scale, seed=789)
    randomizer_scale.bind_handler(handler)
    obj_inst = _get_object(randomizer_scale)
    before_mass = obj_inst.root_physx_view.get_masses().clone().cpu()
    randomizer_scale()
    after_mass = obj_inst.root_physx_view.get_masses().clone().cpu()
    assert ((after_mass / before_mass - common_range[0]).abs() <= 1e-3).all(), "Scale operation failed"

    # Test add operation
    cfg_add = ObjectRandomCfg(
        obj_name="cube",
        physics=PhysicsRandomCfg(
            enabled=True,
            mass_range=common_range,
            distribution=distribution,
            operation="add",
        ),
    )
    randomizer_add = ObjectRandomizer(cfg_add, seed=789)
    randomizer_add.bind_handler(handler)
    obj_inst = _get_object(randomizer_add)
    before_mass = obj_inst.root_physx_view.get_masses().clone().cpu()
    randomizer_add()
    after_mass = obj_inst.root_physx_view.get_masses().clone().cpu()
    assert ((after_mass - before_mass - common_range[0]).abs() <= 1e-3).all(), "Add operation failed"

    # Test abs operation
    cfg_abs = ObjectRandomCfg(
        obj_name="cube",
        physics=PhysicsRandomCfg(
            enabled=True,
            mass_range=common_range,
            distribution=distribution,
            operation="abs",
        ),
    )
    randomizer_abs = ObjectRandomizer(cfg_abs, seed=789)
    randomizer_abs.bind_handler(handler)
    obj_inst = _get_object(randomizer_abs)
    before_mass = obj_inst.root_physx_view.get_masses().clone().cpu()
    randomizer_abs()
    after_mass = obj_inst.root_physx_view.get_masses().clone().cpu()
    assert ((after_mass - common_range[0]).abs() <= 1e-3).all(), "Abs operation failed"

    log.info(f"Object operation types (Type: {distribution}) test passed")


def object_seed(handler, distribution="uniform", common_range=(0.1, 1.0)):
    """Test that object randomization is reproducible with same seed."""
    # Create object randomizer
    cfg = ObjectRandomCfg(
        obj_name="cube",
        physics=PhysicsRandomCfg(
            enabled=True,
            mass_range=common_range,
            distribution=distribution,
            operation="scale",
        ),
    )

    # Test reproducibility
    randomizer = ObjectRandomizer(cfg, seed=789)
    randomizer.bind_handler(handler)

    # Get object instance
    obj_inst = _get_object(randomizer)
    # Store RNG internal state by generating some values
    randomizer.set_seed(42)
    val1 = obj_inst.root_physx_view.get_masses().clone().cpu()

    randomizer.set_seed(42)
    val2 = obj_inst.root_physx_view.get_masses().clone().cpu()

    assert (val1 == val2).all(), "Same seed should produce same random values"
    log.info("Object seed reproducibility test passed")


def object_envid(handler, distribution="uniform", common_range=(0.1, 1.0)):
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
            mass_range=common_range,
            distribution=distribution,
            operation="abs",
        ),
    )

    randomizer = ObjectRandomizer(cfg, seed=789)
    randomizer.bind_handler(handler)

    # Get object instance
    obj_inst = _get_object(randomizer)

    # Mass before randomization (all envs)
    before_mass = obj_inst.root_physx_view.get_masses().clone().cpu()
    # Apply randomization limited to target_env_ids
    randomizer()
    # Mass after randomization (all envs)
    after_mass = obj_inst.root_physx_view.get_masses().clone().cpu()

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
    object_envid,
]


@pytest.mark.isaacsim
@pytest.mark.parametrize("distribution", ["uniform", "log_uniform", "gaussian"])
@pytest.mark.parametrize("test_func", TEST_FUNCTIONS, ids=[f.__name__ for f in TEST_FUNCTIONS])
def test_object_randomizers(handler, test_func, distribution):
    """Run object randomizer checks inside the shared handler process."""
    common_range = (1e-8, 1.0)
    test_func(handler, distribution=distribution, common_range=common_range)


def _get_object(randomizer):
    """Helper function to get object instance from randomizer."""
    obj_name = randomizer.cfg.obj_name
    handler = randomizer._actual_handler
    if obj_name in handler.scene.articulations:
        return handler.scene.articulations[obj_name]
    elif obj_name in handler.scene.rigid_objects:
        return handler.scene.rigid_objects[obj_name]
    else:
        raise ValueError(f"Object {obj_name} not found in the scene")


if __name__ == "__main__":
    pytest.main([__file__, "-k isaacsim"])
