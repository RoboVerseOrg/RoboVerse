"""Object randomizer: physics through the handler's body-property verbs, pose through get/set_states.

Runs on every backend that implements the verbs; restitution, which MuJoCo and Newton do not model, reads NaN
there and a write warns and changes nothing.
"""

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

    # Get current mass before randomization
    current_mass = handler.get_body_masses("cube")
    current_friction = handler.get_body_friction("cube")
    current_restitution = handler.get_body_restitution("cube")

    # Apply randomization
    randomizer()

    new_mass = handler.get_body_masses("cube")
    new_friction = handler.get_body_friction("cube")
    new_restitution = handler.get_body_restitution("cube")
    assert (
        (current_mass != new_mass).all() and (new_mass >= common_range[0]).all() and (new_mass <= common_range[1]).all()
    ), "Mass should have changed after randomization"
    assert (
        (current_friction != new_friction).all()
        and (new_friction >= common_range[0]).all()
        and (new_friction <= common_range[1]).all()
    ), "Friction should have changed after randomization"
    if "restitution" in handler.unsupported_body_properties:
        assert torch.isnan(new_restitution).all(), "an engine without restitution reads NaN"
    else:
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

    # Get current pose before randomization
    root_state_before = handler.get_states(mode="tensor").objects["cube"].root_state.clone()
    pos_before = root_state_before[:, 0:3].cpu()
    rot_before = root_state_before[:, 3:7].cpu()
    r_before, p_before, y_before = euler_xyz_from_quat(rot_before)

    # Apply randomization
    randomizer()

    root_state_after = handler.get_states(mode="tensor").objects["cube"].root_state
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
    before_mass = handler.get_body_masses("cube")
    randomizer_scale()
    after_mass = handler.get_body_masses("cube")
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
    before_mass = handler.get_body_masses("cube")
    randomizer_add()
    after_mass = handler.get_body_masses("cube")
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
    before_mass = handler.get_body_masses("cube")
    randomizer_abs()
    after_mass = handler.get_body_masses("cube")
    assert ((after_mass - common_range[0]).abs() <= 1e-3).all(), "Abs operation failed"

    log.info(f"Object operation types (Type: {distribution}) test passed")


def object_seed(handler, distribution="uniform", common_range=(0.1, 1.0)):
    """The same seed draws the same masses."""
    cfg = ObjectRandomCfg(
        obj_name="cube",
        physics=PhysicsRandomCfg(enabled=True, mass_range=common_range, distribution=distribution, operation="abs"),
    )
    randomizer = ObjectRandomizer(cfg, seed=789)
    randomizer.bind_handler(handler)

    randomizer.set_seed(42)
    randomizer()
    val1 = handler.get_body_masses("cube")
    randomizer.set_seed(43)
    randomizer()
    assert not torch.equal(handler.get_body_masses("cube"), val1), "another seed draws other masses"
    randomizer.set_seed(42)
    randomizer()
    val2 = handler.get_body_masses("cube")

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

    # Mass before randomization (all envs)
    before_mass = handler.get_body_masses("cube")
    # Apply randomization limited to target_env_ids
    randomizer()
    # Mass after randomization (all envs)
    after_mass = handler.get_body_masses("cube")

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


@pytest.mark.sim("isaacsim", "mujoco", "newton")
@pytest.mark.parametrize("distribution", ["uniform", "log_uniform", "gaussian"])
@pytest.mark.parametrize("test_func", TEST_FUNCTIONS, ids=[f.__name__ for f in TEST_FUNCTIONS])
def test_object_randomizers(handler, test_func, distribution):
    """Run object randomizer checks inside the shared handler process."""
    common_range = (1e-8, 1.0)
    test_func(handler, distribution=distribution, common_range=common_range)


@pytest.mark.sim("isaacsim", "mujoco", "newton")
def test_material_randomizer_physical_goes_through_the_verbs(handler):
    """One friction / restitution draw per env on the object's bodies, on any backend with the verbs."""
    from metasim.randomization.material_randomizer import MaterialRandomCfg, MaterialRandomizer, PhysicalMaterialCfg

    cfg = MaterialRandomCfg(
        obj_name="cube", physical=PhysicalMaterialCfg(friction_range=(0.2, 0.3), restitution_range=(0.1, 0.2))
    )
    randomizer = MaterialRandomizer(cfg, seed=7)
    randomizer.bind_handler(handler)
    randomizer()
    friction = handler.get_body_friction("cube")
    assert ((friction >= 0.2) & (friction <= 0.3)).all()
    restitution = handler.get_body_restitution("cube")
    if "restitution" in handler.unsupported_body_properties:
        assert torch.isnan(restitution).all()
    else:
        assert ((restitution >= 0.1) & (restitution <= 0.2)).all()


@pytest.mark.sim("isaacsim", "mujoco", "newton")
def test_body_name_of_a_rigid_object_is_its_own_name(handler):
    cfg = ObjectRandomCfg(
        obj_name="cube",
        body_name="cube",
        physics=PhysicsRandomCfg(enabled=True, mass_range=(0.5, 0.5), operation="abs"),
    )
    randomizer = ObjectRandomizer(cfg, seed=1)
    randomizer.bind_handler(handler)
    randomizer()
    torch.testing.assert_close(handler.get_body_masses("cube"), torch.full((handler.num_envs, 1), 0.5))
    lid = ObjectRandomizer(ObjectRandomCfg(obj_name="cube", body_name="lid", physics=cfg.physics), seed=1)
    lid.bind_handler(handler)
    with pytest.raises(ValueError, match="has no body 'lid'"):
        lid()


if __name__ == "__main__":
    pytest.main([__file__, "-k isaacsim"])
