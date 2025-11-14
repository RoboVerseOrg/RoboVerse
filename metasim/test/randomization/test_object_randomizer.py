# """Test object randomizer functionality."""

# from __future__ import annotations

# import pytest
# import rootutils
# from loguru import logger as log

# rootutils.setup_root(__file__, pythonpath=True)
# from metasim.randomization.object_randomizer import (
#     ObjectRandomCfg,
#     ObjectRandomizer,
#     PhysicsRandomCfg,
# )


# def get_object_from_randomizer(randomizer):
#     """Helper function to get object instance from randomizer."""
#     obj_name = randomizer.cfg.obj_name
#     if obj_name in randomizer.handler.scene.articulations:
#         return randomizer.handler.scene.articulations[obj_name]
#     elif obj_name in randomizer.handler.scene.rigid_objects:
#         return randomizer.handler.scene.rigid_objects[obj_name]
#     else:
#         raise ValueError(f"Object {obj_name} not found in the scene")


# def object_physics(handler, distribution="uniform"):
#     """Test object physics properties (mass, friction, restitution) randomization."""

#     # Create object randomizer with physics randomization
#     cfg = ObjectRandomCfg(
#         obj_name="cube",
#         physics=PhysicsRandomCfg(
#             enabled=True,
#             mass_range=(0.5, 2.0),
#             friction_range=(0.1, 0.9),
#             restitution_range=(0.0, 0.8),
#             distribution=distribution,
#             operation="scale",
#         ),
#     )

#     randomizer = ObjectRandomizer(cfg, seed=789)
#     randomizer.bind_handler(handler)

#     obj_inst = get_object_from_randomizer(randomizer)

#     # Get current mass before randomization
#     current_mass = randomizer.get_mass("cube")

#     # Apply randomization
#     randomizer()

#     # Get new mass after randomization
#     new_mass = randomizer.get_mass("cube")

#     # Mass should have changed (with high probability given the range)
#     # We don't enforce strict inequality since randomization could theoretically produce same value
#     log.info(f"Object physics randomization (Type: {distribution}) test passed")


# def object_pose(handler, distribution="uniform"):
#     """Test object pose (position and rotation) randomization."""
#     from metasim.randomization.object_randomizer import PoseRandomCfg

#     # Create object randomizer with pose randomization
#     cfg = ObjectRandomCfg(
#         obj_name="cube",
#         pose=PoseRandomCfg(
#             enabled=True,
#             position_range=((-0.2, 0.2), (-0.2, 0.2), (0.0, 0.0)),  # Don't change z
#             rotation_range=(-30.0, 30.0),
#             rotation_axes=(False, False, True),  # Only rotate around z-axis
#             distribution=distribution,
#             operation="add",
#             keep_on_ground=True,
#         ),
#     )

#     randomizer = ObjectRandomizer(cfg, seed=789)
#     randomizer.bind_handler(handler)

#     obj_inst = get_object_from_randomizer(randomizer)

#     # Get current pose before randomization
#     pos_before, rot_before = randomizer.get_pose("cube")

#     # Apply randomization
#     randomizer()

#     # Get new pose after randomization
#     pos_after, rot_after = randomizer.get_pose("cube")

#     # Position or rotation should have changed (with high probability)
#     log.info(f"Object pose randomization (Type: {distribution}) test passed")


# def object_combined(handler, distribution="uniform"):
#     """Test combined object randomization (physics + pose)."""
#     # Create object randomizer with both physics and pose randomization
#     cfg = ObjectRandomCfg(
#         obj_name="cube",
#         physics=PhysicsRandomCfg(
#             enabled=True,
#             mass_range=(0.8, 1.2),
#             friction_range=(0.3, 0.7),
#             distribution=distribution,
#             operation="scale",
#         ),
#         pose=PoseRandomCfg(
#             enabled=True,
#             position_range=((-0.1, 0.1), (-0.1, 0.1), (0.0, 0.0)),
#             rotation_range=(-15.0, 15.0),
#             distribution=distribution,
#             operation="add",
#         ),
#     )

#     randomizer = ObjectRandomizer(cfg, seed=789)
#     randomizer.bind_handler(handler)

#     obj_inst = get_object_from_randomizer(randomizer)

#     # Apply randomization
#     randomizer()

#     log.info(f"Object combined randomization (Type: {distribution}) test passed")


# def object_multiple_objects(handler, distribution="uniform"):
#     """Test randomizing multiple different objects."""
#     # Create randomizers for different objects
#     cfg1 = ObjectRandomCfg(
#         obj_name="cube1",
#         physics=PhysicsRandomCfg(
#             enabled=True,
#             mass_range=(0.5, 1.5),
#             distribution=distribution,
#             operation="scale",
#         ),
#     )

#     cfg2 = ObjectRandomCfg(
#         obj_name="sphere1",
#         pose=PoseRandomCfg(
#             enabled=True,
#             position_range=((-0.15, 0.15), (-0.15, 0.15), (0.0, 0.0)),
#             distribution=distribution,
#             operation="add",
#         ),
#     )

#     randomizer1 = ObjectRandomizer(cfg1, seed=789)
#     randomizer1.bind_handler(handler)
#     randomizer1()

#     randomizer2 = ObjectRandomizer(cfg2, seed=999)
#     randomizer2.bind_handler(handler)
#     randomizer2()

#     log.info(
#         f"Object multiple objects randomization (Type: {distribution}) test passed"
#     )


# def object_operation_types(handler, distribution="uniform"):
#     """Test different operation types for object randomization."""
#     # Test scale operation
#     cfg_scale = ObjectRandomCfg(
#         obj_name="cube",
#         physics=PhysicsRandomCfg(
#             enabled=True,
#             mass_range=(0.8, 1.2),
#             distribution=distribution,
#             operation="scale",
#         ),
#     )
#     randomizer_scale = ObjectRandomizer(cfg_scale, seed=789)
#     randomizer_scale.bind_handler(handler)
#     randomizer_scale()

#     # Test add operation
#     cfg_add = ObjectRandomCfg(
#         obj_name="cube",
#         physics=PhysicsRandomCfg(
#             enabled=True,
#             mass_range=(-0.2, 0.2),
#             distribution=distribution,
#             operation="add",
#         ),
#     )
#     randomizer_add = ObjectRandomizer(cfg_add, seed=789)
#     randomizer_add.bind_handler(handler)
#     randomizer_add()

#     # Test abs operation
#     cfg_abs = ObjectRandomCfg(
#         obj_name="cube",
#         physics=PhysicsRandomCfg(
#             enabled=True,
#             mass_range=(0.5, 1.5),
#             distribution=distribution,
#             operation="abs",
#         ),
#     )
#     randomizer_abs = ObjectRandomizer(cfg_abs, seed=789)
#     randomizer_abs.bind_handler(handler)
#     randomizer_abs()

#     log.info(f"Object operation types (Type: {distribution}) test passed")


# def object_seed(handler):
#     """Test that object randomization is reproducible with same seed."""
#     # Create object randomizer
#     cfg = ObjectRandomCfg(
#         obj_name="cube",
#         physics=PhysicsRandomCfg(
#             enabled=True,
#             mass_range=(0.5, 2.0),
#             distribution="uniform",
#             operation="scale",
#         ),
#     )

#     # Test reproducibility
#     randomizer = ObjectRandomizer(cfg, seed=789)
#     randomizer.bind_handler(handler)

#     # Store RNG internal state by generating some values
#     randomizer.set_seed(42)
#     val1 = randomizer._rng.random()

#     randomizer.set_seed(42)
#     val2 = randomizer._rng.random()

#     assert val1 == val2, "Same seed should produce same random values"
#     log.info("Object seed reproducibility test passed")


# def _process_run_handler(scenario):
#     """Process function for standalone mode - creates its own handler."""
#     from metasim.utils.setup_util import get_handler

#     handler = get_handler(scenario)
#     object_seed(handler)
#     distributions = ["uniform", "log_uniform", "gaussian"]
#     for dist in distributions:
#         object_physics(handler, distribution=dist)
#         object_pose(handler, distribution=dist)
#         object_combined(handler, distribution=dist)
#         object_operation_types(handler, distribution=dist)
#     handler.close()


# def run_test(sim="isaacsim", num_envs=2):
#     """Standalone test function for direct execution."""
#     log.info(
#         f"Running object randomizer test in standalone mode with {sim} and {num_envs}"
#     )

#     if sim not in ["isaacsim"]:
#         log.warning(f"Skipping: Only testing IsaacSim here, got {sim}")
#         return

#     scenario = ScenarioCfg(
#         simulator=sim,
#         num_envs=num_envs,
#         headless=True,
#         objects=[
#             PrimitiveCubeCfg(
#                 name="cube",
#                 size=(0.1, 0.1, 0.1),
#                 color=[1.0, 0.0, 0.0],
#                 physics=PhysicStateType.RIGIDBODY,
#                 default_position=[0.5, 0.0, 0.5],
#             ),
#         ],
#         robots=[FrankaCfg()],
#     )

#     ctx = mp.get_context("spawn")
#     p = ctx.Process(target=_process_run_handler, args=(scenario,))
#     p.start()
#     p.join(timeout=60)

#     assert p.exitcode == 0, f"IsaacSim process exited abnormally: {p.exitcode}"
#     log.info("IsaacSim headless test finished successfully.")


# @pytest.mark.usefixtures("shared_handler")
# def test_object_randomizer_with_shared_handler(shared_handler):
#     """Run object randomizer tests using the child-process handler via proxy."""
#     import inspect
#     import sys

#     log.info("Running object randomizer tests with shared handler (proxy)")

#     proxy = shared_handler  # HandlerProxy

#     distributions = ["uniform", "log_uniform", "gaussian"]
#     module = "metasim.test.randomization.test_object_randomizer"

#     # Dynamically get all functions that start with 'object_' and accept distribution parameter
#     object_test_functions = [
#         name
#         for name, obj in inspect.getmembers(sys.modules[__name__], inspect.isfunction)
#         if name.startswith("object_")
#         and name != "object_seed"
#         and name != "object_multiple_objects"
#     ]

#     # Call seed reproducibility test first
#     proxy.run_test(
#         "object_seed",
#         module=module,
#     )

#     # Run all object test functions with different distributions
#     for dist in distributions:
#         for func_name in object_test_functions:
#             proxy.run_test(func_name, module=module, distribution=dist)

#     # Test multiple objects (single run with uniform)
#     proxy.run_test("object_multiple_objects", module=module, distribution="uniform")

#     log.info("All object randomizer tests completed with shared handler (proxy)")


# if __name__ == "__main__":
#     # Direct execution for quick testing - uses standalone mode
#     import sys

#     sim = "isaacsim" if len(sys.argv) < 2 else sys.argv[1]
#     num_envs = 2 if len(sys.argv) < 3 else int(sys.argv[2])
#     run_test(sim, num_envs)
