# """Test material randomizer functionality."""

# from __future__ import annotations

# import pytest
# import rootutils
# from loguru import logger as log

# rootutils.setup_root(__file__, pythonpath=True)

# from metasim.randomization.material_randomizer import (
#     MaterialRandomCfg,
#     MaterialRandomizer,
#     PBRMaterialCfg,
# )
# from metasim.test.randomization.conftest import get_shared_scenario


# def get_material_properties_from_randomizer(randomizer):
#     """Helper function to get material properties from randomizer."""
#     obj_inst = randomizer._get_object_instance(randomizer.cfg.obj_name)
#     if not obj_inst:
#         raise ValueError("Object not found in the scene")
#     return obj_inst


# def physical_material_randomization(handler, distribution="uniform"):
#     """Test physical material (friction, restitution) randomization."""
#     from metasim.randomization.material_randomizer import (
#         MaterialRandomCfg,
#         MaterialRandomizer,
#         PhysicalMaterialCfg,
#     )

#     # Create material randomizer with physical properties
#     cfg = MaterialRandomCfg(
#         obj_name="cube",
#         physical=PhysicalMaterialCfg(
#             friction_range=(0.1, 0.9),
#             restitution_range=(0.0, 0.8),
#             distribution=distribution,
#             enabled=True,
#         ),
#         randomization_mode="physics_only",
#     )

#     randomizer = MaterialRandomizer(cfg, seed=789)
#     randomizer.bind_handler(handler)

#     obj_inst = get_material_properties_from_randomizer(randomizer)

#     # Apply randomization
#     randomizer()

#     # For physical properties, we can check that the randomizer was called successfully
#     # The actual physics properties are internal to the simulation
#     log.info(f"Physical material randomization (Type: {distribution}) test passed")


# def pbr_material_randomization(handler, distribution="uniform"):
#     """Test PBR material (roughness, metallic) randomization."""
#     from metasim.randomization.material_randomizer import PBRMaterialCfg

#     # Create material randomizer with PBR properties
#     cfg = MaterialRandomCfg(
#         obj_name="cube",
#         pbr=PBRMaterialCfg(
#             roughness_range=(0.1, 0.9),
#             metallic_range=(0.0, 1.0),
#             diffuse_color_range=((0.5, 1.0), (0.5, 1.0), (0.5, 1.0)),
#             distribution=distribution,
#             enabled=True,
#         ),
#         randomization_mode="visual_only",
#     )

#     randomizer = MaterialRandomizer(cfg, seed=789)
#     randomizer.bind_handler(handler)

#     obj_inst = get_material_properties_from_randomizer(randomizer)

#     # Apply randomization
#     randomizer()

#     log.info(f"PBR material randomization (Type: {distribution}) test passed")


# def material_multiple_objects(handler, distribution="uniform"):
#     """Test material randomization on multiple objects."""

#     # Create material randomizers for different objects
#     cfg1 = MaterialRandomCfg(
#         obj_name="cube1",
#         pbr=PBRMaterialCfg(
#             roughness_range=(0.1, 0.5),
#             metallic_range=(0.5, 1.0),
#             distribution=distribution,
#             enabled=True,
#         ),
#         randomization_mode="visual_only",
#     )

#     cfg2 = MaterialRandomCfg(
#         obj_name="sphere1",
#         pbr=PBRMaterialCfg(
#             roughness_range=(0.5, 0.9),
#             metallic_range=(0.0, 0.3),
#             distribution=distribution,
#             enabled=True,
#         ),
#         randomization_mode="visual_only",
#     )

#     randomizer1 = MaterialRandomizer(cfg1, seed=789)
#     randomizer1.bind_handler(handler)
#     randomizer1()

#     randomizer2 = MaterialRandomizer(cfg2, seed=999)
#     randomizer2.bind_handler(handler)
#     randomizer2()

#     log.info(
#         f"Material multiple objects randomization (Type: {distribution}) test passed"
#     )


# def material_seed_reproducibility(handler):
#     """Test that material randomization is reproducible with same seed."""
#     # Create material randomizer
#     cfg = MaterialRandomCfg(
#         obj_name="cube",
#         pbr=PBRMaterialCfg(
#             roughness_range=(0.1, 0.9),
#             enabled=True,
#         ),
#     )

#     # Test reproducibility
#     randomizer = MaterialRandomizer(cfg, seed=789)
#     randomizer.bind_handler(handler)

#     # Store RNG internal state by generating some values
#     randomizer.set_seed(42)
#     val1 = randomizer._rng.random()

#     randomizer.set_seed(42)
#     val2 = randomizer._rng.random()

#     assert val1 == val2, "Same seed should produce same random values"
#     log.info("Material seed reproducibility test passed")


# def _process_run_handler(scenario):
#     """Process function for standalone mode - creates its own handler."""
#     from metasim.utils.setup_util import get_handler

#     handler = get_handler(scenario)
#     material_seed_reproducibility(handler)
#     distributions = ["uniform", "log_uniform", "gaussian"]
#     for dist in distributions:
#         physical_material_randomization(handler, distribution=dist)
#         pbr_material_randomization(handler, distribution=dist)

#     # Test multiple objects (only once, not per distribution)
#     material_multiple_objects(handler, distribution="uniform")

#     handler.close()


# def run_test(sim="isaacsim", num_envs=2):
#     """Standalone test function for direct execution."""
#     import multiprocessing as mp

#     log.info(
#         f"Running material randomizer test in standalone mode with {sim} and {num_envs}"
#     )

#     if sim not in ["isaacsim"]:
#         log.warning(f"Skipping: Only testing IsaacSim here, got {sim}")
#         return

#     scenario = get_shared_scenario(sim, num_envs)
#     ctx = mp.get_context("spawn")
#     p = ctx.Process(target=_process_run_handler, args=(scenario,))
#     p.start()
#     p.join(timeout=60)

#     assert p.exitcode == 0, f"IsaacSim process exited abnormally: {p.exitcode}"
#     log.info("IsaacSim headless test finished successfully.")


# @pytest.mark.usefixtures("shared_handler")
# def test_material_randomizer_with_shared_handler(shared_handler):
#     """Run material randomizer tests using the child-process handler via proxy."""
#     import inspect
#     import sys

#     log.info("Running material randomizer tests with shared handler (proxy)")

#     proxy = shared_handler  # HandlerProxy

#     distributions = ["uniform", "log_uniform", "gaussian"]
#     module = "metasim.test.randomization.test_material_randomizer"

#     # Dynamically get all functions that start with material-related prefixes and accept distribution parameter
#     material_test_functions = [
#         name
#         for name, obj in inspect.getmembers(sys.modules[__name__], inspect.isfunction)
#         if (
#             name.startswith("physical_")
#             or name.startswith("pbr_")
#             or name.startswith("material_")
#         )
#         and name != "material_seed_reproducibility"
#         and name != "material_multiple_objects"
#     ]

#     # Call seed reproducibility test first
#     proxy.run_test(
#         "material_seed_reproducibility",
#         module=module,
#     )

#     # Run all material test functions with different distributions
#     for dist in distributions:
#         for func_name in material_test_functions:
#             proxy.run_test(func_name, module=module, distribution=dist)

#     # Test multiple objects (single run with uniform)
#     proxy.run_test("material_multiple_objects", module=module, distribution="uniform")

#     log.info("All material randomizer tests completed with shared handler (proxy)")


# if __name__ == "__main__":
#     # Direct execution for quick testing - uses standalone mode
#     import sys

#     sim = "isaacsim" if len(sys.argv) < 2 else sys.argv[1]
#     num_envs = 2 if len(sys.argv) < 3 else int(sys.argv[2])
#     run_test(sim, num_envs)
