# """Test material randomizer functionality."""

# from __future__ import annotations

# import pytest
# import rootutils
# from loguru import logger as log

# rootutils.setup_root(__file__, pythonpath=True)

# from metasim.randomization.material_randomizer import (
#     MaterialRandomCfg,
#     MaterialRandomizer,
# )
# from metasim.test.randomization.conftest import get_shared_scenario


# def get_pbr_properties(randomizer, env_id: int = 0) -> dict:
#     """Get current PBR properties from the material shader.

#     Args:
#         env_id: Environment ID to query (default: 0)

#     Returns:
#         Dictionary with PBR properties (roughness, metallic, specular, diffuseColor)
#     """
#     try:
#         import omni.isaac.core.utils.prims as prim_utils
#     except ModuleNotFoundError:
#         import isaacsim.core.utils.prims as prim_utils
#     if not randomizer.cfg.pbr:
#         return {}

#     obj_inst = randomizer._get_object_instance(randomizer.cfg.obj_name)
#     root_path = obj_inst.cfg.prim_path
#     env_prim_path = f"{root_path}_{env_id}"

#     prim = prim_utils.get_prim_at_path(env_prim_path)
#     if not prim:
#         return {}

#     # Get bound material
#     material_binding = UsdShade.MaterialBindingAPI(prim)
#     material = material_binding.ComputeBoundMaterial()[0]

#     if not material:
#         return {}

#     # Get shader from material
#     shader = UsdShade.Shader(omni.usd.get_shader_from_material(material, get_prim=True))
#     if not shader:
#         return {}

#     # Extract PBR properties from shader inputs
#     properties = {}

#     roughness_input = shader.GetInput("roughness")
#     if roughness_input:
#         properties["roughness"] = roughness_input.Get()

#     metallic_input = shader.GetInput("metallic")
#     if metallic_input:
#         properties["metallic"] = metallic_input.Get()

#     specular_input = shader.GetInput("specular")
#     if specular_input:
#         properties["specular"] = specular_input.Get()

#     diffuse_input = shader.GetInput("diffuseColor")
#     if diffuse_input:
#         color_val = diffuse_input.Get()
#         if color_val:
#             properties["diffuseColor"] = (color_val[0], color_val[1], color_val[2])

#     return properties


# def material_physical(handler, distribution="uniform"):
#     """Test physical material (friction, restitution) randomization."""
#     from metasim.randomization.material_randomizer import PhysicalMaterialCfg

#     # Create material randomizer with physical properties
#     cfg = MaterialRandomCfg(
#         obj_name="cube",
#         physical=PhysicalMaterialCfg(
#             friction_range=(0.1, 0.9),
#             restitution_range=(0.0, 0.8),
#             distribution=distribution,
#             enabled=True,
#         ),
#     )

#     randomizer = MaterialRandomizer(cfg, seed=789)
#     randomizer.bind_handler(handler)

#     material_prop = randomizer.get_physical_properties()

#     current_friction = material_prop["friction"]
#     current_restitution = material_prop["restitution"]
#     # Apply randomization
#     randomizer()

#     new_material_prop = randomizer.get_physical_properties()
#     new_friction = new_material_prop["friction"]
#     new_restitution = new_material_prop["restitution"]

#     assert (current_friction != new_friction).all(), "Friction should be randomized"
#     assert (
#         current_restitution != new_restitution
#     ).all(), "Restitution should be randomized"
#     assert (current_friction >= 0.1).all() and (
#         current_friction <= 0.9
#     ).all(), "Friction out of range"
#     assert (current_restitution >= 0.0).all() and (
#         current_restitution <= 0.8
#     ).all(), "Restitution out of range"
#     # For physical properties, we can check that the randomizer was called successfully
#     # The actual physics properties are internal to the simulation
#     log.info(f"Physical material randomization (Type: {distribution}) test passed")


# def material_pbr(handler, distribution="uniform"):
#     """Test PBR material (roughness, metallic) randomization."""
#     from metasim.randomization.material_randomizer import PBRMaterialCfg

#     # Create material randomizer with PBR properties
#     cfg = MaterialRandomCfg(
#         obj_name="cube",
#         pbr=PBRMaterialCfg(
#             roughness_range=(0.1, 0.9),
#             metallic_range=(0.0, 1.0),
#             specular_range=(0.0, 1.0),
#             diffuse_color_range=((0.5, 1.0), (0.5, 1.0), (0.5, 1.0)),
#             distribution=distribution,
#             enabled=True,
#         ),
#     )

#     randomizer = MaterialRandomizer(cfg, seed=789)
#     randomizer.bind_handler(handler)

#     current_pbr = randomizer.get_pbr_properties()
#     # Apply randomization - this creates a new material with randomized properties
#     randomizer()

#     # Verify PBR properties were set after randomization
#     # Note: get_pbr_properties retrieves from the shader that was created by _randomize_prim_pbr
#     new_pbr = randomizer.get_pbr_properties()

#     # Validate that properties exist and are within expected ranges
#     assert "roughness" in new_pbr, "Roughness property not found"
#     assert "metallic" in new_pbr, "Metallic property not found"
#     assert "specular" in new_pbr, "Specular property not found"
#     assert "diffuseColor" in new_pbr, "DiffuseColor property not found"

#     assert (
#         0.1 <= new_pbr["roughness"] <= 0.9
#     ), f"Roughness {new_pbr['roughness']} out of range [0.1, 0.9]"
#     assert (
#         0.0 <= new_pbr["metallic"] <= 1.0
#     ), f"Metallic {new_pbr['metallic']} out of range [0.0, 1.0]"
#     assert (
#         0.0 <= new_pbr["specular"] <= 1.0
#     ), f"Specular {new_pbr['specular']} out of range [0.0, 1.0]"

#     # Validate diffuse color components
#     diffuse = new_pbr["diffuseColor"]
#     assert 0.5 <= diffuse[0] <= 1.0, f"Diffuse R {diffuse[0]} out of range [0.5, 1.0]"
#     assert 0.5 <= diffuse[1] <= 1.0, f"Diffuse G {diffuse[1]} out of range [0.5, 1.0]"
#     assert 0.5 <= diffuse[2] <= 1.0, f"Diffuse B {diffuse[2]} out of range [0.5, 1.0]"

#     log.info(f"PBR material randomization (Type: {distribution}) test passed")


# def material_mdl(handler, distribution="uniform"):
#     pass


# def material_multi_objects(handler, distribution="uniform"):
#     pass


# def material_envid(handler, distribution="uniform"):
#     pass


# def material_seed(handler, distribution="uniform"):
#     """Test that material randomization is reproducible with same seed."""
#     from metasim.randomization.material_randomizer import PhysicalMaterialCfg

#     # Create material randomizer with physical properties
#     cfg = MaterialRandomCfg(
#         obj_name="cube",
#         physical=PhysicalMaterialCfg(
#             friction_range=(0.1, 0.9),
#             restitution_range=(0.0, 0.8),
#             distribution=distribution,
#             enabled=True,
#         ),
#     )

#     # Test reproducibility
#     randomizer = MaterialRandomizer(cfg, seed=789)
#     randomizer.bind_handler(handler)
#     # Apply randomization twice with same seed - should give same results
#     randomizer.set_seed(42)
#     randomizer()
#     # Store the state by getting random value after first call
#     val1 = randomizer._rng.random()

#     randomizer.set_seed(42)
#     randomizer()
#     val2 = randomizer._rng.random()

#     assert val1 == val2, "Same seed should produce same random values"
#     log.info("Material seed reproducibility test passed")


# TEST_FUNCTIONS = [
#     material_physical,
#     # material_pbr,
#     material_mdl,
#     material_multi_objects,
#     material_envid,
#     material_seed,
#     material_envid
# ]


# def _process_run_handler(scenario):
#     """Process function for standalone mode - creates its own handler."""
#     from metasim.utils.setup_util import get_handler

#     handler = get_handler(scenario)
#     distributions = ["uniform", "log_uniform", "gaussian"]
#     for dist in distributions:
#         for test_func in TEST_FUNCTIONS:
#             test_func(handler, distribution=dist)
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
#     log.info("Running material randomizer tests with shared handler (proxy)")

#     proxy = shared_handler  # HandlerProxy

#     distributions = ["uniform", "log_uniform", "gaussian"]

#     # Run all material test functions with different distributions
#     for dist in distributions:
#         for test_func in TEST_FUNCTIONS:
#             proxy.run_test(func=test_func, distribution=dist)

#     log.info("All material randomizer tests completed with shared handler (proxy)")


# if __name__ == "__main__":
#     # Direct execution for quick testing - uses standalone mode
#     import sys

#     sim = "isaacsim" if len(sys.argv) < 2 else sys.argv[1]
#     num_envs = 2 if len(sys.argv) < 3 else int(sys.argv[2])
#     run_test(sim, num_envs)
