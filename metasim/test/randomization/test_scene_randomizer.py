# """Test scene randomizer functionality."""

# from __future__ import annotations

# import pytest
# import rootutils
# from loguru import logger as log

# rootutils.setup_root(__file__, pythonpath=True)


# def get_scene_properties_from_randomizer(randomizer):
#     """Helper function to get scene properties from randomizer."""
#     return randomizer.get_scene_properties()


# def scene_floor_creation(handler, material=False):
#     """Test scene floor creation and randomization."""
#     from metasim.randomization.scene_randomizer import (
#         SceneGeometryCfg,
#         SceneMaterialPoolCfg,
#         SceneRandomCfg,
#         SceneRandomizer,
#     )

#     # Create scene randomizer with floor
#     cfg = SceneRandomCfg(
#         floor=SceneGeometryCfg(
#             enabled=True,
#             size=(10.0, 10.0, 0.1),
#             position=(0.0, 0.0, -0.05),
#             material=material,
#         ),
#         floor_materials=(
#             SceneMaterialPoolCfg(
#                 material_paths=[],
#                 selection_strategy="random",
#                 randomize_material_variant=True,
#             )
#             if material
#             else None
#         ),
#         only_if_no_scene=True,
#     )

#     randomizer = SceneRandomizer(cfg, seed=789)
#     randomizer.bind_handler(handler)

#     # Apply randomization
#     randomizer()

#     # Verify scene was created
#     properties = get_scene_properties_from_randomizer(randomizer)

#     log.info(
#         f"Scene floor creation test passed (material={material})"
#     )


# def scene_walls_creation(handler, material=False):
#     """Test scene walls creation."""
#     # Create scene randomizer with walls
#     cfg = SceneRandomCfg(
#         walls=SceneGeometryCfg(
#             enabled=True,
#             size=(10.0, 0.2, 3.0),
#             position=(0.0, 0.0, 0.0),
#             material=material,
#         ),
#         wall_materials=(
#             SceneMaterialPoolCfg(
#                 material_paths=[],
#                 selection_strategy="random",
#             )
#             if material
#             else None
#         ),
#         only_if_no_scene=True,
#     )

#     randomizer = SceneRandomizer(cfg, seed=789)
#     randomizer.bind_handler(handler)

#     # Apply randomization
#     randomizer()

#     # Verify scene was created
#     properties = get_scene_properties_from_randomizer(randomizer)

#     log.info(
#         f"Scene walls creation test passed (material={material})"
#     )


# def scene_ceiling_creation(handler, material=False):
#     """Test scene ceiling creation."""
#     # Create scene randomizer with ceiling
#     cfg = SceneRandomCfg(
#         ceiling=SceneGeometryCfg(
#             enabled=True,
#             size=(10.0, 10.0, 0.1),
#             position=(0.0, 0.0, 3.0),
#             material=material,
#         ),
#         ceiling_materials=(
#             SceneMaterialPoolCfg(
#                 material_paths=[],
#                 selection_strategy="random",
#             )
#             if material
#             else None
#         ),
#         only_if_no_scene=True,
#     )

#     randomizer = SceneRandomizer(cfg, seed=789)
#     randomizer.bind_handler(handler)

#     # Apply randomization
#     randomizer()

#     # Verify scene was created
#     properties = get_scene_properties_from_randomizer(randomizer)

#     log.info(
#         f"Scene ceiling creation test passed (material={material})"
#     )


# def scene_table_creation(handler, material=False):
#     """Test scene table creation."""
#     # Create scene randomizer with table
#     cfg = SceneRandomCfg(
#         table=SceneGeometryCfg(
#             enabled=True,
#             size=(1.5, 1.0, 0.05),
#             position=(0.5, 0.0, 0.4),
#             material=material,
#         ),
#         table_materials=(
#             SceneMaterialPoolCfg(
#                 material_paths=[],
#                 selection_strategy="random",
#             )
#             if material
#             else None
#         ),
#         only_if_no_scene=True,
#     )

#     randomizer = SceneRandomizer(cfg, seed=789)
#     randomizer.bind_handler(handler)

#     # Apply randomization
#     randomizer()

#     # Verify scene was created
#     properties = get_scene_properties_from_randomizer(randomizer)

#     log.info(
#         f"Scene table creation test passed (material={material})"
#     )


# def scene_combined_elements(handler):
#     """Test scene creation with multiple elements (floor + walls + table)."""
#     # Create scene randomizer with multiple elements
#     cfg = SceneRandomCfg(
#         floor=SceneGeometryCfg(
#             enabled=True,
#             size=(10.0, 10.0, 0.1),
#             position=(0.0, 0.0, -0.05),
#             material=False,
#         ),
#         walls=SceneGeometryCfg(
#             enabled=True,
#             size=(10.0, 0.2, 3.0),
#             position=(0.0, 0.0, 0.0),
#             material=False,
#         ),
#         table=SceneGeometryCfg(
#             enabled=True,
#             size=(1.5, 1.0, 0.05),
#             position=(0.5, 0.0, 0.4),
#             material=False,
#         ),
#         only_if_no_scene=True,
#     )

#     randomizer = SceneRandomizer(cfg, seed=789)
#     randomizer.bind_handler(handler)

#     # Apply randomization
#     randomizer()

#     # Verify multiple elements were created
#     properties = get_scene_properties_from_randomizer(randomizer)

#     log.info("Scene combined elements test passed")


# def scene_material_selection_strategies(handler, strategy="random"):
#     """Test different material selection strategies."""
#     # Create scene randomizer with specified strategy
#     cfg = SceneRandomCfg(
#         floor=SceneGeometryCfg(
#             enabled=True,
#             size=(10.0, 10.0, 0.1),
#             position=(0.0, 0.0, -0.05),
#             material=True,
#         ),
#         floor_materials=SceneMaterialPoolCfg(
#             material_paths=[],
#             selection_strategy=strategy,
#         ),
#         only_if_no_scene=True,
#     )

#     randomizer = SceneRandomizer(cfg, seed=789)
#     randomizer.bind_handler(handler)
#     randomizer()

#     log.info(f"Scene material selection strategies test passed (strategy={strategy})")


# def scene_seed(handler):
#     """Test that scene randomization is reproducible with same seed."""
#     # Create scene randomizer
#     cfg = SceneRandomCfg(
#         floor=SceneGeometryCfg(
#             enabled=True,
#             size=(10.0, 10.0, 0.1),
#             position=(0.0, 0.0, -0.05),
#             material=False,
#         ),
#     )

#     # Test reproducibility
#     randomizer = SceneRandomizer(cfg, seed=789)
#     randomizer.bind_handler(handler)

#     # Store RNG internal state by generating some values
#     randomizer.set_seed(42)
#     val1 = randomizer._rng.random()

#     randomizer.set_seed(42)
#     val2 = randomizer._rng.random()

#     assert val1 == val2, "Same seed should produce same random values"
#     log.info("Scene seed reproducibility test passed")


# def _process_run_handler(scenario):
#     """Process function for standalone mode - creates its own handler."""
#     from metasim.utils.setup_util import get_handler

#     handler = get_handler(scenario)
#     scene_seed(handler)

#     # Test creation without material randomization
#     scene_floor_creation(handler, material=False)
#     scene_walls_creation(handler, material=False)
#     scene_ceiling_creation(handler, material=False)
#     scene_table_creation(handler, material=False)

#     # Test creation with material randomization
#     scene_floor_creation(handler, material=True)
#     scene_walls_creation(handler, material=True)
#     scene_ceiling_creation(handler, material=True)
#     scene_table_creation(handler, material=True)

#     # Test combined elements and strategies
#     scene_combined_elements(handler)
#     scene_material_selection_strategies(handler, strategy="random")
#     scene_material_selection_strategies(handler, strategy="sequential")

#     handler.close()


# def run_test(sim="isaacsim", num_envs=2):
#     """Standalone test function for direct execution."""
#     log.info(
#         f"Running scene randomizer test in standalone mode with {sim} and {num_envs}"
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
# def test_scene_randomizer_with_shared_handler(shared_handler):
#     """Run scene randomizer tests using the child-process handler via proxy."""
#     import inspect
#     import sys

#     log.info("Running scene randomizer tests with shared handler (proxy)")

#     proxy = shared_handler  # HandlerProxy

#     module = "metasim.test.randomization.test_scene_randomizer"

#     # Dynamically get scene creation functions that accept material parameter
#     scene_creation_functions = [
#         name
#         for name, obj in inspect.getmembers(sys.modules[__name__], inspect.isfunction)
#         if name.startswith("scene_") and name.endswith("_creation")
#     ]

#     # Get other scene test functions
#     scene_strategy_functions = [
#         name
#         for name, obj in inspect.getmembers(sys.modules[__name__], inspect.isfunction)
#         if name == "scene_material_selection_strategies"
#     ]

#     scene_combined_functions = [
#         name
#         for name, obj in inspect.getmembers(sys.modules[__name__], inspect.isfunction)
#         if name == "scene_combined_elements"
#     ]

#     # Call seed reproducibility test first
#     proxy.run_test(
#         "scene_seed",
#         module=module,
#     )

#     # Test creation without material randomization
#     for func_name in scene_creation_functions:
#         proxy.run_test(func_name, module=module, material=False)

#     # Test creation with material randomization
#     for func_name in scene_creation_functions:
#         proxy.run_test(func_name, module=module, material=True)

#     # Test combined elements
#     for func_name in scene_combined_functions:
#         proxy.run_test(func_name, module=module)

#     # Test material selection strategies
#     for func_name in scene_strategy_functions:
#         proxy.run_test(func_name, module=module, strategy="random")
#         proxy.run_test(func_name, module=module, strategy="sequential")

#     log.info("All scene randomizer tests completed with shared handler (proxy)")


# if __name__ == "__main__":
#     # Direct execution for quick testing - uses standalone mode
#     import sys

#     sim = "isaacsim" if len(sys.argv) < 2 else sys.argv[1]
#     num_envs = 2 if len(sys.argv) < 3 else int(sys.argv[2])
#     run_test(sim, num_envs)
