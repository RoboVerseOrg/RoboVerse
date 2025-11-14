"""Test material randomizer functionality."""

from __future__ import annotations

from typing import Any

import pytest
import rootutils
from loguru import logger as log

rootutils.setup_root(__file__, pythonpath=True)

from metasim.randomization.material_randomizer import (
    MaterialRandomCfg,
    MaterialRandomizer,
)
from metasim.test.randomization.conftest import get_shared_scenario


def material_physical(handler, distribution="uniform"):
    """Test physical material (friction, restitution) randomization."""
    from metasim.randomization.material_randomizer import PhysicalMaterialCfg

    # Create material randomizer with physical properties
    cfg = MaterialRandomCfg(
        obj_name="cube",
        physical=PhysicalMaterialCfg(
            friction_range=(0.1, 1.0),
            restitution_range=(0.1, 1.0),
            distribution=distribution,
            enabled=True,
        ),
    )

    randomizer = MaterialRandomizer(cfg, seed=789)
    randomizer.bind_handler(handler)

    material_prop = randomizer.get_physical_properties()

    current_friction = material_prop["friction"]
    current_restitution = material_prop["restitution"]
    # Apply randomization
    randomizer()

    new_material_prop = randomizer.get_physical_properties()
    new_friction = new_material_prop["friction"]
    new_restitution = new_material_prop["restitution"]

    assert (current_friction != new_friction).all(), "Friction should be randomized"
    assert (current_restitution != new_restitution).all(), "Restitution should be randomized"
    assert (new_friction >= 0.1).all() and (new_friction <= 1.0).all(), "Friction out 1of range"
    assert (new_restitution >= 0.1).all() and (new_restitution <= 1.0).all(), "Restitution out of range"
    # For physical properties, we can check that the randomizer was called successfully
    # The actual physics properties are internal to the simulation
    log.info(f"Physical material randomization (Type: {distribution}) test passed")


def _get_pbr_properties(randomizer: MaterialRandomizer) -> dict[str, Any]:
    """Extract current PBR shader parameters for all environments."""
    if not randomizer.cfg.pbr:
        return {}

    try:
        import omni  # type: ignore[import-not-found]

        try:
            import omni.isaac.core.utils.prims as prim_utils  # type: ignore[import-not-found]
        except ModuleNotFoundError:
            import isaacsim.core.utils.prims as prim_utils  # type: ignore[import-not-found]

        from pxr import UsdShade  # type: ignore[import-not-found]
    except ImportError:
        return {}

    num_envs = randomizer.handler._num_envs
    all_properties: dict[str, list] = {
        "roughness": [],
        "metallic": [],
        "specular": [],
        "diffuseColor": [],
    }

    for env_id in range(num_envs):
        try:
            obj_inst = randomizer._get_object_instance(randomizer.cfg.obj_name)
        except Exception:
            continue

        prim_path = obj_inst.cfg.prim_path.replace("env_.*", f"env_{env_id}")
        prim = prim_utils.get_prim_at_path(prim_path)
        if prim is None:
            continue

        material_binding = UsdShade.MaterialBindingAPI(prim)
        bound_material = material_binding.ComputeBoundMaterial()
        if not bound_material:
            continue

        material = bound_material[0]
        shader_prim = omni.usd.get_shader_from_material(material, get_prim=True)
        if not shader_prim:
            continue
        shader = UsdShade.Shader(shader_prim)
        if not shader:
            continue

        for prop in ("roughness", "metallic", "specular"):
            shader_input = shader.GetInput(prop)
            if shader_input:
                all_properties[prop].append(shader_input.Get())

        diffuse_input = shader.GetInput("diffuseColor")
        if diffuse_input:
            color = diffuse_input.Get()
            if color is not None:
                all_properties["diffuseColor"].append((
                    float(color[0]),
                    float(color[1]),
                    float(color[2]),
                ))

    # Filter out empty properties
    return {k: v for k, v in all_properties.items() if v}


def material_pbr(handler, distribution="uniform"):
    """Test PBR material (roughness, metallic) randomization."""
    from metasim.randomization.material_randomizer import PBRMaterialCfg

    # Create material randomizer with PBR properties
    cfg = MaterialRandomCfg(
        obj_name="cube",
        pbr=PBRMaterialCfg(
            roughness_range=(0.1, 1.0),
            metallic_range=(0.1, 1.0),
            specular_range=(0.1, 1.0),
            diffuse_color_range=((0.5, 1.0), (0.5, 1.0), (0.5, 1.0)),
            distribution=distribution,
            enabled=True,
        ),
    )

    randomizer = MaterialRandomizer(cfg, seed=789)
    randomizer.bind_handler(handler)

    current_pbr = _get_pbr_properties(randomizer)

    # Apply randomization - this creates a new material with randomized properties
    randomizer()

    # Verify PBR properties were set after randomization
    new_pbr = _get_pbr_properties(randomizer)

    assert new_pbr, "PBR properties should be populated after randomization"
    assert new_pbr != current_pbr, "PBR properties should change after randomization"

    num_envs = randomizer.handler._num_envs
    for i in range(num_envs):
        # Validate that properties exist and are within expected ranges
        assert 0.1 <= new_pbr["roughness"][i] <= 1.0, f"Roughness {new_pbr['roughness'][i]} out of range [0.1, 1.0 ]"
        assert 0.1 <= new_pbr["metallic"][i] <= 1.0, f"Metallic {new_pbr['metallic'][i]} out of range [0.1, 1.0]"
        assert 0.1 <= new_pbr["specular"][i] <= 1.0, f"Specular {new_pbr['specular'][i]} out of range [0.1, 1.0]"

        # Validate diffuse color components
        diffuse = new_pbr["diffuseColor"][i]
        assert 0.5 <= diffuse[0] <= 1.0, f"Diffuse R {diffuse[0]} out of range [0.5, 1.0]"
        assert 0.5 <= diffuse[1] <= 1.0, f"Diffuse G {diffuse[1]} out of range [0.5, 1.0]"
        assert 0.5 <= diffuse[2] <= 1.0, f"Diffuse B {diffuse[2]} out of range [0.5, 1.0]"

    log.info(f"PBR material randomization (Type: {distribution}) test passed")


def material_mdl(handler, distribution="uniform"):
    """Test MDL material application and reproducibility basics.

    Validates that an MDL file can be applied to the target object's prims
    without raising errors, and that a material binding exists afterwards
    for each environment. Does not deeply inspect shader internals (which
    depend on IsaacSim runtime), but ensures the material prim is bound.
    """
    from metasim.randomization.material_randomizer import MDLMaterialCfg

    # Use existing MDL asset present in repository
    mdl_path = "roboverse_data/materials/arnold/Wood/Ash.mdl"

    cfg = MaterialRandomCfg(
        obj_name="cube",  # apply to cube
        mdl=MDLMaterialCfg(
            mdl_paths=[mdl_path],
            selection_strategy="random",
            randomize_material_variant=True,
            enabled=True,
            auto_download=False,  # asset exists locally
            validate_paths=True,
        ),
    )

    randomizer = MaterialRandomizer(cfg, seed=789)
    randomizer.bind_handler(handler)

    # Before randomization, capture whether any material is bound (for info only)
    try:
        try:
            import omni.isaac.core.utils.prims as prim_utils  # type: ignore[import-not-found]
        except ModuleNotFoundError:
            import isaacsim.core.utils.prims as prim_utils  # type: ignore[import-not-found]
        from pxr import UsdShade  # type: ignore[import-not-found]
    except ImportError:
        # If simulation-specific modules are unavailable, skip MDL test gracefully.
        log.warning("Skipping MDL material test: simulation modules unavailable")
        return

    obj_inst = randomizer._get_object_instance(randomizer.cfg.obj_name)
    prim_path_template = obj_inst.cfg.prim_path

    bound_before: list = []
    for env_id in range(handler._num_envs):  # type: ignore[attr-defined]
        prim_path = prim_path_template.replace("env_.*", f"env_{env_id}")
        prim = prim_utils.get_prim_at_path(prim_path)
        if prim is None:
            continue
        binding_api = UsdShade.MaterialBindingAPI(prim)
        bound_material = binding_api.ComputeBoundMaterial()
        if bound_material:
            bound_before.append(bound_material[0].GetPath().pathString)

    # Apply MDL randomization
    randomizer()

    bound_after: list = []
    for env_id in range(handler._num_envs):  # type: ignore[attr-defined]
        prim_path = prim_path_template.replace("env_.*", f"env_{env_id}")
        prim = prim_utils.get_prim_at_path(prim_path)
        if prim is None:
            continue
        binding_api = UsdShade.MaterialBindingAPI(prim)
        bound_material = binding_api.ComputeBoundMaterial()
        if bound_material:
            bound_after.append(bound_material[0].GetPath().pathString)

    assert bound_after, "MDL material should be bound after randomization"
    assert len(bound_after) == handler._num_envs, "MDL should bind one material per env"  # type: ignore[attr-defined]
    log.info("MDL material randomization test passed")


def material_multi_objects(handler, distribution="uniform"):
    """Test randomization across multiple distinct objects (cube & sphere).

    Ensures each object's physical properties are randomized independently.
    """
    from metasim.randomization.material_randomizer import PhysicalMaterialCfg

    # Cube randomizer
    cube_cfg = MaterialRandomCfg(
        obj_name="cube",
        physical=PhysicalMaterialCfg(
            friction_range=(0.1, 1.0),
            restitution_range=(0.1, 1.0),
            distribution=distribution,
            enabled=True,
        ),
    )
    cube_rand = MaterialRandomizer(cube_cfg, seed=111)
    cube_rand.bind_handler(handler)
    cube_before = cube_rand.get_physical_properties()

    # Sphere randomizer
    sphere_cfg = MaterialRandomCfg(
        obj_name="sphere",
        physical=PhysicalMaterialCfg(
            friction_range=(0.1, 1.0),
            restitution_range=(0.1, 1.0),
            distribution=distribution,
            enabled=True,
        ),
    )
    sphere_rand = MaterialRandomizer(sphere_cfg, seed=222)
    sphere_rand.bind_handler(handler)
    sphere_before = sphere_rand.get_physical_properties()

    # Apply both randomizations
    cube_rand()
    sphere_rand()

    cube_after = cube_rand.get_physical_properties()
    sphere_after = sphere_rand.get_physical_properties()

    # Assertions per object
    assert (cube_before["friction"] != cube_after["friction"]).any(), "Cube friction should change"
    assert (sphere_before["friction"] != sphere_after["friction"]).any(), "Sphere friction should change"
    assert (cube_before["restitution"] != cube_after["restitution"]).any(), "Cube restitution should change"
    assert (sphere_before["restitution"] != sphere_after["restitution"]).any(), "Sphere restitution should change"

    log.info("Multi-object material randomization test passed")


def material_envid(handler, distribution="uniform"):
    """Test env_ids filtering: only specified envs should change.

    Uses two environments; randomizes only env 0 and verifies env 1 unchanged.
    """
    from metasim.randomization.material_randomizer import PhysicalMaterialCfg

    # Guard: ensure we have at least 2 envs
    num_envs = handler._num_envs  # type: ignore[attr-defined]
    if num_envs < 2:
        log.warning("Skipping env_id test: requires >=2 environments")
        return

    cfg = MaterialRandomCfg(
        obj_name="cube",
        physical=PhysicalMaterialCfg(
            friction_range=(0.1, 1.0),
            restitution_range=(0.1, 1.0),
            distribution=distribution,
            enabled=True,
        ),
        env_ids=[0],  # only randomize env 0
    )

    rand = MaterialRandomizer(cfg, seed=333)
    rand.bind_handler(handler)
    before = rand._get_object_instance(rand.cfg.obj_name).root_physx_view.get_material_properties().clone()
    rand()
    after = rand._get_object_instance(rand.cfg.obj_name).root_physx_view.get_material_properties().clone()
    # Env 0 should differ
    assert before[0, :, 0] != after[0, :, 0], "Env 0 friction should change"
    assert before[0, :, 2] != after[0, :, 2], "Env 0 restitution should change"

    # Other envs should remain same
    for env_id in range(1, num_envs):
        assert before[env_id, :, 0] == after[env_id, :, 0], f"Env {env_id} friction should remain"
        assert before[env_id, :, 2] == after[env_id, :, 2], f"Env {env_id} restitution should remain"

    log.info("Environment ID filtered material randomization test passed")


def material_seed(handler, distribution="uniform"):
    """Test that material randomization is reproducible with same seed."""
    from metasim.randomization.material_randomizer import PhysicalMaterialCfg

    # Create material randomizer with physical properties
    cfg = MaterialRandomCfg(
        obj_name="cube",
        physical=PhysicalMaterialCfg(
            friction_range=(0.1, 1.0),
            restitution_range=(0.1, 1.0),
            distribution=distribution,
            enabled=True,
        ),
    )

    # Test reproducibility
    randomizer = MaterialRandomizer(cfg, seed=789)
    randomizer.bind_handler(handler)
    # Apply randomization twice with same seed - should give same results
    randomizer.set_seed(42)
    randomizer()
    after1 = randomizer.get_physical_properties()

    randomizer.set_seed(42)
    randomizer()
    after2 = randomizer.get_physical_properties()

    for k, v in after1.items():
        val1 = v
        val2 = after2[k]
        assert (val1 == val2).all(), f"Same seed should produce same random values for {k}"

    log.info("Material seed reproducibility test passed")


TEST_FUNCTIONS = [
    material_physical,
    material_pbr,
    material_mdl,
    material_multi_objects,
    material_envid,
    material_seed,
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

    log.info(f"Running material randomizer test in standalone mode with {sim} and {num_envs}")

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
def test_material_randomizer_with_shared_handler(shared_handler):
    """Run material randomizer tests using the child-process handler via proxy."""
    log.info("Running material randomizer tests with shared handler (proxy)")

    proxy = shared_handler  # HandlerProxy

    distributions = ["uniform", "log_uniform", "gaussian"]

    # Run all material test functions with different distributions
    for dist in distributions:
        for test_func in TEST_FUNCTIONS:
            proxy.run_test(func=test_func, distribution=dist)

    log.info("All material randomizer tests completed with shared handler (proxy)")


if __name__ == "__main__":
    # Direct execution for quick testing - uses standalone mode
    import sys

    sim = "isaacsim" if len(sys.argv) < 2 else sys.argv[1]
    num_envs = 2 if len(sys.argv) < 3 else int(sys.argv[2])
    run_test(sim, num_envs)
