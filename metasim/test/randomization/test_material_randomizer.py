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


def material_physical(handler, distribution="uniform", common_range=(1e-8, 1.0)):
    """Test physical material (friction, restitution) randomization."""
    from metasim.randomization.material_randomizer import PhysicalMaterialCfg

    # Create material randomizer with physical properties
    cfg = MaterialRandomCfg(
        obj_name="cube",
        physical=PhysicalMaterialCfg(
            friction_range=common_range,
            restitution_range=common_range,
            distribution=distribution,
            enabled=True,
        ),
    )

    randomizer = MaterialRandomizer(cfg, seed=789)
    randomizer.bind_handler(handler)

    material_prop = _get_physical_properties(handler, cfg.obj_name)
    # Apply randomization
    randomizer()
    new_material_prop = _get_physical_properties(handler, cfg.obj_name)

    for k, v in material_prop.items():
        cur_val = v
        new_val = new_material_prop[k]
        assert (cur_val != new_val).any(), f"{k} should be randomized"
        assert (new_val >= common_range[0]).all() and (new_val <= common_range[1]).all(), f"{k} out of range"
        "Restitution out of range"

    # For physical properties, we can check that the randomizer was called successfully
    # The actual physics properties are internal to the simulation
    log.info(f"Physical material randomization (Type: {distribution}) test passed")


def material_pbr(handler, distribution="uniform", common_range=(1e-8, 1.0)):
    """Test PBR material (roughness, metallic) randomization."""
    from metasim.randomization.material_randomizer import PBRMaterialCfg

    # Create material randomizer with PBR properties
    cfg = MaterialRandomCfg(
        obj_name="cube",
        pbr=PBRMaterialCfg(
            roughness_range=common_range,
            metallic_range=common_range,
            specular_range=common_range,
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
        assert common_range[0] <= new_pbr["roughness"][i] <= common_range[1], (
            f"Roughness {new_pbr['roughness'][i]} out of range {common_range}"
        )
        assert common_range[0] <= new_pbr["metallic"][i] <= common_range[1], (
            f"Metallic {new_pbr['metallic'][i]} out of range {common_range}"
        )
        assert common_range[0] <= new_pbr["specular"][i] <= common_range[1], (
            f"Specular {new_pbr['specular'][i]} out of range {common_range}"
        )

        # Validate diffuse color components
        diffuse = new_pbr["diffuseColor"][i]
        assert common_range[0] <= diffuse[0] <= 1.0, f"Diffuse R {diffuse[0]} out of range {common_range}"
        assert common_range[0] <= diffuse[1] <= 1.0, f"Diffuse G {diffuse[1]} out of range {common_range}"
        assert common_range[0] <= diffuse[2] <= 1.0, f"Diffuse B {diffuse[2]} out of range {common_range}"

    log.info(f"PBR material randomization (Type: {distribution}) test passed")


def material_mdl(handler, distribution="uniform", common_range=(1e-8, 1.0)):
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

    obj_inst = _get_object_instance(handler, cfg.obj_name)
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


def material_multi_objects(handler, distribution="uniform", common_range=(1e-8, 1.0)):
    """Test randomization across multiple distinct objects (cube & sphere).

    Ensures each object's physical properties are randomized independently.
    """
    from metasim.randomization.material_randomizer import PhysicalMaterialCfg

    # Cube randomizer
    cube_cfg = MaterialRandomCfg(
        obj_name="cube",
        physical=PhysicalMaterialCfg(
            friction_range=common_range,
            restitution_range=common_range,
            distribution=distribution,
            enabled=True,
        ),
    )
    cube_rand = MaterialRandomizer(cube_cfg, seed=789)
    cube_rand.bind_handler(handler)
    cube_before = _get_physical_properties(handler, cube_cfg.obj_name)

    # Sphere randomizer
    sphere_cfg = MaterialRandomCfg(
        obj_name="sphere",
        physical=PhysicalMaterialCfg(
            friction_range=common_range,
            restitution_range=common_range,
            distribution=distribution,
            enabled=True,
        ),
    )
    sphere_rand = MaterialRandomizer(sphere_cfg, seed=789)
    sphere_rand.bind_handler(handler)
    sphere_before = _get_physical_properties(handler, sphere_cfg.obj_name)

    # Apply both randomizations
    cube_rand()
    sphere_rand()

    cube_after = _get_physical_properties(handler, cube_cfg.obj_name)
    sphere_after = _get_physical_properties(handler, sphere_cfg.obj_name)

    # Assertions per object
    for k, v in cube_before.items():
        cur_val = v
        new_val = cube_after[k]
        assert (cur_val != new_val).any(), f"Cube {k} should be randomized"
    for k, v in sphere_before.items():
        cur_val = v
        new_val = sphere_after[k]
        assert (cur_val != new_val).any(), f"Sphere {k} should be randomized"

    log.info("Multi-object material randomization test passed")


def material_envid(handler, distribution="uniform", common_range=(1e-8, 1.0)):
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
            friction_range=common_range,
            restitution_range=common_range,
            distribution=distribution,
            enabled=True,
        ),
        env_ids=[0],  # only randomize env 0
    )

    rand = MaterialRandomizer(cfg, seed=789)
    rand.bind_handler(handler)
    before = _get_physical_properties(handler, cfg.obj_name)
    rand()
    after = _get_physical_properties(handler, cfg.obj_name)
    for k, v in before.items():
        val_before = v
        val_after = after[k]
        assert val_before[0] != val_after[0], "Env 0 material properties should change"
        assert (val_before[1:] == val_after[1:]).all(), "Other env material properties should remain same"

    log.info("Environment ID filtered material randomization test passed")


def material_seed(handler, distribution="uniform", common_range=(1e-8, 1.0)):
    """Test that material randomization is reproducible with same seed."""
    from metasim.randomization.material_randomizer import PhysicalMaterialCfg

    # Create material randomizer with physical properties
    cfg = MaterialRandomCfg(
        obj_name="cube",
        physical=PhysicalMaterialCfg(
            friction_range=common_range,
            restitution_range=common_range,
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
    after1 = _get_physical_properties(handler, cfg.obj_name)

    randomizer.set_seed(42)
    randomizer()
    after2 = _get_physical_properties(handler, cfg.obj_name)

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


@pytest.mark.isaacsim
@pytest.mark.parametrize("distribution", ["uniform", "log_uniform", "gaussian"])
@pytest.mark.parametrize("test_func", TEST_FUNCTIONS, ids=[f.__name__ for f in TEST_FUNCTIONS])
def test_material_randomizers(handler, test_func, distribution):
    """Run material randomizer checks inside the shared handler process."""
    common_range = (1e-8, 1.0)
    test_func(handler, distribution=distribution, common_range=common_range)


def _get_object_instance(handler, obj_name):
    """Helper to get object instance from handler."""
    if hasattr(handler, "scene"):
        if obj_name in handler.scene.articulations:
            return handler.scene.articulations[obj_name]
        elif obj_name in handler.scene.rigid_objects:
            return handler.scene.rigid_objects[obj_name]
    raise ValueError(f"Object {obj_name} not found in handler")


def _get_physical_properties(handler, obj_name):
    """Helper to get physical properties from object."""
    obj_inst = _get_object_instance(handler, obj_name)
    materials = obj_inst.root_physx_view.get_material_properties()
    # materials shape: [num_envs, num_bodies, 3] or [num_envs, 3]
    # index 0=static friction, 1=dynamic friction, 2=restitution

    return {"friction": materials[..., 0].detach().clone(), "restitution": materials[..., 2].detach().clone()}


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
            obj_inst = _get_object_instance(randomizer.handler, randomizer.cfg.obj_name)
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


if __name__ == "__main__":
    pytest.main([__file__, "-k isaacsim"])
