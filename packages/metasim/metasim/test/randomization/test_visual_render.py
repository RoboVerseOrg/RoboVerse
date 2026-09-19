"""Real-render validation of recipes; requires the mapped Blender/Isaac Sim environment."""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from metasim.randomization import (
    EnvironmentRandomCfg,
    LightingRandomCfg,
    SensorRandomCfg,
    SurfaceRandomCfg,
    TextureSetCfg,
    ViewRandomCfg,
    VisualRandomizationCfg,
    VisualRandomizer,
    color_temperature_to_rgb,
)

pytestmark = pytest.mark.sim("blender", "isaacsim")


def _resources(handler):
    if handler.scenario.simulator == "blender":
        import bpy

        return (len(bpy.data.materials), len(bpy.data.images), len(bpy.data.objects))
    import omni.usd

    return (sum(1 for _ in omni.usd.get_context().get_stage().Traverse()),)


def _texture_set(directory):
    """A tiny checker albedo and packed ORM map, so the test needs no external assets."""
    import imageio.v2 as imageio

    checker = ((np.indices((8, 8)).sum(axis=0) % 2) * 255).astype(np.uint8)
    base = np.stack([checker, 255 - checker, np.full_like(checker, 64)], axis=-1)
    orm = np.stack([np.full_like(checker, 255), np.full_like(checker, 200), np.zeros_like(checker)], axis=-1)
    imageio.imwrite(directory / "base_color.png", base)
    imageio.imwrite(directory / "orm.png", orm)
    return TextureSetCfg(base_color=str(directory / "base_color.png"), orm=str(directory / "orm.png"))


def _legacy(recipe):
    """The same appearance as a schema-1 writer would have saved it."""
    legacy = copy.deepcopy(recipe)
    legacy.schema_version = 1
    legacy.sensors = {}
    for values in legacy.materials.values():
        for name in ("uv_scale", "uv_rotation", "ior", "uv_projection"):
            del values[name]
        values["textures"] = dict.fromkeys(("base_color", "roughness", "metallic", "normal"))
    for values in legacy.lights.values():
        del values["color_temperature"]
    for values in legacy.cameras.values():
        del values["f_stop"]
        del values["focus_scale"]
    return legacy


def test_recipes_change_pixels_without_drift_physics_steps_or_resource_growth(handler, tmp_path):
    textures = _texture_set(tmp_path)
    sampler = VisualRandomizer(
        VisualRandomizationCfg(
            materials={
                "cube": SurfaceRandomCfg(
                    ior=(1.3, 1.8),
                    uv_scale=((1, 3), (1, 3)),
                    uv_rotation=(-1, 1),
                    textures=(textures,),
                    uv_projection="box",
                )
            },
            lights={"key": LightingRandomCfg(color_temperature=(3000, 3000))},
            cameras={"view": ViewRandomCfg(f_stop=(4, 4))},
            environment=EnvironmentRandomCfg(),
            sensors={"view": SensorRandomCfg()},
        ),
        seed=0,
    ).bind_handler(handler)
    recipe = sampler.sample(sample_id=0)
    sampler.apply(recipe)
    warm = color_temperature_to_rgb(3000)
    if handler.scenario.simulator == "blender":
        import bpy

        assert "metasim_ground" not in bpy.data.objects
        assert "metasim_table" not in bpy.data.objects
        tree = sampler._adapter.materials["cube"].node_tree
        surface = tree.nodes["Surface"]
        surface_pointer = surface.as_pointer()
        assert surface.inputs["IOR"].default_value == pytest.approx(recipe.materials["cube"]["ior"])
        assert surface.inputs["Roughness"].is_linked and surface.inputs["Metallic"].is_linked
        assert tree.nodes["Tex_orm"].image.colorspace_settings.name == "Non-Color"
        layer = bpy.data.objects["cube"].data.uv_layers["metasim_box"]
        assert layer.active_render
        assert max(abs(v) for loop in layer.data for v in loop.uv) == pytest.approx(0.2, abs=1e-6)
        camera = bpy.data.objects["view"].data
        assert camera.dof.use_dof and camera.dof.aperture_fstop == pytest.approx(4)
        assert tuple(bpy.data.objects["key"].data.color) == pytest.approx(warm, abs=1e-6)
    else:
        from pxr import UsdGeom, UsdLux, UsdShade

        stage = sampler._adapter.stage
        assert not stage.GetPrimAtPath("/World/ground")
        prim = stage.GetPrimAtPath("/World/envs/env_0/cube")
        geometry = sampler._adapter._geometry(prim)[0]
        material = UsdShade.MaterialBindingAPI(geometry).ComputeBoundMaterial()[0]
        surface = UsdShade.Shader(stage.GetPrimAtPath(str(material.GetPath()) + "/Surface"))
        transform = UsdShade.Shader(stage.GetPrimAtPath(str(material.GetPath()) + "/UVTransform"))
        assert surface.GetInput("ior").Get() == pytest.approx(recipe.materials["cube"]["ior"])
        assert tuple(transform.GetInput("scale").Get()) == pytest.approx(recipe.materials["cube"]["uv_scale"])
        assert surface.GetInput("roughness").GetConnectedSource()[1] == "g"
        assert surface.GetInput("metallic").GetConnectedSource()[1] == "b"
        st = UsdGeom.PrimvarsAPI(geometry).GetPrimvar("st")
        assert st.GetInterpolation() == UsdGeom.Tokens.faceVarying
        assert np.abs(np.asarray(st.Get())).max() == pytest.approx(0.2, abs=1e-6)
        from metasim.randomization.core.visual_adapter import ISAAC_FSTOP_SCALE

        camera = UsdGeom.Camera(stage.GetPrimAtPath("/World/envs/env_0/view"))
        assert camera.GetFStopAttr().Get() == pytest.approx(4 * ISAAC_FSTOP_SCALE)
        # The authored physical focal length must survive focal augmentation, or the aperture collapses.
        assert camera.GetFocalLengthAttr().Get() == pytest.approx(
            handler.cameras[0].focal_length * recipe.cameras["view"]["focal_scale"]
        )
        assert tuple(UsdLux.LightAPI(stage.GetPrimAtPath("/World/key")).GetColorAttr().Get()) == pytest.approx(
            warm, abs=1e-6
        )
    first = copy.deepcopy(handler.get_states(mode="tensor"))
    expected_intrinsic = torch.tensor(handler.cameras[0].intrinsics, device=first.cameras["view"].intrinsics.device)
    expected_intrinsic[0, 0] *= recipe.cameras["view"]["focal_scale"]
    expected_intrinsic[1, 1] *= recipe.cameras["view"]["focal_scale"]
    assert torch.allclose(first.cameras["view"].intrinsics, expected_intrinsic[None], atol=1e-5)
    position = first.cameras["view"].pos
    expected = torch.tensor(handler.cameras[0].pos, device=position.device) + torch.tensor(
        recipe.cameras["view"]["position_delta"], device=position.device
    )
    if handler.scenario.simulator == "isaacsim":
        expected = expected + handler.scene.env_origins
    assert torch.allclose(position, expected, atol=1e-6)
    capture = recipe.apply_sensor("view", first.cameras["view"].rgb[0], first.cameras["view"].intrinsics[0])
    assert capture.rgb.shape == tuple(first.cameras["view"].rgb[0].shape) and capture.rgb.dtype == np.uint8
    assert capture.intrinsics[0][0] >= float(first.cameras["view"].intrinsics[0, 0, 0]) - 1e-6
    assert capture.distortion[:4] == recipe.sensors["view"]["distortion"]
    resources = _resources(handler)
    for sample_id in range(1, 5):
        sampler.apply(sampler.sample(sample_id=sample_id))
    different = handler.get_states(mode="tensor")
    assert not torch.equal(first.cameras["view"].rgb, different.cameras["view"].rgb)
    sampler.apply(recipe, render=False)
    restored = handler.get_states(mode="tensor")
    assert torch.allclose(first.cameras["view"].pos, restored.cameras["view"].pos, atol=1e-6)
    assert torch.allclose(first.cameras["view"].intrinsics, restored.cameras["view"].intrinsics, atol=1e-5)
    for name in first.objects:
        assert torch.equal(first.objects[name].root_state, restored.objects[name].root_state)
    assert _resources(handler) == resources
    if handler.scenario.simulator == "blender":
        assert sampler._adapter.materials["cube"].node_tree.nodes["Surface"].as_pointer() == surface_pointer
    assert restored.cameras["view"].rgb.float().std() > 3
    # Path-tracing noise may differ: exact reproducibility is a parameter contract.
    assert (restored.cameras["view"].rgb.float() - first.cameras["view"].rgb.float()).abs().mean() < 10
    invalid = copy.deepcopy(recipe)
    invalid.materials["missing"] = invalid.materials.pop("cube")
    with pytest.raises(ValueError, match="target"):
        sampler.apply(invalid)
    assert _resources(handler) == resources
    if handler.num_envs > 1:
        # Local material/camera edits must leave other environments untouched.
        from pxr import UsdShade

        stage = sampler._adapter.stage
        # Cloned envs inherit env_0: a first-ever subset edit on env 0 must not leak to env 1.
        block = sampler.sample(sample_id=30)
        block.lights, block.environment, block.cameras = {}, None, {}
        block.materials = {"block": block.materials.pop("cube")}
        blocks = [sampler._adapter._geometry(stage.GetPrimAtPath(f"/World/envs/env_{i}/block"))[0] for i in (0, 1)]
        untouched_block = UsdShade.MaterialBindingAPI(blocks[1]).ComputeBoundMaterial()[0].GetPath()
        sampler.apply(block, env_ids=[0])
        assert UsdShade.MaterialBindingAPI(blocks[1]).ComputeBoundMaterial()[0].GetPath() == untouched_block
        assert UsdShade.MaterialBindingAPI(blocks[0]).ComputeBoundMaterial()[0].GetPath() != untouched_block
        untouched = stage.GetPrimAtPath("/World/envs/env_1/cube")
        env1_geometry = sampler._adapter._geometry(untouched)[0]
        original_binding = UsdShade.MaterialBindingAPI(env1_geometry).ComputeBoundMaterial()[0].GetPath()
        subset = sampler.sample(sample_id=20)
        subset.lights = {}
        subset.environment = None
        before = copy.deepcopy(restored.cameras["view"])
        sampler.apply(subset, env_ids=[0])
        after = handler.get_states(mode="tensor").cameras["view"]
        assert not torch.equal(before.pos[0], after.pos[0])
        assert not torch.equal(before.intrinsics[0], after.intrinsics[0])
        assert torch.equal(before.pos[1:], after.pos[1:])
        assert torch.equal(before.intrinsics[1:], after.intrinsics[1:])
        assert UsdShade.MaterialBindingAPI(env1_geometry).ComputeBoundMaterial()[0].GetPath() == original_binding
        sampler.apply(recipe)
    sampler.apply(_legacy(recipe))
    if handler.scenario.simulator == "blender":
        import bpy

        assert sampler._adapter.materials["cube"].node_tree.nodes["Surface"].inputs["IOR"].default_value == 1.5
        assert not bpy.data.objects["view"].data.dof.use_dof
        # Box projection is undone: the primitive's authored UV map renders again.
        assert bpy.data.objects["cube"].data.uv_layers.active.name != "metasim_box"
        assert not bpy.data.objects["cube"].data.uv_layers["metasim_box"].active_render
    else:
        from pxr import UsdGeom

        assert surface.GetInput("ior").Get() == 1.5
        assert camera.GetFStopAttr().Get() == 0
        # The mesh cuboid never had st: box projection is removed, so mesh-UV textures are rejected again.
        assert not UsdGeom.PrimvarsAPI(geometry).GetPrimvar("st")
        mesh_uv = copy.deepcopy(recipe)
        mesh_uv.materials["cube"]["uv_projection"] = "mesh"
        with pytest.raises(ValueError, match="st UV primvar"):
            sampler.apply(mesh_uv)
