from pathlib import Path

from roboverse_pack.blender.usd.material_graph.adapters import convert_material, select_adapter_by_policy
from roboverse_pack.blender.usd.material_graph.adapters.mdl_bake import choose_conversion_policy
from roboverse_pack.blender.usd.material_graph.context import MaterialContext
from roboverse_pack.blender.usd.material_graph.kit_bake import kit_bake_command
from roboverse_pack.blender.usd.material_graph.schema import RawMaterialSpec, freeze_values


def _raw(**overrides):
    values = overrides.pop("values", {})
    material_path = overrides.pop("material_path", "/World/Looks/Mat")
    return RawMaterialSpec(
        material_path=material_path,
        material_name=overrides.pop("material_name", material_path.rsplit("/", 1)[-1]),
        shader_ids=overrides.pop("shader_ids", ("OmniPBR",)),
        connected_surface_shader_id=overrides.pop("connected_surface_shader_id", "OmniPBR"),
        mdl_source_asset=overrides.pop("mdl_source_asset", None),
        values=freeze_values(values),
    )


def _context(raw: RawMaterialSpec, *, enable_mdl_bake: bool = False) -> MaterialContext:
    return MaterialContext(
        source_path=Path("scene.usda"),
        texture_base_dir=Path("."),
        material_path=raw.material_path,
        enable_mdl_bake=enable_mdl_bake,
    )


def _complex_mdl_raw() -> RawMaterialSpec:
    return _raw(
        shader_ids=("OmniPBR", "ProceduralNoise"),
        connected_surface_shader_id="OmniPBR",
        mdl_source_asset="ComplexLayeredMaterial.mdl",
        values={"noise_scale": 12.0, "BaseColor_Tex": "textures/base.png"},
    )


def test_existing_preview_surface_policy_preserves_source_graph():
    raw = _raw(
        shader_ids=("UsdPreviewSurface",),
        connected_surface_shader_id="UsdPreviewSurface",
    )

    assert choose_conversion_policy(raw) == "preserve_existing_preview"


def test_complex_procedural_mdl_policy_uses_mdl_bake():
    assert choose_conversion_policy(_complex_mdl_raw()) == "mdl_bake"


def test_complex_mdl_with_known_texture_slots_still_uses_mdl_bake():
    raw = _raw(
        shader_ids=("OmniPBR", "ProceduralNoise"),
        connected_surface_shader_id="OmniPBR",
        mdl_source_asset="OmniPBR.mdl",
        values={"BaseColor_Tex": "textures/base.png"},
    )

    assert choose_conversion_policy(raw) == "mdl_bake"


def test_simple_omnipbr_file_texture_policy_uses_direct_graph():
    raw = _raw(values={"BaseColor_Tex": "textures/wood.png"})

    assert choose_conversion_policy(raw) == "direct_graph"


def test_unconnected_omnipbr_helper_scalar_stays_scalar_fallback():
    raw = _raw(
        shader_ids=("OmniPBR",),
        connected_surface_shader_id="DifferentShader",
        mdl_source_asset="OmniPBR.mdl",
        values={"roughness": 0.5},
    )

    assert choose_conversion_policy(raw) == "scalar_fallback"

    spec = convert_material(raw, _context(raw))

    assert spec.adapter_name == "scalar_fallback"
    assert spec.conversion_policy == "scalar_fallback"


def test_generic_file_texture_policy_still_uses_direct_graph():
    raw = _raw(
        shader_ids=("UnknownShader",),
        connected_surface_shader_id="UnknownShader",
        values={"diffuse_texture": "textures/diffuse.png"},
    )

    assert choose_conversion_policy(raw) == "direct_graph"

    spec = convert_material(raw, _context(raw))

    assert spec.adapter_name == "generic_texture_graph"
    assert spec.conversion_policy == "direct_graph"


def test_disabled_mdl_bake_policy_selects_fallback_adapter():
    adapter = select_adapter_by_policy(_complex_mdl_raw(), enable_mdl_bake=False)

    assert adapter.name in {"scalar_fallback", "semantic_class_fallback"}


def test_convert_material_notes_when_mdl_bake_policy_is_unavailable():
    raw = _complex_mdl_raw()

    spec = convert_material(raw, _context(raw, enable_mdl_bake=False))

    assert spec.adapter_name in {"scalar_fallback", "semantic_class_fallback"}
    assert spec.conversion_policy in {"scalar_fallback", "class_fallback"}
    assert "mdl_bake_unavailable" in spec.quality_notes


def test_kit_bake_command_shape_stringifies_paths():
    command = kit_bake_command(
        Path("/opt/isaacsim/isaac-sim.sh"),
        Path("/tmp/bake_mdl.py"),
        Path("/tmp/source.usda"),
        "/World/Looks/Mat",
        Path("/tmp/cache"),
    )

    assert command == [
        "/opt/isaacsim/isaac-sim.sh",
        "--no-window",
        "--/app/window/enabled=false",
        "--exec",
        "/tmp/bake_mdl.py",
        "--",
        "--source",
        "/tmp/source.usda",
        "--material",
        "/World/Looks/Mat",
        "--cache-dir",
        "/tmp/cache",
    ]
