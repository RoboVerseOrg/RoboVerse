from pathlib import Path

import pytest

from roboverse_pack.blender.usd.material_graph.adapters import select_adapter
from roboverse_pack.blender.usd.material_graph.adapters.omnipbr import OmniPBRAdapter
from roboverse_pack.blender.usd.material_graph.context import MaterialContext
from roboverse_pack.blender.usd.material_graph.schema import RawMaterialSpec, freeze_values
from roboverse_pack.blender.usd.material_graph.slots import SLOT_REGISTRY


def _raw(**overrides):
    values = overrides.pop("values", {})
    return RawMaterialSpec(
        material_path=overrides.pop("material_path", "/World/Looks/Mat"),
        material_name=overrides.pop("material_name", "Mat"),
        shader_ids=overrides.pop("shader_ids", ("OmniPBR",)),
        connected_surface_shader_id=overrides.pop("connected_surface_shader_id", "OmniPBR"),
        mdl_source_asset=overrides.pop("mdl_source_asset", None),
        values=freeze_values(values),
    )


def _context():
    return MaterialContext(
        source_path=Path("scene.usda"),
        texture_base_dir=Path("."),
        material_path="/World/Looks/Mat",
    )


def test_omnipbr_adapter_scores_omnipbr_shader():
    assert OmniPBRAdapter().score(_raw()) == 80


def test_omnipbr_adapter_does_not_score_unconnected_omnipbr_helper():
    raw = _raw(
        shader_ids=("SomeShader", "OmniPBR"),
        connected_surface_shader_id="SomeShader",
        mdl_source_asset="OmniPBR.mdl",
    )

    assert OmniPBRAdapter().score(raw) == 0


def test_omnipbr_adapter_maps_base_color_texture():
    spec = OmniPBRAdapter().convert(_raw(values={"BaseColor_Tex": "textures/wood.png"}), _context())

    assert spec.base_color.texture is not None
    assert spec.base_color.texture.file == "textures/wood.png"
    assert spec.base_color.texture.source_color_space == "sRGB"
    assert spec.base_color.texture.channel == "rgb"
    assert spec.base_color.texture.source_input == "BaseColor_Tex"
    assert spec.conversion_policy == "direct_graph"


def test_omnipbr_adapter_sets_texture_uv_set_and_transform_from_bindings():
    context = MaterialContext(
        source_path=Path("scene.usda"),
        texture_base_dir=Path("."),
        material_path="/World/Looks/Mat",
        bound_prim_paths=("/World/Mesh",),
        uv_primvars_by_prim={"/World/Mesh": ("st", "st1")},
    )

    spec = OmniPBRAdapter().convert(
        _raw(
            values={
                "BaseColor_Tex": "textures/wood.png",
                "MaxTexCoordIndex": 1,
                "BaseColor_UVA": (2.0, 3.0, 0.25, 0.5),
            }
        ),
        context,
    )

    assert spec.base_color.texture is not None
    assert spec.base_color.texture.uv_set.primvar_name == "st1"
    assert spec.base_color.texture.uv_set.requested_index == 1
    assert spec.base_color.texture.uv_set.resolution_status == "matched"
    assert spec.base_color.texture.uv_transform is not None
    assert spec.base_color.texture.uv_transform.scale == (2.0, 3.0)
    assert spec.base_color.texture.uv_transform.translation == (0.25, 0.5)
    assert spec.base_color.texture.uv_transform.source_input == "BaseColor_UVA"


def test_omnipbr_adapter_prefers_slot_uv_index_over_global_index():
    context = MaterialContext(
        source_path=Path("scene.usda"),
        texture_base_dir=Path("."),
        material_path="/World/Looks/Mat",
        bound_prim_paths=("/World/Mesh",),
        uv_primvars_by_prim={"/World/Mesh": ("st", "st1")},
    )

    spec = OmniPBRAdapter().convert(
        _raw(
            values={
                "BaseColor_Tex": "textures/wood.png",
                "MaxTexCoordIndex": 0,
                "BaseColor_MaxTexCoordIndex": 1,
            }
        ),
        context,
    )

    assert spec.base_color.texture is not None
    assert spec.base_color.texture.uv_set.primvar_name == "st1"
    assert spec.base_color.texture.uv_set.requested_index == 1
    assert spec.base_color.texture.uv_set.resolution_status == "matched"


def test_omnipbr_adapter_does_not_false_match_disjoint_bound_uv_sets():
    context = MaterialContext(
        source_path=Path("scene.usda"),
        texture_base_dir=Path("."),
        material_path="/World/Looks/Mat",
        bound_prim_paths=("/World/MeshA", "/World/MeshB"),
        uv_primvars_by_prim={
            "/World/MeshA": ("st1",),
            "/World/MeshB": ("uv",),
        },
    )

    spec = OmniPBRAdapter().convert(
        _raw(values={"BaseColor_Tex": "textures/wood.png", "BaseColor_MaxTexCoordIndex": 1}),
        context,
    )

    assert spec.base_color.texture is not None
    assert spec.base_color.texture.uv_set.primvar_name == "st"
    assert spec.base_color.texture.uv_set.requested_index == 1
    assert spec.base_color.texture.uv_set.resolution_status == "guessed_or_missing"


def test_omnipbr_adapter_inverts_glossiness_scalar_to_roughness():
    spec = OmniPBRAdapter().convert(_raw(values={"glossiness": 0.25}), _context())

    assert spec.roughness is not None
    assert spec.roughness.value == pytest.approx(0.75)
    assert spec.roughness.source_inputs == ("glossiness",)


def test_omnipbr_adapter_sets_specular_workflow_for_specular_slot():
    spec = OmniPBRAdapter().convert(_raw(values={"specular": (0.2, 0.3, 0.4)}), _context())

    assert spec.specular_color is not None
    assert spec.specular_color.value == (0.2, 0.3, 0.4)
    assert spec.use_specular_workflow is True


def test_omnipbr_adapter_inverts_gloss_color_scalar_to_roughness():
    spec = OmniPBRAdapter().convert(_raw(values={"Gloss_Color": 0.2}), _context())

    assert spec.roughness is not None
    assert spec.roughness.value == pytest.approx(0.8)
    assert spec.roughness.source_inputs == ("Gloss_Color",)


def test_omnipbr_adapter_inverts_gloss_color_tuple_to_roughness():
    spec = OmniPBRAdapter().convert(_raw(values={"Gloss_Color": (0.2, 0.2, 0.2)}), _context())

    assert spec.roughness is not None
    assert spec.roughness.value == pytest.approx(0.8)
    assert spec.roughness.source_inputs == ("Gloss_Color",)


def test_omnipbr_adapter_maps_metallic_color_alias():
    spec = OmniPBRAdapter().convert(_raw(values={"Metallic_Color": 0.65}), _context())

    assert spec.metallic is not None
    assert spec.metallic.value == pytest.approx(0.65)
    assert spec.metallic.source_inputs == ("Metallic_Color",)


def test_omnipbr_adapter_maps_metallic_color_tuple_alias_to_scalar():
    spec = OmniPBRAdapter().convert(_raw(values={"Metallic_Color": (0.65, 0.65, 0.65)}), _context())

    assert spec.metallic is not None
    assert spec.metallic.value == pytest.approx(0.65)
    assert spec.metallic.source_inputs == ("Metallic_Color",)


def test_omnipbr_adapter_maps_roughness_alias():
    spec = OmniPBRAdapter().convert(_raw(values={"Roughness": 0.35}), _context())

    assert spec.roughness is not None
    assert spec.roughness.value == pytest.approx(0.35)
    assert spec.roughness.source_inputs == ("Roughness",)


def test_omnipbr_adapter_maps_alpha_texture_to_opacity_channel():
    spec = OmniPBRAdapter().convert(_raw(values={"Alpha_Tex": "textures/alpha.png"}), _context())

    assert spec.opacity is not None
    assert spec.opacity.texture is not None
    assert spec.opacity.texture.file == "textures/alpha.png"
    assert spec.opacity.texture.channel == "a_or_r"
    assert spec.opacity.texture.source_input == "Alpha_Tex"


def test_omnipbr_adapter_uses_shared_and_planned_ior_aliases():
    glass_ior_spec = OmniPBRAdapter().convert(_raw(values={"glass_ior": 1.52}), _context())
    fresnel_spec = OmniPBRAdapter().convert(_raw(values={"FresnelB": 1.33}), _context())

    assert glass_ior_spec.ior == pytest.approx(1.52)
    assert fresnel_spec.ior == pytest.approx(1.33)


def test_slot_registry_includes_planned_uva_aliases():
    assert {"BaseColor_UVA", "Diffuse_UVA"}.issubset(SLOT_REGISTRY["base_color"].uva_aliases)
    assert "Normal_UVA" in SLOT_REGISTRY["normal"].uva_aliases
    assert "Metallic_UVA" in SLOT_REGISTRY["metallic"].uva_aliases
    assert "Roughness_UVA" in SLOT_REGISTRY["roughness"].uva_aliases
    assert "Gloss_UVA" in SLOT_REGISTRY["glossiness"].uva_aliases
    assert "Specular_UVA" in SLOT_REGISTRY["specular"].uva_aliases
    assert {"Emissive_UVA", "Emission_UVA"}.issubset(SLOT_REGISTRY["emissive"].uva_aliases)
    assert {"Opacity_UVA", "Alpha_UVA"}.issubset(SLOT_REGISTRY["opacity"].uva_aliases)


def test_select_adapter_tie_breaks_by_registration_order():
    class FakeAdapter:
        def __init__(self, name):
            self.name = name

        def score(self, raw):
            return 5

        def convert(self, raw, context):
            raise AssertionError("not used")

    earlier = FakeAdapter("earlier")
    later = FakeAdapter("later")

    assert select_adapter(_raw(), adapters=(earlier, later)) is earlier
