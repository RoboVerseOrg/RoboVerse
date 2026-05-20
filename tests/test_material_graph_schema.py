from dataclasses import FrozenInstanceError

import pytest

from roboverse_pack.blender.usd.material_graph.schema import (
    InputSpec,
    PreviewMaterialSpec,
    RawMaterialSpec,
    TextureSpec,
    freeze_values,
)


def test_texture_spec_defaults_to_rgb_and_default_uv_set():
    texture = TextureSpec(file="textures/wood.png", source_color_space="sRGB")

    assert texture.channel == "rgb"
    assert texture.uv_set.primvar_name == "st"
    assert texture.scale is None
    assert texture.bias is None


def test_preview_material_spec_is_frozen():
    spec = PreviewMaterialSpec(
        material_path="/World/Looks/Paint",
        material_name="Paint",
        source_shader_ids=("OmniPBR",),
        mdl_source_asset=None,
        base_color=InputSpec(value=(0.1, 0.2, 0.3)),
        normal=None,
        metallic=InputSpec(value=0.0),
        roughness=InputSpec(value=0.5),
        specular_color=None,
        emissive_color=None,
        opacity=InputSpec(value=1.0),
        ior=1.45,
    )

    assert spec.emissive_color is None
    assert spec.ior == 1.45
    with pytest.raises(FrozenInstanceError):
        spec.material_name = "Primer"


def test_raw_material_spec_carries_uninterpreted_values():
    token = object()
    raw = RawMaterialSpec(
        material_path="/World/Looks/Raw",
        material_name="Raw",
        shader_ids=("OmniPBR",),
        connected_surface_shader_id="OmniPBR",
        mdl_source_asset=None,
        values=freeze_values({"BaseColor_Color": token}),
    )

    assert raw.values["BaseColor_Color"] is token


def test_freeze_values_returns_immutable_copy():
    values = {"roughness": 0.4}
    frozen = freeze_values(values)
    values["roughness"] = 0.9

    assert frozen["roughness"] == 0.4
    with pytest.raises(TypeError):
        frozen["roughness"] = 0.1
