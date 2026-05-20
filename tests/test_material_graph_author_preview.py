from pathlib import Path

import pytest

from roboverse_pack.blender.usd.material_graph.adapters.omnipbr import OmniPBRAdapter
from roboverse_pack.blender.usd.material_graph.author_preview import _resolved_texture_channel, author_material
from roboverse_pack.blender.usd.material_graph.context import MaterialContext
from roboverse_pack.blender.usd.material_graph.schema import (
    InputSpec,
    PreviewMaterialSpec,
    RawMaterialSpec,
    TextureSpec,
    UVSetSpec,
    UVTransformSpec,
    freeze_values,
)

Gf = None
Sdf = None
Usd = None
UsdShade = None


def _spec(**overrides):
    return PreviewMaterialSpec(
        material_path=overrides.pop("material_path", "/World/Looks/Mat"),
        material_name=overrides.pop("material_name", "Mat"),
        source_shader_ids=overrides.pop("source_shader_ids", ("OmniPBR",)),
        mdl_source_asset=overrides.pop("mdl_source_asset", None),
        base_color=overrides.pop("base_color", InputSpec(value=(0.5, 0.5, 0.5))),
        normal=overrides.pop("normal", None),
        metallic=overrides.pop("metallic", None),
        roughness=overrides.pop("roughness", None),
        specular_color=overrides.pop("specular_color", None),
        emissive_color=overrides.pop("emissive_color", None),
        opacity=overrides.pop("opacity", None),
        ior=overrides.pop("ior", None),
        use_specular_workflow=overrides.pop("use_specular_workflow", False),
        material_class=overrides.pop("material_class", None),
        conversion_policy=overrides.pop("conversion_policy", "direct_graph"),
        quality_notes=overrides.pop("quality_notes", ()),
        adapter_name=overrides.pop("adapter_name", None),
    )


def _raw(values):
    return RawMaterialSpec(
        material_path="/World/Looks/Mat",
        material_name="Mat",
        shader_ids=("OmniPBR",),
        connected_surface_shader_id="OmniPBR",
        mdl_source_asset=None,
        values=freeze_values(values),
    )


def _context():
    return MaterialContext(
        source_path=Path("scene.usda"),
        texture_base_dir=Path("."),
        material_path="/World/Looks/Mat",
    )


def _stage_for(spec):
    global Gf, Sdf, Usd, UsdShade
    pxr = pytest.importorskip("pxr")
    from pxr import Gf as _Gf
    from pxr import Sdf as _Sdf
    from pxr import Usd as _Usd
    from pxr import UsdShade as _UsdShade

    Gf = _Gf
    Sdf = _Sdf
    Usd = _Usd
    UsdShade = _UsdShade
    stage = Usd.Stage.CreateInMemory()
    author_material(stage, spec, Gf, Sdf, UsdShade)
    return stage


def _preview(stage, material_path="/World/Looks/Mat"):
    return UsdShade.Shader.Get(stage, f"{material_path}/PreviewSurface")


def _material(stage, material_path="/World/Looks/Mat"):
    return UsdShade.Material.Get(stage, material_path)


def _texture(stage, slot_name, material_path="/World/Looks/Mat"):
    return UsdShade.Shader.Get(stage, f"{material_path}/{slot_name}_Texture")


def _reader(stage, slot_name, material_path="/World/Looks/Mat"):
    return UsdShade.Shader.Get(stage, f"{material_path}/{slot_name}_Primvar")


def _transform(stage, slot_name, material_path="/World/Looks/Mat"):
    return UsdShade.Shader.Get(stage, f"{material_path}/{slot_name}_Transform2d")


def _connected_source(stage, input_name, material_path="/World/Looks/Mat"):
    return _preview(stage, material_path).GetInput(input_name).GetConnectedSource()


def _texture_input(file, channel="rgb", source_color_space="sRGB", **overrides):
    return InputSpec(
        texture=TextureSpec(
            file=file,
            source_color_space=source_color_space,
            channel=channel,
            **overrides,
        )
    )


@pytest.mark.parametrize(
    ("source_input", "expected"),
    [
        ("Opacity_Tex", "a"),
        ("Opacity_Texture", "a"),
        ("Alpha_Tex", "a"),
        ("Alpha_Texture", "a"),
        ("alpha_texture", "a"),
        ("opacity_texture", "r"),
        (None, "r"),
    ],
)
def test_resolved_texture_channel_for_a_or_r(source_input, expected):
    texture = TextureSpec(
        file="textures/opacity.png",
        source_color_space="raw",
        channel="a_or_r",
        source_input=source_input,
    )

    assert _resolved_texture_channel(texture) == expected


@pytest.mark.parametrize(
    ("conversion_policy", "adapter_name"),
    [
        ("direct_graph", "omnipbr"),
        ("scalar_fallback", "scalar_fallback"),
        ("mdl_bake", "mdl_bake"),
    ],
)
def test_authored_material_custom_data_tags_conversion_metadata(conversion_policy, adapter_name):
    stage = _stage_for(_spec(conversion_policy=conversion_policy, adapter_name=adapter_name))

    custom_data = _material(stage).GetPrim().GetCustomData()

    assert custom_data["roboverse:material_conversion_policy"] == conversion_policy
    assert custom_data["roboverse:material_adapter"] == adapter_name
    assert custom_data["roboverse:material_graph_version"] == "v1"


def test_base_color_texture_chain_connects_rgb_to_diffuse_color():
    stage = _stage_for(
        _spec(base_color=_texture_input("textures/albedo.png", source_color_space="sRGB"))
    )

    texture = _texture(stage, "base_color")
    reader = _reader(stage, "base_color")
    connection = _connected_source(stage, "diffuseColor")

    assert texture.GetIdAttr().Get() == "UsdUVTexture"
    assert texture.GetInput("file").Get().path == "textures/albedo.png"
    assert texture.GetInput("sourceColorSpace").Get() == "sRGB"
    assert texture.GetInput("wrapS").Get() == "repeat"
    assert texture.GetInput("wrapT").Get() == "repeat"
    assert reader.GetIdAttr().Get() == "UsdPrimvarReader_float2"
    assert reader.GetInput("varname").Get() == "st"
    assert connection[0].GetPath() == texture.GetPath()
    assert connection[1] == "rgb"


def test_primvar_reader_varname_is_authored_as_string_not_token():
    stage = _stage_for(
        _spec(base_color=_texture_input("textures/albedo.png", source_color_space="sRGB"))
    )

    assert 'string inputs:varname = "st"' in stage.GetRootLayer().ExportToString()


def test_texture_chain_authors_transform2d_and_custom_uv_varname():
    stage = _stage_for(
        _spec(
            base_color=_texture_input(
                "textures/albedo.png",
                source_color_space="sRGB",
                uv_set=UVSetSpec("st1", 1, "matched"),
                uv_transform=UVTransformSpec(
                    scale=(2.0, 3.0),
                    rotation_degrees=45.0,
                    translation=(0.25, 0.5),
                ),
            )
        )
    )

    reader = _reader(stage, "base_color")
    transform = _transform(stage, "base_color")
    texture = _texture(stage, "base_color")

    assert reader.GetInput("varname").Get() == "st1"
    assert transform.GetIdAttr().Get() == "UsdTransform2d"
    assert transform.GetInput("scale").Get() == Gf.Vec2f(2.0, 3.0)
    assert transform.GetInput("rotation").Get() == pytest.approx(45.0)
    assert transform.GetInput("translation").Get() == Gf.Vec2f(0.25, 0.5)
    assert transform.GetInput("in").GetConnectedSource()[0].GetPath() == reader.GetPath()
    assert texture.GetInput("st").GetConnectedSource()[0].GetPath() == transform.GetPath()


def test_normal_texture_authors_raw_source_color_space_and_scale_bias():
    stage = _stage_for(
        _spec(
            normal=_texture_input(
                "textures/normal.png",
                source_color_space="raw",
                scale=(2.0, 2.0, 2.0, 1.0),
                bias=(-1.0, -1.0, -1.0, 0.0),
            )
        )
    )

    texture = _texture(stage, "normal")
    connection = _connected_source(stage, "normal")

    assert texture.GetInput("sourceColorSpace").Get() == "raw"
    assert texture.GetInput("scale").Get() == Gf.Vec4f(2.0, 2.0, 2.0, 1.0)
    assert texture.GetInput("bias").Get() == Gf.Vec4f(-1.0, -1.0, -1.0, 0.0)
    assert connection[0].GetPath() == texture.GetPath()
    assert connection[1] == "rgb"


def test_normal_scalar_value_is_not_authored():
    stage = _stage_for(_spec(normal=InputSpec(value=(0.0, 0.0, 1.0))))

    assert not _preview(stage).GetInput("normal")


def test_metallic_texture_connects_r_channel():
    stage = _stage_for(_spec(metallic=_texture_input("textures/metal.png", channel="r", source_color_space="raw")))

    connection = _connected_source(stage, "metallic")

    assert connection[0].GetPath() == _texture(stage, "metallic").GetPath()
    assert connection[1] == "r"


def test_roughness_texture_connects_r_channel():
    stage = _stage_for(_spec(roughness=_texture_input("textures/rough.png", channel="r", source_color_space="raw")))

    connection = _connected_source(stage, "roughness")

    assert connection[0].GetPath() == _texture(stage, "roughness").GetPath()
    assert connection[1] == "r"


def test_glossiness_texture_inversion_scale_and_bias_are_authored():
    converted = OmniPBRAdapter().convert(_raw({"Glossiness_Tex": "textures/gloss.png"}), _context())
    stage = _stage_for(converted)

    texture = _texture(stage, "roughness")

    assert texture.GetInput("scale").Get() == Gf.Vec4f(-1.0, -1.0, -1.0, -1.0)
    assert texture.GetInput("bias").Get() == Gf.Vec4f(1.0, 1.0, 1.0, 1.0)
    assert _connected_source(stage, "roughness")[1] == "r"


def test_specular_texture_sets_specular_workflow_flag():
    stage = _stage_for(
        _spec(
            specular_color=_texture_input("textures/specular.png", source_color_space="sRGB"),
            use_specular_workflow=True,
        )
    )

    connection = _connected_source(stage, "specularColor")

    assert connection[0].GetPath() == _texture(stage, "specular").GetPath()
    assert connection[1] == "rgb"
    assert _preview(stage).GetInput("useSpecularWorkflow").Get() == 1


def test_emissive_texture_connects_rgb_to_emissive_color():
    stage = _stage_for(_spec(emissive_color=_texture_input("textures/emissive.png", source_color_space="sRGB")))

    connection = _connected_source(stage, "emissiveColor")

    assert connection[0].GetPath() == _texture(stage, "emissive").GetPath()
    assert connection[1] == "rgb"


@pytest.mark.parametrize("source_input", ["Opacity_Tex", "Alpha_Tex"])
def test_opacity_adapter_a_or_r_sources_connect_alpha_channel(source_input):
    converted = OmniPBRAdapter().convert(_raw({source_input: "textures/opacity.png"}), _context())
    stage = _stage_for(converted)

    connection = _connected_source(stage, "opacity")

    assert converted.opacity.texture.source_input == source_input
    assert connection[0].GetPath() == _texture(stage, "opacity").GetPath()
    assert connection[1] == "a"


def test_opacity_a_or_r_non_alpha_source_connects_r_channel():
    stage = _stage_for(
        _spec(
            opacity=InputSpec(
                texture=TextureSpec(
                    file="textures/opacity.png",
                    source_color_space="raw",
                    channel="a_or_r",
                    source_input="opacity_texture",
                )
            )
        )
    )

    assert _connected_source(stage, "opacity")[1] == "r"
