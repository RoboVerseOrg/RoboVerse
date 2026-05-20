"""Author USD PreviewSurface graphs for Blender material overlays."""

from __future__ import annotations

from typing import Any

from .report import adapter_from_policy
from .schema import InputSpec, PreviewMaterialSpec, TextureSpec, UVTransformSpec

_COLOR_INPUTS = {"diffuseColor", "emissiveColor", "specularColor"}
_TEXTURE_OUTPUT_TYPES = {
    "rgb": "Float3",
    "r": "Float",
    "g": "Float",
    "b": "Float",
    "a": "Float",
}
_ALPHA_CHANNEL_SOURCE_INPUTS = {
    "Opacity_Tex",
    "Opacity_Texture",
    "Alpha_Tex",
    "Alpha_Texture",
    "alpha_texture",
}


def connect_surface(material: Any, shader: Any) -> None:
    output = material.CreateSurfaceOutput()
    output.ConnectToSource(shader.ConnectableAPI(), "surface")


def _author_roboverse_custom_data(material: Any, spec: PreviewMaterialSpec) -> None:
    prim = material.GetPrim()
    custom_data = dict(prim.GetCustomData())
    custom_data["roboverse:material_conversion_policy"] = spec.conversion_policy
    custom_data["roboverse:material_adapter"] = spec.adapter_name or adapter_from_policy(spec.conversion_policy)
    custom_data["roboverse:material_graph_version"] = "v1"
    prim.SetCustomData(custom_data)


def _value_type(Sdf: Any, type_name: str) -> Any:
    return getattr(Sdf.ValueTypeNames, type_name)


def author_texture_chain(
    stage: Any,
    material_path: Any,
    slot_name: str,
    texture_spec: TextureSpec,
    Sdf: Any,
    UsdShade: Any,
) -> Any:
    reader = UsdShade.Shader.Define(stage, material_path.AppendChild(f"{slot_name}_Primvar"))
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.String).Set(texture_spec.uv_set.primvar_name)
    reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
    st_source = reader

    if texture_spec.uv_transform is not None:
        st_source = author_transform2d(
            stage,
            material_path,
            slot_name,
            texture_spec.uv_transform,
            reader,
            "result",
            Sdf,
            UsdShade,
        )

    texture = UsdShade.Shader.Define(stage, material_path.AppendChild(f"{slot_name}_Texture"))
    texture.CreateIdAttr("UsdUVTexture")
    texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(texture_spec.file))
    texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st_source.ConnectableAPI(), "result")
    texture.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set(texture_spec.source_color_space)
    if texture_spec.scale is not None:
        texture.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(texture_spec.scale)
    if texture_spec.bias is not None:
        texture.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(texture_spec.bias)
    texture.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set(texture_spec.wrap_s or "repeat")
    texture.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set(texture_spec.wrap_t or "repeat")

    for output_name, type_name in _TEXTURE_OUTPUT_TYPES.items():
        texture.CreateOutput(output_name, _value_type(Sdf, type_name))

    return texture


def author_transform2d(
    stage: Any,
    material_path: Any,
    slot_name: str,
    transform_spec: UVTransformSpec,
    prior_source: Any,
    prior_output: str,
    Sdf: Any,
    UsdShade: Any,
) -> Any:
    transform = UsdShade.Shader.Define(stage, material_path.AppendChild(f"{slot_name}_Transform2d"))
    transform.CreateIdAttr("UsdTransform2d")
    transform.CreateInput("in", Sdf.ValueTypeNames.Float2).ConnectToSource(prior_source.ConnectableAPI(), prior_output)
    transform.CreateInput("scale", Sdf.ValueTypeNames.Float2).Set(transform_spec.scale)
    transform.CreateInput("rotation", Sdf.ValueTypeNames.Float).Set(transform_spec.rotation_degrees)
    transform.CreateInput("translation", Sdf.ValueTypeNames.Float2).Set(transform_spec.translation)
    transform.CreateOutput("result", Sdf.ValueTypeNames.Float2)
    return transform


def _preview_input_type(input_name: str, Sdf: Any) -> Any:
    if input_name in _COLOR_INPUTS:
        return Sdf.ValueTypeNames.Color3f
    if input_name == "normal":
        if hasattr(Sdf.ValueTypeNames, "Normal3f"):
            return Sdf.ValueTypeNames.Normal3f
        if hasattr(Sdf.ValueTypeNames, "Float3"):
            return Sdf.ValueTypeNames.Float3
        return Sdf.ValueTypeNames.Color3f
    return Sdf.ValueTypeNames.Float


def _preview_input_value(value: Any, input_name: str, Gf: Any) -> Any:
    if value is not None and input_name in _COLOR_INPUTS:
        try:
            return Gf.Vec3f(float(value[0]), float(value[1]), float(value[2]))
        except (TypeError, ValueError, IndexError):
            return value
    return value


def _resolved_texture_channel(texture: TextureSpec) -> str:
    if texture.channel != "a_or_r":
        return texture.channel
    if texture.source_input in _ALPHA_CHANNEL_SOURCE_INPUTS:
        return "a"
    return "r"


def connect_or_set(
    stage: Any,
    material_path: Any,
    surface: Any,
    input_name: str,
    slot_name: str,
    spec: InputSpec | None,
    Gf: Any,
    Sdf: Any,
    UsdShade: Any,
) -> None:
    if spec is None:
        return

    if spec.texture is not None:
        texture = author_texture_chain(stage, material_path, slot_name, spec.texture, Sdf, UsdShade)
        channel = _resolved_texture_channel(spec.texture)
        surface.CreateInput(input_name, _preview_input_type(input_name, Sdf)).ConnectToSource(
            texture.ConnectableAPI(), channel
        )
        return

    if spec.value is None:
        return

    surface.CreateInput(input_name, _preview_input_type(input_name, Sdf)).Set(
        _preview_input_value(spec.value, input_name, Gf)
    )


def author_material(stage: Any, spec: PreviewMaterialSpec, Gf: Any, Sdf: Any, UsdShade: Any) -> None:
    material_path = Sdf.Path(spec.material_path)
    overlay_material = UsdShade.Material.Define(stage, material_path)
    _author_roboverse_custom_data(overlay_material, spec)
    preview = UsdShade.Shader.Define(stage, material_path.AppendChild("PreviewSurface"))
    preview.CreateIdAttr("UsdPreviewSurface")
    connect_surface(overlay_material, preview)

    connect_or_set(stage, material_path, preview, "diffuseColor", "base_color", spec.base_color, Gf, Sdf, UsdShade)
    if spec.normal is not None and spec.normal.texture is not None:
        connect_or_set(stage, material_path, preview, "normal", "normal", spec.normal, Gf, Sdf, UsdShade)
    connect_or_set(stage, material_path, preview, "roughness", "roughness", spec.roughness, Gf, Sdf, UsdShade)

    if spec.metallic is not None and (spec.metallic.texture is not None or spec.metallic.value is not None):
        connect_or_set(stage, material_path, preview, "metallic", "metallic", spec.metallic, Gf, Sdf, UsdShade)
    elif spec.material_class == "metal":
        preview.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(1.0)

    if spec.specular_color is not None:
        connect_or_set(
            stage,
            material_path,
            preview,
            "specularColor",
            "specular",
            spec.specular_color,
            Gf,
            Sdf,
            UsdShade,
        )
    if spec.use_specular_workflow:
        preview.CreateInput("useSpecularWorkflow", Sdf.ValueTypeNames.Int).Set(1)

    connect_or_set(
        stage, material_path, preview, "emissiveColor", "emissive", spec.emissive_color, Gf, Sdf, UsdShade
    )

    if spec.opacity is not None and (spec.opacity.texture is not None or spec.opacity.value is not None):
        connect_or_set(stage, material_path, preview, "opacity", "opacity", spec.opacity, Gf, Sdf, UsdShade)
    elif spec.material_class == "glass":
        preview.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(0.35)

    if spec.material_class == "glass":
        preview.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(spec.ior if spec.ior is not None else 1.45)


def author_preview_material(stage: Any, spec: PreviewMaterialSpec, Gf: Any, Sdf: Any, UsdShade: Any) -> None:
    author_material(stage, spec, Gf, Sdf, UsdShade)
