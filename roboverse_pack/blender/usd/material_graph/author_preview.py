"""Author USD PreviewSurface graphs for Blender material overlays."""

from __future__ import annotations

from typing import Any

from .schema import InputSpec, PreviewMaterialSpec, TextureSpec


def connect_surface(material: Any, shader: Any) -> None:
    output = material.CreateSurfaceOutput()
    output.ConnectToSource(shader.ConnectableAPI(), "surface")


def author_texture_chain(
    material_path: Any,
    preview: Any,
    texture_spec: TextureSpec,
    overlay_stage: Any,
    Sdf: Any,
    UsdShade: Any,
) -> None:
    reader = UsdShade.Shader.Define(overlay_stage, material_path.AppendChild("PreviewSTReader"))
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.String).Set("st")
    reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)

    texture = UsdShade.Shader.Define(overlay_stage, material_path.AppendChild("PreviewTexture"))
    texture.CreateIdAttr("UsdUVTexture")
    texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(texture_spec.file))
    texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
    texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)

    preview.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(texture.ConnectableAPI(), "rgb")


def _preview_input_type(input_name: str, Sdf: Any) -> Any:
    if input_name in {"diffuseColor", "emissiveColor", "specularColor"}:
        return Sdf.ValueTypeNames.Color3f
    return Sdf.ValueTypeNames.Float


def _preview_input_value(value: Any, input_name: str, Gf: Any) -> Any:
    if value is not None and input_name in {"diffuseColor", "emissiveColor", "specularColor"}:
        try:
            return Gf.Vec3f(float(value[0]), float(value[1]), float(value[2]))
        except (TypeError, ValueError, IndexError):
            return value
    return value


def _set_preview_input(preview: Any, input_name: str, spec: InputSpec, Gf: Any, Sdf: Any) -> None:
    if spec.value is None:
        return
    preview.CreateInput(input_name, _preview_input_type(input_name, Sdf)).Set(
        _preview_input_value(spec.value, input_name, Gf)
    )


def author_preview_material(stage: Any, spec: PreviewMaterialSpec, Gf: Any, Sdf: Any, UsdShade: Any) -> None:
    material_path = Sdf.Path(spec.material_path)
    overlay_material = UsdShade.Material.Define(stage, material_path)
    preview = UsdShade.Shader.Define(stage, material_path.AppendChild("PreviewSurface"))
    preview.CreateIdAttr("UsdPreviewSurface")
    connect_surface(overlay_material, preview)

    diffuse_texture = spec.base_color.texture
    if diffuse_texture:
        author_texture_chain(material_path, preview, diffuse_texture, stage, Sdf, UsdShade)
    else:
        _set_preview_input(preview, "diffuseColor", spec.base_color, Gf, Sdf)

    if spec.roughness is not None:
        _set_preview_input(preview, "roughness", spec.roughness, Gf, Sdf)

    if spec.metallic is not None and spec.metallic.value is not None:
        _set_preview_input(preview, "metallic", spec.metallic, Gf, Sdf)
    elif spec.material_class == "metal":
        preview.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(1.0)

    if spec.opacity is not None and spec.opacity.value is not None:
        _set_preview_input(preview, "opacity", spec.opacity, Gf, Sdf)
    elif spec.material_class == "glass":
        preview.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(0.35)

    if spec.material_class == "glass":
        preview.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(spec.ior if spec.ior is not None else 1.45)

    if spec.emissive_color is not None:
        _set_preview_input(preview, "emissiveColor", spec.emissive_color, Gf, Sdf)
