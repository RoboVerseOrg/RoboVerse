"""Author USD PreviewSurface graphs for Blender material overlays."""

from __future__ import annotations

from typing import Any


def connect_surface(material: Any, shader: Any) -> None:
    output = material.CreateSurfaceOutput()
    output.ConnectToSource(shader.ConnectableAPI(), "surface")


def author_texture_chain(
    material_path: Any,
    preview: Any,
    texture_asset: str,
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
    texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(texture_asset))
    texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
    texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)

    preview.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(texture.ConnectableAPI(), "rgb")


def author_preview_surface(overlay_stage: Any, material_path: Any, params: dict[str, Any], Sdf: Any, UsdShade: Any) -> None:
    overlay_material = UsdShade.Material.Define(overlay_stage, material_path)
    preview = UsdShade.Shader.Define(overlay_stage, material_path.AppendChild("PreviewSurface"))
    preview.CreateIdAttr("UsdPreviewSurface")
    connect_surface(overlay_material, preview)

    diffuse_texture = params.get("diffuse_texture")
    if diffuse_texture:
        author_texture_chain(material_path, preview, diffuse_texture, overlay_stage, Sdf, UsdShade)
    elif params.get("diffuse_color") is not None:
        preview.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(params["diffuse_color"])

    roughness = params.get("roughness")
    if roughness is not None:
        preview.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)

    metallic = params.get("metallic")
    if metallic is not None:
        preview.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    elif params.get("material_class") == "metal":
        preview.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(1.0)

    opacity = params.get("opacity")
    if opacity is not None:
        preview.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(opacity)
    elif params.get("material_class") == "glass":
        preview.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(0.35)

    if params.get("material_class") == "glass":
        ior = params.get("ior")
        preview.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(ior if ior is not None else 1.45)

    emissive = params.get("emissive")
    if emissive is not None:
        preview.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(emissive)

