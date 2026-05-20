"""Generate a minimal Blender-friendly USD material overlay."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

COLOR_ALIASES = ("diffuseColor", "BaseColor_Color", "diffuse_color_constant", "base_color")
TEXTURE_ALIASES = (
    "BaseColor_Tex",
    "BaseColor_Texture",
    "diffuse_texture",
    "albedo_texture",
    "base_color_texture",
    "diffuseColor_texture",
)
ROUGHNESS_ALIASES = ("roughness", "reflection_roughness_constant")
METALLIC_ALIASES = ("metallic", "metallic_constant")
OPACITY_ALIASES = ("opacity", "opacity_constant")
EMISSIVE_ALIASES = ("emissiveColor", "Emissive_Color")

CLASS_FALLBACKS = {
    "wall": (0.72, 0.72, 0.68),
    "floor": (0.55, 0.50, 0.44),
    "cabinet": (0.60, 0.46, 0.32),
    "glass": (0.78, 0.90, 0.96),
    "screen": (0.02, 0.02, 0.025),
    "metal": (0.55, 0.55, 0.55),
    "fabric": (0.48, 0.42, 0.38),
}


def _find_material_class(material_path: str, shader_id: str | None) -> str | None:
    text = f"{material_path} {shader_id or ''}".lower()
    for material_class in CLASS_FALLBACKS:
        if material_class in text:
            return material_class
    return None


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_vec3(value: Any, Gf: Any) -> Any | None:
    if value is None:
        return None
    try:
        return Gf.Vec3f(float(value[0]), float(value[1]), float(value[2]))
    except (TypeError, ValueError, IndexError):
        return None


def _asset_path_string(value: Any) -> str | None:
    if value is None:
        return None
    path = getattr(value, "path", None)
    return path or str(value)


def _shader_input(shader: Any, aliases: tuple[str, ...]) -> Any:
    if shader is None:
        return None
    for alias in aliases:
        shader_input = shader.GetInput(alias)
        if shader_input:
            value = shader_input.Get()
            if value is not None:
                return value
    return None


def _surface_shader(material: Any) -> Any:
    source = material.ComputeSurfaceSource()
    if isinstance(source, tuple):
        return source[0] if source else None
    return source


def _clear_or_create_layer(path: Path, Sdf: Any) -> Any:
    layer = Sdf.Layer.FindOrOpen(str(path))
    if layer is None:
        layer = Sdf.Layer.CreateNew(str(path))
    else:
        layer.Clear()
    return layer


def _sublayer_path(layer_path: Path, root_parent: Path) -> str:
    if layer_path.is_absolute():
        return str(layer_path)
    return os.path.relpath(layer_path, root_parent)


def _connect_surface(material: Any, shader: Any) -> None:
    output = material.CreateSurfaceOutput()
    output.ConnectToSource(shader.ConnectableAPI(), "surface")


def _author_texture_chain(material_path: Any, preview: Any, texture_asset: str, overlay_stage: Any, Sdf: Any, UsdShade: Any) -> None:
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


def _material_entry(material_path: str) -> dict[str, Any]:
    return {
        "status": "converted",
        "warnings": [],
        "material_class": None,
    }


def generate_blender_overlay(
    input_path: str | Path,
    overlay_path: str | Path,
    root_path: str | Path,
    texture_cache: str | Path,
    resolution: int = 2048,
    samples: int = 16,
) -> dict[str, Any]:
    from pxr import Gf, Sdf, Usd, UsdShade

    source_path = Path(input_path)
    overlay = Path(overlay_path)
    root = Path(root_path)
    cache = Path(texture_cache)
    cache.mkdir(parents=True, exist_ok=True)
    overlay.parent.mkdir(parents=True, exist_ok=True)
    root.parent.mkdir(parents=True, exist_ok=True)

    source_stage = Usd.Stage.Open(str(source_path))
    if source_stage is None:
        raise ValueError(f"Unable to open USD stage: {source_path}")

    overlay_layer = _clear_or_create_layer(overlay, Sdf)
    overlay_stage = Usd.Stage.Open(overlay_layer)
    report = {
        "input_path": str(source_path),
        "overlay_path": str(overlay),
        "root_path": str(root),
        "resolution": resolution,
        "samples": samples,
        "materials": {},
    }

    for prim in source_stage.Traverse():
        if not prim.IsA(UsdShade.Material):
            continue

        material_path = prim.GetPath()
        material_path_str = str(material_path)
        source_material = UsdShade.Material(prim)
        source_shader = _surface_shader(source_material)
        shader_id = None
        if source_shader:
            shader_id_value = source_shader.GetIdAttr().Get()
            shader_id = str(shader_id_value) if shader_id_value else None

        entry = _material_entry(material_path_str)
        material_class = _find_material_class(material_path_str, shader_id)
        entry["material_class"] = material_class

        overlay_material = UsdShade.Material.Define(overlay_stage, material_path)
        preview = UsdShade.Shader.Define(overlay_stage, material_path.AppendChild("PreviewSurface"))
        preview.CreateIdAttr("UsdPreviewSurface")
        _connect_surface(overlay_material, preview)

        diffuse_color = _coerce_vec3(_shader_input(source_shader, COLOR_ALIASES), Gf)
        diffuse_texture = _asset_path_string(_shader_input(source_shader, TEXTURE_ALIASES))
        if diffuse_texture:
            _author_texture_chain(material_path, preview, diffuse_texture, overlay_stage, Sdf, UsdShade)
        else:
            if diffuse_color is None and material_class:
                diffuse_color = Gf.Vec3f(*CLASS_FALLBACKS[material_class])
            if diffuse_color is not None:
                preview.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(diffuse_color)
            else:
                entry["warnings"].append("No diffuse color or texture alias found.")

        roughness = _coerce_float(_shader_input(source_shader, ROUGHNESS_ALIASES))
        if roughness is not None:
            preview.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)

        metallic = _coerce_float(_shader_input(source_shader, METALLIC_ALIASES))
        if metallic is not None:
            preview.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
        elif material_class == "metal":
            preview.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(1.0)

        opacity = _coerce_float(_shader_input(source_shader, OPACITY_ALIASES))
        if opacity is not None:
            preview.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(opacity)
        elif material_class == "glass":
            preview.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(0.35)

        if material_class == "glass":
            ior = _coerce_float(_shader_input(source_shader, ("ior", "glass_ior", "index_of_refraction")))
            preview.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(ior if ior is not None else 1.45)

        emissive = _coerce_vec3(_shader_input(source_shader, EMISSIVE_ALIASES), Gf)
        if emissive is not None:
            preview.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(emissive)

        report["materials"][material_path_str] = entry

    overlay_layer.Save()
    root_layer = _clear_or_create_layer(root, Sdf)
    root_parent = root.parent if str(root.parent) else Path(".")
    root_layer.subLayerPaths = [_sublayer_path(overlay, root_parent), _sublayer_path(source_path, root_parent)]
    root_layer.Save()

    report_json = cache / "conversion_report.json"
    report_md = cache / "conversion_report.md"
    report_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_conversion_report_md(report, report_md)
    return report


def write_conversion_report_md(report: dict[str, Any], path: str | Path) -> None:
    lines = [
        "# Blender Material Overlay Conversion Report",
        "",
        f"Input: `{report.get('input_path', '')}`",
        f"Overlay: `{report.get('overlay_path', '')}`",
        f"Root: `{report.get('root_path', '')}`",
        "",
        "| Material | Status | Class | Warnings |",
        "| --- | --- | --- | --- |",
    ]
    for material_path, entry in sorted(report.get("materials", {}).items()):
        warnings = "; ".join(entry.get("warnings", []))
        lines.append(
            f"| `{material_path}` | {entry.get('status', '')} | {entry.get('material_class') or ''} | {warnings} |"
        )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def verify_overlay_material_coverage(root_path: str | Path, expected_material_paths: Any) -> None:
    from pxr import Usd, UsdShade

    stage = Usd.Stage.Open(str(Path(root_path)))
    if stage is None:
        raise ValueError(f"Unable to open USD stage: {root_path}")

    missing = []
    non_preview = []
    for material_path in expected_material_paths:
        material = UsdShade.Material.Get(stage, material_path)
        if not material:
            missing.append(str(material_path))
            continue
        shader = _surface_shader(material)
        if not shader or shader.GetIdAttr().Get() != "UsdPreviewSurface":
            non_preview.append(str(material_path))

    if missing or non_preview:
        details = []
        if missing:
            details.append(f"missing materials: {', '.join(missing)}")
        if non_preview:
            details.append(f"missing preview surfaces: {', '.join(non_preview)}")
        raise AssertionError("; ".join(details))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a Blender USD material overlay.")
    parser.add_argument("--input", required=True, help="Input USD stage.")
    parser.add_argument("--overlay", required=True, help="Output overlay USD layer.")
    parser.add_argument("--root", required=True, help="Output root USD layer.")
    parser.add_argument("--texture-cache", required=True, help="Texture cache/report directory.")
    parser.add_argument("--resolution", type=int, default=2048)
    parser.add_argument("--samples", type=int, default=16)
    args = parser.parse_args(argv)

    generate_blender_overlay(
        args.input,
        args.overlay,
        args.root,
        args.texture_cache,
        resolution=args.resolution,
        samples=args.samples,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
