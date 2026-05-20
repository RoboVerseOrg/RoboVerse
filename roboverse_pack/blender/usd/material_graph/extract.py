"""Extraction helpers for source USD material graphs."""

from __future__ import annotations

from typing import Any


def coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def coerce_vec3(value: Any, Gf: Any) -> Any | None:
    if value is None:
        return None
    try:
        return Gf.Vec3f(float(value[0]), float(value[1]), float(value[2]))
    except (TypeError, ValueError, IndexError):
        return None


def asset_path_string(value: Any) -> str | None:
    if value is None:
        return None
    path = getattr(value, "path", None)
    return path or str(value)


def shader_input(shader: Any, aliases: tuple[str, ...]) -> Any:
    if shader is None:
        return None
    for alias in aliases:
        shader_input_value = shader.GetInput(alias)
        if shader_input_value:
            value = shader_input_value.Get()
            if value is not None:
                return value
    return None


def surface_shader(material: Any) -> Any:
    source = material.ComputeSurfaceSource()
    if isinstance(source, tuple):
        return source[0] if source else None
    return source


def connected_surface_shader_id(material: Any) -> str | None:
    shader = surface_shader(material)
    if not shader:
        return None
    shader_id_value = shader.GetIdAttr().Get()
    return str(shader_id_value) if shader_id_value else None


def shader_ids(material_prim: Any, Usd: Any, UsdShade: Any) -> list[str]:
    ids = []
    for descendant in Usd.PrimRange(material_prim):
        if descendant.GetPath() == material_prim.GetPath() or not descendant.IsA(UsdShade.Shader):
            continue
        shader = UsdShade.Shader(descendant)
        shader_id = shader.GetIdAttr().Get()
        if shader_id:
            ids.append(str(shader_id))
    return ids


def source_asset_from_material(material_prim: Any, Usd: Any, UsdShade: Any) -> str | None:
    for descendant in Usd.PrimRange(material_prim):
        if descendant.GetPath() == material_prim.GetPath() or not descendant.IsA(UsdShade.Shader):
            continue
        shader = UsdShade.Shader(descendant)
        source_input_value = shader.GetInput("mdl:sourceAsset")
        if source_input_value:
            source_asset = asset_path_string(source_input_value.Get())
            if source_asset is not None:
                return source_asset
    return None

