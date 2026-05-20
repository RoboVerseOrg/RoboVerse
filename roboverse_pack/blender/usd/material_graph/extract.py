"""Extraction helpers for source USD material graphs."""

from __future__ import annotations

from typing import Any

from .schema import RawMaterialSpec, freeze_values


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


def _source_asset_path_string(value: Any) -> str | None:
    if value is None:
        return None
    path = getattr(value, "path", None)
    candidate = str(path) if path is not None else asset_path_string(value)
    if candidate is None or not candidate.strip():
        return None
    return candidate


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


def _source_asset_from_get_source_asset(shader: Any) -> str | None:
    get_source_asset = getattr(shader, "GetSourceAsset", None)
    if get_source_asset is None:
        return None
    try:
        return _source_asset_path_string(get_source_asset("mdl"))
    except (TypeError, AttributeError):
        return None


def _source_asset_from_info_attribute(shader: Any) -> str | None:
    get_prim = getattr(shader, "GetPrim", None)
    prim = get_prim() if get_prim is not None else getattr(shader, "prim", None)
    get_attribute = getattr(prim, "GetAttribute", None)
    if get_attribute is None:
        return None
    attribute = get_attribute("info:mdl:sourceAsset")
    if not attribute:
        return None
    return _source_asset_path_string(attribute.Get())


def _source_asset_from_legacy_input(shader: Any) -> str | None:
    source_input_value = shader.GetInput("mdl:sourceAsset")
    if not source_input_value:
        return None
    return _source_asset_path_string(source_input_value.Get())


def source_asset_from_shader(shader: Any) -> str | None:
    for extractor in (
        _source_asset_from_get_source_asset,
        _source_asset_from_info_attribute,
        _source_asset_from_legacy_input,
    ):
        source_asset = extractor(shader)
        if source_asset is not None:
            return source_asset
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


def _iter_descendants(prim: Any) -> Any:
    for child in prim.GetChildren():
        yield child
        yield from _iter_descendants(child)


def _input_base_name(shader_input_value: Any) -> str:
    get_base_name = getattr(shader_input_value, "GetBaseName", None)
    if get_base_name is not None:
        base_name = get_base_name()
        if base_name:
            return str(base_name)
    full_name = shader_input_value.GetFullName()
    return str(full_name).split(":")[-1]


def _record_shader_inputs(values: dict[str, object], shader: Any) -> None:
    if not shader:
        return
    for shader_input_value in shader.GetInputs():
        value = shader_input_value.Get()
        if value is not None:
            values.setdefault(_input_base_name(shader_input_value), value)


def input_values(material_prim: Any, UsdShade: Any) -> dict[str, object]:
    values: dict[str, object] = {}
    _record_shader_inputs(values, surface_shader(UsdShade.Material(material_prim)))
    for descendant in _iter_descendants(material_prim):
        if descendant.IsA(UsdShade.Shader):
            _record_shader_inputs(values, UsdShade.Shader(descendant))
    return values


def extract_material(material_prim: Any, UsdShade: Any) -> RawMaterialSpec:
    material = UsdShade.Material(material_prim)
    ids: list[str] = []
    mdl_source_asset = None
    connected_shader = surface_shader(material)
    connected_shader_id_value = connected_shader.GetIdAttr().Get() if connected_shader else None
    if connected_shader_id_value:
        ids.append(str(connected_shader_id_value))

    for descendant in _iter_descendants(material_prim):
        if not descendant.IsA(UsdShade.Shader):
            continue
        shader = UsdShade.Shader(descendant)
        shader_id = shader.GetIdAttr().Get()
        if shader_id and str(shader_id) not in ids:
            ids.append(str(shader_id))
        source_asset = source_asset_from_shader(shader)
        if source_asset is not None and mdl_source_asset is None:
            mdl_source_asset = source_asset

    return RawMaterialSpec(
        material_path=str(material_prim.GetPath()),
        material_name=material_prim.GetName(),
        shader_ids=tuple(ids),
        connected_surface_shader_id=connected_surface_shader_id(material),
        mdl_source_asset=mdl_source_asset,
        values=freeze_values(input_values(material_prim, UsdShade)),
    )


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
        source_asset = source_asset_from_shader(shader)
        if source_asset is not None:
            return source_asset
    return None
