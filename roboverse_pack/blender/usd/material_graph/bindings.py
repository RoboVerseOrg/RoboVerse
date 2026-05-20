"""Material binding context and UV primvar resolution helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from .schema import UVSetSpec

_UV_PRIMVAR_RE = re.compile(r"^(?:st\d*|uv\d*|map\d+)$")


@dataclass(frozen=True)
class MaterialBindingContext:
    material_path: str
    bound_prim_paths: tuple[str, ...]
    uv_primvars_by_prim: Mapping[str, tuple[str, ...]]


def _attr_name(attribute: Any) -> str:
    get_name = getattr(attribute, "GetName", None)
    if get_name is not None:
        return str(get_name())
    return str(attribute)


def _uv_primvars(prim: Any, UsdGeom: Any) -> tuple[str, ...]:
    names: set[str] = set()
    get_attributes = getattr(prim, "GetAttributes", None)
    if get_attributes is None:
        return ()
    for attribute in get_attributes():
        full_name = _attr_name(attribute)
        if not full_name.startswith("primvars:"):
            continue
        name = full_name.removeprefix("primvars:").split(":", 1)[0]
        if _UV_PRIMVAR_RE.match(name):
            names.add(name)
    return tuple(sorted(names))


def _bound_material_path(prim: Any, UsdShade: Any) -> str | None:
    try:
        binding = UsdShade.MaterialBindingAPI(prim)
        bound = binding.ComputeBoundMaterial()
    except Exception:
        return None
    material = bound[0] if isinstance(bound, tuple) else bound
    if not material:
        return None
    get_prim = getattr(material, "GetPrim", None)
    material_prim = get_prim() if get_prim is not None else material
    get_path = getattr(material_prim, "GetPath", None)
    if get_path is None:
        return None
    return str(get_path())


def _is_bindable_geometry(prim: Any, UsdGeom: Any) -> bool:
    if UsdGeom is None:
        return True
    is_a = getattr(prim, "IsA", None)
    if is_a is None:
        return True
    for schema_name in ("Gprim", "Mesh"):
        schema = getattr(UsdGeom, schema_name, None)
        if schema is None:
            continue
        try:
            if prim.IsA(schema):
                return True
        except Exception:
            continue
    return False


def collect_material_binding_contexts(stage: Any, UsdShade: Any, UsdGeom: Any) -> dict[str, MaterialBindingContext]:
    bound_paths: dict[str, list[str]] = {}
    uv_sets: dict[str, dict[str, tuple[str, ...]]] = {}

    for prim in stage.Traverse():
        if not _is_bindable_geometry(prim, UsdGeom):
            continue
        material_path = _bound_material_path(prim, UsdShade)
        if material_path is None:
            continue
        prim_path = str(prim.GetPath())
        bound_paths.setdefault(material_path, []).append(prim_path)
        uv_sets.setdefault(material_path, {})[prim_path] = _uv_primvars(prim, UsdGeom)

    return {
        material_path: MaterialBindingContext(
            material_path=material_path,
            bound_prim_paths=tuple(paths),
            uv_primvars_by_prim=MappingProxyType(dict(uv_sets.get(material_path, {}))),
        )
        for material_path, paths in bound_paths.items()
    }


def resolve_uv_set(requested_index: int | None, available_primvars: tuple[str, ...]) -> UVSetSpec:
    available = set(available_primvars)
    if requested_index is None or requested_index == 0:
        preferences = ("st", "st0", "uv", "uv0", "map1")
    else:
        preferences = (
            f"st{requested_index}",
            f"uv{requested_index}",
            f"uv{requested_index + 1}",
            f"map{requested_index + 1}",
        )
    for primvar_name in preferences:
        if primvar_name in available:
            return UVSetSpec(primvar_name, requested_index, "matched")
    return UVSetSpec("st", requested_index, "guessed_or_missing")
