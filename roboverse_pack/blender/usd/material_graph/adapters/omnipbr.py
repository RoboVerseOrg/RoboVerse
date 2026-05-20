"""Adapter for OmniPBR-like MDL material graphs."""

from __future__ import annotations

from typing import Mapping

from ..context import MaterialContext
from ..extract import asset_path_string, coerce_float
from ..fallback import CLASS_FALLBACKS, find_material_class
from ..aliases import IOR_ALIASES
from ..bindings import resolve_uv_set
from ..schema import InputSpec, PreviewMaterialSpec, RawMaterialSpec, TextureSpec
from ..slots import SLOT_REGISTRY, SlotSpec
from ..texture_paths import normalize_texture_asset_path
from ..uv import resolve_slot_uv_transform

_SLOT_PREFIXES = {
    "base_color": "BaseColor",
    "normal": "Normal",
    "metallic": "Metallic",
    "roughness": "Roughness",
    "glossiness": "Gloss",
    "specular": "Specular",
    "emissive": "Emissive",
    "opacity": "Opacity",
}


class OmniPBRAdapter:
    name = "omnipbr"

    def score(self, raw: RawMaterialSpec) -> int:
        if raw.connected_surface_shader_id:
            return 80 if "omnipbr" in raw.connected_surface_shader_id.lower() else 0
        candidates = (*raw.shader_ids, raw.mdl_source_asset or "")
        return 80 if any("omnipbr" in candidate.lower() for candidate in candidates) else 0

    def convert(self, raw: RawMaterialSpec, context: MaterialContext) -> PreviewMaterialSpec:
        quality_notes: list[str] = []
        base_color, notes = _input_from_slot(raw.values, SLOT_REGISTRY["base_color"], context)
        quality_notes.extend(notes)
        material_class = find_material_class(raw.material_path, raw.connected_surface_shader_id)
        if base_color is None:
            if material_class:
                base_color = InputSpec(value=CLASS_FALLBACKS[material_class])
            else:
                base_color = InputSpec(value=(0.5, 0.5, 0.5))
            quality_notes.append("No diffuse color or texture alias found.")

        roughness, notes = _input_from_slot(raw.values, SLOT_REGISTRY["roughness"], context)
        quality_notes.extend(notes)
        if roughness is None:
            roughness, notes = _input_from_slot(raw.values, SLOT_REGISTRY["glossiness"], context)
            quality_notes.extend(notes)

        specular, notes = _input_from_slot(raw.values, SLOT_REGISTRY["specular"], context)
        quality_notes.extend(notes)
        normal, notes = _input_from_slot(raw.values, SLOT_REGISTRY["normal"], context)
        quality_notes.extend(notes)
        metallic, notes = _input_from_slot(raw.values, SLOT_REGISTRY["metallic"], context)
        quality_notes.extend(notes)
        emissive, notes = _input_from_slot(raw.values, SLOT_REGISTRY["emissive"], context)
        quality_notes.extend(notes)
        opacity, notes = _input_from_slot(raw.values, SLOT_REGISTRY["opacity"], context)
        quality_notes.extend(notes)

        return PreviewMaterialSpec(
            material_path=raw.material_path,
            material_name=raw.material_name,
            source_shader_ids=raw.shader_ids,
            mdl_source_asset=raw.mdl_source_asset,
            base_color=base_color,
            normal=normal,
            metallic=metallic,
            roughness=roughness,
            specular_color=specular,
            emissive_color=emissive,
            opacity=opacity,
            ior=_ior(raw.values),
            use_specular_workflow=specular is not None,
            material_class=material_class,
            conversion_policy="direct_graph",
            quality_notes=tuple(quality_notes),
        )


def _input_from_slot(
    values: Mapping[str, object], slot: SlotSpec, context: MaterialContext
) -> tuple[InputSpec | None, tuple[str, ...]]:
    if not _slot_enabled(values, slot):
        return None, ()

    for alias in slot.texture_aliases:
        texture_file = asset_path_string(values.get(alias))
        if not texture_file:
            continue
        path_result = normalize_texture_asset_path(str(texture_file), context.texture_base_dir)
        scale = (-1.0, -1.0, -1.0, -1.0) if slot.invert_to_roughness else slot.scale
        bias = (1.0, 1.0, 1.0, 1.0) if slot.invert_to_roughness else slot.bias
        slot_prefix = _SLOT_PREFIXES[slot.name]
        requested_uv_index = _first_int(values, (f"{slot_prefix}_MaxTexCoordIndex", "MaxTexCoordIndex"))
        return InputSpec(
            texture=TextureSpec(
                file=path_result.normalized,
                source_color_space=slot.color_space,
                channel=slot.texture_output,
                uv_set=resolve_uv_set(requested_uv_index, _common_uv_primvars(context)),
                uv_transform=resolve_slot_uv_transform(values, slot_prefix, slot.uva_aliases),
                scale=scale,
                bias=bias,
                role=slot.name,
                source_input=alias,
            ),
            source_inputs=(alias,),
        ), path_result.warnings

    for alias in slot.value_aliases:
        if alias not in values or values.get(alias) is None:
            continue
        raw_value = values[alias]
        value = _coerce_value(raw_value, slot)
        if value is None:
            continue
        if slot.invert_to_roughness:
            scalar = coerce_float(value)
            if scalar is None:
                continue
            value = 1.0 - scalar
        return InputSpec(value=value, source_inputs=(alias,)), ()
    return None, ()


def _slot_enabled(values: Mapping[str, object], slot: SlotSpec) -> bool:
    for alias in slot.enable_aliases:
        if alias not in values:
            continue
        value = values.get(alias)
        if value is None:
            return False
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return float(value) != 0.0
        if isinstance(value, str):
            return value.strip().lower() not in {"false", "0"}
        return True
    return True


def _coerce_value(value: object, slot: SlotSpec) -> object | None:
    if slot.name in {"metallic", "roughness", "glossiness", "opacity"}:
        return _coerce_scalar_value(value)

    scalar = coerce_float(value)
    if scalar is not None:
        return scalar
    try:
        return (float(value[0]), float(value[1]), float(value[2]))  # type: ignore[index]
    except (TypeError, ValueError, IndexError):
        return None


def _coerce_scalar_value(value: object) -> float | None:
    scalar = coerce_float(value)
    if scalar is not None:
        return scalar
    try:
        return float(value[0])  # type: ignore[index]
    except (TypeError, ValueError, IndexError):
        return None


def _first_int(values: Mapping[str, object], aliases: tuple[str, ...]) -> int | None:
    for alias in aliases:
        if alias not in values or values.get(alias) is None:
            continue
        value = values.get(alias)
        try:
            return int(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            try:
                return int(float(value))  # type: ignore[arg-type]
            except (TypeError, ValueError):
                continue
    return None


def _common_uv_primvars(context: MaterialContext) -> tuple[str, ...]:
    if not context.uv_primvars_by_prim:
        return ("st",)

    paths = context.bound_prim_paths or tuple(context.uv_primvars_by_prim)
    sets = [set(context.uv_primvars_by_prim.get(path, ())) for path in paths]
    if not sets:
        return ("st",)

    intersection = set.intersection(*sets)
    if intersection:
        return tuple(sorted(intersection))

    return ("st",)


def _ior(values: Mapping[str, object]) -> float | None:
    for alias in IOR_ALIASES:
        value = coerce_float(values.get(alias))
        if value is not None:
            return value
    return None
