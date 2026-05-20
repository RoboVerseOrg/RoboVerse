"""Adapter for OmniPBR-like MDL material graphs."""

from __future__ import annotations

from typing import Mapping

from ..context import MaterialContext
from ..extract import asset_path_string, coerce_float
from ..fallback import CLASS_FALLBACKS, find_material_class
from ..aliases import IOR_ALIASES
from ..schema import InputSpec, PreviewMaterialSpec, RawMaterialSpec, TextureSpec
from ..slots import SLOT_REGISTRY, SlotSpec


class OmniPBRAdapter:
    name = "omnipbr"

    def score(self, raw: RawMaterialSpec) -> int:
        if raw.connected_surface_shader_id:
            return 80 if "omnipbr" in raw.connected_surface_shader_id.lower() else 0
        candidates = (*raw.shader_ids, raw.mdl_source_asset or "")
        return 80 if any("omnipbr" in candidate.lower() for candidate in candidates) else 0

    def convert(self, raw: RawMaterialSpec, context: MaterialContext) -> PreviewMaterialSpec:
        base_color = _input_from_slot(raw.values, SLOT_REGISTRY["base_color"])
        material_class = find_material_class(raw.material_path, raw.connected_surface_shader_id)
        quality_notes: list[str] = []
        if base_color is None:
            if material_class:
                base_color = InputSpec(value=CLASS_FALLBACKS[material_class])
            else:
                base_color = InputSpec(value=(0.5, 0.5, 0.5))
            quality_notes.append("No diffuse color or texture alias found.")

        roughness = _input_from_slot(raw.values, SLOT_REGISTRY["roughness"])
        glossiness = _input_from_slot(raw.values, SLOT_REGISTRY["glossiness"])
        if roughness is None and glossiness is not None:
            roughness = glossiness

        specular = _input_from_slot(raw.values, SLOT_REGISTRY["specular"])

        return PreviewMaterialSpec(
            material_path=raw.material_path,
            material_name=raw.material_name,
            source_shader_ids=raw.shader_ids,
            mdl_source_asset=raw.mdl_source_asset,
            base_color=base_color,
            normal=_input_from_slot(raw.values, SLOT_REGISTRY["normal"]),
            metallic=_input_from_slot(raw.values, SLOT_REGISTRY["metallic"]),
            roughness=roughness,
            specular_color=specular,
            emissive_color=_input_from_slot(raw.values, SLOT_REGISTRY["emissive"]),
            opacity=_input_from_slot(raw.values, SLOT_REGISTRY["opacity"]),
            ior=_ior(raw.values),
            use_specular_workflow=specular is not None,
            material_class=material_class,
            conversion_policy="direct_graph",
            quality_notes=tuple(quality_notes),
        )


def _input_from_slot(values: Mapping[str, object], slot: SlotSpec) -> InputSpec | None:
    if not _slot_enabled(values, slot):
        return None

    for alias in slot.texture_aliases:
        texture_file = asset_path_string(values.get(alias))
        if not texture_file:
            continue
        scale = (-1.0, -1.0, -1.0, -1.0) if slot.invert_to_roughness else slot.scale
        bias = (1.0, 1.0, 1.0, 1.0) if slot.invert_to_roughness else slot.bias
        return InputSpec(
            texture=TextureSpec(
                file=texture_file,
                source_color_space=slot.color_space,
                channel=slot.texture_output,
                scale=scale,
                bias=bias,
                source_input=alias,
            ),
            source_inputs=(alias,),
        )

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
        return InputSpec(value=value, source_inputs=(alias,))
    return None


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


def _ior(values: Mapping[str, object]) -> float | None:
    for alias in IOR_ALIASES:
        value = coerce_float(values.get(alias))
        if value is not None:
            return value
    return None
