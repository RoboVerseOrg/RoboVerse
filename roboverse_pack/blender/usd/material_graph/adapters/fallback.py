"""Fallback material adapters."""

from __future__ import annotations

from dataclasses import replace

from ..aliases import IOR_ALIASES
from ..context import MaterialContext
from ..extract import coerce_float
from ..fallback import CLASS_FALLBACKS, find_material_class
from ..normalize import normalize_material
from ..schema import InputSpec, PreviewMaterialSpec, RawMaterialSpec


class ScalarFallbackAdapter:
    name = "scalar_fallback"

    def score(self, raw: RawMaterialSpec) -> int:
        return 20 if raw.values else 0

    def convert(self, raw: RawMaterialSpec, context: MaterialContext) -> PreviewMaterialSpec:
        return replace(normalize_material(raw), conversion_policy="scalar_fallback")


class SemanticClassFallbackAdapter:
    name = "semantic_class_fallback"

    def score(self, raw: RawMaterialSpec) -> int:
        return 1

    def convert(self, raw: RawMaterialSpec, context: MaterialContext) -> PreviewMaterialSpec:
        material_class = find_material_class(raw.material_path, raw.connected_surface_shader_id)
        base_color = CLASS_FALLBACKS.get(material_class, (0.5, 0.5, 0.5))
        opacity_value = _float_for_alias(raw, ("opacity", "opacity_constant"))
        ior_value = _float_for_alias(raw, IOR_ALIASES)
        return PreviewMaterialSpec(
            material_path=raw.material_path,
            material_name=raw.material_name,
            source_shader_ids=raw.shader_ids,
            mdl_source_asset=raw.mdl_source_asset,
            base_color=InputSpec(value=base_color),
            normal=None,
            metallic=None,
            roughness=None,
            specular_color=None,
            emissive_color=None,
            opacity=InputSpec(value=opacity_value) if opacity_value is not None else None,
            ior=ior_value,
            material_class=material_class,
            conversion_policy="class_fallback",
            quality_notes=("semantic class fallback used",),
        )


def _float_for_alias(raw: RawMaterialSpec, aliases: tuple[str, ...]) -> float | None:
    for alias in aliases:
        if alias not in raw.values:
            continue
        value = coerce_float(raw.values.get(alias))
        if value is not None:
            return value
    return None
