"""Adapter for source graphs that already contain UsdPreviewSurface."""

from __future__ import annotations

from dataclasses import replace

from ..context import MaterialContext
from ..normalize import normalize_material
from ..schema import PreviewMaterialSpec, RawMaterialSpec


class ExistingPreviewSurfaceAdapter:
    name = "existing_preview"

    def score(self, raw: RawMaterialSpec) -> int:
        shader_ids = set(raw.shader_ids)
        if raw.connected_surface_shader_id:
            shader_ids.add(raw.connected_surface_shader_id)
        return 100 if "UsdPreviewSurface" in shader_ids else 0

    def convert(self, raw: RawMaterialSpec, context: MaterialContext) -> PreviewMaterialSpec:
        spec = normalize_material(raw)
        return replace(
            spec,
            conversion_policy="preserve_existing_preview",
            quality_notes=("source UsdPreviewSurface graph preserved without overlay opinion",),
        )
