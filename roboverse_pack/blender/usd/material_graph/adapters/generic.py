"""Adapter for simple non-Omni texture graphs."""

from __future__ import annotations

from dataclasses import replace

from ..aliases import TEXTURE_ALIASES
from ..context import MaterialContext
from ..extract import asset_path_string
from ..normalize import normalize_material
from ..schema import PreviewMaterialSpec, RawMaterialSpec
from ..texture_paths import normalize_texture_asset_path


class GenericTextureGraphAdapter:
    name = "generic_texture_graph"

    def score(self, raw: RawMaterialSpec) -> int:
        for alias in TEXTURE_ALIASES:
            if asset_path_string(raw.values.get(alias)):
                return 40
        return 0

    def convert(self, raw: RawMaterialSpec, context: MaterialContext) -> PreviewMaterialSpec:
        spec = replace(normalize_material(raw), conversion_policy="direct_graph")
        texture = spec.base_color.texture
        if texture is None:
            return spec

        path_result = normalize_texture_asset_path(texture.file, context.texture_base_dir)
        base_color = replace(
            spec.base_color,
            texture=replace(texture, file=path_result.normalized, role="base_color"),
        )
        return replace(
            spec,
            base_color=base_color,
            quality_notes=(*spec.quality_notes, *path_result.warnings),
        )
