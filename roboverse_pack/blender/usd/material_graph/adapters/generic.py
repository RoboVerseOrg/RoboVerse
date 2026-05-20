"""Adapter for simple non-Omni texture graphs."""

from __future__ import annotations

from dataclasses import replace

from ..aliases import TEXTURE_ALIASES
from ..context import MaterialContext
from ..extract import asset_path_string
from ..normalize import normalize_material
from ..schema import PreviewMaterialSpec, RawMaterialSpec


class GenericTextureGraphAdapter:
    name = "generic_texture_graph"

    def score(self, raw: RawMaterialSpec) -> int:
        for alias in TEXTURE_ALIASES:
            if asset_path_string(raw.values.get(alias)):
                return 40
        return 0

    def convert(self, raw: RawMaterialSpec, context: MaterialContext) -> PreviewMaterialSpec:
        return replace(normalize_material(raw), conversion_policy="direct_graph")
