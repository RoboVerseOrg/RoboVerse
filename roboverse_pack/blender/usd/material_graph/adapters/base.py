"""Material graph adapter protocol."""

from __future__ import annotations

from typing import Protocol

from ..context import MaterialContext
from ..schema import PreviewMaterialSpec, RawMaterialSpec


class MaterialAdapter(Protocol):
    name: str

    def score(self, raw: RawMaterialSpec) -> int:
        """Return adapter confidence for a raw material."""

    def convert(self, raw: RawMaterialSpec, context: MaterialContext) -> PreviewMaterialSpec:
        """Convert a raw material into the preview material IR."""
