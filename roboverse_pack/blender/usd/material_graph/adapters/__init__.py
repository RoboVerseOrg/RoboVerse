"""Material graph adapter registry.

Score table:
- existing_preview: 100
- omnipbr direct graph: 80
- generic texture graph: 40
- scalar fallback: 20
- semantic class fallback: 1

MDL baking is a later policy-first path, not score-first adapter selection.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from ..context import MaterialContext
from ..schema import PreviewMaterialSpec, RawMaterialSpec
from .base import MaterialAdapter
from .existing_preview import ExistingPreviewSurfaceAdapter
from .fallback import ScalarFallbackAdapter, SemanticClassFallbackAdapter
from .generic import GenericTextureGraphAdapter
from .omnipbr import OmniPBRAdapter

ADAPTERS: tuple[MaterialAdapter, ...] = (
    ExistingPreviewSurfaceAdapter(),
    OmniPBRAdapter(),
    GenericTextureGraphAdapter(),
    ScalarFallbackAdapter(),
    SemanticClassFallbackAdapter(),
)


def select_adapter(
    raw: RawMaterialSpec,
    adapters: Sequence[MaterialAdapter] | None = None,
) -> MaterialAdapter:
    registry = adapters if adapters is not None else ADAPTERS
    return max(registry, key=lambda adapter: adapter.score(raw))


def convert_material(
    raw: RawMaterialSpec,
    context: MaterialContext,
    adapters: Sequence[MaterialAdapter] | None = None,
) -> PreviewMaterialSpec:
    adapter = select_adapter(raw, adapters)
    return replace(adapter.convert(raw, context), adapter_name=adapter.name)
