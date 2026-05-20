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
from .mdl_bake import MdlBakeAdapter, choose_conversion_policy
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


def select_adapter_by_policy(raw: RawMaterialSpec, *, enable_mdl_bake: bool = False) -> MaterialAdapter:
    policy = choose_conversion_policy(raw)
    if policy == "preserve_existing_preview":
        return ExistingPreviewSurfaceAdapter()
    if policy == "mdl_bake":
        if enable_mdl_bake:
            return MdlBakeAdapter()
        if raw.values:
            return ScalarFallbackAdapter()
        return SemanticClassFallbackAdapter()
    if policy == "direct_graph":
        omnipbr = OmniPBRAdapter()
        if omnipbr.score(raw) > 0:
            return omnipbr
        return GenericTextureGraphAdapter()
    if policy == "scalar_fallback":
        return ScalarFallbackAdapter()
    return SemanticClassFallbackAdapter()


def convert_material(
    raw: RawMaterialSpec,
    context: MaterialContext,
    adapters: Sequence[MaterialAdapter] | None = None,
) -> PreviewMaterialSpec:
    if adapters is not None:
        adapter = select_adapter(raw, adapters)
        return replace(adapter.convert(raw, context), adapter_name=adapter.name)

    policy = choose_conversion_policy(raw)
    adapter = select_adapter_by_policy(raw, enable_mdl_bake=context.enable_mdl_bake)
    spec = replace(adapter.convert(raw, context), adapter_name=adapter.name)
    if policy == "mdl_bake" and not context.enable_mdl_bake:
        return replace(spec, quality_notes=(*spec.quality_notes, "mdl_bake_unavailable"))
    return spec
