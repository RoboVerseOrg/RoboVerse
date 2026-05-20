"""Policy adapter for complex MDL materials that require Kit baking."""

from __future__ import annotations

from .. import kit_bake
from ..aliases import IOR_ALIASES, TEXTURE_ALIASES
from ..context import MaterialContext
from ..extract import asset_path_string
from ..schema import ConversionPolicy, PreviewMaterialSpec, RawMaterialSpec
from ..slots import SLOT_REGISTRY

COMPLEX_MDL_HINTS = ("procedural", "noise", "clearcoat", "flake", "subsurface", "layer", "blend")

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


def choose_conversion_policy(raw: RawMaterialSpec) -> ConversionPolicy:
    # Mirror ExistingPreviewSurfaceAdapter so policy-first selection preserves existing graphs.
    shader_ids = set(raw.shader_ids)
    if raw.connected_surface_shader_id:
        shader_ids.add(raw.connected_surface_shader_id)
    if "UsdPreviewSurface" in shader_ids:
        return "preserve_existing_preview"

    if raw.mdl_source_asset and _has_procedural_or_complex_nodes(raw):
        return "mdl_bake"

    if _looks_like_simple_omnipbr_file_texture_material(raw):
        return "direct_graph"

    if _has_direct_texture_alias(raw):
        return "direct_graph"

    if raw.values:
        return "scalar_fallback"
    return "class_fallback"


class MdlBakeAdapter:
    name = "mdl_bake"

    def score(self, raw: RawMaterialSpec) -> float:
        return 90.0 if choose_conversion_policy(raw) == "mdl_bake" else 0.0

    def convert(self, raw: RawMaterialSpec, context: MaterialContext) -> PreviewMaterialSpec:
        return kit_bake.bake_mdl_material_to_preview(raw, context)


def _looks_like_simple_omnipbr_file_texture_material(raw: RawMaterialSpec) -> bool:
    if not _is_omnipbr_like(raw):
        return False

    known_keys = _known_simple_omnipbr_keys()
    return all(key in known_keys for key in raw.values)


def _has_procedural_or_complex_nodes(raw: RawMaterialSpec) -> bool:
    candidates = [*raw.shader_ids, *raw.values.keys()]
    if raw.connected_surface_shader_id:
        candidates.append(raw.connected_surface_shader_id)
    return any(hint in candidate.lower() for candidate in candidates for hint in COMPLEX_MDL_HINTS)


def _has_direct_texture_alias(raw: RawMaterialSpec) -> bool:
    return any(asset_path_string(raw.values.get(alias)) for alias in TEXTURE_ALIASES)


def _is_omnipbr_like(raw: RawMaterialSpec) -> bool:
    if raw.connected_surface_shader_id:
        return "omnipbr" in raw.connected_surface_shader_id.lower()

    candidates = [*raw.shader_ids]
    if raw.mdl_source_asset:
        candidates.append(raw.mdl_source_asset)
    return any("omnipbr" in candidate.lower() for candidate in candidates)


def _known_simple_omnipbr_keys() -> set[str]:
    keys: set[str] = {"MaxTexCoordIndex", *IOR_ALIASES}
    for slot_name, slot in SLOT_REGISTRY.items():
        keys.update(slot.value_aliases)
        keys.update(slot.texture_aliases)
        keys.update(slot.enable_aliases)
        keys.update(slot.uva_aliases)
        keys.update(getattr(slot, "intensity_aliases", ()))

        prefix = _SLOT_PREFIXES.get(slot_name)
        if prefix:
            keys.add(f"{prefix}_MaxTexCoordIndex")
    return keys
