"""Normalize source material inputs into preview authoring parameters."""

from __future__ import annotations

from typing import Any

from .aliases import (
    COLOR_ALIASES,
    EMISSIVE_ALIASES,
    IOR_ALIASES,
    METALLIC_ALIASES,
    OPACITY_ALIASES,
    ROUGHNESS_ALIASES,
    TEXTURE_ALIASES,
)
from .extract import asset_path_string, coerce_float, coerce_vec3, shader_input
from .fallback import CLASS_FALLBACKS, find_material_class


def normalize_preview_parameters(source_shader: Any, material_path: str, shader_id: str | None, Gf: Any) -> dict[str, Any]:
    """Return preview authoring params; ``None`` values mean skip the input."""
    material_class = find_material_class(material_path, shader_id)
    warnings = []

    diffuse_color = coerce_vec3(shader_input(source_shader, COLOR_ALIASES), Gf)
    diffuse_texture = asset_path_string(shader_input(source_shader, TEXTURE_ALIASES))
    if not diffuse_texture:
        if diffuse_color is None and material_class:
            diffuse_color = Gf.Vec3f(*CLASS_FALLBACKS[material_class])
        if diffuse_color is None:
            warnings.append("No diffuse color or texture alias found.")

    return {
        "diffuse_color": diffuse_color,
        "diffuse_texture": diffuse_texture,
        "roughness": coerce_float(shader_input(source_shader, ROUGHNESS_ALIASES)),
        "metallic": coerce_float(shader_input(source_shader, METALLIC_ALIASES)),
        "opacity": coerce_float(shader_input(source_shader, OPACITY_ALIASES)),
        "ior": coerce_float(shader_input(source_shader, IOR_ALIASES)),
        "emissive": coerce_vec3(shader_input(source_shader, EMISSIVE_ALIASES), Gf),
        "material_class": material_class,
        "warnings": warnings,
    }

