"""Normalize source material inputs into preview authoring parameters."""

from __future__ import annotations

from typing import Mapping

from .aliases import (
    COLOR_ALIASES,
    EMISSIVE_ALIASES,
    IOR_ALIASES,
    METALLIC_ALIASES,
    OPACITY_ALIASES,
    ROUGHNESS_ALIASES,
    TEXTURE_ALIASES,
)
from .extract import asset_path_string, coerce_float
from .fallback import CLASS_FALLBACKS, find_material_class
from .schema import ConversionPolicy, InputSpec, PreviewMaterialSpec, RawMaterialSpec, TextureSpec


def _value_for_alias(values: Mapping[str, object], aliases: tuple[str, ...]) -> tuple[object | None, str | None]:
    for alias in aliases:
        value = values.get(alias)
        if value is not None:
            return value, alias
    return None, None


def _coerce_vec3_tuple(value: object | None) -> tuple[float, float, float] | None:
    if value is None:
        return None
    try:
        return (float(value[0]), float(value[1]), float(value[2]))  # type: ignore[index]
    except (TypeError, ValueError, IndexError):
        return None


def _scalar_input(value: object | None, source: str | None) -> InputSpec | None:
    coerced = coerce_float(value)
    if coerced is None:
        return None
    return InputSpec(value=coerced, source_inputs=(source,) if source else ())


def _vec3_input(value: object | None, source: str | None) -> InputSpec | None:
    coerced = _coerce_vec3_tuple(value)
    if coerced is None:
        return None
    return InputSpec(value=coerced, source_inputs=(source,) if source else ())


def normalize_material(raw: RawMaterialSpec) -> PreviewMaterialSpec:
    material_class = find_material_class(raw.material_path, raw.connected_surface_shader_id)
    quality_notes: list[str] = []
    conversion_policy: ConversionPolicy = "direct_graph"

    diffuse_value, diffuse_source = _value_for_alias(raw.values, COLOR_ALIASES)
    diffuse_color = _coerce_vec3_tuple(diffuse_value)
    texture_value, texture_source = _value_for_alias(raw.values, TEXTURE_ALIASES)
    diffuse_texture = asset_path_string(texture_value)

    if diffuse_texture:
        base_color = InputSpec(
            texture=TextureSpec(
                file=diffuse_texture,
                source_color_space="sRGB",
                source_input=texture_source,
            ),
            source_inputs=(texture_source,) if texture_source else (),
        )
    else:
        if diffuse_color is None and material_class:
            diffuse_color = CLASS_FALLBACKS[material_class]
            conversion_policy = "class_fallback"
        if diffuse_color is None:
            quality_notes.append("No diffuse color or texture alias found.")
        base_color = InputSpec(value=diffuse_color, source_inputs=(diffuse_source,) if diffuse_source else ())

    roughness_value, roughness_source = _value_for_alias(raw.values, ROUGHNESS_ALIASES)
    metallic_value, metallic_source = _value_for_alias(raw.values, METALLIC_ALIASES)
    opacity_value, opacity_source = _value_for_alias(raw.values, OPACITY_ALIASES)
    ior_value, ior_source = _value_for_alias(raw.values, IOR_ALIASES)
    emissive_value, emissive_source = _value_for_alias(raw.values, EMISSIVE_ALIASES)

    roughness = _scalar_input(roughness_value, roughness_source)
    metallic = _scalar_input(metallic_value, metallic_source)
    opacity = _scalar_input(opacity_value, opacity_source)
    emissive_color = _vec3_input(emissive_value, emissive_source)
    ior = coerce_float(ior_value)

    return PreviewMaterialSpec(
        material_path=raw.material_path,
        material_name=raw.material_name,
        source_shader_ids=raw.shader_ids,
        mdl_source_asset=raw.mdl_source_asset,
        base_color=base_color,
        normal=None,
        metallic=metallic,
        roughness=roughness,
        specular_color=None,
        emissive_color=emissive_color,
        opacity=opacity,
        ior=ior,
        material_class=material_class,
        conversion_policy=conversion_policy,
        quality_notes=tuple(quality_notes),
    )
