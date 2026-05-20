"""Declarative source input slots for OmniPBR-like material graphs."""

from __future__ import annotations

from dataclasses import dataclass

from .aliases import (
    COLOR_ALIASES,
    EMISSIVE_ALIASES,
    METALLIC_ALIASES,
    OPACITY_ALIASES,
    ROUGHNESS_ALIASES,
    TEXTURE_ALIASES,
)
from .schema import ColorSpace, TextureChannel


@dataclass(frozen=True)
class SlotSpec:
    name: str
    value_aliases: tuple[str, ...] = ()
    texture_aliases: tuple[str, ...] = ()
    enable_aliases: tuple[str, ...] = ()
    uva_aliases: tuple[str, ...] = ()
    color_space: ColorSpace = "raw"
    texture_output: TextureChannel = "rgb"
    scale: tuple[float, float, float, float] | None = None
    bias: tuple[float, float, float, float] | None = None
    invert_to_roughness: bool = False
    requires_specular_workflow: bool = False


SLOT_REGISTRY: dict[str, SlotSpec] = {
    "base_color": SlotSpec(
        name="base_color",
        value_aliases=COLOR_ALIASES,
        texture_aliases=TEXTURE_ALIASES,
        enable_aliases=("enable_base_color", "BaseColor_Enable", "base_color_enabled"),
        uva_aliases=("BaseColor_UVA", "Diffuse_UVA", "BaseColor_UV", "BaseColor_UVSet", "base_color_uv"),
        color_space="sRGB",
    ),
    "normal": SlotSpec(
        name="normal",
        texture_aliases=("Normal_Tex", "Normal_Texture", "normal_texture", "normalmap_texture", "bump_texture"),
        enable_aliases=("enable_normal", "Normal_Enable", "normal_enabled"),
        uva_aliases=("Normal_UVA", "Normal_UV", "Normal_UVSet", "normal_uv"),
        color_space="raw",
        scale=(2.0, 2.0, 2.0, 1.0),
        bias=(-1.0, -1.0, -1.0, 0.0),
    ),
    "metallic": SlotSpec(
        name="metallic",
        value_aliases=METALLIC_ALIASES,
        texture_aliases=("Metallic_Tex", "Metallic_Texture", "metallic_texture"),
        enable_aliases=("enable_metallic", "Metallic_Enable", "metallic_enabled"),
        uva_aliases=("Metallic_UVA", "Metallic_UV", "Metallic_UVSet", "metallic_uv"),
        texture_output="r",
    ),
    "roughness": SlotSpec(
        name="roughness",
        value_aliases=ROUGHNESS_ALIASES,
        texture_aliases=("Roughness_Tex", "Roughness_Texture", "roughness_texture"),
        enable_aliases=("enable_roughness", "Roughness_Enable", "roughness_enabled"),
        uva_aliases=("Roughness_UVA", "Roughness_UV", "Roughness_UVSet", "roughness_uv"),
        texture_output="r",
    ),
    "glossiness": SlotSpec(
        name="glossiness",
        value_aliases=("glossiness", "Glossiness", "Gloss_Color", "glossiness_constant", "reflection_glossiness_constant"),
        texture_aliases=("Glossiness_Tex", "Glossiness_Texture", "Gloss_Tex", "glossiness_texture", "gloss_texture"),
        enable_aliases=("enable_glossiness", "Glossiness_Enable", "glossiness_enabled"),
        uva_aliases=("Gloss_UVA", "Glossiness_UV", "Glossiness_UVSet", "glossiness_uv"),
        texture_output="r",
        invert_to_roughness=True,
    ),
    "specular": SlotSpec(
        name="specular",
        value_aliases=("specular", "specularColor", "Specular_Color", "specular_color"),
        texture_aliases=("Specular_Tex", "Specular_Texture", "specular_texture"),
        enable_aliases=("enable_specular", "Specular_Enable", "specular_enabled"),
        uva_aliases=("Specular_UVA", "Specular_UV", "Specular_UVSet", "specular_uv"),
        color_space="sRGB",
        requires_specular_workflow=True,
    ),
    "emissive": SlotSpec(
        name="emissive",
        value_aliases=EMISSIVE_ALIASES + ("Emission_Color",),
        texture_aliases=(
            "Emissive_Tex",
            "Emissive_Texture",
            "Emission_Tex",
            "Emission_Texture",
            "emissive_texture",
            "emission_texture",
        ),
        enable_aliases=("enable_emissive", "Emissive_Enable", "emissive_enabled"),
        uva_aliases=("Emissive_UVA", "Emission_UVA", "Emissive_UV", "Emissive_UVSet", "emissive_uv"),
        color_space="sRGB",
    ),
    "opacity": SlotSpec(
        name="opacity",
        value_aliases=OPACITY_ALIASES,
        texture_aliases=("Opacity_Tex", "Opacity_Texture", "Alpha_Tex", "opacity_texture", "alpha_texture"),
        enable_aliases=("enable_opacity", "Opacity_Enable", "opacity_enabled"),
        uva_aliases=("Opacity_UVA", "Alpha_UVA", "Opacity_UV", "Opacity_UVSet", "opacity_uv"),
        texture_output="a_or_r",
    ),
}
