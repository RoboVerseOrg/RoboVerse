"""Shader input aliases used by Blender USD material overlay generation."""

from __future__ import annotations

COLOR_ALIASES = ("diffuseColor", "BaseColor_Color", "diffuse_color_constant", "base_color")
TEXTURE_ALIASES = (
    "BaseColor_Tex",
    "BaseColor_Texture",
    "diffuse_texture",
    "albedo_texture",
    "base_color_texture",
    "diffuseColor_texture",
)
ROUGHNESS_ALIASES = ("roughness", "reflection_roughness_constant")
METALLIC_ALIASES = ("metallic", "metallic_constant")
OPACITY_ALIASES = ("opacity", "opacity_constant")
EMISSIVE_ALIASES = ("emissiveColor", "Emissive_Color")
IOR_ALIASES = ("ior", "glass_ior", "index_of_refraction")

