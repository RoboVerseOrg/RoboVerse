"""Fallback material classification for Blender USD preview authoring."""

from __future__ import annotations

CLASS_FALLBACKS = {
    "wall": (0.72, 0.72, 0.68),
    "floor": (0.55, 0.50, 0.44),
    "cabinet": (0.60, 0.46, 0.32),
    "glass": (0.78, 0.90, 0.96),
    "screen": (0.02, 0.02, 0.025),
    "metal": (0.55, 0.55, 0.55),
    "fabric": (0.48, 0.42, 0.38),
}


def find_material_class(material_path: str, shader_id: str | None) -> str | None:
    text = f"{material_path} {shader_id or ''}".lower()
    for material_class in CLASS_FALLBACKS:
        if material_class in text:
            return material_class
    return None


def is_glass_material(material_class: str | None) -> bool:
    return material_class == "glass"


def is_default_gray_color(value: object) -> bool:
    if value is None:
        return False
    try:
        return tuple(float(value[index]) for index in range(3)) == (0.5, 0.5, 0.5)
    except (TypeError, ValueError, IndexError):
        return False

