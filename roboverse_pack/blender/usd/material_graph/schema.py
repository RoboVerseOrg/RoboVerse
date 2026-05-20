"""Typed intermediate representation for USD material graph conversion."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, Mapping

ColorSpace = Literal["sRGB", "raw", "auto"]
TextureChannel = Literal["rgb", "r", "g", "b", "a", "a_or_r"]
ConversionPolicy = Literal[
    "preserve_existing_preview",
    "direct_graph",
    "mdl_bake",
    "scalar_fallback",
    "class_fallback",
    "failed",
]


@dataclass(frozen=True)
class UVTransformSpec:
    scale: tuple[float, float] = (1.0, 1.0)
    rotation_degrees: float = 0.0
    translation: tuple[float, float] = (0.0, 0.0)
    source_input: str | None = None


@dataclass(frozen=True)
class UVSetSpec:
    primvar_name: str = "st"
    requested_index: int | None = None
    resolution_status: str = "default"


@dataclass(frozen=True)
class TextureSpec:
    file: str
    source_color_space: ColorSpace
    channel: TextureChannel = "rgb"
    uv_set: UVSetSpec = field(default_factory=UVSetSpec)
    uv_transform: UVTransformSpec | None = None
    scale: tuple[float, float, float, float] | None = None
    bias: tuple[float, float, float, float] | None = None
    wrap_s: str | None = None
    wrap_t: str | None = None
    role: str | None = None
    source_input: str | None = None


@dataclass(frozen=True)
class InputSpec:
    value: object | None = None
    texture: TextureSpec | None = None
    source_inputs: tuple[str, ...] = ()


@dataclass(frozen=True)
class RawMaterialSpec:
    material_path: str
    material_name: str
    shader_ids: tuple[str, ...]
    connected_surface_shader_id: str | None
    mdl_source_asset: str | None
    values: Mapping[str, object]


@dataclass(frozen=True)
class PreviewMaterialSpec:
    material_path: str
    material_name: str
    source_shader_ids: tuple[str, ...]
    mdl_source_asset: str | None
    base_color: InputSpec
    normal: InputSpec | None
    metallic: InputSpec | None
    roughness: InputSpec | None
    specular_color: InputSpec | None
    emissive_color: InputSpec | None
    opacity: InputSpec | None
    ior: float | None = None
    use_specular_workflow: bool = False
    material_class: str | None = None
    conversion_policy: ConversionPolicy = "direct_graph"
    quality_notes: tuple[str, ...] = ()


def freeze_values(values: Mapping[str, object]) -> Mapping[str, object]:
    return MappingProxyType(dict(values))
