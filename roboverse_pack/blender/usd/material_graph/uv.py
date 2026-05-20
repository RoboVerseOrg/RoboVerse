"""UV transform parsing helpers for OmniPBR-style UVA inputs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

from .schema import UVTransformSpec


def _float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _pair(value: Any) -> tuple[float, float] | None:
    scalar = _float(value)
    if scalar is not None:
        return (scalar, scalar)
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return None
    if len(value) < 2:
        return None
    first = _float(value[0])
    second = _float(value[1])
    if first is None or second is None:
        return None
    return (first, second)


def _first_mapping_value(value: Mapping[str, Any], names: tuple[str, ...]) -> Any:
    for name in names:
        if name in value and value[name] is not None:
            return value[name]
    return None


def parse_uva(value: Any) -> UVTransformSpec | None:
    """Parse common UVA payloads.

    Sequence values follow the OmniPBR-style convention:
    float4 = (uScale, vScale, uOffset, vOffset), float3 = (uniformScale, uOffset, vOffset).
    """

    if isinstance(value, Mapping):
        scale = _pair(_first_mapping_value(value, ("scale",))) or (1.0, 1.0)
        translation = _pair(_first_mapping_value(value, ("translation", "offset"))) or (0.0, 0.0)
        rotation = _float(_first_mapping_value(value, ("rotation", "rotation_degrees"))) or 0.0
        return UVTransformSpec(scale=scale, rotation_degrees=rotation, translation=translation)

    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return None

    try:
        if len(value) == 4:
            scale_u = _float(value[0])
            scale_v = _float(value[1])
            translate_u = _float(value[2])
            translate_v = _float(value[3])
            if None in (scale_u, scale_v, translate_u, translate_v):
                return None
            return UVTransformSpec(
                scale=(scale_u, scale_v),  # type: ignore[arg-type]
                translation=(translate_u, translate_v),  # type: ignore[arg-type]
            )
        if len(value) == 3:
            scale = _float(value[0])
            translate_u = _float(value[1])
            translate_v = _float(value[2])
            if None in (scale, translate_u, translate_v):
                return None
            return UVTransformSpec(
                scale=(scale, scale),  # type: ignore[arg-type]
                translation=(translate_u, translate_v),  # type: ignore[arg-type]
            )
    except (TypeError, IndexError):
        return None
    return None


def resolve_slot_uv_transform(
    values: Mapping[str, object],
    prefix: str,
    uva_aliases: tuple[str, ...],
) -> UVTransformSpec | None:
    for alias in uva_aliases:
        if alias not in values:
            continue
        spec = parse_uva(values.get(alias))
        if spec is not None:
            return replace(spec, source_input=alias)

    field_names = (
        f"{prefix}_UScale",
        f"{prefix}_VScale",
        f"{prefix}_Rotation",
        f"{prefix}_UOffset",
        f"{prefix}_VOffset",
    )
    if not any(name in values and values.get(name) is not None for name in field_names):
        return None

    u_scale = _float(values.get(f"{prefix}_UScale"))
    v_scale = _float(values.get(f"{prefix}_VScale"))
    rotation = _float(values.get(f"{prefix}_Rotation"))
    u_offset = _float(values.get(f"{prefix}_UOffset"))
    v_offset = _float(values.get(f"{prefix}_VOffset"))

    return UVTransformSpec(
        scale=(1.0 if u_scale is None else u_scale, 1.0 if v_scale is None else v_scale),
        rotation_degrees=0.0 if rotation is None else rotation,
        translation=(0.0 if u_offset is None else u_offset, 0.0 if v_offset is None else v_offset),
        source_input=f"{prefix}_*",
    )
