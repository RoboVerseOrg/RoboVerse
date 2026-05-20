"""Report helpers for Blender USD material overlay conversion."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from .schema import InputSpec, PreviewMaterialSpec, TextureSpec, UVTransformSpec


def material_entry(material_path: str) -> dict[str, Any]:
    return {
        "status": "converted",
        "warnings": [],
        "material_class": None,
    }


def adapter_from_policy(policy: str) -> str:
    return {
        "preserve_existing_preview": "existing_preview",
        "direct_graph": "omnipbr",
        "mdl_bake": "mdl_bake",
        "scalar_fallback": "scalar_fallback",
        "class_fallback": "semantic_class_fallback",
        "failed": "failed",
    }.get(policy, "unknown")


def material_entry_from_spec(spec: PreviewMaterialSpec) -> dict[str, Any]:
    status = "skipped" if spec.conversion_policy == "preserve_existing_preview" else "converted"
    return {
        "status": status,
        "warnings": list(spec.quality_notes),
        "material_class": spec.material_class,
    }


def failed_material_entry(error: Exception | str) -> dict[str, Any]:
    warning = str(error)
    if not warning.startswith("conversion failed: "):
        warning = f"conversion failed: {warning}"
    return {
        "status": "failed",
        "warnings": [warning],
        "material_class": None,
    }


_SLOT_FIELDS = {
    "base_color": "base_color",
    "normal": "normal",
    "metallic": "metallic",
    "roughness": "roughness",
    "specular": "specular_color",
    "emissive": "emissive_color",
    "opacity": "opacity",
}


def _empty_summary() -> dict[str, int]:
    return {
        "materials_total": 0,
        "preserve_existing_preview": 0,
        "converted_by_direct_graph": 0,
        "converted_by_mdl_distill_bake": 0,
        "scalar_fallback": 0,
        "class_fallback": 0,
        "failed": 0,
    }


def _empty_texture_slots() -> dict[str, dict[str, int]]:
    slots = {slot: {"texture": 0, "scalar": 0, "missing": 0} for slot in _SLOT_FIELDS}
    slots["glossiness"] = {"texture_inverted": 0, "scalar_inverted": 0}
    return slots


def _empty_uv() -> dict[str, int]:
    return {
        "uv_transforms_found": 0,
        "uv_transforms_converted": 0,
        "multi_uv_requested": 0,
        "multi_uv_resolved": 0,
        "multi_uv_unresolved": 0,
    }


def _empty_udim() -> dict[str, int]:
    return {"materials_with_udim": 0, "normalized": 0}


def _uv_transform_dict(transform: UVTransformSpec) -> dict[str, Any]:
    return {
        "scale": list(transform.scale),
        "rotation_degrees": transform.rotation_degrees,
        "translation": list(transform.translation),
        "source_input": transform.source_input,
    }


def _texture_dict(texture: TextureSpec) -> dict[str, Any]:
    data: dict[str, Any] = {
        "file": texture.file,
        "source_color_space": texture.source_color_space,
        "channel": texture.channel,
        "uv_set": {
            "primvar_name": texture.uv_set.primvar_name,
            "requested_index": texture.uv_set.requested_index,
            "resolution_status": texture.uv_set.resolution_status,
        },
    }
    if texture.source_input is not None:
        data["source_input"] = texture.source_input
    if texture.uv_transform is not None:
        data["uv_transform"] = _uv_transform_dict(texture.uv_transform)
    if texture.scale is not None:
        data["scale"] = list(texture.scale)
    if texture.bias is not None:
        data["bias"] = list(texture.bias)
    return data


def _slot_entry(slot: InputSpec | None) -> dict[str, Any]:
    if slot is None:
        return {"status": "missing"}
    if slot.texture is not None:
        data = {"status": "texture", "source_inputs": list(slot.source_inputs)}
        data.update(_texture_dict(slot.texture))
        return data
    if slot.value is not None:
        return {
            "status": "scalar",
            "value": _jsonable(slot.value),
            "source_inputs": list(slot.source_inputs),
        }
    return {"status": "missing", "source_inputs": list(slot.source_inputs)}


class ConversionReport:
    def __init__(self, **metadata: Any) -> None:
        self._metadata = dict(metadata)
        self._materials: list[dict[str, Any]] = []

    def add_material(self, spec: PreviewMaterialSpec) -> None:
        slots = {
            slot_name: _slot_entry(getattr(spec, field_name))
            for slot_name, field_name in _SLOT_FIELDS.items()
        }
        self._materials.append(
            {
                "material_path": spec.material_path,
                "adapter": spec.adapter_name or adapter_from_policy(spec.conversion_policy),
                "policy": spec.conversion_policy,
                "slots": slots,
                "warnings": list(spec.quality_notes),
            }
        )

    def add_failed_material(self, material_path: str, warning: str) -> None:
        self._materials.append(
            {
                "material_path": material_path,
                "adapter": adapter_from_policy("failed"),
                "policy": "failed",
                "slots": {},
                "warnings": [warning],
            }
        )

    def to_dict(self) -> dict[str, Any]:
        summary = _empty_summary()
        texture_slots = _empty_texture_slots()
        uv = _empty_uv()
        udim = _empty_udim()

        materials: list[dict[str, Any]] = []
        for material in self._materials:
            materials.append(deepcopy(material))
            summary["materials_total"] += 1
            policy = material["policy"]
            if policy == "direct_graph":
                summary["converted_by_direct_graph"] += 1
            elif policy == "mdl_bake":
                summary["converted_by_mdl_distill_bake"] += 1
            elif policy in summary:
                summary[policy] += 1

            material_has_udim = False
            for slot_name, slot in material.get("slots", {}).items():
                if slot_name in texture_slots and slot["status"] in texture_slots[slot_name]:
                    texture_slots[slot_name][slot["status"]] += 1
                if slot_name == "roughness" and _slot_has_glossiness_conversion(slot):
                    key = "texture_inverted" if slot["status"] == "texture" else "scalar_inverted"
                    texture_slots["glossiness"][key] += 1
                if slot["status"] != "texture":
                    continue
                if slot.get("uv_transform") is not None:
                    uv["uv_transforms_found"] += 1
                    uv["uv_transforms_converted"] += 1
                uv_set = slot.get("uv_set", {})
                if uv_set.get("requested_index") is not None:
                    uv["multi_uv_requested"] += 1
                    if uv_set.get("resolution_status") in {"matched", "resolved"}:
                        uv["multi_uv_resolved"] += 1
                    else:
                        uv["multi_uv_unresolved"] += 1
                if "<UDIM>" in slot.get("file", ""):
                    material_has_udim = True
            if material_has_udim:
                udim["materials_with_udim"] += 1
                udim["normalized"] += 1

        data: dict[str, Any] = dict(self._metadata)
        data.update(
            {
                "summary": summary,
                "texture_slots": texture_slots,
                "uv": uv,
                "udim": udim,
                "materials": materials,
            }
        )
        return data


def _slot_has_glossiness_conversion(slot: dict[str, Any]) -> bool:
    sources = list(slot.get("source_inputs", []))
    if slot.get("source_input"):
        sources.append(slot["source_input"])
    return any("gloss" in source.lower() for source in sources)


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, tuple | list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    try:
        return [_jsonable(item) for item in value]
    except TypeError:
        return str(value)


def _report_dict(report: dict[str, Any] | ConversionReport) -> dict[str, Any]:
    if isinstance(report, ConversionReport):
        return report.to_dict()
    return report


def write_conversion_report_md(report: dict[str, Any] | ConversionReport, path: str | Path) -> None:
    report_data = _report_dict(report)
    lines = [
        "# Blender Material Overlay Conversion Report",
        "",
        f"Input: `{report_data.get('input_path', '')}`",
        f"Overlay: `{report_data.get('overlay_path', '')}`",
        f"Root: `{report_data.get('root_path', '')}`",
        "",
        "| Material | Status | Class | Warnings |",
        "| --- | --- | --- | --- |",
    ]
    materials = report_data.get("materials", {})
    if isinstance(materials, dict):
        for material_path, entry in sorted(materials.items()):
            warnings = "; ".join(entry.get("warnings", []))
            lines.append(
                f"| `{material_path}` | {entry.get('status', '')} | {entry.get('material_class') or ''} | {warnings} |"
            )
    else:
        for entry in sorted(materials, key=lambda item: item.get("material_path", "")):
            warnings = "; ".join(entry.get("warnings", []))
            status = "failed" if entry.get("policy") == "failed" else entry.get("policy", "")
            lines.append(f"| `{entry.get('material_path', '')}` | {status} |  | {warnings} |")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_conversion_reports(report: dict[str, Any] | ConversionReport, cache: str | Path) -> None:
    output_dir = Path(cache)
    report_json = output_dir / "conversion_report.json"
    report_md = output_dir / "conversion_report.md"
    report_data = _report_dict(report)
    report_json.write_text(json.dumps(report_data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_conversion_report_md(report, report_md)
