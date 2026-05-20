"""Report helpers for Blender USD material overlay conversion."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schema import PreviewMaterialSpec


def material_entry(material_path: str) -> dict[str, Any]:
    return {
        "status": "converted",
        "warnings": [],
        "material_class": None,
    }


def material_entry_from_spec(spec: PreviewMaterialSpec) -> dict[str, Any]:
    status = "skipped" if spec.conversion_policy == "preserve_existing_preview" else "converted"
    return {
        "status": status,
        "warnings": list(spec.quality_notes),
        "material_class": spec.material_class,
    }


def failed_material_entry(error: Exception) -> dict[str, Any]:
    return {
        "status": "failed",
        "warnings": [f"conversion failed: {error}"],
        "material_class": None,
    }


def write_conversion_report_md(report: dict[str, Any], path: str | Path) -> None:
    lines = [
        "# Blender Material Overlay Conversion Report",
        "",
        f"Input: `{report.get('input_path', '')}`",
        f"Overlay: `{report.get('overlay_path', '')}`",
        f"Root: `{report.get('root_path', '')}`",
        "",
        "| Material | Status | Class | Warnings |",
        "| --- | --- | --- | --- |",
    ]
    for material_path, entry in sorted(report.get("materials", {}).items()):
        warnings = "; ".join(entry.get("warnings", []))
        lines.append(
            f"| `{material_path}` | {entry.get('status', '')} | {entry.get('material_class') or ''} | {warnings} |"
        )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_conversion_reports(report: dict[str, Any], cache: str | Path) -> None:
    output_dir = Path(cache)
    report_json = output_dir / "conversion_report.json"
    report_md = output_dir / "conversion_report.md"
    report_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_conversion_report_md(report, report_md)
