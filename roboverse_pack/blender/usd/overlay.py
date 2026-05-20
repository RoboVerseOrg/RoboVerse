"""Generate a minimal Blender-friendly USD material overlay."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from .material_graph.adapters import convert_material
from .material_graph.author_preview import author_preview_material
from .material_graph.bindings import collect_material_binding_contexts
from .material_graph.context import MaterialContext
from .material_graph.extract import extract_material, iter_material_prims, surface_shader
from .material_graph.report import (
    ConversionReport,
    failed_material_entry,
    material_entry_from_spec,
    write_conversion_reports,
)


def _clear_or_create_layer(path: Path, Sdf: Any) -> Any:
    layer = Sdf.Layer.FindOrOpen(str(path))
    if layer is None:
        layer = Sdf.Layer.CreateNew(str(path))
    else:
        layer.Clear()
    return layer


def _sublayer_path(layer_path: Path, root_parent: Path) -> str:
    if layer_path.is_absolute():
        return str(layer_path)
    return os.path.relpath(layer_path, root_parent)


def _convert_and_author_material(
    raw: Any,
    context: MaterialContext,
    report: dict[str, Any],
    overlay_stage: Any,
    Gf: Any,
    Sdf: Any,
    UsdShade: Any,
    *,
    conversion_report: ConversionReport | None = None,
) -> None:
    try:
        spec = convert_material(raw, context)
        report["materials"][spec.material_path] = material_entry_from_spec(spec)
        if conversion_report is not None:
            conversion_report.add_material(spec)
        if spec.conversion_policy == "preserve_existing_preview":
            return
        author_preview_material(overlay_stage, spec, Gf, Sdf, UsdShade)
    except Exception as exc:
        report["materials"][raw.material_path] = failed_material_entry(exc)
        if conversion_report is not None:
            conversion_report.add_failed_material(raw.material_path, f"conversion failed: {exc}")


def _prim_path_string(prim: Any) -> str:
    try:
        return str(prim.GetPath())
    except Exception:
        return "<unknown material>"


def _extract_convert_and_author_material(
    prim: Any,
    UsdShade: Any,
    source_path: Path,
    report: dict[str, Any],
    overlay_stage: Any,
    Gf: Any,
    Sdf: Any,
    authoring_usdshade: Any,
    binding_contexts: dict[str, Any] | None = None,
    *,
    conversion_report: ConversionReport | None = None,
) -> None:
    material_path = _prim_path_string(prim)
    try:
        raw = extract_material(prim, UsdShade)
        material_path = raw.material_path
        binding_context = (binding_contexts or {}).get(raw.material_path)
        context = MaterialContext(
            source_path=source_path,
            texture_base_dir=source_path.parent,
            material_path=raw.material_path,
            bound_prim_paths=binding_context.bound_prim_paths if binding_context is not None else (),
            uv_primvars_by_prim=binding_context.uv_primvars_by_prim if binding_context is not None else None,
        )
        spec = convert_material(raw, context)
        report["materials"][spec.material_path] = material_entry_from_spec(spec)
        if conversion_report is not None:
            conversion_report.add_material(spec)
        if spec.conversion_policy == "preserve_existing_preview":
            return
        author_preview_material(overlay_stage, spec, Gf, Sdf, authoring_usdshade)
    except Exception as exc:
        report["materials"][material_path] = failed_material_entry(exc)
        if conversion_report is not None:
            conversion_report.add_failed_material(material_path, f"conversion failed: {exc}")


def generate_blender_overlay(
    input_path: str | Path,
    overlay_path: str | Path,
    root_path: str | Path,
    texture_cache: str | Path,
    resolution: int = 2048,
    samples: int = 16,
) -> dict[str, Any]:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

    source_path = Path(input_path)
    overlay = Path(overlay_path)
    root = Path(root_path)
    cache = Path(texture_cache)
    cache.mkdir(parents=True, exist_ok=True)
    overlay.parent.mkdir(parents=True, exist_ok=True)
    root.parent.mkdir(parents=True, exist_ok=True)

    source_stage = Usd.Stage.Open(str(source_path))
    if source_stage is None:
        raise ValueError(f"Unable to open USD stage: {source_path}")

    overlay_layer = _clear_or_create_layer(overlay, Sdf)
    overlay_stage = Usd.Stage.Open(overlay_layer)
    report = {
        "input_path": str(source_path),
        "overlay_path": str(overlay),
        "root_path": str(root),
        "resolution": resolution,
        "samples": samples,
        "materials": {},
    }
    conversion_report = ConversionReport(
        input_path=str(source_path),
        overlay_path=str(overlay),
        root_path=str(root),
        resolution=resolution,
        samples=samples,
    )

    binding_contexts = collect_material_binding_contexts(source_stage, UsdShade, UsdGeom)
    for prim in iter_material_prims(source_stage, UsdShade):
        _extract_convert_and_author_material(
            prim,
            UsdShade,
            source_path,
            report,
            overlay_stage,
            Gf,
            Sdf,
            UsdShade,
            binding_contexts,
            conversion_report=conversion_report,
        )

    overlay_layer.Save()
    root_layer = _clear_or_create_layer(root, Sdf)
    root_parent = root.parent if str(root.parent) else Path(".")
    root_layer.subLayerPaths = [_sublayer_path(overlay, root_parent), _sublayer_path(source_path, root_parent)]
    root_layer.Save()

    deep_report = conversion_report.to_dict()
    report["deep_report"] = deep_report
    write_conversion_reports(conversion_report, cache)
    return report


def verify_overlay_material_coverage(root_path: str | Path, expected_material_paths: Any) -> None:
    from pxr import Usd, UsdShade

    stage = Usd.Stage.Open(str(Path(root_path)))
    if stage is None:
        raise ValueError(f"Unable to open USD stage: {root_path}")

    missing = []
    non_preview = []
    for material_path in expected_material_paths:
        material = UsdShade.Material.Get(stage, material_path)
        if not material:
            missing.append(str(material_path))
            continue
        shader = surface_shader(material)
        if not shader or shader.GetIdAttr().Get() != "UsdPreviewSurface":
            non_preview.append(str(material_path))

    if missing or non_preview:
        details = []
        if missing:
            details.append(f"missing materials: {', '.join(missing)}")
        if non_preview:
            details.append(f"missing preview surfaces: {', '.join(non_preview)}")
        raise AssertionError("; ".join(details))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a Blender USD material overlay.")
    parser.add_argument("--input", required=True, help="Input USD stage.")
    parser.add_argument("--overlay", required=True, help="Output overlay USD layer.")
    parser.add_argument("--root", required=True, help="Output root USD layer.")
    parser.add_argument("--texture-cache", required=True, help="Texture cache/report directory.")
    parser.add_argument("--resolution", type=int, default=2048)
    parser.add_argument("--samples", type=int, default=16)
    args = parser.parse_args(argv)

    generate_blender_overlay(
        args.input,
        args.overlay,
        args.root,
        args.texture_cache,
        resolution=args.resolution,
        samples=args.samples,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
