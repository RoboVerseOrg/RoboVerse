"""Generate a minimal Blender-friendly USD material overlay."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from .material_graph.author_preview import author_preview_material
from .material_graph.extract import extract_material, surface_shader
from .material_graph.normalize import normalize_material
from .material_graph.report import material_entry_from_spec, write_conversion_report_md, write_conversion_reports


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


def generate_blender_overlay(
    input_path: str | Path,
    overlay_path: str | Path,
    root_path: str | Path,
    texture_cache: str | Path,
    resolution: int = 2048,
    samples: int = 16,
) -> dict[str, Any]:
    from pxr import Gf, Sdf, Usd, UsdShade

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

    for prim in source_stage.Traverse():
        if not prim.IsA(UsdShade.Material):
            continue

        raw = extract_material(prim, UsdShade)
        spec = normalize_material(raw)
        author_preview_material(overlay_stage, spec, Gf, Sdf, UsdShade)

        report["materials"][spec.material_path] = material_entry_from_spec(spec)

    overlay_layer.Save()
    root_layer = _clear_or_create_layer(root, Sdf)
    root_parent = root.parent if str(root.parent) else Path(".")
    root_layer.subLayerPaths = [_sublayer_path(overlay, root_parent), _sublayer_path(source_path, root_parent)]
    root_layer.Save()

    write_conversion_reports(report, cache)
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
