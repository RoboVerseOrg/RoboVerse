"""Audit USD material bindings for Blender overlay conversion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _asset_path_string(value: Any) -> str | None:
    if value is None:
        return None
    return getattr(value, "path", str(value))


def audit_usd_materials(input_path: str | Path) -> dict[str, Any]:
    from pxr import Usd, UsdShade

    path = Path(input_path)
    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise ValueError(f"Unable to open USD stage: {path}")

    materials = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdShade.Material):
            continue
        shader_ids = []
        mdl_source_asset = None
        for descendant in Usd.PrimRange(prim):
            if descendant.GetPath() == prim.GetPath() or not descendant.IsA(UsdShade.Shader):
                continue
            shader = UsdShade.Shader(descendant)
            shader_id = shader.GetIdAttr().Get()
            if shader_id:
                shader_ids.append(str(shader_id))
            source_input = shader.GetInput("mdl:sourceAsset")
            if source_input and mdl_source_asset is None:
                mdl_source_asset = _asset_path_string(source_input.Get())
        materials.append(
            {
                "material_path": str(prim.GetPath()),
                "shader_ids": shader_ids,
                "mdl_source_asset": mdl_source_asset,
            }
        )

    return {
        "input_path": str(path),
        "materials_total": len(materials),
        "materials": materials,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit USD materials for Blender overlay conversion.")
    parser.add_argument("--input", required=True, help="Input USD stage.")
    parser.add_argument("--output", required=True, help="Output JSON report path.")
    args = parser.parse_args(argv)

    report = audit_usd_materials(args.input)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
