"""Kit/Isaac Sim command helpers for MDL material baking."""

from __future__ import annotations

from pathlib import Path

from .context import MaterialContext
from .schema import PreviewMaterialSpec, RawMaterialSpec


def kit_bake_command(
    isaacsim_bin: Path,
    script_path: Path,
    source_path: Path,
    material_path: str,
    cache_dir: Path,
) -> list[str]:
    return [
        str(isaacsim_bin),
        "--no-window",
        "--/app/window/enabled=false",
        "--exec",
        str(script_path),
        "--",
        "--source",
        str(source_path),
        "--material",
        str(material_path),
        "--cache-dir",
        str(cache_dir),
    ]


def bake_mdl_material_to_preview(raw: RawMaterialSpec, context: MaterialContext) -> PreviewMaterialSpec:
    raise RuntimeError(
        "MDL bake requires Kit execution; call kit_overlay.py or configure the MDL bake runner before selecting "
        "MdlBakeAdapter"
    )
