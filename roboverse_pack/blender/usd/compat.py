"""Pure-Python compatibility helpers for Blender USD material overlays."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .overlay import generate_blender_overlay

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class BlenderOverlayPaths:
    source: Path
    overlay: Path
    root: Path
    cache: Path
    manifest: Path


def _cache_parent_for_source(source: Path) -> Path:
    parent = source.parent
    if str(parent) in ("", "."):
        return Path("roboverse_data/material_cache")
    parts = parent.parts[1:] if parent.is_absolute() else parent.parts
    return Path("roboverse_data/material_cache").joinpath(*parts)


def blender_overlay_paths(source: str | Path) -> BlenderOverlayPaths:
    source_path = Path(source)
    overlay_path = source_path.with_suffix(".blender_materials.usda")
    root_path = source_path.with_suffix(".blender_root.usda")
    cache_path = _cache_parent_for_source(source_path)
    manifest_path = cache_path / f"{source_path.stem}.blender_overlay_manifest.json"
    return BlenderOverlayPaths(
        source=source_path,
        overlay=overlay_path,
        root=root_path,
        cache=cache_path,
        manifest=manifest_path,
    )


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_overlay_current(
    source: str | Path,
    overlay: str | Path,
    root: str | Path,
    manifest: str | Path,
    *,
    converter_settings: dict[str, Any] | None = None,
) -> bool:
    source_path = Path(source)
    overlay_path = Path(overlay)
    root_path = Path(root)
    manifest_path = Path(manifest)
    if not source_path.exists() or not overlay_path.exists() or not root_path.exists() or not manifest_path.exists():
        return False
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if data.get("schema_version") != SCHEMA_VERSION or data.get("source_sha256") != file_sha256(source_path):
        return False
    if converter_settings is not None and data.get("converter_settings", {}) != converter_settings:
        return False
    return True


def write_manifest(
    manifest: str | Path,
    *,
    source: str | Path,
    overlay: str | Path,
    root: str | Path,
    converter_settings: dict[str, Any] | None = None,
) -> None:
    manifest_path = Path(manifest)
    source_path = Path(source)
    data = {
        "schema_version": SCHEMA_VERSION,
        "source": str(source_path),
        "overlay": str(Path(overlay)),
        "root": str(Path(root)),
        "source_sha256": file_sha256(source_path),
        "converter_settings": converter_settings or {},
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def ensure_blender_material_overlay(
    usd_path: str | Path,
    *,
    force: bool = False,
    resolution: int = 2048,
    samples: int = 16,
) -> Path:
    paths = blender_overlay_paths(usd_path)
    converter_settings = {"resolution": resolution, "samples": samples}
    if not force and is_overlay_current(
        paths.source,
        paths.overlay,
        paths.root,
        paths.manifest,
        converter_settings=converter_settings,
    ):
        return paths.root

    paths.cache.mkdir(parents=True, exist_ok=True)
    generate_blender_overlay(
        paths.source,
        paths.overlay,
        paths.root,
        paths.cache,
        resolution=resolution,
        samples=samples,
    )
    write_manifest(
        paths.manifest,
        source=paths.source,
        overlay=paths.overlay,
        root=paths.root,
        converter_settings=converter_settings,
    )
    return paths.root
