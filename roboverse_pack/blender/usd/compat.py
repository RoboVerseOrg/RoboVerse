"""Pure-Python compatibility helpers for Blender USD material overlays."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .material_graph.texture_paths import find_texture_file_candidates
from .overlay import generate_blender_overlay

SCHEMA_VERSION = 2
CONVERTER_SCHEMA_VERSION = "material_graph_v1"
BLENDER_TARGET = "4.x"
_DEPENDENCY_GROUPS = ("sublayers", "references", "textures", "mdl")


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


def _effective_settings(
    settings: dict[str, Any] | None,
    converter_settings: dict[str, Any] | None,
) -> dict[str, Any]:
    if settings is not None:
        return dict(settings)
    if converter_settings is not None:
        return dict(converter_settings)
    return {}


def _default_converter(converter: dict[str, Any] | None = None) -> dict[str, Any]:
    data = {
        "schema_version": CONVERTER_SCHEMA_VERSION,
        "pxr_version": None,
        "kit_version": None,
        "blender_target": BLENDER_TARGET,
    }
    if converter:
        data.update(converter)
    return data


def _dependency_entry(path: str | Path) -> dict[str, Any]:
    dependency_path = Path(path)
    stat = dependency_path.stat()
    return {
        "path": str(dependency_path),
        "sha256": file_sha256(dependency_path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _normalize_dependencies(dependencies: dict[str, Any] | None) -> dict[str, Any]:
    normalized: dict[str, Any] = {group: [] for group in _DEPENDENCY_GROUPS}
    if not dependencies:
        return normalized

    for group in _DEPENDENCY_GROUPS:
        for item in dependencies.get(group, []) or []:
            if isinstance(item, dict):
                path = item.get("path")
                if path is not None and Path(path).exists():
                    entry = _dependency_entry(path)
                    for key, value in item.items():
                        if key not in entry:
                            entry[key] = value
                    normalized[group].append(entry)
                else:
                    normalized[group].append(dict(item))
            elif Path(item).exists():
                normalized[group].append(_dependency_entry(item))

    missing_textures = dependencies.get("missing_textures")
    if missing_textures:
        normalized["missing_textures"] = [str(path) for path in missing_textures]
    return normalized


def _dependency_is_current(entry: Any) -> bool:
    if not isinstance(entry, dict):
        return False
    path = entry.get("path")
    if not path:
        return False
    dependency_path = Path(path)
    if not dependency_path.exists():
        return False
    try:
        stat = dependency_path.stat()
        if entry.get("size") != stat.st_size or entry.get("mtime_ns") != stat.st_mtime_ns:
            return False
        return entry.get("sha256") == file_sha256(dependency_path)
    except OSError:
        return False


def _missing_texture_still_missing(raw: str, source_parent: Path) -> bool:
    return not find_texture_file_candidates(raw, source_parent)


def _texture_dependencies_from_report(report: dict[str, Any], source_parent: Path) -> dict[str, list[Any]]:
    deep_report = report.get("deep_report", report)
    materials = deep_report.get("materials", [])
    if isinstance(materials, dict):
        return {"textures": [], "missing_textures": []}

    dependencies: list[Path] = []
    missing_textures: list[str] = []
    seen: set[Path] = set()
    seen_missing: set[str] = set()
    for material in materials:
        for slot in material.get("slots", {}).values():
            file_value = slot.get("file")
            if not file_value:
                continue
            candidates = find_texture_file_candidates(str(file_value), source_parent)
            if candidates:
                for candidate in candidates:
                    if candidate not in seen:
                        seen.add(candidate)
                        dependencies.append(candidate)
                continue
            missing = str(file_value)
            if missing not in seen_missing:
                seen_missing.add(missing)
                missing_textures.append(missing)
    return {"textures": dependencies, "missing_textures": missing_textures}


def is_overlay_current(
    source: str | Path,
    overlay: str | Path,
    root: str | Path,
    manifest: str | Path,
    *,
    settings: dict[str, Any] | None = None,
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
    if data.get("converter", {}).get("schema_version") != CONVERTER_SCHEMA_VERSION:
        return False
    for group in _DEPENDENCY_GROUPS:
        for entry in data.get("dependencies", {}).get(group, []):
            if not _dependency_is_current(entry):
                return False
    for missing_texture in data.get("dependencies", {}).get("missing_textures", []):
        if not _missing_texture_still_missing(str(missing_texture), source_path.parent):
            return False
    requested_settings = _effective_settings(settings, converter_settings)
    manifest_settings = data.get("settings", {})
    if requested_settings and any(manifest_settings.get(key) != value for key, value in requested_settings.items()):
        return False
    return True


def write_manifest(
    manifest: str | Path,
    *,
    source: str | Path,
    overlay: str | Path,
    root: str | Path,
    dependencies: dict[str, Any] | None = None,
    settings: dict[str, Any] | None = None,
    converter: dict[str, Any] | None = None,
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
        "converter": _default_converter(converter),
        "dependencies": _normalize_dependencies(dependencies),
        "settings": _effective_settings(settings, converter_settings),
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
    settings = {"resolution": resolution, "samples": samples}
    if not force and is_overlay_current(
        paths.source,
        paths.overlay,
        paths.root,
        paths.manifest,
        settings=settings,
    ):
        return paths.root

    paths.cache.mkdir(parents=True, exist_ok=True)
    report = generate_blender_overlay(
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
        dependencies=_texture_dependencies_from_report(report, paths.source.parent),
        settings=settings,
    )
    return paths.root
