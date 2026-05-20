import json
from pathlib import Path

from roboverse_pack.blender.usd import compat


def test_blender_overlay_paths_are_adjacent_and_cache_mirrors_parent(tmp_path):
    source = tmp_path / "rooms" / "kitchen.usda"
    source.parent.mkdir()
    source.write_text("#usda 1.0\n", encoding="utf-8")

    paths = compat.blender_overlay_paths(source)
    cache_suffix = Path(*source.parent.parts[1:])

    assert paths.source == source
    assert paths.overlay == source.with_suffix(".blender_materials.usda")
    assert paths.root == source.with_suffix(".blender_root.usda")
    assert paths.cache == Path("roboverse_data/material_cache") / cache_suffix
    assert paths.manifest == paths.cache / "kitchen.blender_overlay_manifest.json"


def test_manifest_currentness_uses_source_hash_and_required_outputs(tmp_path):
    source = tmp_path / "scene.usda"
    overlay = tmp_path / "scene.blender_materials.usda"
    root = tmp_path / "scene.blender_root.usda"
    manifest = tmp_path / "manifest.json"
    source.write_text("one", encoding="utf-8")
    overlay.write_text("overlay", encoding="utf-8")
    root.write_text("root", encoding="utf-8")

    compat.write_manifest(manifest, source=source, overlay=overlay, root=root)
    assert compat.is_overlay_current(source, overlay, root, manifest)

    source.write_text("two", encoding="utf-8")
    assert not compat.is_overlay_current(source, overlay, root, manifest)

    manifest.write_text(json.dumps({"schema_version": 2}), encoding="utf-8")
    assert not compat.is_overlay_current(source, overlay, root, manifest)


def test_ensure_blender_material_overlay_reuses_current_manifest(tmp_path, monkeypatch):
    source = tmp_path / "scene.usda"
    source.write_text("#usda 1.0\n", encoding="utf-8")
    paths = compat.blender_overlay_paths(source)
    paths.cache.mkdir(parents=True, exist_ok=True)
    paths.overlay.write_text("overlay", encoding="utf-8")
    paths.root.write_text("root", encoding="utf-8")
    compat.write_manifest(
        paths.manifest,
        source=source,
        overlay=paths.overlay,
        root=paths.root,
        converter_settings={"resolution": 2048, "samples": 16},
    )

    def fail_generate(*args, **kwargs):
        raise AssertionError("generate_blender_overlay should not be called")

    monkeypatch.setattr(compat, "generate_blender_overlay", fail_generate)

    assert compat.ensure_blender_material_overlay(source) == paths.root


def test_ensure_blender_material_overlay_generates_when_forced(tmp_path, monkeypatch):
    source = tmp_path / "scene.usda"
    source.write_text("#usda 1.0\n", encoding="utf-8")
    calls = []

    def fake_generate(input_path, overlay_path, root_path, texture_cache, resolution, samples):
        calls.append((input_path, overlay_path, root_path, texture_cache, resolution, samples))
        overlay_path.write_text("overlay", encoding="utf-8")
        root_path.write_text("root", encoding="utf-8")
        return {"materials": {}}

    monkeypatch.setattr(compat, "generate_blender_overlay", fake_generate)

    root = compat.ensure_blender_material_overlay(source, force=True, resolution=1024, samples=4)
    paths = compat.blender_overlay_paths(source)

    assert root == paths.root
    assert paths.manifest.exists()
    assert calls == [(source, paths.overlay, paths.root, paths.cache, 1024, 4)]


def test_manifest_currentness_compares_converter_settings(tmp_path):
    source = tmp_path / "scene.usda"
    overlay = tmp_path / "scene.blender_materials.usda"
    root = tmp_path / "scene.blender_root.usda"
    manifest = tmp_path / "manifest.json"
    source.write_text("#usda 1.0\n", encoding="utf-8")
    overlay.write_text("overlay", encoding="utf-8")
    root.write_text("root", encoding="utf-8")

    compat.write_manifest(
        manifest,
        source=source,
        overlay=overlay,
        root=root,
        converter_settings={"resolution": 1024, "samples": 4},
    )

    assert compat.is_overlay_current(
        source,
        overlay,
        root,
        manifest,
        converter_settings={"resolution": 1024, "samples": 4},
    )
    assert not compat.is_overlay_current(
        source,
        overlay,
        root,
        manifest,
        converter_settings={"resolution": 2048, "samples": 4},
    )
