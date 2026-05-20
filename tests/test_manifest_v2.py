import json

from roboverse_pack.blender.usd import compat


def _write_outputs(tmp_path):
    source = tmp_path / "scene.usda"
    overlay = tmp_path / "scene.blender_materials.usda"
    root = tmp_path / "scene.blender_root.usda"
    manifest = tmp_path / "manifest.json"
    source.write_text("#usda 1.0\n", encoding="utf-8")
    overlay.write_text("overlay", encoding="utf-8")
    root.write_text("root", encoding="utf-8")
    return source, overlay, root, manifest


def test_changing_referenced_texture_invalidates_overlay(tmp_path):
    source, overlay, root, manifest = _write_outputs(tmp_path)
    texture = tmp_path / "textures" / "wood.png"
    texture.parent.mkdir()
    texture.write_bytes(b"wood-one")

    compat.write_manifest(
        manifest,
        source=source,
        overlay=overlay,
        root=root,
        dependencies={"textures": [texture]},
    )

    assert compat.is_overlay_current(source, overlay, root, manifest)

    texture.write_bytes(b"wood-two")

    assert not compat.is_overlay_current(source, overlay, root, manifest)


def test_subset_settings_match_is_accepted(tmp_path):
    source, overlay, root, manifest = _write_outputs(tmp_path)
    compat.write_manifest(
        manifest,
        source=source,
        overlay=overlay,
        root=root,
        settings={"resolution": 1024, "samples": 4, "preserve_existing_preview": True},
    )

    assert compat.is_overlay_current(source, overlay, root, manifest, settings={"resolution": 1024})
    assert compat.is_overlay_current(source, overlay, root, manifest, settings={})


def test_schema_v1_manifest_is_rejected(tmp_path):
    source, overlay, root, manifest = _write_outputs(tmp_path)
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source": str(source),
                "overlay": str(overlay),
                "root": str(root),
                "source_sha256": compat.file_sha256(source),
            }
        ),
        encoding="utf-8",
    )

    assert not compat.is_overlay_current(source, overlay, root, manifest)


def test_changing_requested_setting_invalidates_overlay(tmp_path):
    source, overlay, root, manifest = _write_outputs(tmp_path)
    compat.write_manifest(
        manifest,
        source=source,
        overlay=overlay,
        root=root,
        settings={"resolution": 1024, "samples": 4},
    )

    assert not compat.is_overlay_current(source, overlay, root, manifest, settings={"resolution": 2048})


def test_missing_dependency_invalidates_overlay(tmp_path):
    source, overlay, root, manifest = _write_outputs(tmp_path)
    texture = tmp_path / "textures" / "wood.png"
    texture.parent.mkdir()
    texture.write_bytes(b"wood")
    compat.write_manifest(
        manifest,
        source=source,
        overlay=overlay,
        root=root,
        dependencies={"textures": [texture]},
    )

    texture.unlink()

    assert not compat.is_overlay_current(source, overlay, root, manifest)


def test_missing_texture_reference_invalidates_when_file_appears(tmp_path):
    source, overlay, root, manifest = _write_outputs(tmp_path)
    compat.write_manifest(
        manifest,
        source=source,
        overlay=overlay,
        root=root,
        dependencies={"missing_textures": ["textures/missing.png"]},
    )

    assert compat.is_overlay_current(source, overlay, root, manifest)

    texture = tmp_path / "textures" / "missing.png"
    texture.parent.mkdir()
    texture.write_bytes(b"now-present")

    assert not compat.is_overlay_current(source, overlay, root, manifest)


def test_udim_texture_report_dependencies_track_tile_changes(tmp_path):
    source, overlay, root, manifest = _write_outputs(tmp_path)
    texture_dir = tmp_path / "textures"
    texture_dir.mkdir()
    tile_1001 = texture_dir / "wall.1001.png"
    tile_1002 = texture_dir / "wall.1002.png"
    tile_1001.write_bytes(b"tile-one")
    tile_1002.write_bytes(b"tile-two")
    report = {
        "deep_report": {
            "materials": [
                {
                    "slots": {
                        "base_color": {
                            "status": "texture",
                            "file": "textures/wall.<UDIM>.png",
                        }
                    }
                }
            ]
        }
    }

    dependencies = compat._texture_dependencies_from_report(report, source.parent)
    assert dependencies["textures"] == [tile_1001, tile_1002]

    compat.write_manifest(
        manifest,
        source=source,
        overlay=overlay,
        root=root,
        dependencies=dependencies,
    )
    assert compat.is_overlay_current(source, overlay, root, manifest)

    tile_1002.write_bytes(b"tile-two-changed")

    assert not compat.is_overlay_current(source, overlay, root, manifest)
