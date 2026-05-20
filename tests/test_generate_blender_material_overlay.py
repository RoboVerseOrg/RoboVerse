import hashlib
import json
import os
import shutil
from pathlib import Path

import pytest

pxr = pytest.importorskip("pxr")
from pxr import Gf, Sdf, Usd, UsdShade

from roboverse_pack.blender.usd.overlay import (
    generate_blender_overlay,
    verify_overlay_material_coverage,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "usd_materials"


def _source_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _copy_fixture(tmp_path, fixture_name):
    source = tmp_path / fixture_name
    if not source.exists():
        shutil.copyfile(FIXTURE_DIR / fixture_name, source)
    return source


def _generate_fixture_overlay(tmp_path, fixture_name):
    source = _copy_fixture(tmp_path, fixture_name)
    overlay = source.with_suffix(".blender_materials.usda")
    root = source.with_suffix(".blender_root.usda")
    report = generate_blender_overlay(source, overlay, root, tmp_path / "cache")
    return overlay, root, report


def _write_overlay_fixture(path):
    stage = Usd.Stage.CreateNew(str(path))
    stage.DefinePrim("/World", "Xform")

    painted = UsdShade.Material.Define(stage, "/World/Looks/Painted")
    painted_shader = UsdShade.Shader.Define(stage, "/World/Looks/Painted/Shader")
    painted_shader.CreateIdAttr("OmniPBR")
    painted_shader.CreateInput("BaseColor_Color", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.1, 0.2, 0.3))
    painted_shader.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(0.7)
    painted_shader.CreateInput("metallic_constant", Sdf.ValueTypeNames.Float).Set(0.4)
    painted.CreateSurfaceOutput().ConnectToSource(painted_shader.ConnectableAPI(), "surface")

    textured = UsdShade.Material.Define(stage, "/World/Looks/Textured")
    textured_shader = UsdShade.Shader.Define(stage, "/World/Looks/Textured/Shader")
    textured_shader.CreateIdAttr("OmniPBR")
    textured_shader.CreateInput("BaseColor_Tex", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath("albedo.png"))
    textured.CreateSurfaceOutput().ConnectToSource(textured_shader.ConnectableAPI(), "surface")

    glass = UsdShade.Material.Define(stage, "/World/Looks/GlassDoor")
    glass_shader = UsdShade.Shader.Define(stage, "/World/Looks/GlassDoor/Shader")
    glass_shader.CreateIdAttr("OmniGlass")
    glass_shader.CreateInput("opacity_constant", Sdf.ValueTypeNames.Float).Set(0.25)
    glass.CreateSurfaceOutput().ConnectToSource(glass_shader.ConnectableAPI(), "surface")

    stage.GetRootLayer().Save()


def _preview_shader(stage, material_path):
    material = UsdShade.Material.Get(stage, material_path)
    source_info = material.ComputeSurfaceSource()
    assert source_info
    return UsdShade.Shader(source_info[0])


def _preview_input(stage, material_path, input_name):
    return _preview_shader(stage, material_path).GetInput(input_name).Get()


def test_root_sublayers_overlay_before_source(tmp_path):
    overlay, root, _report = _generate_fixture_overlay(tmp_path, "omnipbr_basecolor_scalar.usda")
    source = tmp_path / "omnipbr_basecolor_scalar.usda"

    root_layer = Sdf.Layer.FindOrOpen(str(root))
    assert root_layer.subLayerPaths[:2] == [str(overlay), str(source)]


def test_generate_overlay_does_not_modify_source_usd(tmp_path):
    source = _copy_fixture(tmp_path, "omnipbr_basecolor_scalar.usda")
    before_hash = _source_hash(source)

    _generate_fixture_overlay(tmp_path, "omnipbr_basecolor_scalar.usda")

    assert _source_hash(source) == before_hash


def test_direct_diffuse_texture_mapping_is_preserved(tmp_path):
    overlay, _root, _report = _generate_fixture_overlay(tmp_path, "omnipbr_basecolor_texture.usda")

    overlay_text = overlay.read_text(encoding="utf-8")
    assert "UsdUVTexture" in overlay_text
    assert "UsdPrimvarReader_float2" in overlay_text
    assert "inputs:file" in overlay_text
    assert "textures/wood.png" in overlay_text

    stage = Usd.Stage.Open(str(overlay))
    texture_shader = UsdShade.Shader.Get(stage, "/World/Looks/Wood/base_color_Texture")
    reader_shader = UsdShade.Shader.Get(stage, "/World/Looks/Wood/base_color_Primvar")
    preview_shader = UsdShade.Shader.Get(stage, "/World/Looks/Wood/PreviewSurface")

    assert texture_shader.GetIdAttr().Get() == "UsdUVTexture"
    assert texture_shader.GetInput("file").Get().path == "./textures/wood.png"
    assert reader_shader.GetIdAttr().Get() == "UsdPrimvarReader_float2"
    diffuse_connection = preview_shader.GetInput("diffuseColor").GetConnectedSource()
    assert diffuse_connection[0].GetPath() == texture_shader.GetPath()
    assert diffuse_connection[1] == "rgb"


def test_scalar_roughness_metallic_opacity_emissive_are_preserved(tmp_path):
    _overlay, root, _report = _generate_fixture_overlay(tmp_path, "omnipbr_basecolor_scalar.usda")

    stage = Usd.Stage.Open(str(root))
    shader = _preview_shader(stage, "/World/Looks/Paint")

    assert shader.GetInput("roughness").Get() == pytest.approx(0.25)
    assert shader.GetInput("metallic").Get() == pytest.approx(0.0)
    assert shader.GetInput("opacity").Get() == pytest.approx(1.0)
    assert shader.GetInput("emissiveColor").Get() == Gf.Vec3f(0.0, 0.0, 0.0)


def test_glass_class_fallback_is_preserved(tmp_path):
    _overlay, root, report = _generate_fixture_overlay(tmp_path, "omnipbr_glass_scalar.usda")

    entry = report["materials"]["/World/Looks/GlassPane"]
    assert entry["material_class"] == "glass"
    assert entry["status"] == "converted"

    stage = Usd.Stage.Open(str(root))
    shader = _preview_shader(stage, "/World/Looks/GlassPane")
    assert shader.GetInput("opacity").Get() == pytest.approx(0.35)
    assert shader.GetInput("ior").Get() == pytest.approx(1.45)


def test_report_includes_material_status_and_quality_warnings(tmp_path):
    _overlay, _root, report = _generate_fixture_overlay(tmp_path, "omnipbr_basecolor_scalar.usda")

    assert set(report["materials"]) == {"/World/Looks/Paint"}
    material_path = "/World/Looks/Paint"
    entry = report["materials"][material_path]
    assert set(entry) == {"status", "warnings", "material_class"}
    assert entry["status"] == "converted"
    assert entry["warnings"] == []
    assert entry["material_class"] is None


def test_existing_preview_surface_values_are_preserved(tmp_path):
    _overlay, root, report = _generate_fixture_overlay(tmp_path, "preview_surface_existing.usda")

    entry = report["materials"]["/World/Looks/Previewed"]
    assert entry["status"] == "skipped"
    assert entry["material_class"] is None
    assert entry["warnings"] == ["source UsdPreviewSurface graph preserved without overlay opinion"]

    stage = Usd.Stage.Open(str(root))
    shader = _preview_shader(stage, "/World/Looks/Previewed")
    assert shader.GetIdAttr().Get() == "UsdPreviewSurface"
    assert shader.GetInput("diffuseColor").Get() == Gf.Vec3f(0.1, 0.2, 0.3)
    assert shader.GetInput("roughness").Get() == pytest.approx(0.4)


def test_alias_matrix_freezes_current_omnipbr_aliases(tmp_path):
    overlay, root, report = _generate_fixture_overlay(tmp_path, "omnipbr_alias_matrix.usda")

    overlay_text = overlay.read_text(encoding="utf-8")
    for snippet in [
        'def Material "BaseColorColor"',
        'def Material "DiffuseColorConstant"',
        'def Material "BaseColorTex"',
        'def Material "DiffuseTexture"',
        'def Shader "base_color_Texture"',
        "@./textures/base.png@",
        "@./textures/diffuse.png@",
        "float inputs:roughness = 0.31",
        "float inputs:roughness = 0.62",
        "float inputs:metallic = 0.73",
        "float inputs:metallic = 0.84",
        "float inputs:opacity = 0.45",
        "float inputs:opacity = 0.56",
        "color3f inputs:emissiveColor = (0.1, 0.2, 0.3)",
        "color3f inputs:emissiveColor = (0.3, 0.2, 0.1)",
        'def Material "WallPanel"',
        'def Material "FloorTile"',
        'def Material "CabinetDoor"',
        'def Material "GlassFallback"',
    ]:
        assert snippet in overlay_text

    stage = Usd.Stage.Open(str(root))
    assert _preview_input(stage, "/World/Looks/BaseColorColor", "diffuseColor") == Gf.Vec3f(0.2, 0.3, 0.4)
    assert _preview_input(stage, "/World/Looks/DiffuseColorConstant", "diffuseColor") == Gf.Vec3f(0.4, 0.3, 0.2)

    base_texture = _preview_shader(stage, "/World/Looks/BaseColorTex").GetInput("diffuseColor").GetConnectedSource()
    diffuse_texture = _preview_shader(stage, "/World/Looks/DiffuseTexture").GetInput("diffuseColor").GetConnectedSource()
    assert base_texture[0].GetPath() == Sdf.Path("/World/Looks/BaseColorTex/base_color_Texture")
    assert diffuse_texture[0].GetPath() == Sdf.Path("/World/Looks/DiffuseTexture/base_color_Texture")

    assert _preview_input(stage, "/World/Looks/ReflectionRoughness", "roughness") == pytest.approx(0.31)
    assert _preview_input(stage, "/World/Looks/RoughnessAlias", "roughness") == pytest.approx(0.62)
    assert _preview_input(stage, "/World/Looks/MetallicConstant", "metallic") == pytest.approx(0.73)
    assert _preview_input(stage, "/World/Looks/MetallicAlias", "metallic") == pytest.approx(0.84)
    assert _preview_input(stage, "/World/Looks/OpacityConstant", "opacity") == pytest.approx(0.45)
    assert _preview_input(stage, "/World/Looks/OpacityAlias", "opacity") == pytest.approx(0.56)
    assert _preview_input(stage, "/World/Looks/EmissiveColorOmni", "emissiveColor") == Gf.Vec3f(0.1, 0.2, 0.3)
    assert _preview_input(stage, "/World/Looks/EmissiveColorAlias", "emissiveColor") == Gf.Vec3f(0.3, 0.2, 0.1)

    assert _preview_input(stage, "/World/Looks/WallPanel", "diffuseColor") == Gf.Vec3f(0.72, 0.72, 0.68)
    assert _preview_input(stage, "/World/Looks/FloorTile", "diffuseColor") == Gf.Vec3f(0.55, 0.50, 0.44)
    assert _preview_input(stage, "/World/Looks/CabinetDoor", "diffuseColor") == Gf.Vec3f(0.60, 0.46, 0.32)
    assert _preview_input(stage, "/World/Looks/GlassFallback", "diffuseColor") == Gf.Vec3f(0.78, 0.90, 0.96)
    assert _preview_input(stage, "/World/Looks/GlassFallback", "opacity") == pytest.approx(0.35)
    assert _preview_input(stage, "/World/Looks/GlassFallback", "ior") == pytest.approx(1.45)

    assert report["materials"]["/World/Looks/ReflectionRoughness"]["warnings"] == [
        "No diffuse color or texture alias found."
    ]
    for material_path in report["materials"]:
        verify_overlay_material_coverage(root, [material_path])


def test_generate_overlay_authors_root_sublayers_and_preview_values(tmp_path):
    source = tmp_path / "scene.usda"
    overlay = tmp_path / "scene.blender_materials.usda"
    root = tmp_path / "scene.blender_root.usda"
    cache = tmp_path / "cache"
    _write_overlay_fixture(source)
    before_hash = _source_hash(source)

    report = generate_blender_overlay(source, overlay, root, cache, resolution=512, samples=2)

    assert _source_hash(source) == before_hash
    assert overlay.exists()
    assert root.exists()
    root_layer = Sdf.Layer.FindOrOpen(str(root))
    assert root_layer.subLayerPaths[:2] == [str(overlay), str(source)]

    stage = Usd.Stage.Open(str(root))
    painted_shader = _preview_shader(stage, "/World/Looks/Painted")
    assert painted_shader.GetIdAttr().Get() == "UsdPreviewSurface"
    assert painted_shader.GetInput("diffuseColor").Get() == Gf.Vec3f(0.1, 0.2, 0.3)
    assert painted_shader.GetInput("roughness").Get() == pytest.approx(0.7)
    assert painted_shader.GetInput("metallic").Get() == pytest.approx(0.4)

    glass_shader = _preview_shader(stage, "/World/Looks/GlassDoor")
    assert glass_shader.GetInput("opacity").Get() == pytest.approx(0.25)
    assert glass_shader.GetInput("ior").Get() == pytest.approx(1.45)
    assert report["materials"]["/World/Looks/GlassDoor"]["material_class"] == "glass"

    verify_overlay_material_coverage(root, ["/World/Looks/Painted", "/World/Looks/Textured", "/World/Looks/GlassDoor"])


def test_generate_overlay_authors_direct_diffuse_texture_chain_and_reports(tmp_path):
    source = tmp_path / "scene.usda"
    overlay = tmp_path / "scene.blender_materials.usda"
    root = tmp_path / "scene.blender_root.usda"
    cache = tmp_path / "cache"
    _write_overlay_fixture(source)

    generate_blender_overlay(source, overlay, root, cache)

    stage = Usd.Stage.Open(str(overlay))
    texture_shader = UsdShade.Shader.Get(stage, "/World/Looks/Textured/base_color_Texture")
    reader_shader = UsdShade.Shader.Get(stage, "/World/Looks/Textured/base_color_Primvar")
    preview_shader = UsdShade.Shader.Get(stage, "/World/Looks/Textured/PreviewSurface")

    assert texture_shader.GetIdAttr().Get() == "UsdUVTexture"
    assert texture_shader.GetInput("file").Get().path == "albedo.png"
    assert reader_shader.GetIdAttr().Get() == "UsdPrimvarReader_float2"
    assert reader_shader.GetInput("varname").Get() == "st"
    diffuse_connection = preview_shader.GetInput("diffuseColor").GetConnectedSource()
    assert diffuse_connection[0].GetPath() == texture_shader.GetPath()
    assert diffuse_connection[1] == "rgb"

    assert (cache / "conversion_report.json").exists()
    assert (cache / "conversion_report.md").exists()
    report = json.loads((cache / "conversion_report.json").read_text(encoding="utf-8"))
    textured = next(entry for entry in report["materials"] if entry["material_path"] == "/World/Looks/Textured")
    assert textured["policy"] == "direct_graph"
    assert textured["slots"]["base_color"]["status"] == "texture"


def test_generate_overlay_can_rerun_over_existing_overlay(tmp_path):
    source = tmp_path / "scene.usda"
    overlay = tmp_path / "scene.blender_materials.usda"
    root = tmp_path / "scene.blender_root.usda"
    cache = tmp_path / "cache"
    _write_overlay_fixture(source)

    generate_blender_overlay(source, overlay, root, cache)
    report = generate_blender_overlay(source, overlay, root, cache)

    assert report["materials"]["/World/Looks/Painted"]["status"] == "converted"
    verify_overlay_material_coverage(root, report["materials"].keys())


def test_generate_overlay_authors_nested_relative_sublayers_from_root_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source = "assets/scene.usda"
    overlay = "assets/scene.blender_materials.usda"
    root = "assets/scene.blender_root.usda"
    cache = "cache"
    os.makedirs("assets")
    _write_overlay_fixture(source)

    generate_blender_overlay(source, overlay, root, cache)

    root_layer = Sdf.Layer.FindOrOpen(root)
    assert root_layer.subLayerPaths[:2] == ["scene.blender_materials.usda", "scene.usda"]
    verify_overlay_material_coverage(root, ["/World/Looks/Painted", "/World/Looks/Textured", "/World/Looks/GlassDoor"])
