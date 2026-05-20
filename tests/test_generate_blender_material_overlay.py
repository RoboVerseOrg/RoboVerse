import hashlib
import json
import os

import pytest

pxr = pytest.importorskip("pxr")
from pxr import Gf, Sdf, Usd, UsdShade

from roboverse_pack.blender.usd.overlay import (
    generate_blender_overlay,
    verify_overlay_material_coverage,
)


def _source_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    texture_shader = UsdShade.Shader.Get(stage, "/World/Looks/Textured/PreviewTexture")
    reader_shader = UsdShade.Shader.Get(stage, "/World/Looks/Textured/PreviewSTReader")
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
    assert report["materials"]["/World/Looks/Textured"]["status"] == "converted"


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
