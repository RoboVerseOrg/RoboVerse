import json

import pytest

pxr = pytest.importorskip("pxr")
from pxr import Sdf, Usd, UsdShade

from roboverse_pack.blender.usd.audit import audit_usd_materials, main


def _write_audit_fixture(path):
    stage = Usd.Stage.CreateNew(str(path))
    material = UsdShade.Material.Define(stage, "/World/Looks/Wood")
    shader = UsdShade.Shader.Define(stage, "/World/Looks/Wood/Shader")
    shader.CreateIdAttr("OmniPBR")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set((0.2, 0.3, 0.4))
    shader.CreateInput("mdl:sourceAsset", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath("OmniPBR.mdl"))
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    stage.GetRootLayer().Save()


def test_audit_usd_materials_reports_shader_ids_and_mdl_assets(tmp_path):
    usd_path = tmp_path / "material.usda"
    _write_audit_fixture(usd_path)

    report = audit_usd_materials(usd_path)

    assert report["input_path"] == str(usd_path)
    assert report["materials_total"] == 1
    assert report["materials"][0]["material_path"] == "/World/Looks/Wood"
    assert report["materials"][0]["shader_ids"] == ["OmniPBR"]
    assert report["materials"][0]["mdl_source_asset"] == "OmniPBR.mdl"


def test_audit_cli_writes_json(tmp_path):
    usd_path = tmp_path / "material.usda"
    output_path = tmp_path / "audit.json"
    _write_audit_fixture(usd_path)

    assert main(["--input", str(usd_path), "--output", str(output_path)]) == 0

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["materials_total"] == 1

