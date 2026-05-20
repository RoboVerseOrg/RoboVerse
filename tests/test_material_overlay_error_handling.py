from pathlib import Path

from roboverse_pack.blender.usd import overlay
from roboverse_pack.blender.usd.material_graph.context import MaterialContext
from roboverse_pack.blender.usd.material_graph.schema import InputSpec, PreviewMaterialSpec, RawMaterialSpec, freeze_values


def test_material_authoring_failure_records_failed_entry_and_continues(monkeypatch):
    raw = RawMaterialSpec(
        material_path="/World/Looks/Broken",
        material_name="Broken",
        shader_ids=("OmniPBR",),
        connected_surface_shader_id="OmniPBR",
        mdl_source_asset=None,
        values=freeze_values({"BaseColor_Color": (0.1, 0.2, 0.3)}),
    )
    spec = PreviewMaterialSpec(
        material_path=raw.material_path,
        material_name=raw.material_name,
        source_shader_ids=raw.shader_ids,
        mdl_source_asset=None,
        base_color=InputSpec(value=(0.1, 0.2, 0.3)),
        normal=None,
        metallic=None,
        roughness=None,
        specular_color=None,
        emissive_color=None,
        opacity=None,
    )
    context = MaterialContext(
        source_path=Path("scene.usda"),
        texture_base_dir=Path("."),
        material_path=raw.material_path,
    )
    report = {"materials": {}}

    monkeypatch.setattr(overlay, "convert_material", lambda raw_arg, context_arg: spec)

    def fail_authoring(*_args):
        raise RuntimeError("author failed")

    monkeypatch.setattr(overlay, "author_preview_material", fail_authoring)

    overlay._convert_and_author_material(raw, context, report, object(), object(), object(), object())

    assert report["materials"][raw.material_path] == {
        "status": "failed",
        "warnings": ["conversion failed: author failed"],
        "material_class": None,
    }


def test_material_extraction_failure_records_failed_entry(monkeypatch):
    class _Prim:
        def GetPath(self):
            return "/World/Looks/BrokenExtract"

    report = {"materials": {}}

    def fail_extract(*_args):
        raise RuntimeError("extract failed")

    monkeypatch.setattr(overlay, "extract_material", fail_extract)

    overlay._extract_convert_and_author_material(
        _Prim(),
        object(),
        Path("scene.usda"),
        report,
        object(),
        object(),
        object(),
        object(),
    )

    assert report["materials"]["/World/Looks/BrokenExtract"] == {
        "status": "failed",
        "warnings": ["conversion failed: extract failed"],
        "material_class": None,
    }
