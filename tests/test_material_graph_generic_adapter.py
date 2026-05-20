from pathlib import Path

from roboverse_pack.blender.usd.material_graph.adapters import convert_material, select_adapter
from roboverse_pack.blender.usd.material_graph.adapters.fallback import SemanticClassFallbackAdapter
from roboverse_pack.blender.usd.material_graph.adapters.generic import GenericTextureGraphAdapter
from roboverse_pack.blender.usd.material_graph.context import MaterialContext
from roboverse_pack.blender.usd.material_graph.schema import RawMaterialSpec, freeze_values


def _context(material_path="/World/Looks/Mat"):
    return MaterialContext(
        source_path=Path("scene.usda"),
        texture_base_dir=Path("."),
        material_path=material_path,
    )


def _raw(values=None, material_path="/World/Looks/Mat"):
    return RawMaterialSpec(
        material_path=material_path,
        material_name=material_path.rsplit("/", 1)[-1],
        shader_ids=("UnknownShader",),
        connected_surface_shader_id="UnknownShader",
        mdl_source_asset=None,
        values=freeze_values(values or {}),
    )


def test_generic_adapter_converts_unknown_diffuse_texture_to_direct_graph():
    raw = _raw({"diffuse_texture": "textures/diffuse.png"})
    adapter = GenericTextureGraphAdapter()

    assert adapter.score(raw) == 40
    spec = adapter.convert(raw, _context())

    assert spec.conversion_policy == "direct_graph"
    assert spec.base_color.texture is not None
    assert spec.base_color.texture.file == "textures/diffuse.png"


def test_unknown_material_with_no_values_selects_class_fallback():
    raw = _raw(material_path="/World/Looks/GlassPane")
    adapter = select_adapter(raw)
    spec = convert_material(raw, _context(raw.material_path))

    assert isinstance(adapter, SemanticClassFallbackAdapter)
    assert spec.conversion_policy == "class_fallback"
    assert spec.material_class == "glass"
    assert spec.quality_notes == ("semantic class fallback used",)
