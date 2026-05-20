from pathlib import Path

from roboverse_pack.blender.usd.material_graph.adapters.existing_preview import ExistingPreviewSurfaceAdapter
from roboverse_pack.blender.usd.material_graph.context import MaterialContext
from roboverse_pack.blender.usd.material_graph.schema import RawMaterialSpec, freeze_values


def test_existing_preview_policy_preserves_existing_graph():
    raw = RawMaterialSpec(
        material_path="/World/Looks/Previewed",
        material_name="Previewed",
        shader_ids=("UsdPreviewSurface",),
        connected_surface_shader_id="UsdPreviewSurface",
        mdl_source_asset=None,
        values=freeze_values({"diffuseColor": (0.1, 0.2, 0.3), "roughness": 0.4}),
    )
    context = MaterialContext(
        source_path=Path("scene.usda"),
        texture_base_dir=Path("."),
        material_path=raw.material_path,
    )

    adapter = ExistingPreviewSurfaceAdapter()
    spec = adapter.convert(raw, context)

    assert adapter.score(raw) == 100
    assert spec.conversion_policy == "preserve_existing_preview"
    assert spec.base_color.value == (0.1, 0.2, 0.3)
    assert spec.roughness is not None
    assert spec.roughness.value == 0.4
    assert spec.quality_notes == ("source UsdPreviewSurface graph preserved without overlay opinion",)
