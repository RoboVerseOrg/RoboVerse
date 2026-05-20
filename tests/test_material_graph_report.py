from roboverse_pack.blender.usd.material_graph.report import ConversionReport, adapter_from_policy
from roboverse_pack.blender.usd.material_graph.schema import (
    InputSpec,
    PreviewMaterialSpec,
    TextureSpec,
    UVSetSpec,
    UVTransformSpec,
)


def _spec(**overrides):
    data = {
        "material_path": "/World/Looks/Paint",
        "material_name": "Paint",
        "source_shader_ids": ("OmniPBR",),
        "mdl_source_asset": None,
        "base_color": InputSpec(value=(0.2, 0.3, 0.4), source_inputs=("BaseColor_Color",)),
        "normal": None,
        "metallic": None,
        "roughness": None,
        "specular_color": None,
        "emissive_color": None,
        "opacity": None,
        "conversion_policy": "direct_graph",
        "quality_notes": (),
    }
    data.update(overrides)
    return PreviewMaterialSpec(**data)


def test_counts_texture_base_color_and_scalar_roughness():
    report = ConversionReport()
    report.add_material(
        _spec(
            base_color=InputSpec(
                texture=TextureSpec(
                    file="textures/albedo.png",
                    source_color_space="sRGB",
                    source_input="BaseColor_Tex",
                ),
                source_inputs=("BaseColor_Tex",),
            ),
            roughness=InputSpec(value=0.42, source_inputs=("Reflection_Roughness",)),
        )
    )

    data = report.to_dict()

    assert data["summary"]["materials_total"] == 1
    assert data["summary"]["converted_by_direct_graph"] == 1
    assert data["texture_slots"]["base_color"]["texture"] == 1
    assert data["texture_slots"]["roughness"]["scalar"] == 1


def test_material_slot_entry_includes_source_texture_and_uv_details():
    report = ConversionReport()
    report.add_material(
        _spec(
            base_color=InputSpec(
                texture=TextureSpec(
                    file="textures/albedo.png",
                    source_color_space="sRGB",
                    uv_set=UVSetSpec(primvar_name="uv2", requested_index=1, resolution_status="matched"),
                    uv_transform=UVTransformSpec(
                        scale=(2.0, 3.0),
                        rotation_degrees=15.0,
                        translation=(0.25, 0.5),
                        source_input="BaseColor_UVA",
                    ),
                    source_input="BaseColor_Tex",
                ),
                source_inputs=("BaseColor_Tex",),
            )
        )
    )

    material = report.to_dict()["materials"][0]
    slot = material["slots"]["base_color"]

    assert slot["status"] == "texture"
    assert slot["source_input"] == "BaseColor_Tex"
    assert slot["source_inputs"] == ["BaseColor_Tex"]
    assert slot["file"] == "textures/albedo.png"
    assert slot["source_color_space"] == "sRGB"
    assert slot["uv_set"] == {"primvar_name": "uv2", "requested_index": 1, "resolution_status": "matched"}
    assert slot["uv_transform"]["source_input"] == "BaseColor_UVA"


def test_failed_material_increments_failed_count_and_warning():
    report = ConversionReport()
    report.add_failed_material("/World/Looks/Broken", "conversion failed: boom")

    data = report.to_dict()

    assert data["summary"]["materials_total"] == 1
    assert data["summary"]["failed"] == 1
    assert data["materials"][0]["adapter"] == "failed"
    assert data["materials"][0]["warnings"] == ["conversion failed: boom"]


def test_udim_and_uv_summaries_are_recorded():
    report = ConversionReport()
    report.add_material(
        _spec(
            base_color=InputSpec(
                texture=TextureSpec(
                    file="textures/tile_<UDIM>.png",
                    source_color_space="sRGB",
                    uv_set=UVSetSpec(primvar_name="uv1", requested_index=0, resolution_status="matched"),
                    uv_transform=UVTransformSpec(scale=(2.0, 2.0), source_input="BaseColor_UVA"),
                    source_input="BaseColor_Tex",
                ),
                source_inputs=("BaseColor_Tex",),
            )
        )
    )

    data = report.to_dict()

    assert data["uv"]["uv_transforms_found"] == 1
    assert data["uv"]["uv_transforms_converted"] == 1
    assert data["uv"]["multi_uv_requested"] == 1
    assert data["uv"]["multi_uv_resolved"] == 1
    assert data["udim"]["materials_with_udim"] == 1
    assert data["udim"]["normalized"] == 1


def test_adapter_from_policy_mapping():
    assert adapter_from_policy("preserve_existing_preview") == "existing_preview"
    assert adapter_from_policy("direct_graph") == "omnipbr"
    assert adapter_from_policy("mdl_bake") == "mdl_bake"
    assert adapter_from_policy("scalar_fallback") == "scalar_fallback"
    assert adapter_from_policy("class_fallback") == "semantic_class_fallback"
    assert adapter_from_policy("failed") == "failed"


def test_report_uses_actual_adapter_name_when_present():
    report = ConversionReport()
    report.add_material(_spec(adapter_name="generic_texture_graph"))

    data = report.to_dict()

    assert data["materials"][0]["adapter"] == "generic_texture_graph"
    assert data["materials"][0]["policy"] == "direct_graph"
