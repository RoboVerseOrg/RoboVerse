from roboverse_pack.blender.usd.material_graph.schema import UVTransformSpec
from roboverse_pack.blender.usd.material_graph.uv import parse_uva, resolve_slot_uv_transform


def test_parse_uva_float4_uses_scale_and_translation_pairs():
    assert parse_uva((2.0, 3.0, 0.25, 0.5)) == UVTransformSpec(
        scale=(2.0, 3.0),
        translation=(0.25, 0.5),
    )


def test_parse_uva_float3_uses_uniform_scale_and_translation_pair():
    assert parse_uva((2.0, 0.25, 0.5)) == UVTransformSpec(
        scale=(2.0, 2.0),
        translation=(0.25, 0.5),
    )


def test_parse_uva_mapping_accepts_aliases_for_translation_and_rotation():
    assert parse_uva(
        {
            "scale": (2.0, 3.0),
            "offset": (0.25, 0.5),
            "rotation": 45.0,
        }
    ) == UVTransformSpec(
        scale=(2.0, 3.0),
        rotation_degrees=45.0,
        translation=(0.25, 0.5),
    )


def test_parse_uva_mapping_accepts_scalar_scale_translation_and_rotation_degrees():
    assert parse_uva(
        {
            "scale": 2.0,
            "translation": 0.25,
            "rotation_degrees": 90.0,
        }
    ) == UVTransformSpec(
        scale=(2.0, 2.0),
        rotation_degrees=90.0,
        translation=(0.25, 0.25),
    )


def test_parse_uva_returns_none_for_unparseable_values():
    assert parse_uva("not-a-uva") is None
    assert parse_uva((1.0, 2.0)) is None


def test_resolve_slot_uv_transform_prefers_alias_and_records_source_input():
    spec = resolve_slot_uv_transform(
        {"BaseColor_UVA": (2.0, 3.0, 0.25, 0.5), "BaseColor_UScale": 8.0},
        "BaseColor",
        ("BaseColor_UVA",),
    )

    assert spec == UVTransformSpec(
        scale=(2.0, 3.0),
        translation=(0.25, 0.5),
        source_input="BaseColor_UVA",
    )


def test_resolve_slot_uv_transform_falls_back_to_separate_fields():
    spec = resolve_slot_uv_transform(
        {
            "BaseColor_UScale": 2.0,
            "BaseColor_VScale": 3.0,
            "BaseColor_Rotation": 45.0,
            "BaseColor_UOffset": 0.25,
            "BaseColor_VOffset": 0.5,
        },
        "BaseColor",
        ("BaseColor_UVA",),
    )

    assert spec == UVTransformSpec(
        scale=(2.0, 3.0),
        rotation_degrees=45.0,
        translation=(0.25, 0.5),
        source_input="BaseColor_*",
    )


def test_resolve_slot_uv_transform_ignores_empty_separate_defaults():
    assert resolve_slot_uv_transform({}, "BaseColor", ("BaseColor_UVA",)) is None
