from pathlib import Path

from roboverse_pack.blender.usd.material_graph.adapters.generic import GenericTextureGraphAdapter
from roboverse_pack.blender.usd.material_graph.adapters.omnipbr import OmniPBRAdapter
from roboverse_pack.blender.usd.material_graph.context import MaterialContext
from roboverse_pack.blender.usd.material_graph.schema import RawMaterialSpec, freeze_values
from roboverse_pack.blender.usd.material_graph.texture_paths import (
    detect_udim_pattern,
    find_texture_file_candidates,
    normalize_texture_asset_path,
)


def _context(base_dir: Path) -> MaterialContext:
    return MaterialContext(
        source_path=Path("scene.usda"),
        texture_base_dir=base_dir,
        material_path="/World/Looks/Mat",
    )


def _raw(values: dict[str, object], shader_id: str = "OmniPBR") -> RawMaterialSpec:
    return RawMaterialSpec(
        material_path="/World/Looks/Mat",
        material_name="Mat",
        shader_ids=(shader_id,),
        connected_surface_shader_id=shader_id,
        mdl_source_asset=None,
        values=freeze_values(values),
    )


def test_dot_number_udim_pattern_normalizes_to_token():
    info = detect_udim_pattern("textures/wall.1001.png")

    assert info is not None
    assert info.normalized_path == "textures/wall.<UDIM>.png"
    assert info.first_tile == 1001


def test_existing_udim_token_is_kept():
    info = detect_udim_pattern("textures/wall.<UDIM>.png")

    assert info is not None
    assert info.normalized_path == "textures/wall.<UDIM>.png"
    assert info.first_tile is None


def test_printf_udim_token_is_normalized():
    info = detect_udim_pattern("textures/wall.%(UDIM)d.png")

    assert info is not None
    assert info.normalized_path == "textures/wall.<UDIM>.png"
    assert info.first_tile is None


def test_arbitrary_four_digit_version_is_not_udim():
    assert detect_udim_pattern("textures/version_1042.png") is None


def test_existing_relative_texture_candidate_found(tmp_path):
    texture = tmp_path / "textures" / "diffuse.png"
    texture.parent.mkdir()
    texture.write_text("fake image")

    candidates = find_texture_file_candidates(" @textures/diffuse.png@ ", tmp_path)
    result = normalize_texture_asset_path(" @textures/diffuse.png@ ", tmp_path)

    assert candidates == [texture]
    assert result.raw == "textures/diffuse.png"
    assert result.normalized == "textures/diffuse.png"
    assert result.exists is True
    assert result.warnings == ()


def test_udim_candidate_matching_scans_existing_tiles(tmp_path):
    texture = tmp_path / "textures" / "wall.1002.png"
    texture.parent.mkdir()
    texture.write_text("fake image")

    result = normalize_texture_asset_path("textures/wall.1001.png", tmp_path)

    assert result.normalized == "textures/wall.<UDIM>.png"
    assert result.is_udim is True
    assert result.candidates == (texture,)
    assert result.exists is True
    assert result.warnings == ()


def test_missing_texture_warning_mentions_path(tmp_path):
    result = normalize_texture_asset_path("textures/missing.png", tmp_path)

    assert result.exists is False
    assert result.warnings == ("Texture file not found: textures/missing.png",)


def test_missing_udim_warning_mentions_tile(tmp_path):
    result = normalize_texture_asset_path("textures/wall.<UDIM>.png", tmp_path)

    assert result.exists is False
    assert result.warnings == ("UDIM texture tile/file not found: textures/wall.<UDIM>.png",)


def test_omnipbr_adapter_normalizes_udim_path_and_reports_missing(tmp_path):
    spec = OmniPBRAdapter().convert(
        _raw({"BaseColor_Tex": "textures/wall.1001.png"}),
        _context(tmp_path),
    )

    assert spec.base_color.texture is not None
    assert spec.base_color.texture.file == "textures/wall.<UDIM>.png"
    assert spec.base_color.texture.role == "base_color"
    assert spec.quality_notes == ("UDIM texture tile/file not found: textures/wall.<UDIM>.png",)


def test_generic_adapter_keeps_direct_graph_and_reports_missing(tmp_path):
    spec = GenericTextureGraphAdapter().convert(
        _raw({"diffuse_texture": "textures/missing.png"}, shader_id="UnknownShader"),
        _context(tmp_path),
    )

    assert spec.conversion_policy == "direct_graph"
    assert spec.base_color.texture is not None
    assert spec.base_color.texture.file == "textures/missing.png"
    assert spec.base_color.texture.role == "base_color"
    assert spec.quality_notes == ("Texture file not found: textures/missing.png",)
