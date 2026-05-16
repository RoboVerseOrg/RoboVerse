from __future__ import annotations

import pathlib

import pytest
import rootutils

rootutils.setup_root(__file__, pythonpath=True)

from metasim.scenario.objects import ArticulationObjCfg


def test_articulation_file_name_uses_urdf_for_isaacsim_when_usd_missing() -> None:
    cfg = ArticulationObjCfg(
        name="asset",
        usd_path=None,
        urdf_path="/tmp/asset.urdf",
        mjcf_path="/tmp/asset.xml",
    )

    assert cfg.file_name("isaacsim") == "/tmp/asset.urdf"
    assert cfg.file_name("isaaclab") == "/tmp/asset.urdf"


def test_articulation_file_name_prefers_usd_for_isaacsim_when_available() -> None:
    cfg = ArticulationObjCfg(
        name="asset",
        usd_path="/tmp/asset.usd",
        urdf_path="/tmp/asset.urdf",
        mjcf_path="/tmp/asset.xml",
    )

    assert cfg.file_name("isaacsim") == "/tmp/asset.usd"
    assert cfg.file_name("isaaclab") == "/tmp/asset.usd"


def test_resolve_isaacsim_file_path_prefers_existing_usd(tmp_path: pathlib.Path) -> None:
    from metasim.utils.isaacsim_asset_util import resolve_isaacsim_file_path

    usd_path = tmp_path / "asset.usd"
    usd_path.write_text("#usda 1.0\n", encoding="utf-8")

    cfg = ArticulationObjCfg(
        name="asset",
        usd_path=str(usd_path),
        urdf_path=str(tmp_path / "asset.urdf"),
        mjcf_path=str(tmp_path / "asset.xml"),
    )

    assert resolve_isaacsim_file_path(cfg) == str(usd_path)


def test_resolve_isaacsim_file_path_converts_urdf_when_usd_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    from metasim.utils import isaacsim_asset_util

    urdf_path = tmp_path / "asset.urdf"
    urdf_path.write_text("<robot name='asset'/>", encoding="utf-8")
    generated_usd = tmp_path / "asset.usd"

    called: list[str] = []

    def _fake_convert(path: str) -> str:
        called.append(path)
        return str(generated_usd)

    monkeypatch.setattr(isaacsim_asset_util, "convert_urdf_to_usd_cached", _fake_convert)

    cfg = ArticulationObjCfg(
        name="asset",
        usd_path=None,
        urdf_path=str(urdf_path),
        mjcf_path=str(tmp_path / "asset.xml"),
    )

    assert isaacsim_asset_util.resolve_isaacsim_file_path(cfg) == str(generated_usd)
    assert called == [str(urdf_path)]


def test_resolve_isaacsim_file_path_raises_without_usd_or_urdf() -> None:
    from metasim.utils.isaacsim_asset_util import resolve_isaacsim_file_path

    cfg = ArticulationObjCfg(
        name="asset",
        usd_path=None,
        urdf_path=None,
        mjcf_path="/tmp/asset.xml",
    )

    # Commit ``e61505d isaacsim_asset_util: fall through to URDF/MJCF when
    # configured USD missing`` widened the resolver to consider mjcf_path as
    # well, and updated the error message to surface all three accepted paths.
    # The test still asserts the same behaviour (a non-existent mjcf path
    # cannot rescue absent usd/urdf), only the error string changed.
    with pytest.raises(ValueError, match="requires an existing usd_path, urdf_path, or mjcf_path"):
        resolve_isaacsim_file_path(cfg)
