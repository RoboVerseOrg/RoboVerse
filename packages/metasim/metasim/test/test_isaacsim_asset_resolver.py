"""``resolve_isaacsim_file_path`` must skip non-existent USDs and fall
through to URDF / MJCF conversion."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from metasim.utils.isaacsim_asset_util import resolve_isaacsim_file_path


@pytest.mark.general
def test_existing_usd_is_returned(tmp_path):
    usd = tmp_path / "robot.usd"
    usd.write_text("")
    cfg = SimpleNamespace(name="r", usd_path=str(usd), urdf_path=None, mjcf_path=None)
    assert resolve_isaacsim_file_path(cfg) == str(usd.resolve())


@pytest.mark.general
def test_missing_usd_skipped_then_urdf_used(tmp_path, monkeypatch):
    """A configured-but-absent USD path must NOT cause IsaacSim to try
    loading it. The resolver falls through to URDF conversion instead."""
    urdf = tmp_path / "robot.urdf"
    urdf.write_text("<robot/>")
    called: dict[str, str] = {}

    def _stub(p):
        called["urdf"] = p
        return "/tmp/converted.usd"

    monkeypatch.setattr("metasim.utils.isaacsim_asset_util.convert_urdf_to_usd_cached", _stub)
    cfg = SimpleNamespace(
        name="r",
        usd_path=str(tmp_path / "does_not_exist.usd"),
        urdf_path=str(urdf),
        mjcf_path=None,
    )
    out = resolve_isaacsim_file_path(cfg)
    assert called["urdf"] == str(urdf)
    assert out == "/tmp/converted.usd"


@pytest.mark.general
def test_missing_urdf_skipped_then_mjcf_used(tmp_path, monkeypatch):
    """When both USD and URDF point at non-existent paths, the resolver
    falls through to the MJCF converter."""
    mjcf = tmp_path / "robot.xml"
    mjcf.write_text("<mujoco/>")
    called: dict[str, str] = {}

    def _stub(p):
        called["mjcf"] = p
        return "/tmp/from_mjcf.usd"

    monkeypatch.setattr("metasim.utils.isaacsim_asset_util.convert_mjcf_to_usd_cached", _stub)
    cfg = SimpleNamespace(
        name="r",
        usd_path=str(tmp_path / "no.usd"),
        urdf_path=str(tmp_path / "no.urdf"),
        mjcf_path=str(mjcf),
    )
    out = resolve_isaacsim_file_path(cfg)
    assert called["mjcf"] == str(mjcf)
    assert out == "/tmp/from_mjcf.usd"


@pytest.mark.general
def test_no_paths_at_all_raises():
    cfg = SimpleNamespace(name="r", usd_path=None, urdf_path=None, mjcf_path=None)
    with pytest.raises(ValueError, match="usd_path"):
        resolve_isaacsim_file_path(cfg)


@pytest.mark.general
def test_all_paths_present_but_only_usd_used(tmp_path, monkeypatch):
    """When USD exists, neither converter runs — even if URDF/MJCF are present."""
    usd = tmp_path / "robot.usd"
    usd.write_text("")
    urdf = tmp_path / "robot.urdf"
    urdf.write_text("<robot/>")

    monkeypatch.setattr(
        "metasim.utils.isaacsim_asset_util.convert_urdf_to_usd_cached",
        lambda p: pytest.fail("URDF converter should not run when USD exists"),
    )
    monkeypatch.setattr(
        "metasim.utils.isaacsim_asset_util.convert_mjcf_to_usd_cached",
        lambda p: pytest.fail("MJCF converter should not run when USD exists"),
    )

    cfg = SimpleNamespace(name="r", usd_path=str(usd), urdf_path=str(urdf), mjcf_path=str(tmp_path / "robot.xml"))
    assert resolve_isaacsim_file_path(cfg) == str(usd.resolve())
