"""Unit tests for metasim.utils.hf_util.FileDownloader None-handling and error messages."""

from __future__ import annotations

import pytest

import metasim.utils.hf_util as hf_util
from metasim.scenario.objects import (
    ArticulationObjCfg,
    PrimitiveCubeCfg,
    PrimitiveCylinderCfg,
    PrimitiveSphereCfg,
    RigidObjCfg,
)
from metasim.scenario.scenario import ScenarioCfg
from metasim.utils.hf_util import FileDownloader, check_and_download_single


class _FakeScene:
    """Duck-typed stand-in for SceneCfg.

    metasim.scenario.scene.SceneCfg has a pre-existing super().__post_init__
    bug preventing direct instantiation; FileDownloader only calls .file_name(sim).
    """

    def __init__(self, filepath):
        self._filepath = filepath

    def file_name(self, sim):
        return self._filepath


@pytest.mark.general
def test_check_and_download_single_accepts_none():
    """None filepath must be a no-op (regression: old code hit os.path.exists(None) -> TypeError)."""
    check_and_download_single(None)


@pytest.mark.general
def test_check_and_download_single_falls_back_to_private_roboverse_data(monkeypatch):
    """Assets missing from the public data repo can be downloaded from the private company repo."""
    filepath = "roboverse_data/private/object.usd"
    file_exists_calls = []
    download_calls = []

    def fake_file_exists(repo_id, filename, repo_type):
        file_exists_calls.append((repo_id, filename, repo_type))
        return repo_id == "SomaStacksOrg/roboverse_data"

    def fake_hf_hub_download(repo_id, filename, repo_type, local_dir):
        download_calls.append((repo_id, filename, repo_type, local_dir))

    monkeypatch.setattr(hf_util.os.path, "exists", lambda path: False)
    monkeypatch.setattr(hf_util.hf_api, "file_exists", fake_file_exists)
    monkeypatch.setattr(hf_util, "hf_hub_download", fake_hf_hub_download)

    check_and_download_single(filepath)

    assert file_exists_calls == [
        ("RoboVerseOrg/roboverse_data", "private/object.usd", "dataset"),
        ("SomaStacksOrg/roboverse_data", "private/object.usd", "dataset"),
    ]
    assert download_calls == [
        ("SomaStacksOrg/roboverse_data", "private/object.usd", "dataset", "roboverse_data"),
    ]


@pytest.mark.general
def test_add_from_object_raises_when_sim_path_missing():
    """Object lacking the sim-specific asset path should raise ValueError with diagnostic info."""
    scenario = ScenarioCfg(
        simulator="mujoco",
        objects=[
            RigidObjCfg(
                name="test_obj",
                usd_path="fake.usd",
                urdf_path="fake.urdf",
            ),
        ],
    )
    with pytest.raises(ValueError) as excinfo:
        FileDownloader(scenario)

    msg = str(excinfo.value)
    assert "test_obj" in msg
    assert "mjcf" in msg
    assert "mujoco" in msg


@pytest.mark.general
def test_value_error_lists_all_path_fields():
    """Error message should surface every asset path field so the user can tell what's set."""
    scenario = ScenarioCfg(
        simulator="mujoco",
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="a.usd",
                urdf_path="b.urdf",
            ),
        ],
    )
    with pytest.raises(ValueError) as excinfo:
        FileDownloader(scenario)

    msg = str(excinfo.value)
    assert "usd_path=a.usd" in msg
    assert "urdf_path=b.urdf" in msg
    assert "mjcf_path=None" in msg


@pytest.mark.general
def test_rigid_object_with_matching_path_is_added():
    scenario = ScenarioCfg(
        simulator="mujoco",
        objects=[RigidObjCfg(name="obj", mjcf_path="obj.xml")],
    )
    dl = FileDownloader(scenario)
    assert "obj.xml" in dl.files_to_download


@pytest.mark.general
def test_articulation_object_with_matching_path_is_added():
    scenario = ScenarioCfg(
        simulator="pybullet",
        objects=[ArticulationObjCfg(name="arm", urdf_path="arm.urdf")],
    )
    dl = FileDownloader(scenario)
    assert "arm.urdf" in dl.files_to_download


@pytest.mark.general
def test_primitive_objects_are_skipped():
    """Primitives have no asset file — they must not trigger errors or downloads."""
    scenario = ScenarioCfg(
        simulator="mujoco",
        objects=[
            PrimitiveCubeCfg(name="cube", size=[0.1, 0.1, 0.1], color=[1, 0, 0]),
            PrimitiveSphereCfg(name="ball", radius=0.05, color=[0, 1, 0]),
            PrimitiveCylinderCfg(name="cyl", radius=0.05, height=0.1, color=[0, 0, 1]),
        ],
    )
    dl = FileDownloader(scenario)
    assert dl.files_to_download == []


@pytest.mark.general
def test_scene_without_path_is_silently_skipped():
    """SceneCfg legitimately may have no asset — downloader should skip, not raise."""
    scenario = ScenarioCfg(simulator="mujoco", scene=_FakeScene(None))
    dl = FileDownloader(scenario)
    assert dl.files_to_download == []


@pytest.mark.general
def test_scene_with_path_is_added():
    scenario = ScenarioCfg(simulator="mujoco", scene=_FakeScene("scene.xml"))
    dl = FileDownloader(scenario)
    assert "scene.xml" in dl.files_to_download


@pytest.mark.general
def test_extra_resources_are_added():
    scenario = ScenarioCfg(
        simulator="mujoco",
        objects=[
            RigidObjCfg(
                name="obj",
                mjcf_path="main.xml",
                extra_resources=["tex1.png", "tex2.png"],
            ),
        ],
    )
    dl = FileDownloader(scenario)
    assert "main.xml" in dl.files_to_download
    assert "tex1.png" in dl.files_to_download
    assert "tex2.png" in dl.files_to_download


@pytest.mark.general
def test_extra_resources_none_entry_is_skipped():
    """Defensive: list[str] is the declared type, but guard against runtime None too."""
    scenario = ScenarioCfg(
        simulator="mujoco",
        objects=[
            RigidObjCfg(
                name="obj",
                mjcf_path="main.xml",
                extra_resources=["tex.png", None, "other.png"],
            ),
        ],
    )
    dl = FileDownloader(scenario)
    assert "main.xml" in dl.files_to_download
    assert "tex.png" in dl.files_to_download
    assert "other.png" in dl.files_to_download
    assert None not in dl.files_to_download


@pytest.mark.general
def test_mixed_scenario_aggregates_all_files():
    """End-to-end: primitives skipped, scene + object + extras all collected."""
    scenario = ScenarioCfg(
        simulator="mujoco",
        scene=_FakeScene("scene.xml"),
        objects=[
            PrimitiveCubeCfg(name="cube", size=[0.1, 0.1, 0.1], color=[1, 0, 0]),
            RigidObjCfg(
                name="obj",
                mjcf_path="obj.xml",
                extra_resources=["t.png"],
            ),
        ],
    )
    dl = FileDownloader(scenario)
    assert set(dl.files_to_download) == {"scene.xml", "obj.xml", "t.png"}
