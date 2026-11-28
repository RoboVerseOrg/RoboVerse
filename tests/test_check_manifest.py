"""The asset manifest scanner: which `roboverse_data/...` literals are checked, and how f-strings are told apart."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _tool():
    spec = importlib.util.spec_from_file_location("check_manifest", ROOT / "tools" / "assets" / "check_manifest.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["check_manifest"] = module
    spec.loader.exec_module(module)
    return module


def test_a_static_literal_is_checked_even_when_an_fstring_builds_the_same_path(tmp_path, monkeypatch):
    tool = _tool()
    monkeypatch.setattr(tool, "REPO", tmp_path)
    pack = tmp_path / "pack"
    pack.mkdir()
    (pack / "task_a.py").write_text('traj_filepath = "roboverse_data/trajs/foo/v2"\n')
    (pack / "task_b.py").write_text('def p(robot):\n    return f"roboverse_data/trajs/foo/v2/{robot}.pkl"\n')
    (pack / "msg.py").write_text('def m(x):\n    return f"asset {x!r} not under roboverse_data/mirror/. Run it"\n')
    (pack / "doc.py").write_text('"""Assets live under roboverse_data/prose/ on the Hub."""\n\nX = 1\n')
    static, dynamic = tool.references([pack])
    assert not any(k.startswith("roboverse_data/prose") for k in static), "a docstring is prose, not a reference"
    assert "roboverse_data/trajs/foo/v2" in static, "task_a's literal is a real reference and must be checked"
    assert all(loc[0].endswith("task_a.py") for loc in static["roboverse_data/trajs/foo/v2"])
    assert "roboverse_data/trajs/foo/v2/" in dynamic or "roboverse_data/trajs/foo/v2" in dynamic
    assert not any(k.startswith("roboverse_data/mirror") for k in static), (
        "an f-string part is never a static reference"
    )


def test_resolve_accepts_hub_files_directories_and_local_files(tmp_path, monkeypatch):
    tool = _tool()
    files = {"trajs/foo/v2/franka_v2.pkl.gz", "robots/franka/mjcf/panda.xml"}
    prefixes = {"trajs/foo/v2", "robots/franka/mjcf"}
    assert tool.resolve("roboverse_data/robots/franka/mjcf/panda.xml", files, prefixes, hub_only=True)
    assert tool.resolve("roboverse_data/trajs/foo/v2", files, prefixes, hub_only=True), "a stem names a directory"
    assert not tool.resolve("roboverse_data/trajs/bar/v2", files, prefixes, hub_only=True)
    monkeypatch.setattr(tool, "REPO", tmp_path)
    (tmp_path / "roboverse_data" / "local").mkdir(parents=True)
    (tmp_path / "roboverse_data" / "local" / "x.xml").write_text("<mujoco/>")
    assert tool.resolve("roboverse_data/local/x.xml", files, prefixes) and not tool.resolve(
        "roboverse_data/local/x.xml", files, prefixes, hub_only=True
    )
