"""Regression: the RoboTwin asset locator must make ~/projects/robotwin deletable.

Every asset a bridge references is addressed by its RoboTwin-internal relpath and resolved by
``roboverse_pack.tasks.robotwin._locator.robotwin_asset``. These lock the resolution contract
that lets the replay run *without the upstream RoboTwin checkout*: prefer a local clone, else the
vendored ``roboverse_data/robotwin/`` mirror; when ``ROBOTWIN_ASSETS`` is set it is authoritative
(no implicit ``~/projects/robotwin`` fallback), which is how the deletability test forces the
mirror path. No sim or real assets required -- synthetic trees only.
"""

from __future__ import annotations

import importlib.util
import os

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_LOC = os.path.join(_HERE, "..", "roboverse_pack", "tasks", "robotwin", "_locator.py")


def _load_locator():
    spec = importlib.util.spec_from_file_location("rt_locator", _LOC)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def loc():
    return _load_locator()


def _touch(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("x")


def test_prefers_clone_when_present(loc, tmp_path, monkeypatch):
    clone = tmp_path / "clone"
    rel = "assets/objects/020_hammer/visual/textured.obj"
    _touch(str(clone / rel))
    monkeypatch.setenv("ROBOTWIN_ASSETS", str(clone))
    got = loc.robotwin_asset(rel, download=False)
    assert got == os.path.abspath(str(clone / rel))
    assert os.path.isabs(got)


def test_falls_back_to_mirror_when_clone_absent(loc, tmp_path, monkeypatch):
    # ROBOTWIN_ASSETS points at a non-existent clone -> authoritative -> use the mirror.
    monkeypatch.setenv("ROBOTWIN_ASSETS", str(tmp_path / "does_not_exist"))
    monkeypatch.chdir(tmp_path)
    rel = "assets/objects/060_kitchenpot/0/mobility.urdf"
    _touch(str(tmp_path / "roboverse_data" / "robotwin" / rel))
    got = loc.robotwin_asset(rel, download=False)
    assert got == os.path.abspath(os.path.join("roboverse_data", "robotwin", rel))
    assert os.path.exists(got)


def test_env_is_authoritative_no_home_fallback(loc, tmp_path, monkeypatch):
    # Even if ~/projects/robotwin exists, an explicit (bad) ROBOTWIN_ASSETS must NOT fall back to it.
    monkeypatch.setenv("ROBOTWIN_ASSETS", str(tmp_path / "nope"))
    assert loc.robotwin_clone_root() is None


def test_accepts_assets_subdir_form(loc, tmp_path, monkeypatch):
    # ROBOTWIN_ASSETS may point at <root>/assets; the root (its parent) is what we return.
    clone = tmp_path / "clone"
    (clone / "assets").mkdir(parents=True)
    monkeypatch.setenv("ROBOTWIN_ASSETS", str(clone / "assets"))
    assert loc.robotwin_clone_root() == clone


def test_missing_everywhere_raises(loc, tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOTWIN_ASSETS", str(tmp_path / "nope"))
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError):
        loc.robotwin_asset("assets/objects/zzz/visual/missing.glb", download=False)


def test_leading_slash_is_stripped(loc, tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOTWIN_ASSETS", str(tmp_path / "nope"))
    monkeypatch.chdir(tmp_path)
    rel = "assets/objects/x/visual/a.glb"
    _touch(str(tmp_path / "roboverse_data" / "robotwin" / rel))
    # A leading slash must not turn it into an absolute path that escapes the mirror.
    got = loc.robotwin_asset("/" + rel, download=False)
    assert got.endswith(rel)
