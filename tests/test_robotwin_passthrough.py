"""Dep-free regression tests for the RoboTwin passthrough registration.

These lock the two invariants that make the passthrough safe to ship even when
RoboTwin's heavy deps (mplib / curobo / pytorch3d / torch 2.4.1) are absent:

1. task discovery is a pure filename scan that filters infra files, and
2. registration imports no RoboTwin code (only ``gym.make`` does, lazily), so a
   missing / broken checkout degrades to an actionable error, never an import crash.

No RoboTwin checkout is required: ``_discover_task_names`` is pointed at a tmp
``envs/`` dir, and the missing-checkout paths are exercised against a bogus dir.
"""

from __future__ import annotations

import pytest

from roboverse_pack.tasks.robotwin import _passthrough


@pytest.mark.general
def test_discover_filters_infra_files(tmp_path, monkeypatch):
    """Only real task modules are discovered; infra / underscore files are dropped."""
    envs = tmp_path / "envs"
    envs.mkdir()
    for name in ("beat_block_hammer", "stack_blocks_two", "place_shoe"):
        (envs / f"{name}.py").write_text("class Stub: ...\n")
    # Infra / non-task files that MUST be filtered out.
    for name in ("_base_task", "_GLOBAL_CONFIGS", "__init__", "_helper"):
        (envs / f"{name}.py").write_text("# infra\n")
    (envs / "notes.txt").write_text("not python\n")

    names = _passthrough._discover_task_names(str(tmp_path))
    assert names == ["beat_block_hammer", "place_shoe", "stack_blocks_two"]  # sorted, infra removed


@pytest.mark.general
def test_discover_missing_checkout_returns_empty(tmp_path):
    """A missing envs/ dir yields [] (no checkout) rather than raising."""
    assert _passthrough._discover_task_names(str(tmp_path / "nonexistent")) == []


@pytest.mark.general
def test_register_is_lazy_idempotent_and_safe(tmp_path, monkeypatch):
    """Registration scans filenames only, is idempotent, and imports no RoboTwin."""
    import sys

    envs = tmp_path / "envs"
    envs.mkdir()
    for name in ("beat_block_hammer", "stack_blocks_two"):
        (envs / f"{name}.py").write_text("class Stub: ...\n")
    monkeypatch.setenv("ROBOTWIN_DIR", str(tmp_path))

    ids = _passthrough.register_robotwin_passthrough(prefix="RoboTwinTest/")
    assert set(ids) == {"RoboTwinTest/beat_block_hammer", "RoboTwinTest/stack_blocks_two"}
    # Idempotent: a second call returns the same ids and does not raise on re-register.
    ids2 = _passthrough.register_robotwin_passthrough(prefix="RoboTwinTest/")
    assert set(ids2) == set(ids)
    # Registration imported no RoboTwin task module (lazy entry point only).
    assert not any(m == "envs" or m.startswith("envs.") for m in sys.modules)


@pytest.mark.general
def test_list_robotwin_tasks(tmp_path, monkeypatch):
    """list_robotwin_tasks returns prefixed ids (and bare names) from the checkout."""
    envs = tmp_path / "envs"
    envs.mkdir()
    for name in ("beat_block_hammer", "place_shoe"):
        (envs / f"{name}.py").write_text("class Stub: ...\n")
    (envs / "_base_task.py").write_text("# infra\n")
    monkeypatch.setenv("ROBOTWIN_DIR", str(tmp_path))
    assert _passthrough.list_robotwin_tasks() == ["RoboTwin/beat_block_hammer", "RoboTwin/place_shoe"]
    assert _passthrough.list_robotwin_tasks(prefix="") == ["beat_block_hammer", "place_shoe"]
    # Missing checkout -> [].
    monkeypatch.setenv("ROBOTWIN_DIR", str(tmp_path / "nope"))
    assert _passthrough.list_robotwin_tasks() == []


@pytest.mark.general
def test_register_no_checkout_returns_empty(tmp_path, monkeypatch):
    """With ROBOTWIN_DIR pointing nowhere, registration returns [] without error."""
    monkeypatch.setenv("ROBOTWIN_DIR", str(tmp_path / "nope"))
    assert _passthrough.register_robotwin_passthrough(prefix="RoboTwinMissing/") == []


@pytest.mark.general
def test_make_env_missing_checkout_raises_actionable(tmp_path, monkeypatch):
    """gym-make factory raises a clear, actionable error when the checkout is absent."""
    monkeypatch.setenv("ROBOTWIN_DIR", str(tmp_path / "nope"))
    with pytest.raises(RuntimeError, match="RoboTwin checkout not found"):
        _passthrough._make_robotwin_env(task_name="beat_block_hammer")
