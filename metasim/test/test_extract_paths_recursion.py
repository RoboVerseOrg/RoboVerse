"""``extract_paths_from_mjcf`` recurses into ``<include>`` targets (L4).

Pre-fix the function listed the immediate ``<include file="..."/>``
targets as paths but stopped there — it never opened those files to
collect *their* mesh / texture / include references. MJCFs in the
humanoid-bench-style bundles routinely have 2-level-deep includes
(top-level scene includes a robot file which includes a per-link
materials file), so the transitive assets weren't enqueued and missed
the HF download pass. Run-time failure was "asset not found locally"
when the user wasn't looking for that asset by hand.

The fix recurses with a depth limit, guards against cycles via
already-visited tracking, and gracefully degrades when a transitive
include hasn't been fetched yet (returns the include path itself so
the caller can fetch it, but skips re-entering an unreadable file).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from metasim.utils.parse_util import extract_paths_from_mjcf


def _mjcf(tmp_path: Path, name: str, body: str) -> Path:
    """Write a minimal MJCF wrapping ``body``."""
    target = tmp_path / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(f"<mujoco>{body}</mujoco>")
    return target


@pytest.mark.general
def test_flat_mjcf_returns_immediate_paths(tmp_path):
    """Baseline — no includes, just mesh+texture refs."""
    (tmp_path / "meshes").mkdir()
    (tmp_path / "meshes" / "body.stl").write_text("")
    (tmp_path / "tex.png").write_text("")
    f = _mjcf(
        tmp_path,
        "flat.xml",
        '<asset><mesh file="meshes/body.stl"/><texture file="tex.png"/></asset>',
    )

    paths = extract_paths_from_mjcf(str(f))
    resolved = {Path(p).name for p in paths}
    assert "body.stl" in resolved
    assert "tex.png" in resolved


@pytest.mark.general
def test_recurses_into_two_level_include(tmp_path):
    """Top-level → robot.xml → materials.xml — transitive mesh resolved."""
    (tmp_path / "deep.stl").write_text("")
    (tmp_path / "shared.png").write_text("")

    materials = _mjcf(
        tmp_path,
        "materials.xml",
        '<asset><texture file="shared.png"/></asset>',
    )
    _mjcf(
        tmp_path,
        "robot.xml",
        f'<asset><mesh file="deep.stl"/></asset><include file="{materials.name}"/>',
    )
    top = _mjcf(tmp_path, "top.xml", '<include file="robot.xml"/>')

    paths = extract_paths_from_mjcf(str(top))
    names = {Path(p).name for p in paths}

    assert "robot.xml" in names, "first-level include should be listed"
    assert "materials.xml" in names, "second-level include should be listed"
    assert "deep.stl" in names, "transitive mesh should be discovered via recursion"
    assert "shared.png" in names, "transitive texture should be discovered via recursion"


@pytest.mark.general
def test_cycle_is_safe(tmp_path):
    """a.xml includes b.xml includes a.xml — recursion must terminate."""
    _mjcf(tmp_path, "a.xml", '<include file="b.xml"/>')
    _mjcf(tmp_path, "b.xml", '<include file="a.xml"/>')

    paths = extract_paths_from_mjcf(str(tmp_path / "a.xml"))
    names = {Path(p).name for p in paths}
    assert "a.xml" in names or "b.xml" in names  # at least one include surfaced
    # The real guarantee is the call returned at all (no recursion bomb).


@pytest.mark.general
def test_depth_limit_caps_runaway(tmp_path):
    """Chain a.xml → b.xml → c.xml → ... — depth limit terminates."""
    # Build a 20-deep chain.
    for i in range(20):
        body = "" if i == 19 else f'<include file="step{i + 1}.xml"/>'
        _mjcf(tmp_path, f"step{i}.xml", body)

    # Should not hang and should not raise.
    paths = extract_paths_from_mjcf(str(tmp_path / "step0.xml"), _max_include_depth=5)
    # We don't assert which paths come back — just that the call returns.
    assert isinstance(paths, list)


@pytest.mark.general
def test_missing_transitive_include_is_listed_not_fatal(tmp_path):
    """If the included file isn't on disk, surface its path (so the HF
    fetcher can try) but don't try to re-open it."""
    top = _mjcf(tmp_path, "top.xml", '<include file="not_fetched_yet.xml"/>')

    paths = extract_paths_from_mjcf(str(top))
    names = {Path(p).name for p in paths}
    assert "not_fetched_yet.xml" in names


@pytest.mark.general
def test_meshdir_compiler_attribute_honoured(tmp_path):
    """The ``<compiler meshdir="..."/>`` rebases relative mesh paths."""
    (tmp_path / "shared_meshes").mkdir()
    (tmp_path / "shared_meshes" / "shared.stl").write_text("")
    f = _mjcf(
        tmp_path,
        "with_meshdir.xml",
        '<compiler meshdir="shared_meshes"/>'
        '<asset><mesh file="shared.stl"/></asset>',
    )
    paths = extract_paths_from_mjcf(str(f))
    assert any(p.endswith("shared_meshes/shared.stl") for p in paths), paths
