"""OBJ/MTL dependency extraction surfaces I/O failures instead of swallowing.

``_extract_obj_dependencies`` / ``_extract_mtl_textures`` previously wrapped
their file reads in ``except Exception: pass``, so an unreadable OBJ/MTL
silently yielded a PARTIAL dependency list — the missing .mtl/texture then
failed to download and surfaced much later as a confusing missing-asset /
black-material error inside the simulator. The fix narrows the catch to
``OSError`` and logs a warning (matching the careful ``extract_paths_from_mjcf``
sibling), so the failure is visible at discovery time and any genuinely
unexpected exception propagates as a real bug.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from loguru import logger as _loguru_logger

from metasim.utils.parse_util import _extract_obj_dependencies


@pytest.fixture
def loguru_warnings():
    records: list[str] = []
    sink_id = _loguru_logger.add(lambda msg: records.append(str(msg)), level="WARNING")
    try:
        yield records
    finally:
        _loguru_logger.remove(sink_id)


@pytest.mark.general
def test_obj_dependencies_happy_path(tmp_path: Path):
    """A valid OBJ referencing an MTL that references a texture returns both
    (behaviour unchanged by the fix)."""
    (tmp_path / "mat.mtl").write_text("newmtl m\nmap_Kd wood.png\n")
    obj = tmp_path / "mesh.obj"
    obj.write_text("mtllib mat.mtl\nv 0 0 0\n")

    deps = _extract_obj_dependencies(str(obj))
    names = {Path(d).name for d in deps}
    assert "mat.mtl" in names
    assert "wood.png" in names


@pytest.mark.general
def test_obj_dependencies_ioerror_warns_not_silent(tmp_path: Path, loguru_warnings):
    """An unreadable OBJ (here: a directory, which open() rejects with an
    OSError subclass) must WARN and return a (partial/empty) list, not
    silently swallow and not crash."""
    bogus = tmp_path / "looks_like.obj"
    bogus.mkdir()  # exists, but open() raises IsADirectoryError (OSError subclass)

    deps = _extract_obj_dependencies(str(bogus))
    assert isinstance(deps, list)  # graceful, no crash
    out = "\n".join(loguru_warnings)
    assert "looks_like.obj" in out, f"expected a warning naming the unreadable OBJ, got: {out!r}"
