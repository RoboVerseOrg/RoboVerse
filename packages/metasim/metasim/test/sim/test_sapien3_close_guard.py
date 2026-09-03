"""``Sapien3Handler.close`` survives a launch that never built the viewer (L2).

Pre-fix:

    def close(self):
        if not self.headless:
            self.viewer.close()   # AttributeError if launch never built it
        self.scene = None

If ``launch`` raised before ``_build_sapien`` assigned ``self.viewer``
(or if the handler is being closed from a context manager whose
``__enter__`` raised), ``close`` raised ``AttributeError`` and *masked*
the real launch-time exception.

The fix uses ``getattr(self, "viewer", None)`` and wraps the close call
in a try/except so teardown never amplifies an upstream failure.

Tests inspect the source rather than spawning Sapien (which needs the
``sapien`` package + a display).
"""

from __future__ import annotations

import pytest


def _get_close_source() -> str:
    """Read ``Sapien3Handler.close`` source without importing sapien."""
    from pathlib import Path

    import metasim

    repo_root = Path(metasim.__file__).resolve().parent
    text = (repo_root / "sim" / "sapien" / "sapien3.py").read_text()
    # Find the close method body by scanning for its signature.
    marker = "    def close(self):"
    start = text.index(marker)
    # close runs until the next ``def `` at the same indent.
    after = text[start + len(marker) :]
    end = after.find("\n    def ")
    return text[start : start + len(marker) + (end if end != -1 else len(after))]


@pytest.mark.general
def test_close_uses_getattr_for_viewer():
    """The fix uses ``getattr(self, "viewer", None)`` not ``self.viewer``."""
    src = _get_close_source()
    assert 'getattr(self, "viewer"' in src or "getattr(self, 'viewer'" in src, (
        "Sapien3Handler.close must use getattr to tolerate a never-built viewer"
    )


@pytest.mark.general
def test_close_does_not_unconditionally_access_self_viewer():
    """Lock the regression: no line like ``self.viewer.close()`` without a guard."""
    src = _get_close_source()
    # Allow it INSIDE a guarded branch. Reject bare ``self.viewer.close()`` not preceded by a None check.
    for line in src.splitlines():
        stripped = line.strip()
        if stripped == "self.viewer.close()":
            pytest.fail(f"unguarded self.viewer.close() in Sapien3Handler.close:\n{src}")


@pytest.mark.general
def test_close_wraps_viewer_close_in_try_except():
    """A viewer that errors during shutdown must not amplify the real cause
    of failure."""
    src = _get_close_source()
    assert "try" in src and "except" in src, (
        "Sapien3Handler.close must guard viewer.close() in a try/except so a "
        "viewer-shutdown error doesn't mask whatever caused the close to fire"
    )
