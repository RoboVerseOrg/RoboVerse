"""``MJXHandler.close()`` must not crash on camera-less or headless runs (H8).

The previous implementation referenced ``self._renderer`` directly:

    def close(self):
        if hasattr(self, "_viewer") and self._renderer is not None:
            self._viewer.close()

``self._renderer`` is only assigned in launch() when ``self.cameras`` is
non-empty, so a camera-less run hit ``AttributeError: 'MJXHandler' object
has no attribute '_renderer'`` on close — aborting any orderly teardown
downstream.

These tests run the close() logic in isolation against a stub instance —
no mjx / jax dependency required.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_close_method():
    """Pull MJXHandler.close out of mjx.py without importing the module
    (which would trigger jax / mjx imports that aren't gated here)."""
    src = Path(__file__).resolve().parents[2] / "sim" / "mjx" / "mjx.py"
    if not src.is_file():
        pytest.skip(f"mjx.py not at expected location: {src}")
    source = src.read_text()
    # The close() body is small and self-contained. Extract by simple search.
    marker = "    def close(self):"
    idx = source.find(marker)
    if idx < 0:
        pytest.skip("could not locate MJXHandler.close in mjx.py")
    end = source.find("\n    def ", idx + 1)
    close_src = source[idx:end] if end > 0 else source[idx:]
    # Wrap in a callable class for the test
    wrapper = "class _MjxLike:\n" + close_src
    ns: dict = {}
    exec(wrapper, ns)
    return ns["_MjxLike"]


_MjxLike = _load_close_method()


@pytest.mark.general
def test_mjx_close_handles_no_renderer():
    """No cameras → _renderer never assigned; close() should be a no-op,
    not AttributeError."""
    h = _MjxLike()
    # Mimic the post-launch state for a camera-less run: neither attr set.
    h.close()


@pytest.mark.general
def test_mjx_close_handles_no_viewer():
    """Headless → _viewer never assigned; close() should still tear down
    the renderer when one exists."""
    closed = []

    class _FakeRenderer:
        def close(self):
            closed.append("renderer")

    h = _MjxLike()
    h._renderer = _FakeRenderer()
    h.close()
    assert closed == ["renderer"]


@pytest.mark.general
def test_mjx_close_handles_both_present():
    """GUI + cameras → both should be closed in order."""
    closed = []

    class _FakeViewer:
        def close(self):
            closed.append("viewer")

    class _FakeRenderer:
        def close(self):
            closed.append("renderer")

    h = _MjxLike()
    h._viewer = _FakeViewer()
    h._renderer = _FakeRenderer()
    h.close()
    assert closed == ["viewer", "renderer"]


@pytest.mark.general
def test_mjx_close_tolerates_underlying_close_exception():
    """close()-time exceptions in underlying primitives shouldn't abort
    the rest of the teardown — we need the renderer to be released even
    if the viewer raised, etc."""
    closed = []

    class _BadViewer:
        def close(self):
            closed.append("viewer-tried")
            raise RuntimeError("simulated viewer.close() failure")

    class _OkRenderer:
        def close(self):
            closed.append("renderer")

    h = _MjxLike()
    h._viewer = _BadViewer()
    h._renderer = _OkRenderer()
    h.close()  # must not propagate the RuntimeError
    assert closed == ["viewer-tried", "renderer"]
