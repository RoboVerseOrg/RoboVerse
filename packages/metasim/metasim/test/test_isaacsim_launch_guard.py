"""General (non-GPU) tests for the IsaacSim launch-failure guard.

These cover the diagnostic hint that fires when an IsaacSim launch fails with a
numpy ABI mismatch (numpy>=2 against an IsaacSim/Isaac Lab build that requires
numpy<2). The ABI mismatch breaks the ``omni.syntheticdata`` camera-annotation
graph with ``TypeError: Unable to write from unknown dtype`` and previously left
the process hanging on the GPU; the guard now tears the app down and points at
the real cause. The hint logic is pure and needs no simulator/GPU.
"""

from __future__ import annotations

import numpy as np
import pytest

from metasim.sim.isaacsim.isaacsim import IsaacsimHandler


def _capture_hint(err: Exception) -> str:
    """Return the loguru output produced by ``_maybe_hint_numpy_abi`` for ``err``."""
    from loguru import logger as log

    records: list[str] = []
    sink_id = log.add(lambda msg: records.append(str(msg)), level="ERROR")
    try:
        IsaacsimHandler._maybe_hint_numpy_abi(err)
    finally:
        log.remove(sink_id)
    return "\n".join(records)


@pytest.mark.general
def test_numpy_abi_hint_fires_for_syntheticdata_dtype_error(monkeypatch):
    monkeypatch.setattr(np, "__version__", "2.4.5", raising=False)
    err = TypeError("Unable to write from unknown dtype, kind=f, size=0")
    out = _capture_hint(err)
    assert "numpy" in out and "numpy<2" in out


@pytest.mark.general
def test_numpy_abi_hint_fires_for_missing_rendervar(monkeypatch):
    monkeypatch.setattr(np, "__version__", "2.0.0", raising=False)
    err = RuntimeError("missing valid input renderVar LdrColorSD")
    out = _capture_hint(err)
    assert "numpy<2" in out


@pytest.mark.general
def test_numpy_abi_hint_silent_when_numpy_below_2(monkeypatch):
    monkeypatch.setattr(np, "__version__", "1.26.4", raising=False)
    err = TypeError("Unable to write from unknown dtype, kind=f, size=0")
    assert _capture_hint(err) == ""


@pytest.mark.general
def test_numpy_abi_hint_silent_for_unrelated_error(monkeypatch):
    monkeypatch.setattr(np, "__version__", "2.4.5", raising=False)
    err = ValueError("some unrelated launch failure")
    assert _capture_hint(err) == ""
