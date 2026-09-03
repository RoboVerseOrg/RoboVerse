"""``get_traj`` must fail fast with a documented exception, not silently.

Two regressions, both of which previously raised an exception OUTSIDE the
``(FileNotFoundError, KeyError, ValueError)`` set that task base classes catch
(e.g. ``LiberoBaseTask._get_initial_states``), so they surfaced as opaque
crashes far from the real cause instead of a graceful fallback:

- v1-format trajectory: the v1 branch only logged a warning and fell off the
  end, returning ``None`` → callers ``init_states, _, _ = get_traj(...)`` hit
  ``TypeError: cannot unpack non-iterable NoneType``.
- empty trajectory: ``data[0]`` on an empty list raised ``IndexError``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from metasim.utils.demo_util import get_traj


@pytest.mark.general
def test_get_traj_v1_format_raises_valueerror_not_silent_none():
    """A v1-format path (no 'v2' in the name) must raise ValueError, not
    return None (which used to crash callers with an out-of-band TypeError)."""
    robot = SimpleNamespace(name="franka")
    with pytest.raises(ValueError, match="v1"):
        get_traj("/data/demos/franka_v1.pkl.gz", robot)


@pytest.mark.general
def test_get_traj_v2_empty_trajectory_raises_valueerror(monkeypatch, tmp_path):
    """An empty trajectory list must raise ValueError before indexing data[0]
    (previously an IndexError that escaped the caller's catch tuple)."""
    import metasim.utils.demo_util.demo_util_v2 as v2

    robot = SimpleNamespace(name="franka")
    f = tmp_path / "franka_v2.pkl"  # must exist + endswith _v2.pkl for the asserts
    f.write_bytes(b"x")
    monkeypatch.setattr(v2, "load_traj_file", lambda p: {"franka": []})
    with pytest.raises(ValueError, match="empty"):
        v2.get_traj_v2(str(f), robot)
