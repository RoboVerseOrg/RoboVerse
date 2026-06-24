"""Regression test (backend-free) for Libero90BaseTask._get_initial_states hardening.

The method must degrade to None (handler defaults) when the trajectory is
missing/empty or the scenario is robotless, matching the already-hardened
LiberoBaseTask. Previously it indexed ``scenario.robots[0]`` and called get_traj
with no guard, so those cases crashed.
"""

from __future__ import annotations

import pytest


@pytest.mark.general
def test_libero_90_get_initial_states_degrades_gracefully(monkeypatch):
    """Libero90BaseTask._get_initial_states must degrade to None (handler
    defaults) when the traj is missing/empty or the scenario is robotless,
    matching the hardened LiberoBaseTask. Previously it indexed
    ``scenario.robots[0]`` and called get_traj with no guard → crashed.
    """
    from types import SimpleNamespace

    import roboverse_pack.tasks.libero_90.libero_90_base as mod
    from roboverse_pack.tasks.libero_90.libero_90_base import Libero90BaseTask

    t = Libero90BaseTask.__new__(Libero90BaseTask)
    t.scenario = SimpleNamespace(robots=[])  # robotless
    t.num_envs = 1
    t.handler = None

    # 1. No traj_filepath → None (new guard; previously len(None) / robots[0] crash)
    t.traj_filepath = None
    assert t._get_initial_states() is None

    # 2. get_traj raises (missing file / bad key) → None (try/except)
    t.traj_filepath = "does/not/exist.pkl.gz"

    def _raise(*a, **k):
        raise FileNotFoundError("missing")

    monkeypatch.setattr(mod, "get_traj", _raise)
    assert t._get_initial_states() is None

    # 3. get_traj returns empty → None (empty guard before len())
    monkeypatch.setattr(mod, "get_traj", lambda *a, **k: ([], None, None))
    assert t._get_initial_states() is None
