"""General tests for metasim.scenario.grounds terrain configs (no backend)."""

from __future__ import annotations

import pytest

from metasim.scenario import grounds as grounds_mod
from metasim.scenario.grounds import BaseTerrainCfg, PitCfg


@pytest.mark.general
def test_pit_cfg_uses_origin_for_placement():
    """Regression: PitCfg declared 'position' but the terrain builder reads config.origin.

    So a user-set placement was silently ignored (the inherited origin=[0,0] won).
    """
    cfg = PitCfg(origin=[5.0, 5.0])
    assert cfg.origin == [5.0, 5.0]
    assert not hasattr(cfg, "position"), "PitCfg should not have a stray 'position' field"


@pytest.mark.general
def test_all_terrain_cfgs_expose_origin():
    """Every concrete terrain cfg must expose 'origin' (the field _make_* builders read)."""
    for name in dir(grounds_mod):
        obj = getattr(grounds_mod, name)
        if isinstance(obj, type) and issubclass(obj, BaseTerrainCfg) and obj is not BaseTerrainCfg:
            assert hasattr(obj(), "origin"), f"{name} is missing the 'origin' placement field"
