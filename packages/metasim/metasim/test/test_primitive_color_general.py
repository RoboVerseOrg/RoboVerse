"""A primitive object config without ``color`` is rejected at construction, not at backend launch."""

from __future__ import annotations

import pytest

from metasim.scenario.objects import PrimitiveCubeCfg, PrimitiveSphereCfg

pytestmark = pytest.mark.general


def test_primitive_without_color_fails_at_construction():
    with pytest.raises(ValueError, match=r"PrimitiveCubeCfg\('c'\)\.color is required"):
        PrimitiveCubeCfg(name="c", size=(0.1, 0.1, 0.1))
    with pytest.raises(ValueError, match=r"PrimitiveSphereCfg\('s'\)\.color is required"):
        PrimitiveSphereCfg(name="s", radius=0.1)


def test_primitive_with_color_is_unchanged():
    assert PrimitiveCubeCfg(name="c", size=(0.1, 0.1, 0.1), color=[0.8, 0.1, 0.1]).color == [0.8, 0.1, 0.1]
