"""General test for metasim.utils.color_util.hsv_to_rgb (no backend)."""

from __future__ import annotations

import colorsys

import pytest

from metasim.utils.color_util import hsv_to_rgb


@pytest.mark.general
@pytest.mark.parametrize("h", list(range(0, 360, 15)))
@pytest.mark.parametrize("s,v", [(1.0, 1.0), (0.5, 0.8), (0.3, 1.0)])
def test_hsv_to_rgb_matches_colorsys(h, s, v):
    """Regression: f was always 0 (wrong intermediate hues) and round(hi) could be 6.

    round(5.5)==6 fell through every branch and returned None (e.g. h=330).
    """
    got = hsv_to_rgb(h, s, v)
    assert got is not None, f"hsv_to_rgb returned None for h={h}"
    expected = colorsys.hsv_to_rgb(h / 360.0, s, v)
    assert max(abs(a - b) for a, b in zip(got, expected)) < 1e-6


@pytest.mark.general
def test_hsv_to_rgb_h330_not_none():
    """The specific hue that previously returned None (round(5.5)==6)."""
    assert hsv_to_rgb(330, 1.0, 1.0) is not None
