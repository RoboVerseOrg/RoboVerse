"""Sub-module containing utilities for color conversion."""

from __future__ import annotations

## see https://stackoverflow.com/a/1586291


def hsv_to_rgb(h: float, s: float, v: float) -> tuple[float, float, float]:
    """Converts HSV value to RGB values.

    Hue is in range 0-359 (degrees), value/saturation are in range 0-1 (float)

    Direct implementation of:
    http://en.wikipedia.org/wiki/HSL_and_HSV#Conversion_from_HSV_to_RGB
    """
    h, s, v = [float(x) for x in (h, s, v)]

    # Sector index (floor, not round: round(5.5)=6 fell through every branch and
    # returned None). f is the fractional part within the sector — the previous
    # ``(h / 60) - (h / 60)`` was always 0, so intermediate hues came out wrong.
    hi = int(h / 60) % 6
    f = (h / 60) - int(h / 60)
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)

    if hi == 0:
        return v, t, p
    elif hi == 1:
        return q, v, p
    elif hi == 2:
        return p, v, t
    elif hi == 3:
        return p, q, v
    elif hi == 4:
        return t, p, v
    elif hi == 5:
        return v, p, q
