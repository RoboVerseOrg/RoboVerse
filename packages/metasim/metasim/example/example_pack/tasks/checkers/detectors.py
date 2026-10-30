"""Moved to :mod:`metasim.task.checkers.detectors`; this path keeps working as an alias."""

from __future__ import annotations

import sys

from metasim.task.checkers import detectors as _moved

sys.modules[__name__] = _moved
