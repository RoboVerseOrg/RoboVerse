"""Moved to :mod:`metasim.task.checkers.util`; this path keeps working as an alias."""

from __future__ import annotations

import sys

from metasim.task.checkers import util as _moved

sys.modules[__name__] = _moved
