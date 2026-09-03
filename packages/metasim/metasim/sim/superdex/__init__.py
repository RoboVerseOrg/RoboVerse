"""SuperDex (Meta Mochi engine) simulation package.

``SuperdexHandler`` is imported lazily so that the pure-Python helpers (``_assets``) and their
general tests stay importable in environments without the ``superdex`` wheels.
"""

from __future__ import annotations

__all__ = ["SuperdexHandler"]


def __getattr__(name: str):
    if name == "SuperdexHandler":
        from .superdex import SuperdexHandler

        return SuperdexHandler
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
