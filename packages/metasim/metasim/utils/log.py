"""Logging helpers for the library: process-wide one-shot warnings and an opt-in level switch.

MetaSim logs through ``loguru``. Two things used to make its output unusable at scale: warnings
that were deduplicated per *handler instance* (128 envs by 9 actuators gave 1,000 copies of the same
line) and ``hf_util`` announcing every locally present asset at INFO (70% of a typical run's log).
:func:`warn_once` keys a warning on a value of the caller's choosing and emits it once per process;
:func:`configure_logging` applies ``METASIM_LOG_LEVEL`` when the user sets it and does nothing
otherwise, so importing the library never changes an application's logging setup.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Hashable

from loguru import logger as log

_WARNED: set[Hashable] = set()
_CONFIGURED = False


def warn_once(key: Hashable, message: str) -> bool:
    """Emit ``message`` as a warning the first time ``key`` is seen in this process.

    Returns True when the message was emitted. Use a key that identifies the *situation*
    (``("mujoco.forcerange", robot, joint)``), not the handler instance, so parallel envs and
    repeated resets do not repeat it.
    """
    if key in _WARNED:
        return False
    _WARNED.add(key)
    log.opt(depth=1).warning(message)
    return True


def reset_warn_once() -> None:
    """Forget every key (tests)."""
    _WARNED.clear()


def configure_logging(level: str | None = None, *, force: bool = False) -> str | None:
    """Apply a log level to loguru's default sink.

    ``level`` defaults to ``$METASIM_LOG_LEVEL``. When neither is given nothing happens and the
    application keeps whatever sinks it configured. Returns the level applied, or None.
    """
    global _CONFIGURED
    level = level or os.environ.get("METASIM_LOG_LEVEL")
    if not level or (_CONFIGURED and not force):
        return None
    log.remove()
    log.add(sys.stderr, level=level.upper())
    _CONFIGURED = True
    return level.upper()
