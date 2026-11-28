"""One decision about MuJoCo's OpenGL backend, taken once at the process entry.

MuJoCo reads ``MUJOCO_GL`` when the ``mujoco`` module is imported. Its Linux default is GLFW, which needs a display:
on a headless machine the first offscreen render (a camera, a depth task in ``bench health``) dies inside the GL
library instead of raising. :func:`apply_headless_gl` asks for EGL (or OSMesa) when, and only when, this is Linux,
no display is available, nothing was asked for and the library is actually loadable; otherwise MuJoCo's own default
stands. ``python -m metasim`` calls it before dispatching a command, the MuJoCo backend package calls it before
importing ``mujoco``, and content code that builds a handler headless calls it instead of setting the variable
itself, so there is exactly one rule.
"""

from __future__ import annotations

import ctypes
import os
import sys
from collections.abc import Callable, Mapping
from typing import Any

HEADLESS_LIBRARIES: tuple[tuple[str, str], ...] = (("libEGL.so.1", "egl"), ("libOSMesa.so.8", "osmesa"))
"""Shared libraries MuJoCo's offscreen backends need, in order of preference, and the ``MUJOCO_GL`` value each selects."""


def loadable(library: str) -> bool:
    """Whether ``ctypes`` can load ``library`` (what PyOpenGL will do), without spawning ``ldconfig`` or a compiler."""
    try:
        ctypes.CDLL(library)
    except OSError:
        return False
    return True


def choose_headless_gl(
    *,
    environ: Mapping[str, str] | None = None,
    platform: str | None = None,
    mujoco_loaded: bool | None = None,
    is_loadable: Callable[[str], bool] = loadable,
) -> tuple[str | None, str]:
    """The ``MUJOCO_GL`` value to set, or None, and the reason, as a pure function of the inputs.

    Returns ``(None, reason)`` when the variable is already set, a display is available, the platform is not Linux
    (MuJoCo accepts ``egl`` / ``osmesa`` on Linux only), no offscreen library is loadable, or ``mujoco`` was imported
    before the decision could take effect (then the reason says so and the caller warns).
    """
    env: Mapping[str, Any] = os.environ if environ is None else environ
    plat = sys.platform if platform is None else platform
    loaded = ("mujoco" in sys.modules) if mujoco_loaded is None else mujoco_loaded
    if "MUJOCO_GL" in env:
        return None, f"MUJOCO_GL={env['MUJOCO_GL']!r} was set explicitly"
    if env.get("DISPLAY") or env.get("WAYLAND_DISPLAY"):
        return None, "a display is available; MuJoCo's default backend stands"
    if not plat.startswith("linux"):
        return None, f"platform {plat!r}: MuJoCo accepts egl / osmesa on Linux only"
    for library, backend in HEADLESS_LIBRARIES:
        if is_loadable(library):
            if loaded:
                return None, (
                    f"no display and MUJOCO_GL unset, but `mujoco` was imported before this decision; set "
                    f"MUJOCO_GL={backend} before the first import to render headless"
                )
            return backend, f"no display; {library} is loadable"
    return None, "no display and no offscreen GL library (libEGL / libOSMesa); MuJoCo's default stands"


def apply_headless_gl(environ: dict[str, str] | None = None) -> str | None:
    """Set ``MUJOCO_GL`` when :func:`choose_headless_gl` picks a backend; returns what was set (None otherwise)."""
    env = os.environ if environ is None else environ
    backend, reason = choose_headless_gl(environ=env)
    if backend is not None:
        env["MUJOCO_GL"] = backend
        return backend
    if reason.startswith("no display and MUJOCO_GL unset, but"):
        from loguru import logger as log

        log.warning(f"metasim.sim.gl: {reason}")
    return None
