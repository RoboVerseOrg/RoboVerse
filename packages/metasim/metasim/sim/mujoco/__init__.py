# ruff: noqa: F401
"""MuJoCo backend.

MuJoCo picks its OpenGL backend from ``MUJOCO_GL`` when the ``mujoco`` module is imported, and its default on
Linux is GLFW, which needs a display: on a headless machine the first render (a camera, a depth task in
``bench health``) dies inside the GL library, not with a Python error. When no display is available, nothing was
asked for and the machine has libEGL (or OSMesa), this package asks for that backend before the import, the same
choice the nightly gate makes explicitly. Without either library MuJoCo's own default stands (asking for EGL on
such a machine would fail the import itself).
"""

import ctypes.util
import os
import sys


def headless_gl_backend() -> str | None:
    """The offscreen GL backend to ask MuJoCo for on this machine, or None when none is installed."""
    for lib, backend in (("EGL", "egl"), ("OSMesa", "osmesa")):
        if ctypes.util.find_library(lib):
            return backend
    return None


if "MUJOCO_GL" not in os.environ and not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
    _backend = headless_gl_backend()
    if _backend is None:
        pass  # nothing to render with offscreen; MuJoCo's default stands and rendering reports its own error
    elif "mujoco" in sys.modules:
        from loguru import logger as _log

        _log.warning(
            "metasim.sim.mujoco: no display and MUJOCO_GL unset, but `mujoco` was imported before this package, so "
            f"the GLFW backend is already chosen; set MUJOCO_GL={_backend} before the first import to render headless."
        )
    else:
        os.environ["MUJOCO_GL"] = _backend

from .mujoco import MujocoHandler
