# ruff: noqa: F401
"""MuJoCo backend.

MuJoCo picks its OpenGL backend from ``MUJOCO_GL`` when the ``mujoco`` module is imported, and its default on
Linux is GLFW, which needs a display: on a headless machine the first render (a camera, a depth task in
``bench health``) dies inside the GL library, not with a Python error. When no display is available and nothing
was asked for, this package asks for EGL before the import, the same choice the nightly gate makes explicitly.
"""

import os
import sys

if "MUJOCO_GL" not in os.environ and not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
    if "mujoco" in sys.modules:
        from loguru import logger as _log

        _log.warning(
            "metasim.sim.mujoco: no display and MUJOCO_GL unset, but `mujoco` was imported before this package, so "
            "the GLFW backend is already chosen; set MUJOCO_GL=egl before the first import to render headless."
        )
    else:
        os.environ["MUJOCO_GL"] = "egl"

from .mujoco import MujocoHandler
