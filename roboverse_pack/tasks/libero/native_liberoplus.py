"""Native MetaSim LIBERO-plus tasks — perturbed scenes, bitwise 1:1.

`LIBERO-plus <https://github.com/sylvestf/LIBERO-plus>`_ stresses LIBERO policies
with scene perturbations (lighting, distractor objects, table/floor textures,
language rephrasings, …). Each perturbed task is, physically, a LIBERO robosuite
scene with a modified compiled ``model_file`` — so it drops straight into the
native-LIBERO machinery: load the perturbed MJCF on the groundless
``MujocoHandler``, drive it with the native robosuite-free OSC controller, and
judge success natively from the carried-over BDDL goal. No ``import libero`` /
``import robosuite`` at runtime.

Bundles (``native_bundles_plus/<dim>__<suite>__<stem>/``) are built by
``tools/libero_integration/gen_liberoplus.py`` in a dedicated robosuite-1.4 env
(it lazily pulls only each task's asset files out of the 6.4GB LIBERO-plus
``assets.zip`` via HTTP range requests — no full download) and verified bitwise
against raw MuJoCo by ``tools/libero_integration/verify_liberoplus.py``. The
perturbed mesh/texture assets are data, like base LIBERO's: point
``LIBERO_PLUS_ASSETS`` at the LIBERO-plus ``libero/libero/assets`` overlay.
"""

from __future__ import annotations

import os
from pathlib import Path

from metasim.task.registry import register_task

from ._native_util import remap_liberoplus_model
from .native_libero import NativeLiberoEnv, _libero_assets

_PLUS_BUNDLES = Path(__file__).resolve().parent / "native_bundles_plus"
_DEFAULT_PLUS_ASSETS = Path(__file__).resolve().parents[3] / "third_party/LIBERO-plus/libero/libero/assets"


def _plus_assets() -> str:
    return os.environ.get("LIBERO_PLUS_ASSETS", str(_DEFAULT_PLUS_ASSETS))


class NativeLiberoPlusEnv(NativeLiberoEnv):
    """A LIBERO-plus perturbed task on MetaSim's MujocoHandler."""

    bundles_root = _PLUS_BUNDLES

    def _remap_model(self, model_xml: str) -> str:
        return remap_liberoplus_model(model_xml, _libero_assets(), _plus_assets())


def _register_bundled_tasks():
    """Register a ``libero_plus_native.<dim>__<suite>__<stem>`` task per vendored bundle."""
    if not _PLUS_BUNDLES.exists():
        return
    for bdir in sorted(_PLUS_BUNDLES.iterdir()):
        if not (bdir / "model.xml").exists():
            continue
        name = bdir.name
        cls = type(
            f"NativeLiberoPlus_{name}",
            (NativeLiberoPlusEnv,),
            {"bundle": name, "__doc__": f"LIBERO-plus {name} (native)."},
        )
        register_task(f"libero_plus_native.{name}")(cls)


_register_bundled_tasks()
