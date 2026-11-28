"""Asset files behind a config: the format each field promises, and whether a USD stage's composition resolves.

A robot or object config names its assets per format (``mjcf_path``, ``urdf_path``, ``usd_path``, ``mjx_mjcf_path``).
Two defects reach a simulator late and fail there with an unrelated message: a field that names a file of another
format (``usd_path`` pointing at an MJCF: Isaac Sim spawns an empty prim and reports "no rigid bodies"), and a USD
whose payloads or references live next to it but were never fetched (the stage opens, with nothing in it). Both are
decided here, from the paths alone, before a backend starts.

``check_asset_fields`` needs nothing but the config. ``usd_dependencies`` needs the ``pxr`` the process already has
(Isaac Sim's once it is up, or ``usd-core`` imported on purpose); without one the composition is not judged, and the
callers say so instead of guessing. A USD fetched from the Hub comes with its folder, so a fresh machine does not
depend on this check (``metasim.utils.hf_util``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

ASSET_FIELDS: dict[str, tuple[str, ...]] = {
    "mjcf_path": (".xml",),
    "mjx_mjcf_path": (".xml",),
    "urdf_path": (".urdf",),
    "usd_path": (".usd", ".usda", ".usdc", ".usdz"),
}
"""Config field -> the file suffixes that field may name."""

USD_SUFFIXES = ASSET_FIELDS["usd_path"]


def check_asset_fields(cfg) -> list[str]:
    """Problems with the asset fields of ``cfg``: a declared path whose suffix is not the field's format.

    A field that is unset, ``None`` or ``dataclasses.MISSING`` is not declared and is not a problem.
    """
    problems = []
    for name, suffixes in ASSET_FIELDS.items():
        path = getattr(cfg, name, None)
        if not isinstance(path, str) or not path:
            continue
        if not path.lower().endswith(suffixes):
            problems.append(f"{name}={path!r} is not a {' / '.join(suffixes)} file")
    return problems


@dataclass
class UsdDependencies:
    """The files a USD stage composes from (sublayers, references, payloads), followed recursively."""

    path: str
    resolved: list[str] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Every file the stage composes from exists."""
        return not self.unresolved


def _layer_dependencies(layer) -> list[str]:
    """The asset paths one layer composes from, as authored (``GetCompositionAssetDependencies`` when it exists)."""
    getter = getattr(layer, "GetCompositionAssetDependencies", None) or layer.GetExternalReferences
    return [p for p in getter() if p]  # anonymous layers are the empty string


def _pxr_sdf(allow_import: bool):
    """``pxr.Sdf`` from the ``pxr`` this process already has, or ``None``.

    A process that will start Isaac Sim must not import another ``pxr`` first (the pip ``usd-core`` build and Isaac
    Sim's kit build cannot both be loaded: the kit then fails with "module 'omni.usd' has no attribute 'get_context'"),
    so by default this only uses a ``pxr`` that is already imported, which is the case once a USD backend is up.
    """
    import sys

    if "pxr" not in sys.modules and not allow_import:
        return None
    try:
        from pxr import Sdf, Usd  # noqa: F401  # importing Usd registers the .usd / .usdc file formats Sdf opens
    except ImportError:
        return None
    return Sdf


def usd_dependencies(path: str, *, allow_import: bool = False) -> UsdDependencies | None:
    """Every file ``path`` composes from and whether it exists; ``None`` when no ``pxr`` is available.

    ``allow_import=True`` imports ``pxr`` when the process has not; leave it off in a process that may start Isaac Sim.
    """
    Sdf = _pxr_sdf(allow_import)
    if Sdf is None:
        return None
    root = os.path.abspath(path)
    deps = UsdDependencies(path=path)
    seen: set[str] = set()
    stack = [root]
    while stack:
        layer_path = stack.pop()
        if layer_path in seen:
            continue
        seen.add(layer_path)
        if not os.path.exists(layer_path):
            if layer_path != root:
                deps.unresolved.append(layer_path)
            continue
        try:
            layer = Sdf.Layer.FindOrOpen(layer_path)
        except Exception:  # pxr raises on a format it has no plugin for; the file is there but is not a layer
            layer = None
        if layer is None:  # exists but is not a layer this USD can read
            deps.unresolved.append(layer_path)
            continue
        base = os.path.dirname(layer_path)
        for ref in _layer_dependencies(layer):
            target = ref if os.path.isabs(ref) else os.path.normpath(os.path.join(base, ref))
            if os.path.exists(target):
                if target not in deps.resolved:
                    deps.resolved.append(target)
                if target.lower().endswith(USD_SUFFIXES):
                    stack.append(target)
            elif target not in deps.unresolved:
                deps.unresolved.append(target)
    return deps


def fetch_usd_dependencies(path: str, *, fetch=None, allow_import: bool = False) -> UsdDependencies | None:
    """Resolve ``path``'s composition, fetching each missing file with ``fetch`` (the Hub downloader by default).

    Fetching one layer can reveal the layers it composes from, so the walk repeats until nothing new is missing.
    Returns the final state (``None`` without ``pxr``); a file ``fetch`` could not produce stays in ``unresolved``.
    """
    if fetch is None:
        from metasim.utils.hf_util import check_and_download_single as fetch
    tried: set[str] = set()
    while True:
        deps = usd_dependencies(path, allow_import=allow_import)
        if deps is None:
            return None
        todo = [p for p in deps.unresolved if p not in tried]
        if not todo:
            return deps
        for missing in todo:
            tried.add(missing)
            try:
                fetch(missing)
            except Exception:  # the Hub has no such file: it stays unresolved and the caller reports it
                continue
