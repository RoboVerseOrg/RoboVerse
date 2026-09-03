"""Public API surface of MetaSim as data, so changes to it are deliberate.

``collect_api()`` walks the modules in ``PUBLIC_MODULES`` and records every public function and
class (with its public methods and dataclass fields) together with its signature. The result is
stored in ``metasim/test/api_snapshot.json``; ``metasim/test/test_public_api_general.py`` compares
the live surface against it and fails on a removed or re-signatured symbol.

Update the snapshot on purpose with ``python -m metasim api-snapshot --update`` and mention the
change in the CHANGELOG (``Changed`` / ``Removed`` / ``Deprecated``). Additions are reported, not
rejected: adding API is cheap, breaking it is not.

A name is public when it does not start with ``_`` and is defined in the module itself (re-exports
are attributed to the module that defines them).
"""

from __future__ import annotations

import dataclasses
import importlib
import inspect
import json
from pathlib import Path
from typing import Any

#: Modules whose public names form the supported API. Keep this list short and honest: only what
#: downstream code (roboverse_pack, roboverse_learn, users' task packs) is meant to import.
PUBLIC_MODULES: tuple[str, ...] = (
    "metasim",
    "metasim.constants",
    "metasim.types",
    "metasim.scenario.scenario",
    "metasim.scenario.objects",
    "metasim.scenario.robot",
    "metasim.scenario.cameras",
    "metasim.scenario.lights",
    "metasim.scenario.grounds",
    "metasim.scenario.render",
    "metasim.scenario.simulator_params",
    "metasim.sim.base",
    "metasim.sim.hybrid",
    "metasim.sim.parallel",
    "metasim.sim.sim_context",
    "metasim.sim._versions",
    "metasim.task.base",
    "metasim.task.rl_task",
    "metasim.task.registry",
    "metasim.queries.base",
    "metasim.utils.state",
    "metasim.utils.replay",
    "metasim.utils.setup_util",
    "metasim.utils.log",
    "metasim.utils.configclass",
)

SNAPSHOT_PATH = Path(__file__).resolve().parent.parent / "test" / "api_snapshot.json"


def _signature(obj: Any) -> str:
    try:
        return str(inspect.signature(obj))
    except (TypeError, ValueError):
        return "(...)"


def _class_entry(cls: type) -> dict[str, Any]:
    methods: dict[str, str] = {}
    for name, member in cls.__dict__.items():
        if name.startswith("_") and name != "__init__":
            continue
        if isinstance(member, (staticmethod, classmethod)):
            member = member.__func__
        if isinstance(member, property):
            methods[name] = "<property>"
        elif inspect.isfunction(member):
            methods[name] = _signature(member)
    entry: dict[str, Any] = {
        "kind": "class",
        "bases": [b.__name__ for b in cls.__bases__ if b is not object],
        "methods": dict(sorted(methods.items())),
    }
    if dataclasses.is_dataclass(cls):
        entry["fields"] = [f.name for f in dataclasses.fields(cls)]
    return entry


def collect_api(modules: tuple[str, ...] = PUBLIC_MODULES) -> dict[str, dict[str, Any]]:
    """``{module: {name: entry}}`` for every public function/class defined in ``modules``."""
    surface: dict[str, dict[str, Any]] = {}
    for modname in modules:
        module = importlib.import_module(modname)
        names = getattr(module, "__all__", None) or [n for n in vars(module) if not n.startswith("_")]
        entries: dict[str, Any] = {}
        for name in sorted(names):
            obj = getattr(module, name, None)
            if getattr(obj, "__module__", None) != modname:
                continue
            if inspect.isclass(obj):
                entries[name] = _class_entry(obj)
            elif inspect.isfunction(obj):
                entries[name] = {"kind": "function", "signature": _signature(obj)}
        surface[modname] = entries
    return surface


def diff_api(old: dict[str, dict[str, Any]], new: dict[str, dict[str, Any]]) -> tuple[list[str], list[str]]:
    """``(breaking, additions)`` as human-readable lines.

    Breaking = a module, symbol, method or dataclass field that disappeared, or whose signature changed.
    """
    breaking: list[str] = []
    additions: list[str] = []
    for modname, old_entries in old.items():
        if modname not in new:
            breaking.append(f"module removed: {modname}")
            continue
        new_entries = new[modname]
        for name, old_e in old_entries.items():
            new_e = new_entries.get(name)
            if new_e is None:
                breaking.append(f"removed: {modname}.{name}")
                continue
            if old_e["kind"] != new_e["kind"]:
                breaking.append(f"kind changed: {modname}.{name} {old_e['kind']} -> {new_e['kind']}")
                continue
            if old_e["kind"] == "function":
                if old_e["signature"] != new_e["signature"]:
                    breaking.append(f"signature changed: {modname}.{name}{old_e['signature']} -> {new_e['signature']}")
                continue
            for meth, sig in old_e["methods"].items():
                new_sig = new_e["methods"].get(meth)
                if new_sig is None:
                    breaking.append(f"method removed: {modname}.{name}.{meth}")
                elif new_sig != sig:
                    breaking.append(f"signature changed: {modname}.{name}.{meth}{sig} -> {new_sig}")
            for meth in new_e["methods"]:
                if meth not in old_e["methods"]:
                    additions.append(f"method added: {modname}.{name}.{meth}")
            for field in old_e.get("fields", []):
                if field not in new_e.get("fields", []):
                    breaking.append(f"field removed: {modname}.{name}.{field}")
            for field in new_e.get("fields", []):
                if field not in old_e.get("fields", []):
                    additions.append(f"field added: {modname}.{name}.{field}")
        for name in new_entries:
            if name not in old_entries:
                additions.append(f"added: {modname}.{name}")
    for modname in new:
        if modname not in old:
            additions.append(f"module added: {modname}")
    return breaking, additions


def load_snapshot(path: Path = SNAPSHOT_PATH) -> dict[str, dict[str, Any]]:
    """The stored surface (see ``SNAPSHOT_PATH``)."""
    return json.loads(path.read_text())


def write_snapshot(surface: dict[str, dict[str, Any]], path: Path = SNAPSHOT_PATH) -> None:
    """Store ``surface`` as sorted, indented JSON so diffs stay reviewable."""
    path.write_text(json.dumps(surface, indent=1, sort_keys=True) + "\n")
