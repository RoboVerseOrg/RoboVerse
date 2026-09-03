"""Static task index: find ``@register_task("name")`` decorators without importing the modules.

``list_tasks()`` and ``get_task_class()`` used to import *every* module of every task package
(``pkgutil.walk_packages`` + ``import_module``). With RoboVerse that is ~500 modules, two of them
20k-line generated files, plus import-time side effects (asset downloads, optional-dependency
warnings) — about 7 s before the first task can be looked up, and a failure blast radius that grows
with every task family.

This module scans the same packages with :mod:`ast` instead: every ``register_task(...)`` decorator
whose arguments are string literals is recorded as ``name -> module``. Lookups then import only the
one module that registers the requested task. Decorators with non-literal arguments are reported so
the registry can fall back to the full import for those packages.
"""

from __future__ import annotations

import ast
import json
import os
import tempfile
from dataclasses import dataclass, field
from importlib.util import find_spec


@dataclass
class StaticIndex:
    """Result of :func:`build_static_index`."""

    names: dict[str, str] = field(default_factory=dict)
    """lower-cased task name -> fully qualified module that registers it."""
    dynamic_modules: set[str] = field(default_factory=set)
    """Modules with a ``register_task`` call whose arguments are not all string literals."""
    scanned_modules: int = 0
    parse_failures: dict[str, str] = field(default_factory=dict)


def _decorator_names(node: ast.AST) -> list[str] | None:
    """String-literal arguments of a ``register_task(...)`` call, or None if not one / not literal."""
    if not isinstance(node, ast.Call):
        return None
    func = node.func
    fname = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else None
    if fname != "register_task":
        return None
    names = []
    for arg in node.args:
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            names.append(arg.value)
        else:
            return []  # dynamic: caller must import the module to learn the names
    if node.keywords:
        return []
    return names


def _parse_registrations(path: str) -> tuple[list[str], bool]:
    """(literal task names, has_dynamic_registration) for one source file.

    Every ``register_task(...)`` call is inspected — decorators and plain calls such as
    ``register_task(f"family.{n}")(cls)`` in a loop — so a module that registers names the parser
    cannot see is flagged dynamic and imported when the full name list is needed.
    """
    with open(path, encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=path)
    names: list[str] = []
    dynamic = False
    for node in ast.walk(tree):
        found = _decorator_names(node)
        if found is None:
            continue
        if not found:
            dynamic = True
        names.extend(found)
    return names, dynamic


def default_cache_path() -> str:
    """Per-file index cache (JSON). ``$METASIM_CACHE_DIR`` overrides the temp-dir default."""
    return os.path.join(
        os.environ.get("METASIM_CACHE_DIR", os.path.join(tempfile.gettempdir(), "metasim_cache")), "task_index.json"
    )


def _load_cache(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) and data.get("version") == 1 else {}
    except (OSError, ValueError):
        return {}


def _save_cache(path: str, entries: dict) -> None:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.tmp-{os.getpid()}"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"version": 1, "files": entries}, f)
        os.replace(tmp, path)
    except OSError:  # a read-only or full cache location must not break lookups
        pass


def _scan_source(module_name: str, path: str, index: StaticIndex, cache: dict | None = None) -> None:
    try:
        st = os.stat(path)
        stamp = [st.st_mtime, st.st_size]
        entry = cache.get(path) if cache is not None else None
        if entry is not None and entry.get("stamp") == stamp:
            names, dynamic = entry["names"], entry["dynamic"]
        else:
            names, dynamic = _parse_registrations(path)
            if cache is not None:
                cache[path] = {"stamp": stamp, "names": names, "dynamic": dynamic}
    except (OSError, SyntaxError, ValueError) as exc:
        index.parse_failures[module_name] = f"{type(exc).__name__}: {exc}"
        return
    index.scanned_modules += 1
    if dynamic:
        index.dynamic_modules.add(module_name)
    for raw in names:
        key = raw.strip().lower()
        if key:
            index.names.setdefault(key, module_name)


def _module_source_path(module_name: str) -> str | None:
    try:
        spec = find_spec(module_name)
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.origin or not spec.origin.endswith(".py"):
        return None
    return spec.origin


def _iter_package_sources(pkg_name: str, pkg_paths) -> list[tuple[str, str]]:
    """(module_name, source_path) for every ``.py`` under the package directories, without importing.

    ``pkgutil.walk_packages`` imports each sub-package to descend into it, which runs every
    ``__init__`` side effect (some task families import all their modules there) — the very cost this
    index exists to avoid. Walking the directories is enough because a module's name is its path.
    """
    out: list[tuple[str, str]] = []
    for root_dir in pkg_paths:
        root_dir = os.path.abspath(root_dir)
        for dirpath, dirnames, filenames in os.walk(root_dir):
            dirnames[:] = sorted(d for d in dirnames if d != "__pycache__" and not d.startswith("."))
            rel = os.path.relpath(dirpath, root_dir)
            prefix = pkg_name if rel == "." else pkg_name + "." + rel.replace(os.sep, ".")
            for fname in sorted(filenames):
                if not fname.endswith(".py"):
                    continue
                mod = prefix if fname == "__init__.py" else f"{prefix}.{fname[:-3]}"
                out.append((mod, os.path.join(dirpath, fname)))
    return out


def build_static_index(
    package_names: list[str], *, local_modules: list[str] = (), cache_path: str | None = None
) -> StaticIndex:
    """Scan ``package_names`` (importable package roots) and ``local_modules`` for task registrations.

    Nothing is imported: package locations come from ``importlib.util.find_spec`` and every source
    file is parsed with ``ast``. Parse results are cached per file (keyed by mtime + size) in
    ``cache_path`` (default :func:`default_cache_path`; ``""`` disables the cache).
    """
    index = StaticIndex()
    cache_path = default_cache_path() if cache_path is None else cache_path
    cache: dict | None = _load_cache(cache_path).get("files", {}) if cache_path else None
    seen_paths: set[str] = set()
    for pkg_name in package_names:
        if "." not in pkg_name and pkg_name in local_modules:
            continue
        try:
            spec = find_spec(pkg_name)
        except (ImportError, ValueError) as exc:
            index.parse_failures[pkg_name] = f"{type(exc).__name__}: {exc}"
            continue
        if spec is None:
            index.parse_failures[pkg_name] = "ModuleNotFoundError: no such package"
            continue
        if spec.submodule_search_locations:
            for module_name, src in _iter_package_sources(pkg_name, list(spec.submodule_search_locations)):
                if src in seen_paths:
                    continue
                seen_paths.add(src)
                _scan_source(module_name, src, index, cache)
        elif spec.origin and spec.origin.endswith(".py"):
            _scan_source(pkg_name, spec.origin, index, cache)
    for mod in local_modules:
        src = _module_source_path(mod)
        if src is None and os.path.isfile(mod + ".py"):
            src = mod + ".py"
        if src:
            _scan_source(mod, src, index, cache)
    if cache is not None:
        _save_cache(cache_path, cache)
    return index
