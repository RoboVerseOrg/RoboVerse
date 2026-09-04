from __future__ import annotations

import os
import pkgutil
import sys
from importlib import import_module
from importlib.util import find_spec

from loguru import logger as log

from metasim.task._static_index import StaticIndex, build_static_index
from metasim.task.base import BaseTaskEnv
from metasim.utils.package_discovery import get_package_candidates

# Global registry mapping lowercase names to task wrapper classes
TASK_REGISTRY = {}

# Failures encountered during task discovery: module_name -> short error string.
# Surfaced in get_task_class() KeyError so users see import problems right when
# they look up a missing task, rather than having to enable DEBUG logging.
_DISCOVERY_FAILURES: dict[str, str] = {}
#: task name -> modules that registered *different* classes under it. The first class stays in the
#: registry and the module importing keeps going (its other names register); ``get_task_class`` refuses
#: the name on every discovery path.
_NAME_CONFLICTS: dict[str, list[str]] = {}
#: names the index lists under several modules that a lookup has already settled (same class from every
#: claimant): served from the registry until a conflict is recorded or the index is rebuilt.
_RESOLVED: set[str] = set()
_TASK_PACKAGES_CACHE: tuple[tuple, list[str]] | None = None

# Static (AST) index of ``@register_task`` names -> module, built once per process; see
# ``metasim/task/_static_index.py``. ``METASIM_TASK_DISCOVERY=eager`` restores the old
# import-everything behaviour (useful when debugging import-time side effects).
_STATIC_INDEX: StaticIndex | None = None
_STATIC_INDEX_KEY: tuple[str, ...] | None = None
"""Package list the cached index was built for; a different configuration (env vars, cwd) rebuilds it."""
_EAGER_DONE = False


def register_task(*names):
    """Class decorator to register a task under one or more names.

    Usage:
        @register_task("humanoid.walk", "walk")
        class WalkTask(...):
            ...

    Registering the same class again (an alias module) or the same class definition again (a module
    re-executed: reload, a notebook cell, a file run as a script) is a no-op / redefinition. A
    *different* class under a name already taken is a conflict: the first class stays, a warning is
    logged, and ``get_task_class`` raises ``ValueError`` for that name until one of them is renamed.
    """
    if not names:
        raise ValueError("At least one name must be provided to register_task().")

    def _decorator(cls):
        if not issubclass(cls, BaseTaskEnv):
            log.warning(f"Register class {cls!r} is not a subclass of BaseTaskEnv")
        for raw_name in names:
            key = raw_name.strip().lower()
            if not key:
                raise ValueError("Task name cannot be empty or whitespace only.")
            existing = TASK_REGISTRY.get(key)
            if existing is None or _same_definition(existing, cls):
                # the same source re-executed (reload, a notebook cell, a file imported under two module
                # names, a task file run as a script) redefines its own class
                TASK_REGISTRY[key] = cls
                continue
            conflict = _NAME_CONFLICTS.setdefault(key, [existing.__module__])
            if cls.__module__ not in conflict:
                conflict.append(cls.__module__)
            _RESOLVED.discard(key)
            log.warning(
                f"Task name '{key}' is registered to different classes by {existing.__module__} and "
                f"{cls.__module__}; lookups of that name are refused until one is renamed"
            )
        return cls

    return _decorator


def _same_definition(existing: type, cls: type) -> bool:
    """Same class object, or the same qualified name defined in the same source file."""
    if existing is cls:
        return True
    if existing.__qualname__ != cls.__qualname__:
        return False
    if existing.__module__ == cls.__module__:
        return True  # the module re-executed (a notebook ``__main__`` has no file to compare)
    files = []
    for klass in (existing, cls):
        module = sys.modules.get(klass.__module__)
        path = getattr(module, "__file__", None)
        files.append(os.path.realpath(path) if path else None)
    return files[0] is not None and files[0] == files[1]


def _task_packages() -> list[str]:
    """Configured task packages (plus ``_*task.py`` modules in the CWD), in discovery order.

    A CWD module that lives inside a configured package (running from within the package directory)
    is reached by the package walk under its dotted name and is not listed a second time as a bare
    module: one file is one registration on both discovery paths.
    """
    global _TASK_PACKAGES_CACHE
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)
    local_files = tuple(sorted(f for f in os.listdir(cwd) if f.endswith("task.py") and f.startswith("_")))
    env = tuple(sorted((k, v) for k, v in os.environ.items() if k.startswith("METASIM_")))
    cache_key = (cwd, local_files, env)
    if _TASK_PACKAGES_CACHE is not None and _TASK_PACKAGES_CACHE[0] == cache_key:
        return list(_TASK_PACKAGES_CACHE[1])  # entry points and config files do not change within a process
    defaults = ["metasim.example.example_pack.tasks"]
    packages = get_package_candidates("tasks", defaults=defaults, cwd=cwd)
    roots = []
    for pkg in packages:
        try:
            spec = find_spec(pkg)  # imports the parent package: its __init__ may raise anything
        except Exception as exc:
            _DISCOVERY_FAILURES.setdefault(pkg, f"{type(exc).__name__}: {exc}")
            continue
        if spec is not None and spec.submodule_search_locations:
            roots.extend(os.path.realpath(p) for p in spec.submodule_search_locations)
    local_task_modules = []
    for fname in local_files:
        real = os.path.realpath(os.path.join(cwd, fname))
        if any(_reached_by_package_walk(real, root) for root in roots):
            continue
        local_task_modules.append(fname[:-3])
    result = get_package_candidates("tasks", defaults=defaults, local_modules=local_task_modules, cwd=cwd)
    _TASK_PACKAGES_CACHE = (cache_key, list(result))
    return result


def _reached_by_package_walk(file_path: str, root: str) -> bool:
    """True when ``pkgutil.walk_packages`` from ``root`` imports ``file_path``.

    Every directory on the way must be a package; a file under a plain subdirectory is only reachable
    as a working-directory module.
    """
    if not file_path.startswith(root + os.sep):
        return False
    directory = os.path.dirname(file_path)
    while directory != root:
        if not os.path.isfile(os.path.join(directory, "__init__.py")):
            return False
        directory = os.path.dirname(directory)
    return True


def _eager_discovery_requested() -> bool:
    return os.environ.get("METASIM_TASK_DISCOVERY", "").strip().lower() == "eager"


def _static_index() -> StaticIndex:
    """Build (once) the AST index of task names -> module for the configured packages."""
    global _STATIC_INDEX, _STATIC_INDEX_KEY
    packages = _task_packages()
    key = tuple(packages)
    stale = _STATIC_INDEX is not None and _STATIC_INDEX_KEY == key and _packages_now_importable(_STATIC_INDEX, packages)
    if _STATIC_INDEX is None or _STATIC_INDEX_KEY != key or stale:
        local = [m for m in packages if "." not in m]  # cwd ``_*task.py`` modules come through as bare names
        _STATIC_INDEX = build_static_index(packages, local_modules=local)
        _STATIC_INDEX_KEY = key
        _RESOLVED.clear()
        for mod, err in _STATIC_INDEX.parse_failures.items():
            _DISCOVERY_FAILURES.setdefault(mod, err)
            log.warning(f"Task discovery: could not scan '{mod}': {err}")
        for key, modules in sorted(_STATIC_INDEX.collisions.items()):
            log.warning(
                f"Task name '{key}' is registered by several modules: {', '.join(modules)} "
                "(refused on lookup unless they register the same class)"
            )
    return _STATIC_INDEX


def _claimants(index: StaticIndex, key: str) -> list[str]:
    """Every module the index saw registering ``key`` (the one it kept first)."""
    modules = list(index.collisions.get(key, ()))
    first = index.names.get(key)
    if first is not None and first not in modules:
        modules.insert(0, first)
    return modules


def _packages_now_importable(index: StaticIndex, packages: list[str]) -> bool:
    """True when a *package* the index could not scan (its import failed) imports now, so a rebuild sees it.

    A module that failed to parse is not retried here: ``find_spec`` does not compile, so it would
    look fixed and force a rebuild on every call.
    """
    for pkg in index.parse_failures:
        if pkg not in packages:
            continue
        try:
            if find_spec(pkg) is not None:
                _DISCOVERY_FAILURES.pop(pkg, None)
                return True
        except Exception:
            continue
    return False


def _raise_if_conflicting(key: str, claimants: list[str]) -> None:
    modules = _NAME_CONFLICTS.get(key)
    if modules:
        raise ValueError(
            f"Task name '{key}' is registered to different classes by {', '.join(modules)}; rename one of them."
        )
    failed = [m for m in claimants if m in _DISCOVERY_FAILURES]
    if failed and len(claimants) > 1 and key in TASK_REGISTRY:
        # the other claimant could not be imported here, so whether it registers the same class is unknown
        detail = "; ".join(f"{m}: {_DISCOVERY_FAILURES[m]}" for m in failed)
        raise ValueError(
            f"Task name '{key}' is also claimed by {', '.join(failed)}, which failed to import ({detail}); "
            "the name is refused until that module imports (the next lookup retries) or is renamed."
        )


def _import_registering_module(module_name: str) -> None:
    """Import one task module so its ``@register_task`` decorators run; record failures."""
    try:
        import_module(module_name)
        _DISCOVERY_FAILURES.pop(module_name, None)  # fixed since the last attempt
    except Exception as e:
        err_str = f"{type(e).__name__}: {e}"
        _DISCOVERY_FAILURES[module_name] = err_str
        log.warning(f"Task discovery: skip module '{module_name}': {err_str}")


def _discover_task_modules() -> None:
    """Import configured task packages so @register_task decorators run (eager fallback)."""
    global _EAGER_DONE
    if _EAGER_DONE and TASK_REGISTRY:  # an emptied registry (tests, reloads) re-runs discovery
        return
    _EAGER_DONE = True
    packages_to_scan = _task_packages()

    for pkg_name in packages_to_scan:
        try:
            # Import the root package
            pkg = import_module(pkg_name)
        except Exception as e:
            err_str = f"{type(e).__name__}: {e}"
            _DISCOVERY_FAILURES[pkg_name] = err_str
            log.error(f"Task discovery: failed to import package '{pkg_name}': {err_str}")
            continue

        try:
            pkg_path = getattr(pkg, "__path__", None)
            if pkg_path is None:
                continue

            # Scan and import all submodules
            for _finder, module_name, _is_pkg in pkgutil.walk_packages(pkg_path, prefix=pkg.__name__ + "."):
                try:
                    import_module(module_name)
                    _DISCOVERY_FAILURES.pop(module_name, None)
                except Exception as e:
                    err_str = f"{type(e).__name__}: {e}"
                    _DISCOVERY_FAILURES[module_name] = err_str
                    # Log at WARNING (not DEBUG): a discovery failure makes
                    # tasks disappear silently from the registry — the user
                    # then can't run the affected task with no clue why. The
                    # message is also surfaced inside any KeyError raised by
                    # ``get_task_class`` so users hitting an unknown-task
                    # error see related discovery failures.
                    log.warning(f"Task discovery: skip module '{module_name}': {err_str}")
        except Exception as e:
            log.error(f"Task discovery: error scanning package '{pkg_name}': {e}")


def get_task_class(name: str) -> type[BaseTaskEnv]:
    """Return the task wrapper class registered under the given name.

    Name lookup is case-insensitive. Only the modules that register ``name`` are imported (found
    through the static index): every module the index saw claiming the name, so the answer does not
    depend on which of them an earlier lookup happened to import. The eager import of every task
    module happens only when the name is unknown to the index or ``METASIM_TASK_DISCOVERY=eager`` is
    set. A name registered by a module the index cannot read (non-literal ``register_task`` arguments)
    is checked when that module is imported (``list_tasks`` or the eager fallback). A name already in
    the registry (a class the caller registered itself, or one an earlier lookup imported) is answered
    directly unless a conflict is recorded or the index lists another claimant that no lookup has
    settled yet. The eager path consults the index for claimants too (an AST scan, no imports).

    Raises:
        ValueError: ``name`` is registered to different classes by several modules, or another module
            the index saw claiming it failed to import, so the class cannot be established. Rename one;
            a recorded different-classes conflict lasts for the process.
        KeyError: no module registers ``name`` (the message lists the registered names and any task
            modules that failed to import).
    """
    key = name.strip().lower()
    known_collision = _STATIC_INDEX is not None and key in _STATIC_INDEX.collisions and key not in _RESOLVED
    if key in TASK_REGISTRY and key not in _NAME_CONFLICTS and not known_collision:
        return TASK_REGISTRY[key]
    index = _static_index()
    claimants = _claimants(index, key)  # what the index saw, on either path (it imports nothing)
    for module_name in claimants:
        _import_registering_module(module_name)  # a loaded module is not re-executed; a failed one is retried
    if _eager_discovery_requested() or key not in TASK_REGISTRY:
        # eager mode, or dynamic registrations / a name the index could not see: import everything
        _discover_task_modules()
    _raise_if_conflicting(key, claimants)
    _RESOLVED.add(key)
    try:
        return TASK_REGISTRY[key]
    except KeyError as exc:
        available = ", ".join(sorted(TASK_REGISTRY.keys())) or "<none>"
        msg = f"Unknown task '{name}'. Registered tasks: {available}"
        if _DISCOVERY_FAILURES:
            # Surface import errors so users can see if the task they want
            # failed to register because its module raised during import.
            failures = "\n  ".join(f"{m}: {e}" for m, e in sorted(_DISCOVERY_FAILURES.items()))
            msg += f"\n\n{len(_DISCOVERY_FAILURES)} task module(s) failed to import during discovery:\n  {failures}"
        raise KeyError(msg) from exc


def list_tasks():
    """List all task names (sorted) without importing the task modules.

    Names come from the static index plus anything registered so far; modules whose
    ``register_task`` arguments are not string literals are imported to learn their names.
    """
    if _eager_discovery_requested():
        _discover_task_modules()
        return sorted(TASK_REGISTRY.keys())
    index = _static_index()
    for module_name in sorted(index.dynamic_modules):
        _import_registering_module(module_name)
    return sorted(set(index.names) | set(TASK_REGISTRY.keys()))
