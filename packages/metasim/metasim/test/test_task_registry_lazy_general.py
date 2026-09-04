"""The task registry resolves names through a static (AST) index and imports one module per lookup.

A synthetic task package is written to ``tmp_path``: two modules with literal ``@register_task``
decorators, one that registers names in a loop (dynamic), and one whose import fails. Each module
appends to a marker file when imported, so the tests can assert *which* modules a call imported.
"""

from __future__ import annotations

import json
import os
import sys
import textwrap

import pytest

from metasim.task import _static_index, registry


@pytest.fixture
def task_pkg(tmp_path, monkeypatch):
    pkg = tmp_path / "lazypack"
    (pkg / "tasks").mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "tasks" / "__init__.py").write_text("", encoding="utf-8")
    marker = tmp_path / "imported.txt"
    common = textwrap.dedent(
        f"""
        from metasim.task.base import BaseTaskEnv
        from metasim.task.registry import register_task

        with open({str(marker)!r}, "a") as _f:
            _f.write(__name__ + "\\n")
        """
    )
    (pkg / "tasks" / "alpha.py").write_text(
        common
        + textwrap.dedent(
            """
            @register_task("lazy.alpha", "Alpha")
            class AlphaTask(BaseTaskEnv):
                pass
            """
        ),
        encoding="utf-8",
    )
    (pkg / "tasks" / "beta.py").write_text(
        common
        + textwrap.dedent(
            """
            @register_task("lazy.beta")
            class BetaTask(BaseTaskEnv):
                pass
            """
        ),
        encoding="utf-8",
    )
    (pkg / "tasks" / "looped.py").write_text(
        common
        + textwrap.dedent(
            """
            for _i in range(2):
                register_task(f"lazy.looped_{_i}")(type(f"Looped{_i}", (BaseTaskEnv,), {}))
            """
        ),
        encoding="utf-8",
    )
    (pkg / "tasks" / "broken.py").write_text(
        common
        + textwrap.dedent(
            """
            import module_that_does_not_exist_anywhere  # noqa: F401

            @register_task("lazy.broken")
            class BrokenTask(BaseTaskEnv):
                pass
            """
        ),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv("METASIM_TASK_PACKAGES", "lazypack.tasks")
    monkeypatch.setenv("METASIM_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("METASIM_TASK_DISCOVERY", raising=False)
    # fresh registry state for every test
    monkeypatch.setattr(registry, "TASK_REGISTRY", {})
    monkeypatch.setattr(registry, "_DISCOVERY_FAILURES", {})
    monkeypatch.setattr(registry, "_NAME_CONFLICTS", {})
    monkeypatch.setattr(registry, "_RESOLVED", set())
    monkeypatch.setattr(registry, "_TASK_PACKAGES_CACHE", None)
    monkeypatch.setattr(registry, "_STATIC_INDEX", None)
    monkeypatch.setattr(registry, "_EAGER_DONE", False)
    for name in list(sys.modules):
        if name.startswith("lazypack"):
            del sys.modules[name]

    def imported() -> list[str]:
        return marker.read_text(encoding="utf-8").split() if marker.exists() else []

    return imported


@pytest.mark.general
def test_get_task_class_imports_only_the_registering_module(task_pkg):
    cls = registry.get_task_class("Lazy.Alpha")
    assert cls.__name__ == "AlphaTask"
    assert task_pkg() == ["lazypack.tasks.alpha"], "beta / looped / broken must not have been imported"
    assert registry._EAGER_DONE is False


@pytest.mark.general
def test_list_tasks_sees_static_names_without_importing_them(task_pkg):
    names = registry.list_tasks()
    assert {"lazy.alpha", "alpha", "lazy.beta", "lazy.looped_0", "lazy.looped_1", "lazy.broken"} <= set(names)
    # only the dynamic module had to be imported to learn its names
    assert task_pkg() == ["lazypack.tasks.looped"]


@pytest.mark.general
def test_unknown_name_falls_back_to_eager_discovery_and_reports_failures(task_pkg):
    with pytest.raises(KeyError) as excinfo:
        registry.get_task_class("lazy.missing")
    msg = str(excinfo.value)
    assert "lazypack.tasks.broken" in msg and "module_that_does_not_exist_anywhere" in msg
    assert registry._EAGER_DONE is True


@pytest.mark.general
def test_broken_module_error_is_surfaced_on_lookup(task_pkg):
    with pytest.raises(KeyError, match="module_that_does_not_exist_anywhere"):
        registry.get_task_class("lazy.broken")


@pytest.mark.general
def test_eager_env_var_restores_import_everything(task_pkg, monkeypatch):
    monkeypatch.setenv("METASIM_TASK_DISCOVERY", "eager")
    registry.get_task_class("lazy.beta")
    assert set(task_pkg()) == {
        "lazypack.tasks.alpha",
        "lazypack.tasks.beta",
        "lazypack.tasks.looped",
        "lazypack.tasks.broken",
    }


@pytest.mark.general
def test_index_cache_is_written_and_reused(task_pkg, tmp_path):
    registry.list_tasks()
    cache_file = tmp_path / "cache" / "task_index.json"
    assert cache_file.is_file()
    data = json.loads(cache_file.read_text(encoding="utf-8"))
    alpha_path = str(tmp_path / "lazypack" / "tasks" / "alpha.py")
    assert data["files"][alpha_path]["names"] == ["lazy.alpha", "Alpha"]
    assert data["files"][str(tmp_path / "lazypack" / "tasks" / "looped.py")]["dynamic"] is True
    # a second build with an untouched tree reads the cache: parse nothing, same names
    parsed_before = _static_index._parse_registrations
    calls = []

    def _counting(path):
        calls.append(path)
        return parsed_before(path)

    _static_index._parse_registrations = _counting
    try:
        idx = _static_index.build_static_index(["lazypack.tasks"])
    finally:
        _static_index._parse_registrations = parsed_before
    assert calls == []
    assert idx.names["lazy.alpha"] == "lazypack.tasks.alpha"
    # editing a file invalidates only that file
    beta = tmp_path / "lazypack" / "tasks" / "beta.py"
    beta.write_text(beta.read_text(encoding="utf-8").replace('"lazy.beta"', '"lazy.beta2"'), encoding="utf-8")
    os.utime(beta, (os.stat(beta).st_atime + 5, os.stat(beta).st_mtime + 5))
    idx = _static_index.build_static_index(["lazypack.tasks"])
    assert "lazy.beta2" in idx.names and "lazy.beta" not in idx.names


def _write_task(tmp_path, module, body):
    (tmp_path / "lazypack" / "tasks" / f"{module}.py").write_text(
        textwrap.dedent(
            """
            from metasim.task.base import BaseTaskEnv
            from metasim.task.registry import register_task
            """
        )
        + textwrap.dedent(body),
        encoding="utf-8",
    )


@pytest.mark.general
def test_a_task_name_claimed_by_two_classes_is_refused_on_both_discovery_paths(task_pkg, tmp_path, monkeypatch):
    """The static index used to keep whichever module was scanned first, and the eager path swallowed
    the second registration's error and returned the first class: the loser was silently unreachable.
    Now the index records the collision, the lookup imports every claimant and refuses the name when
    they register different classes, whichever path and whichever lookup order."""
    _write_task(
        tmp_path,
        "alpha_again",
        """
        @register_task("lazy.alpha")
        class AlphaAgainTask(BaseTaskEnv):
            pass
        """,
    )
    index = registry._static_index()
    assert index.collisions == {"lazy.alpha": ["lazypack.tasks.alpha", "lazypack.tasks.alpha_again"]}
    assert registry.get_task_class("Alpha").__name__ == "AlphaTask"  # alpha.py's other name resolves first...
    with pytest.raises(ValueError, match=r"different classes by lazypack\.tasks\.alpha, lazypack\.tasks\.alpha_again"):
        registry.get_task_class("lazy.alpha")  # ...and does not make the colliding name resolve to it
    assert "lazy.alpha" in registry.list_tasks()
    assert registry.get_task_class("lazy.beta").__name__ == "BetaTask"

    monkeypatch.setenv("METASIM_TASK_DISCOVERY", "eager")
    monkeypatch.setattr(registry, "TASK_REGISTRY", {})
    monkeypatch.setattr(registry, "_NAME_CONFLICTS", {})
    monkeypatch.setattr(registry, "_EAGER_DONE", False)
    for name in list(sys.modules):
        if name.startswith("lazypack"):
            del sys.modules[name]
    with pytest.raises(ValueError, match="different classes"):
        registry.get_task_class("lazy.alpha")


@pytest.mark.general
def test_re_registering_the_same_class_from_an_alias_module_is_not_a_conflict(task_pkg, tmp_path):
    """``register_task`` allows the same class under the same name twice; an alias module that
    re-exports ``AlphaTask`` is a collision to the index but resolves after both are imported."""
    _write_task(
        tmp_path,
        "aliases",
        """
        from lazypack.tasks.alpha import AlphaTask

        register_task("lazy.alpha")(AlphaTask)
        """,
    )
    assert "lazy.alpha" in registry._static_index().collisions
    assert registry.get_task_class("lazy.alpha").__name__ == "AlphaTask"
    assert "lazypack.tasks.alpha" in sys.modules and "lazypack.tasks.aliases" in sys.modules, "both claimants imported"


@pytest.mark.general
def test_a_file_reachable_as_package_module_and_cwd_module_is_one_registration(task_pkg, tmp_path, monkeypatch):
    """Running from inside the package directory makes ``alpha.py`` both ``lazypack.tasks.alpha`` and the
    bare cwd module ``alpha``; that is the same file, not two claimants."""
    monkeypatch.chdir(tmp_path / "lazypack" / "tasks")
    monkeypatch.syspath_prepend(str(tmp_path / "lazypack" / "tasks"))
    index = _static_index.build_static_index(["lazypack.tasks", "alpha"], local_modules=["alpha"], cache_path="")
    assert index.collisions == {} and index.names["lazy.alpha"] == "lazypack.tasks.alpha"


@pytest.mark.general
def test_only_the_ambiguous_name_is_refused_and_the_losing_module_keeps_its_other_names(task_pkg, tmp_path):
    """``register_task`` no longer aborts the second module at its colliding decorator: the first class
    stays, the module finishes importing, and its other names register. A re-import of that module
    (its earlier import never failed, so this is a no-op) does not turn into a self-conflict either."""
    _write_task(
        tmp_path,
        "alpha_again",
        """
        @register_task("lazy.gamma")
        class GammaTask(BaseTaskEnv):
            pass

        @register_task("lazy.alpha", "lazy.epsilon")
        class AlphaAgainTask(BaseTaskEnv):
            pass

        @register_task("lazy.delta")
        class DeltaTask(BaseTaskEnv):
            pass
        """,
    )
    with pytest.raises(ValueError, match="different classes"):
        registry.get_task_class("lazy.alpha")
    assert registry.get_task_class("lazy.delta").__name__ == "DeltaTask"
    assert registry.get_task_class("lazy.epsilon").__name__ == "AlphaAgainTask"
    assert registry.get_task_class("lazy.gamma").__name__ == "GammaTask"
    assert registry.get_task_class("lazy.gamma").__name__ == "GammaTask"  # the fast path answers now
    assert set(registry._NAME_CONFLICTS) == {"lazy.alpha"}


@pytest.mark.general
def test_reloading_a_module_redefines_its_own_task_instead_of_conflicting(task_pkg):
    import importlib

    first = registry.get_task_class("lazy.beta")
    importlib.reload(sys.modules["lazypack.tasks.beta"])
    second = registry.get_task_class("lazy.beta")
    assert second is not first and second.__name__ == "BetaTask" and registry._NAME_CONFLICTS == {}


@pytest.mark.general
def test_a_claimant_that_fails_to_import_makes_the_name_refused_not_silently_resolved(task_pkg, tmp_path, monkeypatch):
    """Whether the broken claimant registers the same class cannot be established on this machine, so
    the name is refused with the import error, rather than answered differently per environment."""
    _write_task(
        tmp_path,
        "alpha_broken",
        """
        import module_that_does_not_exist_anywhere  # noqa: F401

        @register_task("lazy.alpha")
        class AlphaBrokenTask(BaseTaskEnv):
            pass
        """,
    )
    with pytest.raises(ValueError, match=r"also claimed by lazypack\.tasks\.alpha_broken, which failed to import"):
        registry.get_task_class("lazy.alpha")
    # the eager path refuses the same way (the claimants come from the index, which imports nothing)
    monkeypatch.setenv("METASIM_TASK_DISCOVERY", "eager")
    with pytest.raises(ValueError, match="which failed to import"):
        registry.get_task_class("lazy.alpha")
    # fixed file: the next lookup retries and the name resolves (same class re-registered)
    _write_task(
        tmp_path,
        "alpha_broken",
        """
        from lazypack.tasks.alpha import AlphaTask

        register_task("lazy.alpha")(AlphaTask)
        """,
    )
    monkeypatch.delenv("METASIM_TASK_DISCOVERY")
    import importlib

    importlib.invalidate_caches()
    assert registry.get_task_class("lazy.alpha").__name__ == "AlphaTask"


@pytest.mark.general
def test_a_runtime_registration_is_answered_unless_a_conflict_is_recorded(task_pkg, monkeypatch):
    """A class the caller registered itself is answered directly (no discovery, as before). Once a pack
    module registering the same name to another class has been imported, the name is a conflict."""
    from metasim.task.base import BaseTaskEnv

    @registry.register_task("lazy.beta")
    class MyBeta(BaseTaskEnv):
        pass

    assert registry.get_task_class("lazy.beta") is MyBeta
    registry._import_registering_module("lazypack.tasks.beta")
    with pytest.raises(ValueError, match="different classes"):
        registry.get_task_class("lazy.beta")


@pytest.mark.general
def test_the_same_source_under_two_module_names_redefines_not_conflicts(task_pkg, tmp_path):
    """A task file run as a script (``__main__``) or reached both as ``pkg.mod`` and as a cwd module
    registers the same qualified class from the same file: that is a redefinition, not a conflict."""
    import importlib.util

    first = registry.get_task_class("lazy.beta")
    spec = importlib.util.spec_from_file_location("__main__", tmp_path / "lazypack" / "tasks" / "beta.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["__main__"], saved_main = module, sys.modules["__main__"]
    try:
        spec.loader.exec_module(module)  # registers BetaTask again, from "__main__"
        assert registry.get_task_class("lazy.beta") is module.BetaTask is not first
    finally:
        sys.modules["__main__"] = saved_main
    assert registry._NAME_CONFLICTS == {}


@pytest.mark.general
def test_running_inside_the_package_directory_lists_a_task_file_once_on_the_eager_path(task_pkg, tmp_path, monkeypatch):
    """``_task_packages`` used to add ``_zeta_task`` as a bare cwd module next to ``lazypack.tasks._zeta_task``,
    so the eager path imported the file twice and (now that collisions are recorded) refused the name."""
    (tmp_path / "lazypack" / "tasks" / "_zeta_task.py").write_text(
        textwrap.dedent(
            """
            from metasim.task.base import BaseTaskEnv
            from metasim.task.registry import register_task

            @register_task("lazy.zeta")
            class ZetaTask(BaseTaskEnv):
                pass
            """
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path / "lazypack" / "tasks")
    assert "_zeta_task" not in registry._task_packages()
    monkeypatch.setenv("METASIM_TASK_DISCOVERY", "eager")
    assert registry.get_task_class("lazy.zeta").__name__ == "ZetaTask" and registry._NAME_CONFLICTS == {}


@pytest.mark.general
def test_two_classes_in_one_module_under_one_name_is_a_conflict_not_a_redefinition(task_pkg, tmp_path):
    """Only a re-executed module redefining the *same* class (same qualified name) is a redefinition; a
    copied decorator naming a second class in the same module is refused like any other conflict."""
    _write_task(
        tmp_path,
        "dupe",
        """
        @register_task("lazy.dupe")
        class First(BaseTaskEnv):
            pass

        @register_task("lazy.dupe")
        class Second(BaseTaskEnv):
            pass
        """,
    )
    with pytest.raises(ValueError, match=r"different classes by lazypack\.tasks\.dupe"):
        registry.get_task_class("lazy.dupe")


@pytest.mark.general
def test_a_cwd_task_file_under_a_plain_subdirectory_stays_a_cwd_module(task_pkg, tmp_path, monkeypatch):
    """``pkgutil.walk_packages`` does not enter a directory without ``__init__.py``, so a task file there is
    reachable on the eager path only as a working-directory module and must not be dropped."""
    scratch = tmp_path / "lazypack" / "tasks" / "scratch"
    scratch.mkdir()
    (scratch / "_eta_task.py").write_text(
        textwrap.dedent(
            """
            from metasim.task.base import BaseTaskEnv
            from metasim.task.registry import register_task

            @register_task("lazy.eta")
            class EtaTask(BaseTaskEnv):
                pass
            """
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(scratch)
    assert "_eta_task" in registry._task_packages()
    assert (
        registry.get_task_class("lazy.eta").__name__ == "EtaTask"
    )  # lazy: reached as lazypack.tasks.scratch._eta_task
    with pytest.raises(KeyError):
        registry.get_task_class("no.such.task")  # the eager fallback imports the same file as ``_eta_task``
    assert registry.get_task_class("lazy.eta").__name__ == "EtaTask" and registry._NAME_CONFLICTS == {}
    monkeypatch.setenv("METASIM_TASK_DISCOVERY", "eager")
    assert registry.get_task_class("lazy.eta").__name__ == "EtaTask"


@pytest.mark.general
def test_a_package_whose_init_raises_is_recorded_and_retried_once_it_imports(task_pkg, tmp_path, monkeypatch):
    init = tmp_path / "brokenpack" / "__init__.py"
    init.parent.mkdir()
    init.write_text("raise RuntimeError('gpu check failed')\n", encoding="utf-8")
    (tmp_path / "brokenpack" / "tasks.py").write_text(
        textwrap.dedent(
            """
            from metasim.task.base import BaseTaskEnv
            from metasim.task.registry import register_task

            @register_task("broken.task")
            class BrokenPackTask(BaseTaskEnv):
                pass
            """
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("METASIM_TASK_PACKAGES", "brokenpack.tasks:lazypack.tasks")
    monkeypatch.setattr(registry, "_STATIC_INDEX", None)
    assert registry.get_task_class("lazy.beta").__name__ == "BetaTask"
    assert "RuntimeError: gpu check failed" in registry._DISCOVERY_FAILURES["brokenpack.tasks"]
    with pytest.raises(KeyError, match="gpu check failed"):
        registry.get_task_class("broken.task")
    init.write_text("", encoding="utf-8")  # fixed in the same process
    sys.modules.pop("brokenpack", None)
    assert registry.get_task_class("broken.task").__name__ == "BrokenPackTask"
    assert "brokenpack.tasks" not in registry._DISCOVERY_FAILURES


@pytest.mark.general
def test_a_name_the_caller_registered_survives_a_broken_discovery_config(task_pkg, monkeypatch):
    from metasim.task.base import BaseTaskEnv

    @registry.register_task("lazy.mine")
    class Mine(BaseTaskEnv):
        pass

    monkeypatch.setenv("METASIM_CONFIG", "/nonexistent/metasim.toml")
    monkeypatch.setattr(registry, "_STATIC_INDEX", None)
    assert registry.get_task_class("lazy.mine") is Mine
    with pytest.raises(FileNotFoundError):  # a name that needs discovery still surfaces the config error
        registry.get_task_class("lazy.beta")


@pytest.mark.general
def test_a_settled_alias_name_is_served_from_the_registry(task_pkg, tmp_path, monkeypatch):
    """After one lookup established that every claimant registers the same class, the name is a dict
    hit like any other: the index is not consulted again."""
    _write_task(
        tmp_path,
        "aliases",
        """
        from lazypack.tasks.alpha import AlphaTask

        register_task("lazy.alpha")(AlphaTask)
        """,
    )
    assert registry.get_task_class("lazy.alpha").__name__ == "AlphaTask"

    def _boom():
        raise AssertionError("index consulted again")

    monkeypatch.setattr(registry, "_static_index", _boom)
    assert registry.get_task_class("lazy.alpha").__name__ == "AlphaTask"


@pytest.mark.general
def test_a_module_that_fails_to_parse_does_not_force_an_index_rebuild_per_lookup(task_pkg, tmp_path, monkeypatch):
    (tmp_path / "lazypack" / "tasks" / "unparsable.py").write_text("def (:\n", encoding="utf-8")
    builds = []
    real = registry.build_static_index
    monkeypatch.setattr(registry, "build_static_index", lambda *a, **k: (builds.append(1), real(*a, **k))[1])
    registry.list_tasks()
    registry.list_tasks()
    with pytest.raises(KeyError):
        registry.get_task_class("no.such.task")
    assert len(builds) == 1 and "lazypack.tasks.unparsable" in registry._STATIC_INDEX.parse_failures


@pytest.mark.general
def test_a_notebook_main_without_a_file_redefines_its_task(task_pkg, monkeypatch):
    import types

    from metasim.task.base import BaseTaskEnv

    nb = types.ModuleType("__main__")  # a kernel's __main__ has no __file__
    monkeypatch.setitem(sys.modules, "__main__", nb)
    for _ in range(2):  # the cell run twice
        cls = type("NbTask", (BaseTaskEnv,), {"__module__": "__main__"})
        registry.register_task("nb.task")(cls)
    assert registry.get_task_class("nb.task") is cls and registry._NAME_CONFLICTS == {}


@pytest.mark.general
def test_a_failed_claimant_is_retried_on_the_eager_path_too(task_pkg, tmp_path, monkeypatch):
    broken = tmp_path / "lazypack" / "tasks" / "gamma_broken.py"
    _write_task(tmp_path, "gamma", "@register_task('lazy.gamma')\nclass GammaTask(BaseTaskEnv):\n    pass\n")
    _write_task(
        tmp_path,
        "gamma_broken",
        "import module_that_does_not_exist_anywhere  # noqa: F401\n\n@register_task('lazy.gamma')\nclass GammaBrokenTask(BaseTaskEnv):\n    pass\n",
    )
    monkeypatch.setenv("METASIM_TASK_DISCOVERY", "eager")
    with pytest.raises(ValueError, match="which failed to import"):
        registry.get_task_class("lazy.gamma")
    broken.write_text(
        "from lazypack.tasks.gamma import GammaTask\nfrom metasim.task.registry import register_task\n\n"
        "register_task('lazy.gamma')(GammaTask)\n",
        encoding="utf-8",
    )
    import importlib

    importlib.invalidate_caches()
    assert registry.get_task_class("lazy.gamma").__name__ == "GammaTask"
