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
