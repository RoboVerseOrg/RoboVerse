from __future__ import annotations

import sys
import textwrap
from contextlib import contextmanager
from importlib import import_module
from pathlib import Path

import pytest

from metasim.utils import package_discovery

_ENV_VARS = (
    "METASIM_CONFIG",
    "METASIM_PACKAGES",
    "METASIM_TASK_PACKAGES",
    "METASIM_ROBOT_PACKAGES",
    "METASIM_SCENE_PACKAGES",
    "METASIM_GROUND_PACKAGES",
)


class _FakeEntryPoint:
    def __init__(self, value: str):
        self.value = value


class _FakeEntryPoints:
    def __init__(self, groups: dict[str, list[str]]):
        self._groups = groups

    def select(self, *, group: str):
        return [_FakeEntryPoint(value) for value in self._groups.get(group, [])]


def _write(path: Path, text: str) -> None:
    path.write_text(textwrap.dedent(text).strip() + "\n", encoding="utf-8")


def _is_module_under(name: str, prefix: str) -> bool:
    return name == prefix or name.startswith(prefix + ".")


@contextmanager
def _isolated_task_registry_discovery(
    registry,
    module_prefixes=("metasim.example.example_pack.tasks", "custom_tasks"),
):
    original_registry = dict(registry.TASK_REGISTRY)
    original_failures = dict(registry._DISCOVERY_FAILURES)
    original_modules = {
        module_name: module
        for module_name, module in sys.modules.items()
        if any(_is_module_under(module_name, prefix) for prefix in module_prefixes)
    }

    def _remove_task_modules() -> None:
        for module_name in list(sys.modules):
            if any(_is_module_under(module_name, prefix) for prefix in module_prefixes):
                sys.modules.pop(module_name, None)

    try:
        registry.TASK_REGISTRY.clear()
        registry._DISCOVERY_FAILURES.clear()
        _remove_task_modules()
        yield
    finally:
        _remove_task_modules()
        sys.modules.update(original_modules)
        registry.TASK_REGISTRY.clear()
        registry.TASK_REGISTRY.update(original_registry)
        registry._DISCOVERY_FAILURES.clear()
        registry._DISCOVERY_FAILURES.update(original_failures)


@pytest.fixture(autouse=True)
def _clear_metasim_package_env(monkeypatch):
    for env_var in _ENV_VARS:
        monkeypatch.delenv(env_var, raising=False)


@pytest.mark.general
def test_package_candidates_merge_sources_in_order(tmp_path, monkeypatch):
    _write(
        tmp_path / "metasim.toml",
        """
        [packages]
        roots = ["toml_root"]
        robots = ["toml_robot"]
        """,
    )
    _write(
        tmp_path / "pyproject.toml",
        """
        [tool.metasim.packages]
        roots = ["pyproject_root"]
        robots = ["pyproject_robot"]
        """,
    )
    explicit_cfg = tmp_path / "explicit-metasim.toml"
    _write(
        explicit_cfg,
        """
        [packages]
        roots = ["explicit_root"]
        robots = ["explicit_robot"]
        """,
    )

    monkeypatch.setattr(
        package_discovery,
        "entry_points",
        lambda: _FakeEntryPoints({
            "metasim.packages": ["entry_root"],
            "metasim.robots": ["entry_robot"],
        }),
    )
    monkeypatch.setenv("METASIM_CONFIG", str(explicit_cfg))
    monkeypatch.setenv("METASIM_PACKAGES", "env_root:entry_root")
    monkeypatch.setenv("METASIM_ROBOT_PACKAGES", "env_robot")

    packages = package_discovery.get_package_candidates(
        "robots",
        defaults=["builtin_robot"],
        local_modules=["local_robot"],
        cwd=tmp_path,
    )

    assert packages == [
        "builtin_robot",
        "entry_root.robots",
        "entry_robot",
        "toml_root.robots",
        "toml_robot",
        "pyproject_root.robots",
        "pyproject_robot",
        "explicit_root.robots",
        "explicit_robot",
        "env_root.robots",
        "env_robot",
        "local_robot",
    ]


@pytest.mark.general
def test_env_packages_are_colon_separated_stripped_and_deduped(tmp_path, monkeypatch):
    monkeypatch.setattr(package_discovery, "entry_points", lambda: _FakeEntryPoints({}))
    monkeypatch.setenv("METASIM_PACKAGES", "alpha: beta :alpha::")
    monkeypatch.setenv("METASIM_TASK_PACKAGES", "custom.tasks: beta.tasks ")

    packages = package_discovery.get_package_candidates("tasks", cwd=tmp_path)

    assert packages == ["alpha.tasks", "beta.tasks", "custom.tasks"]


@pytest.mark.general
def test_unknown_config_key_fails_fast(tmp_path, monkeypatch):
    monkeypatch.setattr(package_discovery, "entry_points", lambda: _FakeEntryPoints({}))
    _write(
        tmp_path / "metasim.toml",
        """
        [packages]
        rootz = ["typo"]
        """,
    )

    with pytest.raises(ValueError, match="rootz"):
        package_discovery.get_package_candidates("robots", cwd=tmp_path)


@pytest.mark.general
def test_metasim_config_missing_file_fails_fast(tmp_path, monkeypatch):
    monkeypatch.setattr(package_discovery, "entry_points", lambda: _FakeEntryPoints({}))
    monkeypatch.setenv("METASIM_CONFIG", str(tmp_path / "missing.toml"))

    with pytest.raises(FileNotFoundError, match=r"missing\.toml"):
        package_discovery.get_package_candidates("robots", cwd=tmp_path)


@pytest.mark.general
def test_metasim_config_supports_tool_metasim_packages_table(tmp_path, monkeypatch):
    monkeypatch.setattr(package_discovery, "entry_points", lambda: _FakeEntryPoints({}))
    explicit_cfg = tmp_path / "explicit.toml"
    _write(
        explicit_cfg,
        """
        [tool.metasim.packages]
        roots = ["explicit_root"]
        scenes = ["explicit_scene"]
        """,
    )
    monkeypatch.setenv("METASIM_CONFIG", str(explicit_cfg))

    packages = package_discovery.get_package_candidates("scenes", cwd=tmp_path)

    assert packages == ["explicit_root.scenes", "explicit_scene"]


@pytest.mark.general
def test_toml_package_values_are_normalized(tmp_path, monkeypatch):
    monkeypatch.setattr(package_discovery, "entry_points", lambda: _FakeEntryPoints({}))
    _write(
        tmp_path / "metasim.toml",
        """
        [packages]
        roots = [" root_pkg:unused ", "", "root_pkg"]
        tasks = [" task_pkg:TaskRegistry ", "  "]
        """,
    )

    packages = package_discovery.get_package_candidates("tasks", cwd=tmp_path)

    assert packages == ["root_pkg.tasks", "task_pkg"]


@pytest.mark.general
def test_malformed_known_config_table_path_fails_fast(tmp_path, monkeypatch):
    monkeypatch.setattr(package_discovery, "entry_points", lambda: _FakeEntryPoints({}))
    _write(
        tmp_path / "metasim.toml",
        """
        packages = "roboverse_pack"
        """,
    )

    with pytest.raises(TypeError, match="packages"):
        package_discovery.get_package_candidates("robots", cwd=tmp_path)


@pytest.mark.general
def test_get_robot_uses_builtin_example_franka(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(package_discovery, "entry_points", lambda: _FakeEntryPoints({}))
    monkeypatch.delenv("METASIM_CONFIG", raising=False)

    from metasim.utils.setup_util import get_robot

    robot = get_robot("franka")

    assert robot.name == "franka"
    assert type(robot).__module__ == "metasim.example.example_pack.robots.franka_cfg"


@pytest.mark.general
def test_get_robot_uses_builtin_example_g1_dof29(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(package_discovery, "entry_points", lambda: _FakeEntryPoints({}))
    monkeypatch.delenv("METASIM_CONFIG", raising=False)

    from metasim.utils.setup_util import get_robot

    robot = get_robot("g1_dof29")

    assert robot.name == "g1_dof29"
    assert type(robot).__module__ == "metasim.example.example_pack.robots.g1_cfg"


@pytest.mark.general
def test_get_robot_uses_configured_robot_package(tmp_path, monkeypatch):
    robot_pkg = tmp_path / "custom_robots"
    robot_pkg.mkdir()
    _write(
        robot_pkg / "__init__.py",
        """
        from metasim.scenario.robot import RobotCfg
        from metasim.utils import configclass

        @configclass
        class ToyBotCfg(RobotCfg):
            name: str = "toy_bot"
            num_joints: int = 0
        """,
    )
    _write(
        tmp_path / "metasim.toml",
        """
        [packages]
        robots = ["custom_robots"]
        """,
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(package_discovery, "entry_points", lambda: _FakeEntryPoints({}))
    monkeypatch.delenv("METASIM_CONFIG", raising=False)

    from metasim.utils.setup_util import get_robot

    robot = get_robot("toy_bot")

    assert robot.name == "toy_bot"
    assert type(robot).__module__ == "custom_robots"


@pytest.mark.general
def test_task_registry_uses_builtin_example_tasks(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(package_discovery, "entry_points", lambda: _FakeEntryPoints({}))
    monkeypatch.delenv("METASIM_CONFIG", raising=False)

    from metasim.task import registry

    with _isolated_task_registry_discovery(registry):
        tasks = registry.list_tasks()

        assert "stack_cube" in tasks


@pytest.mark.general
def test_task_registry_uses_configured_task_package(tmp_path, monkeypatch):
    task_pkg = tmp_path / "custom_tasks"
    task_pkg.mkdir()
    _write(task_pkg / "__init__.py", "")
    _write(
        task_pkg / "example_task.py",
        """
        from metasim.task.registry import register_task

        @register_task("custom.example")
        class CustomExampleTask:
            pass
        """,
    )
    _write(
        tmp_path / "metasim.toml",
        """
        [packages]
        tasks = ["custom_tasks"]
        """,
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(package_discovery, "entry_points", lambda: _FakeEntryPoints({}))
    monkeypatch.delenv("METASIM_CONFIG", raising=False)

    from metasim.task import registry

    with _isolated_task_registry_discovery(registry):
        task_cls = registry.get_task_class("custom.example")

        assert task_cls.__name__ == "CustomExampleTask"


@pytest.mark.general
def test_task_registry_surfaces_configured_task_root_import_failure(tmp_path, monkeypatch):
    _write(
        tmp_path / "metasim.toml",
        """
        [packages]
        tasks = ["missing_tasks_pkg"]
        """,
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(package_discovery, "entry_points", lambda: _FakeEntryPoints({}))
    monkeypatch.delenv("METASIM_CONFIG", raising=False)

    from metasim.task import registry

    with _isolated_task_registry_discovery(registry, module_prefixes=("missing_tasks_pkg",)):
        with pytest.raises(KeyError) as exc_info:
            registry.get_task_class("missing.task")

    assert "missing_tasks_pkg" in str(exc_info.value)


@pytest.mark.general
def test_task_registry_builtin_discovery_can_repeat_after_cleanup(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(package_discovery, "entry_points", lambda: _FakeEntryPoints({}))
    monkeypatch.delenv("METASIM_CONFIG", raising=False)

    from metasim.task import registry

    module_name = "metasim.example.example_pack.tasks.stack_cube"
    original_module = import_module(module_name)
    assert module_name in sys.modules

    for _ in range(2):
        with _isolated_task_registry_discovery(registry):
            assert "stack_cube" in registry.list_tasks()

    assert sys.modules[module_name] is original_module
    assert import_module(module_name) is original_module
