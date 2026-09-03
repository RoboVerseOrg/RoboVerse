"""Discover MetaSim content package candidates from local configuration."""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

try:
    import tomllib
except ImportError:  # pragma: no cover - Python < 3.11
    import tomli as tomllib  # type: ignore[no-redef]

try:
    from importlib.metadata import entry_points
except ImportError:  # pragma: no cover - Python < 3.8
    from importlib_metadata import entry_points  # type: ignore[no-redef]


ROLES = ("tasks", "robots", "scenes", "grounds")
_CONFIG_KEYS = ("roots",) + ROLES
_ENTRY_POINT_GROUPS = {
    "tasks": "metasim.tasks",
    "robots": "metasim.robots",
    "scenes": "metasim.scenes",
    "grounds": "metasim.grounds",
}
_ENV_VARS = {
    "tasks": "METASIM_TASK_PACKAGES",
    "robots": "METASIM_ROBOT_PACKAGES",
    "scenes": "METASIM_SCENE_PACKAGES",
    "grounds": "METASIM_GROUND_PACKAGES",
}
_CONFIG_TABLE_PATHS = (("packages",), ("tool", "metasim", "packages"))


@dataclass(frozen=True)
class PackageConfig:
    """Package roots / per-role package lists parsed from ``metasim.toml`` or ``pyproject.toml``."""

    roots: tuple[str, ...] = ()
    tasks: tuple[str, ...] = ()
    robots: tuple[str, ...] = ()
    scenes: tuple[str, ...] = ()
    grounds: tuple[str, ...] = ()

    def packages_for(self, role: str) -> tuple[str, ...]:
        """Return ``<root>.<role>`` for every root plus the explicit packages for ``role``."""
        _validate_role(role)
        root_packages = tuple("%s.%s" % (root, role) for root in self.roots)
        return root_packages + getattr(self, role)


def get_package_candidates(
    role: str,
    defaults: Sequence[str] = (),
    local_modules: Sequence[str] = (),
    cwd: Path | None = None,
) -> list[str]:
    """Return package candidates for a MetaSim content role without importing them."""
    _validate_role(role)
    base_dir = Path.cwd() if cwd is None else Path(cwd)
    candidates = []
    candidates.extend(defaults)
    candidates.extend(_entry_point_packages(role))
    candidates.extend(_nearest_config_packages(base_dir, "metasim.toml", role))
    candidates.extend(_nearest_config_packages(base_dir, "pyproject.toml", role))
    candidates.extend(_explicit_config_packages(role))
    candidates.extend(_env_packages(role))
    candidates.extend(local_modules)
    return _dedupe(candidates)


def _validate_role(role: str) -> None:
    if role not in ROLES:
        raise ValueError("Unknown MetaSim package role %r; expected one of %s" % (role, ", ".join(ROLES)))


def _entry_point_packages(role: str) -> tuple[str, ...]:
    eps = entry_points()
    roots = _select_entry_point_values(eps, "metasim.packages")
    role_packages = _select_entry_point_values(eps, _ENTRY_POINT_GROUPS[role])
    return PackageConfig(roots=roots, **{role: role_packages}).packages_for(role)


def _select_entry_point_values(eps, group: str) -> tuple[str, ...]:
    if hasattr(eps, "select"):
        selected = eps.select(group=group)
    else:  # pragma: no cover - compatibility with older importlib_metadata
        selected = eps.get(group, ())
    return tuple(_normalize_string_list(ep.value for ep in selected))


def _nearest_config_packages(base_dir: Path, filename: str, role: str) -> tuple[str, ...]:
    config_path = _find_nearest(base_dir, filename)
    if config_path is None:
        return ()
    return _load_config(config_path).packages_for(role)


def _find_nearest(base_dir: Path, filename: str) -> Path | None:
    current = base_dir.resolve()
    if current.is_file():
        current = current.parent
    for directory in (current,) + tuple(current.parents):
        candidate = directory / filename
        if candidate.is_file():
            return candidate
    return None


def _explicit_config_packages(role: str) -> tuple[str, ...]:
    config_env = os.environ.get("METASIM_CONFIG")
    if not config_env:
        return ()
    config_path = Path(config_env)
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    return _load_config(config_path).packages_for(role)


def _load_config(path: Path) -> PackageConfig:
    try:
        with path.open("rb") as f:
            data = tomllib.load(f)
    except tomllib.TOMLDecodeError as exc:
        raise ValueError("Invalid TOML in %s: %s" % (path, exc)) from exc

    config = PackageConfig()
    for table_path in _CONFIG_TABLE_PATHS:
        table = _get_table(data, table_path, path)
        config = _merge_configs(config, _parse_config_table(table, path))
    return config


def _merge_configs(left: PackageConfig, right: PackageConfig) -> PackageConfig:
    return PackageConfig(
        roots=left.roots + right.roots,
        tasks=left.tasks + right.tasks,
        robots=left.robots + right.robots,
        scenes=left.scenes + right.scenes,
        grounds=left.grounds + right.grounds,
    )


def _get_table(data: Mapping[str, object], table_path: Sequence[str], path: Path) -> Mapping[str, object]:
    table = data
    traversed = []
    for key in table_path:
        traversed.append(key)
        value = table.get(key)
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise TypeError("Package config table %s in %s must be a table" % (".".join(traversed), path))
        table = value
    return table


def _parse_config_table(table: Mapping[str, object], path: Path) -> PackageConfig:
    unknown_keys = sorted(set(table) - set(_CONFIG_KEYS))
    if unknown_keys:
        raise ValueError("Unknown package config keys in %s: %s" % (path, ", ".join(unknown_keys)))

    values = {}
    for key in _CONFIG_KEYS:
        values[key] = _parse_string_list(table.get(key, ()), key, path)
    return PackageConfig(**values)


def _parse_string_list(value: object, key: str, path: Path) -> tuple[str, ...]:
    if value == ():
        return ()
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise TypeError("Package config key %s in %s must be a list of strings" % (key, path))
    return tuple(_dedupe(_normalize_string_list(value)))


def _env_packages(role: str) -> tuple[str, ...]:
    roots = _parse_env_var(os.environ.get("METASIM_PACKAGES"))
    role_packages = _parse_env_var(os.environ.get(_ENV_VARS[role]))
    return PackageConfig(roots=roots, **{role: role_packages}).packages_for(role)


def _parse_env_var(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(_dedupe(part.strip() for part in value.split(":") if part.strip()))


def _normalize_string_list(values: Iterable[str]) -> Iterable[str]:
    for value in values:
        normalized = _normalize_package_value(value)
        if normalized:
            yield normalized


def _normalize_package_value(value: str) -> str:
    return value.split(":", 1)[0].strip()


def _dedupe(values: Iterable[str]) -> list[str]:
    seen = set()
    deduped = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped
