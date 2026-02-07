from __future__ import annotations

import inspect
import re
from dataclasses import fields, is_dataclass
from typing import Any, Callable


def filter_kwargs_for_callable(fn: Callable[..., Any], kwargs: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of kwargs containing only parameters accepted by fn."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return kwargs

    accepted = {}
    for name, value in kwargs.items():
        if name in sig.parameters:
            accepted[name] = value
    return accepted


def iter_public_fields(obj: Any):
    """Iterate public fields of a config-like object, preserving declaration order when possible."""
    if is_dataclass(obj):
        for f in fields(obj):
            if f.name.startswith("_"):
                continue
            yield f.name, getattr(obj, f.name)
        return

    for name, value in getattr(obj, "__dict__", {}).items():
        if name.startswith("_"):
            continue
        yield name, value


def is_term_cfg(obj: Any) -> bool:
    """Heuristic for identifying IsaacLab-style term cfg objects."""
    func = getattr(obj, "func", None)
    return callable(func)


def is_class_term_cfg(obj: Any) -> bool:
    """Heuristic for identifying IsaacLab-style term cfg objects that point to a term class."""
    return isinstance(getattr(obj, "class_type", None), type)


def resolve_matching_names(
    keys: str | list[str] | tuple[str, ...],
    *,
    candidates: list[str],
    preserve_order: bool = False,
) -> tuple[list[int], list[str]]:
    """Match regex keys against a list of candidate strings.

    This is a minimal, IsaacLab-compatible subset used by the compat layer.
    """
    if isinstance(keys, str):
        keys = [keys]
    else:
        keys = list(keys)

    index_list: list[int] = []
    names_list: list[str] = []
    key_idx_list: list[int] = []

    target_match = [None for _ in range(len(candidates))]
    keys_match = [[] for _ in range(len(keys))]

    for target_index, candidate in enumerate(candidates):
        for key_index, re_key in enumerate(keys):
            if re.fullmatch(re_key, candidate):
                if target_match[target_index] is not None:
                    raise ValueError(
                        f"Multiple matches for '{candidate}': '{target_match[target_index]}' and '{re_key}'"
                    )
                target_match[target_index] = re_key
                index_list.append(target_index)
                names_list.append(candidate)
                key_idx_list.append(key_index)
                keys_match[key_index].append(candidate)

    if preserve_order:
        reordered = [None] * len(index_list)
        global_index = 0
        for key_index in range(len(keys)):
            for idx, entry in enumerate(key_idx_list):
                if entry == key_index:
                    reordered[idx] = global_index
                    global_index += 1

        idx_reorder = [None] * len(index_list)
        names_reorder = [None] * len(index_list)
        for src, dst in enumerate(reordered):
            idx_reorder[dst] = index_list[src]
            names_reorder[dst] = names_list[src]

        index_list = idx_reorder
        names_list = names_reorder

    if keys and not all(keys_match):
        msg = "\n"
        for key, matches in zip(keys, keys_match):
            msg += f"\t{key}: {matches}\n"
        msg += f"Available strings: {candidates}\n"
        raise ValueError(f"Not all regex keys matched.{msg}")

    return index_list, names_list


def resolve_pattern_values(
    pattern_to_value: dict[str, Any] | None, names: list[str], *, default: Any = 0.0
) -> dict[str, Any]:
    """Resolve a regex->value mapping into a per-name dict."""
    out = {name: default for name in names}
    if not pattern_to_value:
        return out
    for pattern, value in pattern_to_value.items():
        rx = re.compile(pattern)
        for name in names:
            if rx.fullmatch(name):
                out[name] = value
    return out


def is_scene_entity_cfg(obj: Any) -> bool:
    """Heuristic for identifying IsaacLab's `SceneEntityCfg` objects (real or shim)."""
    if obj is None:
        return False
    cls = getattr(obj, "__class__", None)
    if cls is None:
        return False
    if getattr(cls, "__name__", "") != "SceneEntityCfg":
        return False
    return hasattr(obj, "name")


def _is_leaf_value(obj: Any) -> bool:
    return obj is None or isinstance(obj, (int, float, str, bool, bytes))


def resolve_scene_entity_cfgs(root_cfg: Any, *, scene: Any) -> None:
    """Resolve `SceneEntityCfg` patterns into explicit indices in-place.

    Many IsaacLab term functions expect `SceneEntityCfg.body_ids` / `joint_ids` to exist and be
    pre-populated (resolved from regex patterns). IsaacLab normally performs this resolution as
    part of manager initialization. MetaSim compat does it once up-front to minimize task changes.
    """
    visited: set[int] = set()

    def _resolve_one(entity_cfg: Any) -> None:
        name = getattr(entity_cfg, "name", None)
        if not isinstance(name, str) or not name:
            return

        # Sensor reference (e.g., SceneEntityCfg("contact_forces", body_names=[...]))
        sensors = getattr(scene, "sensors", None)
        if isinstance(sensors, dict) and name in sensors:
            sensor = sensors[name]
            body_names = getattr(entity_cfg, "body_names", None)
            if body_names is not None and hasattr(sensor, "find_bodies"):
                try:
                    body_ids, _ = sensor.find_bodies(body_names, preserve_order=True)
                    entity_cfg.body_ids = slice(None) if body_names in (".*", [".*"]) else body_ids
                except Exception:
                    pass
            return

        # Asset reference (robot/object)
        try:
            asset = scene[name]
        except Exception:
            return

        joint_names = getattr(entity_cfg, "joint_names", None)
        if joint_names is not None and hasattr(asset, "find_joints"):
            try:
                joint_ids, _ = asset.find_joints(joint_names, preserve_order=True)
                # Preserve a common IsaacLab convention: ".*" means "all joints".
                if joint_names == ".*" or joint_names == [".*"]:
                    entity_cfg.joint_ids = slice(None)
                else:
                    entity_cfg.joint_ids = joint_ids
            except Exception:
                pass

        body_names = getattr(entity_cfg, "body_names", None)
        if body_names is not None and hasattr(asset, "find_bodies"):
            try:
                body_ids, _ = asset.find_bodies(body_names, preserve_order=True)
                if body_names == ".*" or body_names == [".*"]:
                    entity_cfg.body_ids = slice(None)
                else:
                    entity_cfg.body_ids = body_ids
            except Exception:
                pass

    def _walk(obj: Any) -> None:
        oid = id(obj)
        if oid in visited:
            return
        visited.add(oid)

        if _is_leaf_value(obj):
            return
        if callable(obj) or isinstance(obj, type) or inspect.ismodule(obj):
            return

        if is_scene_entity_cfg(obj):
            _resolve_one(obj)

        if isinstance(obj, dict):
            for v in obj.values():
                _walk(v)
            return
        if isinstance(obj, (list, tuple, set)):
            for v in obj:
                _walk(v)
            return

        if is_dataclass(obj):
            for f in fields(obj):
                if f.name.startswith("_"):
                    continue
                try:
                    _walk(getattr(obj, f.name))
                except Exception:
                    continue
            return

        for _name, v in getattr(obj, "__dict__", {}).items():
            if _name.startswith("_"):
                continue
            _walk(v)

    _walk(root_cfg)
