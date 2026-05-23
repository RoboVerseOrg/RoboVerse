"""Meta-World passthrough — expose 50 tasks under MetaWorld/<task-name>."""

from __future__ import annotations

from typing import Any

from loguru import logger as log


def _ensure_imported():
    import metaworld

    return metaworld


def register_metaworld_passthrough(prefix: str = "MetaWorld/") -> list[str]:
    """Register all Meta-World MT50 train + test tasks under MetaWorld/<name> namespace."""
    from gymnasium.envs.registration import register, registry

    metaworld = _ensure_imported()
    # MT50 has both train + test splits, but for simplicity register all 50 from MT50
    mt = metaworld.MT50()
    names = sorted(mt.train_classes.keys())
    registered = []
    failed = []
    for name in names:
        ns_id = f"{prefix}{name}"
        if ns_id in registry:
            registered.append(ns_id)
            continue
        try:
            register(
                id=ns_id,
                entry_point="roboverse_pack.tasks.metaworld._passthrough:_make_metaworld_env",
                kwargs={"name": name},
            )
            registered.append(ns_id)
        except Exception as e:
            failed.append((ns_id, str(e)))
    if failed:
        log.warning(f"[metaworld_passthrough] {len(failed)} failures")
    return registered


def _make_metaworld_env(name: str, **kwargs: Any):
    """Instantiate one Meta-World env."""
    metaworld = _ensure_imported()
    mt = metaworld.MT50()
    env_cls = mt.train_classes[name]
    task_idx = next((i for i, t in enumerate(mt.train_tasks) if t.env_name == name), 0)
    env = env_cls(render_mode=kwargs.pop("render_mode", None))
    env.set_task(mt.train_tasks[task_idx])
    return env
