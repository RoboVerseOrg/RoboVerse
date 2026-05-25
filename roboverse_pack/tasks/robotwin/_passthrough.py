"""RoboTwin passthrough — expose RoboTwin v2 tasks through RoboVerse's gym.make.

Mirrors the ManiSkill passthrough (``maniskill._passthrough_v3``): RoboTwin
tasks are ``Base_Task(gym.Env)`` subclasses in ``<ROBOTWIN_DIR>/envs/<name>.py``
that are configured (config + embodiment + ``setup_demo``) rather than
``gym.make``-able by id. This module registers each task under a
``RoboTwin/<name>`` id with a *lazy* entry point: registration does NOT import
RoboTwin (so it is safe even when RoboTwin's heavy deps — mplib 0.2.1 /
pytorch3d / curobo / torch 2.4.1 — are not installed in this env), and the
underlying task is only imported + set up when the env is actually made.

RoboTwin must be cloned at ``ROBOTWIN_DIR`` (default ``~/projects/robotwin``) and
its dependency stack installed in the active env (its deps conflict with the
RoboVerse env's torch/mplib, so use a dedicated conda env + run
``<ROBOTWIN_DIR>/script/_install.sh``). Until then, ``gym.make("RoboTwin/...")``
raises a clear, actionable error; registration still succeeds so the ids are
discoverable.

Once installed, ``RoboTwin/<name>`` runs the native RoboTwin task → 1:1 by
construction (same code path as RoboTwin's ``collect_data.py``), exactly as the
ManiSkill passthrough is bitwise-identical to native ManiSkill.
"""

from __future__ import annotations

import os

from loguru import logger as log

_DEFAULT_ROBOTWIN_DIR = os.path.expanduser("~/projects/robotwin")
# envs/*.py files that are infrastructure, not tasks.
_NON_TASK_STEMS = {"_base_task", "utils", "__init__"}


def robotwin_dir() -> str:
    """RoboTwin checkout dir (``ROBOTWIN_DIR`` env var or the default)."""
    return os.environ.get("ROBOTWIN_DIR", _DEFAULT_ROBOTWIN_DIR)


def _prepare_robotwin_runtime(rt: str) -> None:
    """Make the active env importable as RoboTwin expects, in place.

    Two environment quirks have to be handled before importing any RoboTwin
    task, and both are no-ops once satisfied:

    * **CWD**: RoboTwin reads assets via *relative* paths (``./assets/...``) at
      import time (e.g. ``envs/utils`` loads ``./assets/objects/objaverse/
      list.json``), so the process CWD must be the checkout root.
    * **warp.torch shim**: RoboTwin's planner uses curobo 0.7.8, which calls
      ``warp.torch.device_from_torch``; warp-lang >= 1.5 moved those helpers to
      the ``warp`` top level and dropped the ``warp.torch`` submodule. We alias
      the old namespace to the new functions so curobo imports unchanged.
    """
    import sys

    if os.path.realpath(os.getcwd()) != os.path.realpath(rt):
        os.chdir(rt)

    if "warp.torch" not in sys.modules:
        try:
            import types

            import warp as wp

            shim = types.ModuleType("warp.torch")
            for fn in (
                "device_from_torch",
                "device_to_torch",
                "dtype_from_torch",
                "dtype_to_torch",
                "from_torch",
                "to_torch",
                "stream_from_torch",
                "stream_to_torch",
            ):
                if hasattr(wp, fn):
                    setattr(shim, fn, getattr(wp, fn))
            sys.modules["warp.torch"] = shim
            wp.torch = shim
        except ImportError:
            pass  # warp not installed; curobo import will report it directly


def _discover_task_names(rt_dir: str) -> list[str]:
    """List RoboTwin task names = ``envs/<name>.py`` files (no import needed).

    Pure filename scan — does not import RoboTwin, so it works even when
    RoboTwin's deps are absent. Returns [] if the checkout is missing.
    """
    envs = os.path.join(rt_dir, "envs")
    if not os.path.isdir(envs):
        return []
    names = []
    for fn in os.listdir(envs):
        if not fn.endswith(".py") or (fn.startswith("_") and fn != "_base_task.py"):
            continue
        stem = fn[:-3]
        if stem in _NON_TASK_STEMS or stem.startswith("_"):
            continue
        # Heuristic: a task module defines a class named exactly like the file.
        names.append(stem)
    return sorted(names)


def _make_robotwin_env(task_name: str, task_config: str = "demo_clean", seed: int = 0, **kwargs):
    """Lazy factory: set up + return the native RoboTwin task (a ``gym.Env``).

    Replicates RoboTwin ``script/collect_data.py`` config assembly (task config
    + embodiment files) and calls ``setup_demo``. Raises an actionable error if
    RoboTwin or its deps are not importable in this env.
    """
    import importlib
    import sys

    import yaml

    rt = robotwin_dir()
    if not os.path.isdir(os.path.join(rt, "envs")):
        raise RuntimeError(
            f"RoboTwin checkout not found at {rt!r}. Clone it (or set $ROBOTWIN_DIR) before "
            f"`gym.make('RoboTwin/{task_name}')`."
        )
    if rt not in sys.path:
        sys.path.insert(0, rt)
    _prepare_robotwin_runtime(rt)
    cfg_dir = os.path.join(rt, "task_config")

    try:
        mod = importlib.import_module(f"envs.{task_name}")
    except Exception as e:
        raise RuntimeError(
            f"Could not import RoboTwin task 'envs.{task_name}' ({type(e).__name__}: {e}). "
            f"RoboTwin's deps are likely not installed in this env (it needs mplib 0.2.1 / "
            f"sapien 3.0.0b1 / pytorch3d / curobo / torch 2.4.1, which conflict with the RoboVerse "
            f"env). Install them in a dedicated env via `{rt}/script/_install.sh` (curobo needs an "
            f"sm-matching CUDA nvcc), then this task runs the native RoboTwin code path (1:1)."
        ) from e

    task_cls = getattr(mod, task_name)
    args = yaml.safe_load(open(os.path.join(cfg_dir, f"{task_config}.yml")))
    args["task_name"] = task_name
    emb = args["embodiment"]
    emb_types = yaml.safe_load(open(os.path.join(cfg_dir, "_embodiment_config.yml")))

    def _emb_file(t: str) -> str:
        fp = emb_types[t]["file_path"]
        return fp if os.path.isabs(fp) else os.path.join(rt, fp)

    if len(emb) == 1:
        args["left_robot_file"] = args["right_robot_file"] = _emb_file(emb[0])
        args["dual_arm_embodied"] = True
    elif len(emb) == 3:
        args["left_robot_file"] = _emb_file(emb[0])
        args["right_robot_file"] = _emb_file(emb[1])
        args["embodiment_dis"] = emb[2]
        args["dual_arm_embodied"] = False
    else:
        raise ValueError(f"embodiment config must have 1 or 3 entries, got {emb}")
    args["left_embodiment_config"] = yaml.safe_load(open(os.path.join(args["left_robot_file"], "config.yml")))
    args["right_embodiment_config"] = yaml.safe_load(open(os.path.join(args["right_robot_file"], "config.yml")))
    args["render_freq"] = 0
    args.update(kwargs)

    env = task_cls()
    env.setup_demo(now_ep_num=0, seed=seed, **args)
    return env


def register_robotwin_passthrough(prefix: str = "RoboTwin/") -> list[str]:
    """Register every RoboTwin task as ``RoboTwin/<name>`` (lazy entry point).

    Idempotent. Returns the registered ids. Safe when RoboTwin/its deps are
    absent — only filenames are scanned; no RoboTwin import happens here.
    """
    from gymnasium.envs.registration import register, registry

    rt = robotwin_dir()
    names = _discover_task_names(rt)
    if not names:
        log.debug(f"[robotwin_passthrough] no RoboTwin tasks found under {rt}/envs (checkout missing?)")
        return []
    registered = []
    for name in names:
        ns_id = f"{prefix}{name}"
        if ns_id in registry:
            registered.append(ns_id)
            continue
        try:
            register(
                id=ns_id,
                entry_point="roboverse_pack.tasks.robotwin._passthrough:_make_robotwin_env",
                kwargs={"task_name": name},
            )
            registered.append(ns_id)
        except Exception as e:
            log.debug(f"[robotwin_passthrough] failed to register {ns_id}: {e}")
    log.info(f"[robotwin_passthrough] registered {len(registered)} RoboTwin tasks under {prefix!r}")
    return registered
