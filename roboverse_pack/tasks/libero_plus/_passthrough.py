"""LIBERO-plus passthrough — expose all LIBERO-plus perturbation tasks 1:1.

LIBERO-plus (*In-depth Robustness Analysis of VLA Models*, 2025) is a drop-in
superset of LIBERO: the same ``OffScreenRenderEnv`` / robosuite / MuJoCo stack
and the same five suite names (``libero_spatial`` / ``libero_object`` /
``libero_goal`` / ``libero_10`` / ``libero_90``), but each suite's task list is
expanded with thousands of *perturbation* variants spanning seven dimensions
(object layout, camera viewpoint, robot init state, language rewrite, lighting,
background texture, sensor noise). Together they are the **10,120-task**
LIBERO-plus benchmark (spatial 2402 / object 2518 / goal 2591 / 10 2519 /
90 90).

Because the underlying simulator is LIBERO-plus's own robosuite/MuJoCo env, this
passthrough is **1:1 by construction** — we build the exact native env that
LIBERO-plus would build (mirroring the LIBERO passthrough one-for-one), so the
observation / dynamics / 128x128 camera bytes are bitwise-identical to native.

Two things differ from the base-LIBERO passthrough, both required for the
perturbation tasks to resolve:

* **Config path.** LIBERO-plus reads its BDDL / asset / init-state roots from a
  ``config.yaml`` located at ``$LIBERO_CONFIG_PATH`` (default ``~/.libero``).
  The base-LIBERO config points at the *base* repo, so it would only enumerate
  the 130 base tasks (or fail). This module points ``LIBERO_CONFIG_PATH`` at a
  dedicated config whose roots resolve into the *installed LIBERO-plus package*,
  written self-contained from the package location (so it works on a fresh
  checkout without depending on any gitignored config dir). A user-supplied
  ``LIBERO_CONFIG_PATH`` is respected and never overwritten.
* **Scale.** ~10k lazy gym ids under ``LiberoPlus/<suite>__<task>``. Registration
  does NOT import LIBERO-plus (safe no-op when its heavy deps -- ``robosuite`` /
  ``bddl`` / ``mujoco`` / ``numpy<1.24`` -- are absent), so the default RoboVerse
  env is unaffected; the env is only built when actually made.

Like the base-LIBERO passthrough, the native env uses the *legacy* gym API
(``step`` -> 4-tuple ``(obs, reward, done, info)``, ``reset`` -> ``obs``); this
is preserved deliberately for bitwise fidelity. Prefer :func:`make_liberoplus_env`
(returns the raw native env) for replay / eval.
"""

from __future__ import annotations

import importlib.util
import os
from typing import Any

import yaml
from loguru import logger as log

#: The five LIBERO(-plus) suites, in canonical order. LIBERO-plus keeps the base
#: suite *names* but expands each task list with perturbation variants.
LIBERO_PLUS_SUITES: tuple[str, ...] = (
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_10",
    "libero_90",
)

#: Suite/task separator inside a gym id (single ``/`` is the gym namespace).
_SEP = "__"

#: Per-suite task counts in the reference LIBERO-plus distribution (HEAD as
#: cloned). Used only as a sanity anchor in tests / detection -- the
#: authoritative counts always come from the live benchmark dict.
REFERENCE_SUITE_COUNTS: dict[str, int] = {
    "libero_spatial": 2402,
    "libero_object": 2518,
    "libero_goal": 2591,
    "libero_10": 2519,
    "libero_90": 90,
}

#: Optional cap (env var) on tasks registered per suite -- purely to keep a smoke
#: import fast; the make_* factories and verification ignore it.
_MAX_PER_SUITE = os.environ.get("LIBERO_PLUS_MAX_TASKS_PER_SUITE")


def _liberoplus_pkg_dir() -> str | None:
    """Return the installed ``libero/libero`` package dir, or ``None`` if absent.

    Located via ``importlib`` *without importing* the package, so we can write a
    correct ``config.yaml`` before LIBERO-plus reads it at import time (the base
    config would otherwise resolve to the wrong repo and break enumeration).
    """
    try:
        spec = importlib.util.find_spec("libero.libero")
    except Exception:
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    return os.path.abspath(next(iter(spec.submodule_search_locations)))


def _stale(cfg_file: str, want: dict[str, str]) -> bool:
    try:
        with open(cfg_file) as f:
            have = yaml.safe_load(f) or {}
    except Exception:
        return True
    return any(have.get(k) != v for k, v in want.items())


def _ensure_config() -> bool:
    """Point ``LIBERO_CONFIG_PATH`` at a config that resolves to LIBERO-plus.

    Returns ``True`` if a usable config is in place. Honors a pre-existing
    ``LIBERO_CONFIG_PATH`` (never overwrites it). Otherwise writes a dedicated
    ``~/.libero_plus/config.yaml`` whose roots point into the installed
    LIBERO-plus package, and exports ``LIBERO_CONFIG_PATH`` to it. Idempotent.
    """
    if os.environ.get("LIBERO_CONFIG_PATH"):
        # User (or a previous call in this process) already chose a config dir.
        return True

    pkg = _liberoplus_pkg_dir()
    if pkg is None:
        return False

    cfg_dir = os.path.expanduser("~/.libero_plus")
    os.makedirs(cfg_dir, exist_ok=True)
    cfg = {
        "benchmark_root": pkg,
        "bddl_files": os.path.join(pkg, "bddl_files"),
        "init_states": os.path.join(pkg, "init_files"),
        "datasets": os.path.normpath(os.path.join(pkg, "..", "datasets")),
        "assets": os.path.join(pkg, "assets"),
    }
    cfg_file = os.path.join(cfg_dir, "config.yaml")
    # Rewrite only when missing/stale so we don't thrash the file every import.
    if not os.path.isfile(cfg_file) or _stale(cfg_file, cfg):
        with open(cfg_file, "w") as f:
            yaml.safe_dump(cfg, f)
    os.environ["LIBERO_CONFIG_PATH"] = cfg_dir
    return True


def _ensure_imported():
    """Import LIBERO-plus's benchmark module (after fixing the config path)."""
    if not _ensure_config():
        raise RuntimeError(
            "LIBERO-plus is not importable in the active env. Install it in a "
            "dedicated conda env (it pins numpy<1.24 / robosuite / bddl / mujoco, "
            "which conflict with the default RoboVerse env):\n"
            "    git clone https://github.com/sylvestf/LIBERO-plus.git\n"
            "    cd LIBERO-plus && pip install -e .\n"
        )
    from libero.libero import benchmark

    return benchmark


def discover_liberoplus_tasks() -> dict[str, list[str]]:
    """Return ``{suite_name: [task_name, ...]}`` from LIBERO-plus's benchmark dict.

    Uses the authoritative ``benchmark.get_benchmark_dict()`` ordering (so task
    index ``i`` matches ``suite.get_task(i)`` / ``get_task_init_states(i)``).
    Returns ``{}`` if LIBERO-plus is not importable, keeping registration safe in
    envs without it.
    """
    try:
        benchmark = _ensure_imported()
    except Exception:
        return {}
    bdict = benchmark.get_benchmark_dict()
    out: dict[str, list[str]] = {}
    for name in LIBERO_PLUS_SUITES:
        if name not in bdict:
            continue
        try:
            suite = bdict[name]()
            out[name] = list(suite.get_task_names())
        except Exception as e:  # pragma: no cover - defensive
            log.debug(f"[liberoplus_passthrough] could not enumerate suite {name!r}: {e}")
    return out


def make_liberoplus_env(
    benchmark_name: str,
    task_id: int,
    *,
    camera_heights: int = 128,
    camera_widths: int = 128,
    seed: int | None = None,
    init_state_index: int | None = None,
    **env_kwargs: Any,
):
    """Build the native LIBERO-plus ``OffScreenRenderEnv`` for ``(suite, task_id)``.

    The canonical 1:1 construction path (identical to base LIBERO, the only
    difference being the LIBERO-plus config/BDDLs): resolve the perturbation
    task's BDDL file, build the offscreen env, optionally seed + reset + apply a
    stored init state. Returns the raw native env (legacy gym 4-tuple ``step``).

    Args:
        benchmark_name: One of :data:`LIBERO_PLUS_SUITES`.
        task_id: Task index within the (perturbation-expanded) suite.
        camera_heights: Offscreen RGB camera height (LIBERO default 128).
        camera_widths: Offscreen RGB camera width (LIBERO default 128).
        seed: If given, ``env.seed(seed)`` before reset.
        init_state_index: If given, ``env.set_init_state(init_states[idx])`` after
            reset, where ``init_states = suite.get_task_init_states(task_id)``.
        **env_kwargs: Extra kwargs forwarded to ``OffScreenRenderEnv``. A
            gym-injected ``render_mode`` is ignored (LIBERO is always offscreen).

    Returns:
        The native LIBERO-plus ``OffScreenRenderEnv`` instance.

    Raises:
        RuntimeError: If LIBERO-plus is not importable in the active env.
    """
    env_kwargs.pop("render_mode", None)  # gym.make may inject this; LIBERO is offscreen-only
    benchmark = _ensure_imported()
    from libero.libero.envs import OffScreenRenderEnv

    bdict = benchmark.get_benchmark_dict()
    if benchmark_name not in bdict:
        raise KeyError(f"Unknown LIBERO-plus suite {benchmark_name!r}; available: {sorted(bdict.keys())}")
    suite = bdict[benchmark_name]()
    n = suite.get_num_tasks()
    if not (0 <= task_id < n):
        raise IndexError(f"task_id {task_id} out of range for suite {benchmark_name!r} (n_tasks={n})")

    # The authoritative path the benchmark/eval feeds to OffScreenRenderEnv. For
    # perturbation tasks this is a *synthetic descriptor* (``..._view_..._initstate_
    # ..._noise_...``) that the env wrapper parses to recover the base BDDL plus the
    # camera/init-state/noise perturbation params -- NOT a real file. So we must not
    # os.path.isfile() it; the wrapper asserts the real base BDDL exists internally.
    bddl_file = suite.get_task_bddl_file_path(task_id)

    env_args = {"bddl_file_name": bddl_file, "camera_heights": camera_heights, "camera_widths": camera_widths}
    env_args.update(env_kwargs)
    env = OffScreenRenderEnv(**env_args)
    if seed is not None:
        env.seed(seed)
    env.reset()
    if init_state_index is not None:
        # LIBERO-plus's get_task_init_states has an upstream fall-through bug
        # (UnboundLocalError) for a few perturbation filename patterns that match
        # none of its _language_/_view_/_table_/_tb_/_light_/_add_/_level branches.
        # The perturbation itself is already baked into the BDDL descriptor at
        # construction (camera/robot/noise params), so when the stored init-state
        # snapshot is unavailable we keep the env's reset state instead of crashing.
        try:
            init_states = suite.get_task_init_states(task_id)
            env.set_init_state(init_states[init_state_index])
        except (UnboundLocalError, FileNotFoundError) as e:
            log.warning(
                f"[liberoplus_passthrough] init-state lookup unavailable for "
                f"{benchmark_name} task_id={task_id} ({type(e).__name__}); "
                f"using the BDDL reset state (perturbation still applied)."
            )
    return env


def make_liberoplus_env_from_bddl(
    bddl_file: str, *, camera_heights: int = 128, camera_widths: int = 128, **env_kwargs: Any
):
    """Build a native LIBERO-plus env directly from a BDDL file path.

    The perturbation (camera / lighting / texture / noise / layout) rides inside
    the BDDL, so this single factory covers any of the ~10k variants 1:1 by file.
    """
    env_kwargs.pop("render_mode", None)
    _ensure_imported()  # fixes config + raises a clear error if LIBERO-plus absent
    from libero.libero.envs import OffScreenRenderEnv

    env_args = {"bddl_file_name": bddl_file, "camera_heights": camera_heights, "camera_widths": camera_widths}
    env_args.update(env_kwargs)
    return OffScreenRenderEnv(**env_args)


def list_liberoplus_tasks(suite: str) -> list[str]:
    """Return the task names for a LIBERO-plus suite (``[]`` if unavailable)."""
    return discover_liberoplus_tasks().get(suite, [])


def suite_task_count(suite: str) -> int:
    """Return the number of tasks in a LIBERO-plus suite (0 if unavailable)."""
    return len(discover_liberoplus_tasks().get(suite, []))


def _make_liberoplus_env(benchmark_name: str, task_id: int, **kwargs: Any):
    """Gym entry-point shim (positional suite + task_id via registered kwargs)."""
    return make_liberoplus_env(benchmark_name, task_id, **kwargs)


def register_liberoplus_passthrough(prefix: str = "LiberoPlus/") -> list[str]:
    """Register every LIBERO-plus task as ``LiberoPlus/<suite>__<task>`` (lazy).

    Idempotent. Returns the registered ids. Safe when LIBERO-plus is absent
    (returns ``[]``). Honors ``LIBERO_PLUS_MAX_TASKS_PER_SUITE`` to cap the count
    for fast smoke imports (the make_* factories ignore the cap).
    """
    from gymnasium.envs.registration import register, registry

    tasks = discover_liberoplus_tasks()
    if not tasks:
        log.debug(
            "[liberoplus_passthrough] LIBERO-plus not importable; no tasks registered "
            "(ids become available once LIBERO-plus is installed in this env)"
        )
        return []

    cap = int(_MAX_PER_SUITE) if _MAX_PER_SUITE else None
    registered: list[str] = []
    failed: list[tuple[str, str]] = []
    for bench_name, task_names in tasks.items():
        names = task_names[:cap] if cap else task_names
        for task_id, task_name in enumerate(names):
            ns_id = f"{prefix}{bench_name}{_SEP}{task_name}"
            if ns_id in registry:
                registered.append(ns_id)
                continue
            try:
                register(
                    id=ns_id,
                    entry_point="roboverse_pack.tasks.libero_plus._passthrough:_make_liberoplus_env",
                    kwargs={"benchmark_name": bench_name, "task_id": task_id},
                    # Native LIBERO-plus env is legacy-gym (4-tuple step); keep gym's
                    # wrappers out of the way so the passthrough stays bitwise-1:1.
                    disable_env_checker=True,
                    order_enforce=False,
                )
                registered.append(ns_id)
            except Exception as e:
                failed.append((ns_id, str(e)))
    if failed:
        log.debug(f"[liberoplus_passthrough] {len(failed)} registration failures (first: {failed[0]})")
    log.info(f"[liberoplus_passthrough] registered {len(registered)} LIBERO-plus tasks under {prefix!r}")
    return registered


# Auto-register on import, mirroring the base-LIBERO passthrough. Safe no-op when
# LIBERO-plus is not installed (``discover_liberoplus_tasks`` returns ``{}``).
try:
    register_liberoplus_passthrough()
except Exception as _e:  # pragma: no cover - registration must never break import
    log.debug(f"[liberoplus_passthrough] auto-registration skipped: {_e}")
