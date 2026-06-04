"""LIBERO passthrough — expose all 130 LIBERO tasks through RoboVerse.

LIBERO (*Lifelong Robot Learning*, Liu et al. 2023) tasks are robosuite / MuJoCo
environments defined by BDDL task files across five suites:
``libero_spatial`` (10), ``libero_object`` (10), ``libero_goal`` (10),
``libero_10`` (10, a.k.a. LIBERO-Long) and ``libero_90`` (90). Together they are
the 130-task LIBERO benchmark (``libero_90`` + ``libero_10`` = LIBERO-100).

This module registers each task under a ``Libero/<suite>__<task_name>`` gym id
with a *lazy* entry point: registration does NOT import LIBERO (so it is safe
even when LIBERO's heavy deps — ``robosuite==1.4.0`` / ``bddl==1.0.1`` /
``robomimic==0.2.0`` / ``numpy<1.24`` — are absent from the active env), and the
underlying env is only built when the env is actually made.

LIBERO must be importable in the active env. Because its pins (numpy<1.24,
robosuite, mujoco) conflict with the default RoboVerse env, install it in a
**dedicated** conda env::

    git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
    cd LIBERO && pip install -e .

Until then, registration registers nothing (the ids only become available once
LIBERO is importable) and :func:`make_libero_env` raises a clear, actionable
error. Once installed, the factory runs the **native** LIBERO env
(``OffScreenRenderEnv``) via the exact code path from LIBERO's README / lifelong
eval → **1:1 by construction**, no cross-sim porting.

Note:
    LIBERO uses the *legacy* gym step API — ``env.step(action)`` returns the
    4-tuple ``(obs, reward, done, info)`` and ``env.reset()`` returns ``obs``
    (not gymnasium's 5-tuple / ``(obs, info)``). This is preserved deliberately
    for bitwise fidelity with the original simulator; downstream 1:1 demo replay
    and the LIBERO benchmark eval both rely on the native interface. Prefer
    :func:`make_libero_env` (returns the raw native env) for replay / eval; the
    gym registration exists primarily for discovery (``list_passthrough_envs``).
"""

from __future__ import annotations

import os
from typing import Any

from loguru import logger as log

#: The five standard LIBERO suites, in canonical order (name -> n_tasks shown for
#: reference only; the authoritative counts come from the benchmark dict).
LIBERO_SUITES: tuple[str, ...] = (
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_10",
    "libero_90",
)

#: Separator between suite and task name inside a gym id. A single ``/`` already
#: denotes the gym namespace, so the suite/task split uses ``__`` (both halves
#: stay valid ``[\w:-]+`` gym-name characters and the split is reversible).
_SEP = "__"


def _ensure_imported():
    """Import LIBERO's benchmark module (raises if LIBERO is not installed)."""
    import libero  # noqa: F401
    from libero.libero import benchmark

    return benchmark


def discover_libero_tasks() -> dict[str, list[str]]:
    """Return ``{suite_name: [task_name, ...]}`` from LIBERO's benchmark dict.

    Uses the authoritative ``benchmark.get_benchmark_dict()`` ordering (so task
    index ``i`` matches ``suite.get_task(i)`` / ``get_task_init_states(i)`` /
    ``get_task_demonstration(i)``). Returns ``{}`` if LIBERO is not importable,
    which keeps :func:`register_libero_passthrough` safe in envs without LIBERO.
    """
    try:
        benchmark = _ensure_imported()
    except Exception:
        return {}
    bdict = benchmark.get_benchmark_dict()
    out: dict[str, list[str]] = {}
    for name in LIBERO_SUITES:
        if name not in bdict:
            continue
        try:
            suite = bdict[name]()
            out[name] = list(suite.get_task_names())
        except Exception as e:  # pragma: no cover - defensive
            log.debug(f"[libero_passthrough] could not enumerate suite {name!r}: {e}")
    return out


def make_libero_env(
    benchmark_name: str,
    task_id: int,
    *,
    camera_heights: int = 128,
    camera_widths: int = 128,
    seed: int | None = None,
    init_state_index: int | None = None,
    **env_kwargs: Any,
):
    """Build the native LIBERO ``OffScreenRenderEnv`` for ``(suite, task_id)``.

    This is the canonical 1:1 construction path (matches LIBERO's README and
    ``libero/lifelong`` eval): resolve the task's BDDL file, build the offscreen
    env, optionally seed + reset + apply a stored init state. Returns the raw
    native env (legacy gym 4-tuple ``step`` API).

    Args:
        benchmark_name: One of :data:`LIBERO_SUITES` (e.g. ``"libero_spatial"``).
        task_id: Task index within the suite (``0 <= task_id < suite.n_tasks``).
        camera_heights: Offscreen RGB camera height (LIBERO default 128).
        camera_widths: Offscreen RGB camera width (LIBERO default 128).
        seed: If given, ``env.seed(seed)`` before reset (LIBERO determinism).
        init_state_index: If given, ``env.set_init_state(init_states[idx])`` after
            reset, where ``init_states = suite.get_task_init_states(task_id)`` —
            the exact init-state path the benchmark eval uses.
        **env_kwargs: Extra kwargs forwarded to ``OffScreenRenderEnv`` (e.g.
            ``camera_depths=True``). A gym-injected ``render_mode`` is ignored
            (LIBERO is always offscreen).

    Returns:
        The native LIBERO ``OffScreenRenderEnv`` instance.

    Raises:
        RuntimeError: If LIBERO is not importable in the active env.
    """
    env_kwargs.pop("render_mode", None)  # gym.make may inject this; LIBERO is offscreen-only
    try:
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
    except Exception as e:
        raise RuntimeError(
            "LIBERO is not importable in the active env. Install it in a dedicated conda env "
            "(it pins numpy<1.24 / robosuite==1.4.0 / bddl==1.0.1 / robomimic==0.2.0, which "
            "conflict with the default RoboVerse env):\n"
            "    git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git\n"
            "    cd LIBERO && pip install -e .\n"
            f"Original import error: {type(e).__name__}: {e}"
        ) from e

    bdict = benchmark.get_benchmark_dict()
    if benchmark_name not in bdict:
        raise KeyError(f"Unknown LIBERO suite {benchmark_name!r}; available: {sorted(bdict.keys())}")
    suite = bdict[benchmark_name]()
    n = suite.get_num_tasks()
    if not (0 <= task_id < n):
        raise IndexError(f"task_id {task_id} out of range for suite {benchmark_name!r} (n_tasks={n})")

    task = suite.get_task(task_id)
    bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    if not os.path.isfile(bddl_file):
        raise FileNotFoundError(f"BDDL file for {benchmark_name}/{task.name} not found: {bddl_file}")

    env_args = {"bddl_file_name": bddl_file, "camera_heights": camera_heights, "camera_widths": camera_widths}
    env_args.update(env_kwargs)
    env = OffScreenRenderEnv(**env_args)
    if seed is not None:
        env.seed(seed)
    env.reset()
    if init_state_index is not None:
        init_states = suite.get_task_init_states(task_id)
        env.set_init_state(init_states[init_state_index])
    return env


def _make_libero_env(benchmark_name: str, task_id: int, **kwargs: Any):
    """Gym entry-point shim (positional-suite + task_id via registered kwargs)."""
    return make_libero_env(benchmark_name, task_id, **kwargs)


def register_libero_passthrough(prefix: str = "Libero/") -> list[str]:
    """Register every LIBERO task as ``Libero/<suite>__<task_name>`` (lazy).

    Idempotent. Returns the registered ids. Safe when LIBERO is absent (returns
    ``[]`` — the ids only become discoverable once LIBERO is importable, since
    the suite/task enumeration is authoritative and comes from LIBERO itself).
    """
    from gymnasium.envs.registration import register, registry

    tasks = discover_libero_tasks()
    if not tasks:
        log.debug(
            "[libero_passthrough] LIBERO not importable; no tasks registered "
            "(ids become available once LIBERO is installed in this env)"
        )
        return []

    registered: list[str] = []
    failed: list[tuple[str, str]] = []
    for bench_name, task_names in tasks.items():
        for task_id, task_name in enumerate(task_names):
            ns_id = f"{prefix}{bench_name}{_SEP}{task_name}"
            if ns_id in registry:
                registered.append(ns_id)
                continue
            try:
                register(
                    id=ns_id,
                    entry_point="roboverse_pack.tasks.libero._passthrough:_make_libero_env",
                    kwargs={"benchmark_name": bench_name, "task_id": task_id},
                    # The native LIBERO env is legacy-gym (4-tuple step); keep gym's
                    # wrappers out of the way so the passthrough stays bitwise-1:1.
                    disable_env_checker=True,
                    order_enforce=False,
                )
                registered.append(ns_id)
            except Exception as e:
                failed.append((ns_id, str(e)))
    if failed:
        log.debug(f"[libero_passthrough] {len(failed)} registration failures (first: {failed[0]})")
    log.info(f"[libero_passthrough] registered {len(registered)} LIBERO tasks under {prefix!r}")
    return registered


# Registering on import mirrors the other ``roboverse_pack.tasks.*`` passthroughs.
# The ``libero`` package ``__init__`` auto-imports every submodule, so importing
# ``roboverse_pack.tasks.libero`` triggers this. It is a safe no-op when LIBERO is
# not installed (``discover_libero_tasks`` returns ``{}``).
try:
    register_libero_passthrough()
except Exception as _e:  # pragma: no cover - registration must never break import
    log.debug(f"[libero_passthrough] auto-registration skipped: {_e}")
