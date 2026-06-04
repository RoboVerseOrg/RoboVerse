"""SimplerEnv passthrough — expose SimplerEnv real-to-sim eval tasks under SimplerEnv/<task>.

Wraps the native `SimplerEnv <https://github.com/simpler-env/SimplerEnv>`_
environments behind the gymnasium API. The wrapper is a *transparent* 1:1
passthrough: it constructs each env with ``simpler_env.make(task_name)`` (the
exact upstream entry point, which fixes ``obs_mode="rgbd"`` and
``prepackaged_config=True`` for the visual-matching evaluation) and forwards
``reset`` / ``step`` verbatim. The observation dict, reward, success flag
(``info["success"]``), per-condition ``info["episode_stats"]``, language
instruction and action space are passed through unchanged — running the original
ManiSkill2-real2sim + SAPIEN physics, not a MetaSim re-implementation.

Registration is import-safe even when SimplerEnv's deps are absent: only
``simpler_env.ENVIRONMENTS`` (a pure-Python list) is read at registration time,
and the heavy env is built lazily inside the entry point. SimplerEnv's stack
(numpy 1.24 / SAPIEN 2.x / the ``mani_skill2_real2sim`` fork) conflicts with the
default RoboVerse env, so install it in a dedicated conda env; until then
``gym.make("SimplerEnv/...")`` raises a clear error while the ids stay
discoverable.

Note: SimplerEnv/SAPIEN keep process-global renderer state, so — exactly as with
native SimplerEnv — only one env should be alive per process at a time. This is a
property of the underlying simulator, not of the wrapper (which forwards verbatim).
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
from loguru import logger as log


def register_simpler_env_passthrough(prefix: str = "SimplerEnv/") -> list[str]:
    """Register all SimplerEnv tasks under the ``SimplerEnv/<task>`` namespace.

    The task list is taken from ``simpler_env.ENVIRONMENTS`` (the authoritative
    upstream list), so the registered ids track the installed SimplerEnv version
    1:1. Returns the list of registered env ids; safe to call repeatedly and safe
    when SimplerEnv is not installed (returns ``[]``).
    """
    from gymnasium.envs.registration import register, registry

    try:
        import simpler_env
    except Exception as e:  # optional dependency; absent in the default env
        log.debug(f"[simpler_env_passthrough] simpler_env not importable: {e}")
        return []

    task_names = list(getattr(simpler_env, "ENVIRONMENTS", []))
    registered = []
    failed = []
    for name in task_names:
        ns_id = f"{prefix}{name}"
        if ns_id in registry:
            registered.append(ns_id)
            continue
        try:
            register(
                id=ns_id,
                entry_point="roboverse_pack.tasks.simpler_env._passthrough:_make_simpler_env",
                kwargs={"task_name": name},
                # Forward the upstream env verbatim: do not impose gymnasium's
                # passive env checker / order-enforcing wrapper on it. Several
                # SimplerEnv (ManiSkill2-real2sim) tasks — e.g. the drawer/place
                # family — build their Dict observation space lazily and expose an
                # empty Dict before the first reset, which the checker rejects with
                # "An empty Dict observation space is not allowed". The native
                # simpler_env.make path is unchecked too, so skipping the check
                # here keeps the passthrough a faithful 1:1 mirror.
                disable_env_checker=True,
                order_enforce=False,
            )
            registered.append(ns_id)
        except Exception as e:
            failed.append((ns_id, str(e)))
    if failed:
        log.warning(f"[simpler_env_passthrough] {len(failed)} failures")
    log.info(f"[simpler_env_passthrough] registered {len(registered)} SimplerEnv tasks under {prefix!r}")
    return registered


def _make_simpler_env(task_name: str, **kwargs: Any):
    return _SimplerEnvGymWrapper(task_name, **kwargs)


class _SimplerEnvGymWrapper(gym.Env):
    """Minimal SimplerEnv -> gymnasium adapter (1:1 passthrough).

    Delegates construction to ``simpler_env.make`` and forwards every call to the
    underlying env. SimplerEnv-specific helpers used by eval loops
    (``get_language_instruction``, ``advance_to_next_subtask``,
    ``is_final_subtask``) are exposed explicitly; any other attribute access falls
    through to the wrapped env via :meth:`__getattr__`.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, task_name: str, **kwargs: Any) -> None:
        super().__init__()
        try:
            import simpler_env
        except Exception as e:
            raise RuntimeError(
                f"Cannot make 'SimplerEnv/{task_name}': simpler_env is not installed in this env "
                f"({type(e).__name__}: {e}). SimplerEnv needs numpy 1.24 / SAPIEN 2.x / the "
                f"mani_skill2_real2sim fork, which conflict with the default RoboVerse env — install "
                f"it in a dedicated conda env (see https://github.com/simpler-env/SimplerEnv)."
            ) from e

        self.task_name = task_name
        # simpler_env.make() applies the canonical obs_mode / prepackaged_config and
        # the per-task gym id + kwargs from simpler_env.ENVIRONMENT_MAP. Using it
        # directly guarantees the env is byte-for-byte the upstream eval env.
        self._env = simpler_env.make(task_name, **kwargs)
        self.observation_space = getattr(self._env, "observation_space", None)
        self.action_space = getattr(self._env, "action_space", None)

    # -- core gymnasium API: pass through verbatim -----------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        # ManiSkill2-real2sim is built on classic gym whose reset() may not accept
        # seed/options; fall back so we never alter the env's own reset semantics.
        try:
            out = self._env.reset(seed=seed, options=options)
        except TypeError:
            if seed is not None and hasattr(self._env, "seed"):
                self._env.seed(seed)
            out = self._env.reset()
        if isinstance(out, tuple) and len(out) == 2 and isinstance(out[1], dict):
            return out  # (obs, info) — gymnasium convention, unchanged
        return out, {}  # old-gym reset returned obs only

    def step(self, action):
        out = self._env.step(action)
        if len(out) == 5:
            return out  # (obs, reward, terminated, truncated, info) — unchanged
        # old-gym 4-tuple (obs, reward, done, info): split done without touching
        # obs / reward / info, so success and stats stay bitwise identical.
        obs, reward, done, info = out
        return obs, reward, bool(done), False, info

    def render(self):
        return self._env.render()

    def close(self):
        return self._env.close()

    # -- SimplerEnv eval-loop helpers (preserved for 1:1 evaluation) -----------

    def get_language_instruction(self, *args, **kwargs):
        return self._env.get_language_instruction(*args, **kwargs)

    def advance_to_next_subtask(self, *args, **kwargs):
        return self._env.advance_to_next_subtask(*args, **kwargs)

    def is_final_subtask(self, *args, **kwargs):
        return self._env.is_final_subtask(*args, **kwargs)

    def __getattr__(self, name: str):
        # Only called for attributes not found normally; delegate to wrapped env.
        # ``_env`` is set in __init__, so guard against recursion before it exists.
        if name == "_env":
            raise AttributeError(name)
        return getattr(self._env, name)
