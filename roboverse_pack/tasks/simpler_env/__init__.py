"""SimplerEnv — real-to-sim RT-1/Octo eval tasks (Google Robot + WidowX/Bridge).

``gymnasium.make("SimplerEnv/<task>")`` now resolves to the **MetaSim-native** task
(``_metasim/`` — a ``BaseTaskEnv`` whose ``ScenarioCfg`` is loaded by the MetaSim ``sapien2``
handler; the SimplerEnv-specific EE controller / overlay / checker layer on the handler's objects).
**Zero dependency on the cloned ``simpler_env`` / ``mani_skill2_real2sim`` packages**, and reads
only ``roboverse_data`` assets. Each MetaSim task reproduces its standalone ``_native`` counterpart
bitwise / machine-eps (which is itself verified vs upstream), so the upstream clone can be removed
and these ids keep working. The ``_native/`` modules remain as the vendored SimplerEnv-specific
logic library (controllers, overlay, grasp/success checkers) that the MetaSim tasks reuse.

The transparent 1:1 passthrough (which forwards to ``simpler_env.make`` for bitwise comparison
against upstream) is still registered, but under the ``SimplerEnvPassthrough/`` namespace and only
when the clone happens to be installed — it is no longer required for ``SimplerEnv/`` to work.
"""

from __future__ import annotations

# Primary: MetaSim-native tasks (ScenarioCfg + sapien2 handler), zero upstream dependency.
try:
    from roboverse_pack.tasks.simpler_env._metasim.registry import register_metasim_simpler_tasks

    register_metasim_simpler_tasks(prefix="SimplerEnv/")
except Exception:  # pragma: no cover - native registration should not fail
    pass

# Secondary (optional): upstream passthrough for bitwise comparison, only if the clone is present.
try:
    from roboverse_pack.tasks.simpler_env._passthrough import register_simpler_env_passthrough

    register_simpler_env_passthrough(prefix="SimplerEnvPassthrough/")
except Exception:
    pass
