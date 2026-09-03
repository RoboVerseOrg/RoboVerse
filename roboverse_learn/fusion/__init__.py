"""RoboVerse IL+RL fusion layer.

A thin, additive bridge that lets the reinforcement-learning (``roboverse_learn/rl``)
and imitation-learning (``roboverse_learn/il``) halves of RoboVerse share one
task/env and one demonstration format, closing the loop in both directions:

* **RL -> IL** (:mod:`roboverse_learn.fusion.rl_to_demo`): roll out a trained
  rsl-rl policy in a RoboVerse task and write successful episodes in the *exact*
  canonical demo layout (``success/demo_XXXX/{metadata.json,rgb.mp4}`` via
  :func:`metasim.utils.save_util.save_demo`). The existing
  ``roboverse_learn/il/data2zarr_dp.py`` + ``il/train.py`` then consume them
  unchanged — no new dataset format is introduced.

* **IL -> RL** (:mod:`roboverse_learn.fusion.bc_warmstart`): copy the weights of
  a behaviour-cloning MLP actor into an rsl-rl ``ActorCritic`` (or extract the
  RL actor as a BC-initialisable MLP), so a policy pre-trained by IL can be
  fine-tuned by RL and vice-versa.

Design rules (see ``AGENTS.md``): everything here is additive and reuses the
existing entry points; nothing in ``rl/`` or ``il/`` is modified. Heavy imports
(simulator backends, rsl-rl) stay lazy so importing this package is cheap.
"""

from __future__ import annotations

__all__ = [
    "collect_demos_from_policy",
    "extract_actor_mlp_state_dict",
    "load_bc_into_actor_critic",
]


def __getattr__(name: str):  # lazy re-export to avoid importing sim/rsl-rl eagerly
    if name == "collect_demos_from_policy":
        from roboverse_learn.fusion.rl_to_demo import collect_demos_from_policy

        return collect_demos_from_policy
    if name in ("load_bc_into_actor_critic", "extract_actor_mlp_state_dict"):
        from roboverse_learn.fusion import bc_warmstart

        return getattr(bc_warmstart, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
