"""MetaSim-native LIBERO tasks — run a LIBERO scene + checker with NO ``libero`` import.

A self-contained bundle (compiled ``model.mjb`` + init states + resolved BDDL goal)
is loaded into MetaSim's MuJoCo handler, driven by the ported OSC_POSE controller,
and evaluated by the ported BDDL checker (:mod:`.checker`). This is the path to
running LIBERO without the upstream library.
"""

from __future__ import annotations
