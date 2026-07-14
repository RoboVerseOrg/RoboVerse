"""RoboVerse unified policy-evaluation harness — typed, embodiment-general core.

The pure, GPU-free core is embodiment inference, typed obs/action specs with
connect-time negotiation, torch-native batched carriers, and unified action chunking /
temporal ensembling. On top of that sit the env adapter, the vectorized runner, the
transports (in-process and WebSocket), and multi-sim parity eval. See ``../DESIGN.md``.
"""

from __future__ import annotations

from typing import Any

from .chunking import ActionChunk, ChunkScheduler, TemporalEnsembler
from .embodiment import Chain, ChainKind, Embodiment, EmbodimentHints, infer_embodiment
from .obs import ActionBatch, ObsBatch
from .policy import Policy, PolicyCard
from .spec import (
    ActionSpec,
    FieldSpec,
    ObsSpec,
    Space,
    SpecMatch,
    derive_action_spec,
    derive_obs_spec,
)

# Heavy names (pull metasim / a simulator backend) are lazy so importing the pure Phase-0
# core (embodiment/spec/chunking) stays light. The eval entry point lives in ``_evaluate``
# (underscored so no submodule named ``evaluate`` shadows the lazily-exposed ``evaluate``
# function — a plain ``evaluate.py`` would be bound by ``from harness import evaluate``
# before ``__getattr__`` runs, returning the module instead of the function).
_LAZY = {
    "evaluate": ("_evaluate", "evaluate"),
    "EvalResult": ("_evaluate", "EvalResult"),
    "ParityReport": ("_evaluate", "ParityReport"),
    "EnvAdapter": ("env_adapter", "EnvAdapter"),
    "VecEvalRunner": ("runner", "VecEvalRunner"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        mod, attr = _LAZY[name]
        import importlib

        return getattr(importlib.import_module(f"{__name__}.{mod}"), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ActionBatch",
    "ActionChunk",
    "ActionSpec",
    "Chain",
    "ChainKind",
    "ChunkScheduler",
    "Embodiment",
    "EmbodimentHints",
    "EnvAdapter",
    "EvalResult",
    "FieldSpec",
    "ObsBatch",
    "ObsSpec",
    "ParityReport",
    "Policy",
    "PolicyCard",
    "Space",
    "SpecMatch",
    "TemporalEnsembler",
    "VecEvalRunner",
    "derive_action_spec",
    "derive_obs_spec",
    "evaluate",
    "infer_embodiment",
]
