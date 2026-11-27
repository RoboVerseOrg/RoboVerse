"""Language instructions for an evaluation: one canonical text, training paraphrases, held-out ones.

A policy that follows instructions should be scored on text it did not see in training, and two runs
of the same protocol must give every episode the same text. A :class:`PromptBank` holds a task's
instruction variants; :meth:`PromptBank.draw` picks one per episode from a hash of the task, the
episode and its seed — independent of every simulator random stream, so changing a backend or
``num_envs`` never changes which instruction an episode got.

The three splits (``EvalProtocol.prompt_split``):

* ``"canonical"`` — the task's own instruction, every episode (the default; what ``task_desc`` was).
* ``"train"`` — the canonical text with probability ``canonical_probability``, otherwise one of the
  training paraphrases. This is the distribution a policy is trained on.
* ``"heldout"`` — one of the paraphrases reserved for evaluation, never produced by ``"train"``.

A task declares its bank as a class attribute ``prompt_bank``; a task with only ``task_desc`` gets a
bank with that text alone, which supports ``"canonical"`` and ``"train"`` (both return it) and refuses
``"heldout"`` (there is nothing held out to measure).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from .protocol import PROMPT_SPLITS

DRAW_REVISION = "roboverse.bench.prompts/1"
"""Bumped whenever the draw changes, so a result says which sampler produced its prompts."""


@dataclass(frozen=True)
class PromptDraw:
    """The instruction one episode was given."""

    prompt: str
    variant: str
    """``"canonical"``, ``"train_<i>"`` or ``"heldout_<i>"`` — which entry of the bank it is."""
    split: str


@dataclass(frozen=True)
class PromptBank:
    """A task's instruction variants.

    Attributes:
        canonical: The instruction as the task states it.
        train: Paraphrases a policy may be trained on.
        heldout: Paraphrases reserved for evaluation; must not overlap ``train`` or ``canonical``.
        canonical_probability: Chance that a ``"train"`` draw returns ``canonical`` (default 0.2).
    """

    canonical: str
    train: tuple[str, ...] = ()
    heldout: tuple[str, ...] = ()
    canonical_probability: float = 0.2

    def __post_init__(self) -> None:
        if not isinstance(self.canonical, str) or not self.canonical.strip():
            raise ValueError("a prompt bank needs a non-empty canonical instruction")
        object.__setattr__(self, "train", tuple(self.train))
        object.__setattr__(self, "heldout", tuple(self.heldout))
        if not 0.0 <= self.canonical_probability <= 1.0:
            raise ValueError(f"canonical_probability must be in [0, 1], got {self.canonical_probability}")
        leaked = sorted(set(self.heldout) & (set(self.train) | {self.canonical}))
        if leaked:
            raise ValueError(f"held-out prompts also appear in the training set: {leaked}")
        if len(set(self.heldout)) != len(self.heldout) or len(set(self.train)) != len(self.train):
            raise ValueError("a prompt bank lists each paraphrase once")

    def draw(self, *, task: str, episode: int, seed: int, split: str = "canonical") -> PromptDraw:
        """The instruction of episode ``episode`` (seeded ``seed``) of ``task`` under ``split``."""
        if split not in PROMPT_SPLITS:
            raise ValueError(f"split must be one of {PROMPT_SPLITS}, got {split!r}")
        if split == "canonical":
            return PromptDraw(self.canonical, "canonical", split)
        key = json.dumps([DRAW_REVISION, str(task), int(episode), int(seed), split], separators=(",", ":"))
        raw = hashlib.sha256(key.encode()).digest()
        if split == "heldout":
            if not self.heldout:
                raise ValueError(
                    f"{task} has no held-out prompts; a held-out language evaluation needs a prompt_bank with "
                    "`heldout` paraphrases"
                )
            i = int.from_bytes(raw[8:16], "big") % len(self.heldout)
            return PromptDraw(self.heldout[i], f"heldout_{i:02d}", split)
        if not self.train or int.from_bytes(raw[:8], "big") / 2**64 < self.canonical_probability:
            return PromptDraw(self.canonical, "canonical", split)
        i = int.from_bytes(raw[8:16], "big") % len(self.train)
        return PromptDraw(self.train[i], f"train_{i:02d}", split)


def task_prompt_bank(env) -> PromptBank | None:
    """The env's ``prompt_bank``, else a bank of its ``task_desc`` alone, else ``None``."""
    bank = getattr(env, "prompt_bank", None)
    if bank is not None:
        if not isinstance(bank, PromptBank):
            raise TypeError(f"{type(env).__name__}.prompt_bank must be a metasim.bench.PromptBank, got {type(bank)}")
        return bank
    desc = getattr(env, "task_desc", None)
    if isinstance(desc, str) and desc.strip():
        return PromptBank(canonical=desc)
    return None
