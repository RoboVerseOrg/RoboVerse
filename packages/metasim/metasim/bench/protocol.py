"""The evaluation protocol: how a task's per-step success flags become one verdict per episode.

A task reports whether its success predicate holds *right now* (``info["success"]``, see
``BaseTaskEnv.success_is_termination``). Whether an episode counts as solved when the predicate
was true once, at the final step, or for ``hold_steps`` consecutive steps is a property of the
*benchmark*, declared here once and written into every result, instead of being re-decided by
each evaluation script.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Literal

SuccessRule = Literal["first", "final", "held"]
SUCCESS_RULES: tuple[str, ...] = ("first", "final", "held")
DEFAULT_EPISODES = 10


@dataclass(frozen=True)
class EvalProtocol:
    """What an evaluation run consists of and how it is scored.

    Attributes:
        episodes: Number of episodes to score (default 10). Instances are drawn round-robin from the
            task's instance pool (``metasim.bench.instances``) unless ``instances`` names them, in which
            case the count is theirs.
        success_rule: ``"first"`` — solved if the predicate held at any step; ``"final"`` — solved if
            it held at the last step of the episode; ``"held"`` — solved if it held for
            ``hold_steps`` consecutive steps at some point.
        hold_steps: Run length for ``"held"``; ignored by the other rules.
        max_episode_steps: Cap on the episode length enforced by the evaluator; ``None`` leaves it to
            the task's own ``max_episode_steps`` / ``_time_out``. A cap shorter than the task's makes
            the evaluator end the episode as a time-out.
        seed: Base seed; episode ``i`` uses ``seed + i``, so two runs of the same protocol on the
            same task draw the same instances in the same order.
        instances: Explicit instance indices into the task's pool, one per episode (overrides the
            round-robin draw and ``episodes`` must equal its length when both are given).
        name: Free label written into the result (e.g. ``"libero-50x3"``).
    """

    episodes: int | None = None
    """``None`` means 10, or the length of ``instances`` when those are given."""
    success_rule: SuccessRule = "first"
    hold_steps: int = 1
    max_episode_steps: int | None = None
    seed: int = 0
    instances: tuple[int, ...] | None = None
    name: str = "default"

    def __post_init__(self) -> None:
        if self.success_rule not in SUCCESS_RULES:
            raise ValueError(f"success_rule must be one of {SUCCESS_RULES}, got {self.success_rule!r}")
        if self.hold_steps < 1:
            raise ValueError(f"hold_steps must be >= 1, got {self.hold_steps}")
        if self.max_episode_steps is not None and self.max_episode_steps < 1:
            raise ValueError(f"max_episode_steps must be >= 1 or None, got {self.max_episode_steps}")
        if self.instances is not None:
            instances = tuple(int(i) for i in self.instances)
            if not instances:
                raise ValueError("instances must name at least one instance when given")
            if any(i < 0 for i in instances):
                raise ValueError(f"instance indices must be >= 0, got {instances}")
            object.__setattr__(self, "instances", instances)
            if self.episodes is None:
                object.__setattr__(self, "episodes", len(instances))
            elif self.episodes != len(instances):
                raise ValueError(
                    f"episodes={self.episodes} disagrees with {len(instances)} explicit instances; drop one"
                )
        elif self.episodes is None:
            object.__setattr__(self, "episodes", DEFAULT_EPISODES)
        if self.episodes < 1:
            raise ValueError(f"episodes must be >= 1, got {self.episodes}")

    def episode_success(self, flags: Sequence[bool]) -> bool:
        """Apply ``success_rule`` to the per-step success flags of one finished episode."""
        flags = [bool(f) for f in flags]
        if not flags:
            return False
        if self.success_rule == "first":
            return any(flags)
        if self.success_rule == "final":
            return flags[-1]
        run = 0
        for flag in flags:
            run = run + 1 if flag else 0
            if run >= self.hold_steps:
                return True
        return False

    def to_dict(self) -> dict:
        """JSON-ready form, written into every result."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> EvalProtocol:
        """Inverse of :meth:`to_dict`."""
        kwargs = dict(data)
        if kwargs.get("instances") is not None:
            kwargs["instances"] = tuple(kwargs["instances"])
        return cls(**kwargs)
