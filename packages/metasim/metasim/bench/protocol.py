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
PromptSplit = Literal["canonical", "train", "heldout"]
PROMPT_SPLITS: tuple[str, ...] = ("canonical", "train", "heldout")
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
        demo_budget_factor: Derive the episode cap from the task's demos: ``ceil(factor x`` the
            ``demo_budget_quantile`` quantile of the demo lengths ``)``. A budget that scales with how long
            the task takes a demonstrator keeps a slow-but-correct policy from timing out on long tasks
            and a dithering one from being given 10x the time on short ones. ``None`` (default) leaves the
            cap to ``max_episode_steps`` / the task; the two cannot both be set. A task without demos
            falls back to its own cap, and the result's notes say which cap was applied.
        demo_budget_quantile: Quantile of the demo lengths the budget is scaled from (default 0.75).
        prompt_split: Which instruction each episode gets from the task's prompt bank
            (``metasim.bench.prompts``): ``"canonical"`` (default) the task's own instruction,
            ``"train"`` the canonical one or a training paraphrase drawn per episode, ``"heldout"`` a
            paraphrase reserved for evaluation, so language generalisation is measured on text a
            policy was not trained on.
    """

    episodes: int | None = None
    """``None`` means 10, or the length of ``instances`` when those are given."""
    success_rule: SuccessRule = "first"
    hold_steps: int = 1
    max_episode_steps: int | None = None
    seed: int = 0
    instances: tuple[int, ...] | None = None
    name: str = "default"
    demo_budget_factor: float | None = None
    demo_budget_quantile: float = 0.75
    prompt_split: PromptSplit = "canonical"

    def __post_init__(self) -> None:
        if self.success_rule not in SUCCESS_RULES:
            raise ValueError(f"success_rule must be one of {SUCCESS_RULES}, got {self.success_rule!r}")
        if self.prompt_split not in PROMPT_SPLITS:
            raise ValueError(f"prompt_split must be one of {PROMPT_SPLITS}, got {self.prompt_split!r}")
        if self.demo_budget_factor is not None:
            if not self.demo_budget_factor > 0:
                raise ValueError(f"demo_budget_factor must be > 0 or None, got {self.demo_budget_factor}")
            if self.max_episode_steps is not None:
                raise ValueError("set either max_episode_steps or demo_budget_factor, not both")
        if not 0.0 < self.demo_budget_quantile <= 1.0:
            raise ValueError(f"demo_budget_quantile must be in (0, 1], got {self.demo_budget_quantile}")
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
