"""Information-access tiers: what a policy may see, call and spend, declared once and written into every result.

Two policies are only comparable when they were evaluated with the same access to information. A VLA that
sees two RGB cameras and its own joints competes in a different class from an agent that may query object
poses from the simulator or call a motion planner. :class:`AccessProfile` names that class. The evaluator
enforces the parts it controls (which observation reaches the policy, the wall-clock budget of an episode),
records the parts only the policy knows (tool and model usage, via ``policy.usage()``), and writes the
profile into the result so leaderboards partition on it instead of mixing tiers in one column.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal

Observation = Literal["privileged", "rgb_depth", "rgb"]
OBSERVATIONS: tuple[str, ...] = ("privileged", "rgb_depth", "rgb")
"""What the harness hands the policy: the full simulator state, RGB + depth images, or RGB only."""

Tools = Literal["none", "skills", "planner", "sim_queries"]
TOOLS: tuple[str, ...] = ("none", "skills", "planner", "sim_queries")
"""The most privileged tool an agent may call: nothing beyond its action, scripted skills (reach / grasp /
place primitives), a motion planner, or the simulator's own queries (object poses, contacts)."""

Memory = Literal["episode", "persistent"]
MEMORIES: tuple[str, ...] = ("episode", "persistent")
"""Whether the policy may carry state from one episode into the next."""

USAGE_FIELDS: tuple[str, ...] = ("llm_calls", "llm_tokens", "tool_calls")
"""Counters a policy may report from ``usage()``; budgets are per episode, usage is pooled over the run."""


@dataclass(frozen=True)
class AccessProfile:
    """One information-access tier.

    Attributes:
        name: The tier id written into results and used to partition leaderboards.
        observation: What the policy observes (:data:`OBSERVATIONS`). The evaluator enforces it: below
            ``"privileged"`` the object states and the ``extras`` are removed, segmentation masks are removed,
            and depth is kept only under ``"rgb_depth"``.
        cameras: The cameras the policy may see (``None`` = every camera the task configures). A name the
            task does not render is an error, not a silently empty view.
        proprioception: Whether the robots' own state (joints, root, bodies, targets) is observed.
        tools: The most privileged tool the policy may call (:data:`TOOLS`). Declared and recorded; the
            harness cannot see inside a policy, so an adapter is expected to honour it.
        memory: ``"episode"`` — the policy starts every episode from scratch; ``"persistent"`` — it may keep
            what it learned across episodes of the run (:data:`MEMORIES`).
        wall_time_s: Wall-clock budget of one episode; an episode over it ends as a time-out (enforced).
        llm_calls / llm_tokens / tool_calls: Per-episode budgets on the counters a policy reports through
            ``usage()``; the run's mean per episode is compared and an overrun is recorded in the result.
    """

    name: str = "privileged"
    observation: Observation = "privileged"
    cameras: tuple[str, ...] | None = None
    proprioception: bool = True
    tools: Tools = "sim_queries"
    memory: Memory = "persistent"
    wall_time_s: float | None = None
    llm_calls: int | None = None
    llm_tokens: int | None = None
    tool_calls: int | None = None

    def __post_init__(self) -> None:
        if not self.name or not isinstance(self.name, str):
            raise ValueError("an access profile needs a name")
        if self.observation not in OBSERVATIONS:
            raise ValueError(f"observation must be one of {OBSERVATIONS}, got {self.observation!r}")
        if self.tools not in TOOLS:
            raise ValueError(f"tools must be one of {TOOLS}, got {self.tools!r}")
        if self.memory not in MEMORIES:
            raise ValueError(f"memory must be one of {MEMORIES}, got {self.memory!r}")
        if self.cameras is not None:
            cameras = tuple(str(c) for c in self.cameras)
            if not cameras:
                raise ValueError("cameras must name at least one camera, or be None for all of them")
            object.__setattr__(self, "cameras", cameras)
        if self.wall_time_s is not None and not self.wall_time_s > 0:
            raise ValueError(f"wall_time_s must be > 0 or None, got {self.wall_time_s}")
        for name in USAGE_FIELDS:
            value = getattr(self, name)
            if value is not None and (not isinstance(value, int) or value < 0):
                raise ValueError(f"{name} must be a non-negative int or None, got {value!r}")

    @property
    def restricts_observation(self) -> bool:
        """Whether the harness has to filter the observation before the policy sees it."""
        return self.observation != "privileged" or self.cameras is not None or not self.proprioception

    @property
    def budgets(self) -> dict[str, int]:
        """The per-episode usage budgets that are set."""
        return {name: getattr(self, name) for name in USAGE_FIELDS if getattr(self, name) is not None}

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form, written into every result's protocol."""
        data = asdict(self)
        if data["cameras"] is not None:
            data["cameras"] = list(data["cameras"])
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AccessProfile:
        """Inverse of :meth:`to_dict`; an unknown field or a ``cameras`` entry that is not a list is an error."""
        kwargs = dict(data)
        unknown = sorted(set(kwargs) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(
                f"access profile has unknown fields {unknown}; the fields are {list(cls.__dataclass_fields__)}"
            )
        cameras = kwargs.get("cameras")
        if cameras is not None:
            if isinstance(cameras, (str, bytes)) or not isinstance(cameras, (list, tuple)):
                raise ValueError(f"cameras must be a list of camera names or null, got {cameras!r}")
            kwargs["cameras"] = tuple(cameras)
        return cls(**kwargs)


ACCESS_PROFILES: dict[str, AccessProfile] = {
    "privileged": AccessProfile(),
    "visual": AccessProfile(name="visual", observation="rgb", tools="none", memory="episode"),
    "visual-depth": AccessProfile(name="visual-depth", observation="rgb_depth", tools="none", memory="episode"),
    "agent-skills": AccessProfile(name="agent-skills", observation="rgb", tools="skills", memory="episode"),
    "agent-planner": AccessProfile(name="agent-planner", observation="rgb_depth", tools="planner", memory="episode"),
    "agent-privileged": AccessProfile(
        name="agent-privileged", observation="privileged", tools="sim_queries", memory="episode"
    ),
}
"""The named tiers: ``privileged`` is today's unrestricted evaluation (the default); ``visual`` and
``visual-depth`` are the VLA / imitation tiers; the ``agent-*`` tiers grade what an LLM agent may call."""


def get_access_profile(spec: str | AccessProfile | None) -> AccessProfile:
    """A profile by preset name, from a JSON file, or passed through; ``None`` is the default tier.

    A file must carry a ``name`` (results are compared by the whole profile, but the name labels them)
    and may not reuse a preset's name for a different profile.
    """
    if spec is None:
        return ACCESS_PROFILES["privileged"]
    if isinstance(spec, AccessProfile):
        return spec
    if spec in ACCESS_PROFILES:
        return ACCESS_PROFILES[spec]
    path = Path(spec)
    if path.suffix == ".json" and path.exists():
        with open(path) as f:
            data = json.load(f)
        if not isinstance(data, dict) or not data.get("name"):
            raise ValueError(f'{path}: an access profile file is a JSON object with a "name" field')
        profile = AccessProfile.from_dict(data)
        preset = ACCESS_PROFILES.get(profile.name)
        if preset is not None and profile != preset:
            raise ValueError(
                f"{path} names the preset tier {profile.name!r} but differs from it; give the profile its own name"
            )
        return profile
    raise ValueError(f"unknown access profile {spec!r}; one of {sorted(ACCESS_PROFILES)} or a JSON file")


def restrict_observation(obs: Any, access: AccessProfile) -> Any:
    """The part of ``obs`` the tier allows; ``obs`` itself when nothing is restricted.

    Works on a :class:`~metasim.types.TensorState`: object states and ``extras`` go below ``"privileged"``,
    segmentation masks go too, depth stays only under ``"rgb_depth"``, cameras are limited to
    ``access.cameras`` and the robots' state to ``access.proprioception``. An observation that is not a
    ``TensorState`` (an RL env's flat tensor) cannot be filtered field by field; restricting it is an error
    rather than a silently unrestricted run, and so is an image tier on a task that renders no camera.
    """
    if not access.restricts_observation:
        return obs
    from metasim.types import TensorState

    if not isinstance(obs, TensorState):
        raise TypeError(
            f"access profile {access.name!r} restricts the observation, but this env observes a "
            f"{type(obs).__name__}, not a TensorState; the tier cannot be enforced on it (use 'privileged')"
        )
    if access.cameras is not None:
        missing = [c for c in access.cameras if c not in obs.cameras]
        if missing:
            raise ValueError(f"access profile {access.name!r} names cameras the task does not render: {missing}")
    cameras = {}
    for name, cam in obs.cameras.items():
        if access.cameras is not None and name not in access.cameras:
            continue
        if access.observation == "privileged":
            cameras[name] = cam
        else:
            cameras[name] = replace(
                cam,
                depth=cam.depth if access.observation == "rgb_depth" else None,
                instance_id_seg=None,
                instance_seg=None,
                instance_id_seg_id2label=None,
                instance_seg_id2label=None,
            )
    if access.observation != "privileged" and not cameras:
        raise ValueError(
            f"access profile {access.name!r} observes images only, but this task renders no camera; add cameras "
            "to the task's scenario or evaluate it under 'privileged'"
        )
    robots = dict(obs.robots) if access.proprioception else {}
    if access.observation == "privileged":
        return replace(obs, robots=robots, cameras=cameras)
    return TensorState(objects={}, robots=robots, cameras=cameras, extras={})


def usage_report(usage: dict[str, Any] | None, access: AccessProfile, episodes: int) -> dict[str, Any] | None:
    """Pooled usage against the per-episode budgets: ``{"per_episode": {...}, "exceeded": [...]}``, or ``None``."""
    budgets = access.budgets
    if not usage and not budgets:
        return None
    usage = dict(usage or {})
    per_episode = {k: (float(v) / episodes if episodes else None) for k, v in usage.items() if k in USAGE_FIELDS}
    exceeded = [k for k, budget in budgets.items() if per_episode.get(k) is not None and per_episode[k] > budget]
    unreported = [k for k in budgets if k not in usage]
    report: dict[str, Any] = {"total": usage, "per_episode": per_episode, "budgets": budgets, "exceeded": exceeded}
    if unreported:
        report["unreported"] = unreported
    return report
