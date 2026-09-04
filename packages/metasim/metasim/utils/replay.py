"""Record → replay fidelity checks for simulator handlers.

"Replayable" has three levels, each a contract a backend either meets or fails loudly:

* **L0 — action replay**: from the same initial state, replaying the same action sequence on the same
  backend reproduces the recorded states (bounded by floating-point noise).
* **L1 — state-anchored replay**: writing any recorded state back with ``set_states`` reproduces it
  exactly on read-back (positions *and* velocities), and one step from it reproduces the recorded next
  state. This is what checkpoints, failure rollback and data augmentation need.
* **L2 — cross-backend replay**: the same actions on another backend; never expected to match, only
  measured (per-step drift after anchoring).

:func:`record` captures a trajectory of full ``TensorState`` snapshots; :func:`verify_action_replay`
and :func:`verify_state_replay` return a :class:`ReplayReport` with the worst deviation and where it
happened, so a test or a data-ingest gate can assert on it and a human can read it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from metasim.types import TensorState


def _flatten(state: TensorState) -> dict[str, torch.Tensor]:
    """Every physical quantity of a ``TensorState`` as ``name -> tensor`` (env 0 … N), detached on CPU."""
    out: dict[str, torch.Tensor] = {}
    for kind in ("robots", "objects"):
        for name, st in getattr(state, kind).items():
            out[f"{kind}/{name}/root"] = st.root_state.detach().cpu().double()
            if getattr(st, "joint_pos", None) is not None:
                out[f"{kind}/{name}/q"] = st.joint_pos.detach().cpu().double()
            if getattr(st, "joint_vel", None) is not None:
                out[f"{kind}/{name}/qd"] = st.joint_vel.detach().cpu().double()
    return out


def state_distance(a: TensorState, b: TensorState) -> tuple[float, str]:
    """Largest absolute difference between two states and the quantity it occurred in."""
    worst, where, _ = _state_distance(a, b)
    return worst, where


def _state_distance(a: TensorState, b: TensorState) -> tuple[float, str, int]:
    """``(worst, where, compared)``: quantities present with the same shape on both sides are compared."""
    fa, fb = _flatten(a), _flatten(b)
    worst, where, compared = 0.0, "", 0
    for key, ta in fa.items():
        tb = fb.get(key)
        if tb is None or tb.shape != ta.shape or ta.numel() == 0:
            continue
        compared += 1
        d = float((ta - tb).abs().max())
        if d > worst:
            worst, where = d, key
    return worst, where, compared


@dataclass
class Trajectory:
    """A recorded rollout: ``states[t]`` is the state *before* ``actions[t]``; ``states[-1]`` is final."""

    states: list[TensorState]
    actions: list[torch.Tensor]

    def __len__(self) -> int:
        return len(self.actions)


@dataclass
class ReplayReport:
    """Outcome of one replay check: worst deviation, where it happened, and the pass/fail verdict."""

    level: str
    passed: bool
    tolerance: float
    worst: float = 0.0
    worst_step: int = -1
    worst_key: str = ""
    per_step: list[float] = field(default_factory=list)
    compared_keys: int = 0
    """Quantities compared per step (``robots/<name>/q`` etc.); zero means the check was vacuous."""

    def __str__(self) -> str:
        verdict = "PASS" if self.passed else "FAIL"
        return (
            f"{self.level} {verdict}: max |Δ| = {self.worst:.3e} at step {self.worst_step} ({self.worst_key or '-'}), "
            f"tolerance {self.tolerance:.1e}, {len(self.per_step)} steps, {self.compared_keys} quantities compared"
        )


def record(handler, initial_state: TensorState, actions: list[torch.Tensor]) -> Trajectory:
    """Write ``initial_state``, apply ``actions`` one per ``simulate()``, and capture every state."""
    handler.set_states(initial_state)
    states = [handler.get_states(mode="tensor")]
    for a in actions:
        handler.set_dof_targets(a)
        handler.simulate()
        states.append(handler.get_states(mode="tensor"))
    return Trajectory(states=states, actions=list(actions))


def _report(level: str, diffs: list[tuple[float, str]], tol: float) -> ReplayReport:
    rep = ReplayReport(level=level, passed=True, tolerance=tol, per_step=[d for d, _ in diffs])
    for t, (d, key) in enumerate(diffs):
        if d > rep.worst:
            rep.worst, rep.worst_step, rep.worst_key = d, t, key
    rep.passed = rep.worst <= tol
    return rep


def verify_action_replay(handler, traj: Trajectory, *, tol: float = 1e-4) -> ReplayReport:
    """L0: replay ``traj.actions`` from ``traj.states[0]`` and compare every state to the recording."""
    handler.set_states(traj.states[0])
    first = _state_distance(handler.get_states(mode="tensor"), traj.states[0])
    diffs = [first[:2]]
    for t, a in enumerate(traj.actions):
        handler.set_dof_targets(a)
        handler.simulate()
        diffs.append(state_distance(handler.get_states(mode="tensor"), traj.states[t + 1]))
    report = _report("L0 action replay", diffs, tol)
    report.compared_keys = first[2]
    if report.compared_keys == 0:
        report.passed = False
        report.worst_key = "nothing compared: no quantity had the same shape on both sides"
    return report


def verify_state_replay(
    handler, traj: Trajectory, *, every: int = 10, tol_roundtrip: float = 1e-6, tol_step: float = 1e-4
) -> tuple[ReplayReport, ReplayReport]:
    """L1 state-anchored replay.

    For every ``every``-th recorded state, write it back and check (a) it reads back unchanged and
    (b) one recorded action from it reproduces the recorded next state.
    """
    roundtrip: list[tuple[float, str]] = []
    one_step: list[tuple[float, str]] = []
    for t in range(0, len(traj.actions), every):
        handler.set_states(traj.states[t])
        roundtrip.append(state_distance(handler.get_states(mode="tensor"), traj.states[t]))
        handler.set_dof_targets(traj.actions[t])
        handler.simulate()
        one_step.append(state_distance(handler.get_states(mode="tensor"), traj.states[t + 1]))
    return _report("L1 state round-trip", roundtrip, tol_roundtrip), _report("L1 one-step", one_step, tol_step)


def _as_float32(state: TensorState) -> TensorState:
    """Every floating tensor field of ``state`` as float32 (the dtype the backends' buffers use)."""
    import dataclasses

    def entity(st):
        kwargs = {}
        for f in dataclasses.fields(st):
            v = getattr(st, f.name)
            kwargs[f.name] = v.float() if isinstance(v, torch.Tensor) and v.is_floating_point() else v
        return type(st)(**kwargs)

    return TensorState(
        objects={k: entity(v) for k, v in state.objects.items()},
        robots={k: entity(v) for k, v in state.robots.items()},
        cameras=state.cameras,
        extras=state.extras,
    )


def verify_episode_replay(handler, episode, *, tol: float = 1e-4, env_step_tol: float = 1e-9) -> ReplayReport:
    """L0 from disk: replay a saved :class:`~metasim.utils.trajectory.EpisodeRecord` on ``handler``.

    The record's time base must match the handler's (a different ``dt`` / ``decimation`` produces a
    different trajectory by construction, so that is an error, not a deviation), and its provenance
    is compared with this machine's in the report's ``worst_key`` when the replay fails: a changed
    asset hash or backend version is the usual reason.
    """
    from metasim.utils.trajectory import check_assets, env_step_seconds

    there = episode.provenance
    simulator = str(handler.scenario.simulator)
    if there.simulator != simulator:
        raise ValueError(f"episode was recorded on {there.simulator!r}, handler is {simulator!r}")
    if there.num_envs != handler.num_envs:
        raise ValueError(
            f"episode has {there.num_envs} env(s), handler has {handler.num_envs}: a broadcast replay would compare "
            "nothing (state_distance skips shape mismatches). Replay on a handler with the recorded num_envs."
        )
    scenario_names = {
        "robots": [r.name for r in handler.scenario.robots],
        "objects": [o.name for o in handler.scenario.objects],
    }
    for kind in ("robots", "objects"):
        missing = [n for n in episode.entities.get(kind, []) if n not in scenario_names[kind]]
        if missing:
            raise ValueError(
                f"episode {kind} {missing} are not in the handler's scenario ({scenario_names[kind]}); "
                "nothing of theirs would be compared"
            )
    for name, names in episode.joint_names.items():
        here = list(handler.get_joint_names(name, sort=True))
        if here != list(names):
            raise ValueError(f"joint names of {name!r} differ: recorded {names}, handler {here}")
    here_step = env_step_seconds(handler)
    if there.env_step_s is None or here_step is None:
        raise ValueError(
            f"time base unknown (recorded env step {there.env_step_s}, handler {here_step}): the backend does not "
            "report physics_dt, so a replay cannot be validated. Set sim_params.dt explicitly on both sides."
        )
    if abs(there.env_step_s - here_step) > env_step_tol:
        raise ValueError(
            f"time base differs: recorded env step {there.env_step_s}s (dt={there.dt}, decimation={there.decimation}); "
            f"handler env step {here_step}s (decimation={getattr(handler.scenario, 'decimation', None)})"
        )
    # records are float64 CPU; tensor-input GPU backends (Newton, Isaac Sim) index CUDA float32
    # buffers, so the initial state and the actions go to the handler's device / dtype first
    from metasim.utils.state import state_to_device

    device = getattr(handler, "device", None) or torch.device("cpu")
    states = [_as_float32(state_to_device(s, device)) for s in episode.states]
    actions = [a.to(device=device, dtype=torch.float32) for a in episode.actions]
    traj = Trajectory(states=states, actions=actions)
    report = verify_action_replay(handler, traj, tol=tol)
    if not report.passed:
        from metasim.utils.trajectory import _backend_versions

        installed = _backend_versions(simulator)
        drift = [k for k, v in check_assets(episode).items() if v != "ok"]
        versions = {k: (v, installed.get(k)) for k, v in there.backend_versions.items() if installed.get(k) != v}
        notes = []
        if drift:
            notes.append(f"assets differ: {drift}")
        if versions:
            notes.append(f"backend versions differ: {versions}")
        if notes:
            report.worst_key = f"{report.worst_key} [{'; '.join(notes)}]"
    return report
