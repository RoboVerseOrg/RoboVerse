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
    fa, fb = _flatten(a), _flatten(b)
    worst, where = 0.0, ""
    for key, ta in fa.items():
        tb = fb.get(key)
        if tb is None or tb.shape != ta.shape or ta.numel() == 0:
            continue
        d = float((ta - tb).abs().max())
        if d > worst:
            worst, where = d, key
    return worst, where


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

    def __str__(self) -> str:
        verdict = "PASS" if self.passed else "FAIL"
        return (
            f"{self.level} {verdict}: max |Δ| = {self.worst:.3e} at step {self.worst_step} ({self.worst_key or '-'}), "
            f"tolerance {self.tolerance:.1e}, {len(self.per_step)} steps"
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
    diffs = [state_distance(handler.get_states(mode="tensor"), traj.states[0])]
    for t, a in enumerate(traj.actions):
        handler.set_dof_targets(a)
        handler.simulate()
        diffs.append(state_distance(handler.get_states(mode="tensor"), traj.states[t + 1]))
    return _report("L0 action replay", diffs, tol)


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
