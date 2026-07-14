"""Bidirectional bridge: ``BaseTaskEnv`` <-> typed ``ObsBatch``/``ActionBatch``, vectorized.

Reads ``handler.get_states(mode="tensor")`` ONCE per step and slices per-chain fields
for all envs at once — no per-env-index python loop. Actions are applied via the
**tensor path**: a single ``(num_envs, total_dof)`` target tensor in the handler's joint
order, so there are no per-env ``dof_pos_target`` python dicts (and no ``.tolist()``
round-trip) on the hot path.

Missing observations are an error, not a zero: a spec that declares ``<arm>.ee_pose`` or
``<cam>.rgb`` and a backend that cannot produce it (no ``body_state`` on pybullet/genesis, an
unrendered camera) raises with the backend named, instead of feeding the policy a constant
identity pose or a black image.

Control modes: ``joint_pos`` (implemented, CPU-testable). ``ee_pose`` needs cuRobo IK
(GPU) and raises for now with a pointer to ``roboverse_learn.il`` IK models.
"""

from __future__ import annotations

import inspect

import torch
from loguru import logger

from .embodiment import Embodiment
from .obs import ActionBatch, ObsBatch
from .spec import EE_POSE_DIM, ActionSpec, ObsSpec, Space


def _t(x) -> torch.Tensor:
    return x if isinstance(x, torch.Tensor) else torch.as_tensor(x)


class EnvAdapter:
    """Translates one RoboVerse ``BaseTaskEnv`` to/from the typed harness carriers."""

    def __init__(self, env, emb: Embodiment, obs_spec: ObsSpec, action_spec: ActionSpec) -> None:
        self.env = env
        self.emb = emb
        self.obs_spec = obs_spec
        self.action_spec = action_spec
        self.handler = env.handler
        self.num_envs = int(getattr(env, "num_envs", 1))
        self.device = torch.device(getattr(env, "device", None) or "cpu")
        if action_spec.control not in {"joint_pos"}:
            raise NotImplementedError(
                f"EnvAdapter control={action_spec.control!r} not wired yet; use 'joint_pos'. "
                "ee_pose needs cuRobo IK (see roboverse_learn.il IK models) — GPU-gated."
            )
        if not emb.chains or not action_spec.fields:
            raise ValueError(
                f"empty embodiment/action-spec for task on {emb.robot_names} — no controllable "
                "joints inferred (robot may define joints via URDF/USD with empty joint_limits). "
                "Provide joint_limits or EmbodimentHints."
            )
        # per-robot sorted joint order (matches joint_pos columns and the tensor action layout)
        self._robot_order = {r: self.handler.get_joint_names(r, sort=True) for r in emb.robot_names}
        # global column index of each joint in the concatenated (num_envs, total_dof) action tensor
        self._col: dict[str, int] = {}
        off = 0
        for r in emb.robot_names:
            for j in self._robot_order[r]:
                self._col[f"{r}/{j}"] = off
                off += 1
        self.total_dof = off
        # Not every task's reset() accepts seed (some override it without the param, in
        # violation of the reset(seed) contract). Forward seed only when accepted, so the
        # harness degrades gracefully instead of crashing (mirrors the MetaSim gym bridge).
        params = inspect.signature(env.reset).parameters
        self._reset_accepts_seed = "seed" in params or any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
        )
        # per-chain (robot, [global cols]) for slicing obs and filling actions
        self._chain_cols = {c.name: (c.robot, [self._col[f"{c.robot}/{j}"] for j in c.joint_names]) for c in emb.chains}
        self._chain_local = {  # column indices within the owning robot's joint_pos tensor
            c.name: (c.robot, [self._robot_order[c.robot].index(j) for j in c.joint_names]) for c in emb.chains
        }
        self._language = getattr(env, "get_language_instruction", None)

    # ------------------------------------------------------------------ obs
    def _states(self):
        return self.handler.get_states(mode="tensor")

    def obs_batch(self, env_ids: torch.Tensor) -> ObsBatch:
        """Build a typed ObsBatch for ``env_ids`` from the current sim state."""
        states = self._states()
        ids = _t(env_ids).to(torch.long)
        tensors: dict[str, torch.Tensor] = {}
        # cache per-robot joint_pos / body_state once
        jp = {r: _t(states.robots[r].joint_pos) for r in self.emb.robot_names}
        for f in self.obs_spec.fields:
            if f.space in (Space.JOINT_POS, Space.GRIPPER):
                robot, local = self._chain_local[f.chain]
                tensors[f.key] = jp[robot][ids][:, local].to(self.device, torch.float32)
            elif f.space == Space.EE_POSE:
                tensors[f.key] = self._ee_pose(f, states, ids)
            elif f.space in (Space.RGB, Space.DEPTH):
                tensors[f.key] = self._camera(f, states, ids)
            # TASK fields are non-tensor and travel in ObsBatch.task (below)
        return ObsBatch(self.obs_spec, ids, tensors, self._task_payload())

    def _task_payload(self) -> dict:
        """Non-tensor payload: the task's language instruction when the env exposes one.

        Only tasks that implement ``get_language_instruction()`` (e.g. the simpler_env ports)
        have one; for everything else the payload is empty and ``derive_obs_spec`` emits no
        ``task.language`` field.
        """
        return {"language": self._language()} if callable(self._language) else {}

    def _ee_pose(self, f, states, ids: torch.Tensor) -> torch.Tensor:
        c = self.emb.chain(f.chain)
        rs = states.robots[c.robot]
        body_state = getattr(rs, "body_state", None)
        names = list(getattr(rs, "body_names", None) or self._body_names(c.robot) or [])
        if body_state is None or not names:
            raise RuntimeError(
                f"obs field {f.key!r} needs per-body state, but backend "
                f"{type(self.handler).__name__} returned body_state/body_names=None for robot "
                f"{c.robot!r} (pybullet and genesis do not expose it). Derive the obs spec with "
                "include_ee_pose=False (evaluate(..., include_ee_pose=False)) or use a backend that "
                "reports body state."
            )
        if c.ee_body_name not in names:
            raise RuntimeError(
                f"obs field {f.key!r}: ee_body_name {c.ee_body_name!r} is not a body of robot "
                f"{c.robot!r} on this backend (bodies: {names}). Fix RobotCfg.ee_body_name or pass "
                "EmbodimentHints(ee_body=...)."
            )
        bs = _t(body_state)[ids]  # (B, nbodies, 13)
        return bs[:, names.index(c.ee_body_name), :EE_POSE_DIM].to(self.device, torch.float32)

    def _body_names(self, robot: str):
        getter = getattr(self.handler, "get_body_names", None)
        return getter(robot) if callable(getter) else None

    def _camera(self, f, states, ids: torch.Tensor) -> torch.Tensor:
        cams = getattr(states, "cameras", None) or {}
        cam = cams.get(f.frame) if hasattr(cams, "get") else None
        attr = "rgb" if f.space == Space.RGB else "depth"
        data = getattr(cam, attr, None) if cam is not None else None
        if data is None:
            raise RuntimeError(
                f"obs field {f.key!r}: camera {f.frame!r} produced no {attr} this step "
                f"(cameras present: {sorted(cams) if hasattr(cams, 'keys') else cams}). The camera must be "
                f"in the scenario with {attr!r} in data_types — pass it to evaluate(cameras=[...]) — and the "
                "backend must render it (headless rendering enabled)."
            )
        arr = _t(data)
        arr = arr[ids] if arr.dim() == len(f.shape) + 1 else arr.unsqueeze(0).expand(len(ids), *f.shape)
        if tuple(arr.shape[1:]) != tuple(f.shape):
            raise RuntimeError(
                f"obs field {f.key!r}: camera {f.frame!r} rendered {tuple(arr.shape[1:])} but the spec "
                f"declares {tuple(f.shape)}; make the CameraCfg width/height match."
            )
        return arr.to(self.device)

    # --------------------------------------------------------------- action
    def apply(self, actions: ActionBatch):
        """Apply an ActionBatch (subset of envs) and step ALL envs; return the step tuple.

        Inactive envs (not in ``actions.env_ids``) hold their current joint *targets* (not their
        measured joint positions — re-targeting the measured position each step would let a
        position-controlled arm droop under gravity instead of holding).
        """
        actions.validate()  # descriptive error on a missing/misshaped field, not a deep crash
        states = self._states()
        # start from the current joint targets (hold), full (num_envs, total_dof). Allocate on the
        # sim device: on CUDA-state backends (mjx/newton/isaacgym) joint_pos/actions are CUDA, and
        # an in-place index-assign into a CPU tensor would raise a cross-device RuntimeError.
        target = torch.zeros(self.num_envs, self.total_dof, dtype=torch.float32, device=self.device)
        for r in self.emb.robot_names:
            cols = [self._col[f"{r}/{j}"] for j in self._robot_order[r]]
            target[:, cols] = self._hold(states.robots[r], r)
        ids = _t(actions.env_ids).to(torch.long)  # CPU index tensor is fine on a CUDA destination
        for f in self.action_spec.fields:
            _, gcols = self._chain_cols[f.chain]
            vals = _t(actions.tensors[f.key]).to(self.device, torch.float32)
            target[ids[:, None], torch.as_tensor(gcols)] = vals
        return self.env.step(target)

    def _hold(self, rs, robot: str) -> torch.Tensor:
        """The hold value for every joint of ``robot``: its current joint_pos_target.

        Backends populate ``joint_pos_target`` from the last commanded target. Before the first
        step of an episode some backends leave it unset (``None``); the measured ``joint_pos`` is
        then the correct seed (the robot is at its reset pose and has not sagged yet).
        """
        target = getattr(rs, "joint_pos_target", None)
        if target is None:
            if not getattr(self, "_hold_warned", False):
                logger.debug(
                    f"[harness] {type(self.handler).__name__} reports joint_pos_target=None for "
                    f"{robot!r}; holding inactive envs at the measured joint_pos for this step."
                )
                self._hold_warned = True
            target = rs.joint_pos
        return _t(target).to(self.device, torch.float32)

    def reset(self, env_ids: torch.Tensor | None = None, seed: int | None = None):
        kw = {}
        if env_ids is not None:
            kw["env_ids"] = _t(env_ids).to(torch.long)
        if seed is not None:
            if self._reset_accepts_seed:
                kw["seed"] = seed
            elif not getattr(self, "_seed_warned", False):
                logger.debug(f"[harness] {type(self.env).__name__}.reset() takes no seed; not seeding this task")
                self._seed_warned = True
        return self.env.reset(**kw)
