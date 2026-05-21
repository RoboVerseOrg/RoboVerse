"""SceneEntityCfg adapter for mjlab → RoboVerse MDP function port.

mjlab's ``SceneEntityCfg`` selects a subset of joints / bodies from a
named scene entity. In RoboVerse the simulator handler reports sorted
joint and body lists per entity (``handler.get_joint_names(name, sort=True)``
and ``handler.get_body_names(name, sort=True)``); this module resolves
the name patterns into integer indices the way mjlab does internally.

Minimal API surface:
    SceneEntityCfg(name, joint_names=(), body_names=())
    resolve_joint_ids(env, cfg) -> torch.LongTensor
    resolve_body_ids(env, cfg) -> torch.LongTensor

Results are cached per ``SceneEntityCfg`` instance on first call to
amortize the name-lookup cost across the training loop.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

import torch


@dataclass
class SceneEntityCfg:
    """Selector for joints/bodies of a named scene entity."""

    name: str
    joint_names: tuple[str, ...] | str = ()
    body_names: tuple[str, ...] | str = ()

    # Resolved-id cache (populated lazily by ``resolve_*_ids``).
    _joint_ids: torch.Tensor | None = field(default=None, init=False, repr=False)
    _body_ids: torch.Tensor | None = field(default=None, init=False, repr=False)
    _joint_ids_device: torch.device | None = field(default=None, init=False, repr=False)
    _body_ids_device: torch.device | None = field(default=None, init=False, repr=False)

    @property
    def joint_ids(self) -> torch.Tensor:
        if self._joint_ids is None:
            raise RuntimeError(f"SceneEntityCfg({self.name!r}).joint_ids accessed before resolve_joint_ids() called")
        return self._joint_ids

    @property
    def body_ids(self) -> torch.Tensor:
        if self._body_ids is None:
            raise RuntimeError(f"SceneEntityCfg({self.name!r}).body_ids accessed before resolve_body_ids() called")
        return self._body_ids


def _match_names(patterns: Iterable[str] | str, available: list[str]) -> list[int]:
    """Resolve a list of name patterns against a sorted available-name list.

    Match strategy (in order):
      1. exact match (``pat in available``)
      2. trailing-segment match (``available[i].endswith("/" + pat)`` or
         ``available[i] == pat``) — handles Newton's full-XML-path joint
         names like ``"cartpole/worldbody/cart/slider"`` when caller
         provides short name ``"slider"``.
      3. Python regex fullmatch (existing behavior for power users).

    A pattern that matches no names raises ValueError — silent drops are
    a common source of subtle reward-function bugs.
    """
    if isinstance(patterns, str):
        patterns = (patterns,)
    if not patterns:
        return list(range(len(available)))
    out: list[int] = []
    for pat in patterns:
        # 1. Exact match
        if pat in available:
            out.append(available.index(pat))
            continue
        # 2. Trailing-segment match (handles Newton "asset/.../slider" vs "slider")
        trail_matches = [
            i for i, name in enumerate(available) if name.endswith("/" + pat) or name.split("/")[-1] == pat
        ]
        if len(trail_matches) == 1:
            out.append(trail_matches[0])
            continue
        if len(trail_matches) > 1:
            out.extend(trail_matches)
            continue
        # 3. Regex
        matches = [i for i, name in enumerate(available) if re.fullmatch(pat, name)]
        if not matches:
            raise ValueError(f"SceneEntityCfg pattern '{pat}' matched no names in {available}")
        out.extend(matches)
    return out


def _entity_is_registered_robot(env, name: str) -> bool:
    """Whether ``name`` is a robot known to the handler's TensorState path."""
    try:
        env.handler.get_joint_names(name, sort=True)
        return True
    except (KeyError, AttributeError):
        return False


def _global_mujoco_joint_names(env) -> list[str]:
    """Sorted-by-id MuJoCo joint name list. Used for scene-MJCF assets
    that aren't registered as RobotCfg entities.
    """
    import mujoco

    physics = env.handler.physics
    m = physics.model.ptr if hasattr(physics.model, "ptr") else physics.model
    return [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(m.njnt)]


def _global_mujoco_body_names(env) -> list[str]:
    import mujoco

    physics = env.handler.physics
    m = physics.model.ptr if hasattr(physics.model, "ptr") else physics.model
    return [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(m.nbody)]


def resolve_joint_ids(env, cfg: SceneEntityCfg) -> torch.Tensor:
    """Return cached LongTensor of joint indices for ``cfg``.

    Resolves against ``handler.get_joint_names(name, sort=True)`` when
    the named entity is a registered robot; otherwise falls back to the
    global MuJoCo joint name list (scene-MJCF assets like the mjlab
    cartpole, which loads as the scene rather than a robot entity).
    """
    if cfg._joint_ids is not None and cfg._joint_ids_device == env.device:
        return cfg._joint_ids
    if _entity_is_registered_robot(env, cfg.name):
        available = env.handler.get_joint_names(cfg.name, sort=True)
    else:
        available = _global_mujoco_joint_names(env)
    ids = _match_names(cfg.joint_names, available)
    cfg._joint_ids = torch.tensor(ids, dtype=torch.long, device=env.device)
    cfg._joint_ids_device = env.device
    return cfg._joint_ids


def resolve_body_ids(env, cfg: SceneEntityCfg) -> torch.Tensor:
    """Return cached LongTensor of body indices for ``cfg`` (robot-name first, global fallback)."""
    if cfg._body_ids is not None and cfg._body_ids_device == env.device:
        return cfg._body_ids
    if _entity_is_registered_robot(env, cfg.name):
        available = env.handler.get_body_names(cfg.name, sort=True)
    else:
        available = _global_mujoco_body_names(env)
    ids = _match_names(cfg.body_names, available)
    cfg._body_ids = torch.tensor(ids, dtype=torch.long, device=env.device)
    cfg._body_ids_device = env.device
    return cfg._body_ids


# ---------------------------------------------------------------------------
# unified joint-state accessor (robot first, scene-MJCF fallback)
# ---------------------------------------------------------------------------


def _joint_id_to_qposadr(env, joint_ids: torch.Tensor) -> torch.Tensor:
    """Map MuJoCo joint indices → qpos offsets (handles freejoint+ball+slide+hinge widths).

    MuJoCo joints occupy different qpos widths: freejoint=7, ball=4, slide/hinge=1.
    Use ``mjModel.jnt_qposadr[jid]`` to get the qpos start offset. For
    hinge/slide that's also the only qpos element; for ball/free this is
    where the multi-element block begins.

    For RoboVerse rewards (joint-pos / joint-vel) we want the SCALAR joint
    angle, so for hinge/slide returning qposadr is correct, and for
    free/ball we shouldn't be selecting them as actuated joints anyway
    (they're typically excluded from ``SceneEntityCfg.joint_names``).
    """
    model = env.handler.physics.model
    mp = model.ptr if hasattr(model, "ptr") else model
    addrs = [int(mp.jnt_qposadr[int(jid)]) for jid in joint_ids.cpu().tolist()]
    return torch.tensor(addrs, dtype=torch.long, device=env.device)


def _joint_id_to_dofadr(env, joint_ids: torch.Tensor) -> torch.Tensor:
    """Map joint indices → qvel/dof offsets (freejoint=6, ball=3, slide/hinge=1)."""
    model = env.handler.physics.model
    mp = model.ptr if hasattr(model, "ptr") else model
    addrs = [int(mp.jnt_dofadr[int(jid)]) for jid in joint_ids.cpu().tolist()]
    return torch.tensor(addrs, dtype=torch.long, device=env.device)


def entity_joint_pos(env, env_states, cfg: SceneEntityCfg) -> torch.Tensor:
    """Joint positions for the entity, batched. Shape ``(num_envs, len(joint_ids))``.

    Reads ``env_states.robots[cfg.name].joint_pos[:, ids]`` when the
    entity is a registered robot; otherwise reads ``physics.data.qpos``
    via ``jnt_qposadr`` lookup (correctly handles MJCF with free / ball
    joints — joint index ≠ qpos offset for those).
    """
    joint_ids = resolve_joint_ids(env, cfg)
    if cfg.name in env_states.robots:
        return env_states.robots[cfg.name].joint_pos[:, joint_ids]
    import numpy as np

    qpos_offsets = _joint_id_to_qposadr(env, joint_ids).cpu().numpy()
    qpos = np.asarray(env.handler.physics.data.qpos, dtype=np.float32)
    sel = torch.as_tensor(qpos[qpos_offsets], device=env.device, dtype=torch.float32)
    return sel.unsqueeze(0).expand(env.num_envs, -1)


def entity_joint_vel(env, env_states, cfg: SceneEntityCfg) -> torch.Tensor:
    """Joint velocities for the entity, batched. Shape ``(num_envs, len(joint_ids))``."""
    joint_ids = resolve_joint_ids(env, cfg)
    if cfg.name in env_states.robots:
        return env_states.robots[cfg.name].joint_vel[:, joint_ids]
    import numpy as np

    dof_offsets = _joint_id_to_dofadr(env, joint_ids).cpu().numpy()
    qvel = np.asarray(env.handler.physics.data.qvel, dtype=np.float32)
    sel = torch.as_tensor(qvel[dof_offsets], device=env.device, dtype=torch.float32)
    return sel.unsqueeze(0).expand(env.num_envs, -1)
