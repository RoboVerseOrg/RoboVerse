"""Sensor ports — ContactSensor / TerrainHeightSensor / BuiltinSensor.

RoboVerse-side ports of mjlab's sensor subsystem
(``mjlab/src/mjlab/sensor/``). Implemented without modifying MetaSim
core: each sensor is a stand-alone Python class that the task
instantiates and registers on ``env._mjlab_sensors``; the corresponding
reward terms in :mod:`mdp.rewards` read from this registry.

Computation strategy:

- :class:`ContactSensor` — each ``update(dt)`` scans
  ``physics.data.contact[:ncon]`` and integrates per-primary force /
  contact-state, then advances air-time / contact-time accumulators.
  Avoids the MjSpec sensor injection path used by mjlab (which would
  require editing the MJCF at load time) and keeps RoboVerse's
  ScenarioCfg path untouched.

- :class:`TerrainHeightSensor` — per-site / per-body downward ray cast
  via ``mujoco.mj_ray``, returning ``height - target_height`` per primary.

- :class:`BuiltinSensor` — wraps a single ``mjData`` field (most
  important one is ``subtree_angmom`` for whole-body angular momentum).

All three expose a ``data`` ``@dataclass`` matching mjlab's, so the
reward functions in :mod:`mdp.rewards` can be a drop-in port of
upstream code (just swap the import path).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch


@dataclass
class ContactSensorCfg:
    """Mirrors mjlab ``ContactSensorCfg``.

    Only the fields RoboVerse uses are typed; other mjlab kwargs are
    accepted via ``**`` for cfg-import parity.
    """

    name: str = "contact"
    primary_bodies: tuple[str, ...] = ()
    """Primary body names (e.g. quadruped feet) to track contact for."""
    secondary_body: str | None = None
    """Optional secondary body name; restrict matched contacts to those
    where the *other* side of the contact is this body. ``None`` matches
    any contact involving a primary body."""
    fields: tuple[str, ...] = ("found", "force")
    """Subset of {found, force, dist, pos}. Only requested fields are
    populated on :class:`ContactData`."""
    num_slots: int = 1
    """Mjlab parity. Always 1 in current RoboVerse path."""
    track_air_time: bool = False
    history_length: int = 0


@dataclass
class ContactData:
    """Per-step contact sensor output. Shape conventions follow mjlab.

    Per-contact fields (``found``/``force``/``dist``) are ``[B, P]`` where
    P = len(primary_bodies). Air-time fields are ``[B, P]``. History is
    ``[B, P, H]`` or ``[B, P, H, 3]`` for vector fields.
    """

    found: torch.Tensor | None = None
    force: torch.Tensor | None = None
    dist: torch.Tensor | None = None
    pos: torch.Tensor | None = None

    current_air_time: torch.Tensor | None = None
    last_air_time: torch.Tensor | None = None
    current_contact_time: torch.Tensor | None = None
    last_contact_time: torch.Tensor | None = None

    force_history: torch.Tensor | None = None
    """``[B, P, H, 3]`` — index 0 is most recent (rolling buffer)."""


class ContactSensor:
    """Per-step body-contact tracker. RoboVerse port of mjlab ContactSensor.

    Resolves ``primary_bodies`` to MuJoCo body IDs at first update; then
    on each step (called by the task's post-physics hook) scans
    ``physics.data.contact[:ncon]`` and accumulates per-body force +
    contact-state. Tracks air / contact time and optional history.

    Usage:

      sensor = ContactSensor(env, ContactSensorCfg(
          name="feet_ground_contact",
          primary_bodies=("FR_foot", "FL_foot", "RR_foot", "RL_foot"),
          secondary_body="floor",
          fields=("found", "force"),
          track_air_time=True,
          history_length=4,
      ))
      env._mjlab_sensors["feet_ground_contact"] = sensor

      # in task._post_physics_step / a post-step event:
      sensor.update(env.step_dt)

      # in a reward fn:
      data = env._mjlab_sensors["feet_ground_contact"].data
    """

    def __init__(self, env, cfg: ContactSensorCfg):
        self.env = env
        self.cfg = cfg
        self.dt = float(env.step_dt)
        device = env.device
        self.device = device

        N = env.num_envs
        P = len(cfg.primary_bodies)
        self._N = N
        self._P = P
        self._fields = set(cfg.fields)

        # Output buffers (kept persistent so update() is allocation-free) +
        # air-time / contact-time state + force history. Allocated for both
        # backends before the backend-specific resolution below.
        self._found = torch.zeros((N, P), device=device, dtype=torch.float32)
        self._force = torch.zeros((N, P, 3), device=device, dtype=torch.float32)
        self._dist = torch.zeros((N, P), device=device, dtype=torch.float32)
        self._current_air_time = torch.zeros((N, P), device=device, dtype=torch.float32)
        self._last_air_time = torch.zeros((N, P), device=device, dtype=torch.float32)
        self._current_contact_time = torch.zeros((N, P), device=device, dtype=torch.float32)
        self._last_contact_time = torch.zeros((N, P), device=device, dtype=torch.float32)
        if cfg.history_length > 0:
            self._force_history = torch.zeros((N, P, cfg.history_length, 3), device=device, dtype=torch.float32)
        else:
            self._force_history = None

        # Newton path: no MuJoCo physics. Defer primary-foot → body-column
        # resolution to the first update (needs the solver's model body names).
        self._newton = not hasattr(env.handler, "physics")
        if self._newton:
            self._newton_body_cols: list[int] | None = None
            return

        import mujoco

        physics = env.handler.physics
        m = physics.model.ptr if hasattr(physics.model, "ptr") else physics.model

        # Resolve primary body IDs (raise if any name is wrong).
        self._primary_body_ids: list[int] = []
        for name in cfg.primary_bodies:
            bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid < 0:
                raise ValueError(f"ContactSensor: primary body '{name}' not found in model.")
            self._primary_body_ids.append(bid)
        # Secondary body id (-1 ⇒ any).
        self._secondary_body_id = -1
        if cfg.secondary_body is not None:
            sb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, cfg.secondary_body)
            if sb < 0:
                # Common case: secondary="terrain" but the geom is unnamed
                # ground — fall back to "any body that is not robot's".
                self._secondary_body_id = -1
            else:
                self._secondary_body_id = sb

        # Pre-compute geom → body table for fast contact-row lookup.
        nbody = m.nbody
        ngeom = m.ngeom
        self._geom_bodyid = np.asarray(m.geom_bodyid).copy()  # (ngeom,)
        self._is_primary = np.zeros(nbody, dtype=bool)
        self._primary_idx_in_cfg = np.full(nbody, -1, dtype=np.int64)
        for i, bid in enumerate(self._primary_body_ids):
            self._is_primary[bid] = True
            self._primary_idx_in_cfg[bid] = i

        # Index of "primary body ancestors" too — mjlab subtree mode picks up
        # geoms attached to descendants. We mirror that by walking body_parentid.
        body_parentid = np.asarray(m.body_parentid).copy()
        self._primary_subtree_root = np.full(nbody, -1, dtype=np.int64)
        for bid in range(nbody):
            cur = bid
            while cur > 0:
                if cur in self._primary_body_ids:
                    self._primary_subtree_root[bid] = self._primary_idx_in_cfg[cur]
                    break
                cur = int(body_parentid[cur])
            else:
                # Allow direct match at the root body itself.
                if bid in self._primary_body_ids:
                    self._primary_subtree_root[bid] = self._primary_idx_in_cfg[bid]

    @property
    def data(self) -> ContactData:
        """Return the current sensor reading (snapshot of internal buffers)."""
        out = ContactData()
        if "found" in self._fields:
            out.found = self._found
        if "force" in self._fields:
            out.force = self._force
        if "dist" in self._fields:
            out.dist = self._dist
        if self.cfg.track_air_time:
            out.current_air_time = self._current_air_time
            out.last_air_time = self._last_air_time
            out.current_contact_time = self._current_contact_time
            out.last_contact_time = self._last_contact_time
        if self._force_history is not None:
            out.force_history = self._force_history
        return out

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None or env_ids.numel() == 0:
            return
        ids = env_ids if isinstance(env_ids, torch.Tensor) else torch.as_tensor(env_ids)
        ids = ids.long().to(self.device)
        self._found[ids] = 0.0
        self._force[ids] = 0.0
        self._dist[ids] = 0.0
        self._current_air_time[ids] = 0.0
        self._last_air_time[ids] = 0.0
        self._current_contact_time[ids] = 0.0
        self._last_contact_time[ids] = 0.0
        if self._force_history is not None:
            self._force_history[ids] = 0.0

    def _update_newton(self, dt: float | None = None) -> None:
        """Newton path: per-env per-foot contact force via the handler's
        mujoco_warp contact reader. Mirrors the MuJoCo update's outputs
        (found / force + air-time) so all sensor-dependent rewards fire.
        """
        forces, names = self.env.handler.get_net_contact_forces_by_body()
        # (num_envs, nbody, 3), names length nbody
        if self._newton_body_cols is None:
            cols: list[int] = []
            for pat in self.cfg.primary_bodies:
                col = next(
                    (j for j, n in enumerate(names) if n == pat or n.endswith("_" + pat) or n.endswith(pat)),
                    -1,
                )
                cols.append(col)
            self._newton_body_cols = cols
        for p, col in enumerate(self._newton_body_cols):
            if 0 <= col < forces.shape[1]:
                self._force[:, p, :] = forces[:, col, :]
            else:
                self._force[:, p, :] = 0.0
        self._dist[:] = 0.0  # penetration distance not tracked on Newton path
        mag = torch.norm(self._force, dim=-1)
        self._found[:] = (mag > 1.0).float()  # ~1 N contact threshold
        if self.cfg.track_air_time:
            self._advance_air_time(dt if dt is not None else self.dt)
        if self._force_history is not None:
            self._force_history = self._force_history.roll(1, dims=2)
            self._force_history[:, :, 0, :] = self._force

    def update(self, dt: float | None = None) -> None:
        """Scan ``data.contact[:ncon]`` and integrate per-primary readings."""
        if self._newton:
            self._update_newton(dt)
            return
        import mujoco

        physics = self.env.handler.physics
        data = physics.data
        mp = physics.model.ptr if hasattr(physics.model, "ptr") else physics.model

        # Zero per-step accumulators.
        found_np = np.zeros((self._N, self._P), dtype=np.float32)
        force_np = np.zeros((self._N, self._P, 3), dtype=np.float32)
        dist_np = np.zeros((self._N, self._P), dtype=np.float32)

        ncon = int(data.ncon)
        # MuJoCo physics.data is single-env (RoboVerse mjuoco path is
        # num_envs=1 today); the env loop here is for forward compat.
        env_idx = 0

        contact = data.contact
        for i in range(ncon):
            con = contact[i]
            b1 = int(self._geom_bodyid[int(con.geom1)])
            b2 = int(self._geom_bodyid[int(con.geom2)])

            # Determine which side (if any) belongs to a primary subtree.
            prim_b1 = self._primary_subtree_root[b1] if b1 < len(self._primary_subtree_root) else -1
            prim_b2 = self._primary_subtree_root[b2] if b2 < len(self._primary_subtree_root) else -1

            if prim_b1 < 0 and prim_b2 < 0:
                continue

            # If both sides are primary (shouldn't normally happen), skip
            # the secondary check.
            if prim_b1 >= 0 and prim_b2 >= 0:
                p_idx = prim_b1
            elif prim_b1 >= 0:
                p_idx = prim_b1
                other = b2
                if self._secondary_body_id >= 0 and other != self._secondary_body_id:
                    # Secondary mismatch — skip unless secondary=any.
                    if self.cfg.secondary_body is not None:
                        continue
            else:
                p_idx = prim_b2
                other = b1
                if self._secondary_body_id >= 0 and other != self._secondary_body_id:
                    if self.cfg.secondary_body is not None:
                        continue

            # Extract contact force in contact frame via mj_contactForce.
            f6 = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(mp, data.ptr, i, f6)
            # f6 = (Fn, Ft1, Ft2, Mn, Mt1, Mt2) in contact frame.
            # For air-time / soft-landing the |F| matters; tangent components
            # are usually small relative to normal. Use the contact-frame F
            # directly (mjlab uses the same when global_frame=False).

            found_np[env_idx, p_idx] += 1.0
            force_np[env_idx, p_idx] += f6[:3].astype(np.float32)
            dist_np[env_idx, p_idx] = min(dist_np[env_idx, p_idx], float(con.dist))

        # Sanitize: mj_contactForce can return non-finite values for
        # degenerate (deeply-penetrating / near-singular) contacts that occur
        # stochastically at spawn. A non-finite contact force is physically
        # meaningless and would poison every sensor-dependent reward
        # (soft_landing, feet_slip, …) with NaN. Clamp to finite at the source.
        np.nan_to_num(force_np, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        np.nan_to_num(dist_np, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        # Copy back to torch buffers.
        self._found.copy_(torch.from_numpy(found_np))
        self._force.copy_(torch.from_numpy(force_np))
        self._dist.copy_(torch.from_numpy(dist_np))

        if self.cfg.track_air_time:
            self._advance_air_time(dt if dt is not None else self.dt)

        if self._force_history is not None:
            self._force_history = self._force_history.roll(1, dims=2)
            self._force_history[:, :, 0, :] = self._force

    def _advance_air_time(self, dt: float) -> None:
        in_contact = self._found > 0  # (N, P)
        elapsed = torch.full((self._N, self._P), dt, device=self.device, dtype=torch.float32)

        is_first_contact = (self._current_air_time > 0) & in_contact
        is_first_detached = (self._current_contact_time > 0) & ~in_contact

        # Latch last_air_time on transition into contact.
        self._last_air_time = torch.where(
            is_first_contact,
            self._current_air_time + elapsed,
            self._last_air_time,
        )
        # current_air_time grows while NOT in contact, else resets.
        self._current_air_time = torch.where(
            ~in_contact,
            self._current_air_time + elapsed,
            torch.zeros_like(self._current_air_time),
        )

        # Mirror for contact-time.
        self._last_contact_time = torch.where(
            is_first_detached,
            self._current_contact_time + elapsed,
            self._last_contact_time,
        )
        self._current_contact_time = torch.where(
            in_contact,
            self._current_contact_time + elapsed,
            torch.zeros_like(self._current_contact_time),
        )

    def compute_first_contact(self, dt: float, abs_tol: float = 1e-6) -> torch.Tensor:
        """Mjlab parity: ``[B, P]`` bool — primaries that landed within the last dt seconds."""
        is_in_contact = self._current_contact_time > 0.0
        within_dt = self._current_contact_time < (dt + abs_tol)
        return is_in_contact & within_dt

    def compute_first_air(self, dt: float, abs_tol: float = 1e-6) -> torch.Tensor:
        is_in_air = self._current_air_time > 0.0
        within_dt = self._current_air_time < (dt + abs_tol)
        return is_in_air & within_dt


# ---------------------------------------------------------------------------
# TerrainHeightSensor — ray-cast downward from each primary site/body.
# ---------------------------------------------------------------------------


@dataclass
class TerrainHeightSensorCfg:
    name: str = "foot_height_scan"
    primary_sites: tuple[str, ...] = ()
    """Site names (or body names — fallback). One ray per primary."""
    primary_bodies: tuple[str, ...] = ()
    """Used if a primary has no site; downward ray from body com."""
    max_distance: float = 1.0
    geom_groups: tuple[int, ...] = (0,)
    """Which MuJoCo geom groups participate in the ray cast (0 = terrain)."""
    target_height: float = 0.0
    """Subtract this from raw ray distance before returning. mjlab sets it
    to the foot's resting height so the value is zero at standstill."""


@dataclass
class TerrainHeightData:
    heights: torch.Tensor | None = None  # (N, P) height above terrain
    num_frames: int = 0  # mjlab parity (len(primary_sites|primary_bodies))


class TerrainHeightSensor:
    """Per-site downward ray-cast sensor. RoboVerse port of mjlab TerrainHeightSensor."""

    def __init__(self, env, cfg: TerrainHeightSensorCfg):
        self.env = env
        self.cfg = cfg
        device = env.device
        self.device = device

        primaries = list(cfg.primary_sites) or list(cfg.primary_bodies)
        if not primaries:
            raise ValueError("TerrainHeightSensor: primary_sites or primary_bodies required.")
        N = env.num_envs
        P = len(primaries)
        self._N = N
        self._P = P
        self._heights = torch.zeros((N, P), device=device, dtype=torch.float32)

        # Newton path: no ray-cast; on flat terrain foot-height-above-ground is
        # just the foot body z (ground plane at z=0). Defer body-column
        # resolution to the first update.
        self._newton = not hasattr(env.handler, "physics")
        if self._newton:
            self._newton_body_cols: list[int] | None = None
            return

        import mujoco

        physics = env.handler.physics
        mp = physics.model.ptr if hasattr(physics.model, "ptr") else physics.model

        self._site_ids: list[int] = []
        self._body_ids: list[int] = []
        for name in cfg.primary_sites:
            sid = mujoco.mj_name2id(mp, mujoco.mjtObj.mjOBJ_SITE, name)
            self._site_ids.append(sid)
        for name in cfg.primary_bodies:
            bid = mujoco.mj_name2id(mp, mujoco.mjtObj.mjOBJ_BODY, name)
            self._body_ids.append(bid)

        # Geom group bitfield.
        bitmask = 0
        for g in cfg.geom_groups:
            bitmask |= 1 << int(g)
        self._geomgroup_bits = bitmask

    @property
    def data(self) -> TerrainHeightData:
        return TerrainHeightData(
            heights=self._heights,
            num_frames=self._P,
        )

    @property
    def num_frames(self) -> int:
        return self._P

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None or env_ids.numel() == 0:
            return
        ids = env_ids.long().to(self.device)
        self._heights[ids] = 0.0

    def _update_newton(self, dt: float | None = None) -> None:
        """Flat-terrain foot height = foot body z (ground at z=0). No ray-cast."""
        st = self.env.handler.get_states(mode="tensor")
        robots = getattr(st, "robots", None) or {}
        rname = self.env.scenario.robots[0].name if self.env.scenario.robots else next(iter(robots), None)
        rs = robots.get(rname) if rname is not None else None
        if rs is None or rs.body_state is None:
            return
        names = rs.body_names
        if self._newton_body_cols is None:
            cols: list[int] = []
            for pat in self.cfg.primary_bodies:
                col = next(
                    (j for j, n in enumerate(names) if n == pat or n.endswith("_" + pat) or n.endswith(pat)),
                    -1,
                )
                cols.append(col)
            self._newton_body_cols = cols
        for p, col in enumerate(self._newton_body_cols):
            if 0 <= col < rs.body_state.shape[1]:
                self._heights[:, p] = rs.body_state[:, col, 2] - self.cfg.target_height
            else:
                self._heights[:, p] = 0.0

    def update(self, dt: float | None = None) -> None:
        if self._newton:
            self._update_newton(dt)
            return
        import mujoco

        physics = self.env.handler.physics
        data = physics.data
        mp = physics.model.ptr if hasattr(physics.model, "ptr") else physics.model

        out_np = np.zeros((self._N, self._P), dtype=np.float32)
        env_idx = 0
        for i in range(self._P):
            if i < len(self._site_ids) and self._site_ids[i] >= 0:
                origin = np.asarray(data.site_xpos[self._site_ids[i]], dtype=np.float64)
            elif i < len(self._body_ids) and self._body_ids[i] >= 0:
                origin = np.asarray(data.xpos[self._body_ids[i]], dtype=np.float64)
            else:
                continue
            direction = np.array([0.0, 0.0, -1.0], dtype=np.float64)
            # mj_ray returns distance along the ray (cast); -1 = no hit.
            geomgroup = np.zeros(6, dtype=np.uint8)
            for g in self.cfg.geom_groups:
                if 0 <= g < 6:
                    geomgroup[g] = 1
            geomid_out = np.zeros(1, dtype=np.int32)
            dist = mujoco.mj_ray(
                mp,
                data.ptr,
                origin,
                direction,
                geomgroup,
                1,
                -1,
                geomid_out,
            )
            # If no hit (dist < 0), set to max_distance as a fallback.
            if dist < 0 or dist > self.cfg.max_distance:
                out_np[env_idx, i] = self.cfg.max_distance - self.cfg.target_height
            else:
                # Height above terrain = how high the site is over the ground;
                # mjlab returns ``foot_height - target_height``.
                out_np[env_idx, i] = float(dist) - self.cfg.target_height
        self._heights.copy_(torch.from_numpy(out_np))


# ---------------------------------------------------------------------------
# BuiltinSensor — wraps a single mjData field by name.
# ---------------------------------------------------------------------------


@dataclass
class BuiltinSensorCfg:
    name: str = "root_angmom"
    field: Literal["subtree_angmom", "subtree_linvel", "subtree_com"] = "subtree_angmom"
    """Which ``mjData`` field to read."""
    body_name: str = "pelvis"


class BuiltinSensor:
    """Wraps a single ``mjData`` subtree-* field for one body.

    For ``subtree_angmom``: returns ``(N, 3)`` whole-body angular momentum
    about the COM of the named body's subtree, mirroring mjlab's
    ``robot/root_angmom`` builtin sensor.
    """

    def __init__(self, env, cfg: BuiltinSensorCfg):
        self.env = env
        self.cfg = cfg
        device = env.device
        self.device = device
        self._data = torch.zeros((env.num_envs, 3), device=device, dtype=torch.float32)

        # Newton path: mujoco_warp computes subtree_* on the batched data, read
        # via handler.get_subtree_field. Defer body-column resolution to update.
        self._newton = not hasattr(env.handler, "physics")
        if self._newton:
            self._newton_body_col: int | None = None
            return

        import mujoco

        physics = env.handler.physics
        mp = physics.model.ptr if hasattr(physics.model, "ptr") else physics.model
        bid = mujoco.mj_name2id(mp, mujoco.mjtObj.mjOBJ_BODY, cfg.body_name)
        if bid < 0:
            raise ValueError(f"BuiltinSensor: body '{cfg.body_name}' not found.")
        self._body_id = bid

    @property
    def data(self) -> torch.Tensor:
        return self._data

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None or env_ids.numel() == 0:
            return
        self._data[env_ids.long().to(self.device)] = 0.0

    def update(self, dt: float | None = None) -> None:
        if self._newton:
            vals, names = self.env.handler.get_subtree_field(self.cfg.field)
            if self._newton_body_col is None:
                self._newton_body_col = next(
                    (
                        j
                        for j, n in enumerate(names)
                        if n == self.cfg.body_name
                        or n.endswith("_" + self.cfg.body_name)
                        or n.endswith(self.cfg.body_name)
                    ),
                    -1,
                )
            col = self._newton_body_col
            if 0 <= col < vals.shape[1]:
                self._data.copy_(vals[:, col, :])
            else:
                self._data.zero_()
            return
        physics = self.env.handler.physics
        d = physics.data
        f = getattr(d, self.cfg.field, None)
        if f is None:
            return
        val = np.asarray(f[self._body_id], dtype=np.float32)
        self._data.copy_(torch.from_numpy(val).unsqueeze(0).expand(self.env.num_envs, -1))
