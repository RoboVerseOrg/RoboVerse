# helper/state_writer.py
from __future__ import annotations

import jax
import jax.numpy as jnp
import torch
from mujoco import mjtJoint

from metasim.types import ObjectState, RobotState


def t2j(arr: torch.Tensor, device: str | torch.device | None = "cuda") -> jnp.ndarray:
    """Torch → JAX (keeps data on the requested device)."""
    if device is not None and arr.device != torch.device(device):
        arr = arr.to(device, non_blocking=True)
    x = jax.dlpack.from_dlpack(torch.utils.dlpack.to_dlpack(arr))
    return x


def j2t(a: jax.Array, device="cuda") -> torch.Tensor:
    """JAX → Torch (keeps data on the requested device)."""
    if device:
        tgt = torch.device(device)
        plat = "gpu" if tgt.type == "cuda" else tgt.type
        if a.device.platform != plat:
            a = jax.device_put(a, jax.devices(plat)[tgt.index or 0])
    return torch.from_dlpack(jax.dlpack.to_dlpack(a))


# -----------------------------------------------------------------------------
#  Root joint writer
# -----------------------------------------------------------------------------
def _write_root_joint(
    qpos: jnp.ndarray,
    qvel: jnp.ndarray,
    idx: jnp.ndarray,
    model,
    root_jid: int,
    root_state: ObjectState | RobotState,  # (N, 13) pose + vel
    *,
    zero_vel: bool = True,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Copy a free / hinge / slide root joint from *root_state* into the
    batched MJX buffers.

    * free  joint  ➜ 7-DoF pose  +  6-DoF vel
    * hinge/slide ➜ 1-DoF pos   +  1-DoF vel

    If *zero_vel* is True, velocities are zeroed regardless of input.
    """
    idx_list = idx.tolist()

    jtype = model.jnt_type[root_jid]
    qadr = model.jnt_qposadr[root_jid]
    vadr = model.jnt_dofadr[root_jid]

    if jtype == mjtJoint.mjJNT_FREE:  # 6-DOF base
        qpos_vals = t2j(root_state[idx_list, :7])  # (N,7)
        qpos = qpos.at[idx, qadr : qadr + 7].set(qpos_vals)

        vel_vals = 0.0 if zero_vel else t2j(root_state[idx_list, 7:13])
        qvel = qvel.at[idx, vadr : vadr + 6].set(vel_vals)

    elif jtype in (mjtJoint.mjJNT_HINGE, mjtJoint.mjJNT_SLIDE):
        qpos = qpos.at[idx, qadr].set(t2j(root_state[idx_list, 0]))

        vel_val = 0.0 if zero_vel else t2j(root_state[idx_list, 7])
        qvel = qvel.at[idx, vadr].set(vel_val)

    return qpos, qvel


# -----------------------------------------------------------------------------
#  Articulated block writer (non-root joints)
# -----------------------------------------------------------------------------
def _write_articulated_block(
    qpos: jnp.ndarray,
    qvel: jnp.ndarray,
    ctrl: jnp.ndarray,
    idx: jnp.ndarray,
    model,
    joint_ids: jnp.ndarray,
    actuator_ids,
    joint_pos,
    joint_vel,
    joint_target,
    *,
    zero_vel: bool = True,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Scatter non-root joint positions / velocities (and optional ctrl targets)
    into the MJX batched buffers.

    Early-return if there is nothing to write (empty *joint_ids* or *joint_pos* is None).
    """
    if joint_ids.size == 0 or joint_pos is None:
        return qpos, qvel, ctrl

    # Vectorised address lookup for these joints
    qadr = model.jnt_qposadr[joint_ids]  # (J,)
    vadr = model.jnt_dofadr[joint_ids]  # (J,)

    # Torch tensors must be sliced with a Python list
    idx_list = idx.tolist()  # e.g. [0, 2, 5]

    # --- qpos ---------------------------------------------------------
    qpos_vals = t2j(joint_pos[idx_list])  # (N,J)
    qpos = qpos.at[idx[:, None], qadr].set(qpos_vals)

    # --- qvel ---------------------------------------------------------
    vel_vals = jnp.zeros_like(t2j(joint_vel[idx_list])) if zero_vel else t2j(joint_vel[idx_list])
    qvel = qvel.at[idx[:, None], vadr].set(vel_vals)

    # --- ctrl targets -------------------------------------------------
    if actuator_ids is not None and joint_target is not None:
        tgt_vals = t2j(joint_target[idx_list])
        ctrl = ctrl.at[idx[:, None], actuator_ids].set(tgt_vals)

    return qpos, qvel, ctrl


def process_entity(
    name: str,
    state: RobotState | ObjectState,
    *,
    qpos: jnp.ndarray,
    qvel: jnp.ndarray,
    ctrl: jnp.ndarray,
    idx: jnp.ndarray,
    model,
    joint_id_map: dict[str, jnp.ndarray],
    actuator_id_map: dict[str, jnp.ndarray],
    fixed_root_names: set[str],
    zero_vel: bool = True,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Write one robot/object state back into MJX buffers."""
    joint_ids = joint_id_map[name]
    root_fixed = name in fixed_root_names

    # ----- root joint -----
    if root_fixed:
        non_root_joint_ids = joint_ids
    else:
        root_jid = int(joint_ids[0])
        qpos, qvel = _write_root_joint(qpos, qvel, idx, model, root_jid, state.root_state, zero_vel=zero_vel)
        non_root_joint_ids = joint_ids[1:] if joint_ids.size > 1 else jnp.empty(0, int)

    # ----- articulated part -----
    if state.joint_pos is not None and non_root_joint_ids.size > 0:
        qpos, qvel, ctrl = _write_articulated_block(
            qpos,
            qvel,
            ctrl,
            idx,
            model,
            non_root_joint_ids,
            actuator_id_map.get(name),
            state.joint_pos,
            state.joint_vel,
            getattr(state, "joint_pos_target", None),
            zero_vel=zero_vel,
        )

    return qpos, qvel, ctrl


_KIND_META = {  # (size-field, adr-field) for each name-pool category
    "joint": ("njnt", "name_jntadr"),
    "actuator": ("nu", "name_actuatoradr"),
    "body": ("nbody", "name_bodyadr"),
}


def _decode_name(pool: bytes, adr: int) -> str:
    """Return the C-string at `adr` inside MuJoCo name pool `pool`."""
    end = pool.find(b"\x00", adr)  # names are null-terminated
    return pool[adr:end].decode()


def _names_ids_mjx(model, kind: str):
    """
    Fetch all names (strings) and their indices (ints) for a given `kind`
    from an mjx model.  `kind` ∈ {"joint", "actuator", "body"}.
    """
    size_attr, adr_attr = _KIND_META[kind]
    size = int(getattr(model, size_attr))
    adr_arr = getattr(model, adr_attr)  # pointer array into the name-pool
    pool = model.names

    names = [_decode_name(pool, int(adr_arr[i])) for i in range(size)]
    ids = list(range(size))
    return names, ids  # parallel lists


def sorted_joint_info(model, prefix: str):
    """
    Return (qadr, vadr) address arrays for all joints whose full name starts
    with ``prefix``, sorted alphabetically.

    Example
    -------
    Joints:
        panda_joint1
        panda_joint2
        panda_joint_finger1

    After sorting::
        panda_joint1 → panda_joint2 → panda_joint_finger1
    """
    names, ids = _names_ids_mjx(model, "joint")
    matches = [(n, i) for n, i in zip(names, ids) if n.startswith(prefix)]
    if not matches:
        raise ValueError(f"No joints begin with '{prefix}'")
    matches.sort(key=lambda t: t[0])  # alpha order by full name
    _, j_ids = zip(*matches)
    qadr = model.jnt_qposadr[list(j_ids)]
    vadr = model.jnt_dofadr[list(j_ids)]
    return jnp.asarray(qadr), jnp.asarray(vadr)


def sorted_actuator_ids(model, prefix: str):
    """Return **sorted** actuator IDs whose name begins with `prefix`."""
    names, ids = _names_ids_mjx(model, "actuator")
    sel = [(n, i) for n, i in zip(names, ids) if n.startswith(prefix)]
    sel.sort(key=lambda t: t[0])
    return [i for _, i in sel]


def sorted_body_ids(model, prefix: str):
    """
    Return ``(body_ids, local_names)`` for every body whose full name starts
    with ``prefix/`` (excluding the root ``prefix`` body itself), sorted
    alphabetically by full name.

    Example
    -------
    Bodies:
        panda_link0
        panda_link3
        panda_left_finger

    After sorting::
        panda_link0  → panda_link3 → panda_left_finger
    """
    names, ids = _names_ids_mjx(model, "body")
    filt = [(n, i) for n, i in zip(names, ids) if n.startswith(prefix) and n != prefix]
    filt.sort(key=lambda t: t[0])
    body_ids = [i for _, i in filt]
    local_names = [n.split("/")[-1] for n, _ in filt]
    return body_ids, local_names
