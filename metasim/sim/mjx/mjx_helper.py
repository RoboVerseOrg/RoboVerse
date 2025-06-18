# helper/state_writer.py
from __future__ import annotations

from functools import partial
import functools
import jax
import jax.numpy as jnp
import torch
from mujoco import mjtJoint
import numpy as np

from metasim.types import ObjectState, RobotState


# -----------------------------------------------------------------------------  
#  Torch ↔ JAX conversion helpers
# -----------------------------------------------------------------------------
def t2j(arr: torch.Tensor, device: str | torch.device | None = "cuda") -> jnp.ndarray:
    """Convert a PyTorch tensor to a JAX array, keeping it on the requested device."""
    if device is not None and arr.device != torch.device(device):
        arr = arr.to(device, non_blocking=True)
    return jax.dlpack.from_dlpack(torch.utils.dlpack.to_dlpack(arr))


def j2t(a: jax.Array, device="cuda") -> torch.Tensor:
    """Convert a JAX array to a PyTorch tensor, keeping it on the requested device."""
    if device:
        tgt = torch.device(device)
        plat = "gpu" if tgt.type == "cuda" else tgt.type
        if a.device.platform != plat:
            a = jax.device_put(a, jax.devices(plat)[tgt.index or 0])
    return torch.from_dlpack(jax.dlpack.to_dlpack(a))


# -----------------------------------------------------------------------------  
#  Root-joint writer  (STATIC ints as compile-time constants)
# -----------------------------------------------------------------------------
@functools.partial(
    jax.jit,
    static_argnums=(3, 4, 5),          # jtype, qadr, vadr are compile-time constants
    static_argnames=("zero_vel",),
)
def _write_root_joint(
    qpos: jnp.ndarray,
    qvel: jnp.ndarray,
    idx:  jnp.ndarray,                 # (N,) env indices
    jtype: int,                        # python int  (joint type)
    qadr:  int,                        # python int  (start addr in qpos)
    vadr:  int,                        # python int  (start addr in qvel)
    root_state: jnp.ndarray,           # (B, 13) pose + vel, already JAX
    *, 
    zero_vel: bool = True,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Scatter root-joint pose/vel into batched MJX buffers."""

    if jtype == mjtJoint.mjJNT_FREE:            # 6-DoF root
        # pose (quat + xyz)
        qpos_vals = root_state[idx, :7]         # (N,7)
        qpos = qpos.at[idx, qadr : qadr + 7].set(qpos_vals)

        # velocity (angVel + linVel)
        vel_vals = 0.0 if zero_vel else root_state[idx, 7:13]
        qvel = qvel.at[idx, vadr : vadr + 6].set(vel_vals)

    elif jtype in (mjtJoint.mjJNT_HINGE, mjtJoint.mjJNT_SLIDE):
        # single-DoF root
        qpos = qpos.at[idx, qadr].set(root_state[idx, 0])

        vel_val = 0.0 if zero_vel else root_state[idx, 7]
        qvel = qvel.at[idx, vadr].set(vel_val)

    return qpos, qvel


@functools.partial(
    jax.jit,
    static_argnums=(4, 5, 6),              # qadr_list, vadr_list, act_list
    static_argnames=("zero_vel",),
)
def _write_articulated_block(
    qpos: jnp.ndarray,
    qvel: jnp.ndarray,
    ctrl: jnp.ndarray,
    idx:  jnp.ndarray,                     # (N,)
    qadr_list: tuple[int, ...],            # static pose addresses
    vadr_list: tuple[int, ...],            # static vel  addresses
    act_list: tuple[int, ...]|None,   # static actuator ids / None
    joint_pos: jnp.ndarray | None,
    joint_vel: jnp.ndarray | None,
    joint_target: jnp.ndarray | None,
    *,
    zero_vel: bool = True,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:

    if joint_pos is None or len(qadr_list) == 0:
        return qpos, qvel, ctrl

    qadr = jnp.asarray(qadr_list, dtype=jnp.int32)
    vadr = jnp.asarray(vadr_list, dtype=jnp.int32)

    # positions
    qpos_vals = joint_pos[idx]                         # (N, J)
    qpos = qpos.at[idx[:, None], qadr].set(qpos_vals)

    # velocities
    vel_vals = jnp.zeros_like(joint_vel[idx]) if zero_vel else joint_vel[idx]
    qvel = qvel.at[idx[:, None], vadr].set(vel_vals)

    # actuator targets
    if act_list is not None and joint_target is not None:
        act_ids = jnp.asarray(act_list, dtype=jnp.int32)
        tgt_vals = joint_target[idx]
        ctrl = ctrl.at[idx[:, None], act_ids].set(tgt_vals)

    return qpos, qvel, ctrl



def process_entity(
    name: str,
    state,                                     # RobotState | ObjectState (torch)
    qpos: jnp.ndarray,
    qvel: jnp.ndarray,
    ctrl: jnp.ndarray,
    idx:  jnp.ndarray,
    model,                                     # mjx.Model
    joint_id_map: dict[str, jnp.ndarray],
    actuator_id_map: dict[str, jnp.ndarray],
    fixed_root_names: set[str],
    *,
    zero_vel: bool = True,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:

    root_state_jax = t2j(state.root_state)
    joint_pos_jax  = t2j(state.joint_pos)        if state.joint_pos        is not None else None
    joint_vel_jax  = t2j(state.joint_vel)        if state.joint_vel        is not None else None
    joint_tgt_jax  = t2j(state.joint_pos_target) if getattr(state, "joint_pos_target", None) is not None else None

    joint_ids  = jnp.asarray(joint_id_map[name])
    root_fixed = name in fixed_root_names

    # static model arrays (NumPy)
    jnt_type_np    = model.jnt_type
    jnt_qposadr_np = model.jnt_qposadr
    jnt_dofadr_np  = model.jnt_dofadr


    if root_fixed:
        non_root_joint_ids = joint_ids
    else:
        root_jid   = int(joint_ids[0])
        root_jtype = int(jnt_type_np[root_jid])
        root_qadr  = int(jnt_qposadr_np[root_jid])
        root_vadr  = int(jnt_dofadr_np[root_jid])

        qpos, qvel = _write_root_joint(
            qpos, qvel, idx,
            root_jtype, root_qadr, root_vadr,
            root_state_jax,
            zero_vel=zero_vel,
        )

        non_root_joint_ids = (
            joint_ids[1:] if joint_ids.size > 1 else jnp.empty(0, int)
        )

    if joint_pos_jax is not None and non_root_joint_ids.size > 0:

        qadr_list = tuple(int(jnt_qposadr_np[j]) for j in non_root_joint_ids)
        vadr_list = tuple(int(jnt_dofadr_np [j]) for j in non_root_joint_ids)
        act_ids   = actuator_id_map.get(name)
        act_list  = None if act_ids is None else tuple(int(a) for a in act_ids)
        qpos, qvel, ctrl = _write_articulated_block(
            qpos, qvel, ctrl,
            idx,
            qadr_list,
            vadr_list,
            act_list,
            joint_pos_jax,
            joint_vel_jax,
            joint_tgt_jax,
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



@functools.partial(jax.jit, static_argnums=())
def pack_body_state(data, env_idx, body_ids):
    """
    Build a (N, B, 13) tensor for the selected bodies.

    Parameters
    ----------
    data      : mjx_env.Data          shape (N, …)
    env_idx   : jnp.ndarray[int32]    shape (N,)   indices of environments
    body_ids  : jnp.ndarray[int32]    shape (B,)   body IDs to extract

    Returns
    -------
    jnp.ndarray  shape (N, B, 13)
        [pos(3), quat(4), lin_vel_world(3), ang_vel_world(3)]
    """
    # position and orientation
    pos  = data.xpos[env_idx[:, None], body_ids]          # (N, B, 3)
    quat = data.xquat[env_idx[:, None], body_ids]         # (N, B, 4)

    # angular and linear velocities in world
    w    = data.cvel[env_idx[:, None], body_ids, 0:3]     # (N, B, 3)
    v_cm = data.cvel[env_idx[:, None], body_ids, 3:6]     # (N, B, 3)

    # convert linear vel from COM frame to world origin
    offset = pos - data.subtree_com[env_idx[:, None], body_ids]  # (N, B, 3)
    v_org  = v_cm + jnp.cross(w, offset)                         # (N, B, 3)

    return jnp.concatenate([pos, quat, v_org, w], axis=-1)       # (N, B, 13)


# ------------------------------------------------------------- pack_root_state
def _pack_root_state(data, env_idx, bid):
    """Return (N, 13) root body state for a single body id."""
    return pack_body_state(data, env_idx, jnp.asarray([bid]))[:, 0]

# jit-compile with bid (scalar) static so it does not recompile per call
pack_root_state = jax.jit(_pack_root_state, static_argnums=(2,))