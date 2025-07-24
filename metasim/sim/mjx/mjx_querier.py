# mjx_query_helper.py

import jax
import jax.numpy as jnp
import mujoco
import torch

from metasim.cfg.query_type import ContactForce, SitePos


class MJXQuerier:
    """
    Add new Query types by inserting into QUERY_MAP.
    """

    _site_cache = {}
    _body_cache = {}

    QUERY_MAP = {
        SitePos: "site_pos",
        ContactForce: "contact_force",
        # add here
    }

    # public entry ------------------------------------------------------------
    @classmethod
    def query(cls, q, handler, robot_name=None):
        fn_name = cls.QUERY_MAP[type(q)]
        return j2t(getattr(cls, fn_name)(q, handler, robot_name))

        # query func collection ---------------------------------------------------

    @classmethod
    def site_pos(cls, q: SitePos, handler, robot_name):
        """Return (N_env, 3) site position."""
        key = id(handler._mj_model)
        cache = cls._site_cache.setdefault(key, {})

        if not cache:
            cache.update({
                mujoco.mj_id2name(handler._mj_model, mujoco.mjtObj.mjOBJ_SITE, i): i
                for i in range(handler._mj_model.nsite)
                if mujoco.mj_id2name(handler._mj_model, mujoco.mjtObj.mjOBJ_SITE, i)
            })

        full_name = f"{robot_name}/{q.site}" if "/" not in q.site else q.site
        sid = cache[full_name]

        return handler._data.site_xpos[:, sid]

    @classmethod
    def contact_force(cls, q: ContactForce, handler, robot_name):
        """Return (N_env, 6) force torque of one body."""
        key = id(handler._mj_model)
        cache = cls._body_cache.setdefault(key, {})
        if not cache:
            cache.update({
                mujoco.mj_id2name(handler._mj_model, mujoco.mjtObj.mjOBJ_BODY, i): i
                for i in range(handler._mj_model.nbody)
                if mujoco.mj_id2name(handler._mj_model, mujoco.mjtObj.mjOBJ_BODY, i)
            })

        full_name = f"{robot_name}/{q.sensor_name}" if "/" not in q.site else q.site
        bid = cache[full_name]

        contact_force = handler._data.cfrc_ext[:, jnp.asarray([bid], jnp.int32)]
        return contact_force[:, 0, :]  # (N_env, 6)


def j2t(a: jax.Array, device="cuda") -> torch.Tensor:
    """Convert a JAX array to a PyTorch tensor, keeping it on the requested device."""
    if device:
        tgt = torch.device(device)
        plat = "gpu" if tgt.type == "cuda" else tgt.type
        if a.device.platform != plat:
            a = jax.device_put(a, jax.devices(plat)[tgt.index or 0])
    return torch.from_dlpack(jax.dlpack.to_dlpack(a))


# -----------------------------------------------------------------------------
# usage in handler
# -----------------------------------------------------------------------------
# value = MJXQuerier.query(query_obj, handler)
