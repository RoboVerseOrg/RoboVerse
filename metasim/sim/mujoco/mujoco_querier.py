# mjx_query_helper.py

import jax.numpy as jnp
import mujoco

from metasim.cfg.query_type import ContactForce, SitePos


class MujocoQuerier:
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
    def query(cls, q, handler):
        fn_name = cls.QUERY_MAP[type(q)]
        return getattr(cls, fn_name)(q, handler)

    # query func collection ------------------------------------------------------------
    @classmethod
    def site_pos(cls, q: SitePos, handler):
        """Return (N_env, 3) site position."""
        key = id(handler._mj_model)
        cache = cls._site_cache.setdefault(key, {})
        if not cache:
            cache.update({
                mujoco.mj_id2name(handler._mj_model, mujoco.mjtObj.mjOBJ_SITE, i): i
                for i in range(handler._mj_model.nsite)
                if mujoco.mj_id2name(handler._mj_model, mujoco.mjtObj.mjOBJ_SITE, i)
            })
        sid = cache[q.site]
        return handler._data.site_xpos[:, sid]

    @classmethod
    def contact_force(cls, q: ContactForce, handler):
        """Return (N_env, 6) force torque of one body."""
        key = id(handler._mj_model)
        cache = cls._body_cache.setdefault(key, {})
        if not cache:
            cache.update({
                mujoco.mj_id2name(handler._mj_model, mujoco.mjtObj.mjOBJ_BODY, i): i
                for i in range(handler._mj_model.nbody)
                if mujoco.mj_id2name(handler._mj_model, mujoco.mjtObj.mjOBJ_BODY, i)
            })
        bid = cache[q.sensor_name]

        contact_force = handler._data.cfrc_ext[:, jnp.asarray([bid], jnp.int32)]
        return contact_force[:, 0, :]  # (N_env, 6)


# -----------------------------------------------------------------------------
# usage in handler
# -----------------------------------------------------------------------------
# value = MujocoQuerier.query(query_obj, handler)
