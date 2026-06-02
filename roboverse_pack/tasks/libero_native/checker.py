"""Ported LIBERO BDDL success checker — evaluates the goal with NO ``libero`` import.

LIBERO checks task success by AND-ing the BDDL ``goal_state`` predicates against
the live object states. This module reimplements the geometric primitives those
predicates reduce to, reading them straight from a MuJoCo ``model``/``data`` pair
(e.g. a MetaSim handler's physics). It is the piece that lets a LIBERO task be
*evaluated* inside MetaSim without the upstream library.

Predicate reduction (from ``libero/libero/envs``):

* ``In(obj, region_site)`` — the region is a ``SiteObjectState`` whose
  ``check_contact`` is always ``True`` (no dynamics for site objects), so it
  reduces to ``in_box``: the object's body position lies inside the region site's
  oriented box (``site_pos ± |site_mat @ site_half_size|``, lower z extended by
  1 cm) — verbatim from ``objects/site_object.py::in_box``.
* ``On(a, b)`` with ``b`` an object — ``ObjectState.check_ontop``: the surface ``b``
  sits below the placed object ``a`` (``z[b] <= z[a]``) AND their ``contact_geoms``
  touch AND ``||a.xy - b.xy|| < 0.03``.
* ``On(a, b)`` with ``b`` a region site — ``SiteObjectState.check_ontop`` via
  ``SiteObject.under``: ``a``'s body lies in the site's oriented box above its floor
  (``size[2]-0.005 < (mat·(a-site))_z < size[2]+0.10`` and ``|·xy| < size[:2]``) AND
  the region's parent object's ``contact_geoms`` touch ``a``'s. Contact uses the
  exact robosuite ``contact_geoms`` sets (resolved to geom ids by the exporter).
* ``Open/Close/TurnOn/TurnOff(obj)`` — articulated-joint predicates. LIBERO reads
  each of the object's joint qpos and compares it to a per-object threshold
  (``articulated_objects.py``); ``Open``/``TurnOn`` are satisfied if **any** joint
  passes, ``Close``/``TurnOff`` only if **all** joints pass. The export step probes
  the live object's own ``is_open``/``is_close``/``turn_on``/``turn_off`` method to
  recover the exact ``(op, threshold)`` per object + the qpos addresses, so this
  checker reduces the predicate to ``op(qpos[addr], thr)`` with no libero import.

A goal is exported (see ``scripts/native/export_libero_task.py``) as a list of
predicate dicts with model names already resolved, so this checker needs only the
names + the MuJoCo state.
"""

from __future__ import annotations

import operator

import mujoco
import numpy as np

# comparison operators recovered by the exporter when probing the articulated
# object's own threshold method (e.g. ``qpos < max(open_ranges)`` -> "lt").
_OPS = {"lt": operator.lt, "le": operator.le, "gt": operator.gt, "ge": operator.ge}


def _site(model, data, name):
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    if sid < 0:
        raise KeyError(f"site {name!r} not in model")
    return np.array(data.site_xpos[sid]), np.array(data.site_xmat[sid]).reshape(3, 3), np.array(model.site_size[sid])


def _body_xpos(model, data, name):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if bid < 0:
        raise KeyError(f"body {name!r} not in model")
    return np.array(data.xpos[bid])


def in_box(site_pos, site_mat, site_half_size, other_pos) -> bool:
    """Verbatim LIBERO ``SiteObject.in_box`` (axis-aligned-ish containment test)."""
    total = np.abs(site_mat @ site_half_size)
    ub = site_pos + total
    lb = site_pos - total
    lb[2] -= 0.01
    return bool(np.all(other_pos > lb) and np.all(other_pos < ub))


def _in_contact(data, geoms_a, geoms_b) -> bool:
    """Robosuite ``check_contact``: any active contact between the two geom-id sets."""
    a, b = set(geoms_a), set(geoms_b)
    for i in range(data.ncon):
        c = data.contact[i]
        if (c.geom1 in a and c.geom2 in b) or (c.geom1 in b and c.geom2 in a):
            return True
    return False


def under(site_pos, site_mat, site_size, other_pos) -> bool:
    """Verbatim LIBERO ``SiteObject.under`` (object resting in a region site box)."""
    delta = site_mat @ (other_pos - site_pos)
    return bool(site_size[2] - 0.005 < delta[2] < site_size[2] + 0.10 and np.all(np.abs(delta[:2]) < site_size[:2]))


def eval_predicate(model, data, p) -> bool:
    """Evaluate one exported predicate dict against the current MuJoCo state."""
    fn = p["fn"]
    if fn == "in":  # object inside a region site
        sp, sm, ss = _site(model, data, p["region"])
        return in_box(sp, sm, ss, _body_xpos(model, data, p["obj"]))
    if fn == "on":  # placed object `obj` (arg1) resting on surface object `obj2` (arg2)
        a, b = _body_xpos(model, data, p["obj"]), _body_xpos(model, data, p["obj2"])
        return (
            bool(b[2] <= a[2])  # surface below the placed object (ObjectState.check_ontop)
            and _in_contact(data, p["obj_geoms"], p["obj2_geoms"])
            and float(np.linalg.norm(a[:2] - b[:2])) < 0.03
        )
    if fn == "on_site":  # placed object `obj` (arg1) resting in region site `site` (arg2)
        sp, sm, ss = _site(model, data, p["site"])
        if not under(sp, sm, ss, _body_xpos(model, data, p["obj"])):
            return False
        # a region with no backing object (table region) has no contact term
        return p["parent_geoms"] is None or _in_contact(data, p["parent_geoms"], p["obj_geoms"])
    if fn == "joint":  # open / close / turnon / turnoff on an articulated object
        op = _OPS[p["op"]]
        checks = [bool(op(float(data.qpos[addr]), p["thr"])) for addr in p["qpos_addr"]]
        return any(checks) if p["reduce"] == "any" else all(checks)
    raise NotImplementedError(f"predicate {fn!r} not ported")


def check_success(model, data, goal) -> bool:
    """AND over all goal predicates (LIBERO ``_check_success`` semantics)."""
    return all(eval_predicate(model, data, p) for p in goal)
