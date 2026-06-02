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
* ``On(a, b)`` — ``a.z <= b.z`` AND geom contact between the two bodies AND
  ``||a.xy - b.xy|| < 0.03`` -- verbatim from ``ObjectState.check_ontop``.

A goal is exported (see ``scripts/native/export_libero_task.py``) as a list of
predicate dicts with model names already resolved, so this checker needs only the
names + the MuJoCo state.
"""

from __future__ import annotations

import mujoco
import numpy as np


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


def _body_geom_ids(model, body_name):
    """All geom ids whose body is the named body or any of its descendants."""
    root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    bodies = {root}
    for b in range(model.nbody):  # walk descendants via body_parentid
        p = b
        while p != 0:
            if p == root:
                bodies.add(b)
                break
            p = int(model.body_parentid[p])
    return {g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in bodies}


def _in_contact(model, data, geoms_a, geoms_b) -> bool:
    for i in range(data.ncon):
        c = data.contact[i]
        if (c.geom1 in geoms_a and c.geom2 in geoms_b) or (c.geom1 in geoms_b and c.geom2 in geoms_a):
            return True
    return False


def eval_predicate(model, data, p) -> bool:
    """Evaluate one exported predicate dict against the current MuJoCo state."""
    fn = p["fn"]
    if fn == "in":  # object inside a region site
        sp, sm, ss = _site(model, data, p["region"])
        return in_box(sp, sm, ss, _body_xpos(model, data, p["obj"]))
    if fn == "on":  # object a on top of object b
        a, b = _body_xpos(model, data, p["obj"]), _body_xpos(model, data, p["obj2"])
        ga, gb = _body_geom_ids(model, p["obj"]), _body_geom_ids(model, p["obj2"])
        return bool(a[2] <= b[2]) and _in_contact(model, data, ga, gb) and float(np.linalg.norm(a[:2] - b[:2])) < 0.03
    raise NotImplementedError(f"predicate {fn!r} not ported")


def check_success(model, data, goal) -> bool:
    """AND over all goal predicates (LIBERO ``_check_success`` semantics)."""
    return all(eval_predicate(model, data, p) for p in goal)
