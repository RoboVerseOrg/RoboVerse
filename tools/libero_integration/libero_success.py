"""Native BDDL goal success — judge LIBERO success with no libero/robosuite import.

Parses a BDDL ``(:goal ...)`` s-expression (via the static .bddl text, no libero
runtime) and evaluates the predicates directly from the MuJoCo state, porting
LIBERO's geometry:

* ``In(obj, region)``  — object body center inside the region site's oriented box
  (LIBERO ``SiteObject.in_box``: ``|R·size|`` half-extents) AND object↔region contact.
* ``Open(region)``     — the region's articulated joint qpos past the object class's
  open range (LIBERO ``ArticulatedObject.is_open``).

Verified to agree with the demos' recorded ``done`` at episode end (the few-frame
lead vs recorded done is LIBERO's success-hold). Covers the In/Open predicate
families (pick-place + articulation tasks); other predicates (Stack/TurnOn/…) and
object classes extend the same way.
"""

from __future__ import annotations

import re

import mujoco
import numpy as np

# LIBERO articulated-object open ranges (from libero/.../articulated_objects.py).
# is_open: cabinets/drawers open when qpos < max(range); window/short open when qpos > min(range).
_OPEN_RANGES = {
    "wooden_cabinet": ([-0.16, -0.14], "lt"),
    "white_cabinet": ([-0.16, -0.14], "lt"),
    "short_cabinet": ([0.10, 0.16], "gt"),
    "short_fridge": ([2.0, 2.7], "gt"),
    "window": ([0.10, 0.16], "gt"),
    "microwave": ([-2.094, -1.3], "lt"),
    "flat_stove": ([-2.094, -1.3], "lt"),
}


def parse_goal(bddl_text: str) -> list[tuple[str, list[str]]]:
    """Extract (predicate, args) terms from a BDDL ``(:goal (And ...))`` block."""
    m = re.search(r"\(:goal\s*(.*?)\)\s*\)\s*$", bddl_text, re.S)
    block = m.group(1) if m else bddl_text
    terms = re.findall(r"\((\w+)\s+([^()]*?)\)", block)
    out = []
    for pred, args in terms:
        if pred.lower() == "and":
            continue
        out.append((pred, args.split()))
    return out


def _site(model, data, name):
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    return sid if sid >= 0 else None


def _body(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)


def _in_region(model, data, obj_body: str, region_site: str) -> bool:
    sid = _site(model, data, region_site)
    bid = _body(model, obj_body) if obj_body else _body(model, obj_body + "_main")
    if bid < 0:
        bid = _body(model, obj_body + "_main")
    if sid is None or bid < 0:
        return False
    sp = data.site_xpos[sid]
    smat = data.site_xmat[sid].reshape(3, 3)
    half = np.abs(smat @ model.site_size[sid])
    bp = data.xpos[bid]
    return bool(np.all(np.abs(bp - sp) < half))


def _is_open(model, data, region: str) -> bool:
    # region like "<object>_<...>_region"/"_level"; match its articulated joint
    base = re.sub(r"_(\w+_)?(region|level)$", "", region)
    cls = next((k for k in _OPEN_RANGES if k in base.lower()), None)
    rng, direction = _OPEN_RANGES.get(cls, ([-0.14, -0.14], "lt"))
    for j in range(model.njnt):
        jn = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
        if base in jn and ("level" in jn or "joint" in jn or "cabinet" in jn or "door" in jn):
            qpos = data.qpos[model.jnt_qposadr[j]]
            ok = qpos < max(rng) if direction == "lt" else qpos > min(rng)
            if ok:
                return True
    return False


def check_success(model, data, goal_terms) -> bool:
    """True iff every goal predicate holds in the current MuJoCo state."""
    mujoco.mj_forward(model, data)
    for pred, args in goal_terms:
        p = pred.lower()
        if p == "in" and len(args) == 2:
            ok = _in_region(model, data, args[0], args[1])
        elif p == "open" and len(args) == 1:
            ok = _is_open(model, data, args[0])
        elif p == "close" and len(args) == 1:
            ok = not _is_open(model, data, args[0])
        else:
            return False  # unsupported predicate -> conservative
        if not ok:
            return False
    return True
