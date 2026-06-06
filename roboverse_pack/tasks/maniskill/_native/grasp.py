"""ManiSkill-faithful ``is_grasped`` for the Panda, on the sapien3 contact query.

Mirrors ``mani_skill.agents.robots.panda.Panda.is_grasping``: the object is grasped when *both*
fingers contact it with force ≥ ``min_force`` and the contact-force direction is within
``max_angle`` of the finger's closing axis. Uses the handler's
:meth:`Sapien3Handler.get_pairwise_contact_force`, so it needs no runtime ``mani_skill`` import.
"""

from __future__ import annotations

import numpy as np

_MIN_FORCE = 0.5
_MAX_ANGLE = 85.0


def _link_by_name(robot, name):
    for link in robot.get_links():
        if link.get_name() == name:
            return link
    return None


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 180.0
    cos = float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
    return float(np.rad2deg(np.arccos(cos)))


def grasp_forces(handler, object_name: str, robot_name: str = "panda"):
    """Return ``(left_force_vec, right_force_vec)`` (N) between each finger and the object."""
    import sapien.physx as physx

    robot = handler.object_ids[robot_name]
    f1 = _link_by_name(robot, "panda_leftfinger")
    f2 = _link_by_name(robot, "panda_rightfinger")
    obj = handler.object_ids[object_name]
    body = obj.find_component_by_type(physx.PhysxRigidDynamicComponent)
    if body is None:
        body = obj.find_component_by_type(physx.PhysxRigidStaticComponent)
    lf = handler.get_pairwise_contact_force(f1, body)
    rf = handler.get_pairwise_contact_force(f2, body)
    return lf, rf, f1, f2


def is_grasped(
    handler, object_name: str, robot_name: str = "panda", min_force: float = _MIN_FORCE, max_angle: float = _MAX_ANGLE
) -> bool:
    """True when the Panda is grasping ``object_name`` (ManiSkill criterion)."""
    lf, rf, f1, f2 = grasp_forces(handler, object_name, robot_name)
    ldir = f1.get_pose().to_transformation_matrix()[:3, 1]
    rdir = -f2.get_pose().to_transformation_matrix()[:3, 1]
    lflag = np.linalg.norm(lf) >= min_force and _angle_deg(ldir, lf) <= max_angle
    rflag = np.linalg.norm(rf) >= min_force and _angle_deg(rdir, rf) <= max_angle
    return bool(lflag and rflag)
