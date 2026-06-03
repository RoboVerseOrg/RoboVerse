"""Grasp + success evaluation for the SimplerEnv coke-can task — vendored, no upstream dep.

Reproduces, as standalone pure-SAPIEN/numpy code:
  * ``GoogleRobot.check_grasp`` (mani_skill2_real2sim/agents/robots/googlerobot.py @ ef7a4d4)
    — per-finger contact-impulse + impulse-vs-open-direction angle test.
  * the contact-impulse helpers (utils/sapien_utils.py: get_pairwise_contacts /
    compute_total_impulse / get_pairwise_contact_impulse).
  * the coke-can ``evaluate`` success rule (envs/custom_scenes/grasp_single_in_scene.py):
    consecutive_grasp >= 5 OR (lifted with no non-robot contact, lift > 0.02m accumulated
    >= 5 significant lifts), require_lifting_obj_for_success + success_from_episode_stats.

So the native port computes ``info["success"]`` identically to upstream with NO dependency
on ``mani_skill2_real2sim``. All functions take live SAPIEN objects (contacts, links, actor).
"""

from __future__ import annotations

import numpy as np

from .control._vendor_utils import compute_angle_between

# ----------------------------------------------------------------------------- #
# Contact-impulse helpers (verbatim logic from utils/sapien_utils.py)
# ----------------------------------------------------------------------------- #


def get_pairwise_contacts(contacts, actor0, actor1, collision_shape0=None, collision_shape1=None):
    pairwise = []
    for contact in contacts:
        if collision_shape0 is not None and contact.collision_shape0 != collision_shape0:
            continue
        if collision_shape1 is not None and contact.collision_shape1 != collision_shape1:
            continue
        if contact.actor0 == actor0 and contact.actor1 == actor1:
            pairwise.append((contact, True))
        elif contact.actor0 == actor1 and contact.actor1 == actor0:
            pairwise.append((contact, False))
    return pairwise


def compute_total_impulse(contact_infos):
    total = np.zeros(3)
    for contact, flag in contact_infos:
        impulse = np.sum([point.impulse for point in contact.points], axis=0)
        total += impulse * (1 if flag else -1)
    return total


def get_pairwise_contact_impulse(contacts, actor0, actor1):
    return compute_total_impulse(get_pairwise_contacts(contacts, actor0, actor1))


# ----------------------------------------------------------------------------- #
# GoogleRobot grasp check
# ----------------------------------------------------------------------------- #

FINGER_LINK_NAMES = {
    "finger_right": "link_finger_right",
    "finger_right_tip": "link_finger_tip_right",
    "finger_left": "link_finger_left",
    "finger_left_tip": "link_finger_tip_left",
}


class GoogleRobotGraspChecker:
    """Replicates GoogleRobot.check_grasp on a SAPIEN articulation + scene."""

    def __init__(self, robot, scene):
        self.robot = robot
        self.scene = scene
        links = {x.get_name(): x for x in robot.get_links()}
        self.finger_right_link = links[FINGER_LINK_NAMES["finger_right"]]
        self.finger_right_tip_link = links[FINGER_LINK_NAMES["finger_right_tip"]]
        self.finger_left_link = links[FINGER_LINK_NAMES["finger_left"]]
        self.finger_left_tip_link = links[FINGER_LINK_NAMES["finger_left_tip"]]

    def check_grasp(self, actor, min_impulse=1e-6, max_angle=80) -> bool:
        contacts = self.scene.get_contacts()
        limpulse_tip = get_pairwise_contact_impulse(contacts, self.finger_left_tip_link, actor)
        limpulse_finger = get_pairwise_contact_impulse(contacts, self.finger_left_link, actor)
        rimpulse_tip = get_pairwise_contact_impulse(contacts, self.finger_right_tip_link, actor)
        rimpulse_finger = get_pairwise_contact_impulse(contacts, self.finger_right_link, actor)

        ldirection_tip = self.finger_left_tip_link.pose.to_transformation_matrix()[:3, 1]
        ldirection_finger = self.finger_left_link.pose.to_transformation_matrix()[:3, 1]
        rdirection_tip = self.finger_right_tip_link.pose.to_transformation_matrix()[:3, 1]
        rdirection_finger = self.finger_right_link.pose.to_transformation_matrix()[:3, 1]

        langle = min(
            compute_angle_between(ldirection_tip, limpulse_tip),
            compute_angle_between(ldirection_finger, limpulse_finger),
        )
        rangle = min(
            compute_angle_between(rdirection_tip, rimpulse_tip),
            compute_angle_between(rdirection_finger, rimpulse_finger),
        )
        lflag = (max(np.linalg.norm(limpulse_tip), np.linalg.norm(limpulse_finger)) >= min_impulse) and np.rad2deg(
            langle
        ) <= max_angle
        rflag = (max(np.linalg.norm(rimpulse_tip), np.linalg.norm(rimpulse_finger)) >= min_impulse) and np.rad2deg(
            rangle
        ) <= max_angle
        return bool(lflag and rflag)


# ----------------------------------------------------------------------------- #
# Coke-can pick success evaluation (grasp_single_in_scene.py::evaluate)
# ----------------------------------------------------------------------------- #


class PickSuccessEvaluator:
    """Reproduces GraspSingle...InScene.evaluate success bookkeeping across an episode.

    Defaults match the coke-can env: require_lifting_obj_for_success=True,
    success_from_episode_stats=True.
    """

    def __init__(self, require_lifting_obj_for_success=True, success_from_episode_stats=True):
        self.require_lifting = require_lifting_obj_for_success
        self.from_stats = success_from_episode_stats
        self.reset(0.0)

    def reset(self, obj_height_after_settle: float):
        self.obj_height_after_settle = obj_height_after_settle
        self.consecutive_grasp = 0
        self.lifted_obj = False
        self.episode_stats = {"n_lift_significant": 0, "consec_grasp": False, "grasped": False}

    def step(self, *, is_grasped: bool, obj_z: float, contacts, obj, robot_link_names) -> dict:
        if is_grasped:
            self.consecutive_grasp += 1
        else:
            self.consecutive_grasp = 0
            self.lifted_obj = False

        # no contact with non-robot actors (impulse > 1e-6) => "flag"
        flag = True
        for contact in contacts:
            a0, a1 = contact.actor0, contact.actor1
            other = None
            if a0.name == obj.name:
                other = a1.name
            elif a1.name == obj.name:
                other = a0.name
            if other is not None:
                impulse = np.sum([p.impulse for p in contact.points], axis=0)
                if (other not in robot_link_names) and (np.linalg.norm(impulse) > 1e-6):
                    flag = False
                    break

        consecutive_grasp = self.consecutive_grasp >= 5
        diff_h = obj_z - self.obj_height_after_settle
        self.lifted_obj = self.lifted_obj or (flag and (diff_h > 0.01))
        lifted_significantly = self.lifted_obj and (diff_h > 0.02)

        success = self.lifted_obj if self.require_lifting else consecutive_grasp
        self.episode_stats["n_lift_significant"] += int(lifted_significantly)
        self.episode_stats["consec_grasp"] = self.episode_stats["consec_grasp"] or consecutive_grasp
        self.episode_stats["grasped"] = self.episode_stats["grasped"] or is_grasped
        if self.from_stats:
            success = success or (self.episode_stats["n_lift_significant"] >= 5)

        return {
            "is_grasped": is_grasped,
            "consecutive_grasp": consecutive_grasp,
            "lifted_object": self.lifted_obj,
            "lifted_object_significantly": lifted_significantly,
            "success": bool(success),
            "episode_stats": dict(self.episode_stats),
        }
