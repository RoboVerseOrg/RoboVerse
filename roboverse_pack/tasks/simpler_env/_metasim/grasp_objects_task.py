"""SimplerEnv pick_object + move_near tasks via the MetaSim API (1 + 3 tasks) — fully ScenarioCfg.

google_robot + google_pick_coke GLB arena + mounted intrinsic camera + the family's FULL candidate
object set are all declared in the ScenarioCfg and loaded by the sapien2 handler; visual-matching
lighting installed on the handler scene. reset selects the per-episode object (pick_object) / triplet
(move_near), positions it/them (fall/settle) and parks the rest out of frame. Reuses the vendored
grasp checker / pick-success evaluator and the multi-object move-near checker.
"""

from __future__ import annotations

import json
import os

import numpy as np
import sapien.core as sapien
from transforms3d.euler import euler2quat
from transforms3d.quaternions import quat2mat

from metasim.scenario.objects import NonConvexRigidObjCfg
from metasim.scenario.scenario import ScenarioCfg

from .._native._assets import paths
from .._native.control import CombinedController
from .._native.grasp import GoogleRobotGraspChecker, PickSuccessEvaluator
from .._native.move_near import MOVE_NEAR_CONFIG, _instruction_obj_name, _xy_configs
from .._native.overlay import load_overlay_image
from .._native.robot_config import google_robot_deployed_controller_configs
from .._native.scene import INIT_QPOS, ROBOT_INIT_HEIGHT, SCENE_OFFSET, SCENE_POSE_Q
from .base import HIDDEN_POS, SimplerMetaSimTask, apply_external_states, candidate_mesh_objects, set_actor_collision
from .drawer_task import _google_robot_cfg, _overhead_camera_cfg, _sim_params

ROBOT_NAME, ARENA_NAME, CAM_NAME = "google_robot", "arena", "overhead_camera"
SCENE_TABLE_HEIGHT, OBJ_FRICTION = 0.87, 0.5
_ORIENT = {
    "upright": euler2quat(np.pi / 2, 0, 0),
    "laid_vertically": euler2quat(0, 0, np.pi / 2),
    "lr_switch": euler2quat(0, 0, np.pi),
}
PICK_OBJECT_IDS = [
    "opened_pepsi_can",
    "opened_coke_can",
    "opened_sprite_can",
    "opened_fanta_can",
    "opened_redbull_can",
    "blue_plastic_bottle",
    "apple",
    "orange",
    "sponge",
    "bridge_spoon_generated_modified",
    "bridge_carrot_generated_modified",
    "green_cube_3cm",
    "yellow_cube_3cm",
    "eggplant",
]


def _google_arena_cfg():
    return NonConvexRigidObjCfg(
        name=ARENA_NAME,
        usd_path=paths()["coke_scene"],
        mesh_pose=[0.0, 0.0, 0.0, *SCENE_POSE_Q],
        default_position=tuple((-SCENE_OFFSET).tolist()),
        default_orientation=(1.0, 0.0, 0.0, 0.0),
        fix_base_link=True,
    )


class _GoogleArenaTask(SimplerMetaSimTask):
    """Shared: google robot + GLB arena + candidate objects + mounted camera + visual-matching lighting."""

    robot_name, cam_name = ROBOT_NAME, CAM_NAME
    sim_freq, control_freq = 513, 3

    def _extra_objects(self):
        return []

    def _build_scenario(self):
        return ScenarioCfg(
            simulator="sapien2",
            robots=[_google_robot_cfg()],
            objects=[_google_arena_cfg(), *self._extra_objects()],
            cameras=[_overhead_camera_cfg()],
            sim_params=_sim_params(),
            add_default_ground=False,
            num_envs=1,
            headless=True,
            gravity=(0, 0, -9.81),
        )

    def _install_lighting(self):
        self.scene.set_ambient_light([0.3, 0.3, 0.3])
        self.scene.add_directional_light([0, 0, -1], [2.2, 2.2, 2.2], shadow=False, scale=5, shadow_map_size=2048)
        self.scene.add_directional_light([-1, -0.5, -1], [0.7, 0.7, 0.7])
        self.scene.add_directional_light([1, 1, -1], [0.7, 0.7, 0.7])

    def _make_controller(self):
        return CombinedController(
            google_robot_deployed_controller_configs(), self.robot, self.control_freq, sim_freq=self.sim_freq
        )

    def _park(self, ids):
        for mid in ids:
            o = self.handler.object_ids[mid]
            o.set_pose(sapien.Pose(HIDDEN_POS))
            o.set_velocity(np.zeros(3))
            o.set_angular_velocity(np.zeros(3))
            o.lock_motion(1, 1, 1, 1, 1, 1)
            set_actor_collision(o, False)  # inactive: no collision -> no perturbation of the active solve


class SimplerPickObjectTask(_GoogleArenaTask):
    max_episode_steps = 80

    def _extra_objects(self):
        with open(os.path.join(paths()["model_db_dir"], "info_pick_custom_v0.json")) as f:
            db = json.load(f)
        dmap = {k: db[k].get("density", 1000) for k in PICK_OBJECT_IDS}
        return candidate_mesh_objects(
            paths()["models_dir"], PICK_OBJECT_IDS, friction=None, damping=0.1, density_map=dmap
        )

    def _wire(self):
        self.grasp_checker = GoogleRobotGraspChecker(self.robot, self.scene)
        self.evaluator = PickSuccessEvaluator()
        self._overlay = load_overlay_image(paths()["coke_overlay"])
        self.obj = None

    def _foreground_ids(self):
        return [link.get_id() for link in self.robot.get_links()] + [self.obj.get_id()]

    def reset(self, states=None, env_ids=None, seed=0, options=None):
        """Reset the task. ``states``/``env_ids``/``seed`` are the ``BaseTaskEnv.reset`` contract;
        ``options`` is the SimplerEnv episode spec from the gym adapter."""
        orientation = np.random.RandomState(seed).choice(list(_ORIENT.keys()))
        self.model_id = PICK_OBJECT_IDS[np.random.RandomState(seed).randint(len(PICK_OBJECT_IDS))]
        self._park(PICK_OBJECT_IDS)
        self.obj = self.handler.object_ids[self.model_id]
        set_actor_collision(self.obj, True)
        self.obj.lock_motion(0, 0, 0, 1, 1, 0)
        rng = np.random.RandomState(seed)
        obj_xy = rng.uniform([-0.35, -0.02], [-0.12, 0.42], [2])
        ix, iy = rng.uniform(0.30, 0.40), rng.uniform(0.0, 0.2)
        self.robot.set_root_pose(sapien.Pose([-10, 0, 0]))
        self.robot.set_qpos(INIT_QPOS)
        self.robot.set_qvel(np.zeros(self.robot.dof))
        self.obj.set_pose(sapien.Pose(np.hstack([obj_xy, SCENE_TABLE_HEIGHT + 0.5]), _ORIENT[orientation]))
        self._settle(0.5)
        self.obj.lock_motion(0, 0, 0, 0, 0, 0)
        self.obj.set_pose(self.obj.pose)
        self.obj.set_velocity(np.zeros(3))
        self.obj.set_angular_velocity(np.zeros(3))
        self._settle(0.5)
        if np.linalg.norm(self.obj.velocity) > 1e-3 or np.linalg.norm(self.obj.angular_velocity) > 1e-2:
            self._settle(1.5)
        self.obj_height_after_settle = float(self.obj.pose.p[2])
        self.robot.set_root_pose(sapien.Pose([ix, iy, ROBOT_INIT_HEIGHT], [0, 0, 0, 1]))
        self.robot.set_qpos(INIT_QPOS)
        self.robot.set_qvel(np.zeros(self.robot.dof))
        if apply_external_states(self, states, env_ids):
            # the success checker keys off the settled height, so re-read it from the applied states
            self.obj_height_after_settle = float(self.obj.pose.p[2])
        self.controller.reset()
        self.evaluator.reset(self.obj_height_after_settle)
        self._elapsed = 0
        return self.get_obs(), {"model_id": self.model_id, "orientation": orientation}

    def _evaluate(self):
        g = self.grasp_checker.check_grasp(self.obj, max_angle=80)
        return self.evaluator.step(
            is_grasped=g,
            obj_z=float(self.obj.pose.p[2]),
            contacts=self.scene.get_contacts(),
            obj=self.obj,
            robot_link_names=[lk.get_name() for lk in self.robot.get_links()],
        )

    def get_language_instruction(self):
        return f"pick {self.model_id}"


_SRC_IDS = [i for i in range(3) for j in range(3) if i != j]
_TGT_IDS = [j for i in range(3) for j in range(3) if i != j]


class SimplerMoveNearTask(_GoogleArenaTask):
    max_episode_steps = 80
    variant = "v1"

    def _candidate_ids(self):
        cfg = MOVE_NEAR_CONFIG[self.variant]
        seen = []
        for trip in cfg["triplets"]:
            for m in trip:
                if m not in seen:
                    seen.append(m)
        return seen

    def _extra_objects(self):
        cfg = MOVE_NEAR_CONFIG[self.variant]
        with open(os.path.join(paths()["model_db_dir"], cfg["model_json"])) as f:
            db = json.load(f)
        ids = self._candidate_ids()
        dmap = {k: cfg["special_density"].get(k, db[k].get("density", 1000)) for k in ids}
        return candidate_mesh_objects(paths()["models_dir"], ids, friction=OBJ_FRICTION, damping=0.1, density_map=dmap)

    def _wire(self):
        cfg = MOVE_NEAR_CONFIG[self.variant]
        with open(os.path.join(paths()["model_db_dir"], cfg["model_json"])) as f:
            self.model_db = json.load(f)
        self.cfg = cfg
        self._overlay = load_overlay_image(os.path.join(paths()["overlay_dir"], "google_move_near_real_eval_1.png"))
        self.objs = []

    def _foreground_ids(self):
        return [link.get_id() for link in self.robot.get_links()] + [o.get_id() for o in self.objs]

    def reset(self, states=None, env_ids=None, seed=0, options=None):
        """Reset the task. ``states``/``env_ids``/``seed`` are the ``BaseTaskEnv.reset`` contract;
        ``options`` is the SimplerEnv episode spec from the gym adapter."""
        n = 5 * len(_SRC_IDS) * 2
        episode_id = np.random.RandomState(seed).randint(n)
        triplet = list(self.cfg["triplets"][episode_id // (len(_SRC_IDS) * 2)])
        self.src_id, self.tgt_id = _SRC_IDS[episode_id % len(_SRC_IDS)], _TGT_IDS[episode_id % len(_TGT_IDS)]
        xy = _xy_configs()[(episode_id % (len(_SRC_IDS) * 2)) // len(_SRC_IDS)]
        quat = [self.cfg["quat"][m] for m in triplet]
        self.model_ids = triplet
        self._park(self._candidate_ids())
        self.objs = [self.handler.object_ids[m] for m in triplet]
        self.bbox = [
            np.array(self.model_db[m]["bbox"]["max"]) - np.array(self.model_db[m]["bbox"]["min"]) for m in triplet
        ]
        for o in self.objs:
            set_actor_collision(o, True)
            o.lock_motion(0, 0, 0, 1, 1, 0)
        rng = np.random.RandomState(seed)
        rx, ry = rng.uniform(0.30, 0.40), rng.uniform(0.0, 0.2)
        self.robot.set_root_pose(sapien.Pose([-10, 0, 0]))
        self.robot.set_qpos(INIT_QPOS)
        self.robot.set_qvel(np.zeros(self.robot.dof))
        z = SCENE_TABLE_HEIGHT + 0.5
        for o, xy_i, q_i in zip(self.objs, xy, quat):
            o.set_pose(sapien.Pose(np.hstack([xy_i, z]), q_i))
        self._settle(0.5)
        for o in self.objs:
            o.lock_motion(0, 0, 0, 0, 0, 0)
            o.set_pose(o.pose)
            o.set_velocity(np.zeros(3))
            o.set_angular_velocity(np.zeros(3))
        self._settle(0.5)
        if (
            sum(np.linalg.norm(o.velocity) for o in self.objs) > 1e-3
            or sum(np.linalg.norm(o.angular_velocity) for o in self.objs) > 1e-2
        ):
            self._settle(1.5)
        self.xyz_after_settle = [o.pose.p.copy() for o in self.objs]
        self.src_bbox_world = quat2mat(self.objs[self.src_id].pose.q) @ self.bbox[self.src_id]
        self.tgt_bbox_world = quat2mat(self.objs[self.tgt_id].pose.q) @ self.bbox[self.tgt_id]
        self.robot.set_root_pose(sapien.Pose([rx, ry, ROBOT_INIT_HEIGHT], [0, 0, 0, 1]))
        self.robot.set_qpos(INIT_QPOS)
        self.robot.set_qvel(np.zeros(self.robot.dof))
        if apply_external_states(self, states, env_ids):
            # success is measured against the post-reset object poses, so re-derive them
            self.xyz_after_settle = [o.pose.p.copy() for o in self.objs]
            self.src_bbox_world = quat2mat(self.objs[self.src_id].pose.q) @ self.bbox[self.src_id]
            self.tgt_bbox_world = quat2mat(self.objs[self.tgt_id].pose.q) @ self.bbox[self.tgt_id]
        self.controller.reset()
        self._elapsed = 0
        return self.get_obs(), {"episode_id": int(episode_id), "model_ids": triplet}

    def _com(self, o):
        return o.pose.transform(o.cmass_local_pose)

    def _evaluate(self):
        src, tgt = self.src_id, self.tgt_id
        sp, tp = self._com(self.objs[src]), self._com(self.objs[tgt])
        other = [i for i in range(len(self.objs)) if i not in (src, tgt)]
        keep = (
            all(self.objs[i].pose.p[2] - self.xyz_after_settle[i][2] > -0.02 for i in other)
            and (sp.p[2] - self.xyz_after_settle[src][2] > -0.15)
            and (tp.p[2] - self.xyz_after_settle[tgt][2] > -0.15)
        )
        smove = np.linalg.norm(self.xyz_after_settle[src][:2] - self.objs[src].pose.p[:2])
        omove = [
            np.linalg.norm(self.xyz_after_settle[i][:2] - self.objs[i].pose.p[:2])
            for i in range(len(self.objs))
            if i != src
        ]
        moved = (smove > 0.03) and all(x < smove for x in omove)
        dist = np.linalg.norm(sp.p[:2] - tp.p[:2])
        near = dist < (np.linalg.norm(self.tgt_bbox_world[:2]) / 2 + np.linalg.norm(self.src_bbox_world[:2]) / 2 + 0.10)
        dother = [np.linalg.norm(sp.p[:2] - self.objs[i].pose.p[:2]) for i in range(len(self.objs)) if i != src]
        closest = all(dist < x + 0.01 for x in dother)
        success = bool(keep and moved and near and closest)
        return {
            "success": success,
            "all_obj_keep_height": keep,
            "moved_correct_obj": bool(moved),
            "near_tgt_obj": bool(near),
            "is_closest_to_tgt": closest,
        }

    def get_language_instruction(self):
        return f"move {_instruction_obj_name(self.model_ids[self.src_id])} near {_instruction_obj_name(self.model_ids[self.tgt_id])}"


class MoveNearV0(SimplerMoveNearTask):
    variant = "v0"


class MoveNearV1(SimplerMoveNearTask):
    variant = "v1"
