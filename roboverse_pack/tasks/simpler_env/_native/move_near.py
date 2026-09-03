"""Native MoveNear (google_robot) — pure SAPIEN, zero upstream dep.

Reproduces MoveNearGoogleBakedTexInScene-v0/-v1 (the 3 eval tasks google_robot_move_near[_v0/v1])
on RoboVerse/MetaSim's own SAPIEN 2.x with no mani_skill2_real2sim/simpler_env import. Vendors:
  * the triplet / xy-config / per-object orientation / special-density tables (verbatim),
  * the episode_id -> (triplet, source/target obj, xy, quat) selection (RNG draw order matched),
  * the multi-object success checker (height-keep + moved-correct + near-target + closest-to-target).
Reuses the scene/robot/camera/lighting builders' constants from ``scene.py``.
"""

from __future__ import annotations

import json
import os

import numpy as np
import sapien.core as sapien
from transforms3d.euler import euler2quat
from transforms3d.quaternions import quat2mat

from .control import CombinedController
from .robot_config import google_robot_deployed_controller_configs
from .scene import (
    _LINK_MATERIAL,
    _URDF_MATERIALS,
    CAMERA_H,
    CAMERA_INTRINSIC,
    CAMERA_Q,
    CAMERA_W,
    CONTROL_FREQ,
    INIT_QPOS,
    ROBOT_INIT_HEIGHT,
    SCENE_OFFSET,
    SCENE_POSE_Q,
    SIM_FREQ,
    real2sim_scene_config,
)

SUBSTEPS = SIM_FREQ // CONTROL_FREQ
SCENE_TABLE_HEIGHT = 0.87
MAX_EPISODE_STEPS = 80
OBJ_FRICTION = 0.5  # CustomSceneEnv.obj_static_friction / obj_dynamic_friction

_RM_WORDS = {"opened", "light", "generated", "modified", "objaverse", "bridge", "baked", "v2"}


def _instruction_obj_name(s):
    out = []
    for w in s.split("_"):
        if w[-2:] == "cm" or w in _RM_WORDS:
            continue
        out.append(w)
    return " ".join(out)


def _xy_configs():
    return [
        ([-0.33, 0.04], [-0.33, 0.34], [-0.13, 0.19]),
        ([-0.13, 0.04], [-0.33, 0.19], [-0.13, 0.34]),
    ]


# v0 / v1 object configs (verbatim from MoveNearGoogleBakedTexInScene{,V1})
MOVE_NEAR_CONFIG = {
    "v0": {
        "model_json": "info_pick_custom_baked_tex_v0.json",
        "triplets": [
            ("blue_plastic_bottle", "baked_opened_pepsi_can", "orange"),
            ("baked_opened_7up_can", "baked_apple", "baked_sponge"),
            ("baked_opened_coke_can", "baked_opened_redbull_can", "baked_apple"),
            ("baked_sponge", "blue_plastic_bottle", "baked_opened_7up_can"),
            ("orange", "baked_opened_pepsi_can", "baked_opened_redbull_can"),
        ],
        "quat": {
            "blue_plastic_bottle": euler2quat(np.pi / 2, 0, np.pi / 2),
            "baked_opened_pepsi_can": euler2quat(np.pi / 2, 0, 0),
            "orange": euler2quat(0, 0, np.pi / 2),
            "baked_opened_7up_can": euler2quat(np.pi / 2, 0, 0),
            "baked_apple": [1.0, 0.0, 0.0, 0.0],
            "baked_sponge": euler2quat(0, 0, np.pi / 2),
            "baked_opened_coke_can": euler2quat(np.pi / 2, 0, 0),
            "baked_opened_redbull_can": euler2quat(np.pi / 2, 0, 0),
        },
        "special_density": {"baked_apple": 200, "orange": 200},
    },
    "v1": {
        "model_json": "info_pick_custom_baked_tex_v1.json",
        "triplets": [
            ("blue_plastic_bottle", "baked_opened_pepsi_can_v2", "orange"),
            ("baked_opened_7up_can_v2", "baked_apple_v2", "baked_sponge_v2"),
            ("baked_opened_coke_can_v2", "baked_opened_redbull_can_v2", "baked_apple_v2"),
            ("baked_sponge_v2", "blue_plastic_bottle", "baked_opened_7up_can_v2"),
            ("orange", "baked_opened_pepsi_can_v2", "baked_opened_redbull_can_v2"),
        ],
        "quat": {
            "blue_plastic_bottle": euler2quat(np.pi / 2, 0, np.pi / 2),
            "baked_opened_pepsi_can_v2": euler2quat(np.pi / 2, 0, 0),
            "orange": euler2quat(0, 0, np.pi / 2),
            "baked_opened_7up_can_v2": euler2quat(np.pi / 2, 0, 0),
            "baked_apple_v2": [1.0, 0.0, 0.0, 0.0],
            "baked_sponge_v2": euler2quat(0, 0, np.pi / 2),
            "baked_opened_coke_can_v2": euler2quat(np.pi / 2, 0, 0),
            "baked_opened_redbull_can_v2": euler2quat(np.pi / 2, 0, 0),
        },
        "special_density": {"baked_apple_v2": 200, "orange": 200},
    },
}

# source/target obj id pairs: (i,j) for i!=j in 3x3
_SRC_IDS = [i for i in range(3) for j in range(3) if i != j]
_TGT_IDS = [j for i in range(3) for j in range(3) if i != j]
N_EPISODES = 5 * len(_SRC_IDS) * 2  # triplets x src/tgt x xy_configs = 60


def _visual_file(model_dir):
    for name in ("textured.obj", "textured.dae", "textured.glb"):
        p = os.path.join(model_dir, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"no textured.* in {model_dir}")


class NativeMoveNearEnv:
    def __init__(self, *, variant, urdf_path, arena_glb, models_dir, model_db_dir, overlay_png):
        self.variant = variant
        self.cfg = MOVE_NEAR_CONFIG[variant]
        self.urdf_path = str(urdf_path)
        self.arena_glb = str(arena_glb)
        self.models_dir = str(models_dir)
        self.overlay_png = str(overlay_png)
        with open(os.path.join(model_db_dir, self.cfg["model_json"])) as f:
            self.model_db = json.load(f)
        self.engine = self.scene = self.renderer = self.robot = self.camera = None
        self.controller = None
        self.objs = []
        self._built = False
        self._elapsed = 0

    # ---- build ---------------------------------------------------------------
    def build(self):
        self.engine = sapien.Engine()
        self.renderer = sapien.SapienRenderer()
        self.engine.set_renderer(self.renderer)
        self.scene = self.engine.create_scene(real2sim_scene_config())
        self.scene.set_timestep(1.0 / SIM_FREQ)
        # visual-matching default lighting (same as grasp_single)
        self.scene.set_ambient_light([0.3, 0.3, 0.3])
        self.scene.add_directional_light([0, 0, -1], [2.2, 2.2, 2.2], shadow=False, scale=5, shadow_map_size=2048)
        self.scene.add_directional_light([-1, -0.5, -1], [0.7, 0.7, 0.7])
        self.scene.add_directional_light([1, 1, -1], [0.7, 0.7, 0.7])
        # arena
        b = self.scene.create_actor_builder()
        sp = sapien.Pose(q=SCENE_POSE_Q)
        b.add_nonconvex_collision_from_file(self.arena_glb, sp)
        b.add_visual_from_file(self.arena_glb, sp)
        self.arena = b.build_static(name="arena")
        self.arena.set_pose(sapien.Pose(-SCENE_OFFSET))
        # robot
        loader = self.scene.create_urdf_loader()
        loader.fix_root_link = True
        loader.load_multiple_collisions_from_file = True
        mats = {n: self.scene.create_physical_material(**c) for n, c in _URDF_MATERIALS.items()}
        urdf_cfg = {"link": {lk: {"material": mats[m]} for lk, m in _LINK_MATERIAL.items()}}
        self.robot = loader.load(self.urdf_path, urdf_cfg)
        self.robot.set_qpos(INIT_QPOS)
        self.link_camera = next(x for x in self.robot.get_links() if x.get_name() == "link_camera")
        self.camera = self.scene.add_mounted_camera(
            "overhead_camera",
            self.link_camera,
            sapien.Pose(p=[0, 0, 0], q=CAMERA_Q),
            CAMERA_W,
            CAMERA_H,
            np.pi / 2,
            0.01,
            10,
        )
        self.camera.set_focal_lengths(CAMERA_INTRINSIC[0, 0], CAMERA_INTRINSIC[1, 1])
        self.camera.set_principal_point(CAMERA_INTRINSIC[0, 2], CAMERA_INTRINSIC[1, 2])
        self.controller = CombinedController(
            google_robot_deployed_controller_configs(), self.robot, CONTROL_FREQ, sim_freq=SIM_FREQ
        )
        self._built = True
        return self

    def _build_object(self, model_id):
        mat = self.scene.create_physical_material(OBJ_FRICTION, OBJ_FRICTION, 0.0)
        density = self.cfg["special_density"].get(model_id, self.model_db[model_id].get("density", 1000))
        model_dir = os.path.join(self.models_dir, model_id)
        b = self.scene.create_actor_builder()
        b.add_multiple_collisions_from_file(
            filename=os.path.join(model_dir, "collision.obj"), scale=[1.0] * 3, material=mat, density=density
        )
        b.add_visual_from_file(filename=_visual_file(model_dir), scale=[1.0] * 3)
        obj = b.build(name=model_id)
        obj.set_damping(0.1, 0.1)  # MoveNearInSceneEnv._load_actors
        return obj

    # ---- reset ---------------------------------------------------------------
    def episode_layout(self, seed, episode_id=None):
        """Reproduce MoveNearGoogle reset selection (episode_id drawn on RandomState(seed))."""
        if episode_id is None:
            episode_id = np.random.RandomState(seed).randint(N_EPISODES)
        triplet = self.cfg["triplets"][episode_id // (len(_SRC_IDS) * 2)]
        src = _SRC_IDS[episode_id % len(_SRC_IDS)]
        tgt = _TGT_IDS[episode_id % len(_TGT_IDS)]
        xy = _xy_configs()[(episode_id % (len(_SRC_IDS) * 2)) // len(_SRC_IDS)]
        quat = [self.cfg["quat"][m] for m in triplet]
        return episode_id, list(triplet), src, tgt, xy, quat

    def reset(self, seed=0, options=None):
        if not self._built:
            self.build()
        options = options or {}
        episode_id, triplet, src, tgt, xy, quat = self.episode_layout(
            seed, (options.get("obj_init_options") or {}).get("episode_id")
        )
        self.episode_id, self.src_id, self.tgt_id = episode_id, src, tgt
        self.model_ids = triplet

        # (re)build objects for this triplet
        for o in self.objs:
            self.scene.remove_actor(o)
        self.objs = [self._build_object(m) for m in triplet]
        self.bbox_sizes = []
        for m in triplet:
            bb = self.model_db[m]["bbox"]
            self.bbox_sizes.append(np.array(bb["max"]) - np.array(bb["min"]))

        # robot init draws on a fresh RandomState(seed) (re-seeded before initialize_episode;
        # object xys/quats are provided so the first draws are robot init_x/init_y)
        rng = np.random.RandomState(seed)
        robot_xy = (options.get("robot_init_options") or {}).get("init_xy")
        if robot_xy is None:
            robot_xy = [rng.uniform(0.30, 0.40), rng.uniform(0.0, 0.2)]

        # _initialize_actors: objects fall + settle (robot far away)
        self.robot.set_root_pose(sapien.Pose([-10, 0, 0]))
        self.robot.set_qpos(INIT_QPOS)
        self.robot.set_qvel(np.zeros(self.robot.dof))
        z = SCENE_TABLE_HEIGHT + 0.5
        for o, xy_i, q_i in zip(self.objs, xy, quat, strict=False):
            o.set_pose(sapien.Pose(np.hstack([xy_i, z]), q_i))
            o.lock_motion(0, 0, 0, 1, 1, 0)
        self._settle(0.5)
        for o in self.objs:
            o.lock_motion(0, 0, 0, 0, 0, 0)
            o.set_pose(o.pose)
            o.set_velocity(np.zeros(3))
            o.set_angular_velocity(np.zeros(3))
        self._settle(0.5)
        lin = sum(np.linalg.norm(o.velocity) for o in self.objs)
        ang = sum(np.linalg.norm(o.angular_velocity) for o in self.objs)
        if lin > 1e-3 or ang > 1e-2:
            self._settle(1.5)
        self.xyz_after_settle = [o.pose.p.copy() for o in self.objs]
        # bbox in world frame at settle (rotated)
        self.src_bbox_world = quat2mat(self.objs[src].pose.q) @ self.bbox_sizes[src]
        self.tgt_bbox_world = quat2mat(self.objs[tgt].pose.q) @ self.bbox_sizes[tgt]

        # _initialize_agent
        self.robot.set_root_pose(sapien.Pose([robot_xy[0], robot_xy[1], ROBOT_INIT_HEIGHT], [0, 0, 0, 1]))
        self.robot.set_qpos(INIT_QPOS)
        self.robot.set_qvel(np.zeros(self.robot.dof))
        self.controller.reset()
        self._elapsed = 0
        return self.get_obs(), {"episode_id": episode_id, "model_ids": triplet}

    def _settle(self, t):
        for _ in range(int(SIM_FREQ * t)):
            self.scene.step()

    # ---- step ----------------------------------------------------------------
    def step(self, action):
        self.controller.set_action(np.asarray(action, dtype=np.float32))
        for _ in range(SUBSTEPS):
            self.controller.before_simulation_step()
            self.scene.step()
        self._elapsed += 1
        info = self.evaluate()
        return (
            self.get_obs(),
            (1.0 if info["success"] else 0.0),
            bool(info["success"]),
            self._elapsed >= MAX_EPISODE_STEPS,
            info,
        )

    # ---- obs + eval ----------------------------------------------------------
    def render_color(self):
        self.scene.update_render()
        self.camera.take_picture()
        return self.camera.get_float_texture("Color")[..., :3]

    def get_obs(self):
        rgb = np.clip(self.render_color() * 255, 0, 255).astype(np.uint8)
        return {"image": {"overhead_camera": {"rgb": rgb}}, "agent": {"qpos": self.robot.get_qpos().copy()}}

    def _com_pose(self, obj):
        return obj.pose.transform(obj.cmass_local_pose)

    def evaluate(self):
        src, tgt = self.src_id, self.tgt_id
        src_pose, tgt_pose = self._com_pose(self.objs[src]), self._com_pose(self.objs[tgt])
        # height keep
        other = [i for i in range(len(self.objs)) if i not in (src, tgt)]
        other_keep = all(self.objs[i].pose.p[2] - self.xyz_after_settle[i][2] > -0.02 for i in other)
        all_keep = (
            other_keep
            and (src_pose.p[2] - self.xyz_after_settle[src][2] > -0.15)
            and (tgt_pose.p[2] - self.xyz_after_settle[tgt][2] > -0.15)
        )
        # moved correct
        src_move = np.linalg.norm(self.xyz_after_settle[src][:2] - self.objs[src].pose.p[:2])
        other_move = [
            np.linalg.norm(self.xyz_after_settle[i][:2] - self.objs[i].pose.p[:2])
            for i in range(len(self.objs))
            if i != src
        ]
        moved_correct = (src_move > 0.03) and all(x < src_move for x in other_move)
        # near target
        dist = np.linalg.norm(src_pose.p[:2] - tgt_pose.p[:2])
        near = dist < (np.linalg.norm(self.tgt_bbox_world[:2]) / 2 + np.linalg.norm(self.src_bbox_world[:2]) / 2 + 0.10)
        # closest to target
        dist_others = [
            np.linalg.norm(src_pose.p[:2] - self.objs[i].pose.p[:2]) for i in range(len(self.objs)) if i != src
        ]
        is_closest = all(dist < x + 0.01 for x in dist_others)
        success = bool(all_keep and moved_correct and near and is_closest)
        return {
            "all_obj_keep_height": all_keep,
            "moved_correct_obj": moved_correct,
            "near_tgt_obj": bool(near),
            "is_closest_to_tgt": is_closest,
            "success": success,
        }

    def get_language_instruction(self):
        return f"move {_instruction_obj_name(self.model_ids[self.src_id])} near {_instruction_obj_name(self.model_ids[self.tgt_id])}"
