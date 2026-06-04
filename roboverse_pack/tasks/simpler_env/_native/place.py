"""Native PlaceIntoClosedDrawer (google_robot) — pure SAPIEN, zero upstream dep.

Reproduces PlaceIntoClosedDrawerCustomInScene-v0 (+ top/middle/bottom + place_apple) — the 5 eval
tasks google_robot_place_in_closed[_top/_middle/_bottom]_drawer + place_apple_in_closed_top_drawer
— on RoboVerse/MetaSim's own SAPIEN 2.x with no upstream import. Builds on the drawer cabinet
(mk_station, dummy_drawer scene, simple lighting) and adds:
  * a target object that falls onto the cabinet top and settles (contact_offset=0.005 override),
  * the 2-subtask success rule (open drawer -> place object): success = subtask==1 and target
    drawer joint qpos>=0.05 and object-vs-drawer contact impulse seen at least once,
  * force-advance to subtask 1 at 100 steps.
model_id = random_choice(sorted model_ids) [rng.randint] (or fixed, e.g. place_apple);
drawer_id = rng.choice(drawer_ids); obj/robot init xy from a re-seeded RandomState(seed).
"""

from __future__ import annotations

import json
import os

import numpy as np
import sapien.core as sapien

from .control import CombinedController
from .grasp import compute_total_impulse, get_pairwise_contacts
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
    SIM_FREQ,
    real2sim_scene_config,
)

SUBSTEPS = SIM_FREQ // CONTROL_FREQ
MAX_EPISODE_STEPS = 200
CABINET_POSE = sapien.Pose([-0.295, 0, 0.017], [1, 0, 0, 0])
CABINET_JOINT_FRICTION = 0.05
SCENE_TABLE_HEIGHT = 0.87
OBJ_FRICTION = 0.5
FORCE_ADVANCE_STEPS = 100


def _visual_file(model_dir):
    for name in ("textured.obj", "textured.dae", "textured.glb"):
        p = os.path.join(model_dir, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(model_dir)


def _place_scene_config():
    sc = real2sim_scene_config()
    sc.contact_offset = 0.005  # PlaceObjectInClosedDrawerInSceneEnv._get_default_scene_config
    return sc


class NativePlaceInDrawerEnv:
    def __init__(
        self,
        *,
        drawer_ids,
        urdf_path,
        cabinet_urdf,
        models_dir,
        model_db_dir,
        model_db_json="info_pick_custom_baked_tex_v1.json",
        fixed_model_id=None,
    ):
        self.drawer_ids = list(drawer_ids)
        self.urdf_path = str(urdf_path)
        self.cabinet_urdf = str(cabinet_urdf)
        self.models_dir = str(models_dir)
        self.fixed_model_id = fixed_model_id
        with open(os.path.join(model_db_dir, model_db_json)) as f:
            self.model_db = json.load(f)
        self.sorted_model_ids = sorted(self.model_db.keys())
        self.engine = self.scene = self.renderer = None
        self.robot = self.cabinet = self.camera = self.controller = self.obj = None
        self._built = False
        self._elapsed = 0

    def build(self):
        self.engine = sapien.Engine()
        self.renderer = sapien.SapienRenderer()
        self.engine.set_renderer(self.renderer)
        self.scene = self.engine.create_scene(_place_scene_config())
        self.scene.set_timestep(1.0 / SIM_FREQ)
        # simple lighting (drawer eval)
        self.scene.set_ambient_light([1.0, 1.0, 1.0])
        angle = 75
        self.scene.add_directional_light([-np.cos(np.deg2rad(angle)), 0, -np.sin(np.deg2rad(angle))], [1.0, 1.0, 1.0])
        # dummy_drawer arena (visual-only white box)
        b = self.scene.create_actor_builder()
        b.add_box_visual(half_size=np.array([10.0, 10.0, 0.017]), color=[1, 1, 1])
        self.arena = b.build_static(name="arena")
        self.arena.set_pose(sapien.Pose(-SCENE_OFFSET))
        # cabinet
        cl = self.scene.create_urdf_loader()
        cl.fix_root_link = True
        self.cabinet = cl.load(self.cabinet_urdf)
        self.cabinet.set_name("cabinet")
        self.cabinet.set_pose(CABINET_POSE)
        for j in self.cabinet.get_active_joints():
            j.set_friction(CABINET_JOINT_FRICTION)
            j.set_drive_property(stiffness=0, damping=1)
        self.joint_names = [j.name for j in self.cabinet.get_active_joints()]
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
        density = self.model_db[model_id].get("density", 1000)
        md = os.path.join(self.models_dir, model_id)
        b = self.scene.create_actor_builder()
        b.add_multiple_collisions_from_file(
            filename=os.path.join(md, "collision.obj"), scale=[1.0] * 3, material=mat, density=density
        )
        b.add_visual_from_file(filename=_visual_file(md), scale=[1.0] * 3)
        o = b.build(name=model_id)
        o.set_damping(0.1, 0.1)
        return o

    def reset(self, seed=0, options=None):
        if not self._built:
            self.build()
        options = options or {}
        # model_id: random_choice util = sorted[rng.randint(len)] (or fixed)
        if self.fixed_model_id is not None:
            self.model_id = self.fixed_model_id
        else:
            self.model_id = self.sorted_model_ids[np.random.RandomState(seed).randint(len(self.sorted_model_ids))]
        # drawer_id: rng.choice (np), separate fresh rng
        self.drawer_id = str(np.random.RandomState(seed).choice(self.drawer_ids))
        self.joint_idx = self.joint_names.index(f"{self.drawer_id}_drawer_joint")

        if self.obj is not None:
            self.scene.remove_actor(self.obj)
        self.obj = self._build_object(self.model_id)

        # initialize_episode draws on a fresh RandomState(seed): obj_xy first, then robot xy
        rng = np.random.RandomState(seed)
        obj_xy = (options.get("obj_init_options") or {}).get("init_xy")
        if obj_xy is None:
            obj_xy = rng.uniform([-0.10, -0.00], [-0.05, 0.1], [2])
        robot_xy = (options.get("robot_init_options") or {}).get("init_xy")
        if robot_xy is None:
            robot_xy = [rng.uniform(0.30, 0.40), rng.uniform(0.0, 0.2)]

        # cabinet starts closed
        self.cabinet.set_qpos(np.zeros(self.cabinet.dof))
        self.cabinet.set_qvel(np.zeros(self.cabinet.dof))
        # object falls + settles (robot far away)
        self.robot.set_root_pose(sapien.Pose([-10, 0, 0]))
        self.robot.set_qpos(INIT_QPOS)
        self.robot.set_qvel(np.zeros(self.robot.dof))
        z = SCENE_TABLE_HEIGHT + 0.5
        self.obj.set_pose(sapien.Pose(np.hstack([obj_xy, z]), [1, 0, 0, 0]))
        self.obj.lock_motion(0, 0, 0, 1, 1, 0)
        self._settle(0.5)
        self.obj.lock_motion(0, 0, 0, 0, 0, 0)
        self.obj.set_pose(self.obj.pose)
        self.obj.set_velocity(np.zeros(3))
        self.obj.set_angular_velocity(np.zeros(3))
        self._settle(0.5)
        if np.linalg.norm(self.obj.velocity) > 1e-3 or np.linalg.norm(self.obj.angular_velocity) > 1e-2:
            self._settle(1.5)
        self.obj_height_after_settle = float(self.obj.pose.p[2])

        # robot placed at init
        self.robot.set_root_pose(sapien.Pose([robot_xy[0], robot_xy[1], ROBOT_INIT_HEIGHT], [0, 0, 0, 1]))
        self.robot.set_qpos(INIT_QPOS)
        self.robot.set_qvel(np.zeros(self.robot.dof))
        self.controller.reset()

        self.drawer_link = next(x for x in self.cabinet.get_links() if x.get_name() == f"{self.drawer_id}_drawer")
        self.drawer_collision = self.drawer_link.get_collision_shapes()[2]
        self.cur_subtask_id = 0
        self.episode_stats = {"qpos": 0.0, "is_drawer_open": False, "has_contact": 0}
        self._elapsed = 0
        return self.get_obs(), {"model_id": self.model_id, "drawer_id": self.drawer_id}

    def _settle(self, t):
        for _ in range(int(SIM_FREQ * t)):
            self.scene.step()

    def advance_to_next_subtask(self):
        self.cur_subtask_id = 1

    def step(self, action):
        if self._elapsed >= FORCE_ADVANCE_STEPS:
            self.advance_to_next_subtask()
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

    def render_color(self):
        self.scene.update_render()
        self.camera.take_picture()
        return self.camera.get_float_texture("Color")[..., :3]

    def get_obs(self):
        rgb = np.clip(self.render_color() * 255, 0, 255).astype(np.uint8)
        return {"image": {"overhead_camera": {"rgb": rgb}}, "agent": {"qpos": self.robot.get_qpos().copy()}}

    def evaluate(self):
        qpos = float(self.cabinet.get_qpos()[self.joint_idx])
        self.episode_stats["qpos"] = qpos
        is_open = qpos >= 0.15
        self.episode_stats["is_drawer_open"] = self.episode_stats["is_drawer_open"] or is_open
        contacts = get_pairwise_contacts(
            self.scene.get_contacts(), self.obj, self.drawer_link, collision_shape1=self.drawer_collision
        )
        has_contact = np.linalg.norm(compute_total_impulse(contacts)) > 1e-6
        self.episode_stats["has_contact"] += int(has_contact)
        success = (self.cur_subtask_id == 1) and (qpos >= 0.05) and (self.episode_stats["has_contact"] >= 1)
        return {
            "success": bool(success),
            "qpos": qpos,
            "has_contact": int(self.episode_stats["has_contact"]),
            "is_drawer_open": bool(self.episode_stats["is_drawer_open"]),
        }

    def get_language_instruction(self):
        if self.cur_subtask_id == 0:
            return f"open {self.drawer_id} drawer"
        return f"place {self.model_id} into {self.drawer_id} drawer"
