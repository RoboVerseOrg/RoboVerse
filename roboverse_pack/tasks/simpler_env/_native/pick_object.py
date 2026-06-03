"""Native GraspSingleRandomObject (google_robot pick_object) — pure SAPIEN, zero upstream dep.

Reproduces GraspSingleRandomObjectInScene-v0 (the eval task google_robot_pick_object): the coke
grasp scene/robot/camera/checker, but the target object is drawn each episode from a 14-object
set and its orientation is drawn too. Reuses the vendored grasp checker + pick-success evaluator
and the google_pick_coke scene; the only new logic is model selection + per-reset object build.

Draw order (verified vs upstream): orientation = RandomState(seed).choice([upright,
laid_vertically, lr_switch]); model_id = model_ids[RandomState(seed).randint(len)] (random_choice
util); then a re-seeded RandomState(seed) draws obj_init_xy, robot init_x, init_y.
"""

from __future__ import annotations

import json
import os

import numpy as np
import sapien.core as sapien
from transforms3d.euler import euler2quat

from .control import CombinedController
from .grasp import GoogleRobotGraspChecker, PickSuccessEvaluator
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
MODEL_IDS = [
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


def _visual_file(md):
    for n in ("textured.obj", "textured.dae", "textured.glb"):
        p = os.path.join(md, n)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(md)


class NativePickObjectEnv:
    def __init__(
        self,
        *,
        urdf_path,
        arena_glb,
        models_dir,
        model_db_dir,
        model_db_json="info_pick_custom_v0.json",
        overlay_png=None,
    ):
        self.urdf_path = str(urdf_path)
        self.arena_glb = str(arena_glb)
        self.models_dir = str(models_dir)
        with open(os.path.join(model_db_dir, model_db_json)) as f:
            self.model_db = json.load(f)
        self._orient = {
            "upright": euler2quat(np.pi / 2, 0, 0),
            "laid_vertically": euler2quat(0, 0, np.pi / 2),
            "lr_switch": euler2quat(0, 0, np.pi),
        }
        self.engine = self.scene = self.renderer = None
        self.robot = self.camera = self.controller = self.obj = None
        self.evaluator = PickSuccessEvaluator()
        self._built = False
        self._elapsed = 0

    def build(self):
        self.engine = sapien.Engine()
        self.renderer = sapien.SapienRenderer()
        self.engine.set_renderer(self.renderer)
        self.scene = self.engine.create_scene(real2sim_scene_config())
        self.scene.set_timestep(1.0 / SIM_FREQ)
        self.scene.set_ambient_light([0.3, 0.3, 0.3])
        self.scene.add_directional_light([0, 0, -1], [2.2, 2.2, 2.2], shadow=False, scale=5, shadow_map_size=2048)
        self.scene.add_directional_light([-1, -0.5, -1], [0.7, 0.7, 0.7])
        self.scene.add_directional_light([1, 1, -1], [0.7, 0.7, 0.7])
        b = self.scene.create_actor_builder()
        sp = sapien.Pose(q=SCENE_POSE_Q)
        b.add_nonconvex_collision_from_file(self.arena_glb, sp)
        b.add_visual_from_file(self.arena_glb, sp)
        self.arena = b.build_static(name="arena")
        self.arena.set_pose(sapien.Pose(-SCENE_OFFSET))
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
        self.grasp_checker = GoogleRobotGraspChecker(self.robot, self.scene)
        self._built = True
        return self

    def _build_object(self, model_id):
        density = self.model_db[model_id].get("density", 1000)
        md = os.path.join(self.models_dir, model_id)
        b = self.scene.create_actor_builder()
        b.add_multiple_collisions_from_file(
            filename=os.path.join(md, "collision.obj"), scale=[1.0] * 3, density=density
        )
        b.add_visual_from_file(filename=_visual_file(md), scale=[1.0] * 3)
        o = b.build(name=model_id)
        o.set_damping(0.1, 0.1)
        return o

    def reset(self, seed=0, options=None):
        if not self._built:
            self.build()
        options = options or {}
        orientation = np.random.RandomState(seed).choice(list(self._orient.keys()))
        self.model_id = MODEL_IDS[np.random.RandomState(seed).randint(len(MODEL_IDS))]
        if self.obj is not None:
            self.scene.remove_actor(self.obj)
        self.obj = self._build_object(self.model_id)

        rng = np.random.RandomState(seed)
        obj_xy = rng.uniform([-0.35, -0.02], [-0.12, 0.42], [2])
        init_x, init_y = rng.uniform(0.30, 0.40), rng.uniform(0.0, 0.2)

        self.robot.set_root_pose(sapien.Pose([-10, 0, 0]))
        self.robot.set_qpos(INIT_QPOS)
        self.robot.set_qvel(np.zeros(self.robot.dof))
        z = SCENE_TABLE_HEIGHT + 0.5
        self.obj.set_pose(sapien.Pose(np.hstack([obj_xy, z]), self._orient[orientation]))
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

        self.robot.set_root_pose(sapien.Pose([init_x, init_y, ROBOT_INIT_HEIGHT], [0, 0, 0, 1]))
        self.robot.set_qpos(INIT_QPOS)
        self.robot.set_qvel(np.zeros(self.robot.dof))
        self.controller.reset()
        self.evaluator.reset(self.obj_height_after_settle)
        self._elapsed = 0
        return self.get_obs(), {"model_id": self.model_id, "orientation": orientation}

    def _settle(self, t):
        for _ in range(int(SIM_FREQ * t)):
            self.scene.step()

    def step(self, action):
        self.controller.set_action(np.asarray(action, dtype=np.float32))
        for _ in range(SUBSTEPS):
            self.controller.before_simulation_step()
            self.scene.step()
        self._elapsed += 1
        info = self._evaluate()
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

    def _evaluate(self):
        is_grasped = self.grasp_checker.check_grasp(self.obj, max_angle=80)
        return self.evaluator.step(
            is_grasped=is_grasped,
            obj_z=float(self.obj.pose.p[2]),
            contacts=self.scene.get_contacts(),
            obj=self.obj,
            robot_link_names=[lk.get_name() for lk in self.robot.get_links()],
        )

    def get_language_instruction(self):
        return f"pick {self.model_id}"
