"""Native PutOnBridge (widowx) — pure SAPIEN, zero upstream dep.

Reproduces the 4 widowx Bridge eval tasks on RoboVerse/MetaSim's own SAPIEN 2.x with no upstream
import: PutSpoonOnTableCloth, PutCarrotOnPlate, StackGreenCubeOnYellowCubeBakedTex,
PutEggplantInBasket. Builds the bridge scene + wx250s robot + 3rd-view camera (sink camera for
eggplant) + source/target objects (+ sink), vendors the put_on success checker (moved-correct +
src-on-target via bbox + grasp + contact) and the episode_id -> (xy,quat) selection.
control_freq=5, sim_freq=500 (substeps 100), control_mode align2 + gripper_pd_joint_pos.
"""

from __future__ import annotations

import json
import os

import numpy as np
import sapien.core as sapien
from transforms3d.euler import euler2quat
from transforms3d.quaternions import quat2mat

from .control import CombinedController
from .scene import (
    _LINK_MATERIAL,  # noqa: F401  (unused; google materials not needed)
    real2sim_scene_config,
)
from .widowx_config import (
    CAMERA_H,
    CAMERA_INTRINSIC,
    CAMERA_W,
    DEFAULT_CAMERA,
    INIT_QPOS,
    ROBOT_INIT_HEIGHT,
    SCENE_OFFSET,
    SCENE_POSE_Q,
    SINK_CAMERA,
    SINK_QPOS,
    WIDOWX_CONTROL_FREQ,
    WIDOWX_SIM_FREQ,
    WIDOWX_SUBSTEPS,
    WidowXGraspChecker,
    apply_robot_render_material,
    apply_widowx_urdf_materials,
    widowx_deployed_controller_configs,
)

SCENE_TABLE_HEIGHT = 0.87
OBJ_FRICTION = 0.5


def _grid(xy_center, hx, hy):
    g = (np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) * 2 - 1) * np.array([hx, hy])[None] + np.array(xy_center)[None]
    out = []
    for i, a in enumerate(g):
        for j, b in enumerate(g):
            if i != j:
                out.append(np.array([a, b]))
    return out


def _spoon_cfg():
    return dict(
        src="bridge_spoon_generated_modified",
        tgt="table_cloth_generated_shorter",
        model_json="info_bridge_custom_v0.json",
        xy=_grid([-0.16, 0.0], 0.075, 0.075),
        quat=[np.array([[1, 0, 0, 0], [1, 0, 0, 0]]), np.array([euler2quat(0, 0, np.pi / 2), [1, 0, 0, 0]])],
        success_complete=False,
        z_off=0.02,
        sink=False,
        camera=DEFAULT_CAMERA,
        scene="bridge_table_1_v1",
        instr="put the spoon on the towel",
    )


def _carrot_cfg():
    return dict(
        src="bridge_carrot_generated_modified",
        tgt="bridge_plate_objaverse_larger",
        model_json="info_bridge_custom_v0.json",
        xy=_grid([-0.16, 0.0], 0.075, 0.075),
        quat=[
            np.array([euler2quat(0, 0, np.pi), [1, 0, 0, 0]]),
            np.array([euler2quat(0, 0, -np.pi / 2), [1, 0, 0, 0]]),
        ],
        success_complete=True,
        z_off=0.02,
        sink=False,
        camera=DEFAULT_CAMERA,
        scene="bridge_table_1_v1",
        instr="put carrot on plate",
    )


def _stack_cfg():
    xy = []
    for hx, hy in zip([0.05, 0.1], [0.05, 0.1], strict=False):
        xy += _grid([-0.16, 0.0], hx, hy)
    return dict(
        src="baked_green_cube_3cm",
        tgt="baked_yellow_cube_3cm",
        model_json="info_bridge_custom_baked_tex_v0.json",
        xy=xy,
        quat=[np.array([[1, 0, 0, 0], [1, 0, 0, 0]])],
        success_complete=True,
        z_off=0.02,
        sink=False,
        camera=DEFAULT_CAMERA,
        scene="bridge_table_1_v1",
        instr="stack the green block on the yellow block",
    )


def _eggplant_cfg():
    target_xy = np.array([-0.125, 0.025])
    xy_center = [-0.105, 0.206]
    grid = [
        np.array([x + xy_center[0], y + xy_center[1]])
        for x in np.linspace(-0.01, 0.01, 2)
        for y in np.linspace(-0.015, 0.015, 4)
    ]
    xy = [np.stack([pos, target_xy], axis=0) for pos in grid]
    quat = [
        np.array([euler2quat(0, 0, 0), [1, 0, 0, 0]]),
        np.array([euler2quat(0, 0, np.pi / 4), [1, 0, 0, 0]]),
        np.array([euler2quat(0, 0, -np.pi / 4), [1, 0, 0, 0]]),
    ]
    return dict(
        src="eggplant",
        tgt="dummy_sink_target_plane",
        model_json="info_bridge_custom_v0.json",
        xy=xy,
        quat=quat,
        success_complete=False,
        z_off=0.06,
        sink=True,
        camera=SINK_CAMERA,
        scene="bridge_table_1_v2",
        instr="put eggplant into yellow basket",
    )


TASKS = {"spoon": _spoon_cfg, "carrot": _carrot_cfg, "stack": _stack_cfg, "eggplant": _eggplant_cfg}


def _visual_file(md):
    for n in ("textured.obj", "textured.dae", "textured.glb"):
        p = os.path.join(md, n)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(md)


class NativePutOnEnv:
    def __init__(self, *, task, urdf_path, scenes_dir, models_dir, model_db_dir):
        self.cfg = TASKS[task]()
        self.urdf_path = str(urdf_path)
        self.arena_glb = os.path.join(scenes_dir, self.cfg["scene"] + ".glb")
        self.models_dir = str(models_dir)
        with open(os.path.join(model_db_dir, self.cfg["model_json"])) as f:
            self.model_db = json.load(f)
        self.engine = self.scene = self.renderer = self.robot = self.camera = self.controller = None
        self.objs = []
        self.sink = None
        self._built = False
        self._elapsed = 0

    def build(self):
        self.engine = sapien.Engine()
        self.renderer = sapien.SapienRenderer()
        self.engine.set_renderer(self.renderer)
        self.scene = self.engine.create_scene(real2sim_scene_config())
        self.scene.set_timestep(1.0 / WIDOWX_SIM_FREQ)
        # Bridge tasks use the MoveNear-default lighting; eggplant overrides with one dim light.
        self.scene.set_ambient_light([0.3, 0.3, 0.3])
        if self.cfg["sink"]:  # PutEggplantInBasketScene._setup_lighting
            self.scene.add_directional_light(
                [0, 0, -1], [0.3, 0.3, 0.3], position=[0, 0, 1], shadow=False, scale=5, shadow_map_size=2048
            )
        else:
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
        self.robot = loader.load(self.urdf_path)
        apply_widowx_urdf_materials(self.scene, self.robot)
        apply_robot_render_material(self.robot)
        self.qpos = SINK_QPOS if self.cfg["sink"] else INIT_QPOS
        self.robot.set_qpos(self.qpos)
        self.base_link = next(x for x in self.robot.get_links() if x.get_name() == "base_link")
        cam = self.cfg["camera"]
        self.camera = self.scene.add_mounted_camera(
            "3rd_view_camera",
            self.base_link,
            sapien.Pose(p=cam["p"], q=cam["q"]),
            CAMERA_W,
            CAMERA_H,
            np.pi / 2,
            0.01,
            10,
        )
        self.camera.set_focal_lengths(CAMERA_INTRINSIC[0, 0], CAMERA_INTRINSIC[1, 1])
        self.camera.set_principal_point(CAMERA_INTRINSIC[0, 2], CAMERA_INTRINSIC[1, 2])
        self.controller = CombinedController(
            widowx_deployed_controller_configs(), self.robot, WIDOWX_CONTROL_FREQ, sim_freq=WIDOWX_SIM_FREQ
        )
        self.grasp_checker = WidowXGraspChecker(self.robot, self.scene)
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

    def episode_layout(self, seed, episode_id=None):
        n = len(self.cfg["xy"]) * len(self.cfg["quat"])
        if episode_id is None:
            episode_id = np.random.RandomState(seed).randint(n)
        xy = self.cfg["xy"][(episode_id % n) // len(self.cfg["quat"])]
        quat = self.cfg["quat"][episode_id % len(self.cfg["quat"])]
        return episode_id, xy, quat

    def reset(self, seed=0, options=None):
        if not self._built:
            self.build()
        options = options or {}
        episode_id, xy, quat = self.episode_layout(seed, (options.get("obj_init_options") or {}).get("episode_id"))
        self.episode_id = episode_id
        for o in self.objs:
            self.scene.remove_actor(o)
        self.objs = [self._build_object(self.cfg["src"]), self._build_object(self.cfg["tgt"])]
        self.bbox = []
        for m in (self.cfg["src"], self.cfg["tgt"]):
            bb = self.model_db[m]["bbox"]
            self.bbox.append(np.array(bb["max"]) - np.array(bb["min"]))

        # robot far away; sink placed (eggplant); objects fall + settle
        self.robot.set_root_pose(sapien.Pose([-10, 0, 0]))
        self.robot.set_qpos(self.qpos)
        self.robot.set_qvel(np.zeros(self.robot.dof))
        if self.cfg["sink"]:
            if self.sink is None:
                self.sink = self._build_object("sink")
            self.sink.set_pose(sapien.Pose([-0.16, 0.13, 0.88], [1, 0, 0, 0]))
            self.sink.lock_motion()
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
        if (
            sum(np.linalg.norm(o.velocity) for o in self.objs) > 1e-3
            or sum(np.linalg.norm(o.angular_velocity) for o in self.objs) > 1e-2
        ):
            self._settle(1.5)
        self.xyz_after_settle = [o.pose.p.copy() for o in self.objs]
        self.src_bbox_world = quat2mat(self.objs[0].pose.q) @ self.bbox[0]
        self.tgt_bbox_world = quat2mat(self.objs[1].pose.q) @ self.bbox[1]

        # widowx robot init (fixed, no draw): bridge tasks init_xy [0.147,0.028];
        # eggplant uses sink setup init [0.127,0.06]. Both height 0.870.
        rio = options.get("robot_init_options") or {}
        rxy = rio.get("init_xy", [0.127, 0.06] if self.cfg["sink"] else [0.147, 0.028])
        rh = rio.get("init_height", 0.85 if self.cfg["sink"] else ROBOT_INIT_HEIGHT)
        self.robot.set_root_pose(sapien.Pose([rxy[0], rxy[1], rh], [0, 0, 0, 1]))
        self.robot.set_qpos(self.qpos)
        self.robot.set_qvel(np.zeros(self.robot.dof))
        self.controller.reset()
        self.consecutive_grasp = 0
        self._elapsed = 0
        return self.get_obs(), {"episode_id": episode_id}

    def _settle(self, t):
        for _ in range(int(WIDOWX_SIM_FREQ * t)):
            self.scene.step()

    def step(self, action):
        self.controller.set_action(np.asarray(action, dtype=np.float32))
        for _ in range(WIDOWX_SUBSTEPS):
            self.controller.before_simulation_step()
            self.scene.step()
        self._elapsed += 1
        info = self.evaluate()
        return self.get_obs(), (1.0 if info["success"] else 0.0), bool(info["success"]), self._elapsed >= 120, info

    def render_color(self):
        self.scene.update_render()
        self.camera.take_picture()
        return self.camera.get_float_texture("Color")[..., :3]

    def get_obs(self):
        rgb = np.clip(self.render_color() * 255, 0, 255).astype(np.uint8)
        return {"image": {"3rd_view_camera": {"rgb": rgb}}, "agent": {"qpos": self.robot.get_qpos().copy()}}

    def _com(self, o):
        return o.pose.transform(o.cmass_local_pose)

    def evaluate(self):
        src, tgt = self.objs[0], self.objs[1]
        src_pose, tgt_pose = self._com(src), self._com(tgt)
        src_move = np.linalg.norm(self.xyz_after_settle[0][:2] - src.pose.p[:2])
        other_move = [np.linalg.norm(self.xyz_after_settle[1][:2] - tgt.pose.p[:2])]
        moved_correct = (src_move > 0.03) and all(x < src_move for x in other_move)
        is_grasped = self.grasp_checker.check_grasp(src)
        self.consecutive_grasp = self.consecutive_grasp + 1 if is_grasped else 0
        tgt_half, src_half = self.tgt_bbox_world / 2, self.src_bbox_world / 2
        offset = src_pose.p - tgt_pose.p
        xy_flag = np.linalg.norm(offset[:2]) <= np.linalg.norm(tgt_half[:2]) + 0.003
        z_flag = (offset[2] > 0) and (offset[2] - tgt_half[2] - src_half[2] <= self.cfg["z_off"])
        src_on_target = bool(xy_flag and z_flag)
        if self.cfg["success_complete"]:
            contacts = self.scene.get_contacts()
            ignore = [tgt.name] + [x.get_name() for x in self.robot.get_links()]
            flag = True
            for c in contacts:
                other = None
                if c.actor0.name == src.name:
                    other = c.actor1.name
                elif c.actor1.name == src.name:
                    other = c.actor0.name
                if other is not None:
                    imp = np.sum([p.impulse for p in c.points], axis=0)
                    if other not in ignore and np.linalg.norm(imp) > 1e-6:
                        flag = False
                        break
            src_on_target = src_on_target and flag
        return {
            "moved_correct_obj": bool(moved_correct),
            "is_src_obj_grasped": bool(is_grasped),
            "consecutive_grasp": bool(self.consecutive_grasp >= 5),
            "src_on_target": src_on_target,
            "success": src_on_target,
        }

    def get_language_instruction(self):
        return self.cfg["instr"]
