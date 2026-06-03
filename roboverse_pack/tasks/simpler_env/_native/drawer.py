"""Native Drawer (google_robot open/close) — pure SAPIEN, zero upstream dep.

Reproduces Open/CloseDrawerCustomInScene-v0 (+ top/middle/bottom) — the 8 eval tasks
google_robot_{open,close}[_top/_middle/_bottom]_drawer — on RoboVerse/MetaSim's own SAPIEN 2.x
with no mani_skill2_real2sim/simpler_env import. The cabinet is the self-contained box-primitive
``mk_station.urdf``; the scene is the procedural ``dummy_drawer`` white box (visual only).

Open: cabinet starts closed (qpos 0), success = target drawer joint qpos >= 0.15.
Close: cabinet starts at qpos 0.2, success = qpos <= 0.05.
drawer_id is drawn from drawer_ids via RandomState(seed).choice (matched), robot init xy from a
re-seeded RandomState(seed).uniform — same draw structure as the other families.
"""

from __future__ import annotations

import numpy as np
import sapien.core as sapien

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
    SIM_FREQ,
    real2sim_scene_config,
)

SUBSTEPS = SIM_FREQ // CONTROL_FREQ
MAX_EPISODE_STEPS = 113
CABINET_POSE = sapien.Pose([-0.295, 0, 0.017], [1, 0, 0, 0])
CABINET_JOINT_FRICTION = 0.05
CLOSE_INIT_QPOS = 0.2


class NativeDrawerEnv:
    def __init__(self, *, is_close, drawer_ids, urdf_path, cabinet_urdf):
        self.is_close = is_close
        self.drawer_ids = list(drawer_ids)
        self.urdf_path = str(urdf_path)
        self.cabinet_urdf = str(cabinet_urdf)
        self.engine = self.scene = self.renderer = None
        self.robot = self.cabinet = self.camera = self.controller = None
        self._built = False
        self._elapsed = 0
        self.drawer_id = None
        self.joint_idx = None

    def build(self):
        self.engine = sapien.Engine()
        self.renderer = sapien.SapienRenderer()
        self.engine.set_renderer(self.renderer)
        self.scene = self.engine.create_scene(real2sim_scene_config())
        self.scene.set_timestep(1.0 / SIM_FREQ)
        # drawer eval lighting: light_mode="simple" (OpenDrawerInSceneEnv._setup_lighting)
        self.scene.set_ambient_light([1.0, 1.0, 1.0])
        angle = 75
        self.scene.add_directional_light([-np.cos(np.deg2rad(angle)), 0, -np.sin(np.deg2rad(angle))], [1.0, 1.0, 1.0])
        # dummy_drawer arena: visual-only white box (no collision), offset -scene_offset
        b = self.scene.create_actor_builder()
        b.add_box_visual(half_size=np.array([10.0, 10.0, 0.017]), color=[1, 1, 1])
        self.arena = b.build_static(name="arena")
        self.arena.set_pose(sapien.Pose(-SCENE_OFFSET))
        # cabinet articulation (box-primitive urdf)
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

    def reset(self, seed=0, options=None):
        if not self._built:
            self.build()
        options = options or {}
        self.drawer_id = str(np.random.RandomState(seed).choice(self.drawer_ids))
        self.joint_idx = self.joint_names.index(f"{self.drawer_id}_drawer_joint")

        rng = np.random.RandomState(seed)  # re-seeded; robot init draws first (no falling objects)
        robot_xy = (options.get("robot_init_options") or {}).get("init_xy")
        if robot_xy is None:
            robot_xy = [rng.uniform(0.30, 0.40), rng.uniform(0.0, 0.2)]

        self.robot.set_root_pose(sapien.Pose([robot_xy[0], robot_xy[1], ROBOT_INIT_HEIGHT], [0, 0, 0, 1]))
        self.robot.set_qpos(INIT_QPOS)
        self.robot.set_qvel(np.zeros(self.robot.dof))
        # cabinet qpos: open task -> all 0 (closed); close task -> target joint at 0.2
        qpos = np.zeros(self.cabinet.dof)
        if self.is_close:
            qpos[self.joint_idx] = CLOSE_INIT_QPOS
        self.cabinet.set_qpos(qpos)
        self.cabinet.set_qvel(np.zeros(self.cabinet.dof))
        self.controller.reset()
        self._elapsed = 0
        return self.get_obs(), {"drawer_id": self.drawer_id}

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

    def render_color(self):
        self.scene.update_render()
        self.camera.take_picture()
        return self.camera.get_float_texture("Color")[..., :3]

    def get_obs(self):
        rgb = np.clip(self.render_color() * 255, 0, 255).astype(np.uint8)
        return {"image": {"overhead_camera": {"rgb": rgb}}, "agent": {"qpos": self.robot.get_qpos().copy()}}

    def evaluate(self):
        qpos = float(self.cabinet.get_qpos()[self.joint_idx])
        success = (qpos <= 0.05) if self.is_close else (qpos >= 0.15)
        return {"success": bool(success), "qpos": qpos}

    def get_language_instruction(self):
        return f"{'close' if self.is_close else 'open'} {self.drawer_id} drawer"
