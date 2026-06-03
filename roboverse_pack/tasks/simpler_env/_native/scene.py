"""Native (deletable) SimplerEnv coke-can scene — pure SAPIEN 2.x, no upstream env class.

Builds the visual-matching ``google_pick_coke_can_1_v4`` scene from scratch (arena GLB +
google_robot + opened_coke_can + mounted overhead camera + visual-matching lighting), using
ONLY vendored components and migrated ``roboverse_data`` assets — no ``mani_skill2_real2sim``
import. Reproduces the parts of ManiSkill2-real2sim ``CustomSceneEnv`` /
``GraspSingleInSceneEnv`` that determine the policy-facing observation:

  * SceneConfig          = real2sim ``_get_default_scene_config`` (M0/M2 verified)
  * lighting             = ``GraspSingleInSceneEnv._setup_lighting`` default branch
  * arena                = nonconvex GLB at scene_pose, offset ``-scene_offset``
  * robot                = google_robot URDF, urdf_config friction materials applied
  * overhead camera      = mounted on ``link_camera``, intrinsic 425/425/320/256, 640x512
  * object               = opened_coke_can (density 50, scale 1.0)
  * greenscreen overlay  = vendored ``apply_greenscreen_overlay`` (M4 verified)

This is the renderer/obs half of the native port; control (M3/M6), checker (M5) and engine
(M0/M2) are verified separately. ``parity_render`` compares this scene's overhead obs image
against upstream given identical state.
"""

from __future__ import annotations

import numpy as np
import sapien.core as sapien

from .overlay import apply_greenscreen_overlay, foreground_actor_ids, load_overlay_image

# ----------------------------------------------------------------------------- #
# Exact upstream config (mani_skill2_real2sim, google_robot visual-matching)
# ----------------------------------------------------------------------------- #
SIM_FREQ = 513
CONTROL_FREQ = 3
SCENE_OFFSET = np.array([-1.6616, -3.0337, 0.0])
SCENE_POSE_Q = [0.707, 0.707, 0, 0]
ROBOT_INIT_HEIGHT = 0.06205 + 0.017
CAMERA_Q = [0.5, 0.5, -0.5, 0.5]  # opencv->ros
CAMERA_INTRINSIC = np.array([[425.0, 0, 320.0], [0, 425.0, 256.0], [0, 0, 1]])
CAMERA_W, CAMERA_H = 640, 512
COKE_DENSITY = 50.0
COKE_SCALE = 1.0

# google_robot finger friction materials (GoogleRobotDefaultConfig.urdf_config)
_FINGER_FRICTION = 2.0
_URDF_MATERIALS = {
    "finger_mat": dict(static_friction=_FINGER_FRICTION, dynamic_friction=_FINGER_FRICTION, restitution=0.0),
    "finger_tip_mat": dict(static_friction=_FINGER_FRICTION, dynamic_friction=_FINGER_FRICTION, restitution=0.0),
    "finger_nail_mat": dict(static_friction=0.1, dynamic_friction=0.1, restitution=0.0),
    "base_mat": dict(static_friction=0.1, dynamic_friction=0.0, restitution=0.0),
    "wheel_mat": dict(static_friction=1.0, dynamic_friction=0.0, restitution=0.0),
}
_LINK_MATERIAL = {
    "link_base": "base_mat",
    "link_wheel_left": "wheel_mat",
    "link_wheel_right": "wheel_mat",
    "link_finger_left": "finger_mat",
    "link_finger_right": "finger_mat",
    "link_finger_tip_left": "finger_tip_mat",
    "link_finger_tip_right": "finger_tip_mat",
    "link_finger_nail_left": "finger_nail_mat",
    "link_finger_nail_right": "finger_nail_mat",
}

INIT_QPOS = np.array([
    -0.2639457174606611,
    0.0831913360274175,
    0.5017611504652179,
    1.156859026208673,
    0.028583671314766423,
    1.592598203487462,
    -1.080652960128774,
    0,
    0,
    -0.00285961,
    0.7851361,
])


def real2sim_scene_config(gravity=(0.0, 0.0, -9.81)):
    sc = sapien.SceneConfig()
    sc.default_dynamic_friction = 1.0
    sc.default_static_friction = 1.0
    sc.default_restitution = 0.0
    sc.contact_offset = 0.02
    sc.enable_pcm = False
    sc.solver_iterations = 25
    sc.enable_tgs = True
    sc.solver_velocity_iterations = 1
    sc.gravity = np.array(gravity)
    return sc


class NativeCokeCanScene:
    """Self-contained SAPIEN coke-can scene reproducing the upstream policy observation."""

    def __init__(self, *, urdf_path, arena_glb, coke_collision, coke_visual, overlay_png):
        self.urdf_path = str(urdf_path)
        self.arena_glb = str(arena_glb)
        self.coke_collision = str(coke_collision)
        self.coke_visual = str(coke_visual)
        self.overlay_png = str(overlay_png)
        self.engine = self.scene = self.renderer = None
        self.robot = self.obj = self.arena = None
        self.camera = self.link_camera = None

    # ---- build ---------------------------------------------------------------
    def build(self):
        self.engine = sapien.Engine()
        self.renderer = sapien.SapienRenderer()
        self.engine.set_renderer(self.renderer)
        self.scene = self.engine.create_scene(real2sim_scene_config())
        self.scene.set_timestep(1.0 / SIM_FREQ)
        self._setup_lighting()
        self._load_arena()
        self._load_robot()
        self._load_object()
        self._mount_camera()
        return self

    def _setup_lighting(self):
        # GraspSingleInSceneEnv._setup_lighting, default (no special) branch
        self.scene.set_ambient_light([0.3, 0.3, 0.3])
        self.scene.add_directional_light([0, 0, -1], [2.2, 2.2, 2.2], shadow=False, scale=5, shadow_map_size=2048)
        self.scene.add_directional_light([-1, -0.5, -1], [0.7, 0.7, 0.7])
        self.scene.add_directional_light([1, 1, -1], [0.7, 0.7, 0.7])

    def _load_arena(self):
        builder = self.scene.create_actor_builder()
        scene_pose = sapien.Pose(q=SCENE_POSE_Q)
        builder.add_nonconvex_collision_from_file(self.arena_glb, scene_pose)
        builder.add_visual_from_file(self.arena_glb, scene_pose)
        self.arena = builder.build_static(name="arena")
        self.arena.set_pose(sapien.Pose(-SCENE_OFFSET))

    def _load_robot(self):
        loader = self.scene.create_urdf_loader()
        loader.fix_root_link = True
        loader.load_multiple_collisions_from_file = True
        # build urdf_config (friction materials), matching parse_urdf_config
        materials = {name: self.scene.create_physical_material(**cfg) for name, cfg in _URDF_MATERIALS.items()}
        link_cfg = {link: {"material": materials[m]} for link, m in _LINK_MATERIAL.items()}
        urdf_config = {"link": link_cfg}
        self.robot = loader.load(self.urdf_path, urdf_config)
        self.robot.set_qpos(INIT_QPOS)
        self.robot.set_qvel(np.zeros(self.robot.dof))
        self.link_camera = next(x for x in self.robot.get_links() if x.get_name() == "link_camera")

    def _load_object(self):
        builder = self.scene.create_actor_builder()
        builder.add_multiple_collisions_from_file(
            filename=self.coke_collision, scale=[COKE_SCALE] * 3, density=COKE_DENSITY
        )
        builder.add_visual_from_file(filename=self.coke_visual, scale=[COKE_SCALE] * 3)
        self.obj = builder.build(name="opened_coke_can")
        self.obj.set_damping(0.1, 0.1)

    def _mount_camera(self):
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

    # ---- state + render ------------------------------------------------------
    def set_state(self, *, robot_qpos, robot_pose, obj_pose):
        # set_root_pose + set_qpos update link transforms kinematically (no physics step,
        # which would sag the robot and drift the mounted-camera pose).
        self.robot.set_root_pose(sapien.Pose(robot_pose[:3], robot_pose[3:]))
        self.robot.set_qpos(np.asarray(robot_qpos))
        self.obj.set_pose(sapien.Pose(obj_pose[:3], obj_pose[3:]))

    def render_color(self):
        self.scene.update_render()
        self.camera.take_picture()
        color = self.camera.get_float_texture("Color")[..., :3]  # (H, W, 3) float
        return color

    def render_obs(self):
        """Policy-facing overhead image (uint8) with greenscreen overlay applied."""
        color = self.render_color()
        rgb = np.clip(color * 255, 0, 255).astype(np.uint8)
        seg = self.camera.get_uint32_texture("Segmentation")[..., 1]  # actor-level seg
        robot_ids = [link.get_id() for link in self.robot.get_links()]
        overlay = load_overlay_image(self.overlay_png)
        fg = foreground_actor_ids(robot_ids, [self.obj.get_id()], [])
        out = rgb.astype(np.float64) / 255.0
        out = apply_greenscreen_overlay(out, seg, overlay, foreground_actor_ids=fg)
        return np.clip(out * 255, 0, 255).astype(np.uint8)
