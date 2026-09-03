"""SimplerEnv coke-can task, fully through the MetaSim API.

This is the MetaSim-native rewrite of the SimplerEnv pick-coke-can family: the scene, google_robot,
coke object and the mounted overhead camera are all declared in a ``ScenarioCfg`` and loaded by the
MetaSim ``sapien2`` handler (with the opt-in SceneConfig / no-ground / mounted-intrinsic-camera /
mesh-object / load-multiple-collisions / skip-default-lights extensions). The genuinely
SimplerEnv-specific behaviour — the EE-delta + ruckig controller, the visual-matching greenscreen
overlay, the grasp/lift success checker, and the fall-and-settle reset — is layered on top as a
task, operating on the handler's loaded objects (``handler.object_ids`` / ``handler.camera_ids`` /
``handler.scene``). No ``mani_skill2_real2sim`` / ``simpler_env`` import.
"""

from __future__ import annotations

import numpy as np
import sapien.core as sapien
from transforms3d.euler import euler2quat

from metasim.scenario.cameras import PinholeCameraCfg
from metasim.scenario.objects import NonConvexRigidObjCfg, RigidObjCfg
from metasim.scenario.robot import BaseActuatorCfg, RobotCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.scenario.simulator_params import SimParamCfg
from metasim.task.base import BaseTaskEnv

# reuse the verified vendored SimplerEnv-specific pieces (raw-SAPIEN, no upstream dep)
from .._native._assets import paths
from .._native.control import CombinedController
from .._native.grasp import GoogleRobotGraspChecker, PickSuccessEvaluator
from .._native.overlay import apply_greenscreen_overlay, foreground_actor_ids, load_overlay_image
from .._native.robot_config import (
    ARM_DAMPING,
    ARM_JOINT_NAMES,
    ARM_STIFFNESS,
    GRIPPER_JOINT_NAMES,
    google_robot_deployed_controller_configs,
)
from .._native.scene import (
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
)
from .base import apply_external_states

SUBSTEPS = SIM_FREQ // CONTROL_FREQ
SCENE_TABLE_HEIGHT = 0.87
ROBOT_NAME = "google_robot"
COKE_NAME = "opened_coke_can"
ARENA_NAME = "arena"
CAM_NAME = "overhead_camera"
_ORIENT = {
    "upright": euler2quat(np.pi / 2, 0, 0),
    "laid_vertically": euler2quat(0, 0, np.pi / 2),
    "lr_switch": euler2quat(0, 0, np.pi),
}
# google_robot finger friction (urdf_config), applied to the handler-loaded articulation
_FINGER_LINKS = ["link_finger_left", "link_finger_right", "link_finger_tip_left", "link_finger_tip_right"]


def build_coke_scenario() -> ScenarioCfg:
    p = paths()
    arm_n, grip_n = ARM_JOINT_NAMES, GRIPPER_JOINT_NAMES
    # actuators: provide the deployed PD so the handler sets drive props (the EE controller re-sets
    # them on reset anyway); one entry per active joint as the handler requires.
    actuators = {}
    for i, jn in enumerate(arm_n):
        actuators[jn] = BaseActuatorCfg(stiffness=float(ARM_STIFFNESS[i]), damping=float(ARM_DAMPING[i]))
    for jn in grip_n:
        actuators[jn] = BaseActuatorCfg(stiffness=200.0, damping=8.0)
    default_qpos = {jn: float(INIT_QPOS[i]) for i, jn in enumerate(arm_n + grip_n)}

    robot = RobotCfg(
        name=ROBOT_NAME,
        urdf_path=p["urdf_google"],
        fix_base_link=True,
        default_joint_positions=default_qpos,
        actuators=actuators,
    )
    arena = NonConvexRigidObjCfg(
        name=ARENA_NAME,
        usd_path=p["coke_scene"],
        mesh_pose=[0.0, 0.0, 0.0, *SCENE_POSE_Q],
        default_position=tuple((-SCENE_OFFSET).tolist()),
        default_orientation=(1.0, 0.0, 0.0, 0.0),
        fix_base_link=True,
    )
    coke = RigidObjCfg(
        name=COKE_NAME,
        mesh_path=p["coke_visual"],
        collision_mesh_path=p["coke_collision"],
        mesh_density=50.0,
        linear_damping=0.1,
        angular_damping=0.1,
        fix_base_link=False,
        default_position=(0.0, 0.0, SCENE_TABLE_HEIGHT),
    )
    cam = PinholeCameraCfg(
        name=CAM_NAME,
        width=CAMERA_W,
        height=CAMERA_H,
        data_types=["rgb"],
        mount_to=ROBOT_NAME,
        mount_link="link_camera",
        mount_pos=(0.0, 0.0, 0.0),
        mount_quat=tuple(CAMERA_Q),
        intrinsic=[[float(x) for x in row] for row in CAMERA_INTRINSIC],
        clipping_range=(0.01, 10.0),
    )
    sim = SimParamCfg(
        dt=1.0 / SIM_FREQ,
        sapien_enable_tgs=True,
        sapien_enable_pcm=False,
        sapien_default_static_friction=1.0,
        sapien_default_dynamic_friction=1.0,
        sapien_default_restitution=0.0,
        sapien_apply_scene_solver=True,
        contact_offset=0.02,
        num_position_iterations=25,
        num_velocity_iterations=1,
        sapien_disable_default_lights=True,
        sapien_load_multiple_collisions=True,
    )
    return ScenarioCfg(
        simulator="sapien2",
        robots=[robot],
        objects=[arena, coke],
        cameras=[cam],
        sim_params=sim,
        add_default_ground=False,
        num_envs=1,
        headless=True,
        gravity=(0.0, 0.0, -9.81),
    )


class SimplerCokeTask(BaseTaskEnv):
    """pick-coke-can on MetaSim sapien2; default_orientation None => drawn from the episode seed."""

    scenario = None  # set per-instance (build_coke_scenario) since it resolves asset paths
    max_episode_steps = 80
    default_orientation = None  # subclasses override (lr_switch / laid_vertically / upright)
    _instruction = "pick coke can"

    def __init__(self, scenario=None, device=None):
        super().__init__(scenario if scenario is not None else build_coke_scenario(), device=device)
        self._setup_simpler()

    # ---- one-time wiring on the handler-loaded objects -----------------------
    def _setup_simpler(self):
        h = self.handler
        self.scene = h.scene
        self.robot = h.object_ids[ROBOT_NAME]
        self.obj = h.object_ids[COKE_NAME]
        self.camera = h.camera_ids[CAM_NAME]
        # SimplerEnv visual-matching lighting (handler default lights skipped via sim_params)
        self.scene.set_ambient_light([0.3, 0.3, 0.3])
        self.scene.add_directional_light([0, 0, -1], [2.2, 2.2, 2.2], shadow=False, scale=5, shadow_map_size=2048)
        self.scene.add_directional_light([-1, -0.5, -1], [0.7, 0.7, 0.7])
        self.scene.add_directional_light([1, 1, -1], [0.7, 0.7, 0.7])
        # finger friction material (google urdf_config: finger_friction=2.0)
        mat = self.scene.create_physical_material(2.0, 2.0, 0.0)
        links = {lk.get_name(): lk for lk in self.robot.get_links()}
        for ln in _FINGER_LINKS:
            if ln in links:
                for cs in links[ln].get_collision_shapes():
                    cs.set_physical_material(mat)
        self.controller = CombinedController(
            google_robot_deployed_controller_configs(), self.robot, CONTROL_FREQ, sim_freq=SIM_FREQ
        )
        self.grasp_checker = GoogleRobotGraspChecker(self.robot, self.scene)
        self.evaluator = PickSuccessEvaluator()
        self._overlay = load_overlay_image(paths()["coke_overlay"])
        self._elapsed = 0
        self.obj_height_after_settle = None

    # ---- reset (RNG repro + fall/settle, on the handler objects) -------------
    def reset(self, states=None, env_ids=None, seed=0, options=None):
        """Reset the task. ``states``/``env_ids``/``seed`` are the ``BaseTaskEnv.reset`` contract;
        ``options`` is the SimplerEnv episode spec (obj/robot init options) from the gym adapter."""
        options = options or {}
        oio = options.get("obj_init_options") or {}
        if oio.get("init_rot_quat") is None:
            orientation = oio.get("orientation", self.default_orientation)
            if orientation is None:
                orientation = np.random.RandomState(seed).choice(list(_ORIENT.keys()))
            quat = _ORIENT[orientation]
        else:
            quat = np.array(oio["init_rot_quat"])
            orientation = oio.get("orientation")
        self.orientation = orientation

        rng = np.random.RandomState(seed)
        obj_xy = oio.get("init_xy") or rng.uniform([-0.35, -0.02], [-0.12, 0.42], [2])
        rio = options.get("robot_init_options") or {}
        rxy = rio.get("init_xy") or [rng.uniform(0.30, 0.40), rng.uniform(0.0, 0.2)]

        self.robot.set_root_pose(sapien.Pose([-10, 0, 0]))
        self.robot.set_qpos(INIT_QPOS)
        self.robot.set_qvel(np.zeros(self.robot.dof))
        z = SCENE_TABLE_HEIGHT + 0.5
        self.obj.set_pose(sapien.Pose(np.hstack([obj_xy, z]), quat))
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

        self.robot.set_root_pose(sapien.Pose([rxy[0], rxy[1], ROBOT_INIT_HEIGHT], [0, 0, 0, 1]))
        self.robot.set_qpos(INIT_QPOS)
        self.robot.set_qvel(np.zeros(self.robot.dof))
        if apply_external_states(self, states, env_ids):
            # the success checker keys off the settled height, so re-read it from the applied states
            self.obj_height_after_settle = float(self.obj.pose.p[2])
        self.controller.reset()
        self.evaluator.reset(self.obj_height_after_settle)
        self._elapsed = 0
        return self.get_obs(), {"orientation": orientation}

    def _settle(self, t):
        for _ in range(int(SIM_FREQ * t)):
            self.scene.step()

    # ---- step (EE controller on the handler articulation) --------------------
    def step(self, action):
        self.controller.set_action(np.asarray(action, dtype=np.float32))
        for _ in range(SUBSTEPS):
            self.controller.before_simulation_step()
            self.scene.step()
        self._elapsed += 1
        info = self._evaluate()
        return (
            self.get_obs(),
            1.0 if info["success"] else 0.0,
            bool(info["success"]),
            self._elapsed >= self.max_episode_steps,
            info,
        )

    # ---- obs (handler camera + vendored overlay) ----------------------------
    def render_color(self):
        self.scene.update_render()
        self.camera.take_picture()
        return self.camera.get_float_texture("Color")[..., :3]

    def get_obs(self, overlay=True):
        color = self.render_color()
        if not overlay:
            return {"image": {CAM_NAME: {"rgb": np.clip(color * 255, 0, 255).astype(np.uint8)}}}
        seg = self.camera.get_uint32_texture("Segmentation")[..., 1]
        fg = foreground_actor_ids([lk.get_id() for lk in self.robot.get_links()], [self.obj.get_id()], [])
        ov = apply_greenscreen_overlay(color.astype(np.float64), seg, self._overlay, foreground_actor_ids=fg)
        return {
            "image": {CAM_NAME: {"rgb": np.clip(ov * 255, 0, 255).astype(np.uint8)}},
            "agent": {"qpos": self.robot.get_qpos().copy()},
        }

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
        return self._instruction


# ---- concrete variants (orientation as class attr) + MetaSim task registry ----
from metasim.task.registry import register_task


@register_task("simpler.google_robot_pick_coke_can")
class PickCokeCan(SimplerCokeTask):
    default_orientation = None


@register_task("simpler.google_robot_pick_horizontal_coke_can")
class PickHorizontalCokeCan(SimplerCokeTask):
    default_orientation = "lr_switch"


@register_task("simpler.google_robot_pick_vertical_coke_can")
class PickVerticalCokeCan(SimplerCokeTask):
    default_orientation = "laid_vertically"


@register_task("simpler.google_robot_pick_standing_coke_can")
class PickStandingCokeCan(SimplerCokeTask):
    default_orientation = "upright"
