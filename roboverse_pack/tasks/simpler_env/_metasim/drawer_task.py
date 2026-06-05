"""SimplerEnv open/close-drawer tasks via the MetaSim API (8 tasks).

google_robot loaded as a RobotCfg, the mk_station cabinet as an ArticulationObjCfg, the mounted
intrinsic overhead camera as a PinholeCameraCfg, SimplerEnv SceneConfig via SimParamCfg — all
loaded by the sapien2 handler. The procedural ``dummy_drawer`` white floor (visual-only static box)
and the SimplerEnv "simple" lighting are installed on the handler scene. open: closed drawer ->
qpos>=0.15; close: qpos 0.2 -> <=0.05.
"""

from __future__ import annotations

import os

import numpy as np
import sapien.core as sapien
from transforms3d.euler import euler2quat

from metasim.scenario.cameras import PinholeCameraCfg
from metasim.scenario.objects import ArticulationObjCfg, PrimitiveCubeCfg
from metasim.scenario.robot import BaseActuatorCfg, RobotCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.scenario.simulator_params import SimParamCfg

from .._native._assets import paths
from .._native.control import CombinedController
from .._native.overlay import load_overlay_image
from .._native.robot_config import (
    ARM_DAMPING,
    ARM_JOINT_NAMES,
    ARM_STIFFNESS,
    GRIPPER_JOINT_NAMES,
    google_robot_deployed_controller_configs,
)
from .._native.scene import INIT_QPOS, ROBOT_INIT_HEIGHT, SCENE_OFFSET
from .base import SimplerMetaSimTask

ROBOT_NAME, CAB_NAME, CAM_NAME = "google_robot", "cabinet", "overhead_camera"
CABINET_POSE_P = (-0.295, 0.0, 0.017)
CABINET_JOINT_FRICTION = 0.05
CLOSE_INIT_QPOS = 0.2

# SimplerEnv visual-matching "station" config (open_drawer_in_scene.py): 9 real-image overlays, each
# with a matching fixed robot base xy + small yaw. The robot base is head-mounted-camera-relevant, so
# the right station is what makes the obs match the RT-1 training distribution (and the handle reachable).
_STATION_OVERLAY_IDS = ["a0", "a1", "a2", "b0", "b1", "b2", "c0", "c1", "c2"]
_STATION_ROBOT_XS = [0.644, 0.765, 0.889, 0.652, 0.752, 0.851, 0.665, 0.765, 0.865]
_STATION_ROBOT_YS = [-0.179, -0.182, -0.203, 0.009, 0.009, 0.035, 0.224, 0.222, 0.222]
_STATION_ROBOT_ROTZS = [-0.03, -0.02, -0.06, 0.0, 0.0, 0.0, 0.0, -0.025, -0.025]


def _google_robot_cfg():
    actuators = {
        jn: BaseActuatorCfg(stiffness=float(ARM_STIFFNESS[i]), damping=float(ARM_DAMPING[i]))
        for i, jn in enumerate(ARM_JOINT_NAMES)
    }
    for jn in GRIPPER_JOINT_NAMES:
        actuators[jn] = BaseActuatorCfg(stiffness=200.0, damping=8.0)
    qpos = {jn: float(INIT_QPOS[i]) for i, jn in enumerate(ARM_JOINT_NAMES + GRIPPER_JOINT_NAMES)}
    return RobotCfg(
        name=ROBOT_NAME,
        urdf_path=paths()["urdf_google"],
        fix_base_link=True,
        default_joint_positions=qpos,
        actuators=actuators,
    )


def _overhead_camera_cfg():
    from .._native.scene import CAMERA_H, CAMERA_INTRINSIC, CAMERA_Q, CAMERA_W

    return PinholeCameraCfg(
        name=CAM_NAME,
        width=CAMERA_W,
        height=CAMERA_H,
        data_types=["rgb"],
        mount_to=ROBOT_NAME,
        mount_link="link_camera",
        mount_pos=(0.0, 0.0, 0.0),
        mount_quat=tuple(CAMERA_Q),
        intrinsic=[[float(x) for x in r] for r in CAMERA_INTRINSIC],
        clipping_range=(0.01, 10.0),
    )


def dummy_drawer_floor_cfg():
    """dummy_drawer procedural white floor as a visual-only static box (handler-loaded)."""
    return PrimitiveCubeCfg(
        name="arena",
        size=[20.0, 20.0, 0.034],
        color=[1.0, 1.0, 1.0],
        fix_base_link=True,
        collision_enabled=False,
        default_position=tuple((-SCENE_OFFSET).tolist()),
    )


def _sim_params():
    return SimParamCfg(
        dt=1.0 / 513,
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


class SimplerDrawerTask(SimplerMetaSimTask):
    robot_name, cam_name = ROBOT_NAME, CAM_NAME
    sim_freq, control_freq = 513, 3
    max_episode_steps = 113
    overlay_path = None  # raw render path; overlay assets selected at eval time
    is_close = False
    drawer_ids = ("top", "middle", "bottom")

    def _build_scenario(self):
        cabinet = ArticulationObjCfg(
            name=CAB_NAME,
            urdf_path=paths()["cabinet_recolor_urdf"],
            fix_base_link=True,
            default_position=CABINET_POSE_P,
            default_orientation=(1.0, 0.0, 0.0, 0.0),
        )
        return ScenarioCfg(
            simulator="sapien2",
            robots=[_google_robot_cfg()],
            objects=[dummy_drawer_floor_cfg(), cabinet],
            cameras=[_overhead_camera_cfg()],
            sim_params=_sim_params(),
            add_default_ground=False,
            num_envs=1,
            headless=True,
            gravity=(0, 0, -9.81),
        )

    def _install_lighting(self):
        self.scene.set_ambient_light([1.0, 1.0, 1.0])
        ang = 75
        self.scene.add_directional_light([-np.cos(np.deg2rad(ang)), 0, -np.sin(np.deg2rad(ang))], [1.0, 1.0, 1.0])

    def _make_controller(self):
        return CombinedController(
            google_robot_deployed_controller_configs(), self.robot, self.control_freq, sim_freq=self.sim_freq
        )

    def _wire(self):
        # dummy_drawer floor is now a cfg object (handler-loaded); just wire the cabinet.
        self.cabinet = self.handler.object_ids[CAB_NAME]
        for j in self.cabinet.get_active_joints():
            j.set_friction(CABINET_JOINT_FRICTION)
            j.set_drive_property(stiffness=0, damping=1)
        self.joint_names = [j.name for j in self.cabinet.get_active_joints()]

    def reset(self, env_ids=None, options=None, seed=0):
        options = options or {}
        rng = np.random.RandomState(seed)
        self.drawer_id = str(rng.choice(list(self.drawer_ids)))
        self.joint_idx = self.joint_names.index(f"{self.drawer_id}_drawer_joint")
        # pick a visual-matching station: fixes the robot base xy + yaw AND the matching overlay
        rio = options.get("robot_init_options") or {}
        idx = int(rio["station"]) if rio.get("station") is not None else int(rng.choice(len(_STATION_OVERLAY_IDS)))
        rxy = rio.get("init_xy") or [_STATION_ROBOT_XS[idx], _STATION_ROBOT_YS[idx]]
        quat = (sapien.Pose(q=euler2quat(0, 0, _STATION_ROBOT_ROTZS[idx])) * sapien.Pose(q=[0, 0, 0, 1])).q
        self._overlay = load_overlay_image(
            os.path.join(paths()["overlay_dir"], f"open_drawer_{_STATION_OVERLAY_IDS[idx]}.png")
        )
        self.robot.set_root_pose(sapien.Pose([rxy[0], rxy[1], ROBOT_INIT_HEIGHT], quat))
        self.robot.set_qpos(INIT_QPOS)
        self.robot.set_qvel(np.zeros(self.robot.dof))
        qpos = np.zeros(self.cabinet.dof)
        if self.is_close:
            qpos[self.joint_idx] = CLOSE_INIT_QPOS
        self.cabinet.set_qpos(qpos)
        self.cabinet.set_qvel(np.zeros(self.cabinet.dof))
        self.controller.reset()
        self._elapsed = 0
        return self.get_obs(), {"drawer_id": self.drawer_id}

    def _foreground_ids(self):
        # keep robot + cabinet in front of the real-image overlay (the white floor is background)
        return [lk.get_id() for lk in self.robot.get_links()] + [lk.get_id() for lk in self.cabinet.get_links()]

    def _evaluate(self):
        qpos = float(self.cabinet.get_qpos()[self.joint_idx])
        success = (qpos <= 0.05) if self.is_close else (qpos >= 0.15)
        return {"success": bool(success), "qpos": qpos}

    def get_language_instruction(self):
        return f"{'close' if self.is_close else 'open'} {self.drawer_id} drawer"


# concrete task variants (8)
class OpenDrawer(SimplerDrawerTask):
    is_close, drawer_ids = False, ("top", "middle", "bottom")


class OpenTopDrawer(SimplerDrawerTask):
    is_close, drawer_ids = False, ("top",)


class OpenMiddleDrawer(SimplerDrawerTask):
    is_close, drawer_ids = False, ("middle",)


class OpenBottomDrawer(SimplerDrawerTask):
    is_close, drawer_ids = False, ("bottom",)


class CloseDrawer(SimplerDrawerTask):
    is_close, drawer_ids = True, ("top", "middle", "bottom")


class CloseTopDrawer(SimplerDrawerTask):
    is_close, drawer_ids = True, ("top",)


class CloseMiddleDrawer(SimplerDrawerTask):
    is_close, drawer_ids = True, ("middle",)


class CloseBottomDrawer(SimplerDrawerTask):
    is_close, drawer_ids = True, ("bottom",)
