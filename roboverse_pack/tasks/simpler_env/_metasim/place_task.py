"""SimplerEnv place-into-closed-drawer tasks via the MetaSim API (5 tasks) — fully ScenarioCfg.

google_robot + mk_station cabinet + dummy_drawer floor + mounted camera + the FULL candidate object
set (info_pick_custom_baked_tex_v1) are all declared in the ScenarioCfg and loaded by the sapien2
handler (contact_offset=0.005). reset selects the per-episode target object (random, or fixed apple),
positions it (fall/settle) and parks the rest out of frame. 2-subtask checker: open drawer
(qpos>=0.15) then place (success = subtask1 & qpos>=0.05 & object-drawer contact); force-advance@100.
"""

from __future__ import annotations

import json
import os

import numpy as np
import sapien.core as sapien
from transforms3d.euler import euler2quat

from metasim.scenario.objects import ArticulationObjCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.scenario.simulator_params import SimParamCfg

from .._native._assets import paths
from .._native.control import CombinedController
from .._native.grasp import compute_total_impulse, get_pairwise_contacts
from .._native.overlay import load_overlay_image
from .._native.robot_config import google_robot_deployed_controller_configs
from .._native.scene import INIT_QPOS, ROBOT_INIT_HEIGHT
from .base import HIDDEN_POS, SimplerMetaSimTask, apply_external_states, candidate_mesh_objects, set_actor_collision
from .drawer_task import CABINET_POSE_P, _google_robot_cfg, _overhead_camera_cfg, dummy_drawer_floor_cfg

CAB_NAME = "cabinet"
SCENE_TABLE_HEIGHT, OBJ_FRICTION, FORCE_ADVANCE_STEPS = 0.87, 0.5, 100
MODEL_DB_JSON = "info_pick_custom_baked_tex_v1.json"

# SimplerEnv visual-matching stations for place (place_in_closed_drawer_in_scene.py): 3 overlays
_STATION_OVERLAY_IDS = ["a0", "b0", "c0"]
_STATION_ROBOT_XS = [0.644, 0.652, 0.665]
_STATION_ROBOT_YS = [-0.179, 0.009, 0.224]
_STATION_ROBOT_ROTZS = [-0.03, 0.0, 0.0]


def _place_sim_params():
    return SimParamCfg(
        dt=1.0 / 513,
        sapien_enable_tgs=True,
        sapien_enable_pcm=False,
        sapien_default_static_friction=1.0,
        sapien_default_dynamic_friction=1.0,
        sapien_default_restitution=0.0,
        sapien_apply_scene_solver=True,
        contact_offset=0.005,
        num_position_iterations=25,
        num_velocity_iterations=1,
        sapien_disable_default_lights=True,
        sapien_load_multiple_collisions=True,
    )


def _load_model_db():
    with open(os.path.join(paths()["model_db_dir"], MODEL_DB_JSON)) as f:
        return json.load(f)


class SimplerPlaceTask(SimplerMetaSimTask):
    robot_name, cam_name = "google_robot", "overhead_camera"
    sim_freq, control_freq = 513, 3
    max_episode_steps = 200
    drawer_ids = ("top", "middle", "bottom")
    fixed_model_id = None

    def _build_scenario(self):
        db = _load_model_db()
        self._candidate_ids = sorted(db.keys())
        cabinet = ArticulationObjCfg(
            name=CAB_NAME,
            urdf_path=paths()["cabinet_recolor_urdf"],
            fix_base_link=True,
            default_position=CABINET_POSE_P,
            default_orientation=(1.0, 0.0, 0.0, 0.0),
        )
        density_map = {k: db[k].get("density", 1000) for k in self._candidate_ids}
        objs = [dummy_drawer_floor_cfg(), cabinet] + candidate_mesh_objects(
            paths()["models_dir"], self._candidate_ids, friction=OBJ_FRICTION, damping=0.1, density_map=density_map
        )
        return ScenarioCfg(
            simulator="sapien2",
            robots=[_google_robot_cfg()],
            objects=objs,
            cameras=[_overhead_camera_cfg()],
            sim_params=_place_sim_params(),
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
        self.model_db = _load_model_db()
        self.sorted_model_ids = sorted(self.model_db.keys())
        self.cabinet = self.handler.object_ids[CAB_NAME]
        for j in self.cabinet.get_active_joints():
            j.set_friction(0.05)
            j.set_drive_property(stiffness=0, damping=1)
        self.joint_names = [j.name for j in self.cabinet.get_active_joints()]

    def _park_all_candidates(self):
        for mid in self.sorted_model_ids:
            o = self.handler.object_ids[mid]
            o.set_pose(sapien.Pose(HIDDEN_POS))
            o.set_velocity(np.zeros(3))
            o.set_angular_velocity(np.zeros(3))
            o.lock_motion(1, 1, 1, 1, 1, 1)
            set_actor_collision(o, False)

    def reset(self, states=None, env_ids=None, seed=0, options=None):
        """Reset the task. ``states``/``env_ids``/``seed`` are the ``BaseTaskEnv.reset`` contract;
        ``options`` is the SimplerEnv episode spec (obj/robot init options) from the gym adapter."""
        options = options or {}
        if self.fixed_model_id is not None:
            self.model_id = self.fixed_model_id
        else:
            self.model_id = self.sorted_model_ids[np.random.RandomState(seed).randint(len(self.sorted_model_ids))]
        self.drawer_id = str(np.random.RandomState(seed).choice(list(self.drawer_ids)))
        self.joint_idx = self.joint_names.index(f"{self.drawer_id}_drawer_joint")
        self._park_all_candidates()
        self.obj = self.handler.object_ids[self.model_id]
        set_actor_collision(self.obj, True)
        self.obj.lock_motion(0, 0, 0, 1, 1, 0)  # release for the fall (xy-rot locked)

        rng = np.random.RandomState(seed)
        obj_xy = (options.get("obj_init_options") or {}).get("init_xy")
        if obj_xy is None:
            obj_xy = rng.uniform([-0.10, -0.00], [-0.05, 0.1], [2])
        # visual-matching station: robot base xy + yaw + matching real-image overlay
        rio = options.get("robot_init_options") or {}
        idx = int(rio["station"]) if rio.get("station") is not None else int(rng.choice(len(_STATION_OVERLAY_IDS)))
        rxy = rio.get("init_xy") or [_STATION_ROBOT_XS[idx], _STATION_ROBOT_YS[idx]]
        rquat = (sapien.Pose(q=euler2quat(0, 0, _STATION_ROBOT_ROTZS[idx])) * sapien.Pose(q=[0, 0, 0, 1])).q
        self._overlay = load_overlay_image(
            os.path.join(paths()["overlay_dir"], f"open_drawer_{_STATION_OVERLAY_IDS[idx]}.png")
        )

        self.cabinet.set_qpos(np.zeros(self.cabinet.dof))
        self.cabinet.set_qvel(np.zeros(self.cabinet.dof))
        self.robot.set_root_pose(sapien.Pose([-10, 0, 0]))
        self.robot.set_qpos(INIT_QPOS)
        self.robot.set_qvel(np.zeros(self.robot.dof))
        z = SCENE_TABLE_HEIGHT + 0.5
        self.obj.set_pose(sapien.Pose(np.hstack([obj_xy, z]), [1, 0, 0, 0]))
        self._settle(0.5)
        self.obj.lock_motion(0, 0, 0, 0, 0, 0)
        self.obj.set_pose(self.obj.pose)
        self.obj.set_velocity(np.zeros(3))
        self.obj.set_angular_velocity(np.zeros(3))
        self._settle(0.5)
        if np.linalg.norm(self.obj.velocity) > 1e-3 or np.linalg.norm(self.obj.angular_velocity) > 1e-2:
            self._settle(1.5)
        self.obj_height_after_settle = float(self.obj.pose.p[2])

        self.robot.set_root_pose(sapien.Pose([rxy[0], rxy[1], ROBOT_INIT_HEIGHT], rquat))
        self.robot.set_qpos(INIT_QPOS)
        self.robot.set_qvel(np.zeros(self.robot.dof))
        if apply_external_states(self, states, env_ids):
            # the success checker keys off the settled height, so re-read it from the applied states
            self.obj_height_after_settle = float(self.obj.pose.p[2])
        self.controller.reset()
        self.drawer_link = next(x for x in self.cabinet.get_links() if x.get_name() == f"{self.drawer_id}_drawer")
        self.drawer_collision = self.drawer_link.get_collision_shapes()[2]
        self.cur_subtask_id = 0
        self.has_contact = 0
        self._elapsed = 0
        return self.get_obs(), {"model_id": self.model_id, "drawer_id": self.drawer_id}

    def _foreground_ids(self):
        return (
            [link.get_id() for link in self.robot.get_links()]
            + [lk.get_id() for lk in self.cabinet.get_links()]
            + [self.obj.get_id()]
        )

    def step(self, action):
        if self._elapsed >= FORCE_ADVANCE_STEPS:
            self.cur_subtask_id = 1
        return super().step(action)

    def _evaluate(self):
        qpos = float(self.cabinet.get_qpos()[self.joint_idx])
        contacts = get_pairwise_contacts(
            self.scene.get_contacts(), self.obj, self.drawer_link, collision_shape1=self.drawer_collision
        )
        self.has_contact += int(np.linalg.norm(compute_total_impulse(contacts)) > 1e-6)
        success = (self.cur_subtask_id == 1) and (qpos >= 0.05) and (self.has_contact >= 1)
        return {"success": bool(success), "qpos": qpos, "has_contact": int(self.has_contact)}

    def get_language_instruction(self):
        if self.cur_subtask_id == 0:
            return f"open {self.drawer_id} drawer"
        return f"place {self.model_id} into {self.drawer_id} drawer"


class PlaceInClosedDrawer(SimplerPlaceTask):
    drawer_ids, fixed_model_id = ("top", "middle", "bottom"), None


class PlaceInClosedTopDrawer(SimplerPlaceTask):
    drawer_ids, fixed_model_id = ("top",), None


class PlaceInClosedMiddleDrawer(SimplerPlaceTask):
    drawer_ids, fixed_model_id = ("middle",), None


class PlaceInClosedBottomDrawer(SimplerPlaceTask):
    drawer_ids, fixed_model_id = ("bottom",), None


class PlaceAppleInClosedTopDrawer(SimplerPlaceTask):
    drawer_ids, fixed_model_id = ("top",), "baked_apple_v2"
