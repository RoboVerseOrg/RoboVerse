"""SimplerEnv WidowX Bridge put-on tasks via the MetaSim API (4 tasks).

wx250s RobotCfg + bridge GLB arena (NonConvexRigidObjCfg) + source/target mesh objects (+ sink for
eggplant) + mounted intrinsic 3rd-view camera — all in a ScenarioCfg loaded by the sapien2 handler
(control 5/500, align2 + gripper_pd_joint_pos). Per-task lighting (bright bridge / dim eggplant),
finger friction + robot render material, and the put-on success checker layer on the handler
objects. Objects are repositioned per episode_id at reset.
"""

from __future__ import annotations

import json
import os

import numpy as np
import sapien.core as sapien
from transforms3d.quaternions import quat2mat

from metasim.scenario.cameras import PinholeCameraCfg
from metasim.scenario.objects import NonConvexRigidObjCfg, RigidObjCfg
from metasim.scenario.robot import BaseActuatorCfg, RobotCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.scenario.simulator_params import SimParamCfg

from .._native._assets import paths
from .._native.control import CombinedController
from .._native.overlay import load_overlay_image
from .._native.put_on import TASKS as _PUTON_CFG
from .._native.widowx_config import (
    ARM_DAMPING,
    ARM_JOINT_NAMES,
    ARM_STIFFNESS,
    CAMERA_H,
    CAMERA_INTRINSIC,
    CAMERA_W,
    DEFAULT_CAMERA,
    GRIPPER_DAMPING,
    GRIPPER_JOINT_NAMES,
    GRIPPER_STIFFNESS,
    INIT_QPOS,
    ROBOT_INIT_HEIGHT,
    SCENE_OFFSET,
    SCENE_POSE_Q,
    SINK_CAMERA,
    SINK_QPOS,
    WidowXGraspChecker,
    apply_robot_render_material,
    apply_widowx_urdf_materials,
    widowx_deployed_controller_configs,
)
from .base import SimplerMetaSimTask, apply_external_states

ROBOT_NAME, ARENA_NAME, CAM_NAME = "widowx", "arena", "3rd_view_camera"
OBJ_FRICTION = 0.5


def _widowx_robot_cfg(qpos_arr):
    actuators = {
        jn: BaseActuatorCfg(stiffness=float(ARM_STIFFNESS[i]), damping=float(ARM_DAMPING[i]))
        for i, jn in enumerate(ARM_JOINT_NAMES)
    }
    for jn in GRIPPER_JOINT_NAMES:
        actuators[jn] = BaseActuatorCfg(stiffness=float(GRIPPER_STIFFNESS), damping=float(GRIPPER_DAMPING))
    qpos = {jn: float(qpos_arr[i]) for i, jn in enumerate(ARM_JOINT_NAMES + GRIPPER_JOINT_NAMES)}
    return RobotCfg(
        name=ROBOT_NAME,
        urdf_path=paths()["urdf_widowx"],
        fix_base_link=True,
        default_joint_positions=qpos,
        actuators=actuators,
    )


def _obj_cfg(name, model_id, fix=False):
    md = os.path.join(paths()["models_dir"], model_id)
    vis = None
    for n in ("textured.obj", "textured.dae", "textured.glb"):
        if os.path.exists(os.path.join(md, n)):
            vis = os.path.join(md, n)
            break
    return RigidObjCfg(
        name=name,
        mesh_path=vis,
        collision_mesh_path=os.path.join(md, "collision.obj"),
        static_friction=OBJ_FRICTION,
        dynamic_friction=OBJ_FRICTION,
        restitution=0.0,
        linear_damping=0.1,
        angular_damping=0.1,
        fix_base_link=fix,
        default_position=(0.0, 0.0, 1.0),
    )


class SimplerPutOnTask(SimplerMetaSimTask):
    robot_name, cam_name = ROBOT_NAME, CAM_NAME
    sim_freq, control_freq = 500, 5
    max_episode_steps = 120
    task_key = "spoon"

    def __init__(self, scenario=None, device=None):
        self.cfg = _PUTON_CFG[self.task_key]()
        super().__init__(scenario=scenario, device=device)

    def _build_scenario(self):
        cfg = self.cfg
        qpos = SINK_QPOS if cfg["sink"] else INIT_QPOS
        arena = NonConvexRigidObjCfg(
            name=ARENA_NAME,
            usd_path=os.path.join(paths()["scenes_dir"], cfg["scene"] + ".glb"),
            mesh_pose=[0.0, 0.0, 0.0, *SCENE_POSE_Q],
            default_position=tuple((-SCENE_OFFSET).tolist()),
            default_orientation=(1.0, 0.0, 0.0, 0.0),
            fix_base_link=True,
        )
        objs = [arena, _obj_cfg("source_obj", cfg["src"]), _obj_cfg("target_obj", cfg["tgt"])]
        if cfg["sink"]:
            objs.append(_obj_cfg("sink", "sink", fix=False))  # dynamic + lock_motion in reset (== _native)
        cam = PinholeCameraCfg(
            name=CAM_NAME,
            width=CAMERA_W,
            height=CAMERA_H,
            data_types=["rgb"],
            mount_to=ROBOT_NAME,
            mount_link="base_link",
            mount_pos=tuple((SINK_CAMERA if cfg["sink"] else DEFAULT_CAMERA)["p"]),
            mount_quat=tuple((SINK_CAMERA if cfg["sink"] else DEFAULT_CAMERA)["q"]),
            intrinsic=[[float(x) for x in r] for r in CAMERA_INTRINSIC],
            clipping_range=(0.01, 10.0),
        )
        sim = SimParamCfg(
            dt=1.0 / 500,
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
            robots=[_widowx_robot_cfg(qpos)],
            objects=objs,
            cameras=[cam],
            sim_params=sim,
            add_default_ground=False,
            num_envs=1,
            headless=True,
            gravity=(0, 0, -9.81),
        )

    def _install_lighting(self):
        self.scene.set_ambient_light([0.3, 0.3, 0.3])
        if self.cfg["sink"]:
            self.scene.add_directional_light(
                [0, 0, -1], [0.3, 0.3, 0.3], position=[0, 0, 1], shadow=False, scale=5, shadow_map_size=2048
            )
        else:
            self.scene.add_directional_light([0, 0, -1], [2.2, 2.2, 2.2], shadow=False, scale=5, shadow_map_size=2048)
            self.scene.add_directional_light([-1, -0.5, -1], [0.7, 0.7, 0.7])
            self.scene.add_directional_light([1, 1, -1], [0.7, 0.7, 0.7])

    def _make_controller(self):
        return CombinedController(
            widowx_deployed_controller_configs(), self.robot, self.control_freq, sim_freq=self.sim_freq
        )

    def _wire(self):
        apply_widowx_urdf_materials(self.scene, self.robot)
        apply_robot_render_material(self.robot)
        self.qpos = SINK_QPOS if self.cfg["sink"] else INIT_QPOS
        self.objs = [self.handler.object_ids["source_obj"], self.handler.object_ids["target_obj"]]
        self.sink = self.handler.object_ids.get("sink")
        self.grasp_checker = WidowXGraspChecker(self.robot, self.scene)
        with open(os.path.join(paths()["model_db_dir"], self.cfg["model_json"])) as f:
            self.model_db = json.load(f)
        # SimplerEnv bridge visual-matching overlay (sink setup uses a different real image).
        # The sink is intentionally left out of the foreground so it reads as background (== upstream).
        overlay_name = "bridge_sink.png" if self.cfg["sink"] else "bridge_real_eval_1.png"
        self._overlay = load_overlay_image(os.path.join(paths()["overlay_dir"], overlay_name))

    def _foreground_ids(self):
        return [link.get_id() for link in self.robot.get_links()] + [o.get_id() for o in self.objs]

    def reset(self, states=None, env_ids=None, seed=0, options=None):
        """Reset the task. ``states``/``env_ids``/``seed`` are the ``BaseTaskEnv.reset`` contract;
        ``options`` is the SimplerEnv episode spec (episode id, robot init options) from the gym
        adapter."""
        cfg = self.cfg
        n = len(cfg["xy"]) * len(cfg["quat"])
        episode_id = (options or {}).get("episode_id")
        if episode_id is None:
            episode_id = np.random.RandomState(seed).randint(n)
        xy = cfg["xy"][(episode_id % n) // len(cfg["quat"])]
        quat = cfg["quat"][episode_id % len(cfg["quat"])]
        self.bbox = [
            np.array(self.model_db[m]["bbox"]["max"]) - np.array(self.model_db[m]["bbox"]["min"])
            for m in (cfg["src"], cfg["tgt"])
        ]
        self.robot.set_root_pose(sapien.Pose([-10, 0, 0]))
        self.robot.set_qpos(self.qpos)
        self.robot.set_qvel(np.zeros(self.robot.dof))
        if self.sink is not None:
            self.sink.set_pose(sapien.Pose([-0.16, 0.13, 0.88], [1, 0, 0, 0]))
            self.sink.lock_motion()
        z = 0.87 + 0.5
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
        rxy = [0.127, 0.06] if cfg["sink"] else [0.147, 0.028]
        rh = 0.85 if cfg["sink"] else ROBOT_INIT_HEIGHT
        rio = (options or {}).get("robot_init_options") or {}
        rxy = rio.get("init_xy", rxy)
        rh = rio.get("init_height", rh)
        self.robot.set_root_pose(sapien.Pose([rxy[0], rxy[1], rh], [0, 0, 0, 1]))
        self.robot.set_qpos(self.qpos)
        self.robot.set_qvel(np.zeros(self.robot.dof))
        if apply_external_states(self, states, env_ids):
            # success is measured against the post-reset object poses, so re-derive them
            self.xyz_after_settle = [o.pose.p.copy() for o in self.objs]
            self.src_bbox_world = quat2mat(self.objs[0].pose.q) @ self.bbox[0]
            self.tgt_bbox_world = quat2mat(self.objs[1].pose.q) @ self.bbox[1]
        self.controller.reset()
        self.consecutive_grasp = 0
        self._elapsed = 0
        return self.get_obs(), {"episode_id": int(episode_id)}

    def _com(self, o):
        return o.pose.transform(o.cmass_local_pose)

    def _evaluate(self):
        cfg = self.cfg
        src, tgt = self.objs[0], self.objs[1]
        sp, tp = self._com(src), self._com(tgt)
        smove = np.linalg.norm(self.xyz_after_settle[0][:2] - src.pose.p[:2])
        omove = [np.linalg.norm(self.xyz_after_settle[1][:2] - tgt.pose.p[:2])]
        moved = (smove > 0.03) and all(x < smove for x in omove)
        g = self.grasp_checker.check_grasp(src)
        self.consecutive_grasp = self.consecutive_grasp + 1 if g else 0
        th, sh = self.tgt_bbox_world / 2, self.src_bbox_world / 2
        off = sp.p - tp.p
        xy_flag = np.linalg.norm(off[:2]) <= np.linalg.norm(th[:2]) + 0.003
        z_flag = (off[2] > 0) and (off[2] - th[2] - sh[2] <= cfg["z_off"])
        src_on = bool(xy_flag and z_flag)
        if cfg["success_complete"]:
            ignore = [tgt.name] + [x.get_name() for x in self.robot.get_links()]
            flag = True
            for c in self.scene.get_contacts():
                other = (
                    c.actor1.name
                    if c.actor0.name == src.name
                    else (c.actor0.name if c.actor1.name == src.name else None)
                )
                if other is not None:
                    imp = np.sum([p.impulse for p in c.points], axis=0)
                    if other not in ignore and np.linalg.norm(imp) > 1e-6:
                        flag = False
                        break
            src_on = src_on and flag
        return {
            "success": src_on,
            "moved_correct_obj": bool(moved),
            "is_src_obj_grasped": bool(g),
            "consecutive_grasp": bool(self.consecutive_grasp >= 5),
            "src_on_target": src_on,
        }

    def get_language_instruction(self):
        return self.cfg["instr"]


class PutSpoonOnTowel(SimplerPutOnTask):
    task_key = "spoon"


class PutCarrotOnPlate(SimplerPutOnTask):
    task_key = "carrot"


class StackGreenOnYellow(SimplerPutOnTask):
    task_key = "stack"


class PutEggplantInBasket(SimplerPutOnTask):
    task_key = "eggplant"
