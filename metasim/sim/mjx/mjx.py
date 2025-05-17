from __future__ import annotations

import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import torch
from dm_control import mjcf
from loguru import logger as log
from mujoco import mjtJoint, mjx

from metasim.cfg.objects import (
    ArticulationObjCfg,
    PrimitiveCubeCfg,
    PrimitiveCylinderCfg,
    PrimitiveSphereCfg,
)
from metasim.cfg.scenario import ScenarioCfg
from metasim.constants import TaskType
from metasim.sim import BaseSimHandler, EnvWrapper, GymEnvWrapper
from metasim.types import Action
from metasim.utils.state import CameraState, ObjectState, RobotState, TensorState, list_state_to_tensor


def _j2t(arr: jax.Array, device: str | torch.device | None = "cuda") -> torch.Tensor:
    if device is not None:
        tgt = torch.device(device)
        plat = "gpu" if tgt.type == "cuda" else tgt.type
        if arr.device.platform != plat:
            arr = jax.device_put(arr, jax.devices(plat)[tgt.index or 0])
    t = torch.from_dlpack(jax.dlpack.to_dlpack(arr))
    return t


def _t2j(arr: torch.Tensor, device: str | torch.device | None = "cuda") -> jnp.ndarray:
    if device is not None and arr.device != torch.device(device):
        arr = arr.to(device, non_blocking=True)
    x = jax.dlpack.from_dlpack(torch.utils.dlpack.to_dlpack(arr))
    return x


class MJXHandler(BaseSimHandler):
    def __init__(self, scenario: ScenarioCfg, *, seed: int | None = None):
        super().__init__(scenario)

        self._scenario = scenario
        self._seed = seed or 0
        self._mjx_model = None
        self._robot = scenario.robot
        self._robot_path = self._robot.mjcf_path
        self.cameras = []
        for camera in scenario.cameras:
            self.cameras.append(camera)

        self._renderer = None

        self._episode_length_buf = torch.zeros(self.num_envs, dtype=torch.int32)
        self.replay_traj = False
        self.use_taskdecimation = False
        self._object_root_path_cache: dict[str, str] = {}
        self._object_root_bid_cache: dict[str, int] = {}
        self._fix_path_cache: dict[str, int] = {}
        self._gravity_compensation = not self._robot.enabled_gravity

        if self.use_taskdecimation:
            self.decimation = self.scenario.decimation
        elif self.replay_traj:
            log.warning("Warning: hard coding decimation to 1 for object states")
            self.decimation = 1
        elif self.task is not None and self.task.task_type == TaskType.LOCOMOTION:
            self.decimation = self.scenario.decimation
        else:
            log.warning("Warning: hard coding decimation to 25 for replaying trajectories")
            self.decimation = 25

    def launch(self) -> None:
        mjcf_root = self._init_mujoco()

        tmp_dir = tempfile.mkdtemp()
        mjcf.export_with_assets(mjcf_root, tmp_dir)
        xml_path = next(Path(tmp_dir).glob("*.xml"))
        self._mj_model = mujoco.MjModel.from_xml_path(str(xml_path))

        self.body_names = [
            mujoco.mj_id2name(self._mj_model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(self._mj_model.nbody)
        ]
        self.robot_body_names = [n for n in self.body_names if n.startswith(self._mujoco_robot_name)]

        if self.cameras:
            max_w = max(c.width for c in self.cameras)
            max_h = max(c.height for c in self.cameras)
            self._renderer = mujoco.Renderer(self._mj_model, width=max_w, height=max_h)
            self._render_data = mujoco.MjData(self._mj_model)

        log.info(f"MJXHandler launched · envs={self.num_envs}")

    def simulate(self) -> None:
        if self._gravity_compensation:
            self._disable_robotgravity()
        self._data = self._substep(self._mjx_model, self._data)

    def get_states(self, env_ids: list[int] | None = None):
        data = self._data  # mjx_env.Data, shape (N, …)
        N = data.qpos.shape[0]
        idx_np = np.arange(N) if env_ids is None else np.asarray(env_ids, dtype=int)
        idx = jnp.asarray(idx_np, dtype=jnp.int32)  # jax array for slicing
        B = idx.shape[0]  # batch size actually returned

        robots: dict[str, RobotState] = {}
        objects: dict[str, ObjectState] = {}

        # ===================== Robot =====================================
        r_cfg = self._scenario.robot
        prefix = f"{r_cfg.name}/"

        qadr_r, vadr_r, _ = self._sorted_joint_info(prefix)
        aid_r = self._sorted_actuator_ids(prefix)

        root_bid_r = self._object_root_bid_cache.get(
            r_cfg.name,
            self._sorted_body_ids(prefix)[0][0],
        )

        bid_r, bnames_r = self._sorted_body_ids(prefix)
        if root_bid_r not in bid_r:
            bid_r.insert(0, root_bid_r)
            bnames_r.insert(0, self.mj_objects[r_cfg.name].full_identifier)

        root_state_r = jnp.concatenate(
            [data.xpos[idx, root_bid_r], data.xquat[idx, root_bid_r], data.cvel[idx, root_bid_r]],
            axis=-1,  # (B, 13)
        )
        body_state_r = jnp.concatenate(
            [data.xpos[idx[:, None], bid_r], data.xquat[idx[:, None], bid_r], data.cvel[idx[:, None], bid_r]],
            axis=-1,  # (B, Bbody, 13)
        )

        robots[r_cfg.name] = RobotState(
            root_state=_j2t(root_state_r),
            body_names=bnames_r,
            body_state=_j2t(body_state_r),
            joint_pos=_j2t(data.qpos[idx[:, None], qadr_r]),
            joint_vel=_j2t(data.qvel[idx[:, None], vadr_r]),
            joint_pos_target=_j2t(data.ctrl[idx[:, None], aid_r]),
            joint_vel_target=None,
            joint_effort_target=_j2t(data.actuator_force[idx[:, None], aid_r]),
        )

        # ===================== Objects ===================================
        for obj in self._scenario.objects:
            prefix = f"{obj.name}/"

            root_bid_o = self._object_root_bid_cache[obj.name]
            bid_o, bnames_o = self._sorted_body_ids(prefix)
            if root_bid_o not in bid_o:
                bid_o.insert(0, root_bid_o)
                bnames_o.insert(0, self.mj_objects[obj.name].full_identifier)

            root_state_o = jnp.concatenate(
                [data.xpos[idx, root_bid_o], data.xquat[idx, root_bid_o], data.cvel[idx, root_bid_o]],
                axis=-1,  # (B, 13)
            )

            if isinstance(obj, ArticulationObjCfg):
                qadr_o, vadr_o, _ = self._sorted_joint_info(prefix)
                body_state_o = jnp.concatenate(
                    [data.xpos[idx[:, None], bid_o], data.xquat[idx[:, None], bid_o], data.cvel[idx[:, None], bid_o]],
                    axis=-1,  # (B, Bbody, 13)
                )
                objects[obj.name] = ObjectState(
                    root_state=_j2t(root_state_o),
                    body_names=bnames_o,
                    body_state=_j2t(body_state_o),
                    joint_pos=_j2t(data.qpos[idx[:, None], qadr_o]),
                    joint_vel=_j2t(data.qvel[idx[:, None], vadr_o]),
                )
            else:  # rigid object without joints
                objects[obj.name] = ObjectState(
                    root_state=_j2t(root_state_o),
                )

        # ===================== Cameras ===================================
        camera_states = {}
        want_any_rgb = any("rgb" in cam.data_types for cam in self.cameras)
        want_any_dep = any("depth" in cam.data_types for cam in self.cameras)

        if want_any_rgb or want_any_dep:
            for cam in self.cameras:
                cam_id = f"{cam.name}_custom"
                want_rgb = "rgb" in cam.data_types
                want_dep = "depth" in cam.data_types

                rgb_frames, dep_frames = [], []

                for env_id in idx_np:
                    slice_data = jax.tree_util.tree_map(lambda x: x[env_id], data)  # noqa: B023
                    mjx.get_data_into(self._render_data, self._mj_model, slice_data)
                    mujoco.mj_forward(self._mj_model, self._render_data)

                    if want_rgb:
                        self._renderer.disable_depth_rendering()
                        self._renderer.update_scene(self._render_data, camera=cam_id)
                        rgb = self._renderer.render()
                        rgb_frames.append(torch.from_numpy(rgb.copy()))

                    if want_dep:
                        self._renderer.enable_depth_rendering()
                        self._renderer.update_scene(self._render_data, camera=cam_id)
                        depth = self._renderer.render()
                        dep_frames.append(torch.from_numpy(depth.copy()))

                def _stk(frames):
                    return None if not frames else torch.stack(frames, dim=0)

                camera_states[cam.name] = CameraState(
                    rgb=_stk(rgb_frames) if want_rgb else None,
                    depth=_stk(dep_frames) if want_dep else None,
                )

        return TensorState(objects=objects, robots=robots, cameras=camera_states, sensors={})

    def set_states(
        self,
        ts: TensorState,
        env_ids: list[int] | None = None,
        *,
        zero_vel: bool = True,
    ) -> None:
        ts = list_state_to_tensor(self, ts)
        self._init_mjx_once(ts)

        data = self._data
        model = self._mjx_model

        N = data.qpos.shape[0]
        idx = jnp.arange(N, dtype=jnp.int32) if env_ids is None else jnp.asarray(env_ids, dtype=jnp.int32)
        self._ensure_id_cache(ts)

        qpos, qvel, ctrl = data.qpos, data.qvel, data.ctrl

        def _write_root_free(qpos, qvel, root_jid, root_state):
            jtype = model.jnt_type[root_jid]
            qadr = model.jnt_qposadr[root_jid]
            vadr = model.jnt_dofadr[root_jid]

            if jtype == mjtJoint.mjJNT_FREE:
                qpos_ = _t2j(root_state[:, :7])
                qpos = qpos.at[idx, qadr : qadr + 7].set(qpos_)
                if zero_vel:
                    qvel = qvel.at[idx, vadr : vadr + 6].set(0.0)
                else:
                    qvel_ = _t2j(root_state[:, 7:13])
                    qvel = qvel.at[idx, vadr : vadr + 6].set(qvel_)
            elif jtype in (mjtJoint.mjJNT_HINGE, mjtJoint.mjJNT_SLIDE):
                qpos = qpos.at[idx, qadr].set(_t2j(root_state[:, 0]))
                qvel = qvel.at[idx, vadr].set(0.0 if zero_vel else _t2j(root_state[:, 7]))
            return qpos, qvel

        def _write_joints_ctrl(qpos, qvel, ctrl, j_ids, a_ids, joint_pos, joint_vel, target):
            if j_ids.size == 0 or joint_pos is None:
                return qpos, qvel, ctrl

            qadr = model.jnt_qposadr[j_ids]
            vadr = model.jnt_dofadr[j_ids]
            qpos = qpos.at[idx[:, None], qadr].set(_t2j(joint_pos))
            qvel = qvel.at[idx[:, None], vadr].set(jnp.zeros_like(_t2j(joint_vel)) if zero_vel else _t2j(joint_vel))
            if a_ids is not None and target is not None:
                ctrl = ctrl.at[idx[:, None], a_ids].set(_t2j(target))
            return qpos, qvel, ctrl

        def _process_entity(qpos, qvel, ctrl, name, st, j_map, a_map):
            fixed_root = name in self._fix_path_cache
            j_ids = j_map.get(name)

            if fixed_root:
                non_root_jids = j_ids
            else:
                root_jid = int(j_ids[0])
                qpos, qvel = _write_root_free(qpos, qvel, root_jid, st.root_state)
                non_root_jids = j_ids[1:] if j_ids.size > 1 else jnp.empty(0, int)

            if st.joint_pos is not None and non_root_jids.size > 0:
                qpos, qvel, ctrl = _write_joints_ctrl(
                    qpos,
                    qvel,
                    ctrl,
                    non_root_jids,
                    a_map.get(name),
                    st.joint_pos,
                    st.joint_vel,
                    getattr(st, "joint_pos_target", None),
                )
            return qpos, qvel, ctrl

        for n, r in ts.robots.items():
            qpos, qvel, ctrl = _process_entity(qpos, qvel, ctrl, n, r, self._robot_joint_ids, self._robot_act_ids)

        for n, o in ts.objects.items():
            qpos, qvel, ctrl = _process_entity(qpos, qvel, ctrl, n, o, self._object_joint_ids, self._object_act_ids)

        self._data = self._data.replace(qpos=qpos, qvel=qvel, ctrl=ctrl)

        self._data = jax.vmap(lambda d: mjx.forward(self._mjx_model, d))(self._data)

    def _ensure_id_cache(self, ts: TensorState):
        if hasattr(self, "_robot_joint_ids"):
            return

        capi = self._mj_model

        self._robot_joint_ids, self._robot_act_ids = {}, {}
        for rname in ts.robots:
            jfull = [f"{rname}/{jn}" for jn in self._get_jnames(rname, sort=True)]
            jids = [mujoco.mj_name2id(capi, mujoco.mjtObj.mjOBJ_JOINT, n) for n in jfull]
            aids = self._sorted_actuator_ids(f"{rname}/")
            self._robot_joint_ids[rname] = jnp.asarray(jids, dtype=jnp.int32)
            self._robot_act_ids[rname] = jnp.asarray(aids, dtype=jnp.int32)

        self._object_joint_ids, self._object_act_ids = {}, {}
        for oname in ts.objects:
            jfull = [f"{oname}/{jn}" for jn in self._get_jnames(oname, sort=True)]
            jids = [mujoco.mj_name2id(capi, mujoco.mjtObj.mjOBJ_JOINT, n) for n in jfull]
            aids = self._sorted_actuator_ids(f"{oname}/")
            self._object_joint_ids[oname] = jnp.asarray(jids, dtype=jnp.int32)
            self._object_act_ids[oname] = jnp.asarray(aids, dtype=jnp.int32)

    def _disable_robotgravity(self):
        model = self._mjx_model
        data = self._data

        g = jnp.array([0.0, 0.0, -9.81])
        ids = jnp.asarray([
            mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_BODY, name) for name in self.robot_body_names
        ])
        mass = model.body_mass[ids]
        force = -g * mass[:, None]
        xfrc = data.xfrc_applied.at[:, :, :].set(0.0)
        xfrc = xfrc.at[:, ids, 0:3].set(force)

        self._data = data.replace(xfrc_applied=xfrc)

    def _init_mjx_once(self, ts: TensorState) -> None:
        if getattr(self, "_mjx_done", False):
            return

        def _write_fixed_body(name, root_state):
            pos = root_state[0, :3].cpu().numpy()
            quat = root_state[0, 3:7].cpu().numpy()
            full = self._fix_path_cache[name]
            bid = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_BODY, full)
            self._mj_model.body_pos[bid] = pos
            self._mj_model.body_quat[bid] = quat

        for n in self._fix_path_cache:
            if n in ts.objects:
                _write_fixed_body(n, ts.objects[n].root_state)
            elif n in ts.robots:
                _write_fixed_body(n, ts.robots[n].root_state)
        self._init_mjx()
        self._mjx_done = True

    def set_dof_targets(
        self,
        obj_name: str,
        actions: list[Action],
    ) -> None:
        """
        replay_traj = False → targets go to Data.ctrl
        replay_traj = True  → qpos is overwritten (teleport)
        """
        self._actions_cache = actions

        # -------- build (N, J) tensor ---------------------------------
        jnames_local = self.get_joint_names(obj_name, sort=True)
        tgt_torch = torch.stack(
            [
                torch.tensor(
                    [actions[e]["dof_pos_target"][jn] for jn in jnames_local],
                    dtype=torch.float32,
                )
                for e in range(self.num_envs)
            ],
            dim=0,
        )  # (N, J)
        tgt_jax = _t2j(tgt_torch)

        # -------- id maps ---------------------------------------------
        if obj_name == self._scenario.robot.name:
            j_ids = self._robot_joint_ids[obj_name]
            a_ids = self._robot_act_ids.get(obj_name)
        else:
            j_ids = self._object_joint_ids[obj_name]
            a_ids = self._object_act_ids.get(obj_name)

        model = self._mjx_model
        qadr = model.jnt_qposadr[j_ids]  # (J,)

        data = self._data
        if self.replay_traj:
            new_qpos = data.qpos.at[:, qadr].set(tgt_jax)
            self._data = data.replace(qpos=new_qpos)
        else:
            new_ctrl = data.ctrl.at[:, a_ids].set(tgt_jax)
            self._data = data.replace(ctrl=new_ctrl)

    def close(self):
        pass

    ############################################################
    ## Utils
    ############################################################
    def _init_mujoco(self) -> mjcf.RootElement:
        """Build MJCF tree (one robot, no task-xml branch)."""
        mjcf_model = mjcf.RootElement()

        ## Optional: Add ground grid
        # mjcf_model.asset.add('texture', name="texplane", type="2d", builtin="checker", width=512, height=512, rgb1=[0.2, 0.3, 0.4], rgb2=[0.1, 0.2, 0.3])
        # mjcf_model.asset.add('material', name="matplane", reflectance="0.", texture="texplane", texrepeat=[1, 1], texuniform=True)

        camera_max_width = 640
        camera_max_height = 480
        for camera in self.cameras:
            direction = np.array([
                camera.look_at[0] - camera.pos[0],
                camera.look_at[1] - camera.pos[1],
                camera.look_at[2] - camera.pos[2],
            ])
            direction = direction / np.linalg.norm(direction)
            up = np.array([0, 0, 1])
            right = np.cross(direction, up)
            right = right / np.linalg.norm(right)
            up = np.cross(right, direction)

            camera_params = {
                "pos": f"{camera.pos[0]} {camera.pos[1]} {camera.pos[2]}",
                "mode": "fixed",
                "fovy": camera.vertical_fov,
                "xyaxes": f"{right[0]} {right[1]} {right[2]} {up[0]} {up[1]} {up[2]}",
                "resolution": f"{camera.width} {camera.height}",
            }
            mjcf_model.worldbody.add("camera", name=f"{camera.name}_custom", **camera_params)

            camera_max_width = max(camera_max_width, camera.width)
            camera_max_height = max(camera_max_height, camera.height)

        for child in mjcf_model.visual._children:
            if child.tag == "global":
                child.offwidth = camera_max_width
                child.offheight = camera_max_height

        # Add ground grid, light, and skybox
        mjcf_model.asset.add(
            "texture",
            name="texplane",
            type="2d",
            builtin="checker",
            width=512,
            height=512,
            rgb1=[0, 0, 0],
            rgb2=[1.0, 1.0, 1.0],
        )
        mjcf_model.asset.add(
            "material", name="matplane", reflectance="0.2", texture="texplane", texrepeat=[1, 1], texuniform=True
        )
        ground = mjcf_model.worldbody.add(
            "geom",
            type="plane",
            pos="0 0 0",
            size="100 100 0.001",
            quat="1 0 0 0",
            condim="3",
            conaffinity="15",
            material="matplane",
        )

        self.object_body_names = []
        self.mj_objects = {}
        object_paths = []
        for obj in self.objects:
            object_paths.append(obj.mjcf_path)

        for obj, obj_path in zip(self.objects, object_paths):
            if isinstance(obj, (PrimitiveCubeCfg, PrimitiveCylinderCfg, PrimitiveSphereCfg)):
                obj_mjcf = mjcf.from_xml_string(self._create_primitive_xml(obj))
            else:
                obj_mjcf = mjcf.from_path(obj_path)
            obj_mjcf.model = obj.name

            if obj.fix_base_link:
                obj_attached = mjcf_model.attach(obj_mjcf)
                self._fix_path_cache[obj.name] = obj_attached.full_identifier
            else:
                obj_attached = mjcf_model.attach(obj_mjcf)
                obj_attached.add("freejoint")
            full_path = obj_attached.full_identifier
            self.object_body_names.append(full_path)
            self._object_root_path_cache[obj.name] = full_path
            self.mj_objects[obj.name] = obj_attached
        robot_xml = mjcf.from_path(self._robot_path)

        if self._robot.fix_base_link:
            robot_attached = mjcf_model.attach(robot_xml)
            self._fix_path_cache[self._robot.name] = robot_attached.full_identifier
        else:
            robot_attached = mjcf_model.attach(robot_xml)
            robot_attached.add("freejoint")

        full_path = robot_attached.full_identifier
        self._robot_root_path_cache = {self._robot.name: full_path}
        self.mj_objects[self._robot.name] = robot_attached
        self._mujoco_robot_name = full_path

        return mjcf_model

    ############################################################
    ## Misc
    ###########################################################
    def refresh_render(self) -> None:
        pass

    def get_body_names(self, obj_name: str, sort: bool = True) -> list[str]:
        if isinstance(self.object_dict[obj_name], ArticulationObjCfg):
            m = self._mj_model
            names = [self._mj_model.body(i).name for i in range(self._mj_model.nbody)]
            names = [name.split("/")[-1] for name in names if name.split("/")[0] == obj_name]
            names = [name for name in names if name != ""]
            if sort:
                names.sort()
            return names
        else:
            return []

    def get_joint_names(self, obj_name: str, sort: bool = True) -> list[str]:
        if isinstance(self.object_dict[obj_name], ArticulationObjCfg):
            joint_names = [
                self._mj_model.joint(joint_id).name
                for joint_id in range(self._mj_model.njnt)
                if self._mj_model.joint(joint_id).name.startswith(obj_name + "/")
            ]
            joint_names = [name.split("/")[-1] for name in joint_names]
            joint_names = [name for name in joint_names if name != ""]
            if sort:
                joint_names.sort()
            return joint_names
        else:
            return []

    def _get_jnames(self, obj_name: str, sort: bool = True) -> list[str]:
        joint_names = [
            self._mj_model.joint(joint_id).name
            for joint_id in range(self._mj_model.njnt)
            if self._mj_model.joint(joint_id).name.startswith(obj_name + "/")
        ]
        joint_names = [name.split("/")[-1] for name in joint_names]
        if sort:
            joint_names.sort()
        return joint_names

    # ------------------------------------------------------------
    #  MJX helpers
    # ------------------------------------------------------------
    def _build_joint_name_map(self) -> None:
        pool = self._mjx_model.names
        adr = self._mjx_model.name_jntadr
        robot_prefix = self._scenario.robot.name
        self._joint_name2id = {}

        for jid, a in enumerate(adr):
            raw = pool[int(a) : pool.find(b"\0", int(a))].decode()
            self._joint_name2id[raw] = jid
            self._joint_name2id[raw.split("/")[-1]] = jid
            if "/" not in raw:
                self._joint_name2id[f"{robot_prefix}/{raw}"] = jid

    def _build_root_bid_cache(self) -> None:
        for name, mjcf_body in self.mj_objects.items():
            full = mjcf_body.full_identifier
            bid = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_BODY, full)
            self._object_root_bid_cache[name] = bid

    def _init_mjx(self) -> None:
        if self._mj_model.opt.solver == mujoco.mjtSolver.mjSOL_PGS:
            self._mj_model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
        self._mjx_model = mjx.put_model(self._mj_model)
        self._build_joint_name_map()
        self._build_root_bid_cache()

        # batched empty data
        data_single = mjx.make_data(self._mjx_model)

        def _broadcast(x):
            return jax.tree_util.tree_map(lambda y: jnp.broadcast_to(y, (self.num_envs, *y.shape)), x)

        self._data = _broadcast(data_single)

        # sub-step kernel
        self._substep = self._make_substep(self.decimation)

    def _make_substep(self, n_sub: int):
        def _one_env(model, data):
            def body(d, _):
                d = mjx.step(model, d)
                return d, None

            data, _ = jax.lax.scan(body, data, None, length=100)
            return data

        batched = jax.vmap(_one_env, in_axes=(None, 0))
        return jax.jit(batched)

    def _create_primitive_xml(self, obj):
        if isinstance(obj, PrimitiveCubeCfg):
            size_str = f"{obj.half_size[0]} {obj.half_size[1]} {obj.half_size[2]}"
            type_str = "box"
        elif isinstance(obj, PrimitiveCylinderCfg):
            size_str = f"{obj.radius} {obj.height}"
            type_str = "cylinder"
        elif isinstance(obj, PrimitiveSphereCfg):
            size_str = f"{obj.radius}"
            type_str = "sphere"
        else:
            raise ValueError("Unknown primitive type")

        rgba_str = f"{obj.color[0]} {obj.color[1]} {obj.color[2]} 1"
        xml = f"""
        <mujoco model="{obj.name}_model">
        <worldbody>
            <body name="{type_str}_body" pos="{0} {0} {0}">
            <geom name="{type_str}_geom" type="{type_str}" size="{size_str}" rgba="{rgba_str}"/>
            </body>
        </worldbody>
        </mujoco>
        """
        return xml.strip()

    _KIND_META = {
        "joint": ("njnt", "name_jntadr"),
        "actuator": ("nu", "name_actuatoradr"),
        "body": ("nbody", "name_bodyadr"),
    }

    def _decode_name(self, pool: bytes, adr: int) -> str:
        end = pool.find(b"\x00", adr)
        return pool[adr:end].decode()

    def _names_ids_mjx(self, kind: str):
        model = self._mjx_model
        size_attr, adr_attr = self._KIND_META[kind]
        size = int(getattr(model, size_attr))
        adr_arr = getattr(model, adr_attr)
        pool = model.names
        names = [self._decode_name(pool, int(adr_arr[i])) for i in range(size)]
        ids = list(range(size))
        return names, ids

    def _sorted_joint_info(self, prefix: str):
        names, ids = self._names_ids_mjx("joint")
        filt = [(n, i) for n, i in zip(names, ids) if n.startswith(prefix)]
        if not filt:
            raise ValueError(f"No joints start with '{prefix}'")
        filt.sort(key=lambda t: t[0])
        names_sorted, j_ids = zip(*filt)

        model = self._mjx_model
        qadr = model.jnt_qposadr[list(j_ids)]
        vadr = model.jnt_dofadr[list(j_ids)]
        local = [n.split("/")[-1] for n in names_sorted]
        return jnp.asarray(qadr), jnp.asarray(vadr), local

    def _sorted_actuator_ids(self, prefix: str) -> list[int]:
        """
        Return actuator ids whose *name* starts with `prefix`,
        ordered by **lexicographical order of actuator names**,
        exactly一致 with _get_actuator_reindex on MuJoCo-CPU.
        """
        names, ids = self._names_ids_mjx("actuator")  # parallel lists

        # pick actuators under the given prefix
        selected = [(n, i) for n, i in zip(names, ids) if n.startswith(prefix)]

        # sort by name, keep the original aid
        selected.sort(key=lambda t: t[0])

        return [aid for _, aid in selected]

    def _sorted_body_ids(self, prefix: str):
        names, ids = self._names_ids_mjx("body")
        filt = [(n, i) for n, i in zip(names, ids) if n.startswith(prefix) and n != prefix]
        filt.sort(key=lambda t: t[0])
        body_ids = [i for _, i in filt]
        local_names = [n.split("/")[-1] for n, _ in filt]
        return body_ids, local_names

    @property
    def num_envs(self) -> int:
        return self._scenario.num_envs

    @property
    def episode_length_buf(self) -> list[int]:
        return [self._episode_length_buf]

    @property
    def actions_cache(self) -> list[Action]:
        return self._actions_cache

    @property
    def device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


MJXEnv: type[EnvWrapper[MJXHandler]] = GymEnvWrapper(MJXHandler)
