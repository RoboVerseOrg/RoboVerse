from __future__ import annotations

import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import re
import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import torch
from dm_control import mjcf
from loguru import logger as log
from mujoco import mjx

from metasim.cfg.objects import (
    ArticulationObjCfg,
    PrimitiveCubeCfg,
    PrimitiveCylinderCfg,
    PrimitiveSphereCfg,
)
from metasim.cfg.robots import BaseRobotCfg
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
        self._seed     = seed or 0
        self._mjx_model  = None
        self._robot      = scenario.robot
        self._robot_path = self._robot.mjcf_path
        self.cameras = []
        for camera in scenario.cameras:
            self.cameras.append(camera)

        self._renderer     = None

        self._episode_length_buf = torch.zeros(self.num_envs, dtype=torch.int32)
        self.replay_traj         = False
        self.use_taskdecimation  = False

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
        self._mj_model = self._init_mujoco()
        for geom in self._mj_model.find_all('geom'):
            if geom.mesh is not None:
                geom.contype = 0
                geom.conaffinity = 0

        self._physics = mjcf.Physics.from_mjcf_model(self._mj_model)

        self.body_names = [self._physics.model.body(i).name for i in range(self._physics.model.nbody)]
        self.robot_body_names = [
            body_name for body_name in self.body_names if body_name.startswith(self._mujoco_robot_name)
        ]


        self._mjx_model = self._make_mjx_model()

        pool = self._mjx_model.names
        adr  = self._mjx_model.name_jntadr        

        robot_prefix = self._scenario.robot.name

        self._joint_name2id = {}
        for i, a in enumerate(adr):
            raw = pool[int(a) : pool.find(b"\0", int(a))].decode()
            self._joint_name2id[raw] = i
            short = raw.split("/")[-1]
            self._joint_name2id[short] = i

            if "/" not in raw:
                self._joint_name2id[f"{robot_prefix}/{raw}"] = i

        data= mjx.make_data(self._mjx_model)

        def broadcast_tree(x, N):
            return jax.tree_util.tree_map(
                lambda y: jnp.broadcast_to(y, (N, *y.shape)), x)
        
        #batch data
        self._data =  broadcast_tree(data, self.num_envs)
        self._robot_joint_ids: dict[str, list[int]] = {}

       
        self._substep = self._make_substep(self.decimation)
        if self._scenario.cameras:
            from madrona_mjx.renderer import BatchRenderer
            ref_w, ref_h = self._scenario.cameras[0].width, self._scenario.cameras[0].height

            # Assert all cameras share same resolution
            for cam in self._scenario.cameras[1:]:
                if cam.width != ref_w or cam.height != ref_h:
                    log.warning(
                        f"Camera {cam.name} has different resolution than the first camera. "
                        f"Using the first camera's resolution for rendering."
                    )

            #TODO multi renderer for different cameras
            self._renderer = BatchRenderer(
                m=self._mjx_model,
                gpu_id = 0,  #TODO choose GPU
                num_worlds = self.num_envs,

                #cameras should have same resolution
                batch_render_view_width  = self._scenario.cameras[0].width,
                batch_render_view_height = self._scenario.cameras[0].height,

                enabled_geom_groups = np.array([0, 1, 2, 3, 4, 5], dtype=np.int32),
                enabled_cameras     = np.arange(len(self.cameras), dtype=np.int32),
                use_rasterizer      = False,
                add_cam_debug_geo   = False,
                viz_gpu_hdls        = None,
            )
            self._render_tokens = self._renderer.init(self._data, self._mjx_model)
        # create env
        log.info(f"MJXHandler launched · envs={self.num_envs}")

    def simulate(self) -> None:
        self._data = self._substep(self._mjx_model, self._data)

    def get_states(self) -> TensorState:

        data = self._data 
        N = data.qpos.shape[0]
        robots, objects = {}, {}
        # ---------------------- Robot -------------------------------------------
        r_cfg  = self._scenario.robot
        prefix = f"{r_cfg.name}/"

        qadr_r, vadr_r, jnames_r  = self._sorted_joint_info(prefix)
        aid_r                     = self._sorted_actuator_ids(prefix)
        bid_r, bnames_r           = self._sorted_body_ids(prefix)
        root_id_r                 = bid_r[0]

        root_state_r = jnp.concatenate([
            data.xpos[:, root_id_r],        # (N,3)
            data.xquat[:, root_id_r],       # (N,4)
            data.cvel[:, root_id_r]         # (N,6)
        ], axis=-1)                         # (N,13)

        body_state_r = jnp.concatenate([
            data.xpos[:, bid_r],            # (N,B,3)
            data.xquat[:, bid_r],           # (N,B,4)
            data.cvel[:, bid_r]             # (N,B,6)
        ], axis=-1)                         # (N,B,13)

        joint_pos         = _j2t(data.qpos[:, qadr_r])
        robots[r_cfg.name] = RobotState(
            root_state        = _j2t(root_state_r),
            body_names        = bnames_r,
            body_state        = _j2t(body_state_r),
            joint_pos         = _j2t(data.qpos[:, qadr_r]),        # (N,J)
            joint_vel         = _j2t(data.qvel[:, vadr_r]),       # (N,J)
            joint_pos_target  = _j2t(data.ctrl[:, aid_r]),        # (N,J)
            joint_vel_target  = None,
            joint_effort_target=_j2t(data.actuator_force[:, aid_r])  # (N,J)
        )

        # ---------------------- Objects -----------------------------------------
        for obj in self._scenario.objects:
            prefix = f"{obj.name}/"
            bid_o, bnames_o = self._sorted_body_ids(prefix)
            root_id_o       = bid_o[0]

            root_state_o = jnp.concatenate([
                data.xpos[:, root_id_o],
                data.xquat[:, root_id_o],
                data.cvel[:, root_id_o]
            ], axis=-1)                                           # (N,13)

            if isinstance(obj, BaseRobotCfg):                     # articulated
                qadr_o, vadr_o, jnames_o = self._sorted_joint_info(prefix)
                body_state_o = jnp.concatenate([
                    data.xpos[:, bid_o],
                    data.xquat[:, bid_o],
                    data.cvel[:, bid_o]
                ], axis=-1)                                       # (N,B,13)
                joint_pos  = _j2t(data.qpos[:, qadr_o])
                joint_vel  = _j2t(data.qvel[:, vadr_o])
            else:
                body_state_o = joint_pos = joint_vel = None

            objects[obj.name] = ObjectState(
                root_state = _j2t(root_state_o),
                body_names = bnames_o if body_state_o is not None else None,
                body_state = _j2t(body_state_o) if body_state_o is not None else None,
                joint_pos  = joint_pos,
                joint_vel  = joint_vel,
            )

        # ---------------------- Cameras -----------------------------
        if self._renderer is not None :
            rgb_batch, depth_batch = self.render_batch(data)
            rgb_batch = _j2t(rgb_batch)                          # (N,K,H,W,4)
            depth_batch = _j2t(depth_batch)                      # (N,K,H,W,1)
        cameras = {}
        for cam_i, cam_cfg in enumerate(self.cameras):
            rgb   = depth = None
            if "rgb" in cam_cfg.data_types and rgb_batch is not None:
                rgb = rgb_batch[:, cam_i, ..., :3]         # (N,H,W,3)
            if "depth" in cam_cfg.data_types and depth_batch is not None:
                depth = depth_batch[:, cam_i, ...]         # (N,H,W,1)
            cameras[cam_cfg.name] = CameraState(rgb=rgb, depth=depth)

        return TensorState(objects=objects, robots=robots, cameras=cameras, sensors={})

    def render_batch(self, data: mjx.Data):
        if self._renderer is None or self.render_tokens is None:
            return None, None
        # rgb  : (N, K, H, W, 4)  (RGBA)
        # depth: (N, K, H, W, 1)
        tokens, rgb, depth = self._renderer.render(self._render_tokens, data)
        self._render_tokens = tokens
        return rgb, depth

    def set_states(
        self,
        ts: TensorState,
        env_ids: list[int] | None = None,
        zero_vel: bool = True,
    ) -> None:

        ts= list_state_to_tensor(self,ts)  # convert back to nested dict


        data  = self._data
        model = self._mjx_model
        N     = data.qpos.shape[0]

        idx = jnp.arange(N, dtype=int) if env_ids is None else jnp.array(env_ids, dtype=int)

        qpos, qvel, ctrl = data.qpos, data.qvel, data.ctrl

        capi = self._mj_model

        self._robot_joint_ids = {}
        for rname in ts.robots.keys():
            jnames = self.get_joint_names(rname, sort=True)
            self._robot_joint_ids[rname] = [
                mujoco.mj_name2id(capi, mujoco.mjtObj.mjOBJ_JOINT, jn)
                for jn in jnames
            ]
        self._object_joint_ids = {}
        for oname in ts.objects.keys():
            jnames = self.get_joint_names(oname, sort=True)
            full = [f"{oname}/{jn}" for jn in jnames]
            self._object_joint_ids[oname] = [
                mujoco.mj_name2id(capi, mujoco.mjtObj.mjOBJ_JOINT, fn)
                for fn in full
            ]


        def _scatter_root(qpos, qvel, root_t):
            qpos = qpos.at[idx, :3].set(root_t[:, :3])
            qpos = qpos.at[idx, 3:7].set(root_t[:, 3:7])
            if zero_vel:
                qvel = qvel.at[idx, :6].set(0.)
            else:
                qvel = qvel.at[idx, :3].set(root_t[:, 7:10])
                qvel = qvel.at[idx, 3:6].set(root_t[:, 10:13])
            return qpos, qvel

        # ------------------------ ROBOTS --------------------------------------
        for name, r in ts.robots.items():
            j_names = self.get_joint_names(name, sort=True)
            j_ids = jnp.array(self._robot_joint_ids[name], dtype=int)
            qadr = model.jnt_qposadr[j_ids]
            vadr = model.jnt_dofadr[j_ids]

            qpos, qvel = _scatter_root(qpos, qvel, r.root_state)
            if r.joint_pos is not None:
                qpos = qpos.at[idx[:, None], qadr].set(_t2j(r.joint_pos))
            qvel = qvel.at[idx[:, None], vadr].set(
                jnp.zeros_like(_t2j(r.joint_vel)) if zero_vel else _t2j(r.joint_vel)
            )
            ctrl = ctrl.at[idx[:, None], vadr].set(_t2j(r.joint_pos_target))

        # ------------------------ OBJECTS ------------------------------------
        for name, o in ts.objects.items():
            qpos, qvel = _scatter_root(qpos, qvel, o.root_state)

            if o.joint_pos is not None:
                j_names = self.get_joint_names(name, sort=True)
                # use name2id for object joints too
                j_ids = jnp.array(self._object_joint_ids[name], dtype=int)
                qadr = model.jnt_qposadr[j_ids]
                vadr = model.jnt_dofadr[j_ids]

                qpos = qpos.at[idx[:, None], qadr].set(_t2j(o.joint_pos))
                qvel = qvel.at[idx[:, None], vadr].set(
                    jnp.zeros_like(_t2j(o.joint_vel)) if zero_vel else _t2j(o.joint_vel)
                )

        self._data = self._data.replace(qpos=qpos, qvel=qvel)
        self._data = self._data.replace(ctrl=ctrl)  # (N,nv)

    def set_dof_targets(self, obj_name: str, actions: list[Action]) -> None:
        """
        Normal mode → write targets into Data.ctrl             (N,J)→(N,nv)
        Replay mode → overwrite qpos directly (teleport)       (N,J)→(N,nq)
        """
        self._actions_cache = actions
        N = self.num_envs

        jnames_local = self.get_joint_names(obj_name, sort=True)

        # ---------- (N, J) torch tensor ----------
        tgt_torch = torch.stack(
            [
                torch.tensor(
                    [actions[e]["dof_pos_target"][jn] for jn in jnames_local],
                    dtype=torch.float32
                )
                for e in range(N)
            ],
            dim=0,
        )
        tgt_jax = _t2j(tgt_torch)                                      # (N, J)

        # ----------  joint ids ----------
        if obj_name == self._scenario.robot.name:
            full_names = jnames_local
        else:
            full_names = [f"{obj_name}/{jn}" for jn in jnames_local]


        j_ids = [self._joint_name2id[n] for n in full_names]

        model = self._mjx_model
        vadr  = model.jnt_dofadr[j_ids]      # (J,)
        qadr  = model.jnt_qposadr[j_ids]     # (J,)

        data = self._data
        if self.replay_traj:
            qpos = data.qpos.at[:, qadr].set(tgt_jax)
            self._data = data.replace(qpos=qpos, qvel=data.qvel)
        else:
            ctrl = data.ctrl.at[:, vadr].set(tgt_jax)
            self._data = data.replace(ctrl=ctrl)



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
        for i, (obj, obj_path) in enumerate(zip(self.objects, object_paths)):
            if isinstance(obj, (PrimitiveCubeCfg, PrimitiveCylinderCfg, PrimitiveSphereCfg)):
                xml_str = self._create_primitive_xml(obj)
                obj_mjcf = mjcf.from_xml_string(xml_str)
            else:
                obj_mjcf = mjcf.from_path(obj_path)
            obj_attached = mjcf_model.attach(obj_mjcf)
            if not obj.fix_base_link:
                obj_attached.add("freejoint")
            self.object_body_names.append(obj_attached.full_identifier)
            self.mj_objects[obj.name] = obj_mjcf

        robot_xml = mjcf.from_path(self._robot_path)
        robot_attached = mjcf_model.attach(robot_xml)
        if not self._robot.fix_base_link:
            robot_attached.add("freejoint")
        self.robot_attached = robot_attached
        self.mj_objects[self._robot.name] = robot_xml
        self._mujoco_robot_name = robot_xml.full_identifier
        return mjcf_model


    ############################################################
    ## Misc
    ###########################################################
    def refresh_render(self) -> None:
        pass


    def get_body_names(self, obj_name: str, sort: bool = True) -> list[str]:
        if isinstance(self.object_dict[obj_name], ArticulationObjCfg):
            m=self._mj_model
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
        
    def _make_mjx_model(self) -> mjcf.RootElement:
        robot_dir = Path(self._scenario.robot.mjcf_path).parent
        asset_dir = robot_dir / "assets"

        xml_str = self._mj_model.to_xml_string()
        xml_str = re.sub(r'-[0-9a-f]{40}\.(stl|obj|dae)', r'.\1', xml_str)

        tmp_xml = Path(tempfile.mktemp(suffix=".xml", dir=asset_dir))
        tmp_xml.write_text(xml_str)


        self._mj_model  = mujoco.MjModel.from_xml_path(str(tmp_xml))
        tmp_xml.unlink(missing_ok=True)
        return mjx.put_model(self._mj_model)


    def _make_substep(self, n_sub: int):
        def _one_env(model, data):
            def body(d, _):
                return mjx.step(model, d), None
            data, _ = jax.lax.scan(body, data, None, length=n_sub)
            return data

        batched = jax.vmap(_one_env, in_axes=(None, 0))
        return jax.jit(batched, donate_argnums=(1,))



    def _create_primitive_xml(self, obj) -> str:
            """Generate a minimal MJCF string for primitive objects."""
            if isinstance(obj, PrimitiveCubeCfg):
                size, gtype = (
                    f"{obj.half_size[0]} {obj.half_size[1]} {obj.half_size[2]}",
                    "box",
                )
            elif isinstance(obj, PrimitiveCylinderCfg):
                size, gtype = f"{obj.radius} {obj.height}", "cylinder"
            elif isinstance(obj, PrimitiveSphereCfg):
                size, gtype = f"{obj.radius}", "sphere"
            else:
                raise ValueError("Unknown primitive type")

            rgba = f"{obj.color[0]} {obj.color[1]} {obj.color[2]} 1"
            return (
                f'<mujoco model="{obj.name}_model">'
                f'<worldbody><body name="{gtype}_body" pos="0 0 0">'
                f'<geom type="{gtype}" size="{size}" rgba="{rgba}"/>'
                f"</body></worldbody></mujoco>"
            )
    _KIND_META = {
        "joint"    : ("njnt" , "name_jntadr"),
        "actuator" : ("nu"   , "name_actuatoradr"),
        "body"     : ("nbody", "name_bodyadr"),
    }


    def _decode_name(self, pool: bytes, adr: int) -> str:
        end = pool.find(b"\x00", adr)
        return pool[adr:end].decode()

    def _names_ids_mjx(self, kind: str):
        model = self._mjx_model
        size_attr, adr_attr = self._KIND_META[kind]
        size   = int(getattr(model, size_attr))
        adr_arr= getattr(model, adr_attr)
        pool   = model.names
        names  = [self._decode_name(pool, int(adr_arr[i])) for i in range(size)]
        ids    = list(range(size))
        return names, ids

    def _sorted_joint_info(self, prefix: str):
        names, ids = self._names_ids_mjx("joint")
        filt = [(n, i) for n, i in zip(names, ids) if n.startswith(prefix)]
        if not filt:
            raise ValueError(f"No joints start with '{prefix}'")
        filt.sort(key=lambda t: t[0])
        names_sorted, j_ids = zip(*filt)

        model = self._mjx_model
        qadr  = model.jnt_qposadr[list(j_ids)]
        vadr  = model.jnt_dofadr[list(j_ids)]
        local = [n.split("/")[-1] for n in names_sorted]
        return jnp.asarray(qadr), jnp.asarray(vadr), local

    def _sorted_actuator_ids(self, prefix: str):
        names, ids = self._names_ids_mjx("actuator")
        filt = [(n, i) for n, i in zip(names, ids) if n.startswith(prefix)]
        return [i for n, i in sorted(filt, key=lambda t: t[0])]

    def _sorted_body_ids(self, prefix: str):
        names, ids = self._names_ids_mjx("body")
        filt = [(n, i) for n, i in zip(names, ids)
                if n.startswith(prefix) and n != prefix]
        filt.sort(key=lambda t: t[0])
        body_ids    = [i for _, i in filt]
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
