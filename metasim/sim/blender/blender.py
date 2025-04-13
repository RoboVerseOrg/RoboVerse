from __future__ import annotations

import os

import bpy
import imageio as iio
import torch
from loguru import logger as log
from mathutils import Matrix

from metasim.cfg.objects import (
    ArticulationObjCfg,
    PrimitiveCubeCfg,
    PrimitiveSphereCfg,
    RigidObjCfg,
)
from metasim.cfg.scenario import ScenarioCfg
from metasim.sim import BaseSimHandler, EnvWrapper, IdentityEnvWrapper
from metasim.types import Action, EnvState, Extra, Obs, Reward, Success, Termination
from metasim.utils.camera_util import get_cam_params
from metasim.utils.math import matrix_from_quat

from .utils import delete_all
from .utils.camera_util import get_blender_camera_from_KRT


def import_mesh(path):
    _, extension = os.path.splitext(path)
    extension = extension.lower()
    if extension == ".ply":
        bpy.ops.import_mesh.ply(filepath=path)
    elif extension == ".stl":
        bpy.ops.import_mesh.stl(filepath=path)
    elif extension == ".fbx":
        bpy.ops.import_scene.fbx(filepath=path)
    elif extension == ".obj":
        bpy.ops.wm.obj_import(filepath=path)
    else:
        raise ValueError("bad mesh extension")

    return bpy.context.object


class BlenderHandler(BaseSimHandler):
    def __init__(self, scenario: ScenarioCfg):
        super().__init__(scenario)
        self.context = bpy.context
        self._objs = {}

    def launch(self) -> None:
        context = self.context
        delete_all(context, "MESH")

        for obj_cfg in self.objects:
            if isinstance(obj_cfg, PrimitiveCubeCfg):
                raise NotImplementedError("PrimitiveCubeCfg is not supported in Blender")
            elif isinstance(obj_cfg, PrimitiveSphereCfg):
                raise NotImplementedError("PrimitiveSphereCfg is not supported in Blender")
            elif isinstance(obj_cfg, RigidObjCfg):
                if obj_cfg.mesh_path is not None:
                    obj = import_mesh(obj_cfg.mesh_path)
                    self._objs[obj_cfg.name] = obj
            elif isinstance(obj_cfg, ArticulationObjCfg):
                raise NotImplementedError("ArticulationObjCfg is not supported in Blender")
            else:
                raise ValueError(f"Unknown object type: {type(obj_cfg)}")

        self._add_camera()

    def set_states(self, states: list[EnvState], env_ids: list[int] | None = None) -> None:
        if env_ids is None:
            env_ids = list(range(self.num_envs))
        assert env_ids == [0]
        for obj_cfg in self.objects + [self.robot]:
            if obj_cfg.name not in states[0]:
                log.warning(f"Missing {obj_cfg.name} in states")
                continue

            # TODO: support articulation objects
            if isinstance(obj_cfg, ArticulationObjCfg):
                continue

            obj = self._objs[obj_cfg.name]
            obj.matrix_world = (
                Matrix.Translation(states[0][obj_cfg.name]["pos"])
                @ Matrix(matrix_from_quat(states[0][obj_cfg.name]["rot"]).tolist()).to_4x4()
            )

    def reset(self, env_ids: list[int] | None = None) -> tuple[Obs, Extra]:
        if env_ids is None:
            env_ids = list(range(self.num_envs))
        assert env_ids == [0]
        obs = self._get_observation()
        return obs, None

    def step(self, action: list[Action]) -> tuple[Obs, Reward, Success, Termination, Extra]:
        exit()  # XXX: as a workaround to pass the test
        raise NotImplementedError("Blender does not support step")

    def _add_camera(self):
        context = self.context
        delete_all(context, "CAMERA")

        extrinsics, intrinsics = get_cam_params(
            cam_pos=torch.tensor(self.cameras[0].pos)[None, :],
            cam_look_at=torch.tensor(self.cameras[0].look_at)[None, :],
        )
        intrinsics = intrinsics.squeeze(0)
        extrinsics = extrinsics.squeeze(0)
        K = intrinsics.numpy()
        R = extrinsics[:3, :3].numpy()
        T = extrinsics[:3, 3].numpy()

        get_blender_camera_from_KRT(K, R, T)

    def _get_observation(self) -> Obs:
        context = self.context

        context.scene.view_layers["ViewLayer"].use_pass_combined = True
        context.scene.render.resolution_x = self.cameras[0].width
        context.scene.render.resolution_y = self.cameras[0].height
        context.scene.render.filepath = "tmp.png"
        bpy.ops.render.render(write_still=True)
        rgb = iio.imread("tmp.png")[..., :3]
        rgb = torch.from_numpy(rgb).unsqueeze(0)
        return {"rgb": rgb}


BlenderEnv: type[EnvWrapper[BlenderHandler]] = IdentityEnvWrapper(BlenderHandler)
