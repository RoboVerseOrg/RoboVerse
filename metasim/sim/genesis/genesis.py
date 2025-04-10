from __future__ import annotations

import genesis as gs
import numpy as np
import rootutils
import torch
from genesis.engine.entities.rigid_entity import RigidEntity, RigidJoint
from genesis.vis.camera import Camera
from loguru import logger as log

rootutils.setup_root(__file__, pythonpath=True)
from metasim.cfg.objects import ArticulationObjCfg, BaseObjCfg, PrimitiveCubeCfg, PrimitiveSphereCfg, RigidObjCfg
from metasim.cfg.robots import BaseRobotCfg
from metasim.cfg.scenario import ScenarioCfg
from metasim.sim import BaseSimHandler, GymEnvWrapper
from metasim.types import Action, EnvState


class GenesisHandler(BaseSimHandler):
    def __init__(self, scenario: ScenarioCfg):
        super().__init__(scenario)
        self._actions_cache: list[Action] = []
        self.object_inst_dict: dict[str, RigidEntity] = {}
        self.camera_inst_dict: dict[str, Camera] = {}

    def launch(self) -> None:
        gs.init(backend=gs.gpu)  # TODO: add option for cpu
        self.scene_inst = gs.Scene(
            sim_options=gs.options.SimOptions(substeps=1),  # TODO: substeps > 1 doesn't work
            vis_options=gs.options.VisOptions(n_rendered_envs=self.scenario.num_envs),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(3.5, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            renderer=gs.renderers.Rasterizer(),
            show_viewer=not self.headless,
        )

        ## Add ground
        self.scene_inst.add_entity(gs.morphs.Plane())

        ## Add robot
        self.robot_inst: RigidEntity = self.scene_inst.add_entity(
            gs.morphs.URDF(
                file=self.robot.urdf_path,
                fixed=self.robot.fix_base_link,
                merge_fixed_links=self.robot.collapse_fixed_joints,
            ),
            material=gs.materials.Rigid(gravity_compensation=1 if not self.robot.enabled_gravity else 0),
        )
        self.object_inst_dict[self.robot.name] = self.robot_inst

        ## Add objects
        for obj in self.scenario.objects:
            if isinstance(obj.scale, tuple) or isinstance(obj.scale, list):
                obj.scale = obj.scale[0]
                log.warning(
                    f"Genesis does not support different scaling for each axis for {obj.name}, using scale={obj.scale}"
                )
            if isinstance(obj, PrimitiveCubeCfg):
                obj_inst = self.scene_inst.add_entity(
                    gs.morphs.Box(size=obj.size), surface=gs.surfaces.Default(color=obj.color)
                )
            elif isinstance(obj, PrimitiveSphereCfg):
                obj_inst = self.scene_inst.add_entity(
                    gs.morphs.Sphere(radius=obj.radius), surface=gs.surfaces.Default(color=obj.color)
                )
            elif isinstance(obj, RigidObjCfg):
                obj_inst = self.scene_inst.add_entity(
                    gs.morphs.URDF(file=obj.urdf_path, fixed=obj.fix_base_link, scale=obj.scale),
                )
            elif isinstance(obj, ArticulationObjCfg):
                obj_inst = self.scene_inst.add_entity(
                    gs.morphs.URDF(file=obj.urdf_path, fixed=obj.fix_base_link, scale=obj.scale),
                )
            else:
                raise NotImplementedError(f"Object type {type(obj)} not supported")
            self.object_inst_dict[obj.name] = obj_inst

        ## Add cameras
        for camera in self.cameras:
            camera_inst = self.scene_inst.add_camera(
                res=(camera.width, camera.height),
                pos=camera.pos,
                lookat=camera.look_at,
                fov=camera.vertical_fov,
            )
            self.camera_inst_dict[camera.name] = camera_inst

        self.scene_inst.build(n_envs=self.scenario.num_envs, env_spacing=(2, 2))

    def get_states(self, env_ids: list[int] | None = None) -> list[EnvState]:
        if env_ids is None:
            env_ids = list(range(self.num_envs))
        states = []

        states_concat = {}
        for obj in self.objects + [self.robot]:
            obj_inst = self.object_inst_dict[obj.name]
            states_concat[obj.name] = {}
            states_concat[obj.name]["pos"] = obj_inst.get_pos(envs_idx=env_ids).cpu()
            states_concat[obj.name]["rot"] = obj_inst.get_quat(envs_idx=env_ids).cpu()
            if isinstance(obj, ArticulationObjCfg):
                states_concat[obj.name]["dof_pos"] = obj_inst.get_qpos(envs_idx=env_ids).cpu()
        camera_obs = {}
        for camera_name, camera_inst in self.camera_inst_dict.items():
            rgb, _, _, _ = camera_inst.render()
            rgb = torch.from_numpy(rgb.copy())
            camera_obs[camera_name] = {"rgb": rgb}

        for env_id in env_ids:
            env_state = {"objects": {}, "robots": {}, "cameras": camera_obs}
            for obj in self.objects + [self.robot]:
                obj_inst = self.object_inst_dict[obj.name]
                obj_state = {}
                obj_state["pos"] = states_concat[obj.name]["pos"][env_id]
                obj_state["rot"] = states_concat[obj.name]["rot"][env_id]
                if isinstance(obj, ArticulationObjCfg):
                    obj_state["dof_pos"] = {
                        jn: states_concat[obj.name]["dof_pos"][env_id][jid]
                        for jid, jn in enumerate(self.get_object_joint_names(obj))
                    }
                if isinstance(obj, BaseRobotCfg):
                    ## TODO: read from simulator instead of cache
                    if self.actions_cache:
                        obj_state["dof_pos_target"] = {
                            jn: self.actions_cache[env_id]["dof_pos_target"][jn]
                            for jid, jn in enumerate(self.get_object_joint_names(obj))
                        }
                    else:
                        obj_state["dof_pos_target"] = None
                if obj.name == self.robot.name:
                    env_state["robots"][obj.name] = obj_state
                else:
                    env_state["objects"][obj.name] = obj_state
            states.append(env_state)
        return states

    def set_states(self, states: list[EnvState], env_ids: list[int] | None = None) -> None:
        if env_ids is None:
            env_ids = list(range(self.num_envs))
        states_flat = [state["objects"] | state["robots"] for state in states]
        for obj in self.objects + [self.robot]:
            obj_inst = self.object_inst_dict[obj.name]
            obj_inst.set_pos(np.array([states_flat[env_id][obj.name]["pos"] for env_id in env_ids]))
            obj_inst.set_quat(np.array([states_flat[env_id][obj.name]["rot"] for env_id in env_ids]))
            if isinstance(obj, ArticulationObjCfg):
                if obj.fix_base_link:
                    obj_inst.set_qpos(
                        np.array([
                            [states_flat[env_id][obj.name]["dof_pos"][jn] for jn in self.get_object_joint_names(obj)]
                            for env_id in env_ids
                        ]),
                        envs_idx=env_ids,
                    )
                else:
                    joint_names = self.get_object_joint_names(obj)
                    qs_idx_local = torch.arange(1, 1 + len(joint_names), dtype=torch.int32, device=gs.device).tolist()
                    obj_inst.set_qpos(
                        np.array([
                            [states_flat[env_id][obj.name]["dof_pos"][jn] for jn in joint_names] for env_id in env_ids
                        ]),
                        qs_idx_local=qs_idx_local,
                        envs_idx=env_ids,
                    )

    def set_dof_targets(self, obj_name: str, actions: list[Action]) -> None:
        self._actions_cache = actions
        position = [
            [actions[env_id]["dof_pos_target"][jn] for jn in self.get_object_joint_names(self.object_dict[obj_name])]
            for env_id in range(self.num_envs)
        ]
        if self.object_dict[obj_name].fix_base_link:
            self.robot_inst.control_dofs_position(
                position=position,
                dofs_idx_local=[j.dof_idx_local for j in self.robot_inst.joints if j.dof_idx_local is not None],
            )
        else:
            self.robot_inst.control_dofs_position(
                position=position,
                dofs_idx_local=[
                    j.dof_idx_local
                    for j in self.robot_inst.joints
                    if j.dof_idx_local is not None and j.name != self.robot_inst.base_joint.name
                ],
            )

    def get_observation(self):
        states = self.get_states()
        rgbs = [state["cameras"][self.cameras[0].name]["rgb"] for state in states]
        obs = {"rgb": torch.stack(rgbs, dim=0)}
        return obs

    def simulate(self):
        for _ in range(self.scenario.decimation):
            self.scene_inst.step()

    def close(self):
        pass

    def get_object_joint_names(self, object: BaseObjCfg) -> list[str]:
        if isinstance(object, ArticulationObjCfg):
            joints: list[RigidJoint] = self.object_inst_dict[object.name].joints
            return [
                j.name
                for j in joints
                if j.dof_idx_local is not None and j.name != self.object_inst_dict[object.name].base_joint.name
            ]
        else:
            return []

    @property
    def num_envs(self) -> int:
        return self.scene_inst.n_envs

    @property
    def actions_cache(self) -> list[Action]:
        return self._actions_cache


GenesisEnv = GymEnvWrapper(GenesisHandler)
