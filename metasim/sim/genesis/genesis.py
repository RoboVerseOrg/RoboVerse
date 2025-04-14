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
from metasim.cfg.scenario import ScenarioCfg
from metasim.sim import BaseSimHandler, GymEnvWrapper
from metasim.types import Action, EnvState
from metasim.utils.state import CameraState, ObjectState, RobotState, TensorState


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

        object_states = {}
        for obj in self.objects:
            obj_inst = self.object_inst_dict[obj.name]
            if isinstance(obj, ArticulationObjCfg):
                joint_reindex = self.get_object_joint_reindex(obj.name)
                state = ObjectState(
                    root_state=torch.cat(
                        [
                            obj_inst.get_pos(envs_idx=env_ids),
                            obj_inst.get_quat(envs_idx=env_ids),
                            obj_inst.get_vel(envs_idx=env_ids),
                            obj_inst.get_ang(envs_idx=env_ids),
                        ],
                        dim=-1,
                    ),
                    joint_pos=obj_inst.get_dofs_position(envs_idx=env_ids)[:, joint_reindex],
                    joint_vel=obj_inst.get_dofs_velocity(envs_idx=env_ids)[:, joint_reindex],
                )
            else:
                state = ObjectState(
                    root_state=torch.cat(
                        [
                            obj_inst.get_pos(envs_idx=env_ids),
                            obj_inst.get_quat(envs_idx=env_ids),
                            obj_inst.get_vel(envs_idx=env_ids),
                            obj_inst.get_ang(envs_idx=env_ids),
                        ],
                        dim=-1,
                    ),
                )
            object_states[obj.name] = state

        robot_states = {}
        for obj in [self.robot]:
            obj_inst = self.object_inst_dict[obj.name]
            joint_reindex = self.get_object_joint_reindex(obj.name)
            state = RobotState(
                root_state=torch.cat(
                    [
                        obj_inst.get_pos(envs_idx=env_ids),
                        obj_inst.get_quat(envs_idx=env_ids),
                        obj_inst.get_vel(envs_idx=env_ids),
                        obj_inst.get_ang(envs_idx=env_ids),
                    ],
                    dim=-1,
                ),
                joint_pos=obj_inst.get_dofs_position(envs_idx=env_ids)[:, joint_reindex],
                joint_vel=obj_inst.get_dofs_velocity(envs_idx=env_ids)[:, joint_reindex],
                joint_pos_target=None,  # TODO
                joint_vel_target=None,  # TODO
                joint_effort_target=None,  # TODO
            )
            robot_states[obj.name] = state

        camera_states = {}
        for camera in self.cameras:
            camera_inst = self.camera_inst_dict[camera.name]
            rgb, depth, _, _ = camera_inst.render(depth=True)
            state = CameraState(
                rgb=torch.from_numpy(rgb.copy()).unsqueeze(0).repeat_interleave(self.num_envs, dim=0),  # XXX
                depth=torch.from_numpy(depth.copy()).unsqueeze(0).repeat_interleave(self.num_envs, dim=0),  # XXX
            )
            camera_states[camera.name] = state

        return TensorState(objects=object_states, robots=robot_states, cameras=camera_states)

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

    def simulate(self):
        for _ in range(self.scenario.decimation):
            self.scene_inst.step()

    def refresh_render(self):
        """Refresh the render."""
        if not self.headless:
            self.scene_inst.viewer.update()
        self.scene_inst.visualizer.update()

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

    @property
    def device(self) -> torch.device:
        return gs.device


GenesisEnv = GymEnvWrapper(GenesisHandler)
