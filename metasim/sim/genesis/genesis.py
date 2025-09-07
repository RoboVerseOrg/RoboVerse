from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import genesis as gs
import numpy as np
import torch
from genesis.engine.entities.rigid_entity import RigidEntity, RigidJoint
from genesis.vis.camera import Camera
from loguru import logger as log

from metasim.queries.base import BaseQueryType
from metasim.scenario.objects import (
    ArticulationObjCfg,
    PrimitiveCubeCfg,
    PrimitiveSphereCfg,
    RigidObjCfg,
    _FileBasedMixin,
)
from metasim.scenario.robot import RobotCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.sim import BaseSimHandler
from metasim.types import Action, DictEnvState
from metasim.utils.state import CameraState, ObjectState, RobotState, TensorState

# Apply IGL compatibility patch
try:
    import genesis.engine.entities.rigid_entity.rigid_geom as _rigid_geom_module
    import igl as _igl

    _original_compute_sd = _rigid_geom_module.RigidGeom._compute_sd

    def _patched_compute_sd(self, query_points):
        """Patched version that handles different IGL return values"""
        result = _igl.signed_distance(query_points, self._sdf_verts, self._sdf_faces)
        if isinstance(result, tuple):
            return result[0] if len(result) > 0 else None
        return result

    _rigid_geom_module.RigidGeom._compute_sd = _patched_compute_sd
except Exception:
    pass


class GenesisHandler(BaseSimHandler):
    def __init__(self, scenario: ScenarioCfg, optional_queries: dict[str, BaseQueryType] | None = None):
        super().__init__(scenario, optional_queries)
        self._actions_cache: list[Action] = []
        self.object_inst_dict: dict[str, RigidEntity] = {}
        self.camera_inst_dict: dict[str, Camera] = {}
        self.robot = self.robots[0] if self.robots else None

    def launch(self) -> None:
        super().launch()
        gs.init(backend=gs.gpu)  # TODO: add option for cpu
        self.scene_inst = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=self.scenario.sim_params.dt if self.scenario.sim_params.dt is not None else 1 / 100,
                substeps=1,
            ),  # TODO: substeps > 1 doesn't work
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
        try:
            self.scene_inst.add_entity(gs.morphs.Plane())
        except (ValueError, Exception) as e:
            # Fallback if Plane has issues
            log.warning(f"Could not add ground plane: {e}")
            pass

        if self.robot:
            # Sanitize URDF to remove invalid empty collision nodes which
            # cause genesis' urdf parser to crash.
            robot_urdf_path = self._sanitize_urdf(self.robot.urdf_path)
            self.robot_inst: RigidEntity = self.scene_inst.add_entity(
                gs.morphs.URDF(
                    file=robot_urdf_path,
                    fixed=self.robot.fix_base_link,
                    merge_fixed_links=self.robot.collapse_fixed_joints,
                ),
                material=gs.materials.Rigid(gravity_compensation=1 if not self.robot.enabled_gravity else 0),
            )
            self.object_inst_dict[self.robot.name] = self.robot_inst

        ## Add objects
        for obj in self.scenario.objects:
            if isinstance(obj, _FileBasedMixin):
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
                urdf_path = self._sanitize_urdf(obj.urdf_path) if obj.urdf_path else None
                obj_inst = self.scene_inst.add_entity(
                    gs.morphs.URDF(file=urdf_path, fixed=obj.fix_base_link, scale=obj.scale),
                )
            elif isinstance(obj, ArticulationObjCfg):
                urdf_path = self._sanitize_urdf(obj.urdf_path) if obj.urdf_path else None
                obj_inst = self.scene_inst.add_entity(
                    gs.morphs.URDF(file=urdf_path, fixed=obj.fix_base_link, scale=obj.scale),
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

        self.scene_inst.build(
            n_envs=self.scenario.num_envs, env_spacing=(self.scenario.env_spacing, self.scenario.env_spacing)
        )

    def _get_states(self, env_ids: list[int] | None = None) -> list[DictEnvState]:
        if env_ids is None:
            env_ids = list(range(self.num_envs))

        object_states = {}
        for obj in self.objects:
            obj_inst = self.object_inst_dict[obj.name]
            if isinstance(obj, ArticulationObjCfg):
                joint_reindex = self.get_joint_reindex(obj.name)
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
                    body_names=None,
                    body_state=None,  # TODO
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
        for obj in self.robots:
            obj_inst = self.object_inst_dict[obj.name]
            joint_reindex = self.get_joint_reindex(obj.name)
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
                body_names=None,
                body_state=None,  # TODO
                joint_pos=obj_inst.get_dofs_position(envs_idx=env_ids)[:, joint_reindex],
                joint_vel=obj_inst.get_dofs_velocity(envs_idx=env_ids)[:, joint_reindex],
                joint_pos_target=None,  # TODO
                joint_vel_target=None,  # TODO
                joint_effort_target=self._get_effort_targets()
                if self._get_control_mode(obj.name) == "effort"
                else None,
            )
            robot_states[obj.name] = state

        camera_states = {}
        for camera in self.cameras:
            camera_inst = self.camera_inst_dict[camera.name]
            rgb, depth, _, _ = camera_inst.render(depth=True)

            # Ensure tensors and normalize RGB to [0, 255] for consistency
            if isinstance(rgb, np.ndarray):
                rgb_t = torch.from_numpy(rgb.copy())
            else:
                rgb_t = torch.as_tensor(rgb)
            if rgb_t.is_floating_point():
                # If already [0,1], scale to [0,255]
                try:
                    maxv = float(rgb_t.max().item())
                except Exception:
                    maxv = 1.0
                if maxv <= 1.01:
                    rgb_t = (rgb_t * 255.0).clamp(0, 255)

            if isinstance(depth, np.ndarray):
                depth_t = torch.from_numpy(depth.copy())
            else:
                depth_t = torch.as_tensor(depth)

            state = CameraState(
                rgb=rgb_t.unsqueeze(0).repeat_interleave(self.num_envs, dim=0),
                depth=depth_t.unsqueeze(0).repeat_interleave(self.num_envs, dim=0),
            )
            camera_states[camera.name] = state

        extras = self.get_extra()  # extra observations TODO: add extra observations
        return TensorState(objects=object_states, robots=robot_states, cameras=camera_states, extras=extras)

    def _set_states(self, states: list[DictEnvState], env_ids: list[int] | None = None) -> None:
        if env_ids is None:
            env_ids = list(range(self.num_envs))
        states_flat = [state["objects"] | state["robots"] for state in states]

        all_objects = self.objects + self.robots
        for obj in all_objects:
            if obj.name not in self.object_inst_dict:
                log.warning(f"Object {obj.name} not found in object_inst_dict")
                continue

            obj_inst = self.object_inst_dict[obj.name]

            positions = torch.stack(
                [torch.as_tensor(states_flat[env_id][obj.name]["pos"]) for env_id in env_ids], dim=0
            )
            rotations = torch.stack(
                [torch.as_tensor(states_flat[env_id][obj.name]["rot"]) for env_id in env_ids], dim=0
            )

            obj_inst.set_pos(positions)
            obj_inst.set_quat(rotations)

            is_articulated = isinstance(obj, (ArticulationObjCfg, RobotCfg))

            if is_articulated:
                joint_names = self._get_joint_names(obj.name, sort=False)

                if joint_names and "dof_pos" in states_flat[0][obj.name]:
                    joint_positions = []
                    for env_id in env_ids:
                        env_joint_pos = []
                        for jn in joint_names:
                            if jn in states_flat[env_id][obj.name]["dof_pos"]:
                                pos_val = states_flat[env_id][obj.name]["dof_pos"][jn]
                                env_joint_pos.append(float(pos_val))
                            else:
                                env_joint_pos.append(0.0)
                        joint_positions.append(env_joint_pos)

                    joint_pos_array = torch.tensor(joint_positions, dtype=torch.float32)

                    if obj.fix_base_link:
                        obj_inst.set_qpos(
                            joint_pos_array,
                            envs_idx=env_ids,
                        )
                    else:
                        # Use actual local q indices per joint (skips FREE root qs properly)
                        qs_idx_local: list[int] = []
                        for j in obj_inst.joints:
                            if j.name in joint_names:
                                try:
                                    qs_idx_local.extend(list(j.qs_idx_local))
                                except Exception:
                                    pass
                        obj_inst.set_qpos(
                            joint_pos_array,
                            qs_idx_local=qs_idx_local,
                            envs_idx=env_ids,
                        )

        self.scene_inst.step()

        if not self.headless and hasattr(self.scene_inst, "viewer") and self.scene_inst.viewer:
            self.scene_inst.viewer.update()

    def _set_dof_targets(self, actions: list[Action] | torch.Tensor) -> None:
        self._actions_cache = actions

        if not self.robot:
            return

        obj_name = self.robot.name
        obj_inst = self.object_inst_dict[obj_name]
        control_mode = self._get_control_mode(obj_name)
        sim_joint_names = self._get_joint_names(obj_name, sort=False)

        # Fast-path: tensor input (VectorEnv). Assume position control.
        if isinstance(actions, torch.Tensor):
            # Map tensor joint order (RobotCfg order) -> simulator joint order
            cfg_joint_order = list(self.object_dict[obj_name].joint_limits.keys())
            idxs = [cfg_joint_order.index(jn) for jn in sim_joint_names if jn in cfg_joint_order]
            if len(idxs) == 0:
                return
            # Select and shape per Genesis expectations
            if actions.dim() == 1:
                actions = actions.unsqueeze(0)
            position = actions[:, idxs]

            dofs_idx_local: list[int] = []
            for j in obj_inst.joints:
                if j.name in sim_joint_names and j.dofs_idx_local is not None:
                    dofs_idx_local.extend(j.dofs_idx_local)

            if dofs_idx_local:
                obj_inst.control_dofs_position(position=position, dofs_idx_local=dofs_idx_local)
            return

        # Dict/list input path
        joint_names = sim_joint_names

        if control_mode == "effort":
            if (
                isinstance(actions, list)
                and len(actions) > 0
                and obj_name in actions[0]
                and "dof_effort_target" in actions[0][obj_name]
            ):
                available_joints = set(actions[0][obj_name]["dof_effort_target"].keys())
                joint_names = [jn for jn in joint_names if jn in available_joints]

            effort = [
                [actions[env_id][obj_name]["dof_effort_target"][jn] for jn in joint_names]
                for env_id in range(self.num_envs)
            ]

            dofs_idx_local = []
            for j in obj_inst.joints:
                if j.dofs_idx_local is not None and j.name in joint_names:
                    dofs_idx_local.extend(j.dofs_idx_local)

            if dofs_idx_local:
                obj_inst.control_dofs_force(
                    force=effort,
                    dofs_idx_local=dofs_idx_local,
                )
        else:
            if (
                isinstance(actions, list)
                and len(actions) > 0
                and obj_name in actions[0]
                and "dof_pos_target" in actions[0][obj_name]
            ):
                available_joints = set(actions[0][obj_name]["dof_pos_target"].keys())
                joint_names = [jn for jn in joint_names if jn in available_joints]

            position = [
                [actions[env_id][obj_name]["dof_pos_target"][jn] for jn in joint_names]
                for env_id in range(self.num_envs)
            ]

            dofs_idx_local = []
            for j in obj_inst.joints:
                if j.dofs_idx_local is not None and j.name in joint_names:
                    dofs_idx_local.extend(j.dofs_idx_local)

            if dofs_idx_local:
                if self.num_envs == 1:
                    position = position[0]
                obj_inst.control_dofs_position(
                    position=position,
                    dofs_idx_local=dofs_idx_local,
                )

    def _simulate(self):
        for _ in range(self.scenario.decimation):
            self.scene_inst.step()

    def refresh_render(self):
        """Refresh the render."""
        if not self.headless and hasattr(self.scene_inst, "viewer") and self.scene_inst.viewer:
            self.scene_inst.viewer.update()

    def _get_effort_targets(self) -> torch.Tensor | None:
        """Get the effort targets from cached actions."""
        if not self._actions_cache or not self.robot:
            return None

        joint_names = self._get_joint_names(self.robot.name, sort=False)
        effort_targets = []
        for action in self._actions_cache:
            if "dof_effort_target" in action[self.robot.name] and action[self.robot.name]["dof_effort_target"]:
                effort_values = [action[self.robot.name]["dof_effort_target"][jn] for jn in joint_names]
                effort_targets.append(effort_values)

        if effort_targets:
            return torch.tensor(effort_targets, dtype=torch.float32)
        return None

    def _get_control_mode(self, obj_name: str) -> str:
        """Get the control mode for the object."""
        if hasattr(self.object_dict[obj_name], "control_type"):
            control_types = list(set(self.object_dict[obj_name].control_type.values()))
            if len(control_types) > 1:
                raise ValueError(f"Multiple control types not supported: {control_types}")
            return control_types[0] if control_types else "position"
        return "position"

    def _get_joint_names(self, obj_name: str, sort: bool = True) -> list[str]:
        obj_cfg = self.object_dict[obj_name]
        if isinstance(obj_cfg, (ArticulationObjCfg, RobotCfg)):
            joints: list[RigidJoint] = self.object_inst_dict[obj_name].joints
            # Exclude FREE and FIXED joints (e.g., floating base), and only include joints with DoFs
            joint_names = []
            for j in joints:
                try:
                    if j.type in (gs.JOINT_TYPE.FREE, gs.JOINT_TYPE.FIXED):
                        continue
                except Exception:
                    pass
                try:
                    dofs = getattr(j, "dofs_idx_local", None)
                    if dofs is None or len(dofs) == 0:
                        continue
                except Exception:
                    continue
                joint_names.append(j.name)

            if sort:
                joint_names.sort()

            return joint_names
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

    def close(self) -> None:
        """Close the simulation and clean up resources."""
        self.scene_inst = None
        self.object_inst_dict = {}
        self.camera_inst_dict = {}

    def _sanitize_urdf(self, urdf_path: str | None) -> str | None:
        """
        removing empty <collision> nodes.
        """
        if urdf_path is None:
            return None

        try:
            src = Path(urdf_path)
            if not src.exists():
                return urdf_path
            tree = ET.parse(src)
            root = tree.getroot()
            changed = False

            for link in root.findall(".//link"):
                to_remove = []
                for coll in link.findall("collision"):
                    if coll.find("geometry") is None:
                        to_remove.append(coll)
                for coll in to_remove:
                    link.remove(coll)
                    changed = True

            if not changed:
                return urdf_path

            dst = src.with_suffix(".genesis.urdf")
            tree.write(dst, encoding="utf-8", xml_declaration=True)
            return str(dst)
        except Exception as e:
            log.warning(f"URDF sanitize failed for {urdf_path}: {e}. Using original file.")
            return urdf_path
