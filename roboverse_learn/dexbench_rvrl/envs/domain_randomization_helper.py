"""Domain Randomization Example for MetaSim."""

from __future__ import annotations

import rootutils

rootutils.setup_root(__file__, pythonpath=True)

import os
from typing import Literal

import numpy as np
import torch
import tyro
from loguru import logger as log
from rich.logging import RichHandler

from metasim.constants import PhysicStateType


from metasim.randomization import *
from metasim.randomization.camera_randomizer import (
    CameraImageRandomCfg,
    CameraIntrinsicsRandomCfg,
    CameraOrientationRandomCfg,
    CameraPositionRandomCfg,
    CameraLookAtRandomCfg,
    CameraRandomCfg,
    CameraRandomizer,
)
from metasim.randomization.light_randomizer import LightRandomCfg, LightRandomizer
from metasim.randomization.material_randomizer import MaterialRandomCfg, MaterialRandomizer, MDLMaterialCfg, PhysicalMaterialCfg, PBRMaterialCfg
from metasim.randomization.object_randomizer import ObjectRandomCfg, ObjectRandomizer, PhysicsRandomCfg, PoseRandomCfg
from metasim.randomization.presets import CameraPresets, LightPresets, MaterialPresets, ObjectPresets, ScenePresets, MaterialProperties, CameraProperties, LightProperties, MDLCollections 
from metasim.randomization.scene_randomizer import SceneGeometryCfg, SceneMaterialPoolCfg, SceneRandomCfg, SceneRandomizer
from metasim.randomization.light_randomizer import LightColorRandomCfg, LightIntensityRandomCfg, LightOrientationRandomCfg
from metasim.randomization.presets.scene_presets import SceneMaterialCollections
from metasim.utils import configclass

log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])

class DomainRandomizationHelper:
    """Helper class for domain randomization tasks."""
    def __init__(self, cfg, num_envs, lights, robots, objects, cameras, handler, env_spacing, seed, device):
        self.cfg = cfg
        self.lights = lights
        self.robots = robots
        self.objects = objects
        self.cameras = cameras
        self.handler = handler
        self.env_spacing = env_spacing
        self.seed = seed
        
        self.enable_table = cfg.get("enable_table", False)
        self.table_cfg = cfg.get("table_cfg", None)
        self.randomizer = {}
        self.randomize_cfg = cfg.get("randomize_cfg", {})
        self.material_cfg = cfg.get("material_cfg", {})
        self.light_randomize_freq = cfg.get("light_randomize_freq", 200)
        self.env_setting_randomize_freq = cfg.get("env_setting_randomize_freq", 1)
        self.env_reset_num = np.zeros(num_envs, dtype=np.int32)
        for robot in robots:
            robot_randomize_cfg = self.randomize_cfg.get(robot.name, None)
            if robot_randomize_cfg:
                self.randomizer[robot] = ObjectRandomizer(
                    ObjectPresets.combined(
                        obj_name=robot.name,
                        mass_range=robot_randomize_cfg.get("mass_range", None),
                        friction_range=robot_randomize_cfg.get("friction_range", None),
                        restitution_range=robot_randomize_cfg.get("restitution_range", None),
                        position_range=robot_randomize_cfg.get("position_range", None),
                        rotation_range=robot_randomize_cfg.get("rotation_range", None),
                        rotation_axes=robot_randomize_cfg.get("rotation_axes", (False, False, False)),
                    ),
                    seed=seed,
                    device=device,
                )
                self.randomizer[robot.name].bind_handler(handler)
            robot_material_cfg = self.material_cfg.get(robot.name, None)
            if robot_material_cfg:
                mdl_cfg = None
                pbr_cfg = None
                if robot_material_cfg.get("material_path", None):
                    mdl_cfg = MDLMaterialCfg(mdl_paths=robot_material_cfg["material_path"], enabled=True)
                else:
                    pbr_cfg = PBRMaterialCfg(
                        roughness_range=robot_material_cfg.get("roughness_range", None),
                        metallic_range=robot_material_cfg.get("metallic_range", None),
                        diffuse_color_range=robot_material_cfg.get("diffuse_color_range", None),
                        enabled=True,
                    )
                config = MaterialRandomCfg(
                    obj_name=robot.name,
                    physical=PhysicalMaterialCfg(
                        friction_range=robot_material_cfg.get("friction_range", None),
                        restitution_range=robot_material_cfg.get("restitution_range", None),
                        enabled=robot_material_cfg.get("randomize_physical", False),
                    ),
                    mdl=mdl_cfg if mdl_cfg else None,
                    pbr=pbr_cfg if pbr_cfg else None,
                    randomization_mode="combined",
                )
                material_randomizer = MaterialRandomizer(
                    config,
                    seed=seed,
                    device=device,
                )
                material_randomizer.bind_handler(handler)
                self.randomizer[f"material_{robot.name}"] = material_randomizer
        for obj in objects:
            obj_randomize_cfg = self.randomize_cfg.get(obj.name, None)
            if obj_randomize_cfg:
                self.randomizer[obj] = ObjectRandomizer(
                    ObjectPresets.combined(
                        obj_name=obj.name,
                        mass_range=obj_randomize_cfg.get("mass_range", None),
                        friction_range=obj_randomize_cfg.get("friction_range", None),
                        restitution_range=obj_randomize_cfg.get("restitution_range", None),
                        position_range=obj_randomize_cfg.get("position_range", None),
                        rotation_range=obj_randomize_cfg.get("rotation_range", None),
                        rotation_axes=obj_randomize_cfg.get("rotation_axes", (False, False, False)),
                    ),
                    seed=seed,
                    device=device,
                )
                self.randomizer[obj.name].bind_handler(handler)
            obj_material_cfg = self.material_cfg.get(obj.name, None)
            if obj_material_cfg:
                mdl_cfg = None
                pbr_cfg = None
                if obj_material_cfg.get("material_path", None):
                    mdl_cfg = MDLMaterialCfg(mdl_paths=obj_material_cfg["material_path"], enabled=True)
                else:
                    pbr_cfg = PBRMaterialCfg(
                        roughness_range=obj_material_cfg.get("roughness_range", None),
                        metallic_range=obj_material_cfg.get("metallic_range", None),
                        diffuse_color_range=obj_material_cfg.get("diffuse_color_range", None),
                        enabled=True,
                    )
                config = MaterialRandomCfg(
                    obj_name=obj.name,
                    physical=PhysicalMaterialCfg(
                        friction_range=obj_material_cfg.get("friction_range", None),
                        restitution_range=obj_material_cfg.get("restitution_range", None),
                        enabled=obj_material_cfg.get("randomize_physical", False),
                    ),
                    mdl=mdl_cfg if mdl_cfg else None,
                    pbr=pbr_cfg if pbr_cfg else None,
                    randomization_mode="combined",
                )
                material_randomizer = MaterialRandomizer(
                    config,
                    seed=seed,
                    device=device,
                )
                material_randomizer.bind_handler(handler)
                self.randomizer[f"material_{obj.name}"] = material_randomizer
                
        for light in lights:
            light_randomize_cfg = self.randomize_cfg.get(light.name, None)
            intensity_range = light_randomize_cfg.get("intensity_range", (1.0, 1.0)) if light_randomize_cfg else (1.0, 1.0)
            intensity_range = (light.intensity * intensity_range[0], light.intensity * intensity_range[1])
            if light_randomize_cfg:
                light_randomizer = LightRandomizer(
                    LightRandomCfg(
                        light_name=light.name,
                        intensity=LightIntensityRandomCfg(
                            intensity_range=intensity_range,
                            distribution="uniform",
                            enabled=True,
                        ),
                        color=LightColorRandomCfg(
                            color_range=light_randomize_cfg.get("color_range", ((1.0, 1.0), (1.0, 1.0), (1.0, 1.0))),
                        ),
                        orientation=LightOrientationRandomCfg(
                            angle_range=LightProperties.ORIENTATION_LARGE,
                            relative_to_origin=True,
                            distribution="uniform",
                            enabled=light_randomize_cfg.get("randomize_orientation", False),
                        ),
                    ),
                    seed=seed,
                    device=device,
                )
                light_randomizer.bind_handler(handler)
                self.randomizer[light.name] = light_randomizer
        
        for camera in cameras:
            camera_randomize_cfg = self.randomize_cfg.get(camera.name, None)
            if camera_randomize_cfg:
                randomization_mode = camera_randomize_cfg.get("randomization_mode", "combined")
                camera_randomizer = CameraRandomizer(
                    CameraRandomCfg(
                        camera_name=camera.name,
                        position=CameraPositionRandomCfg(
                            delta_range=camera_randomize_cfg.get("position_delta_range", ((-0.05, 0.05), (-0.05, 0.05), (-0.05, 0.05))),
                            use_delta=True,
                            enabled=camera_randomize_cfg.get("randomize_position", True),
                        ),
                        init_position=torch.tensor(camera.pos, device=device).unsqueeze(0).repeat(num_envs, 1) + self.handler.scene.env_origins,
                        look_at=CameraLookAtRandomCfg(
                            look_at_delta=camera_randomize_cfg.get("look_at_delta_range", ((-0.05, 0.05), (-0.05, 0.05), (-0.05, 0.05))),
                            use_delta=True,
                            enabled=camera_randomize_cfg.get("randomize_look_at", True),
                        ),
                        init_look_at=torch.tensor(camera.look_at, device=device).unsqueeze(0).repeat(num_envs, 1) + self.handler.scene.env_origins,
                        intrinsics=CameraIntrinsicsRandomCfg(
                            focal_length_range=camera_randomize_cfg.get("focal_length_range", None),
                            horizontal_aperture_range=camera_randomize_cfg.get("horizontal_aperture_range", None),
                            focus_distance_range=camera_randomize_cfg.get("focus_distance_range", None),
                            clipping_range=camera_randomize_cfg.get("clipping_range", None),
                            distribution="uniform",
                            enabled=camera_randomize_cfg.get("randomize_intrinsics", False),
                        ),
                        randomization_mode=randomization_mode,
                    ),
                    seed=seed,
                    device=device,
                )
                camera_randomizer.bind_handler(handler)
                self.randomizer[camera.name] = camera_randomizer

        self.scene_randomizer = None
        self.enable_floor = cfg.get("enable_floor", True)
        self.enable_walls = cfg.get("enable_walls", False)
        self.enable_ceiling = cfg.get("enable_ceiling", False)
        

        if self.enable_floor or self.enable_walls or self.enable_ceiling or self.enable_table:

            log.info("Initializing Scene Randomizer")

            floor_cfg = walls_cfg = ceiling_cfg = table_cfg = None
            floor_materials_cfg = wall_materials_cfg = ceiling_materials_cfg = table_materials_cfg = None

            room_size = self.env_spacing
            wall_height = self.cfg.get("wall_height", 5.0)
            wall_thickness = self.cfg.get("wall_thickness", 0.2)

            if self.enable_floor:
                floor_cfg = SceneGeometryCfg(
                    enabled=True,
                    size=(room_size, room_size, 0.01),
                    position=(0.0, 0.0, 0.005),
                    material_randomization=True,
                )
                floor_materials_cfg = SceneMaterialPoolCfg(
                    material_paths=self.cfg.get("floor_materials", SceneMaterialCollections.floor_materials()),
                    selection_strategy="random",
                )
                log.info(f"  - Floor enabled, {len(floor_materials_cfg.material_paths)} materials)")

            if self.enable_walls:
                walls_cfg = SceneGeometryCfg(
                    enabled=True,
                    size=(room_size, wall_thickness, wall_height),
                    position=(0.0, 0.0, wall_height / 2),
                    material_randomization=True,
                )
                wall_materials_cfg = SceneMaterialPoolCfg(
                    material_paths=self.cfg.get("wall_materials", SceneMaterialCollections.wall_materials()),
                    selection_strategy="random",
                )
                log.info(f"  - Walls enabled, {len(wall_materials_cfg.material_paths)} materials)")

            if self.enable_ceiling and self.enable_walls:
                ceiling_cfg = SceneGeometryCfg(
                    enabled=True,
                    size=(room_size, room_size, wall_thickness),
                    position=(0.0, 0.0, wall_height + wall_thickness / 2),
                    material_randomization=True,
                )
                ceiling_materials_cfg = SceneMaterialPoolCfg(
                    material_paths=self.cfg.get("ceiling_materials", SceneMaterialCollections.ceiling_materials()),
                    selection_strategy="random",
                )
                log.info("  - Ceiling enabled")
            elif self.enable_ceiling and not self.enable_walls:
                log.warning("  - Ceiling disabled (requires --enable-walls)")

            if self.enable_table and self.table_cfg:
                table_cfg_dict = self.table_cfg
                table_thickness = table_cfg_dict.get("thickness", 0.1)
                table_center_z = table_cfg_dict["height"] - table_thickness / 2

                table_cfg = SceneGeometryCfg(
                    enabled=True,
                    size=(table_cfg_dict["width"], table_cfg_dict["depth"], table_thickness),
                    position=(table_cfg_dict["x_pos"], table_cfg_dict["y_pos"], table_center_z),
                    material_randomization=True,
                )
                table_materials_cfg = SceneMaterialPoolCfg(
                    material_paths=self.cfg.get("table_materials", SceneMaterialCollections.table_materials()),
                    selection_strategy="random",
                )
                log.info(
                    f"  - Table enabled ({table_cfg_dict['width']}x{table_cfg_dict['depth']}m at z={table_cfg_dict['height']}m, ~{len(table_materials_cfg.material_paths)} materials)"
                )

            # Create scene configuration
            scene_cfg = SceneRandomCfg(
                floor=floor_cfg,
                walls=walls_cfg,
                ceiling=ceiling_cfg,
                table=table_cfg,
                floor_materials=floor_materials_cfg,
                wall_materials=wall_materials_cfg,
                ceiling_materials=ceiling_materials_cfg,
                table_materials=table_materials_cfg,
                only_if_no_scene=True,
            )

            self.scene_randomizer = SceneRandomizer(scene_cfg, seed=self.seed, device=device)
            self.scene_randomizer.bind_handler(handler)
            log.info("Scene randomizer initialized successfully")
    
    def _apply_material_to_prim(self, material_path, root_prim, env_ids):
        """Apply material to a given prim across specified environments."""
        for env_id in env_ids:
            prim_path = root_prim.replace("env_.*", f"env_{env_id}")
            try:
                SceneRandomizer._apply_material_to_prim(material_path=material_path, prim_path=prim_path)
                log.debug(f"Applied material {material_path} to {prim_path}")
            except Exception as e:
                log.warning(f"Failed to apply material {material_path} to {prim_path}: {e}")
        
    def randomization(self, env_ids, step_count=0):
        self.env_reset_num[env_ids] += 1
        self.env_reset_num[env_ids] = self.env_reset_num[env_ids] % self.env_setting_randomize_freq
        if np.any(self.env_reset_num[env_ids] == 0):
            mask = self.env_reset_num[env_ids] == 0
            env_ids_to_randomize = np.array(env_ids)[mask].tolist()
            
            # randomize_envs -> [0, 2]
            for obj in self.objects:
                if obj.name in self.randomizer:
                    self.randomizer[obj.name](env_ids_to_randomize)
                if f"material_{obj.name}" in self.randomizer:
                    self.randomizer[f"material_{obj.name}"](env_ids_to_randomize)
            for robot in self.robots:
                if robot.name in self.randomizer:
                    self.randomizer[robot.name](env_ids_to_randomize)
                if f"material_{robot.name}" in self.randomizer:
                    self.randomizer[f"material_{robot.name}"](env_ids_to_randomize)
            for camera in self.cameras:
                if camera.name in self.randomizer:
                    self.randomizer[camera.name](env_ids_to_randomize)
            if self.scene_randomizer:
                self.scene_randomizer(env_ids_to_randomize)
        if step_count % self.light_randomize_freq == 0:
            for light in self.lights:
                if light.name in self.randomizer:
                    self.randomizer[light.name]()