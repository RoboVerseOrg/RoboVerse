"""
IsaacSim Domain Randomizer Implementation.

This module provides comprehensive domain randomization for IsaacSim including:
- Camera pose randomization 
- Lighting randomization (5 light types)
- Material and reflection randomization
- Clutter object placement
- Surface material randomization (table/ground/wall)

Uses delayed imports to avoid IsaacSim runtime initialization issues.
All IsaacSim-specific modules are imported only when needed at runtime.
"""

import copy
import os
import random
from typing import Iterator, Literal, Dict, Any

import numpy as np
import yaml
from loguru import logger as log


class IsaacSimDomainRandomizer:
    """
    Domain randomizer for IsaacSim with delayed imports.
    
    Uses runtime imports to avoid initialization issues with IsaacSim modules.
    """
    
    def __init__(self, sim_handler, randomization_cfg):
        """
        Initialize the domain randomizer.
        
        Args:
            sim_handler: The IsaacSim simulation handler
            randomization_cfg: RandomizationCfg object with flags for each randomization type
        """
        self.sim_handler = sim_handler
        self.cfg = randomization_cfg
        self.device = sim_handler.device if hasattr(sim_handler, 'device') else 'cpu'
        
        # Load camera position candidates
        self._load_camera_candidates()
        
        log.info(f"IsaacSim Domain Randomizer initialized with level {self.cfg.level}")
    
    def _check_isaac_availability(self):
        """Check if IsaacSim modules are available."""
        try:
            import torch
            import omni
            return True
        except ImportError:
            return False
    
    def _get_isaac_modules(self):
        """Get IsaacSim modules with runtime import."""
        modules = {}
        
        try:
            import torch
            import omni
            modules['torch'] = torch
            modules['omni'] = omni
            
            # Import USD/PXR
            try:
                from pxr import Gf, Sdf, Usd, UsdShade, UsdGeom
                modules.update({'Gf': Gf, 'Sdf': Sdf, 'Usd': Usd, 'UsdShade': UsdShade, 'UsdGeom': UsdGeom})
                modules['usd_available'] = True
            except ImportError:
                modules['usd_available'] = False
                log.debug("USD/PXR not available")
            
            # Import material library
            try:
                from omni.kit.material.library import get_material_prim_path
                modules['get_material_prim_path'] = get_material_prim_path
            except ImportError:
                def get_material_prim_path(name):
                    return False, f"/World/Looks/{name}"
                modules['get_material_prim_path'] = get_material_prim_path
                log.debug("Material library not available, using fallback")
            
            # Import prim utils
            try:
                import omni.isaac.core.utils.prims as prim_utils
                modules['prim_utils'] = prim_utils
            except ImportError:
                try:
                    import isaacsim.core.utils.prims as prim_utils
                    modules['prim_utils'] = prim_utils
                except ImportError:
                    modules['prim_utils'] = None
                    log.debug("Prim utils not available")
            
            return modules
            
        except ImportError as e:
            log.warning(f"Failed to import IsaacSim modules: {e}")
            return {}
    
    def _load_camera_candidates(self):
        """Load camera position candidates for camera randomization."""
        try:
            candidates_path = "metasim/sim/isaacsim/cfg/randomization/camera_pos_candidates.txt"
            if os.path.exists(candidates_path):
                self._phi_theta_candidates = np.loadtxt(candidates_path)
                log.debug(f"Loaded {len(self._phi_theta_candidates)} camera position candidates")
            else:
                # Create default candidates if file doesn't exist
                self._phi_theta_candidates = np.array([
                    [60.0/180*np.pi, 0.0],  # front
                    [45.0/180*np.pi, np.pi/4],  # front-right
                    [45.0/180*np.pi, -np.pi/4], # front-left
                    [30.0/180*np.pi, np.pi/2],  # right
                    [30.0/180*np.pi, -np.pi/2], # left
                ])
                log.warning(f"Camera candidates file not found, using defaults")
        except Exception as e:
            log.warning(f"Failed to load camera candidates: {e}")
            self._phi_theta_candidates = np.array([[60.0/180*np.pi, 0.0]])
    
    def reset_randomization(self):
        """Reset randomization state for a new episode."""
        log.debug("Domain randomization reset for new episode")
    
    def apply_scene_randomization(self):
        """Apply scene-level randomizations that should happen BEFORE physics initialization."""
        if not self._check_isaac_availability():
            log.warning("IsaacSim not available, skipping scene randomization")
            return

        if not hasattr(self.sim_handler, 'scene') or self.sim_handler.scene is None:
            log.warning("Scene not available, skipping scene randomization")
            return
            
        log.info("Applying scene-level domain randomization...")

        try:
            # Apply only scene-level randomizations (materials)
            # These should not affect physics simulation state
            if self.cfg.table:
                self._randomize_table_material()
            if self.cfg.ground:
                self._randomize_ground_material()
            if self.cfg.wall:
                self._randomize_wall_material()
            # Apply scene clutter objects if configured
            if self.cfg.scene:
                self._randomize_scene()

            log.info("Scene-level domain randomization applied successfully")

        except Exception as e:
            log.warning(f"Scene-level domain randomization failed: {e}")
            
    def apply_visual_randomization(self):
        """Apply visual-only randomizations that should happen AFTER physics initialization."""
        if not self._check_isaac_availability():
            log.warning("IsaacSim not available, skipping visual randomization")
            return

        if not hasattr(self.sim_handler, 'scene') or self.sim_handler.scene is None:
            log.warning("Scene not available, skipping visual randomization")
            return
            
        log.info("Applying visual domain randomization...")

        try:
            # Apply only visual randomizations (camera, lights, reflection)
            # These should not affect physics simulation state
            if self.cfg.camera:
                self._randomize_camera_poses()
            if self.cfg.light:
                self._randomize_lighting()
            if self.cfg.reflection:
                self._randomize_reflection()

            log.info("Visual domain randomization applied successfully")

        except Exception as e:
            log.warning(f"Visual domain randomization failed: {e}")
    
    def apply_dynamic_lighting(self):
        """Apply per-frame dynamic lighting changes."""
        if not self.cfg.light:
            return
            
        modules = self._get_isaac_modules()
        if not modules or not modules.get('usd_available', False):
            return
            
        try:
            omni = modules['omni']
            Gf = modules['Gf']
            
            stage = omni.usd.get_context().get_stage()
            if not stage:
                return
                
            # Find and randomly adjust existing lights
            light_paths = ["/World/defaultLight", "/World/Light", "/World/light_0", "/World/light_1"]
            
            for light_path in light_paths:
                light_prim = stage.GetPrimAtPath(light_path)
                if light_prim and light_prim.IsValid():
                    # Randomly adjust color (subtle changes)
                    color_variation = np.random.uniform(0.9, 1.1, 3)
                    color_attr = light_prim.GetAttribute("color3f")
                    if color_attr:
                        color_attr.Set(Gf.Vec3f(*color_variation))
                        
                    # Randomly adjust intensity (subtle changes)
                    intensity_variation = np.random.uniform(0.8, 1.2)
                    intensity_attr = light_prim.GetAttribute("intensity")
                    if intensity_attr:
                        current_intensity = intensity_attr.Get()
                        if current_intensity:
                            intensity_attr.Set(current_intensity * intensity_variation)
                        
        except Exception as e:
            log.debug(f"Dynamic lighting adjustment failed: {e}")
    
    def _randomize_camera_poses(self):
        """Randomize camera poses using IsaacLab method."""
        log.debug("Randomizing camera poses...")
        
        modules = self._get_isaac_modules()
        if not modules:
            log.warning("IsaacSim modules not available for camera randomization")
            return
        
        torch = modules['torch']
        
        try:
            for camera in self.sim_handler._cameras:
                if camera.mount_to is not None:
                    continue  # Skip mounted cameras
                    
                # Use IsaacLab-style camera randomization
                randomized_camera = self._randomize_single_camera(camera)
                
                # Update camera configuration
                camera.pos = randomized_camera.pos
                camera.look_at = randomized_camera.look_at
                camera.focus_distance = getattr(randomized_camera, 'focus_distance', 2.0)
                
                # Update actual camera in scene if available
                if hasattr(self.sim_handler, 'scene') and camera.name in self.sim_handler.scene.sensors:
                    try:
                        camera_inst = self.sim_handler.scene.sensors[camera.name]
                        eyes = torch.tensor(camera.pos, dtype=torch.float32, device=self.device)[None, :]
                        targets = torch.tensor(camera.look_at, dtype=torch.float32, device=self.device)[None, :]
                        
                        # Add environment origins offset if available
                        if hasattr(self.sim_handler.scene, 'env_origins'):
                            eyes = eyes + self.sim_handler.scene.env_origins
                            targets = targets + self.sim_handler.scene.env_origins
                            
                        camera_inst.set_world_poses_from_view(eyes=eyes, targets=targets)
                        log.debug(f"Updated camera {camera.name}: pos={camera.pos}, look_at={camera.look_at}")
                        
                    except Exception as e:
                        log.warning(f"Failed to update camera {camera.name} in scene: {e}")
                        
        except Exception as e:
            log.warning(f"Camera randomization failed: {e}")
    
    def _randomize_single_camera(self, original_camera):
        """Randomize a single camera using IsaacLab patterns."""
        randomized_camera = copy.deepcopy(original_camera)
        
        # Use different randomization modes (from IsaacLab)
        mode = np.random.choice(["semisphere", "front", "front_uniform_random", "front_select"])
        
        # Get object position for look_at target (approximate table center)
        obj_pos = (0.0, 0.0, 0.8)  # Default table center
        if hasattr(self.sim_handler, 'objects') and self.sim_handler.objects:
            try:
                # Get first object position as reference
                obj_pose, _ = self.sim_handler._get_pose(self.sim_handler.objects[0].name)
                obj_pos = obj_pose[0].cpu().numpy()
            except:
                pass
        
        # Robot orientation (assume identity for simplicity)
        robot_quat = (1.0, 0.0, 0.0, 0.0)
        
        if mode == "semisphere":
            distance = np.random.uniform(1.5, 3.0)  # Closer than IsaacLab for tabletop
            theta_to_robot = np.random.uniform(0, 2 * np.pi)
            phi = np.random.uniform(0, 70 / 180 * np.pi)
            look_at_offset = np.random.uniform([-0.1, -0.1, 0], [0.1, 0.1, 0])
        elif mode == "front":
            distance = 1.5
            theta_to_robot = 0.0
            phi = 45.0 / 180.0 * np.pi  # Lower angle for table view
            look_at_offset = np.array([0.0, 0.0, 0.0])
        elif mode == "front_uniform_random":
            distance = np.random.uniform(1.2, 2.0)
            theta_to_robot = np.random.normal(0, np.pi / 6)  # Smaller variation
            phi = np.random.uniform(30, 60) / 180.0 * np.pi  # Better angles for table
            look_at_offset = np.array([0.0, 0.0, 0.0])
        elif mode == "front_select":
            distance = 1.5
            phi, theta_to_robot = self._phi_theta_candidates[
                np.random.randint(0, self._phi_theta_candidates.shape[0])
            ]
            look_at_offset = np.array([0.0, 0.0, 0.0])
        
        # Calculate robot orientation (simplified)
        robot_theta = np.arctan2(robot_quat[2], robot_quat[3]) * 2  # Approximate
        theta = robot_theta + theta_to_robot
        
        # Calculate camera position
        pos = (np.array([
            np.cos(theta) * np.sin(phi), 
            np.sin(theta) * np.sin(phi), 
            np.cos(phi)
        ]) * distance).tolist()
        
        # Calculate look_at position
        look_at = (np.array(obj_pos) + look_at_offset).tolist()
        
        randomized_camera.pos = pos
        randomized_camera.look_at = look_at
        randomized_camera.focus_distance = distance
        
        return randomized_camera
    
    def _randomize_lighting(self):
        """Randomize lighting using IsaacSim's 5 predefined light types."""
        log.debug("Randomizing lighting...")
        
        modules = self._get_isaac_modules()
        if not modules or not modules.get('usd_available', False):
            log.debug("USD not available, skipping lighting randomization")
            return
        
        try:
            omni = modules['omni']
            stage = omni.usd.get_context().get_stage()
            if not stage:
                return
                
            # Clear existing lights first (optional - for complete randomization)
            if np.random.random() < 0.3:  # 30% chance to clear and recreate
                self._clear_existing_lights(stage)
                
            # Create a randomized lighting setup using IsaacSim's 5 light types:
            # 1. DistantLight (directional, like sun)
            # 2. DomeLight (environment lighting)  
            # 3. CylinderLight (tube lighting)
            # 4. SphereLight (point lighting)
            # 5. DiskLight (area lighting)
            
            self._create_comprehensive_lighting_setup(stage, modules)
                    
        except Exception as e:
            log.warning(f"Lighting randomization failed: {e}")
            
    def _clear_existing_lights(self, stage):
        """Clear existing lights for complete randomization."""
        try:
            light_types = ["DistantLight", "DomeLight", "CylinderLight", "SphereLight", "DiskLight"]
            for prim in stage.Traverse():
                if prim.GetTypeName() in light_types:
                    stage.RemovePrim(prim.GetPath())
        except Exception as e:
            log.debug(f"Failed to clear existing lights: {e}")
            
    def _create_comprehensive_lighting_setup(self, stage, modules):
        """Create a reasonable lighting setup (simplified to avoid too many lights)."""
        try:
            # Strategy: Start with basic setup, then randomly enhance
            lighting_strategy = np.random.choice([
                "simple",      # Basic dome + distant light
                "enhanced",    # Basic + one accent light
                "complex"      # Multiple lights (less common)
            ], p=[0.5, 0.3, 0.2])  # Prefer simpler setups
            
            if lighting_strategy == "simple":
                # Basic setup: dome light + one directional light
                self._create_dome_light(stage, modules)
                self._create_distant_light(stage, modules, 0)
                
            elif lighting_strategy == "enhanced":
                # Enhanced setup: basic + one accent light
                self._create_dome_light(stage, modules) 
                self._create_distant_light(stage, modules, 0)
                
                # Add ONE accent light (randomly choose type)
                accent_type = np.random.choice(["disk", "sphere"])
                if accent_type == "disk":
                    self._create_disk_light(stage, modules)
                else:
                    self._create_sphere_light(stage, modules, 0)
                    
            else:  # complex
                # Complex setup: multiple lights (but still reasonable)
                self._create_dome_light(stage, modules)
                self._create_distant_light(stage, modules, 0)
                
                # Maybe add a second distant light from different angle
                if np.random.random() < 0.6:
                    self._create_distant_light(stage, modules, 1)
                    
                # Add area light for task area
                if np.random.random() < 0.7:
                    self._create_disk_light(stage, modules)
                    
                # Rarely add cylinder light for ambient
                if np.random.random() < 0.3:
                    self._create_cylinder_light(stage, modules)
                    
            log.debug(f"Created lighting setup: {lighting_strategy}")
                
        except Exception as e:
            log.warning(f"Failed to create lighting setup: {e}")
            
    def _create_dome_light(self, stage, modules):
        """Create randomized dome light for environment lighting."""
        try:
            Gf, Sdf = modules.get('Gf'), modules.get('Sdf')
            if not all([Gf, Sdf]):
                return
                
            dome_path = "/World/DomeLight"
            dome_light = stage.DefinePrim(dome_path, "DomeLight")
            
            # Random intensity (0.5 to 3.0)
            intensity = np.random.uniform(0.5, 3.0)
            dome_light.CreateAttribute("intensity", Sdf.ValueTypeNames.Float).Set(intensity)
            
            # Random color temperature (3000K to 8000K)
            color_temp = np.random.uniform(3000, 8000)
            dome_light.CreateAttribute("colorTemperature", Sdf.ValueTypeNames.Float).Set(color_temp)
            
            # Random exposure (-1 to 2)
            exposure = np.random.uniform(-1.0, 2.0)
            dome_light.CreateAttribute("exposure", Sdf.ValueTypeNames.Float).Set(exposure)
            
            log.debug(f"Created dome light: intensity={intensity:.2f}, temp={color_temp:.0f}K")
            
        except Exception as e:
            log.warning(f"Failed to create dome light: {e}")
            
    def _create_distant_light(self, stage, modules, index):
        """Create randomized distant light (directional)."""
        try:
            Gf, Sdf = modules.get('Gf'), modules.get('Sdf')
            if not all([Gf, Sdf]):
                return
                
            distant_path = f"/World/DistantLight_{index}"
            distant_light = stage.DefinePrim(distant_path, "DistantLight")
            
            # Random intensity (0.5 to 4.0)
            intensity = np.random.uniform(0.5, 4.0)
            distant_light.CreateAttribute("intensity", Sdf.ValueTypeNames.Float).Set(intensity)
            
            # Random color (warm to cool)
            color = np.random.uniform(0.8, 1.0, 3)
            distant_light.CreateAttribute("color3f", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
            
            # Random angle (soft to hard shadows)
            angle = np.random.uniform(0.5, 15.0)
            distant_light.CreateAttribute("angle", Sdf.ValueTypeNames.Float).Set(angle)
            
            # Random direction (elevation and azimuth)
            elevation = np.random.uniform(20, 80)  # degrees
            azimuth = np.random.uniform(0, 360)    # degrees
            
            # Convert to rotation
            from math import radians, cos, sin
            elev_rad = radians(elevation)
            azim_rad = radians(azimuth)
            
            # Calculate direction vector (pointing towards the scene)
            x = sin(elev_rad) * cos(azim_rad)
            y = sin(elev_rad) * sin(azim_rad) 
            z = -cos(elev_rad)  # Negative Z to point down
            
            # Set rotation to point in calculated direction
            # This is a simplified rotation - you might need more complex quaternion math
            xform = distant_light.CreateAttribute("xformOp:rotateXYZ", Sdf.ValueTypeNames.Float3)
            xform.Set((elevation, azimuth, 0))
            
            log.debug(f"Created distant light {index}: intensity={intensity:.2f}, angle={angle:.1f}°")
            
        except Exception as e:
            log.warning(f"Failed to create distant light {index}: {e}")
            
    def _create_disk_light(self, stage, modules):
        """Create randomized disk light for area lighting."""
        try:
            Gf, Sdf = modules.get('Gf'), modules.get('Sdf')
            if not all([Gf, Sdf]):
                return
                
            disk_path = "/World/DiskLight"
            disk_light = stage.DefinePrim(disk_path, "DiskLight")
            
            # Random intensity (1.0 to 10.0)
            intensity = np.random.uniform(1.0, 10.0)
            disk_light.CreateAttribute("intensity", Sdf.ValueTypeNames.Float).Set(intensity)
            
            # Random radius (0.3 to 1.5 meters)
            radius = np.random.uniform(0.3, 1.5)
            disk_light.CreateAttribute("radius", Sdf.ValueTypeNames.Float).Set(radius)
            
            # Random color
            color = np.random.uniform(0.8, 1.0, 3)
            disk_light.CreateAttribute("color3f", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
            
            # Position above the table/scene
            pos_x = np.random.uniform(-1.0, 1.0)
            pos_y = np.random.uniform(-1.0, 1.0)
            pos_z = np.random.uniform(1.5, 3.0)  # Above the scene
            
            translate = disk_light.CreateAttribute("xformOp:translate", Sdf.ValueTypeNames.Double3)
            translate.Set((pos_x, pos_y, pos_z))
            
            # Point downward
            rotate = disk_light.CreateAttribute("xformOp:rotateXYZ", Sdf.ValueTypeNames.Float3)
            rotate.Set((90, 0, 0))  # Point down
            
            log.debug(f"Created disk light: intensity={intensity:.2f}, radius={radius:.2f}m")
            
        except Exception as e:
            log.warning(f"Failed to create disk light: {e}")
            
    def _create_sphere_light(self, stage, modules, index):
        """Create randomized sphere light for point lighting."""
        try:
            Gf, Sdf = modules.get('Gf'), modules.get('Sdf')
            if not all([Gf, Sdf]):
                return
                
            sphere_path = f"/World/SphereLight_{index}"
            sphere_light = stage.DefinePrim(sphere_path, "SphereLight")
            
            # Random intensity (0.5 to 8.0)
            intensity = np.random.uniform(0.5, 8.0)
            sphere_light.CreateAttribute("intensity", Sdf.ValueTypeNames.Float).Set(intensity)
            
            # Random radius (0.05 to 0.3 meters)
            radius = np.random.uniform(0.05, 0.3)
            sphere_light.CreateAttribute("radius", Sdf.ValueTypeNames.Float).Set(radius)
            
            # Random color (can be more varied for accent lighting)
            color = np.random.uniform(0.7, 1.0, 3)
            sphere_light.CreateAttribute("color3f", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
            
            # Random position around the scene
            pos_x = np.random.uniform(-2.0, 2.0)
            pos_y = np.random.uniform(-2.0, 2.0)
            pos_z = np.random.uniform(0.5, 2.5)
            
            translate = sphere_light.CreateAttribute("xformOp:translate", Sdf.ValueTypeNames.Double3)
            translate.Set((pos_x, pos_y, pos_z))
            
            log.debug(f"Created sphere light {index}: intensity={intensity:.2f}, radius={radius:.3f}m")
            
        except Exception as e:
            log.warning(f"Failed to create sphere light {index}: {e}")
            
    def _create_cylinder_light(self, stage, modules):
        """Create randomized cylinder light for tube lighting."""
        try:
            Gf, Sdf = modules.get('Gf'), modules.get('Sdf')
            if not all([Gf, Sdf]):
                return
                
            cylinder_path = "/World/CylinderLight"
            cylinder_light = stage.DefinePrim(cylinder_path, "CylinderLight")
            
            # Random intensity (1.0 to 6.0)
            intensity = np.random.uniform(1.0, 6.0)
            cylinder_light.CreateAttribute("intensity", Sdf.ValueTypeNames.Float).Set(intensity)
            
            # Random length (0.5 to 2.0 meters)
            length = np.random.uniform(0.5, 2.0)
            cylinder_light.CreateAttribute("length", Sdf.ValueTypeNames.Float).Set(length)
            
            # Random radius (0.02 to 0.1 meters)
            radius = np.random.uniform(0.02, 0.1)
            cylinder_light.CreateAttribute("radius", Sdf.ValueTypeNames.Float).Set(radius)
            
            # Random color
            color = np.random.uniform(0.8, 1.0, 3)
            cylinder_light.CreateAttribute("color3f", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
            
            # Random position and orientation
            pos_x = np.random.uniform(-1.5, 1.5)
            pos_y = np.random.uniform(-1.5, 1.5) 
            pos_z = np.random.uniform(1.0, 2.5)
            
            translate = cylinder_light.CreateAttribute("xformOp:translate", Sdf.ValueTypeNames.Double3)
            translate.Set((pos_x, pos_y, pos_z))
            
            # Random orientation
            rot_x = np.random.uniform(0, 360)
            rot_y = np.random.uniform(0, 360)
            rot_z = np.random.uniform(0, 360)
            
            rotate = cylinder_light.CreateAttribute("xformOp:rotateXYZ", Sdf.ValueTypeNames.Float3)
            rotate.Set((rot_x, rot_y, rot_z))
            
            log.debug(f"Created cylinder light: intensity={intensity:.2f}, length={length:.2f}m")
            
        except Exception as e:
            log.warning(f"Failed to create cylinder light: {e}")
    
    def _randomize_reflection(self):
        """Randomize material reflection properties, based on IsaacLab's ReflectionRandomizer."""
        log.debug("Randomizing material reflection properties...")
        
        modules = self._get_isaac_modules()
        if not modules or not modules.get('usd_available', False):
            log.debug("USD not available, skipping reflection randomization")
            return
            
        try:
            # Use IsaacLab's approach: randomize material properties for all scene objects
            self._randomize_scene_material_properties(modules)
            
            log.debug("Applied material reflection randomization")
            
        except Exception as e:
            log.warning(f"Material reflection randomization failed: {e}")
            
    def _randomize_scene_material_properties(self, modules):
        """Randomize material properties for all scene objects (based on IsaacLab)."""
        try:
            stage = modules['omni'].usd.get_context().get_stage()
            UsdShade = modules.get('UsdShade')
            Gf = modules.get('Gf')
            Sdf = modules.get('Sdf')
            prim_utils = modules.get('prim_utils')
            
            if not all([stage, UsdShade, Gf, Sdf]):
                return
                
            # Apply reflection randomization to specific ground geometry only
            # Focus on the actual visual ground mesh, not all the metadata
            target_candidates = [
                "/World/ground/terrain/Environment/Geometry",  # Main ground geometry 
                "/World/ground/terrain/GroundPlane",           # Backup ground plane
            ]
            
            processed_count = 0
            for target_path in target_candidates:
                target_prim = stage.GetPrimAtPath(target_path)
                if target_prim and target_prim.IsValid():
                    log.info(f"Applying reflection to: {target_path}")
                    self._randomize_single_prim_material(target_prim, modules)
                    processed_count += 1
                    break  # Only process the first valid one
            
            if processed_count == 0:
                log.debug("No ground geometry found for reflection randomization")
            else:
                log.info(f"Reflection applied to {processed_count} ground object")
                    
        except Exception as e:
            log.warning(f"Failed to randomize scene material properties: {e}")
    
    def _randomize_single_prim_material(self, prim, modules):
        """Randomize material properties for a single prim (based on IsaacLab approach)."""
        try:
            UsdShade = modules.get('UsdShade')
            Gf = modules.get('Gf')
            Sdf = modules.get('Sdf')
            stage = modules['omni'].usd.get_context().get_stage()
            
            if not all([UsdShade, Gf, Sdf, stage]):
                return
                
            # Get or create material for this prim
            material = UsdShade.MaterialBindingAPI(prim).GetDirectBinding().GetMaterial()
            if not material:
                # Create a new material if none exists
                import random
                mtl_name = f"material_{random.randint(0, 1000000)}"
                try:
                    from omni.kit.material.library import get_material_prim_path
                    _, mtl_prim_path = get_material_prim_path(mtl_name)
                except ImportError:
                    mtl_prim_path = f"/World/Looks/{mtl_name}"
                    
                material = UsdShade.Material.Define(stage, mtl_prim_path)
                
                # Bind material to prim
                try:
                    import omni.kit.commands
                    omni.kit.commands.execute("BindMaterial", 
                                            prim_path=str(prim.GetPath()), 
                                            material_path=mtl_prim_path)
                except Exception:
                    # Fallback binding
                    UsdShade.MaterialBindingAPI(prim).Bind(material)
                    
            # Get or create shader
            try:
                import omni.usd
                shader = UsdShade.Shader(omni.usd.get_shader_from_material(material, get_prim=True))
            except Exception:
                shader = None
                
            if not shader:
                # Create shader (UsdPreviewSurface)
                shader_path = material.GetPrim().GetPath().AppendChild("Shader")
                shader = UsdShade.Shader.Define(stage, shader_path)
                shader.CreateIdAttr("UsdPreviewSurface")
                
                # Connect shader to material surface
                material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
                
            # Now randomize material properties (based on IsaacLab's logic)
            try:
                impl_source = shader.GetImplementationSource()
            except Exception:
                impl_source = UsdShade.Tokens.id  # Default to UsdPreviewSurface
                
            if impl_source == UsdShade.Tokens.sourceAsset:
                # MDL material - randomize MDL-specific properties
                log.debug(f"Applying MDL reflection properties to {prim.GetPath()}")
                self._randomize_mdl_material_properties(shader, modules)
            else:
                # UsdPreviewSurface - randomize standard PBR properties
                log.debug(f"Applying PBR reflection properties to {prim.GetPath()}")
                self._randomize_preview_surface_properties(shader, modules)
                
        except Exception as e:
            log.debug(f"Failed to randomize material for prim {prim.GetPath()}: {e}")
            
    def _randomize_mdl_material_properties(self, shader, modules):
        """Randomize MDL material properties."""
        try:
            Sdf = modules.get('Sdf')
            if not Sdf:
                return
                
            # MDL-specific properties (using correct parameter names)
            # Try common MDL parameter names that actually exist
            mdl_properties = {
                "reflection_roughness": np.random.uniform(0.0, 0.6),     # More common name
                "metallic": np.random.uniform(0.0, 0.4),                 # Standard name  
                "specular_reflection": np.random.uniform(0.4, 1.0),     # Alternative name
                "roughness": np.random.uniform(0.0, 0.6),               # Fallback
                "diffuse_roughness": np.random.uniform(0.0, 0.6),       # Alternative
            }
            
            for prop_name, value in mdl_properties.items():
                prop_input = shader.CreateInput(prop_name, Sdf.ValueTypeNames.Float)
                prop_input.Set(float(value))
                
        except Exception as e:
            log.debug(f"Failed to randomize MDL properties: {e}")
            
    def _randomize_preview_surface_properties(self, shader, modules):
        """Randomize UsdPreviewSurface properties."""
        try:
            Gf = modules.get('Gf')
            Sdf = modules.get('Sdf')
            if not all([Gf, Sdf]):
                return
                
            # UsdPreviewSurface properties (adjusted for better reflections)
            pbr_properties = {
                "roughness": np.random.uniform(0.0, 0.7),  # Keep some surfaces smooth for reflection
                "metallic": np.random.uniform(0.0, 0.5),   # Moderate metallic values
                "specular": np.random.uniform(0.3, 1.0)    # Ensure some specular reflection
            }
            
            for prop_name, value in pbr_properties.items():
                prop_input = shader.CreateInput(prop_name, Sdf.ValueTypeNames.Float)
                prop_input.Set(float(value))
                # log.debug(f"Set {prop_name} = {value:.3f}")  # Too verbose
                
            # DO NOT modify diffuseColor - this would override material textures!
            # Reflection randomization should only change reflection properties,
            # not the base appearance/color/texture of the material.
            
            # Only add subtle emission if we want brighter surfaces (10% chance)
            if np.random.random() < 0.1:  # Reduced chance, more subtle
                emission = np.random.uniform(0.05, 0.15)  # Very subtle glow
                emission_input = shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f)
                emission_input.Set(Gf.Vec3f(emission, emission, emission))
                log.info(f"Added subtle ground glow: {emission:.3f}")  # Keep this one
                
        except Exception as e:
            log.debug(f"Failed to randomize UsdPreviewSurface properties: {e}")
    
    def _randomize_table_material(self):
        """Randomize table material."""
        self._randomize_surface_material("table")
    
    def _randomize_ground_material(self):
        """Randomize ground material."""
        self._randomize_surface_material("ground")
    
    def _randomize_wall_material(self):
        """Randomize wall material."""
        self._randomize_surface_material("wall")
    
    def _randomize_surface_material(self, surface_type: str):
        """Randomize material for a specific surface type using MDL material pools."""
        modules = self._get_isaac_modules()
        if not modules or not modules.get('usd_available', False):
            log.debug(f"USD not available, skipping {surface_type} material randomization")
            return
            
        try:
            # Load material paths from YAML
            mdl_paths = self._get_mdl_paths_from_yaml(f"{surface_type}_mdl_paths.yml")
            if not mdl_paths:
                log.warning(f"No MDL paths found for {surface_type}")
                return
                
            # Randomly select an MDL material
            selected_mdl = np.random.choice(mdl_paths)
            log.debug(f"Selected {surface_type} material: {selected_mdl}")
            
            # Find and apply to surface prims
            stage = modules['omni'].usd.get_context().get_stage()
            if not stage:
                return
                
            # Find surface prims based on type
            surface_prims = self._find_surface_prims(stage, surface_type)
            
            for prim_path in surface_prims:
                success = self._apply_mdl_to_prim(stage, prim_path, selected_mdl, modules)
                if success:
                    log.debug(f"Applied {surface_type} material to {prim_path}")
                    
            # Also randomize material properties
            self._randomize_material_properties(stage, surface_prims, modules)
            
        except Exception as e:
            log.warning(f"Failed to randomize {surface_type} materials: {e}")
            
    def _find_surface_prims(self, stage, surface_type: str) -> list[str]:
        """Find prims that match the surface type."""
        surface_prims = []
        
        # Define search patterns for different surface types
        search_patterns = {
            "table": ["table", "desk", "surface", "tabletop"],
            "ground": ["ground", "floor", "plane"],
            "wall": ["wall", "backdrop", "background"]
        }
        
        patterns = search_patterns.get(surface_type, [surface_type])
        
        try:
            # Search through all prims
            for prim in stage.Traverse():
                prim_name = str(prim.GetPath()).lower()
                
                # Check if prim name contains any of the patterns
                for pattern in patterns:
                    if pattern in prim_name and prim.IsA("Mesh"):
                        surface_prims.append(str(prim.GetPath()))
                        break
                        
        except Exception as e:
            log.warning(f"Error finding surface prims: {e}")
            
        return surface_prims
        
    def _randomize_material_properties(self, stage, prim_paths: list[str], modules):
        """Randomize additional material properties like roughness, metallic, etc."""
        try:
            UsdShade = modules.get('UsdShade')
            Gf = modules.get('Gf')
            if not UsdShade or not Gf:
                return
                
            for prim_path in prim_paths:
                prim = stage.GetPrimAtPath(prim_path)
                if not prim.IsValid():
                    continue
                    
                # Find or create material
                material_api = UsdShade.MaterialBindingAPI(prim)
                binding = material_api.GetDirectBinding()
                material = binding.GetMaterial() if binding else None
                
                if material:
                    # Randomize PBR properties
                    self._set_random_pbr_properties(material, modules)
                    
        except Exception as e:
            log.warning(f"Failed to randomize material properties: {e}")
            
    def _set_random_pbr_properties(self, material, modules):
        """Set random PBR material properties."""
        try:
            UsdShade = modules.get('UsdShade')
            Gf = modules.get('Gf')
            Sdf = modules.get('Sdf')
            if not all([UsdShade, Gf, Sdf]):
                return
                
            material_prim = material.GetPrim()
            
            # Random roughness (0.1 to 0.9)
            roughness = np.random.uniform(0.1, 0.9)
            roughness_input = material_prim.GetAttribute("inputs:roughness")
            if not roughness_input:
                roughness_input = material_prim.CreateAttribute("inputs:roughness", Sdf.ValueTypeNames.Float)
            roughness_input.Set(float(roughness))
            
            # Random metallic (0.0 to 0.8)
            metallic = np.random.uniform(0.0, 0.8)
            metallic_input = material_prim.GetAttribute("inputs:metallic")
            if not metallic_input:
                metallic_input = material_prim.CreateAttribute("inputs:metallic", Sdf.ValueTypeNames.Float)
            metallic_input.Set(float(metallic))
            
            # Random base color tint
            tint = np.random.uniform(0.8, 1.2, 3)  # Slight color variation
            tint_input = material_prim.GetAttribute("inputs:diffuse_tint")
            if not tint_input:
                tint_input = material_prim.CreateAttribute("inputs:diffuse_tint", Sdf.ValueTypeNames.Color3f)
            tint_input.Set(Gf.Vec3f(*tint))
            
        except Exception as e:
            log.warning(f"Failed to set PBR properties: {e}")
    
    def _randomize_scene(self):
        """Randomize scene elements."""
        log.debug("Randomizing scene elements...")
        
        modules = self._get_isaac_modules()
        if not modules or not modules.get('usd_available', False):
            log.debug("USD not available, skipping scene randomization")
            return
            
        try:
            # Add simple clutter objects
            self._add_clutter_objects()
            
        except Exception as e:
            log.warning(f"Scene randomization failed: {e}")
    
    def _get_mdl_paths_from_yaml(self, yaml_filename: str):
        """Load MDL paths from YAML configuration file."""
        try:
            import os
            yaml_path = os.path.join(
                os.path.dirname(__file__), 
                "../cfg/randomization", 
                yaml_filename
            )
            
            if not os.path.exists(yaml_path):
                log.warning(f"YAML file not found: {yaml_path}")
                return []
                
            with open(yaml_path, 'r') as f:
                data = yaml.load(f, Loader=yaml.FullLoader)
                
            # Extract MDL paths from YAML structure
            mdl_paths = []
            if isinstance(data, dict):
                # Support different dataset formats
                for dataset_name, dataset_data in data.items():
                    if isinstance(dataset_data, dict):
                        # Handle train/test/val splits
                        for split_name, paths in dataset_data.items():
                            if isinstance(paths, list):
                                mdl_paths.extend(paths)
                    elif isinstance(dataset_data, list):
                        # Direct list of paths
                        mdl_paths.extend(dataset_data)
            elif isinstance(data, list):
                # Direct list format
                mdl_paths = data
                
            log.debug(f"Loaded {len(mdl_paths)} MDL paths from {yaml_filename}")
            return mdl_paths
            
        except Exception as e:
            log.warning(f"Failed to load MDL paths from {yaml_filename}: {e}")
            return []
    
    def _apply_mdl_to_prim(self, stage, prim_path: str, mdl_path: str, modules):
        """Apply MDL material to a prim (based on IsaacLab's material_util)."""
        try:
            import os
            from metasim.utils.hf_util import check_and_download_single
            
            # Download MDL file if it doesn't exist
            if not os.path.exists(mdl_path):
                log.debug(f"Downloading MDL file: {mdl_path}")
                check_and_download_single(mdl_path)
                
            if not os.path.exists(mdl_path):
                log.warning(f"MDL file not found after download: {mdl_path}")
                return False
                
            if not mdl_path.endswith(".mdl"):
                log.warning(f"File is not MDL: {mdl_path}")
                return False
                
            # Get material name
            mtl_name = os.path.basename(mdl_path).replace(".mdl", "")
            
            # Try to get material path
            try:
                from omni.kit.material.library import get_material_prim_path
                _, mtl_prim_path = get_material_prim_path(mtl_name)
            except ImportError:
                mtl_prim_path = f"/World/Looks/{mtl_name}"
                
            # Create MDL material
            try:
                import omni.kit.commands
                success, result = omni.kit.commands.execute(
                    "CreateMdlMaterialPrim",
                    mtl_url=mdl_path,
                    mtl_name=mtl_name,
                    mtl_path=mtl_prim_path,
                    select_new_prim=False,
                )
                if not success:
                    log.warning(f"Failed to create MDL material: {mtl_name}")
                    return False
                    
                # Bind material to prim
                UsdShade = modules.get('UsdShade')
                strength = UsdShade.Tokens.strongerThanDescendants if UsdShade else None
                
                success, result = omni.kit.commands.execute(
                    "BindMaterial",
                    prim_path=prim_path,
                    material_path=mtl_prim_path,
                    strength=strength,
                )
                if not success:
                    log.warning(f"Failed to bind material to {prim_path}")
                    return False
                    
                log.debug(f"Successfully applied MDL {mtl_name} to {prim_path}")
                return True
                
            except Exception as e:
                log.warning(f"Failed to execute MDL commands: {e}")
                return False
                
        except Exception as e:
            log.warning(f"Failed to apply MDL {mdl_path} to {prim_path}: {e}")
            return False
    
    def _add_clutter_objects(self):
        """
        Add random clutter objects using metasim's proper object system.
        
        Inspired by IsaacLab's test approach of creating organized scene hierarchy
        with Xform containers for better object management.
        """
        if self.cfg.level < 2:  # Only add clutter objects at higher randomization levels
            return
            
        try:
            # Generate clutter object configurations using metasim's object system
            clutter_configs = self._create_metasim_clutter_objects()
            if not clutter_configs:
                log.debug("No clutter objects to add")
                return
                
            # Add objects to sim handler's scenario AND handler's objects list
            success = self._add_objects_to_scenario(clutter_configs)
            if success:
                # Also add to handler's objects list so they get found during state setting
                self._add_objects_to_handler(clutter_configs)
                # Load the new objects into the scene
                self._load_clutter_objects_to_scene(clutter_configs)
                log.info(f"Added {len(clutter_configs)} clutter objects using metasim system")
            else:
                log.debug("Failed to add clutter objects to scenario")
                
        except Exception as e:
            log.warning(f"Failed to add clutter objects: {e}")
            
    def _create_metasim_clutter_objects(self):
        """Create clutter objects using metasim's object configuration system."""
        try:
            import numpy as np
            from metasim.scenario.objects import PrimitiveCubeCfg, PrimitiveSphereCfg, PrimitiveCylinderCfg, RigidObjCfg, ArticulationObjCfg
            from metasim.constants import PhysicStateType
            
            # Get YAML configurations for reference
            yaml_configs = self._get_clutter_object_paths()
            if not yaml_configs:
                return []
                
            # Number of objects based on randomization level (conservative to prevent overlaps)
            num_objects = min(self.cfg.level, 3)  # 1-3 objects (reduced)
            
            clutter_objects = []
            placed_positions = []  # Track positions for collision avoidance
            
            for i in range(num_objects):
                # Select random configuration from YAML
                config = np.random.choice(yaml_configs)
                obj_type = config.get('type', 'primitive')
                obj_name = f"clutter_{i}_{config.get('name', 'object')}"
                
                # Find valid position
                position = self._find_valid_clutter_position(placed_positions, config)
                if position is None:
                    continue
                    
                placed_positions.append(position)
                
                # Create object configuration based on type
                if obj_type == 'primitive':
                    obj_cfg = self._create_primitive_metasim_cfg(config, obj_name, position)
                elif obj_type == 'rigid':
                    obj_cfg = self._create_rigid_metasim_cfg(config, obj_name, position)
                elif obj_type == 'articulation':
                    obj_cfg = self._create_articulation_metasim_cfg(config, obj_name, position)
                else:
                    log.debug(f"Unsupported object type: {obj_type}")
                    continue
                    
                if obj_cfg:
                    clutter_objects.append(obj_cfg)
                    
            return clutter_objects
            
        except Exception as e:
            log.warning(f"Failed to create metasim clutter objects: {e}")
            return []
            
    def _find_valid_clutter_position(self, placed_positions, config):
        """Find a valid position for a clutter object avoiding collisions with improved safety margins."""
        try:
            import numpy as np
            
            # Table bounds (similar to IsaacLab test approach)
            table_center = np.array([0.0, 0.0, 0.75])
            table_bounds = np.array([0.7, 0.3, 0.1])  # x, y, z extents
            
            obj_radius = config.get('radius', 0.05)
            scale_max = max(config.get('scale_range', [0.8, 1.2]))
            effective_radius = obj_radius * scale_max
            
            # Much larger minimum separation to prevent overlaps
            min_distance = max(0.15, effective_radius * 4.0)
            
            # Also avoid robot workspace (center area)
            robot_workspace_radius = 0.2
            
            # Try up to 100 times to find a valid position
            for attempt in range(100):
                # Use IsaacLab test approach: random in range then offset
                rand_pos = np.random.rand(3) - 0.5  # [-0.5, 0.5]
                rand_pos *= table_bounds  # Scale to table size
                rand_pos += table_center  # Offset to table center
                
                x, y, z = rand_pos[0], rand_pos[1], rand_pos[2] + config.get('z_offset', 0.05)
                
                # Check distance from robot workspace center
                robot_distance = np.sqrt(x**2 + y**2)
                if robot_distance < robot_workspace_radius:
                    continue
                
                # Check distance from existing objects
                valid = True
                for pos in placed_positions:
                    # Calculate 2D distance (on table surface)
                    distance = np.sqrt((x - pos[0])**2 + (y - pos[1])**2)
                    if distance < min_distance:
                        valid = False
                        break
                        
                if valid:
                    log.debug(f"Found valid position after {attempt + 1} attempts: [{x:.3f}, {y:.3f}, {z:.3f}]")
                    return [x, y, z]
                    
            log.warning(f"Could not find valid position after 100 attempts for object with radius {effective_radius}")
            return None
            
        except Exception as e:
            log.debug(f"Failed to find valid position: {e}")
            return None
            
    def _create_primitive_metasim_cfg(self, config, name, position):
        """Create a primitive object configuration using metasim's system."""
        try:
            import numpy as np
            from metasim.scenario.objects import PrimitiveCubeCfg, PrimitiveSphereCfg, PrimitiveCylinderCfg
            from metasim.constants import PhysicStateType
            
            # Support all available metasim primitive types
            available_types = ['Cube', 'Sphere', 'Cylinder']
            usd_type = config.get('usd_type', np.random.choice(available_types))
            scale_range = config.get('scale_range', [0.02, 0.08])
            color_range = config.get('color_range', [[0.1, 0.9], [0.1, 0.9], [0.1, 0.9]])
            
            # Random scale and color (similar to IsaacLab test approach)
            scale = np.random.uniform(*scale_range)
            color = [np.random.uniform(*color_range[i]) for i in range(3)]
            
            # Random rotation around Z-axis
            rot_z = np.random.uniform(0, 2 * np.pi)
            quat = [np.cos(rot_z/2), 0, 0, np.sin(rot_z/2)]  # Z-axis rotation
            
            if usd_type == 'Sphere':
                return PrimitiveSphereCfg(
                    name=name,
                    radius=scale,
                    color=color,
                    physics=PhysicStateType.RIGIDBODY
                )
            elif usd_type == 'Cylinder':
                return PrimitiveCylinderCfg(
                    name=name,
                    radius=scale,
                    height=scale * 2,  # Height is 2x radius
                    color=color,
                    physics=PhysicStateType.RIGIDBODY
                )
            else:  # Default to Cube
                return PrimitiveCubeCfg(
                    name=name,
                    size=(scale, scale, scale),
                    color=color,
                    physics=PhysicStateType.RIGIDBODY
                )
                
        except Exception as e:
            log.debug(f"Failed to create primitive metasim config: {e}")
            return None
            
    def _create_rigid_metasim_cfg(self, config, name, position):
        """Create a rigid object configuration using metasim's system."""
        try:
            import numpy as np
            import os
            from metasim.scenario.objects import RigidObjCfg
            from metasim.constants import PhysicStateType
            
            obj_path = config.get('path', '')
            if not obj_path or not os.path.exists(obj_path):
                log.debug(f"USD file not found: {obj_path}")
                return None
                
            scale_range = config.get('scale_range', [0.8, 1.2])
            scale = np.random.uniform(*scale_range)
            
            # Random rotation around Z-axis
            rot_z = np.random.uniform(0, 2 * np.pi)
            quat = [np.cos(rot_z/2), 0, 0, np.sin(rot_z/2)]  # Z-axis rotation
            
            return RigidObjCfg(
                name=name,
                scale=(scale, scale, scale),
                physics=PhysicStateType.RIGIDBODY,  # This automatically sets collision_enabled=True
                usd_path=obj_path
            )
            
        except Exception as e:
            log.debug(f"Failed to create rigid metasim config: {e}")
            return None
            
    def _create_articulation_metasim_cfg(self, config, name, position):
        """Create an articulation object configuration using metasim's system."""
        try:
            import numpy as np
            import os
            from metasim.scenario.objects import ArticulationObjCfg
            
            obj_path = config.get('path', '')
            if not obj_path or not os.path.exists(obj_path):
                log.debug(f"USD file not found: {obj_path}")
                return None
                
            scale_range = config.get('scale_range', [0.8, 1.2])
            scale = np.random.uniform(*scale_range)
            
            # Get articulation-specific settings
            fix_base_link = config.get('fix_base_link', True)
            
            return ArticulationObjCfg(
                name=name,
                scale=(scale, scale, scale),
                fix_base_link=fix_base_link,
                usd_path=obj_path,
                # Note: URDF and MJCF paths could be added if available in config
                urdf_path=obj_path.replace('/usd/', '/urdf/').replace('.usd', '_unique.urdf'),
                mjcf_path=obj_path.replace('/usd/', '/mjcf/').replace('.usd', '_unique.mjcf')
            )
            
        except Exception as e:
            log.debug(f"Failed to create articulation metasim config: {e}")
            return None
            
    def _add_objects_to_scenario(self, clutter_objects):
        """Add clutter objects to the simulation handler's scenario."""
        try:
            # Check if sim_handler has scenario
            if not hasattr(self.sim_handler, 'scenario_cfg') or not self.sim_handler.scenario_cfg:
                log.debug("No scenario configuration available")
                return False
                
            # Add objects to scenario configuration
            if not hasattr(self.sim_handler.scenario_cfg, 'objects'):
                self.sim_handler.scenario_cfg.objects = []
            elif self.sim_handler.scenario_cfg.objects is None:
                self.sim_handler.scenario_cfg.objects = []
                
            # Extend existing objects with clutter
            self.sim_handler.scenario_cfg.objects.extend(clutter_objects)
            
            log.debug(f"Added {len(clutter_objects)} objects to scenario configuration")
            return True
            
        except Exception as e:
            log.debug(f"Failed to add objects to scenario: {e}")
            return False
            
    def _add_objects_to_handler(self, clutter_objects):
        """Add clutter objects to simulation handler's objects list."""
        try:
            # Add objects to the handler's objects list so they can be found during state setting
            if hasattr(self.sim_handler, 'objects') and self.sim_handler.objects is not None:
                self.sim_handler.objects.extend(clutter_objects)
                
                # Also update the object_dict for quick lookup
                if hasattr(self.sim_handler, 'object_dict'):
                    for obj in clutter_objects:
                        self.sim_handler.object_dict[obj.name] = obj
                        
                log.debug(f"Added {len(clutter_objects)} objects to handler's objects list")
                return True
            else:
                log.debug("Handler objects list not available")
                return False
        except Exception as e:
            log.debug(f"Failed to add objects to handler: {e}")
            return False
            
    def _load_clutter_objects_to_scene(self, clutter_objects):
        """Load clutter objects into the IsaacSim scene."""
        try:
            # Use the handler's _add_object method to add each object to the scene
            for obj_cfg in clutter_objects:
                if hasattr(self.sim_handler, '_add_object'):
                    self.sim_handler._add_object(obj_cfg)
                    log.debug(f"Loaded clutter object {obj_cfg.name} to scene")
                else:
                    log.debug("Handler _add_object method not available")
                    return False
            return True
        except Exception as e:
            log.debug(f"Failed to load clutter objects to scene: {e}")
            return False

    def _get_clutter_object_paths(self):
        """Get clutter object configurations from YAML."""
        try:
            import os
            yaml_path = os.path.join(
                os.path.dirname(__file__),
                "../cfg/randomization",
                "objects_paths.yml"
            )
            
            if not os.path.exists(yaml_path):
                log.warning(f"Clutter objects YAML not found: {yaml_path}")
                return []
                
            import yaml
            with open(yaml_path, 'r') as f:
                data = yaml.safe_load(f)
                
            # Extract object configurations
            clutter_objects = []
            
            # Get available objects (real USD files)
            if 'available_objects' in data:
                for split in ['train', 'val', 'test']:
                    if split in data['available_objects']:
                        clutter_objects.extend(data['available_objects'][split])
                        
            # Get primitive objects as fallback
            if 'primitive_objects' in data and len(clutter_objects) < 5:
                for split in ['train', 'val', 'test']:
                    if split in data['primitive_objects']:
                        clutter_objects.extend(data['primitive_objects'][split])
                        
            return clutter_objects[:20]  # Limit for performance
            
        except Exception as e:
            log.warning(f"Failed to load clutter object configurations: {e}")
            return []
