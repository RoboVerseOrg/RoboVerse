"""IsaacSim Backend Adapter - Unified USD operation interface.

The IsaacSimAdapter provides a clean abstraction layer over IsaacSim/USD APIs,
enabling randomizers to perform operations without direct USD API dependencies.

This adapter handles:
- Transform operations (set/get position, rotation, scale)
- Material operations (MDL and PBR materials)
- Light operations (intensity, color, position)
- Physics operations (limited, mainly for reference)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from metasim.sim.isaacsim import IsaacsimHandler


class IsaacSimAdapter:
    """Adapter for IsaacSim backend operations.

    Provides unified interface for USD operations, abstracting away the complexity
    of IsaacSim/USD APIs.

    Usage:
        adapter = IsaacSimAdapter(handler)
        adapter.set_transform("/World/table", position=(0, 0, 0.7))
        adapter.apply_mdl_material("/World/table", "materials/wood/oak.mdl")
    """

    def __init__(self, handler: IsaacsimHandler):
        """Initialize IsaacSim adapter.

        Args:
            handler: IsaacSim handler instance
        """
        self.handler = handler

        # Lazy import IsaacSim modules
        try:
            import omni

            try:
                import omni.isaac.core.utils.prims as prim_utils
            except ModuleNotFoundError:
                import isaacsim.core.utils.prims as prim_utils

            from omni.kit.material.library import get_material_prim_path
            from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade
        except ImportError as e:
            raise ImportError(f"IsaacSim modules not available: {e}") from e

        self.omni = omni
        self.prim_utils = prim_utils
        self.get_material_prim_path = get_material_prim_path
        self.Gf = Gf
        self.Sdf = Sdf
        self.UsdGeom = UsdGeom
        self.UsdPhysics = UsdPhysics
        self.UsdShade = UsdShade
        self.stage = omni.usd.get_context().get_stage()

    # -------------------------------------------------------------------------
    # Transform Operations
    # -------------------------------------------------------------------------

    def set_transform(
        self,
        prim_path: str,
        position: tuple[float, float, float] | None = None,
        rotation: tuple[float, float, float, float] | None = None,
        scale: tuple[float, float, float] | None = None,
    ):
        """Set object transform via USD API.

        This is used for Dynamic Objects (created by SceneRandomizer).
        For Static Objects, use Handler API instead.

        Args:
            prim_path: USD prim path
            position: Position (x, y, z) in meters
            rotation: Rotation as quaternion (w, x, y, z)
            scale: Scale (x, y, z)

        Raises:
            ValueError: If prim path is invalid
        """
        prim = self.stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            raise ValueError(f"Invalid prim path: {prim_path}")

        xform = self.UsdGeom.Xformable(prim)
        if not xform:
            raise ValueError(f"Prim at {prim_path} is not transformable")

        # Get existing xform ops
        xform_ops = xform.GetOrderedXformOps()

        if position is not None:
            # Find or create translate op
            translate_ops = [op for op in xform_ops if op.GetOpType() == self.UsdGeom.XformOp.TypeTranslate]
            if translate_ops:
                translate_ops[0].Set(self.Gf.Vec3d(*position))
            else:
                xform.AddTranslateOp().Set(self.Gf.Vec3d(*position))

        if rotation is not None:
            # rotation is quaternion (w, x, y, z)
            orient_ops = [op for op in xform_ops if op.GetOpType() == self.UsdGeom.XformOp.TypeOrient]
            quat = self.Gf.Quatd(rotation[0], self.Gf.Vec3d(rotation[1], rotation[2], rotation[3]))
            if orient_ops:
                orient_ops[0].Set(quat)
            else:
                xform.AddOrientOp().Set(quat)

        if scale is not None:
            scale_ops = [op for op in xform_ops if op.GetOpType() == self.UsdGeom.XformOp.TypeScale]
            if scale_ops:
                scale_ops[0].Set(self.Gf.Vec3d(*scale))
            else:
                xform.AddScaleOp().Set(self.Gf.Vec3d(*scale))

    def get_transform(self, prim_path: str) -> tuple[tuple, tuple, tuple]:
        """Get object transform.

        Args:
            prim_path: USD prim path

        Returns:
            Tuple of (position, rotation, scale)
        """
        prim = self.stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            raise ValueError(f"Invalid prim path: {prim_path}")

        xform = self.UsdGeom.Xformable(prim)
        xform_ops = xform.GetOrderedXformOps()

        position = (0.0, 0.0, 0.0)
        rotation = (1.0, 0.0, 0.0, 0.0)
        scale = (1.0, 1.0, 1.0)

        for op in xform_ops:
            if op.GetOpType() == self.UsdGeom.XformOp.TypeTranslate:
                vec = op.Get()
                position = (vec[0], vec[1], vec[2])
            elif op.GetOpType() == self.UsdGeom.XformOp.TypeOrient:
                quat = op.Get()
                rotation = (quat.GetReal(), quat.GetImaginary()[0], quat.GetImaginary()[1], quat.GetImaginary()[2])
            elif op.GetOpType() == self.UsdGeom.XformOp.TypeScale:
                vec = op.Get()
                scale = (vec[0], vec[1], vec[2])

        return position, rotation, scale

    # -------------------------------------------------------------------------
    # Asset Download Operations
    # -------------------------------------------------------------------------

    def ensure_usd_downloaded(self, usd_path: str, auto_download: bool = True) -> bool:
        """Ensure USD asset is downloaded and ready.

        Args:
            usd_path: Path to USD/URDF file
            auto_download: Auto-download if missing

        Returns:
            True if file is ready, False otherwise
        """
        import os

        from metasim.utils.hf_util import check_and_download_single

        if os.path.exists(usd_path):
            return True

        if not auto_download:
            logger.warning(f"USD file not found: {usd_path}")
            return False

        logger.info(f"Downloading missing USD: {usd_path}")
        try:
            result = check_and_download_single(usd_path)
            if result and os.path.exists(usd_path):
                return True
            else:
                logger.error(f"Failed to download USD: {usd_path}")
                return False
        except Exception as e:
            logger.error(f"Error downloading USD {usd_path}: {e}")
            return False

    # -------------------------------------------------------------------------
    # Material Operations
    # -------------------------------------------------------------------------

    def apply_mdl_material(
        self,
        prim_path: str,
        mdl_path: str,
        material_name: str | None = None,
        auto_download: bool = True,
    ):
        """Apply MDL material to a prim using IsaacSim's correct workflow.

        Args:
            prim_path: USD prim path
            mdl_path: Path to MDL file
            material_name: Specific material variant name (None = uses export material from MDL)
            auto_download: Auto-download missing MDL and textures

        Raises:
            ValueError: If prim path is invalid
            FileNotFoundError: If MDL file not found and auto_download=False
        """
        import os

        import isaacsim.core.utils.prims as prim_utils
        import omni.kit.commands
        from omni.kit.material.library import get_material_prim_path
        from pxr import UsdGeom, UsdShade

        from metasim.utils.hf_util import check_and_download_single, extract_texture_paths_from_mdl

        # Convert to absolute path
        mdl_path = os.path.abspath(mdl_path)

        # Download MDL if missing
        if not os.path.exists(mdl_path):
            if not auto_download:
                raise FileNotFoundError(f"MDL file not found: {mdl_path}")

            logger.info(f"Downloading missing MDL: {mdl_path}")
            check_and_download_single(mdl_path)
            # Verify file exists after download attempt
            if not os.path.exists(mdl_path):
                raise RuntimeError(f"Failed to download MDL: {mdl_path}")

        # Download textures
        try:
            texture_paths = extract_texture_paths_from_mdl(mdl_path)
            for tex_path in texture_paths:
                if not os.path.exists(tex_path) and auto_download:
                    check_and_download_single(tex_path)
        except Exception as e:
            logger.debug(f"Failed to download textures: {e}")

        # Get prim
        prim = prim_utils.get_prim_at_path(prim_path)
        if not prim or not prim.IsValid():
            raise ValueError(f"Invalid prim path: {prim_path}")

        # Ensure UV coordinates (critical for textures!)
        # For articulations/complex objects, apply to all child meshes
        self._ensure_uv_for_prim_recursive(prim)

        # Material naming:
        # - mdl_basename: Name from MDL file (e.g., "Plywood")
        # - material_name: Variant within MDL (optional, e.g., "Plywood::Oak")
        # - mtl_prim_name: Unique prim name to avoid conflicts
        mdl_basename = os.path.basename(mdl_path).removesuffix(".mdl")
        prim_basename = prim_path.split("/")[-1]
        mtl_prim_unique = f"{mdl_basename}_{prim_basename}_{id(prim_path)}"  # Fully unique

        _, mtl_prim_path = get_material_prim_path(mtl_prim_unique)

        # Create material prim if needed
        if not self.stage.GetPrimAtPath(mtl_prim_path).IsValid():
            # mtl_name should be the actual material name in MDL file
            actual_mdl_name = material_name if material_name else mdl_basename
            success, _ = omni.kit.commands.execute(
                "CreateMdlMaterialPrim",
                mtl_url=mdl_path,
                mtl_name=actual_mdl_name,  # Use MDL internal name
                mtl_path=mtl_prim_path,  # But create at unique path
                select_new_prim=False,
            )
            if not success:
                raise RuntimeError(f"Failed to create material from {mdl_path}")

        # Ensure double-sided rendering
        if prim.IsA(UsdGeom.Mesh):
            mesh = UsdGeom.Mesh(prim)
            mesh.CreateDoubleSidedAttr(True)

        # Bind material to geometry
        success, _ = omni.kit.commands.execute(
            "BindMaterial",
            prim_path=prim.GetPath(),
            material_path=mtl_prim_path,
            strength=UsdShade.Tokens.strongerThanDescendants,
        )

        if not success:
            raise RuntimeError(f"Failed to bind material {actual_mdl_name} to {prim_path}")

    def _ensure_uv_for_prim_recursive(self, prim):
        """Recursively ensure UV coordinates for prim and all child meshes.

        This is critical for articulations and complex objects with multiple meshes.
        """
        from pxr import UsdGeom

        # Apply to current prim
        self._ensure_uv_for_prim(prim)

        # Recursively apply to all children
        for child in prim.GetChildren():
            if child.IsA(UsdGeom.Mesh) or child.IsA(UsdGeom.Gprim):
                self._ensure_uv_for_prim(child)
            elif child.GetChildren():  # Has children, continue recursion
                self._ensure_uv_for_prim_recursive(child)

    def _ensure_uv_for_prim(self, prim):
        """Ensure UV coordinates for mesh (required for textures).

        This generates proper UV coordinates for procedural geometry to display textures correctly.
        """
        from math import isfinite

        from pxr import Gf, Sdf, UsdGeom, Vt

        if not prim.IsA(UsdGeom.Mesh):
            # For non-mesh gprims (Cube, Sphere, etc.), use basic UV
            if prim.IsA(UsdGeom.Gprim):
                self._ensure_basic_uv_for_gprim(prim)
            return

        mesh = UsdGeom.Mesh(prim)

        # Check if UV coordinates already exist
        primvars_api = UsdGeom.PrimvarsAPI(mesh)
        st_primvar = primvars_api.GetPrimvar("st")

        if st_primvar and st_primvar.HasValue():
            return  # Already has UVs

        # Generate UV coordinates based on bounding box projection
        pts = mesh.GetPointsAttr().Get() or []
        idx = mesh.GetFaceVertexIndicesAttr().Get() or []

        if not pts or not idx:
            return

        # Calculate bounding box
        xs, ys, zs = [p[0] for p in pts], [p[1] for p in pts], [p[2] for p in pts]
        rx = (max(xs) - min(xs)) if xs else 0
        ry = (max(ys) - min(ys)) if ys else 0
        rz = (max(zs) - min(zs)) if zs else 0

        # Pick largest two axes for projection
        axes = sorted([("x", rx), ("y", ry), ("z", rz)], key=lambda t: t[1], reverse=True)
        a, b = axes[0][0], axes[1][0]

        def comp(p, axis):
            return p[0] if axis == "x" else (p[1] if axis == "y" else p[2])

        # Generate UVs
        tile = 0.2  # Tiling factor
        st_list = []
        for vid in idx:
            p = pts[vid]
            u = comp(p, a) * tile
            v = comp(p, b) * tile
            u = 0.0 if not isfinite(u) else u
            v = 0.0 if not isfinite(v) else v
            st_list.append((u, v))

        # Create UV primvar
        try:
            st = Vt.Vec2fArray([Gf.Vec2f(u, v) for (u, v) in st_list])
            pv = primvars_api.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
            pv.Set(st)
            pv.SetInterpolation(UsdGeom.Tokens.faceVarying)
        except Exception as e:
            logger.debug(f"Failed to create UVs: {e}")

    def force_pose_nudge(self, prim_paths: list[str]):
        """Apply tiny temporary translation to force RTX BLAS update.

        Required for IsaacSim 4.5+ to ensure material changes are immediately visible.
        Without this, materials may appear unchanged until the object moves naturally.

        Args:
            prim_paths: Prim paths that had material changes
        """
        if not prim_paths:
            return

        try:
            from pxr import Gf, UsdGeom

            nudged_ops = []

            for prim_path in prim_paths:
                prim = self.stage.GetPrimAtPath(prim_path)
                if not prim or not prim.IsValid():
                    continue

                xformable = UsdGeom.Xformable(prim)
                if not xformable:
                    continue

                # Get or create translate op
                translate_op = None
                for op in xformable.GetOrderedXformOps():
                    if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                        translate_op = op
                        break

                if translate_op is None:
                    translate_op = xformable.AddTranslateOp(opSuffix="dr_refresh")

                # Save original
                original_val = translate_op.Get()
                if original_val is None:
                    original_val = Gf.Vec3d(0, 0, 0)

                # Apply tiny offset
                translate_op.Set(original_val + Gf.Vec3d(1e-4, 0, 0))
                nudged_ops.append((translate_op, original_val))

            # Flush with offset
            if not (hasattr(self.handler, "_defer_all_visual_flushes") and self.handler._defer_all_visual_flushes):
                if hasattr(self.handler, "flush_visual_updates"):
                    self.handler.flush_visual_updates(settle_passes=1)

            # Restore original
            for op, original_val in nudged_ops:
                op.Set(original_val)

            # Final flush
            if not (hasattr(self.handler, "_defer_all_visual_flushes") and self.handler._defer_all_visual_flushes):
                if hasattr(self.handler, "flush_visual_updates"):
                    self.handler.flush_visual_updates(settle_passes=1)

        except Exception as e:
            logger.debug(f"Pose nudge failed (non-critical): {e}")

    def _ensure_basic_uv_for_gprim(self, gprim):
        """Generate UV coordinates for geometric primitives (Cube, Sphere, etc.)."""
        from pxr import Gf, Sdf, UsdGeom, Vt

        pvapi = UsdGeom.PrimvarsAPI(gprim)
        if pvapi.HasPrimvar("st"):
            return

        prim_type = gprim.GetTypeName()
        tile = 1.0

        if prim_type == "Cube":
            # 6 faces, 4 vertices each = 24 UVs
            cube_uvs = []
            for _ in range(6):
                cube_uvs.extend([(0.0, 0.0), (tile, 0.0), (tile, tile), (0.0, tile)])
            st = Vt.Vec2fArray([Gf.Vec2f(u, v) for (u, v) in cube_uvs])
        elif prim_type == "Sphere":
            # Spherical mapping (simplified)
            st = Vt.Vec2fArray([Gf.Vec2f(0.5, 0.5)])  # Fallback
        else:
            # Default mapping
            st = Vt.Vec2fArray([Gf.Vec2f(0.0, 0.0)])

        try:
            pv = pvapi.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
            pv.Set(st)
        except Exception as e:
            logger.debug(f"Failed to create basic UVs for {prim_type}: {e}")

    def apply_pbr_material(self, prim_path: str, pbr_config: dict):
        """Apply PBR material to a prim.

        Args:
            prim_path: USD prim path
            pbr_config: PBR configuration dict with keys:
                - roughness: float
                - metallic: float
                - specular: float
                - diffuse_color: tuple[float, float, float]

        Raises:
            ValueError: If prim path is invalid
        """
        prim = self.stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            raise ValueError(f"Invalid prim path: {prim_path}")

        # Create unique material
        mtl_name = f"pbr_material_{hash(prim_path)}"
        _, mtl_prim_path = self.get_material_prim_path(mtl_name)
        material = self.UsdShade.Material.Define(self.stage, mtl_prim_path)

        # Bind material
        self.omni.kit.commands.execute(
            "BindMaterial",
            prim_path=prim.GetPath(),
            material_path=mtl_prim_path,
            strength=self.UsdShade.Tokens.strongerThanDescendants,
        )

        # Create shader
        shader_path = material.GetPrim().GetPath().AppendChild("Shader")
        shader = self.UsdShade.Shader.Define(self.stage, shader_path)
        shader.CreateIdAttr("UsdPreviewSurface")
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

        # Set PBR properties
        if "roughness" in pbr_config:
            shader.CreateInput("roughness", self.Sdf.ValueTypeNames.Float).Set(float(pbr_config["roughness"]))

        if "metallic" in pbr_config:
            shader.CreateInput("metallic", self.Sdf.ValueTypeNames.Float).Set(float(pbr_config["metallic"]))

        if "specular" in pbr_config:
            shader.CreateInput("specular", self.Sdf.ValueTypeNames.Float).Set(float(pbr_config["specular"]))

        if "diffuse_color" in pbr_config:
            color = pbr_config["diffuse_color"]
            shader.CreateInput("diffuseColor", self.Sdf.ValueTypeNames.Color3f).Set(
                self.Gf.Vec3f(color[0], color[1], color[2])
            )

    # -------------------------------------------------------------------------
    # Light Operations
    # -------------------------------------------------------------------------

    def set_light_intensity(self, prim_path: str, intensity: float):
        """Set light intensity.

        Args:
            prim_path: USD prim path to light
            intensity: Light intensity value

        Raises:
            ValueError: If prim path is invalid or not a light
        """
        prim = self.stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            raise ValueError(f"Invalid prim path: {prim_path}")

        intensity_attr = prim.GetAttribute("inputs:intensity")
        if not intensity_attr:
            raise ValueError(f"Prim at {prim_path} does not have intensity attribute (not a light?)")

        intensity_attr.Set(float(intensity))

    def set_light_color(self, prim_path: str, color: tuple[float, float, float]):
        """Set light color.

        Args:
            prim_path: USD prim path to light
            color: RGB color (0-1 range)

        Raises:
            ValueError: If prim path is invalid or not a light
        """
        prim = self.stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            raise ValueError(f"Invalid prim path: {prim_path}")

        color_attr = prim.GetAttribute("inputs:color")
        if not color_attr:
            raise ValueError(f"Prim at {prim_path} does not have color attribute (not a light?)")

        color_attr.Set(self.Gf.Vec3f(color[0], color[1], color[2]))

    def get_light_intensity(self, prim_path: str) -> float:
        """Get light intensity.

        Args:
            prim_path: USD prim path to light

        Returns:
            Light intensity value
        """
        prim = self.stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            raise ValueError(f"Invalid prim path: {prim_path}")

        intensity_attr = prim.GetAttribute("inputs:intensity")
        if not intensity_attr:
            return 0.0

        return float(intensity_attr.Get())

    # -------------------------------------------------------------------------
    # Physics Operations (Limited - mainly for Dynamic Objects)
    # -------------------------------------------------------------------------

    def disable_physics(self, prim_path: str, recursive: bool = True):
        """Disable physics for a prim (make it pure visual).

        This recursively removes RigidBodyAPI and CollisionAPI.

        Args:
            prim_path: USD prim path
            recursive: Whether to process descendants
        """
        prim = self.stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            return

        if recursive:
            for descendant in prim.GetAllChildren():
                self.disable_physics(str(descendant.GetPath()), recursive=True)

        # Remove physics APIs
        if prim.HasAPI(self.UsdPhysics.RigidBodyAPI):
            prim.RemoveAPI(self.UsdPhysics.RigidBodyAPI)

        if prim.HasAPI(self.UsdPhysics.CollisionAPI):
            prim.RemoveAPI(self.UsdPhysics.CollisionAPI)

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------

    def is_valid_prim(self, prim_path: str) -> bool:
        """Check if a prim path is valid.

        Args:
            prim_path: USD prim path

        Returns:
            True if valid, False otherwise
        """
        return self.prim_utils.is_prim_path_valid(prim_path)

    def create_xform(self, prim_path: str) -> bool:
        """Create an Xform prim at the given path.

        Args:
            prim_path: USD prim path

        Returns:
            True if successful
        """
        try:
            prim = self.stage.DefinePrim(prim_path, "Xform")
            return prim and prim.IsValid()
        except Exception as e:
            logger.error(f"Failed to create Xform at {prim_path}: {e}")
            return False

    def delete_prim(self, prim_path: str):
        """Delete a prim.

        Args:
            prim_path: USD prim path
        """
        if self.is_valid_prim(prim_path):
            self.prim_utils.delete_prim(prim_path)
