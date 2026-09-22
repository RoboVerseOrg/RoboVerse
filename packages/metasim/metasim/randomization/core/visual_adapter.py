"""Renderer adapters for realized visual recipes; optional imports stay at bind time.

These editors own only their generated materials/images and use bounded names.
They never step physics. A recipe is preflighted as a whole before edits begin;
backend failures propagate (edits are not a transactional rollback).
"""

from __future__ import annotations

import copy
import hashlib
import math

import numpy as np


def _key(name):
    return hashlib.blake2b(name.encode(), digest_size=12).hexdigest()


def _camera_pose(cfg, values):
    pos = tuple(a + b for a, b in zip(cfg.pos, values["position_delta"], strict=False))
    target = tuple(a + b for a, b in zip(cfg.look_at, values["look_at_delta"], strict=False))
    if sum((a - b) ** 2 for a, b in zip(pos, target, strict=False)) < 1e-12:
        raise ValueError(f"Camera {cfg.name!r} position and look-at point coincide")
    return pos, target


def _focus(values, pos, target):
    """Thin-lens parameters: an f-stop of zero keeps the renderer's pinhole camera."""
    f_stop = float(values.get("f_stop", 0.0))
    return f_stop, math.dist(pos, target) * float(values.get("focus_scale", 1.0))


# Isaac Sim derives the thin-lens aperture from the USD ``focalLength`` read in tenths of the
# metre stage unit (24 -> 2.4 m) while Blender reads the same 24 as millimetres, so the same
# f-number opens Isaac's aperture 100x wider. Measured: Isaac f/140 matches Blender f/1.4.
ISAAC_FSTOP_SCALE = 100.0
BOX_UV_LAYER = "metasim_box"
IDLE_IMAGE_POOL = 8
"""Unused Blender image datablocks kept loaded, so alternating recipes do not re-decode them."""


def _box_uv(point, axis):
    """Metric planar UV of a point on a face whose normal is dominated by ``axis``."""
    if axis == 0:
        return point[1], point[2]
    if axis == 1:
        return point[0], point[2]
    return point[0], point[1]


class BlenderVisualAdapter:
    """Edit a launched Blender scene, retaining one material per named target."""

    def __init__(self, handler):
        import bpy

        self.bpy = bpy
        self.handler = handler
        self.cameras = {c.name: copy.deepcopy(c) for c in handler.cameras}
        self.lights = {}
        self.materials = {}
        self.images = {}
        self._material_topologies = {}
        # mesh pointer -> the authored render UV layer name (None if the mesh had none)
        self._box_uv_meshes = {}
        self._world_has_hdri = None
        self.world = None
        for obj in bpy.context.scene.objects:
            if obj.type == "LIGHT":
                self.lights[obj.name] = (obj, float(obj.data.energy), obj.location.copy())

    def _meshes(self, name):
        root = self.handler._objs.get(name)
        if root is None:
            root = self.bpy.context.scene.objects.get(name)
        if root is None:
            raise ValueError(f"Unknown Blender material target: {name!r}")
        meshes = [obj for obj in [root, *root.children_recursive] if obj.type == "MESH"]
        if not meshes:
            raise ValueError(f"Blender material target {name!r} has no meshes")
        if any(obj.instance_type != "NONE" for obj in [root, *root.children_recursive]):
            raise NotImplementedError("Realize collection instances before material augmentation")
        return meshes

    def validate(self, recipe, env_ids):
        """Resolve every target before any scene mutation."""
        if self.handler.num_envs != 1:
            raise NotImplementedError("Blender visual augmentation supports one environment per process")
        selected = set()
        for name, values in recipe.materials.items():
            meshes = self._meshes(name)
            names = {obj.name for obj in meshes}
            if selected & names:
                raise ValueError("Material targets overlap; select disjoint object hierarchies")
            selected.update(names)
            textured = any(values["textures"].values())
            if (
                textured
                and values.get("uv_projection", "mesh") == "mesh"
                and any(not [layer for layer in obj.data.uv_layers if layer.name != BOX_UV_LAYER] for obj in meshes)
            ):
                raise ValueError(f"Texture target {name!r} requires UV coordinates on every mesh")
        for name in recipe.lights:
            if name not in self.lights:
                raise ValueError(f"Unknown Blender light {name!r}; use environment for world/dome lighting")
        for name, values in recipe.cameras.items():
            cfg = self.cameras.get(name)
            if cfg is None or name not in self.handler._camera_objs:
                raise ValueError(f"Unknown camera: {name!r}")
            if cfg.mount_to is not None or cfg.intrinsic is not None:
                raise NotImplementedError("Visual camera jitter requires an unmounted aperture-based pinhole camera")
            _camera_pose(cfg, values)

    def _image(self, path, color_space):
        key = (path, color_space)
        if key not in self.images:
            # Own the datablock: never change colorspace on an imported asset's image.
            image = self.bpy.data.images.load(path, check_existing=False)
            image.colorspace_settings.name = color_space
        else:
            image = self.images.pop(key)
        self.images[key] = image  # re-inserted: the mapping is ordered oldest-used first
        return image

    def _box_uvs(self, obj):
        if obj.data.users > 1:
            obj.data = obj.data.copy()
        mesh = obj.data
        layer = mesh.uv_layers.get(BOX_UV_LAYER)
        if layer is None or mesh.as_pointer() not in self._box_uv_meshes:
            authored = next((layer for layer in mesh.uv_layers if layer.active_render), None)
            self._box_uv_meshes[mesh.as_pointer()] = authored.name if authored else None
            layer = layer or mesh.uv_layers.new(name=BOX_UV_LAYER)
            scale = obj.matrix_world.to_scale()
            for poly in mesh.polygons:
                axis = max(range(3), key=lambda i: abs(poly.normal[i]))
                for index in poly.loop_indices:
                    co = mesh.vertices[mesh.loops[index].vertex_index].co
                    layer.data[index].uv = _box_uv([co[i] * scale[i] for i in range(3)], axis)
        layer.active_render = True
        mesh.uv_layers.active = layer

    def _restore_uvs(self, obj):
        """Make the mesh's authored UV layer the render layer again after box projection."""
        authored = self._box_uv_meshes.get(obj.data.as_pointer())
        layer = obj.data.uv_layers.get(authored) if authored else None
        if layer is not None:
            layer.active_render = True
            obj.data.uv_layers.active = layer

    def _material(self, name, values):
        mat = self.materials.get(name)
        if mat is None:
            mat = self.bpy.data.materials.new(f"metasim_visual_{_key(name)}")
            mat.use_nodes = True
            self.materials[name] = mat
        tree = mat.node_tree
        topology = tuple(sorted(channel for channel, path in values["textures"].items() if path))
        # Keep shader nodes alive while topology stays constant. Rebuilding every
        # episode needlessly invalidates Cycles shader caches and allocates nodes.
        if self._material_topologies.get(name) != topology:
            tree.nodes.clear()
            surface = tree.nodes.new("ShaderNodeBsdfPrincipled")
            surface.name = "Surface"
            out = tree.nodes.new("ShaderNodeOutputMaterial")
            tree.links.new(surface.outputs["BSDF"], out.inputs["Surface"])
            if topology:
                coord = tree.nodes.new("ShaderNodeTexCoord")
                mapping = tree.nodes.new("ShaderNodeMapping")
                mapping.name = "UV"
                mapping.vector_type = "POINT"
                tree.links.new(coord.outputs["UV"], mapping.inputs["Vector"])
            for channel in topology:
                tex = tree.nodes.new("ShaderNodeTexImage")
                tex.name = "Tex_" + channel
                tree.links.new(mapping.outputs["Vector"], tex.inputs["Vector"])
                output = tex.outputs["Color"]
                if channel == "base_color":
                    tree.links.new(output, surface.inputs["Base Color"])
                elif channel == "normal":
                    normal = tree.nodes.new("ShaderNodeNormalMap")
                    tree.links.new(output, normal.inputs["Color"])
                    tree.links.new(normal.outputs["Normal"], surface.inputs["Normal"])
                else:
                    separate = tree.nodes.new("ShaderNodeSeparateColor")
                    separate.mode = "RGB"
                    tree.links.new(output, separate.inputs["Color"])
                    if channel == "orm":
                        tree.links.new(separate.outputs["Green"], surface.inputs["Roughness"])
                        tree.links.new(separate.outputs["Blue"], surface.inputs["Metallic"])
                    else:
                        socket = "Roughness" if channel == "roughness" else "Metallic"
                        tree.links.new(separate.outputs["Red"], surface.inputs[socket])
            self._material_topologies[name] = topology
        surface = tree.nodes["Surface"]
        surface.inputs["Base Color"].default_value = (*values["color"], 1.0)
        surface.inputs["Roughness"].default_value = values["roughness"]
        surface.inputs["Metallic"].default_value = values["metallic"]
        surface.inputs["IOR"].default_value = values.get("ior", 1.5)
        if topology:
            mapping = tree.nodes["UV"]
            mapping.inputs["Scale"].default_value = (*values.get("uv_scale", (1, 1)), 1)
            mapping.inputs["Rotation"].default_value[2] = values.get("uv_rotation", 0)
        for channel in topology:
            tree.nodes["Tex_" + channel].image = self._image(
                values["textures"][channel], "sRGB" if channel == "base_color" else "Non-Color"
            )
        for obj in self._meshes(name):
            if topology and values.get("uv_projection", "mesh") == "box":
                self._box_uvs(obj)
            elif obj.data.as_pointer() in self._box_uv_meshes:
                self._restore_uvs(obj)
            if not obj.material_slots:
                if obj.data.users > 1:
                    obj.data = obj.data.copy()
                obj.data.materials.append(mat)
            for slot in obj.material_slots:
                slot.link = "OBJECT"
                slot.material = mat

    def _environment(self, values):
        scene = self.bpy.context.scene
        if self.world is None:
            self.world = self.bpy.data.worlds.new("metasim_visual_world")
            self.world.use_nodes = True
        scene.world = self.world
        tree = self.world.node_tree
        has_hdri = values["hdri_path"] is not None
        if self._world_has_hdri != has_hdri:
            tree.nodes.clear()
            background = tree.nodes.new("ShaderNodeBackground")
            background.name = "Background"
            out = tree.nodes.new("ShaderNodeOutputWorld")
            tree.links.new(background.outputs["Background"], out.inputs["Surface"])
            if has_hdri:
                tex = tree.nodes.new("ShaderNodeTexEnvironment")
                tex.name = "Environment"
                mapping = tree.nodes.new("ShaderNodeMapping")
                mapping.name = "Rotation"
                coord = tree.nodes.new("ShaderNodeTexCoord")
                tree.links.new(coord.outputs["Generated"], mapping.inputs["Vector"])
                tree.links.new(mapping.outputs["Vector"], tex.inputs["Vector"])
                tree.links.new(tex.outputs["Color"], background.inputs["Color"])
            self._world_has_hdri = has_hdri
        tree.nodes["Background"].inputs["Strength"].default_value = values["strength"]
        tree.nodes["Background"].inputs["Color"].default_value = (*values["color"], 1.0)
        if has_hdri:
            tree.nodes["Environment"].image = self._image(values["hdri_path"], "Linear Rec.709")
            tree.nodes["Rotation"].inputs["Rotation"].default_value[2] = values["rotation"]

    def apply(self, recipe, env_ids):
        """Apply preflighted values relative to binding-time baselines."""
        from mathutils import Vector

        for name, values in recipe.materials.items():
            self._material(name, values)
        for name, values in recipe.lights.items():
            obj, energy, pos = self.lights[name]
            obj.data.energy = energy * values["intensity_scale"]
            obj.data.color = values["color"]
            obj.location = pos + Vector(values["position_delta"])
        for name, values in recipe.cameras.items():
            cfg = self.cameras[name]
            pos, target = _camera_pose(cfg, values)
            obj = self.handler._camera_objs[name]
            obj.location = pos
            obj.rotation_mode = "QUATERNION"
            obj.rotation_quaternion = (Vector(target) - Vector(pos)).to_track_quat("-Z", "Y")
            obj.data.lens = cfg.focal_length * values["focal_scale"]
            f_stop, focus_distance = _focus(values, pos, target)
            obj.data.dof.use_dof = f_stop > 0
            if f_stop > 0:
                obj.data.dof.focus_object = None
                obj.data.dof.aperture_fstop = f_stop
                obj.data.dof.focus_distance = focus_distance
        if recipe.environment is not None:
            self._environment(recipe.environment)
        # Removing only our unused image datablocks bounds memory even when replaying
        # recipes from an unbounded stream of different texture paths. A small pool of
        # recently used ones survives, so alternating recipes do not reload and decode
        # the same 4K HDRI every time.
        idle = [key for key, image in self.images.items() if image.users == 0]
        for key in idle[: max(0, len(idle) - IDLE_IMAGE_POOL)]:
            self.bpy.data.images.remove(self.images.pop(key))
        self.bpy.context.view_layer.update()

    def invalidate(self):
        """Mark both the renderer frame and public state cache dirty."""
        self.handler._render_dirty = True
        self.handler._invalidate_state_caches()

    def render(self):
        """Render cameras without advancing simulation."""
        self.handler.refresh_render()


class IsaacSimVisualAdapter:
    """Author USD Preview Surface materials and edit Isaac Lab camera sensors."""

    def __init__(self, handler):
        import omni.usd
        from pxr import UsdGeom, UsdLux

        self.handler = handler
        self.stage = omni.usd.get_context().get_stage()
        if self.stage is None:
            raise RuntimeError("Launch Isaac Sim before binding visual randomization")
        self.cameras = {c.name: copy.deepcopy(c) for c in handler.cameras}
        self.lights = {}
        # prim path -> the authored st primvar (values, interpolation) or None, saved before box projection
        self._box_uv_prims = {}
        self.scope = "/World/MetaSimVisual"
        if self.stage.GetPrimAtPath(self.scope):
            raise RuntimeError("Only one visual adapter may own an Isaac Sim stage")
        # Claim the scope at bind time; the guard above is what makes a second adapter fail.
        UsdGeom.Scope.Define(self.stage, self.scope)
        self.scene_domes = []
        for prim in self.stage.Traverse():
            if prim.IsA(UsdLux.DomeLight):
                # Scenario domes and the skies shipped inside interior scenes: muted while a
                # recipe environment is active, never edited.
                self.scene_domes.append(prim)
            elif prim.HasAPI(UsdLux.LightAPI):
                self.lights[str(prim.GetPath())] = (
                    UsdLux.LightAPI(prim).GetIntensityAttr().Get(),
                    UsdGeom.Xformable(prim).GetLocalTransformation(),
                )

    def _roots(self, name, env_ids):
        # Only handler-declared objects/robots and explicit ground are targetable.
        names = {obj.name for obj in [*self.handler.objects, *self.handler.robots]}
        if name == "ground":
            paths = ["/World/ground"]
            if set(env_ids) != set(range(self.handler.num_envs)):
                raise ValueError("Ground is shared; apply its material to all environments")
        elif name in names:
            paths = [f"/World/envs/env_{i}/{name}" for i in env_ids]
        else:
            raise ValueError(f"Unknown Isaac Sim material target: {name!r}")
        roots = [self.stage.GetPrimAtPath(path) for path in paths]
        if any(not p.IsValid() for p in roots):
            raise ValueError(f"Material target {name!r} has missing prims: {paths}")
        return roots

    def _geometry(self, root):
        from pxr import Usd, UsdGeom

        prims = list(Usd.PrimRange(root))
        if any(p.IsInstance() or p.IsInstanceProxy() for p in prims):
            raise NotImplementedError("Disable USD instancing on material targets before visual augmentation")
        return [p for p in prims if p.IsA(UsdGeom.Gprim)]

    def _light(self, name):
        from pxr import UsdLux

        path = f"/World/{name}"
        if path not in self.lights:
            if self.stage.GetPrimAtPath(path).IsA(UsdLux.DomeLight):
                raise ValueError("Use environment for dome lighting")
            raise ValueError(f"Unknown Isaac Sim light: {name!r}; use its explicit scenario name")
        return self.stage.GetPrimAtPath(path)

    def _camera_prims(self, name, env_ids):
        from pxr import UsdGeom

        prims = [self.stage.GetPrimAtPath(f"/World/envs/env_{i}/{name}") for i in env_ids]
        if any(not p.IsValid() or not p.IsA(UsdGeom.Camera) for p in prims):
            raise ValueError(f"Camera {name!r} has no USD camera prim in every selected environment")
        return prims

    def validate(self, recipe, env_ids):
        """Resolve per-env geometry, UVs, lights and camera capabilities upfront."""
        gs_scene = getattr(self.handler.scenario, "gs_scene", None)
        if recipe.cameras and gs_scene is not None and gs_scene.with_gs_background:
            raise NotImplementedError("Camera augmentation with Gaussian-splat background compositing is unsupported")
        from pxr import UsdGeom

        for name, values in recipe.materials.items():
            for root in self._roots(name, env_ids):
                geometry = self._geometry(root)
                if not geometry:
                    raise ValueError(f"Material target {name!r} has no render geometry")
                if not any(values["textures"].values()):
                    continue
                for prim in geometry:
                    if values.get("uv_projection", "mesh") == "box":
                        if not prim.IsA(UsdGeom.Mesh):
                            raise NotImplementedError(f"Box UV projection needs a UsdGeom.Mesh, got {prim.GetPath()}")
                        continue
                    if prim.GetPath() in self._box_uv_prims:
                        authored = self._box_uv_prims[prim.GetPath()] is not None
                    else:
                        uv = UsdGeom.PrimvarsAPI(prim).FindPrimvarWithInheritance("st")
                        authored = bool(uv) and uv.HasValue()
                    if not authored:
                        raise ValueError(f"Texture target {prim.GetPath()} requires an authored st UV primvar")
        for name in recipe.lights:
            self._light(name)
        for name, values in recipe.cameras.items():
            cfg = self.cameras.get(name)
            if cfg is None or name not in self.handler.scene.sensors:
                raise ValueError(f"Unknown camera: {name!r}")
            if cfg.mount_to is not None or cfg.intrinsic is not None:
                raise NotImplementedError("Visual camera jitter requires an unmounted aperture-based pinhole camera")
            _camera_pose(cfg, values)
            self._camera_prims(name, env_ids)

    def _record_uvs(self, prim):
        from pxr import UsdGeom

        if self._box_uv_prims.get(prim.GetPath(), False) is not False:
            return
        # Record before authoring any prim of the batch: cloned envs inherit env_0, so a
        # sibling would otherwise read env_0's fresh box UVs as its own authored ones.
        authored = UsdGeom.PrimvarsAPI(prim).GetPrimvar("st")
        self._box_uv_prims[prim.GetPath()] = (
            (authored.Get(), authored.GetInterpolation(), authored.GetIndices() if authored.IsIndexed() else None)
            if authored and authored.HasValue()
            else None
        )

    def _author_box_uvs(self, prim):
        from pxr import Gf, Sdf, Usd, UsdGeom, Vt

        mesh = UsdGeom.Mesh(prim)
        points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
        matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        scale = np.asarray([Gf.Vec3d(*matrix.GetRow3(i)).GetLength() for i in range(3)])
        points = points * scale
        uvs = []
        offset = 0
        for count in counts:
            face = indices[offset : offset + int(count)]
            normal = np.cross(points[face[1]] - points[face[0]], points[face[2]] - points[face[0]])
            axis = int(np.argmax(np.abs(normal)))
            uvs.extend(Gf.Vec2f(*(float(v) for v in _box_uv(points[vertex], axis))) for vertex in face)
            offset += int(count)
        primvar = UsdGeom.PrimvarsAPI(prim).CreatePrimvar(
            "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying
        )
        primvar.Set(Vt.Vec2fArray(uvs))
        # Our values are one per face-vertex; an imported mesh's index array must not remap them.
        primvar.BlockIndices()

    def _pin_uvs(self, prim):
        """Author a prim's current st locally so an inherited env_0 edit cannot change it."""
        from pxr import Sdf, UsdGeom

        current = UsdGeom.PrimvarsAPI(prim).GetPrimvar("st")
        if not current or not current.HasValue():
            return
        values, interpolation = current.Get(), current.GetInterpolation()
        indices = current.GetIndices() if current.IsIndexed() else None
        pinned = UsdGeom.PrimvarsAPI(prim).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, interpolation)
        pinned.Set(values)
        if indices is not None:
            pinned.SetIndices(indices)

    def _restore_uvs(self, prim):
        """Put back the authored st primvar (or remove ours) after box projection."""
        from pxr import Sdf, UsdGeom

        saved = self._box_uv_prims.pop(prim.GetPath())
        api = UsdGeom.PrimvarsAPI(prim)
        if saved is None:
            api.RemovePrimvar("st")
            return
        values, interpolation, indices = saved
        primvar = api.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, interpolation)
        primvar.Set(values)
        if indices is not None:
            primvar.SetIndices(indices)
        else:
            primvar.BlockIndices()

    def _material(self, path, values):
        from pxr import Gf, Sdf, UsdShade

        material = UsdShade.Material.Define(self.stage, path)
        shader = UsdShade.Shader.Define(self.stage, path + "/Surface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        shader.CreateInput("useSpecularWorkflow", Sdf.ValueTypeNames.Int).Set(0)
        uv = UsdShade.Shader.Define(self.stage, path + "/UV")
        uv.CreateIdAttr("UsdPrimvarReader_float2")
        uv.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        uv.CreateOutput("result", Sdf.ValueTypeNames.Float2)
        transform = UsdShade.Shader.Define(self.stage, path + "/UVTransform")
        transform.CreateIdAttr("UsdTransform2d")
        transform.CreateInput("in", Sdf.ValueTypeNames.Float2).ConnectToSource(uv.ConnectableAPI(), "result")
        transform.CreateInput("scale", Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(*values.get("uv_scale", (1, 1))))
        transform.CreateInput("rotation", Sdf.ValueTypeNames.Float).Set(math.degrees(values.get("uv_rotation", 0)))
        transform.CreateOutput("result", Sdf.ValueTypeNames.Float2)
        shader.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(values.get("ior", 1.5))
        sockets = {
            "base_color": ("diffuseColor", Sdf.ValueTypeNames.Color3f, Gf.Vec3f(*values["color"]), "rgb"),
            "roughness": ("roughness", Sdf.ValueTypeNames.Float, values["roughness"], "r"),
            "metallic": ("metallic", Sdf.ValueTypeNames.Float, values["metallic"], "r"),
            "normal": ("normal", Sdf.ValueTypeNames.Normal3f, Gf.Vec3f(0, 0, 1), "rgb"),
        }
        for socket, value_type, fallback, _ in sockets.values():
            inp = shader.CreateInput(socket, value_type)
            inp.DisconnectSource()
            inp.Set(fallback)
        textures = values["textures"]
        for channel in (*sockets, "orm"):
            tex_path = path + "/Tex_" + channel
            asset = textures.get(channel)
            if asset is None:
                if self.stage.GetPrimAtPath(tex_path):
                    self.stage.RemovePrim(tex_path)
                continue
            tex = UsdShade.Shader.Define(self.stage, tex_path)
            tex.CreateIdAttr("UsdUVTexture")
            tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(asset))
            tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set(
                "sRGB" if channel == "base_color" else "raw"
            )
            tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(transform.ConnectableAPI(), "result")
            tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
            tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
            if channel == "normal":
                tex.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(2, 2, 2, 1))
                tex.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(-1, -1, -1, 0))
            if channel == "orm":
                # glTF packing: G roughness, B metallic (R occlusion is not used by Preview Surface).
                for output, socket in (("g", "roughness"), ("b", "metallic")):
                    tex.CreateOutput(output, Sdf.ValueTypeNames.Float)
                    shader.GetInput(socket).ConnectToSource(tex.ConnectableAPI(), output)
                continue
            socket, _, _, output = sockets[channel]
            tex.CreateOutput(output, Sdf.ValueTypeNames.Float if output == "r" else Sdf.ValueTypeNames.Float3)
            shader.GetInput(socket).ConnectToSource(tex.ConnectableAPI(), output)
        return material

    def _environment(self, values):
        from pxr import Gf, Sdf, UsdGeom, UsdLux

        for prim in self.scene_domes:
            if prim.IsValid():
                UsdLux.LightAPI(prim).CreateIntensityAttr().Set(0.0)
        dome = UsdLux.DomeLight.Define(self.stage, self.scope + "/Environment")
        dome.CreateIntensityAttr(500.0 * values["strength"])
        dome.CreateExposureAttr(0.0)
        dome.CreateColorAttr(Gf.Vec3f(1) if values["hdri_path"] else Gf.Vec3f(*values["color"]))
        dome.CreateTextureFileAttr(Sdf.AssetPath(values["hdri_path"] or ""))
        dome.CreateTextureFormatAttr("latlong")
        transform = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), math.degrees(values["rotation"])))
        UsdGeom.Xformable(dome.GetPrim()).MakeMatrixXform().Set(transform)

    def apply(self, recipe, env_ids):
        """Edit render properties and sensors, never rigid-body poses."""
        import torch
        from pxr import Gf, UsdGeom, UsdLux, UsdShade

        UsdGeom.Scope.Define(self.stage, self.scope)
        excluded = [i for i in range(self.handler.num_envs) if i not in env_ids]
        for name, values in recipe.materials.items():
            textured_box = any(values["textures"].values()) and values.get("uv_projection", "mesh") == "box"
            if 0 in env_ids and excluded and name != "ground":
                # Cloned envs inherit env_0's prims, so pin the other envs to their current
                # material before env_0's binding changes underneath them.
                for root in self._roots(name, excluded):
                    for prim in self._geometry(root):
                        binding = UsdShade.MaterialBindingAPI.Apply(prim)
                        current = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
                        if current:
                            binding.Bind(current, UsdShade.Tokens.strongerThanDescendants)
                        else:
                            binding.UnbindDirectBinding()
                        if textured_box:
                            self._pin_uvs(prim)
            roots = self._roots(name, env_ids)
            if textured_box:
                for root in roots:
                    for prim in self._geometry(root):
                        self._record_uvs(prim)
            for root in roots:
                material = self._material(self.scope + "/M_" + _key(str(root.GetPath())), values)
                for prim in self._geometry(root):
                    if textured_box:
                        self._author_box_uvs(prim)
                    elif prim.GetPath() in self._box_uv_prims:
                        self._restore_uvs(prim)
                    # Strong parent binding also overrides imported face-subset materials.
                    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, UsdShade.Tokens.strongerThanDescendants)
        for name, values in recipe.lights.items():
            prim = self._light(name)
            intensity, baseline = self.lights[str(prim.GetPath())]
            light = UsdLux.LightAPI(prim)
            light.GetIntensityAttr().Set(intensity * values["intensity_scale"])
            light.GetColorAttr().Set(Gf.Vec3f(*values["color"]))
            matrix = Gf.Matrix4d(baseline)
            matrix.SetTranslateOnly(baseline.ExtractTranslation() + Gf.Vec3d(*values["position_delta"]))
            UsdGeom.Xformable(prim).MakeMatrixXform().Set(matrix)
        for name, values in recipe.cameras.items():
            cfg = self.cameras[name]
            sensor = self.handler.scene.sensors[name]
            pos, target = _camera_pose(cfg, values)
            device = self.handler.device
            # The handler owns camera poses: it re-asserts them on every state push.
            self.handler.set_camera_pose(name, pos, target, env_ids=env_ids)
            intrinsic = torch.tensor(cfg.intrinsics, device=device).repeat(len(env_ids), 1, 1)
            intrinsic[:, 0, 0] *= values["focal_scale"]
            intrinsic[:, 1, 1] *= values["focal_scale"]
            # Keep the authored physical focal length: without it Isaac Lab rewrites focalLength
            # to 1/width, which leaves the thin-lens aperture (focalLength / fStop) almost closed.
            sensor.set_intrinsic_matrices(
                intrinsic, focal_length=float(cfg.focal_length) * values["focal_scale"], env_ids=env_ids
            )
            f_stop, focus_distance = _focus(values, pos, target)
            for prim in self._camera_prims(name, env_ids):
                usd_camera = UsdGeom.Camera(prim)
                usd_camera.CreateFStopAttr().Set(f_stop * ISAAC_FSTOP_SCALE)
                usd_camera.CreateFocusDistanceAttr().Set(focus_distance if f_stop > 0 else float(cfg.focus_distance))
        if recipe.environment is not None:
            self._environment(recipe.environment)

    def invalidate(self):
        """Invalidate image freshness as well as the public state cache."""
        self.handler._render_current = False
        self.handler._visual_refresh_pending = True
        self.handler._invalidate_state_caches()

    def render(self):
        """Refresh with the handler's configured path-tracing sample budget."""
        self.handler.refresh_render()
