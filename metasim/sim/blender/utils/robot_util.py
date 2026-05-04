"""URDF loading + forward kinematics for the Blender backend.

Approach (mirrors the rerun visualizer in ``metasim/utils/rerun/rerun_util.py``):

* Parse the URDF directly with ``xml.etree`` — no yourdfpy. The roboverse
  URDFs use ``package://meshes/...`` where the segment after ``package://`` is
  *the start of a relative path under the URDF dir*, not a ROS package name.
  Our own ``resolve_mesh_path`` tries a fixed list of candidates and picks
  the first that exists.

* Load each visual mesh with ``trimesh`` and build a Blender mesh directly
  via ``bpy.data.meshes.new`` + ``from_pydata``. We do NOT round-trip through
  glb — Blender's gltf importer applies a Y-up→Z-up 90° rotation, which double-
  rotates URDF meshes that are already authored in Z-up.

* Express the kinematic tree with Blender parent-child empties:
  ``<robot>``  (empty, world transform from root state) →
  ``<robot>/<link>``  (empty, local = link transform in robot frame) →
  ``<robot>/<link>/visual_i``  (mesh, local = visual.origin in link frame).
  Each FK update writes one ``matrix_local`` per link; visual meshes follow.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import bpy
import numpy as np
import trimesh
from loguru import logger as log
from mathutils import Matrix


def _make_transform(xyz: list[float], rpy: list[float]) -> np.ndarray:
    """Build a 4x4 homogeneous transform from URDF xyz + rpy (ZYX intrinsic)."""
    T = np.eye(4)
    T[:3, 3] = xyz
    cr, sr = np.cos(rpy[0]), np.sin(rpy[0])
    cp, sp = np.cos(rpy[1]), np.sin(rpy[1])
    cy, sy = np.cos(rpy[2]), np.sin(rpy[2])
    T[0, 0] = cy * cp
    T[0, 1] = cy * sp * sr - sy * cr
    T[0, 2] = cy * sp * cr + sy * sr
    T[1, 0] = sy * cp
    T[1, 1] = sy * sp * sr + cy * cr
    T[1, 2] = sy * sp * cr - cy * sr
    T[2, 0] = -sp
    T[2, 1] = cp * sr
    T[2, 2] = cp * cr
    return T


def _axis_angle_to_transform(axis: np.ndarray, angle: float) -> np.ndarray:
    a = np.asarray(axis, dtype=float)
    a = a / (np.linalg.norm(a) + 1e-12)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    R = np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)
    T = np.eye(4)
    T[:3, :3] = R
    return T


def resolve_mesh_path(filename: str, urdf_dir: Path) -> str | None:
    """Return an existing absolute path for ``filename`` or None.

    Tries the URDF dir + many common subfolders. ``package://X/Y`` is treated
    as either ``urdf_dir/X/Y`` or ``urdf_dir/Y`` (the second is the roboverse
    convention where ``X`` is the actual first directory, not a package name).
    """
    if os.path.isabs(filename) and os.path.exists(filename):
        return filename

    candidates: list[Path] = []
    if filename.startswith("package://"):
        rel = filename[len("package://") :]
        candidates += [urdf_dir / rel, urdf_dir.parent / rel, urdf_dir.parent.parent / rel]
        head, sep, tail = rel.partition("/")
        if sep:
            candidates.append(urdf_dir / tail)
    elif filename.startswith("file://"):
        candidates.append(Path(filename[len("file://") :]))
    else:
        base = Path(filename).name
        candidates += [
            urdf_dir / filename,
            urdf_dir / "meshes" / filename,
            urdf_dir / "meshes" / base,
            urdf_dir / "meshes" / "visual" / filename,
            urdf_dir / "meshes" / "visual" / base,
            urdf_dir / "visual" / filename,
            urdf_dir / "visual" / base,
            urdf_dir.parent / "meshes" / filename,
            urdf_dir.parent / "meshes" / base,
        ]
    try:
        candidates.append((urdf_dir / filename).resolve())
    except Exception:
        pass

    for cand in candidates:
        try:
            resolved = cand.resolve() if not cand.is_absolute() else cand
            if resolved.exists():
                return str(resolved)
        except Exception:
            continue
    return None


@dataclass
class _Visual:
    type: str  # "mesh" / "box" / "sphere" / "cylinder"
    origin: np.ndarray  # 4x4
    color: list[float]  # RGBA (0-1)
    filename: str | None = None
    scale: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    size: list[float] | None = None  # box
    radius: float | None = None  # sphere / cylinder
    length: float | None = None  # cylinder


@dataclass
class _Joint:
    parent: str
    child: str
    origin: np.ndarray  # 4x4
    type: str
    axis: list[float]


@dataclass
class URDFData:
    links: dict[str, list[_Visual]]
    joints: dict[str, _Joint]
    parent_map: dict[str, tuple[str, str]]  # child -> (parent, joint)
    root_link: str
    urdf_dir: Path


def parse_urdf(urdf_path: str) -> URDFData:
    urdf_dir = Path(urdf_path).parent
    root = ET.parse(urdf_path).getroot()

    links: dict[str, list[_Visual]] = {}
    joints: dict[str, _Joint] = {}
    parent_map: dict[str, tuple[str, str]] = {}
    all_links: set[str] = set()
    child_links: set[str] = set()

    for link_el in root.findall(".//link"):
        link_name = link_el.get("name", "unnamed")
        all_links.add(link_name)
        visuals: list[_Visual] = []
        for vis_el in link_el.findall("visual"):
            origin = np.eye(4)
            o = vis_el.find("origin")
            if o is not None:
                xyz = [float(x) for x in o.get("xyz", "0 0 0").split()]
                rpy = [float(x) for x in o.get("rpy", "0 0 0").split()]
                origin = _make_transform(xyz, rpy)

            mat_el = vis_el.find("material")
            color = [0.8, 0.8, 0.8, 1.0]
            if mat_el is not None:
                ce = mat_el.find("color")
                if ce is not None:
                    color = [float(x) for x in ce.get("rgba", "0.8 0.8 0.8 1").split()]

            geom = vis_el.find("geometry")
            if geom is None:
                continue
            mesh_el = geom.find("mesh")
            box_el = geom.find("box")
            sph_el = geom.find("sphere")
            cyl_el = geom.find("cylinder")
            if mesh_el is not None:
                fn = mesh_el.get("filename", "")
                resolved = resolve_mesh_path(fn, urdf_dir)
                if resolved:
                    sc = [float(x) for x in mesh_el.get("scale", "1 1 1").split()]
                    visuals.append(_Visual(type="mesh", origin=origin, color=color, filename=resolved, scale=sc))
            elif box_el is not None:
                size = [float(x) for x in box_el.get("size", "1 1 1").split()]
                visuals.append(_Visual(type="box", origin=origin, color=color, size=size))
            elif sph_el is not None:
                visuals.append(
                    _Visual(type="sphere", origin=origin, color=color, radius=float(sph_el.get("radius", "1")))
                )
            elif cyl_el is not None:
                visuals.append(
                    _Visual(
                        type="cylinder",
                        origin=origin,
                        color=color,
                        radius=float(cyl_el.get("radius", "1")),
                        length=float(cyl_el.get("length", "1")),
                    )
                )
        links[link_name] = visuals

    for joint_el in root.findall(".//joint"):
        jname = joint_el.get("name", "unnamed")
        jtype = joint_el.get("type", "fixed")
        p = joint_el.find("parent")
        c = joint_el.find("child")
        if p is None or c is None:
            continue
        parent_link = p.get("link")
        child_link = c.get("link")

        o = joint_el.find("origin")
        if o is not None:
            xyz = [float(x) for x in o.get("xyz", "0 0 0").split()]
            rpy = [float(x) for x in o.get("rpy", "0 0 0").split()]
            j_origin = _make_transform(xyz, rpy)
        else:
            j_origin = np.eye(4)
        ax = joint_el.find("axis")
        axis = [float(x) for x in ax.get("xyz", "0 0 1").split()] if ax is not None else [0, 0, 1]

        joints[jname] = _Joint(parent=parent_link, child=child_link, origin=j_origin, type=jtype, axis=axis)
        parent_map[child_link] = (parent_link, jname)
        child_links.add(child_link)

    roots = all_links - child_links
    root_link = next(iter(roots)) if roots else next(iter(all_links))
    return URDFData(links=links, joints=joints, parent_map=parent_map, root_link=root_link, urdf_dir=urdf_dir)


def compute_link_local_transforms(urdf: URDFData, dof_pos: dict[str, float] | None) -> dict[str, np.ndarray]:
    """Return link_name -> 4x4 transform in the robot's root frame."""
    dof_pos = dof_pos or {}
    cache: dict[str, np.ndarray] = {}

    def get(link: str) -> np.ndarray:
        if link in cache:
            return cache[link]
        if link == urdf.root_link or link not in urdf.parent_map:
            cache[link] = np.eye(4)
            return cache[link]
        parent, jname = urdf.parent_map[link]
        T_parent = get(parent)
        j = urdf.joints[jname]
        q = float(dof_pos.get(jname, 0.0))
        if j.type in ("revolute", "continuous"):
            T_motion = _axis_angle_to_transform(j.axis, q)
        elif j.type == "prismatic":
            T_motion = np.eye(4)
            T_motion[:3, 3] = np.array(j.axis) * q
        else:
            T_motion = np.eye(4)
        cache[link] = T_parent @ j.origin @ T_motion
        return cache[link]

    for ln in urdf.links:
        get(ln)
    return cache


def _extract_visual_info(visual) -> dict:
    """Pull rgba / metallic / roughness / texture image / UVs from a trimesh visual.

    trimesh's DAE loader sometimes stores ``baseColorFactor`` as 0-255 ints even
    though the glTF spec defines it as 0-1; we sniff the magnitude and rescale.
    Returns a dict with keys ``rgba``, ``metallic``, ``roughness``, ``texture``
    (PIL.Image or None) and ``uv`` (np.ndarray or None).
    """
    info = {
        "rgba": (0.8, 0.8, 0.8, 1.0),
        "metallic": 0.0,
        "roughness": 0.8,
        "texture": None,
        "uv": None,
    }
    if visual is None:
        return info

    def _to_unit(arr) -> np.ndarray:
        a = np.asarray(arr, dtype=np.float64)
        if a.size and a.max() > 1.5:
            a = a / 255.0
        return a

    mat = getattr(visual, "material", None)
    base = None
    if mat is not None:
        base = getattr(mat, "baseColorFactor", None)
        if base is None:
            base = getattr(mat, "main_color", None)
        if base is None and hasattr(mat, "diffuse"):
            base = mat.diffuse
        if base is not None:
            b = _to_unit(base)
            if b.size >= 3:
                info["rgba"] = (
                    float(b[0]),
                    float(b[1]),
                    float(b[2]),
                    float(b[3]) if b.size >= 4 else 1.0,
                )
        info["metallic"] = float(getattr(mat, "metallicFactor", info["metallic"]) or 0.0)
        info["roughness"] = float(getattr(mat, "roughnessFactor", info["roughness"]) or 0.8)
        # Image-based textures (SimpleMaterial.image, PBRMaterial.baseColorTexture).
        for attr in ("baseColorTexture", "image"):
            img = getattr(mat, attr, None)
            if img is not None:
                info["texture"] = img
                break
    elif hasattr(visual, "main_color") and visual.main_color is not None:
        c = _to_unit(visual.main_color)
        info["rgba"] = (float(c[0]), float(c[1]), float(c[2]), float(c[3]) if c.size >= 4 else 1.0)

    if hasattr(visual, "uv") and visual.uv is not None:
        info["uv"] = np.asarray(visual.uv, dtype=np.float64)
    return info


def _pil_to_bpy_image(pil_img, name: str) -> bpy.types.Image | None:
    """Copy a PIL image into a fresh bpy.types.Image (RGBA, float32)."""
    try:
        arr = np.asarray(pil_img)
    except Exception as e:
        log.warning(f"failed to convert texture to array: {e}")
        return None
    if arr.ndim == 2:
        arr = np.dstack([arr, arr, arr])
    if arr.shape[-1] == 3:
        alpha = np.full(arr.shape[:2] + (1,), 255, dtype=arr.dtype)
        arr = np.concatenate([arr, alpha], axis=-1)
    h, w = arr.shape[:2]
    img = bpy.data.images.new(name=name, width=w, height=h, alpha=True)
    # Blender stores pixels bottom-up, in 0-1 floats, RGBA flat.
    flat = (np.flipud(arr).astype(np.float32) / 255.0).reshape(-1)
    img.pixels.foreach_set(flat)
    img.update()
    return img


def _build_bpy_mesh_from_arrays(
    name: str,
    verts: np.ndarray,
    faces: np.ndarray,
    rgba: tuple[float, float, float, float],
    metallic: float = 0.0,
    roughness: float = 0.8,
    uv: np.ndarray | None = None,
    texture_image=None,  # PIL.Image
) -> bpy.types.Object:
    """Build a Blender mesh + Principled-BSDF material; optionally wire up a UV-mapped texture."""
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts.tolist(), [], faces.tolist())
    me.update(calc_edges=True)
    obj = bpy.data.objects.new(name, me)
    bpy.context.collection.objects.link(obj)

    mat = bpy.data.materials.new(name=f"{name}_mat")
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes.get("Principled BSDF")
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = (rgba[0], rgba[1], rgba[2], rgba[3])
        if "Metallic" in bsdf.inputs:
            bsdf.inputs["Metallic"].default_value = metallic
        if "Roughness" in bsdf.inputs:
            bsdf.inputs["Roughness"].default_value = roughness

    # Texture: per-vertex UVs are mirrored onto per-loop UVs, then linked to BSDF.
    if uv is not None and texture_image is not None and bsdf is not None:
        bpy_img = _pil_to_bpy_image(texture_image, name=f"{name}_tex")
        if bpy_img is not None and len(uv) >= len(verts):
            uv_layer = me.uv_layers.new(name="UVMap")
            for poly in me.polygons:
                for li in range(poly.loop_start, poly.loop_start + poly.loop_total):
                    v_idx = me.loops[li].vertex_index
                    if v_idx < len(uv):
                        uv_layer.data[li].uv = (float(uv[v_idx][0]), float(uv[v_idx][1]))
            tex_node = nt.nodes.new("ShaderNodeTexImage")
            tex_node.image = bpy_img
            tex_node.location = (-300, 0)
            nt.links.new(tex_node.outputs["Color"], bsdf.inputs["Base Color"])

    obj.data.materials.append(mat)
    return obj


def _trimesh_to_bpy_objects(
    mesh_path: str,
    scale: list[float],
    name: str,
    fallback_rgba: tuple[float, float, float, float],
) -> list[bpy.types.Object]:
    """Load a mesh via trimesh and create one Blender object per sub-geometry.

    Multi-material DAEs (Franka, etc.) come back as a ``trimesh.Scene`` whose
    sub-geometries each carry their own ``PBRMaterial``. We preserve the
    per-region colour by emitting one Blender mesh per sub-geometry.

    Pure single-material files yield a single object.

    Returns a list of Blender objects (caller is responsible for parenting and
    setting ``pass_index`` / ``matrix_local``).
    """
    try:
        loaded = trimesh.load(mesh_path, skip_materials=False)
    except Exception as e:
        log.warning(f"trimesh failed to load {mesh_path}: {e}")
        return []

    scale_np = np.asarray(scale, dtype=np.float64) if scale != [1.0, 1.0, 1.0] else None
    geometries: list[tuple[str, trimesh.Trimesh]] = []
    if isinstance(loaded, trimesh.Scene):
        # ``dump(concatenate=False)`` walks the scene graph and returns each
        # sub-mesh with its world-space vertices baked in (and visuals/PBR
        # materials preserved).
        for j, geom in enumerate(loaded.dump(concatenate=False)):
            if hasattr(geom, "vertices") and len(geom.vertices) > 0:
                geometries.append((f"{name}_{j}", geom))
    elif loaded is not None and hasattr(loaded, "vertices") and len(loaded.vertices) > 0:
        geometries.append((name, loaded))

    if not geometries:
        log.warning(f"empty mesh: {mesh_path}")
        return []

    out: list[bpy.types.Object] = []
    for i, (sub_name, geom) in enumerate(geometries):
        verts = np.asarray(geom.vertices, dtype=np.float64)
        if scale_np is not None:
            verts = verts * scale_np
        faces = np.asarray(geom.faces, dtype=np.int64)
        info = _extract_visual_info(getattr(geom, "visual", None))
        # No material? Fall back to the URDF-supplied colour.
        if info["rgba"] == (0.8, 0.8, 0.8, 1.0) and getattr(geom, "visual", None) is None:
            info["rgba"] = fallback_rgba
        obj = _build_bpy_mesh_from_arrays(
            f"{name}__{i}" if len(geometries) > 1 else name,
            verts,
            faces,
            info["rgba"],
            info["metallic"],
            info["roughness"],
            uv=info["uv"],
            texture_image=info["texture"],
        )
        out.append(obj)
    return out


def _spawn_urdf_primitive(v, name: str) -> bpy.types.Object | None:
    """Build a Blender primitive (box/sphere/cylinder) for a URDF visual.

    Vertices live in the link-local frame; the visual.origin is applied
    later via ``matrix_local`` parenting.
    """
    if v.type == "box":
        sx, sy, sz = (float(v.size[0]), float(v.size[1]), float(v.size[2]))
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, 0.0))
        obj = bpy.context.object
        obj.scale = (sx, sy, sz)
    elif v.type == "sphere":
        bpy.ops.mesh.primitive_uv_sphere_add(radius=float(v.radius), location=(0.0, 0.0, 0.0))
        obj = bpy.context.object
    elif v.type == "cylinder":
        bpy.ops.mesh.primitive_cylinder_add(
            radius=float(v.radius),
            depth=float(v.length),
            location=(0.0, 0.0, 0.0),
        )
        obj = bpy.context.object
    else:
        return None

    obj.name = name
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    rgba = tuple(v.color) if v.color else (0.8, 0.8, 0.8, 1.0)
    mat = bpy.data.materials.new(name=f"{name}_mat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        rgba_full = (rgba[0], rgba[1], rgba[2], rgba[3] if len(rgba) >= 4 else 1.0)
        bsdf.inputs["Base Color"].default_value = rgba_full
    obj.data.materials.append(mat)
    return obj


@dataclass
class RobotVisualState:
    urdf: URDFData
    root_empty: bpy.types.Object  # parented to world, holds robot pose
    link_empties: dict[str, bpy.types.Object]  # link_name -> empty parented to root_empty
    visual_objs: list[bpy.types.Object]  # mesh objects parented to their link_empty
    pass_index: int


def load_robot_visuals(
    urdf_path: str,
    name_prefix: str,
    pass_index: int,
) -> RobotVisualState:
    """Build the full empty/mesh hierarchy for one robot."""
    urdf = parse_urdf(urdf_path)

    root_empty = bpy.data.objects.new(f"{name_prefix}", None)
    bpy.context.collection.objects.link(root_empty)

    link_empties: dict[str, bpy.types.Object] = {}
    for link_name in urdf.links:
        e = bpy.data.objects.new(f"{name_prefix}__{link_name}", None)
        bpy.context.collection.objects.link(e)
        e.parent = root_empty
        link_empties[link_name] = e

    visual_objs: list[bpy.types.Object] = []
    for link_name, visuals in urdf.links.items():
        link_empty = link_empties[link_name]
        for i, v in enumerate(visuals):
            sub_objs: list[bpy.types.Object] = []
            visual_name = f"{name_prefix}__{link_name}__visual_{i}"
            if v.type == "mesh":
                sub_objs = _trimesh_to_bpy_objects(
                    v.filename,
                    v.scale,
                    name=visual_name,
                    fallback_rgba=tuple(v.color) if v.color else (0.8, 0.8, 0.8, 1.0),
                )
            elif v.type in ("box", "sphere", "cylinder"):
                obj = _spawn_urdf_primitive(v, visual_name)
                if obj is not None:
                    sub_objs = [obj]
            else:
                continue

            visual_local = Matrix(v.origin.tolist())
            for obj in sub_objs:
                obj.parent = link_empty
                obj.matrix_local = visual_local
                obj.pass_index = pass_index
                visual_objs.append(obj)

    log.info(f"loaded {name_prefix}: {len(urdf.links)} links, {len(visual_objs)} visual meshes")
    return RobotVisualState(
        urdf=urdf,
        root_empty=root_empty,
        link_empties=link_empties,
        visual_objs=visual_objs,
        pass_index=pass_index,
    )


def update_robot_pose(
    robot: RobotVisualState,
    root_world: Matrix,
    dof_pos: dict[str, float] | None,
) -> None:
    """Set the robot's root transform and run FK to refresh link locals."""
    robot.root_empty.matrix_world = root_world
    link_locals = compute_link_local_transforms(robot.urdf, dof_pos)
    for link_name, T in link_locals.items():
        e = robot.link_empties.get(link_name)
        if e is not None:
            e.matrix_local = Matrix(T.tolist())
