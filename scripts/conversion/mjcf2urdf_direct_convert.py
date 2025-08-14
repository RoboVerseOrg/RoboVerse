#!/usr/bin/env python3

# MJCF 2 URDF Parser. Still in development.
# Mesh rendering is supported. Texture rendering for this version is not supported.
# MJCF ettiqutes: there needs to be at least 1 body in the model.
# Rendered URDF does not contains the following attributes in mujoco:
# - camera
# - sites
# - some more
# Rendered URDF only supports isaacgym/isaacsim/isaaclab for now. Mujoco URDF importing is not supported.

import os
import pathlib
import shutil
import sys
import xml.etree.ElementTree as ET
from xml.dom import minidom

import mujoco
import numpy as np
from mesh2obj import msh_to_obj


def parse_mjcf(path: str) -> mujoco.MjModel:
    """
    Wraps MuJoCo's XML parsing to return an MjModel.
    """
    try:
        mj = mujoco.MjModel.from_xml_path(path)
    except Exception as e:
        raise Exception(
            f"There is an error in the MJCF file. Please fix the MJCF file error first before doing the conversion: {e}"
        ) from e
    return mj


def parse_material(mj: mujoco.MjModel, i_g: int, xml_path: str) -> dict:
    """
    Parse material/texture information from MuJoCo geom.
    """
    mat_info = {}
    rgba = mj.geom_rgba[i_g]
    if rgba[3] > 0:
        mat_info["color"] = rgba[:4].tolist()
    mat_id = mj.geom_matid[i_g]
    if mat_id >= 0:
        mat_adr = mj.name_matadr[mat_id]
        if mat_id + 1 < mj.nmat:
            mat_end = mj.name_matadr[mat_id + 1]
            mat_name = mj.names[mat_adr:mat_end].decode("utf-8").replace("\x00", "")
        else:
            mat_name = mj.names[mat_adr:].decode("utf-8").split("\x00")[0]
        mat_info["material_name"] = mat_name
        if mj.ntex > 0:
            tex_id = 0
            tex_adr = mj.name_texadr[tex_id]
            if tex_id + 1 < mj.ntex:
                tex_end = mj.name_texadr[tex_id + 1]
                tex_name = mj.names[tex_adr:tex_end].decode("utf-8").replace("\x00", "")
            else:
                tex_name = mj.names[tex_adr:].decode("utf-8").split("\x00")[0]
            import xml.etree.ElementTree as ET

            tree = ET.parse(xml_path)
            root = tree.getroot()
            tex_elem = root.find(f".//texture[@name='{tex_name}']")
            if tex_elem is not None and "file" in tex_elem.attrib:
                tex_file = tex_elem.attrib["file"]
                mat_info["texture_name"] = tex_name
                mat_info["texture_file"] = tex_file
    return mat_info


def parse_link(mj: mujoco.MjModel, i_l: int, q_offset: int, dof_offset: int, scale: float, xml_path: str) -> tuple:
    """
    Extract link (body) info and associated elements (geoms) from MuJoCo.
    Returns:
      l_info (dict), j_info (dict)
    """
    l_info = dict()
    name_start = mj.name_bodyadr[i_l]
    if i_l + 1 < mj.nbody:
        name_end = mj.name_bodyadr[i_l + 1]
        l_info["name"] = mj.names[name_start:name_end].decode("utf-8").replace("\x00", "")
    else:
        l_info["name"] = mj.names[name_start:].decode("utf-8").split("\x00")[0]
    l_info["pos"] = (mj.body_pos[i_l] * scale).tolist()
    l_info["quat"] = mj.body_quat[i_l].tolist()
    l_info["inertial_pos"] = (mj.body_ipos[i_l] * scale).tolist()
    l_info["inertial_quat"] = mj.body_iquat[i_l].tolist()
    l_info["inertial_i"] = [
        mj.body_inertia[i_l, 0] * scale**5,
        mj.body_inertia[i_l, 1] * scale**5,
        mj.body_inertia[i_l, 2] * scale**5,
    ]
    l_info["inertial_mass"] = float(mj.body_mass[i_l]) * scale**3
    l_info["parent_idx"] = int(mj.body_parentid[i_l] - 1)
    l_info["parent_name"] = "world_link" if l_info["parent_idx"] < 0 else None
    geom_start = mj.body_geomadr[i_l]
    geom_num = mj.body_geomnum[i_l]
    l_info["geoms"] = []
    for g in range(geom_num):
        i_g = geom_start + g
        geom_info = {
            "type": mj.geom_type[i_g],
            "pos": (mj.geom_pos[i_g] * scale).tolist(),
            "quat": mj.geom_quat[i_g].tolist(),
            "size": (mj.geom_size[i_g] * scale).tolist(),
            "rgba": mj.geom_rgba[i_g].tolist(),
            "friction": mj.geom_friction[i_g].tolist(),
            "group": int(mj.geom_group[i_g]),
            "contype": int(mj.geom_contype[i_g]),
            "conaffinity": int(mj.geom_conaffinity[i_g]),
        }
        if mj.geom_type[i_g] == mujoco.mjtGeom.mjGEOM_MESH:
            mesh_id = mj.geom_dataid[i_g]
            if mesh_id >= 0:
                mesh_adr = mj.name_meshadr[mesh_id]
                if mesh_id + 1 < mj.nmesh:
                    mesh_end = mj.name_meshadr[mesh_id + 1]
                    mesh_name = mj.names[mesh_adr:mesh_end].decode("utf-8").replace("\x00", "")
                else:
                    mesh_name = mj.names[mesh_adr:].decode("utf-8").split("\x00")[0]
                geom_info["mesh_name"] = mesh_name
                geom_info["mesh_id"] = mesh_id
                geom_info["mesh_scale"] = (mj.mesh_scale[mesh_id] * scale).tolist()
        mat_id = mj.geom_matid[i_g]
        if mat_id >= 0:
            mat_info = parse_material(mj, i_g, xml_path)
            geom_info["material"] = mat_info
        l_info["geoms"].append(geom_info)
    j_info = dict()
    j_info["n_qs"] = 0
    j_info["n_dofs"] = 0
    j_info["type"] = "fixed"
    jnt_adr = mj.body_jntadr[i_l]
    jnt_num = mj.body_jntnum[i_l]
    if jnt_adr == -1 or jnt_num == 0:
        j_info["name"] = l_info["name"] + "_joint"
        j_info["pos"] = [0, 0, 0]
        j_info["quat"] = [1, 0, 0, 0]
        j_info["type"] = "fixed"
        j_info["parent"] = l_info["parent_name"] or "world_link"
    else:
        i_j = jnt_adr
        j_info["name"] = l_info["name"] + "_joint"
        j_info["pos"] = (mj.jnt_pos[i_j] * scale).tolist()
        j_info["quat"] = [1, 0, 0, 0]
        j_info["parent"] = l_info["parent_name"] or "world_link"
        mj_type = mj.jnt_type[i_j]
        mj_limit = mj.jnt_range[i_j] if mj.jnt_limited[i_j] else np.array([-np.inf, np.inf])
        if mj_type == mujoco.mjtJoint.mjJNT_FREE:
            j_info["type"] = "floating"
            j_info["n_qs"] = 7
            j_info["n_dofs"] = 6
        elif mj_type == mujoco.mjtJoint.mjJNT_HINGE:
            j_info["type"] = "revolute"
            j_info["n_qs"] = 1
            j_info["n_dofs"] = 1
            j_info["axis"] = mj.jnt_axis[i_j].tolist()
            j_info["limit"] = mj_limit.tolist()
        elif mj_type == mujoco.mjtJoint.mjJNT_SLIDE:
            j_info["type"] = "prismatic"
            j_info["n_qs"] = 1
            j_info["n_dofs"] = 1
            j_info["axis"] = mj.jnt_axis[i_j].tolist()
            j_info["limit"] = mj_limit.tolist()
        else:
            j_info["type"] = "fixed"
    return l_info, j_info


def process_body_recursive(
    mj: mujoco.MjModel,
    i_l: int,
    robot_links: list,
    robot_joints: list,
    scale: float,
    mesh_dir: str,
    out_urdf_path: str,
    xml_path: str,
) -> None:
    """
    Recursively process a body and its children, building up link/joint dicts.
    """
    l_info, j_info = parse_link(mj, i_l, q_offset=0, dof_offset=0, scale=scale, xml_path=xml_path)
    if l_info["name"] and l_info["name"] != "world_link":
        link_dict = {
            "name": l_info["name"],
            "inertial": None,
            "visuals": [],
            "collisions": [],
        }
        if l_info["inertial_mass"] > 1e-8:
            link_dict["inertial"] = {
                "origin_pos": l_info["inertial_pos"],
                "origin_quat": l_info["inertial_quat"],
                "mass": l_info["inertial_mass"],
                "inertia": [
                    [l_info["inertial_i"][0], 0.0, 0.0],
                    [0.0, l_info["inertial_i"][1], 0.0],
                    [0.0, 0.0, l_info["inertial_i"][2]],
                ],
            }
        for idx, geom in enumerate(l_info["geoms"]):
            geometry_dict = get_urdf_geometry_dict(geom, mj, mesh_dir, out_urdf_path, xml_path)
            if geometry_dict:
                visual_dict = {
                    "origin_pos": geom["pos"],
                    "origin_quat": geom["quat"],
                    "geometry": geometry_dict,
                    "material": None,
                }
                if "material" in geom:
                    mat_info = geom["material"]
                    material_dict = {"name": mat_info.get("material_name", f"mat_{l_info['name']}_{idx}")}
                    if "color" in mat_info:
                        material_dict["color"] = mat_info["color"]
                    visual_dict["material"] = material_dict
                link_dict["visuals"].append(visual_dict)
                collision_dict = {
                    "origin_pos": geom["pos"],
                    "origin_quat": geom["quat"],
                    "geometry": geometry_dict,
                }
                link_dict["collisions"].append(collision_dict)
        robot_links.append(link_dict)
        joint_dict = {
            "name": j_info["name"],
            "type": j_info["type"],
            "parent": j_info["parent"],
            "child": l_info["name"],
            "origin_pos": j_info["pos"],
            "origin_quat": j_info["quat"],
        }
        if j_info["type"] in ["revolute", "prismatic"]:
            joint_dict["axis"] = j_info.get("axis", [0, 0, 0])
            joint_dict["limit"] = j_info.get("limit", [-9999, 9999])
        robot_joints.append(joint_dict)
    for child_idx in range(1, mj.nbody):
        if mj.body_parentid[child_idx] == i_l:
            process_body_recursive(mj, child_idx, robot_links, robot_joints, scale, mesh_dir, out_urdf_path, xml_path)


def write_mtl_file(material_name: str, texture_path: str, filepath: str) -> None:
    """
    Write material library file (.mtl) to link texture with OBJ.
    """
    with open(filepath, "w") as f:
        f.write(f"newmtl {material_name}\n")
        f.write(f"map_Kd {texture_path}\n")


def write_obj_file(
    vertices: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray,
    texcoords: np.ndarray,
    scale: list,
    filepath: str,
    material_name: str,
) -> None:
    """
    Write mesh data to OBJ file with proper UV mapping and material reference.
    """
    mtl_filename = os.path.splitext(os.path.basename(filepath))[0] + ".mtl"
    mtl_filepath = os.path.join(os.path.dirname(filepath), mtl_filename)
    texture_filename = "texture_map.png"
    write_mtl_file(material_name, texture_filename, mtl_filepath)
    with open(filepath, "w") as f:
        f.write(f"mtllib {mtl_filename}\n")
        for v in vertices:
            scaled_v = [v[0] * scale[0], v[1] * scale[1], v[2] * scale[2]]
            f.write(f"v {scaled_v[0]} {scaled_v[1]} {scaled_v[2]}\n")
        for vt in texcoords:
            f.write(f"vt {vt[0]} {1.0 - vt[1]}\n")
        for vn in normals:
            f.write(f"vn {vn[0]} {vn[1]} {vn[2]}\n")
        f.write(f"usemtl {material_name}\n")
        for face in faces:
            v1, v2, v3 = face + 1
            f.write(f"f {v1}/{v1}/{v1} {v2}/{v2}/{v2} {v3}/{v3}/{v3}\n")


def get_mesh_and_texture_paths(xml_path: str) -> tuple:
    """
    Extract mesh and texture information from MJCF/XML file.
    Returns:
        mesh_files (dict): Mapping of mesh names to their file paths
        mesh_textures (dict): Mapping of mesh names to list of texture names they use
        texture_files (dict): Mapping of texture names to their file paths
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        mesh_files = {}  # mesh_name -> file path
        mesh_textures = {}  # mesh_name -> [texture_names]
        texture_files = {}  # texture_name -> file path

        # First pass: collect all meshes and textures
        asset_elem = root.find(".//asset")
        if asset_elem is not None:
            # Get all meshes
            for mesh in asset_elem.findall("mesh"):
                if "name" in mesh.attrib and "file" in mesh.attrib:
                    mesh_files[mesh.attrib["name"]] = mesh.attrib["file"]
                    mesh_textures[mesh.attrib["name"]] = []

            # Get all textures
            for texture in asset_elem.findall("texture"):
                if "name" in texture.attrib and "file" in texture.attrib:
                    texture_files[texture.attrib["name"]] = texture.attrib["file"]

        # Second pass: link meshes to textures through materials
        for geom in root.findall(".//geom"):
            if "mesh" in geom.attrib and "material" in geom.attrib:
                mesh_name = geom.attrib["mesh"]
                material_name = geom.attrib["material"]

                # Find material definition
                material = root.find(f".//material[@name='{material_name}']")
                if material is not None and "texture" in material.attrib:
                    texture_name = material.attrib["texture"]
                    if mesh_name in mesh_textures and texture_name not in mesh_textures[mesh_name]:
                        mesh_textures[mesh_name].append(texture_name)

        return mesh_files, mesh_textures, texture_files

    except Exception as e:
        print(f"Warning: Failed to parse XML file for mesh and texture paths: {e}")
        return {}, {}, {}


def get_urdf_geometry_dict(geom: dict, mj: mujoco.MjModel, mesh_dir: str, out_urdf_path: str, xml_path: str) -> dict:
    mj_type = geom["type"]
    if mj_type == mujoco.mjtGeom.mjGEOM_MESH:
        if "mesh_name" in geom:
            mesh_id = geom["mesh_id"]
            mesh_name = geom["mesh_name"]
            mesh_scale = geom.get("mesh_scale", [1.0, 1.0, 1.0])

            # Create destination mesh directory
            dst_mesh_dir = os.path.join(os.path.dirname(out_urdf_path), "meshes")
            os.makedirs(dst_mesh_dir, exist_ok=True)

            # Get mesh and texture paths
            mesh_files, mesh_textures, texture_files = get_mesh_and_texture_paths(xml_path)

            if mesh_name in mesh_files:
                # Get mesh file path
                rel_mesh_path = mesh_files[mesh_name]
                abs_mesh_path = os.path.join(os.path.dirname(xml_path), rel_mesh_path)

                if os.path.exists(abs_mesh_path):
                    try:
                        # Check file extension
                        file_ext = os.path.splitext(abs_mesh_path)[1].lower()

                        if file_ext == ".stl":
                            # For STL files, simply copy to destination
                            dst_filename = f"{mesh_name}.stl"
                            dst_path = os.path.join(dst_mesh_dir, dst_filename)
                            shutil.copy2(abs_mesh_path, dst_path)

                            return {
                                "type": "mesh",
                                "filename": f"package://meshes/{dst_filename}",
                                "scale": f"{mesh_scale[0]} {mesh_scale[1]} {mesh_scale[2]}",
                            }

                        elif file_ext == ".msh":
                            # Get XML filename without extension
                            xml_filename = os.path.splitext(os.path.basename(xml_path))[0]
                            obj_filename = f"{xml_filename}.obj"
                            obj_path = os.path.join(dst_mesh_dir, obj_filename)

                            # Convert .msh to .obj
                            obj_content = msh_to_obj(pathlib.Path(abs_mesh_path))

                            # Check if there's texture information in the XML
                            if "material" in geom and "texture_file" in geom["material"]:
                                texture_file = geom["material"]["texture_file"]
                                texture_name = geom["material"]["texture_name"]

                                # Create MTL file
                                mtl_filename = f"{xml_filename}.mtl"
                                mtl_path = os.path.join(dst_mesh_dir, mtl_filename)

                                # Copy texture file
                                src_texture_path = os.path.join(os.path.dirname(xml_path), texture_file)
                                dst_texture_path = os.path.join(dst_mesh_dir, os.path.basename(texture_file))
                                if os.path.exists(src_texture_path):
                                    shutil.copy2(src_texture_path, dst_texture_path)

                                    # Write MTL content
                                    with open(mtl_path, "w") as f:
                                        f.write(f"newmtl {texture_name}\n")
                                        f.write(f"map_Kd {os.path.basename(texture_file)}\n")

                                    # Write OBJ with MTL reference
                                    with open(obj_path, "w") as f:
                                        f.write(f"mtllib {mtl_filename}\n")
                                        f.write(f"usemtl {texture_name}\n")
                                        f.write(obj_content)
                            else:
                                # Write OBJ without texture
                                with open(obj_path, "w") as f:
                                    f.write(obj_content)
                            return {
                                "type": "mesh",
                                "filename": f"package://meshes/{obj_filename}",
                                "scale": f"{mesh_scale[0]} {mesh_scale[1]} {mesh_scale[2]}",
                            }
                        else:
                            print(f"Warning: Unsupported mesh format {file_ext} for {mesh_name}")
                            return {"type": "box", "size": [0.01, 0.01, 0.01]}

                    except Exception as e:
                        print(f"Warning: Failed to process mesh {mesh_name}: {e}")
                        return {"type": "box", "size": [0.01, 0.01, 0.01]}

    # For all non-mesh types or if mesh processing fails, return None
    return None


def quat_to_rpy(quat: list) -> list:
    """
    Convert quaternion [w, x, y, z] to RPY angles [roll, pitch, yaw].
    """
    w, x, y, z = quat
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)
    else:
        pitch = np.arcsin(sinp)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return [roll, pitch, yaw]


def build_urdf_xml(robot_name: str, links: list, joints: list) -> ET.Element:
    """
    Build an XML Element for the <robot> tag, containing <link> and <joint> tags.
    Returns the root <robot> element (as an xml.etree.ElementTree.Element).
    """
    robot_el = ET.Element("robot")
    robot_el.set("name", robot_name)
    for link in links:
        link_el = ET.SubElement(robot_el, "link")
        link_el.set("name", link["name"])
        if link["inertial"] is not None:
            inertial_el = ET.SubElement(link_el, "inertial")
            origin_el = ET.SubElement(inertial_el, "origin")
            xyz = link["inertial"]["origin_pos"]
            rpy = quat_to_rpy(link["inertial"]["origin_quat"])
            origin_el.set("xyz", f"{xyz[0]} {xyz[1]} {xyz[2]}")
            origin_el.set("rpy", f"{rpy[0]} {rpy[1]} {rpy[2]}")
            mass_el = ET.SubElement(inertial_el, "mass")
            mass_el.set("value", f"{link['inertial']['mass']}")
            ixx = link["inertial"]["inertia"][0][0]
            iyy = link["inertial"]["inertia"][1][1]
            izz = link["inertial"]["inertia"][2][2]
            ixy = ixz = iyz = 0.0
            inertia_el = ET.SubElement(inertial_el, "inertia")
            inertia_el.set("ixx", f"{ixx}")
            inertia_el.set("ixy", f"{ixy}")
            inertia_el.set("ixz", f"{ixz}")
            inertia_el.set("iyy", f"{iyy}")
            inertia_el.set("iyz", f"{iyz}")
            inertia_el.set("izz", f"{izz}")
        for viz in link["visuals"]:
            visual_el = ET.SubElement(link_el, "visual")
            origin_el = ET.SubElement(visual_el, "origin")
            xyz = viz["origin_pos"]
            rpy = quat_to_rpy(viz["origin_quat"])
            origin_el.set("xyz", f"{xyz[0]} {xyz[1]} {xyz[2]}")
            origin_el.set("rpy", f"{rpy[0]} {rpy[1]} {rpy[2]}")
            geometry_el = ET.SubElement(visual_el, "geometry")
            add_geometry_subnode(geometry_el, viz["geometry"])
            mat = viz["material"]
            if mat:
                material_el = ET.SubElement(visual_el, "material")
                material_el.set("name", mat["name"])
                if "color" in mat:
                    color_el = ET.SubElement(material_el, "color")
                    rgba = mat["color"]
                    color_el.set("rgba", f"{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}")
                if "texture" in mat:
                    texture_el = ET.SubElement(material_el, "texture")
                    texture_el.set("filename", mat["texture"])
        for col in link["collisions"]:
            collision_el = ET.SubElement(link_el, "collision")
            origin_el = ET.SubElement(collision_el, "origin")
            xyz = col["origin_pos"]
            rpy = quat_to_rpy(col["origin_quat"])
            origin_el.set("xyz", f"{xyz[0]} {xyz[1]} {xyz[2]}")
            origin_el.set("rpy", f"{rpy[0]} {rpy[1]} {rpy[2]}")
            geometry_el = ET.SubElement(collision_el, "geometry")
            add_geometry_subnode(geometry_el, col["geometry"])
    for joint in joints:
        joint_el = ET.SubElement(robot_el, "joint")
        joint_el.set("name", joint["name"])
        joint_el.set("type", joint["type"])
        origin_el = ET.SubElement(joint_el, "origin")
        xyz = joint["origin_pos"]
        rpy = quat_to_rpy(joint["origin_quat"])
        origin_el.set("xyz", f"{xyz[0]} {xyz[1]} {xyz[2]}")
        origin_el.set("rpy", f"{rpy[0]} {rpy[1]} {rpy[2]}")
        parent_el = ET.SubElement(joint_el, "parent")
        parent_el.set("link", joint["parent"])
        child_el = ET.SubElement(joint_el, "child")
        child_el.set("link", joint["child"])
        if joint["type"] in ["revolute", "prismatic"]:
            axis_el = ET.SubElement(joint_el, "axis")
            ax = joint["axis"]
            axis_el.set("xyz", f"{ax[0]} {ax[1]} {ax[2]}")
            limit_el = ET.SubElement(joint_el, "limit")
            lim = joint["limit"]
            limit_el.set("lower", f"{lim[0]}")
            limit_el.set("upper", f"{lim[1]}")
            limit_el.set("effort", "1000.0")
            limit_el.set("velocity", "1.0")
    return robot_el


def add_geometry_subnode(geometry_el: ET.Element, geometry_dict: dict) -> None:
    """
    Given a <geometry> element and a dictionary describing shape,
    add the appropriate child (<box>, <sphere>, <mesh>, etc.).
    """
    # Handle case where geometry_dict is None
    if geometry_dict is None:
        box_el = ET.SubElement(geometry_el, "box")
        box_el.set("size", "0.01 0.01 0.01")
        return

    # Handle case where type is missing
    gtype = geometry_dict.get("type", "box")

    if gtype == "box":
        size = geometry_dict.get("size", [0.01, 0.01, 0.01])
        box_el = ET.SubElement(geometry_el, "box")
        box_el.set("size", f"{size[0]} {size[1]} {size[2]}")
    elif gtype == "sphere":
        radius = geometry_dict.get("radius", 0.01)
        sphere_el = ET.SubElement(geometry_el, "sphere")
        sphere_el.set("radius", f"{radius}")
    elif gtype == "mesh":
        if "filename" in geometry_dict:
            mesh_el = ET.SubElement(geometry_el, "mesh")
            mesh_el.set("filename", geometry_dict["filename"])
            if "scale" in geometry_dict:
                mesh_el.set("scale", geometry_dict["scale"])
        else:
            # Fall back to default box if mesh filename is missing
            box_el = ET.SubElement(geometry_el, "box")
            box_el.set("size", "0.01 0.01 0.01")
    else:
        # Default fallback
        box_el = ET.SubElement(geometry_el, "box")
        box_el.set("size", "0.01 0.01 0.01")


def write_urdf_xml(robot_el: ET.Element, out_urdf_path: str) -> None:
    """
    Write the <robot> element tree to an XML file in a pretty-printed way.
    """
    rough_string = ET.tostring(robot_el, "utf-8")
    reparsed = minidom.parseString(rough_string)
    pretty_xml = reparsed.toprettyxml(indent="  ")
    with open(out_urdf_path, "w") as f:
        f.write(pretty_xml)


def mjcf_to_urdf(mjcf_path: str, out_urdf_path: str, scale: float = 1.0) -> None:
    """
    Convert an MJCF XML file to a URDF XML file without using urdfpy.
    """
    mj = parse_mjcf(mjcf_path)
    mesh_dir = os.path.dirname(os.path.abspath(mjcf_path))
    os.makedirs(os.path.dirname(os.path.abspath(out_urdf_path)), exist_ok=True)
    robot_links = []
    robot_joints = []
    world_link_dict = {
        "name": "world_link",
        "inertial": None,
        "visuals": [],
        "collisions": [],
    }
    robot_links.append(world_link_dict)
    process_body_recursive(mj, 1, robot_links, robot_joints, scale, mesh_dir, out_urdf_path, mjcf_path)
    robot_name = os.path.basename(mjcf_path).split(".")[0]
    robot_el = build_urdf_xml(robot_name, robot_links, robot_joints)
    write_urdf_xml(robot_el, out_urdf_path)
    print(f"URDF successfully written to: {out_urdf_path}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python mjcf2urdf.py <input_mjcf.xml> <output.urdf> [<scale=1.0>]")
        sys.exit(1)
    mjcf_file = sys.argv[1]
    urdf_file = sys.argv[2]
    scale = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    if not os.path.exists(mjcf_file):
        print(f"Error: MJCF file {mjcf_file} does not exist.")
        sys.exit(1)
    mjcf_to_urdf(mjcf_file, urdf_file, scale=scale)
