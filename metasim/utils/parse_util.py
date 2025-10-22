"""This file contains the utility functions for parsing URDF and MJCF files."""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from pathlib import Path


def extract_mesh_paths_from_urdf(urdf_file_path):
    """Extract all mesh file paths from a URDF XML file and convert them to absolute paths.

    Also recursively extracts material (.mtl) and texture files referenced by OBJ meshes.

    Args:
        urdf_file_path (str): Path to the URDF XML file

    Returns:
        list: List of absolute paths to all referenced mesh files and their dependencies
    """
    # Parse the XML file
    tree = ET.parse(urdf_file_path)
    root = tree.getroot()

    # Find all mesh elements
    mesh_elements = root.findall(".//mesh")

    # Extract the filename attributes and convert to absolute paths
    mesh_paths = []
    for mesh in mesh_elements:
        if "filename" in mesh.attrib:
            path = mesh.attrib["filename"]

            # Handle package:// URLs by replacing them with absolute paths
            if path.startswith("package://"):
                # Remove the "package://" prefix
                path = path[len("package://") :]

                # Assuming the package directory is relative to the URDF file location
                # You might need to adjust this based on your project structure
                urdf_dir = os.path.dirname(os.path.abspath(urdf_file_path))
                absolute_path = os.path.normpath(os.path.join(urdf_dir, path))
                mesh_paths.append(absolute_path)
            else:
                # If it's already an absolute path or a relative path, just normalize it
                if not os.path.isabs(path):
                    urdf_dir = os.path.dirname(os.path.abspath(urdf_file_path))
                    path = os.path.normpath(os.path.join(urdf_dir, path))
                mesh_paths.append(path)

    # Extract material and texture files from OBJ meshes
    additional_paths = []
    for mesh_path in mesh_paths:
        if mesh_path.endswith(".obj"):
            additional_paths.extend(_extract_obj_dependencies(mesh_path))

    mesh_paths.extend(additional_paths)
    return mesh_paths


def _extract_obj_dependencies(obj_file_path):
    """Extract material (.mtl) and texture file paths referenced by an OBJ file.

    Returns paths even if files don't exist locally - the download system will handle it.

    Args:
        obj_file_path (str): Path to the OBJ file

    Returns:
        list: List of absolute paths to material and texture files
    """
    dependencies = []

    if not os.path.exists(obj_file_path):
        return dependencies

    obj_dir = os.path.dirname(obj_file_path)

    try:
        with open(obj_file_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("mtllib "):
                    mtl_filename = line[7:].strip()
                    mtl_path = os.path.normpath(os.path.join(obj_dir, mtl_filename))
                    dependencies.append(mtl_path)

                    if os.path.exists(mtl_path):
                        dependencies.extend(_extract_mtl_textures(mtl_path))
    except Exception:
        pass

    return dependencies


def _extract_mtl_textures(mtl_file_path):
    """Extract texture file paths referenced by an MTL file.

    Args:
        mtl_file_path (str): Path to the MTL file

    Returns:
        list: List of absolute paths to texture files
    """
    textures = []
    mtl_dir = os.path.dirname(mtl_file_path)

    try:
        with open(mtl_file_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("map_"):
                    parts = line.split(None, 1)
                    if len(parts) == 2:
                        texture_filename = parts[1].strip()
                        texture_path = os.path.normpath(os.path.join(mtl_dir, texture_filename))
                        textures.append(texture_path)
    except Exception:
        pass

    return textures


def extract_paths_from_mjcf(xml_file_path: str) -> list[str]:
    """Extract all referenced mesh, texture, and include-xml file paths from a MuJoCo XML file.

    Args:
        xml_file_path (str): Path to the MuJoCo XML file

    Returns:
        list: List of absolute paths to all referenced mesh, texture, and include-xml files
    """
    path = Path(xml_file_path)
    mujoco_xml = path.read_text()
    root = ET.fromstring(mujoco_xml)

    # Handle texture paths
    texture_nodes = root.findall(".//texture")
    texture_relpaths = [texture.get("file") for texture in texture_nodes if texture.get("file") is not None]
    texture_abspaths = [path.parent / rel for rel in texture_relpaths]

    # Parse meshdir
    mesh_basepath = path.parent
    compiler_node = root.find(".//compiler")
    if compiler_node is not None and compiler_node.get("meshdir") is not None:
        mesh_basepath = mesh_basepath / compiler_node.get("meshdir")

    # Handle mesh paths
    mesh_nodes = root.findall(".//mesh")
    mesh_relpaths = [mesh.get("file") for mesh in mesh_nodes if mesh.get("file") is not None]
    mesh_abspaths = [mesh_basepath / rel for rel in mesh_relpaths]

    # Handler include-xml
    include_nodes = root.findall(".//include")
    include_relpaths = [include.get("file") for include in include_nodes if include.get("file") is not None]
    include_abspaths = [path.parent / rel for rel in include_relpaths]

    paths = texture_abspaths + mesh_abspaths + include_abspaths
    paths = [str(path.resolve()) for path in paths]
    return paths
