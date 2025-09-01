"""This file contains the utility functions for parsing URDF and MJCF files."""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from pathlib import Path


def extract_mesh_paths_from_urdf(urdf_file_path):
    """Extract all mesh file paths from a URDF XML file and convert them to absolute paths.

    Args:
        urdf_file_path (str): Path to the URDF XML file

    Returns:
        list: List of absolute paths to all referenced mesh files
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

    return mesh_paths


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
