"""This file contains the utility functions for parsing URDF and MJCF files."""

import os
import xml.etree.ElementTree as ET


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


def extract_mesh_paths_from_mjcf(xml_file_path):
    """Extract all referenced mesh file paths from a MuJoCo XML file.

    Args:
        xml_file_path (str): Path to the MuJoCo XML file

    Returns:
        list: List of absolute paths to all referenced mesh files
    """
    # Parse the XML file
    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    # Get the mesh directory if specified
    meshdir = root.find(".//compiler").get("meshdir", "")

    # Base directory of the XML file
    base_dir = os.path.dirname(os.path.abspath(xml_file_path))

    # Full mesh directory path
    mesh_dir_path = os.path.join(base_dir, meshdir) if meshdir else base_dir

    # Find all mesh elements
    mesh_files = []

    # Check for mesh references in 'mesh' elements
    for mesh in root.findall(".//mesh"):
        file_name = mesh.get("file")
        if file_name:
            mesh_files.append(os.path.join(mesh_dir_path, file_name))

    # Also check for mesh references in 'geom' elements with type="mesh"
    for geom in root.findall(".//geom[@type='mesh']"):
        mesh_name = geom.get("mesh")
        if mesh_name:
            # Find the corresponding mesh element to get the file path
            mesh_elem = root.find(f".//mesh[@name='{mesh_name}']")
            if mesh_elem is not None:
                file_name = mesh_elem.get("file")
                if file_name:
                    mesh_files.append(os.path.join(mesh_dir_path, file_name))

    return mesh_files
