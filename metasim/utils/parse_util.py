"""This file contains the utility functions for parsing URDF and MJCF files."""

from __future__ import annotations

import os
from pathlib import Path

from metasim.utils.xml_safe import ET  # defused parser for untrusted MJCF/URDF


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
        # Explicitly read as UTF-8 and replace undecodable bytes to avoid Unicode errors on Windows
        with open(obj_file_path, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if line.startswith("mtllib "):
                    mtl_filename = line[7:].strip()
                    mtl_path = os.path.normpath(os.path.join(obj_dir, mtl_filename))
                    dependencies.append(mtl_path)

                    if os.path.exists(mtl_path):
                        dependencies.extend(_extract_mtl_textures(mtl_path))
    except OSError as err:
        # Best-effort discovery: an I/O failure yields a PARTIAL dependency
        # list, which later surfaces as a confusing missing-asset/black-material
        # error at load time. Surface it here instead of silently swallowing.
        # The file is opened with errors="replace" so decode errors never reach
        # here; a genuinely unexpected exception should propagate as a real bug.
        from loguru import logger as _log

        _log.warning(f"Failed to read OBJ '{obj_file_path}' for dependencies: {err}")

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
        # Explicitly read as UTF-8 and replace undecodable bytes to avoid Unicode errors on Windows
        with open(mtl_file_path, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if line.startswith("map_"):
                    parts = line.split(None, 1)
                    if len(parts) == 2:
                        texture_filename = parts[1].strip()
                        texture_path = os.path.normpath(os.path.join(mtl_dir, texture_filename))
                        textures.append(texture_path)
    except OSError as err:
        # See _extract_obj_dependencies: surface I/O failures instead of
        # silently returning a partial texture list.
        from loguru import logger as _log

        _log.warning(f"Failed to read MTL '{mtl_file_path}' for textures: {err}")

    return textures


def extract_paths_from_mjcf(xml_file_path: str, _max_include_depth: int = 8) -> list[str]:
    """Extract referenced mesh, texture, and include-xml paths from a MuJoCo XML.

    Follows ``<include file="..."/>`` transitively so a scene that pulls
    in a robot file which itself includes a materials file resolves the
    deeper mesh/texture references too. Already-visited absolute paths
    are tracked to break include cycles.

    Args:
        xml_file_path: Path to the MuJoCo XML file.
        _max_include_depth: Hard cap on include recursion, guarding
            against a pathological mutually-recursive XML.

    Returns:
        list[str]: Absolute paths to every referenced mesh, texture, and
        ``<include>`` XML, plus the transitive references of those includes.
    """
    visited: set[str] = set()

    def _collect(file_path: str, depth: int) -> list[str]:
        path = Path(file_path)
        resolved = str(path.resolve())
        if resolved in visited:
            return []
        visited.add(resolved)
        if depth > _max_include_depth:
            return []

        try:
            mujoco_xml = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            # If a transitive include points to a file we haven't fetched
            # yet, we still want the caller (hf_util) to attempt download
            # — return the absolute path so it's enqueued, but stop
            # recursing into a file we can't read.
            return []

        try:
            root = ET.fromstring(mujoco_xml)
        except ET.ParseError as err:
            # Some shipped MJCF files (e.g. originally roboverse_data/robots/
            # franka/mjcf/mjx_panda.xml before its fix) are tolerated by
            # MuJoCo's native parser but trip up Python's strict ET — unbalanced
            # ``<body>`` tags, missing wrappers, etc. Treat as "no extractable
            # paths to download" so the backend's own loader gets a chance to
            # surface a more specific error at launch time.
            from loguru import logger as _log

            _log.warning(
                f"extract_paths_from_mjcf: strict-XML parse failed on {file_path} ({err}); "
                f"skipping asset extraction. Backend's own MJCF parser may still accept this file."
            )
            return []

        # Parse compiler asset directories. Per the MJCF spec, ``assetdir``
        # sets the default search path for BOTH meshes and textures;
        # ``meshdir`` / ``texturedir`` override it for their respective asset
        # types. Honouring ``assetdir`` lets robots that use only the combined
        # attribute (e.g. google_robot, hello_robot_stretch in mujoco_menagerie)
        # resolve meshes locally instead of falling through to a doomed
        # HuggingFace fetch.
        compiler_node = root.find(".//compiler")
        assetdir = compiler_node.get("assetdir") if compiler_node is not None else None
        meshdir = compiler_node.get("meshdir") if compiler_node is not None else None
        texturedir = compiler_node.get("texturedir") if compiler_node is not None else None

        texture_basepath = path.parent
        if texturedir is not None:
            texture_basepath = texture_basepath / texturedir
        elif assetdir is not None:
            texture_basepath = texture_basepath / assetdir
        texture_relpaths = [t.get("file") for t in root.findall(".//texture") if t.get("file") is not None]
        texture_abspaths = [texture_basepath / rel for rel in texture_relpaths]

        mesh_basepath = path.parent
        if meshdir is not None:
            mesh_basepath = mesh_basepath / meshdir
        elif assetdir is not None:
            mesh_basepath = mesh_basepath / assetdir
        mesh_relpaths = [m.get("file") for m in root.findall(".//mesh") if m.get("file") is not None]
        mesh_abspaths = [mesh_basepath / rel for rel in mesh_relpaths]

        include_relpaths = [i.get("file") for i in root.findall(".//include") if i.get("file") is not None]
        include_abspaths = [path.parent / rel for rel in include_relpaths]

        own_paths = [str(p.resolve()) for p in texture_abspaths + mesh_abspaths + include_abspaths]

        # Recurse into includes that exist on disk; for missing ones we
        # already enqueued the absolute path above.
        for inc_abs in include_abspaths:
            if inc_abs.exists():
                own_paths.extend(_collect(str(inc_abs), depth + 1))

        return own_paths

    return _collect(xml_file_path, depth=0)
