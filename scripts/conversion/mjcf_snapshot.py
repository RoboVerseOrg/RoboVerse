#!/usr/bin/env python3
"""
Script to take snapshots of MJCF models.
This script loads MJCF files, adds a camera if not present, renders them, and saves the images.
The camera position is automatically adjusted based on the object's size.
"""

import argparse
import glob
import os
import shutil
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pandas as pd
from dm_control import mjcf
from PIL import Image


def calculate_object_bounds(physics):
    """Calculate the bounding box of all geometries in the model."""
    # Get all geometry positions and sizes
    geom_positions = physics.data.geom_xpos
    geom_sizes = physics.model.geom_size
    geom_types = physics.model.geom_type

    # Initialize bounds
    min_bounds = np.array([np.inf, np.inf, np.inf])
    max_bounds = np.array([-np.inf, -np.inf, -np.inf])

    # Process each geometry to determine overall bounds
    for i in range(len(geom_positions)):
        pos = geom_positions[i]
        size = geom_sizes[i]
        geom_type = geom_types[i]

        # Handle different geometry types
        if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
            # For sphere, use radius (size[0])
            min_bounds = np.minimum(min_bounds, pos - size[0])
            max_bounds = np.maximum(max_bounds, pos + size[0])
        elif geom_type == mujoco.mjtGeom.mjGEOM_BOX:
            # For box, use half-sizes
            min_bounds = np.minimum(min_bounds, pos - size)
            max_bounds = np.maximum(max_bounds, pos + size)
        elif geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER or geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
            # For cylinder/capsule, use radius (size[0]) for x/y and half-height (size[1]) for z
            min_bounds = np.minimum(min_bounds, pos - np.array([size[0], size[0], size[1]]))
            max_bounds = np.maximum(max_bounds, pos + np.array([size[0], size[0], size[1]]))
        else:
            # For other geometry types, just use the largest size value
            max_size = np.max(size)
            min_bounds = np.minimum(min_bounds, pos - max_size)
            max_bounds = np.maximum(max_bounds, pos + max_size)

    # If no geometries found or bounds are infinity, use default values
    if np.isinf(min_bounds).any() or np.isinf(max_bounds).any():
        min_bounds = np.array([-0.5, -0.5, 0])
        max_bounds = np.array([0.5, 0.5, 1.0])

    return min_bounds, max_bounds


def calculate_camera_position(bounds_min, bounds_max, margin_factor=1.5):
    """Calculate optimal camera position based on object bounds."""
    # Calculate center and size of the object
    center = (bounds_min + bounds_max) / 2
    size = bounds_max - bounds_min

    # Calculate the required camera distance based on the object size
    max_dim = np.max(size)
    distance = max_dim * margin_factor

    # Position camera at an angle to show 3D structure
    camera_pos = center + np.array([distance, -distance, distance / 2])

    return camera_pos, center


def find_asset_files(mjcf_path):
    """Find all mesh and texture files referenced in the MJCF file."""
    asset_files = []

    try:
        # Parse the XML file
        tree = ET.parse(mjcf_path)
        root = tree.getroot()

        # Get the directory of the MJCF file
        base_dir = os.path.dirname(mjcf_path)

        # Find mesh directories specified in compiler tags
        mesh_dirs = [""]  # Default is the same directory
        for compiler in root.findall(".//compiler"):
            if "meshdir" in compiler.attrib:
                mesh_dirs.append(compiler.attrib["meshdir"])

        # Find texture directories specified in compiler tags
        texture_dirs = [""]  # Default is the same directory
        for compiler in root.findall(".//compiler"):
            if "texturedir" in compiler.attrib:
                texture_dirs.append(compiler.attrib["texturedir"])

        # Find all mesh files
        for mesh in root.findall(".//mesh"):
            if "file" in mesh.attrib:
                mesh_file = mesh.attrib["file"]
                # Try all potential mesh directories
                for mesh_dir in mesh_dirs:
                    potential_path = os.path.join(base_dir, mesh_dir, mesh_file)
                    if os.path.exists(potential_path):
                        asset_files.append(potential_path)
                        break

        # Find all texture files
        for texture in root.findall(".//texture"):
            if "file" in texture.attrib:
                texture_file = texture.attrib["file"]
                # Try all potential texture directories
                for texture_dir in texture_dirs:
                    potential_path = os.path.join(base_dir, texture_dir, texture_file)
                    if os.path.exists(potential_path):
                        asset_files.append(potential_path)
                        break

        # Find all included files
        for include in root.findall(".//include"):
            if "file" in include.attrib:
                include_file = include.attrib["file"]
                include_path = os.path.join(base_dir, include_file)
                if os.path.exists(include_path):
                    asset_files.append(include_path)
                    # Recursively find assets in included files
                    asset_files.extend(find_asset_files(include_path))

    except Exception as e:
        print(f"Error parsing assets in {mjcf_path}: {e}")

    return asset_files


def copy_mjcf_with_assets(mjcf_path, output_dir):
    """Copy MJCF file and all its assets to output directory, maintaining relative paths."""
    try:
        # Create model name based on the file name
        model_name = os.path.splitext(os.path.basename(mjcf_path))[0]
        model_dir = os.path.join(output_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)

        # Find all asset files
        asset_files = find_asset_files(mjcf_path)
        asset_files.append(mjcf_path)  # Add the MJCF file itself

        # Get the base directory of the MJCF file
        base_dir = os.path.dirname(mjcf_path)

        # Copy each asset file, maintaining directory structure
        for asset_path in asset_files:
            # Determine relative path (keep directory structure)
            rel_path = os.path.relpath(asset_path, base_dir)
            target_path = os.path.join(model_dir, rel_path)

            # Create directories if needed
            os.makedirs(os.path.dirname(target_path), exist_ok=True)

            # Copy the file
            shutil.copy2(asset_path, target_path)

        return model_dir, None
    except Exception as e:
        error_msg = f"Error copying assets: {e!s}"
        print(f"Error copying assets for {mjcf_path}: {e}")
        return None, error_msg


def add_scene_lighting(mjcf_model, bounds_min, bounds_max):
    """Add appropriate lighting to the scene based on object bounds."""
    # Get the center of the object
    center = (bounds_min + bounds_max) / 2

    # Check if the model already has lights
    has_lights = False
    for light in mjcf_model.find_all("light"):
        has_lights = True
        break

    if has_lights:
        # Model already has lighting, so let's not interfere
        return

    # Calculate lighting positions based on object bounds
    # We'll add lights from multiple directions for good illumination
    size = bounds_max - bounds_min
    max_dim = np.max(size)
    light_distance = max_dim * 2.0  # Position lights further than camera

    # Add a soft ambient key light from front-top-right
    light1_pos = center + np.array([light_distance, -light_distance, light_distance])
    light1_dir = center - light1_pos
    light1_dir = light1_dir / np.linalg.norm(light1_dir)
    mjcf_model.worldbody.add(
        "light",
        name="key_light",
        pos=f"{light1_pos[0]} {light1_pos[1]} {light1_pos[2]}",
        dir=f"{light1_dir[0]} {light1_dir[1]} {light1_dir[2]}",
        directional="true",
        castshadow="true",
        diffuse="0.6 0.6 0.6",
        specular="0.2 0.2 0.2",
    )

    # Add a fill light from the opposite side (back-left)
    light2_pos = center + np.array([-light_distance * 0.7, light_distance * 0.7, light_distance * 0.5])
    light2_dir = center - light2_pos
    light2_dir = light2_dir / np.linalg.norm(light2_dir)
    mjcf_model.worldbody.add(
        "light",
        name="fill_light",
        pos=f"{light2_pos[0]} {light2_pos[1]} {light2_pos[2]}",
        dir=f"{light2_dir[0]} {light2_dir[1]} {light2_dir[2]}",
        directional="true",
        castshadow="false",
        diffuse="0.4 0.4 0.4",
        specular="0.1 0.1 0.1",
    )

    # Add a top light for better ambient illumination
    light3_pos = center + np.array([0, 0, light_distance])
    light3_dir = center - light3_pos
    light3_dir = light3_dir / np.linalg.norm(light3_dir)
    mjcf_model.worldbody.add(
        "light",
        name="top_light",
        pos=f"{light3_pos[0]} {light3_pos[1]} {light3_pos[2]}",
        dir=f"{light3_dir[0]} {light3_dir[1]} {light3_dir[2]}",
        directional="true",
        castshadow="false",
        diffuse="0.3 0.3 0.3",
        specular="0.1 0.1 0.1",
    )


def process_mjcf_file(mjcf_path, images_dir, mjcf_dir, resolution=(640, 480), fov=45.0):
    """Process a single MJCF file and save a snapshot."""
    result = {
        "input_file": mjcf_path,
        "model_name": os.path.splitext(os.path.basename(mjcf_path))[0],
        "image_path": "",
        "render_success": False,
        "render_error_message": "",
        "mjcf_success": False,
        "mjcf_error_message": "",
        "mjcf_dir": "",
        "processing_time": 0,
    }

    start_time = time.time()

    try:
        print(f"Processing {result['model_name']}...")

        # Try to load with dm_control
        mjcf_model = mjcf.from_path(mjcf_path)

        # Convert to Physics object
        physics = mjcf.Physics.from_mjcf_model(mjcf_model)

        # Make sure we have one simulation step to properly position everything
        physics.step()

        # Calculate object bounds and optimal camera position
        bounds_min, bounds_max = calculate_object_bounds(physics)
        camera_pos, look_at = calculate_camera_position(bounds_min, bounds_max)

        # Add appropriate lighting to the scene
        add_scene_lighting(mjcf_model, bounds_min, bounds_max)

        # Calculate camera orientation based on look_at direction
        direction = look_at - camera_pos
        direction = direction / np.linalg.norm(direction)
        up = np.array([0, 0, 1])
        right = np.cross(direction, up)
        if np.linalg.norm(right) < 1e-6:  # If direction is aligned with up
            up = np.array([0, 1, 0])
            right = np.cross(direction, up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, direction)

        # Check if model has a camera
        has_camera = False
        camera_name = None
        for camera in mjcf_model.find_all("camera"):
            has_camera = True
            camera_name = camera.name
            # Modify existing camera to look at the object
            camera.pos = camera_pos.tolist()
            camera.xyaxes = f"{right[0]} {right[1]} {right[2]} {up[0]} {up[1]} {up[2]}"
            camera.fovy = fov
            break

        # Add a default camera if none exists
        if not has_camera:
            # Add camera to worldbody
            camera_params = {
                "pos": f"{camera_pos[0]} {camera_pos[1]} {camera_pos[2]}",
                "mode": "fixed",
                "fovy": fov,
                "xyaxes": f"{right[0]} {right[1]} {right[2]} {up[0]} {up[1]} {up[2]}",
            }
            camera = mjcf_model.worldbody.add("camera", name="snapshot_camera", **camera_params)
            camera_name = "snapshot_camera"

        # Re-create physics to apply camera and lighting changes
        physics = mjcf.Physics.from_mjcf_model(mjcf_model)
        physics.step()

        # Render the scene
        try:
            img = physics.render(camera_id=camera_name, width=resolution[0], height=resolution[1])

            # Save the image to the images directory
            output_path = os.path.join(images_dir, f"{result['model_name']}.jpg")
            Image.fromarray(img).save(output_path)
            result["mjcf_image_path"] = output_path
            result["mjcf_render_success"] = True
            print(f"Saved snapshot to {output_path}")
        except Exception as render_error:
            result["mjcf_render_error_message"] = f"Rendering error: {render_error!s}"
            print(f"Error rendering {result['model_name']}: {render_error}")

        # Copy MJCF file and its assets to the mjcf directory
        try:
            mjcf_output_dir, error = copy_mjcf_with_assets(mjcf_path, mjcf_dir)
            if mjcf_output_dir:
                result["mjcf_dir"] = mjcf_output_dir
                result["mjcf_success"] = True
                print(f"Copied MJCF and assets for {result['model_name']}")
            else:
                result["mjcf_error_message"] = f"MJCF copying error: {error}"
                print(f"Failed to copy MJCF and assets for {result['model_name']}")
        except Exception as mjcf_error:
            result["mjcf_error_message"] = f"MJCF copying exception: {mjcf_error!s}"
            print(f"Exception copying MJCF for {result['model_name']}: {mjcf_error}")

    except Exception as e:
        error_msg = str(e)
        result["mjcf_error_message"] = f"Model loading error: {error_msg}"
        print(f"Error processing {mjcf_path}: {e}")

    # Calculate processing time
    result["processing_time"] = time.time() - start_time

    return result


def main():
    parser = argparse.ArgumentParser(description="Take snapshots of MJCF models")
    parser.add_argument("input_dir", help="Directory containing MJCF files")
    parser.add_argument("output_dir", help="Directory to save snapshots")
    parser.add_argument("--width", type=int, default=640, help="Image width")
    parser.add_argument("--height", type=int, default=480, help="Image height")
    parser.add_argument("--fov", type=float, default=45.0, help="Camera field of view in degrees")
    parser.add_argument("--extensions", type=str, default="mjcf,xml", help="Comma-separated file extensions to process")
    args = parser.parse_args()

    # Create output directories
    images_dir = os.path.join(args.output_dir, "images")
    mjcf_dir = os.path.join(args.output_dir, "mjcf")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(mjcf_dir, exist_ok=True)

    # Get all MJCF files in the input directory (recursively)
    extensions = args.extensions.split(",")
    mjcf_files = []
    for ext in extensions:
        mjcf_files.extend(glob.glob(os.path.join(args.input_dir, f"**/*.{ext}"), recursive=True))

    if not mjcf_files:
        print(f"No MJCF files found in {args.input_dir}")
        return

    print(f"Found {len(mjcf_files)} MJCF files to process")

    # Process each file
    resolution = (args.width, args.height)
    results = []

    for mjcf_path in mjcf_files:
        result = process_mjcf_file(mjcf_path, images_dir, mjcf_dir, resolution, args.fov)
        results.append(result)

    # Create a DataFrame from the results
    df = pd.DataFrame(results)

    # Save status to CSV
    csv_path = os.path.join(args.output_dir, "processing_status.csv")
    df.to_csv(csv_path, index=False)
    print(f"Status report saved to {csv_path}")

    # Summary
    render_success_count = df["render_success"].sum()
    mjcf_success_count = df["mjcf_success"].sum()
    print(f"Rendering successful for {render_success_count}/{len(mjcf_files)} MJCF files")
    print(f"MJCF copying successful for {mjcf_success_count}/{len(mjcf_files)} MJCF files")

    # If there were failures, print a summary
    if render_success_count < len(mjcf_files) or mjcf_success_count < len(mjcf_files):
        print("\nFailed models:")
        render_failed_df = df[~df["render_success"]]
        mjcf_failed_df = df[~df["mjcf_success"]]

        if not render_failed_df.empty:
            print("\nRendering failures:")
            for _, row in render_failed_df.iterrows():
                print(f"  {row['model_name']}: {row['render_error_message']}")

        if not mjcf_failed_df.empty:
            print("\nMJCF copying failures:")
            for _, row in mjcf_failed_df.iterrows():
                print(f"  {row['model_name']}: {row['mjcf_error_message']}")


if __name__ == "__main__":
    main()
