#!/usr/bin/env python3
"""
Script to take snapshots of URDF models using Isaac Gym.
This script loads URDF files, positions a camera, renders them, and saves the images.
It also updates the existing CSV file from MJCF processing with URDF results.
"""

import argparse
import glob
import os
import traceback

import numpy as np
import pandas as pd
from isaacgym import gymapi
from PIL import Image


def calculate_object_bounds(gym, env, actor_handle):
    """Calculate the bounding box of the URDF model by checking all rigid bodies."""
    # Get the number of bodies in the actor
    body_count = gym.get_actor_rigid_body_count(env, actor_handle)

    # Initialize min and max bounds to extreme values
    min_bounds = np.array([float("inf"), float("inf"), float("inf")])
    max_bounds = np.array([-float("inf"), -float("inf"), -float("inf")])

    # Iterate through each body and update bounds
    for i in range(body_count):
        body_handle = gym.get_actor_rigid_body_handle(env, actor_handle, i)
        # Get the AABB for this specific body
        body_aabb = gym.get_rigid_body_aabb(env, body_handle)

        # Update min bounds
        min_bounds[0] = min(min_bounds[0], body_aabb.min.x)
        min_bounds[1] = min(min_bounds[1], body_aabb.min.y)
        min_bounds[2] = min(min_bounds[2], body_aabb.min.z)

        # Update max bounds
        max_bounds[0] = max(max_bounds[0], body_aabb.max.x)
        max_bounds[1] = max(max_bounds[1], body_aabb.max.y)
        max_bounds[2] = max(max_bounds[2], body_aabb.max.z)

    # If no bodies were found, use a small default bounding box
    if np.any(np.isinf(min_bounds)) or np.any(np.isinf(max_bounds)):
        min_bounds = np.array([-0.1, -0.1, -0.1])
        max_bounds = np.array([0.1, 0.1, 0.1])

    return min_bounds, max_bounds


def calculate_camera_position(bounds_min, bounds_max):
    """Calculate optimal camera position and look-at point based on object bounds."""
    # Calculate object center and size
    center = (bounds_min + bounds_max) / 2
    size = bounds_max - bounds_min
    max_dimension = np.max(size)

    # Calculate distance based on the object's size (add margin for better framing)
    distance = max_dimension * 1.5

    # Position the camera at an angle (30 degrees from horizontal, 45 degrees around vertical)
    elevation_angle = np.radians(30)
    azimuth_angle = np.radians(45)

    # Calculate camera position in spherical coordinates
    x = center[0] + distance * np.cos(elevation_angle) * np.sin(azimuth_angle)
    y = center[1] + distance * np.cos(elevation_angle) * np.cos(azimuth_angle)
    z = center[2] + distance * np.sin(elevation_angle)

    camera_pos = np.array([x, y, z])
    look_at = center

    return camera_pos, look_at


def process_urdf_file(urdf_path, gym, sim, env, viewer, images_dir, resolution=(640, 480)):
    """Process a single URDF file and take a snapshot using Isaac Gym."""
    result = {"urdf_image_path": "", "urdf_error": "", "urdf_success": False}

    try:
        # Create actor properties
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.flip_visual_attachments = False
        asset_options.use_mesh_materials = True

        # Load the URDF asset
        asset = gym.load_asset(sim, os.path.dirname(urdf_path), os.path.basename(urdf_path), asset_options)

        if asset is None:
            result["urdf_error"] = "Failed to load URDF asset"
            return result

        # Create an actor from the asset
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0.0, 0.0, 0.0)  # Position at origin
        pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)  # No rotation

        actor_handle = gym.create_actor(env, asset, pose, "urdf_model", 0, 0)

        if actor_handle == gymapi.INVALID_HANDLE:
            result["urdf_error"] = "Failed to create actor from URDF"
            return result

        # Step the simulation to ensure everything is properly initialized
        gym.simulate(sim)
        gym.fetch_results(sim, True)

        # Use fixed camera position that works well for most objects
        camera_pos = gymapi.Vec3(1.5, 1.5, 1.5)  # Position camera at a fixed distance
        look_at = gymapi.Vec3(0.0, 0.0, 0.0)  # Look at the origin

        # Add a camera
        camera_props = gymapi.CameraProperties()
        camera_props.width = resolution[0]
        camera_props.height = resolution[1]
        camera_props.horizontal_fov = 45.0
        camera_props.near_plane = 0.01
        camera_props.far_plane = 100.0

        # Create camera and set its position
        camera_handle = gym.create_camera_sensor(env, camera_props)
        gym.set_camera_location(camera_handle, env, camera_pos, look_at)

        # Wait for a moment to ensure rendering is ready
        for _ in range(5):
            gym.simulate(sim)
            gym.fetch_results(sim, True)
            gym.step_graphics(sim)
            gym.render_all_camera_sensors(sim)

        # Capture the image from the camera
        image = gym.get_camera_image(sim, env, camera_handle, gymapi.IMAGE_COLOR)

        # Convert the image from the camera format to an RGB image
        image = image.reshape(resolution[1], resolution[0], 4)  # RGBA format
        image = image[:, :, :3]  # Keep only RGB channels

        # Save the image
        model_name = os.path.splitext(os.path.basename(urdf_path))[0]
        output_path = os.path.join(images_dir, f"{model_name}.jpg")
        Image.fromarray(image).save(output_path)

        result["urdf_image_path"] = output_path
        result["urdf_success"] = True

        # No need to remove actor - we'll recreate the environment for each model

    except Exception as e:
        error_msg = f"Error processing URDF: {e!s}\n{traceback.format_exc()}"
        result["urdf_error"] = error_msg
        print(f"Error processing {urdf_path}: {error_msg}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Take snapshots of URDF models and update CSV results")
    parser.add_argument("urdf_dir", help="Directory containing URDF files")
    parser.add_argument("csv_file", help="Existing CSV file from MJCF processing")
    parser.add_argument("output_dir", help="Directory to save outputs")
    parser.add_argument("--width", type=int, default=640, help="Image width")
    parser.add_argument("--height", type=int, default=480, help="Image height")
    parser.add_argument("--extensions", type=str, default="urdf", help="Comma-separated file extensions to process")
    args = parser.parse_args()

    # Create output directory for images
    images_dir = os.path.join(args.output_dir, "urdf_images")
    os.makedirs(images_dir, exist_ok=True)

    # Read the existing CSV file
    if not os.path.exists(args.csv_file):
        print(f"CSV file {args.csv_file} does not exist!")
        return

    df = pd.read_csv(args.csv_file)

    # Extract model names from the CSV
    model_names = set(df["model_name"].values)

    # Initialize Isaac Gym
    gym = gymapi.acquire_gym()

    # Create simulator
    sim_params = gymapi.SimParams()
    sim_params.substeps = 2
    sim_params.dt = 1.0 / 60.0
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)

    # Set physics parameters
    sim_params.physx.use_gpu = True
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.contact_offset = 0.01
    sim_params.physx.rest_offset = 0.0

    # Create simulation instance
    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
    if sim is None:
        print("Failed to create simulation instance")
        return

    # Create a ground plane
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0, 0, 1)  # Z-up
    plane_params.distance = 0.0
    plane_params.static_friction = 1.0
    plane_params.dynamic_friction = 1.0
    plane_params.restitution = 0.0

    # Add ground plane
    gym.add_ground(sim, plane_params)

    # Create viewer
    viewer = None
    # Uncomment this line to visualize the rendering process:
    # viewer = gym.create_viewer(sim, gymapi.CameraProperties())

    # Get all URDF files in the input directory (recursively)
    extensions = args.extensions.split(",")
    urdf_files = []
    for ext in extensions:
        urdf_files.extend(glob.glob(os.path.join(args.urdf_dir, f"**/*.{ext}"), recursive=True))

    if not urdf_files:
        print(f"No URDF files found in {args.urdf_dir}")
        return

    print(f"Found {len(urdf_files)} URDF files to process")

    # Initialize dataframe with new columns
    df["urdf_image_path"] = ""
    df["urdf_error"] = ""
    df["urdf_success"] = False

    # Track processed model names to identify those missing in the CSV
    processed_models = set()

    # Process each URDF file
    resolution = (args.width, args.height)

    for urdf_path in urdf_files:
        model_name = os.path.splitext(os.path.basename(urdf_path))[0]
        processed_models.add(model_name)

        # Create a new environment for each model
        env_spacing = 2.0
        env_lower = gymapi.Vec3(-env_spacing, -env_spacing, 0.0)
        env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)
        env = gym.create_env(sim, env_lower, env_upper, 1)

        # Check if model name exists in CSV
        if model_name in model_names:
            # Process the URDF file
            print(f"Processing URDF for {model_name}...")
            result = process_urdf_file(urdf_path, gym, sim, env, viewer, images_dir, resolution)

            # Update the DataFrame for this model
            idx = df[df["model_name"] == model_name].index
            for key, value in result.items():
                df.loc[idx, key] = value
        else:
            print(f"Model {model_name} not found in the CSV file, skipping...")

        # Destroy the environment to clean up
        gym.destroy_env(env)

    # Save the updated CSV
    updated_csv_path = os.path.join(args.output_dir, "processing_status_with_urdf.csv")
    df.to_csv(updated_csv_path, index=False)
    print(f"Updated status report saved to {updated_csv_path}")

    # Print summary
    urdf_success_count = df["urdf_success"].sum()
    print(f"URDF rendering successful for {urdf_success_count}/{len(model_names)} models")

    # Report models that were in URDF directory but not in CSV
    missing_in_csv = processed_models - model_names
    if missing_in_csv:
        print(f"\n{len(missing_in_csv)} URDF models not found in CSV:")
        for model in sorted(missing_in_csv):
            print(f"  {model}")

    # Clean up
    if viewer is not None:
        gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)


if __name__ == "__main__":
    main()
