#!/usr/bin/env python3
"""
Script to take snapshots of USD models using Isaac Lab.
This script loads USD files, positions a camera, renders them, and saves the images.
It also updates the existing CSV file from MJCF processing with USD results.

IMPORTANT: This script must be run from within Omniverse/Isaac Lab.
It cannot be run from a standard Python environment.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import glob
import os
import traceback
import uuid

import numpy as np
import pandas as pd

# Configure command-line arguments
parser = argparse.ArgumentParser(description="Take snapshots of USD models and update CSV results")
parser.add_argument("usd_dir", help="Directory containing USD files")
parser.add_argument("csv_file", help="Existing CSV file from processing")
parser.add_argument("output_dir", help="Directory to save outputs")
parser.add_argument("--width", type=int, default=640, help="Image width")
parser.add_argument("--height", type=int, default=480, help="Image height")
parser.add_argument(
    "--extensions", type=str, default="usd", help="Comma-separated file extensions to process (usd, usda, usdc)"
)

# Launch the app
from omni.isaac.lab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import omni

# Import more IsaacLab modules
import omni.kit.commands

# Now that the app is launched, we can import Omniverse modules
from omni.isaac.core import World
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.utils.stage import add_reference_to_stage, get_current_stage
from omni.isaac.core.utils.viewports import set_camera_view
from pxr import Gf, UsdGeom


def create_camera(
    camera_path, focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1000.0)
):
    """Create a camera with the specified parameters using USD API directly."""
    # Create the camera prim
    stage = get_current_stage()
    omni.kit.commands.execute("CreatePrimWithDefaultXform", prim_type="Camera", prim_path=camera_path)

    # Get the camera prim
    camera_prim = stage.GetPrimAtPath(camera_path)

    # Set camera attributes directly using the USD API
    if camera_prim:
        camera = UsdGeom.Camera(camera_prim)
        camera.CreateFocalLengthAttr().Set(focal_length)
        camera.CreateFocusDistanceAttr().Set(focus_distance)
        camera.CreateHorizontalApertureAttr().Set(horizontal_aperture)
        camera.CreateClippingRangeAttr().Set(Gf.Vec2f(clipping_range[0], clipping_range[1]))
        return True
    return False


def process_usd_file(usd_path, world, images_dir, resolution=(640, 480)):
    """Process a single USD file and take a snapshot using Isaac Lab."""
    result = {"usd_image_path": "", "usd_error": "", "usd_success": False}

    camera_path = None

    try:
        # Reset the world to clear previous objects
        world.reset()

        # Get current stage
        stage = get_current_stage()

        # Add USD reference to stage
        model_prim_path = "/World/USD_Model"
        add_reference_to_stage(usd_path, model_prim_path)

        # Step the simulation to ensure everything is properly initialized
        world.step(render=True)

        # Create a camera using the standard IsaacLab approach
        camera_id = str(uuid.uuid4())[:8]
        camera_name = f"snapshot_camera_{camera_id}"
        camera_path = f"/World/{camera_name}"

        # Create camera using direct USD API instead of commands
        if not create_camera(camera_path):
            result["usd_error"] = "Failed to create camera"
            return result

        # Create XFormPrim for the camera
        camera_prim = XFormPrim(camera_path)

        # Position camera
        eye_position = np.array([1.5, 1.5, 1.5])
        target_position = np.array([0.0, 0.0, 0.0])

        # Set the camera position
        camera_prim.set_world_pose(
            position=eye_position,
            orientation=np.array([0.0, 0.0, 0.0, 1.0]),  # Identity quaternion
        )

        # Configure the viewport camera to match our camera
        set_camera_view(eye=eye_position, target=target_position, camera_prim_path=camera_path)

        # Step a few times to render properly
        for _ in range(5):
            world.step(render=True)

        # Capture image using the Viewport
        from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport

        # Capture image
        model_name = os.path.splitext(os.path.basename(usd_path))[0]
        output_path = os.path.join(images_dir, f"{model_name}.jpg")

        # Get the active viewport
        viewport = get_active_viewport()
        if viewport is not None:
            # Render and capture viewport
            capture_viewport_to_file(viewport, output_path, resolution[0], resolution[1])
            print(f"Saved image to {output_path} via viewport utility")

            # Verify the file exists
            if os.path.exists(output_path):
                result["usd_image_path"] = output_path
                result["usd_success"] = True
                print(f"Successfully saved image for {model_name}")
            else:
                result["usd_error"] = f"Image file was not created at {output_path}"
        else:
            result["usd_error"] = "Failed to get active viewport"

    except Exception as e:
        error_msg = f"Error processing USD: {e!s}\n{traceback.format_exc()}"
        result["usd_error"] = error_msg
        print(f"Error processing {usd_path}: {error_msg}")

    finally:
        # Clean up - remove the camera in a finally block to ensure it always happens
        if camera_path:
            try:
                omni.kit.commands.execute("DeletePrims", paths=[camera_path])
            except Exception as cleanup_error:
                print(f"Warning: Failed to delete camera prim: {cleanup_error}")

    return result


def main():
    # Create output directory for images
    images_dir = os.path.join(args_cli.output_dir, "usd_images")
    os.makedirs(images_dir, exist_ok=True)
    print(f"Output directory for images: {images_dir}")

    # Read the existing CSV file
    if not os.path.exists(args_cli.csv_file):
        print(f"CSV file {args_cli.csv_file} does not exist!")
        return

    df = pd.read_csv(args_cli.csv_file)

    # Extract model names from the CSV
    model_names = set(df["model_name"].values)

    # Initialize World (not SimulationContext)
    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()

    # Add a light - lower intensity to fix brightness issue
    light_path = "/World/Light"
    omni.kit.commands.execute("CreatePrimWithDefaultXform", prim_type="DistantLight", prim_path=light_path)
    omni.kit.commands.execute("SetLightIntensity", light_path=light_path, intensity=100.0)

    # Add ambient light for better overall lighting
    ambient_light_path = "/World/AmbientLight"
    omni.kit.commands.execute("CreatePrimWithDefaultXform", prim_type="DomeLight", prim_path=ambient_light_path)
    omni.kit.commands.execute("SetLightIntensity", light_path=ambient_light_path, intensity=50.0)

    # Get all USD files in the input directory (recursively)
    extensions = args_cli.extensions.split(",")
    usd_files = []
    for ext in extensions:
        usd_files.extend(glob.glob(os.path.join(args_cli.usd_dir, f"**/*.{ext}"), recursive=True))

    if not usd_files:
        print(f"No USD files found in {args_cli.usd_dir}")
        return

    print(f"Found {len(usd_files)} USD files to process")

    # Initialize dataframe with new columns for USD results
    if "usd_image_path" not in df.columns:
        df["usd_image_path"] = ""
    if "usd_error" not in df.columns:
        df["usd_error"] = ""
    if "usd_success" not in df.columns:
        df["usd_success"] = False

    # Track processed model names to identify those missing in the CSV
    processed_models = set()
    success_count = 0

    # Process each USD file
    resolution = (args_cli.width, args_cli.height)

    for usd_path in usd_files:
        model_name = os.path.splitext(os.path.basename(usd_path))[0]
        processed_models.add(model_name)

        # Check if model name exists in CSV
        if model_name in model_names:
            # Process the USD file
            print(f"Processing USD for {model_name}...")
            result = process_usd_file(usd_path, world, images_dir, resolution)

            # Update the DataFrame for this model
            idx = df[df["model_name"] == model_name].index
            for key, value in result.items():
                df.loc[idx, key] = value

            # Update success count manually for accurate reporting
            if result["usd_success"]:
                success_count += 1
        else:
            print(f"Model {model_name} not found in the CSV file, skipping...")

    # Save the updated CSV
    updated_csv_path = os.path.join(args_cli.output_dir, "processing_status_with_usd.csv")
    df.to_csv(updated_csv_path, index=False)
    print(f"Updated status report saved to {updated_csv_path}")

    # Print summary - use our manual count for accuracy
    print(f"USD rendering successful for {success_count}/{len(model_names)} models")

    # Report models that were in USD directory but not in CSV
    missing_in_csv = processed_models - model_names
    if missing_in_csv:
        print(f"\n{len(missing_in_csv)} USD models not found in CSV:")
        for model in sorted(missing_in_csv):
            print(f"  {model}")

    # Clean up
    world.stop()


if __name__ == "__main__":
    # Run the main function
    main()
    # Close the app properly
    simulation_app.close()
