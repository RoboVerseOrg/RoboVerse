"""Domain Randomization Example for MetaSim."""

from __future__ import annotations

import rootutils

rootutils.setup_root(__file__, pythonpath=True)

import os
from typing import Literal

import numpy as np
import torch
import tyro
from loguru import logger as log
from rich.logging import RichHandler

from metasim.constants import PhysicStateType
from metasim.randomization import (
    CameraPresets,
    CameraRandomizer,
    LightPresets,
    LightRandomizer,
    MaterialPresets,
    MaterialRandomizer,
    ObjectPresets,
    ObjectRandomizer,
)
from metasim.randomization.presets.light_presets import LightScenarios
from metasim.scenario.cameras import PinholeCameraCfg
from metasim.scenario.lights import DistantLightCfg, SphereLightCfg
from metasim.scenario.objects import (
    ArticulationObjCfg,
    PrimitiveCubeCfg,
    PrimitiveSphereCfg,
)
from metasim.scenario.scenario import ScenarioCfg
from metasim.utils import configclass
from metasim.utils.obs_utils import ObsSaver
from metasim.utils.setup_util import get_handler

log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])


def log_randomization_result(
    randomizer_type: str,
    obj_name: str,
    property_name: str,
    before_value,
    after_value,
    unit: str = "",
):
    """Log randomization results in a consistent format."""
    if isinstance(before_value, torch.Tensor):
        before_str = str(before_value.cpu().numpy().round(3))
    else:
        before_str = str(before_value)

    if isinstance(after_value, torch.Tensor):
        after_str = str(after_value.cpu().numpy().round(3))
    else:
        after_str = str(after_value)

    log.info(f"  [{randomizer_type}] {obj_name}.{property_name}: {before_str} -> {after_str} {unit}")


def log_randomization_header(randomizer_name: str, description: str = ""):
    """Log a consistent header for randomization sections."""
    log.info("=" * 60)
    if description:
        log.info(f"{randomizer_name}: {description}")
    else:
        log.info(randomizer_name)


def run_domain_randomization(args):
    """Demonstrate domain randomization with specified simulator."""
    log.info(f"=== {args.sim.upper()} Domain Randomization Demo ===")

    # Set up reproducible randomization
    if args.seed is not None:
        log.info(f"Reproducibility: Using seed {args.seed}")
        # Set global seeds for PyTorch, NumPy, etc.
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)

    # Create scenario and update simulator
    scenario = ScenarioCfg(
        robots=["franka"],
        num_envs=args.num_envs,  # Multiple environments for parallel testing
        simulator=args.sim,  # Will be overridden
        headless=args.headless,  # Will be overridden
    )

    # Add single camera for video recording and randomization
    scenario.cameras = [
        PinholeCameraCfg(
            name="main_camera",
            width=1024,
            height=1024,
            pos=(1.5, -1.5, 1.5),
            look_at=(0.0, 0.0, 0.0),
        )
    ]

    # Demo: Using LightScenarios for complex lighting setups
    if args.lighting_scenario == "indoor_room":
        # Use LightScenarios for indoor room setup
        scenario.lights = [
            DistantLightCfg(
                name="ceiling_light",
                intensity=800.0,
                color=(1.0, 1.0, 0.9),
                polar=0.0,
                azimuth=0.0,
                is_global=True,
            ),
            SphereLightCfg(
                name="window_light",
                intensity=1200.0,
                color=(0.9, 0.95, 1.0),
                radius=0.3,
                pos=(2.0, 1.0, 2.5),
                is_global=False,
            ),
            SphereLightCfg(
                name="desk_lamp",
                intensity=400.0,
                color=(1.0, 0.8, 0.6),
                radius=0.1,
                pos=(-1.0, -1.0, 1.0),
                is_global=False,
            ),
        ]
    elif args.lighting_scenario == "outdoor_scene":
        # Use LightScenarios for outdoor setup
        scenario.lights = [
            DistantLightCfg(
                name="sun_light",
                intensity=2000.0,
                color=(1.0, 0.95, 0.8),
                polar=45.0,
                azimuth=30.0,
                is_global=True,
            ),
            DistantLightCfg(
                name="sky_light",
                intensity=600.0,
                color=(0.7, 0.8, 1.0),
                polar=80.0,
                azimuth=0.0,
                is_global=True,
            ),
        ]
    elif args.lighting_scenario == "studio":
        # Use LightScenarios for studio setup
        scenario.lights = [
            DistantLightCfg(
                name="key_light",
                intensity=1500.0,
                color=(1.0, 1.0, 1.0),
                polar=30.0,
                azimuth=45.0,
                is_global=True,
            ),
            SphereLightCfg(
                name="fill_light",
                intensity=800.0,
                color=(0.9, 0.9, 1.0),
                radius=0.5,
                pos=(1.0, -2.0, 2.0),
                is_global=False,
            ),
            SphereLightCfg(
                name="rim_light",
                intensity=600.0,
                color=(1.0, 0.9, 0.8),
                radius=0.2,
                pos=(-2.0, 0.5, 1.5),
                is_global=False,
            ),
        ]
    elif args.lighting_scenario == "demo":
        # Demo mode: maximum dramatic changes for visibility with strong shadows
        # Use multiple lights to create clear shadow effects
        scenario.lights = [
            # Main distant light for overall scene lighting and color changes
            DistantLightCfg(
                name="rainbow_light",
                intensity=3000.0,
                color=(1.0, 0.0, 0.0),
                polar=60.0,
                azimuth=45.0,
                is_global=True,
            ),
            # Strong sphere light for position-based shadow changes (lower initial position)
            SphereLightCfg(
                name="disco_light",
                intensity=15000.0,
                color=(1.0, 1.0, 1.0),
                radius=0.3,
                pos=(0.0, 0.0, 2.0),
                is_global=False,
            ),
            # Second sphere light for cross-shadows (lower initial position)
            SphereLightCfg(
                name="shadow_light",
                intensity=12000.0,
                color=(1.0, 1.0, 1.0),
                radius=0.2,
                pos=(0.0, 0.0, 2.0),
                is_global=False,
            ),
        ]
    else:
        # Default simple setup
        scenario.lights = [
            DistantLightCfg(
                name="main_light",
                intensity=1000.0,
                color=(1.0, 1.0, 1.0),
                polar=45.0,
                azimuth=30.0,
                is_global=True,
            ),
            SphereLightCfg(
                name="ambient_light",
                intensity=500.0,
                color=(0.9, 0.8, 0.7),
                radius=0.5,
                pos=(0.0, 0.0, 2.0),
                is_global=False,
            ),
        ]

    # Add objects (same as 0_static_scene.py)
    scenario.objects = [
        PrimitiveCubeCfg(
            name="cube",
            size=(0.1, 0.1, 0.1),
            color=[1.0, 0.0, 0.0],
            physics=PhysicStateType.RIGIDBODY,
        ),
        PrimitiveSphereCfg(
            name="sphere",
            radius=0.1,
            color=[0.0, 0.0, 1.0],
            physics=PhysicStateType.RIGIDBODY,
        ),
        ArticulationObjCfg(
            name="box_base",
            fix_base_link=True,
            usd_path="roboverse_data/assets/rlbench/close_box/box_base/usd/box_base.usd",
            urdf_path="roboverse_data/assets/rlbench/close_box/box_base/urdf/box_base_unique.urdf",
            mjcf_path="roboverse_data/assets/rlbench/close_box/box_base/mjcf/box_base_unique.mjcf",
        ),
    ]

    # Get handler based on simulator
    handler = get_handler(scenario)
    init_states = [
        {
            "objects": {
                "cube": {
                    "pos": torch.tensor([0.3, -0.2, 0.05]),
                    "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                },
                "sphere": {
                    "pos": torch.tensor([0.4, -0.6, 0.05]),
                    "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                },
                "box_base": {
                    "pos": torch.tensor([0.5, 0.2, 0.1]),
                    "rot": torch.tensor([0.0, 0.7071, 0.0, 0.7071]),
                    "dof_pos": {"box_joint": 0.0},
                },
            },
            "robots": {
                "franka": {
                    "pos": torch.tensor([0.0, 0.0, 0.0]),
                    "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                    "dof_pos": {
                        "panda_joint1": 0.0,
                        "panda_joint2": -0.785398,
                        "panda_joint3": 0.0,
                        "panda_joint4": -2.356194,
                        "panda_joint5": 0.0,
                        "panda_joint6": 1.570796,
                        "panda_joint7": 0.785398,
                        "panda_finger_joint1": 0.04,
                        "panda_finger_joint2": 0.04,
                    },
                },
            },
        }
    ] * scenario.num_envs

    handler.set_states(init_states)

    # Initialize video recording
    os.makedirs("get_started/output", exist_ok=True)
    obs_saver = ObsSaver(video_path=f"get_started/output/12_domain_randomization_{args.sim}.mp4")
    obs = handler.get_states(mode="tensor")
    obs_saver.add(obs)

    # Initialize object randomizers using unified approach
    log.info("================================================")
    log.info("Using ObjectRandomizer (unified approach)")

    # Cube: Grasping target with comprehensive randomization
    cube_randomizer = ObjectRandomizer(ObjectPresets.grasping_target("cube"), seed=args.seed)
    cube_randomizer.bind_handler(handler)

    # Sphere: Bouncy object with varied physics and pose
    sphere_randomizer = ObjectRandomizer(ObjectPresets.bouncy_object("sphere"), seed=args.seed)
    sphere_randomizer.bind_handler(handler)

    # Robot: Base randomization for payload simulation
    franka_randomizer = ObjectRandomizer(ObjectPresets.robot_base("franka"), seed=args.seed)
    franka_randomizer.bind_handler(handler)

    # Store for later use
    object_randomizers = [cube_randomizer, sphere_randomizer, franka_randomizer]

    # Initialize material randomizers with different strategies

    # Cube: Wood with MDL textures (combined mode - physics + visual)
    cube_material_randomizer = MaterialRandomizer(
        MaterialPresets.wood_object("cube", use_mdl=True, randomization_mode="combined"),
        seed=args.seed,
    )
    cube_material_randomizer.bind_handler(handler)

    # Sphere: Rubber with high bounce (combined mode - physics + visual)
    sphere_material_randomizer = MaterialRandomizer(
        MaterialPresets.rubber_object("sphere", randomization_mode="combined"),
        seed=args.seed,
    )
    sphere_material_randomizer.bind_handler(handler)

    # Box: Metal with MDL textures (combined mode - physics + visual)
    box_material_randomizer = MaterialRandomizer(
        MaterialPresets.wood_object("box_base", use_mdl=True, randomization_mode="combined"),
        seed=args.seed,
    )
    box_material_randomizer.bind_handler(handler)

    # Initialize light randomizers using LightScenarios for complex setups
    light_randomizers = []

    if args.lighting_scenario == "indoor_room":
        # Get indoor room scenario configurations
        light_configs = LightScenarios.indoor_room()
        log.info("Using Indoor Room lighting scenario with 3 lights")

        # Create randomizers for each light in the scenario
        for config in light_configs:
            randomizer = LightRandomizer(config, seed=args.seed)
            randomizer.bind_handler(handler)
            light_randomizers.append(randomizer)

    elif args.lighting_scenario == "outdoor_scene":
        # Get outdoor scene scenario configurations
        light_configs = LightScenarios.outdoor_scene()
        log.info("Using Outdoor Scene lighting scenario with 2 lights")

        for config in light_configs:
            randomizer = LightRandomizer(config, seed=args.seed)
            randomizer.bind_handler(handler)
            light_randomizers.append(randomizer)

    elif args.lighting_scenario == "studio":
        # Get studio scenario configurations
        light_configs = LightScenarios.three_point_studio()
        log.info("Using Three-Point Studio lighting scenario with 3 lights")

        for config in light_configs:
            randomizer = LightRandomizer(config, seed=args.seed)
            randomizer.bind_handler(handler)
            light_randomizers.append(randomizer)

    elif args.lighting_scenario == "demo":
        # Demo mode: use extreme colors and positions for maximum visual impact with shadows
        log.info("Using Demo mode with EXTREME changes, shadows, and debugging (3 lights)")

        # Use demo presets for maximum visual impact
        rainbow_randomizer = LightRandomizer(
            LightPresets.demo_colors("rainbow_light", randomization_mode="color_only"),
            seed=args.seed,
        )
        rainbow_randomizer.bind_handler(handler)

        # Use position randomizers for shadow effects
        position_randomizer = LightRandomizer(
            LightPresets.demo_positions("disco_light", randomization_mode="position_only"),
            seed=args.seed,
        )
        position_randomizer.bind_handler(handler)

        shadow_randomizer = LightRandomizer(
            LightPresets.demo_positions("shadow_light", randomization_mode="position_only"),
            seed=args.seed,
        )
        shadow_randomizer.bind_handler(handler)

        light_randomizers = [rainbow_randomizer, position_randomizer, shadow_randomizer]
    else:
        # Default simple setup using individual presets
        log.info("Using default lighting setup with 2 lights")
        main_light_randomizer = LightRandomizer(
            LightPresets.outdoor_daylight("main_light", randomization_mode="combined"),
            seed=args.seed,
        )
        main_light_randomizer.bind_handler(handler)

        ambient_light_randomizer = LightRandomizer(
            LightPresets.indoor_ambient("ambient_light", randomization_mode="combined"),
            seed=args.seed,
        )
        ambient_light_randomizer.bind_handler(handler)

        light_randomizers = [main_light_randomizer, ambient_light_randomizer]

    # Initialize single camera randomizer
    camera_randomizers = []

    if args.camera_scenario == "position_only":
        camera_randomizer = CameraRandomizer(
            CameraPresets.surveillance_camera("main_camera", randomization_mode="position_only"),
            seed=args.seed,
        )
        log.info("Using position-only camera randomization")
    elif args.camera_scenario == "orientation_only":
        camera_randomizer = CameraRandomizer(
            CameraPresets.surveillance_camera("main_camera", randomization_mode="orientation_only"),
            seed=args.seed,
        )
        log.info("Using orientation-only camera randomization (rotation deltas)")
    elif args.camera_scenario == "look_at_only":
        camera_randomizer = CameraRandomizer(
            CameraPresets.surveillance_camera("main_camera", randomization_mode="look_at_only"),
            seed=args.seed,
        )
        log.info("Using look-at-only camera randomization (target point changes)")
    elif args.camera_scenario == "intrinsics_only":
        camera_randomizer = CameraRandomizer(
            CameraPresets.surveillance_camera("main_camera", randomization_mode="intrinsics_only"),
            seed=args.seed,
        )
        log.info("Using intrinsics-only camera randomization")
    elif args.camera_scenario == "image_only":
        camera_randomizer = CameraRandomizer(
            CameraPresets.surveillance_camera("main_camera", randomization_mode="image_only"),
            seed=args.seed,
        )
        log.info("Using image-only camera randomization")
    elif args.camera_scenario == "combined":
        camera_randomizer = CameraRandomizer(
            CameraPresets.surveillance_camera("main_camera", randomization_mode="combined"),
            seed=args.seed,
        )
        log.info("Using combined camera randomization")
    else:
        camera_randomizer = CameraRandomizer(
            CameraPresets.surveillance_camera("main_camera", randomization_mode="combined"),
            seed=args.seed,
        )
        log.info("Using default camera randomization")

    camera_randomizer.bind_handler(handler)
    camera_randomizers = [camera_randomizer]

    # Get initial object properties for comparison
    cube_properties = cube_randomizer.get_properties()
    sphere_properties = sphere_randomizer.get_properties()
    franka_properties = franka_randomizer.get_properties()

    # Get initial material and light properties for comparison
    initial_cube_physical = cube_material_randomizer.get_physical_properties()
    initial_sphere_physical = sphere_material_randomizer.get_physical_properties()

    # Get initial light properties from all light randomizers
    initial_light_properties = {}
    for i, randomizer in enumerate(light_randomizers):
        try:
            properties = randomizer.get_light_properties()
            initial_light_properties[f"light_{i}"] = properties
        except Exception as e:
            log.warning(f"Failed to get initial properties for light {i}: {e}")
            initial_light_properties[f"light_{i}"] = {}

    # Get initial camera properties from all camera randomizers
    initial_camera_properties = {}
    for i, randomizer in enumerate(camera_randomizers):
        try:
            properties = randomizer.get_camera_properties()
            initial_camera_properties[f"camera_{i}"] = properties
        except Exception as e:
            log.warning(f"Failed to get initial properties for camera {i}: {e}")
            initial_camera_properties[f"camera_{i}"] = {}

    # Store initial values for comparison
    initial_values = {
        "cube_physical": initial_cube_physical,
        "sphere_physical": initial_sphere_physical,
        "lights": initial_light_properties,
        "cameras": initial_camera_properties,
    }

    # Run object randomization using unified approach
    log_randomization_header("OBJECT RANDOMIZATION", "Unified ObjectRandomizer approach")

    # Randomize cube (grasping target preset)
    log.info("Cube (grasping target preset):")
    cube_randomizer()
    cube_new_props = cube_randomizer.get_properties()
    if "mass" in cube_new_props and "mass" in cube_properties:
        log_randomization_result(
            "Object",
            "cube",
            "mass",
            cube_properties["mass"],
            cube_new_props["mass"],
            "kg",
        )
    if "friction" in cube_new_props and "friction" in cube_properties:
        log_randomization_result(
            "Object",
            "cube",
            "friction",
            cube_properties["friction"][0],
            cube_new_props["friction"][0],
            "",
        )
    if "position" in cube_new_props and "position" in cube_properties:
        log_randomization_result(
            "Object",
            "cube",
            "position",
            cube_properties["position"],
            cube_new_props["position"],
            "m",
        )

    # Randomize sphere (bouncy object preset)
    log.info("Sphere (bouncy object preset):")
    sphere_randomizer()
    sphere_new_props = sphere_randomizer.get_properties()
    if "mass" in sphere_new_props and "mass" in sphere_properties:
        log_randomization_result(
            "Object",
            "sphere",
            "mass",
            sphere_properties["mass"],
            sphere_new_props["mass"],
            "kg",
        )
    if "restitution" in sphere_new_props and "restitution" in sphere_properties:
        log_randomization_result(
            "Object",
            "sphere",
            "restitution",
            sphere_properties["restitution"],
            sphere_new_props["restitution"],
            "",
        )
    if "position" in sphere_new_props and "position" in sphere_properties:
        log_randomization_result(
            "Object",
            "sphere",
            "position",
            sphere_properties["position"],
            sphere_new_props["position"],
            "m",
        )

    # Randomize franka (robot base preset)
    log.info("Franka (robot base preset):")
    franka_randomizer()
    franka_new_props = franka_randomizer.get_properties()
    if "mass" in franka_new_props and "mass" in franka_properties:
        log_randomization_result(
            "Object",
            "franka",
            "mass",
            franka_properties["mass"],
            franka_new_props["mass"],
            "kg",
        )
    if "friction" in franka_new_props and "friction" in franka_properties:
        log_randomization_result(
            "Object",
            "franka",
            "friction",
            franka_properties["friction"][0],
            franka_new_props["friction"][0],
            "",
        )

    # run material randomization
    log_randomization_header("MATERIAL RANDOMIZATION", "Visual appearance + physics properties")

    log.info("Cube (Wood material with MDL):")
    cube_material_randomizer()
    randomized_cube_physical = cube_material_randomizer.get_physical_properties()
    log.info("  Applied: Wood MDL texture + Physics properties")
    if "friction" in initial_values["cube_physical"] and "friction" in randomized_cube_physical:
        log_randomization_result(
            "Material",
            "cube",
            "friction",
            initial_values["cube_physical"]["friction"][0],
            randomized_cube_physical["friction"][0],
            "",
        )

    log.info("Sphere (Rubber material with PBR):")
    sphere_material_randomizer()
    randomized_sphere_physical = sphere_material_randomizer.get_physical_properties()
    log.info("  Applied: Rubber PBR + Physics (high bounce)")
    if "friction" in initial_values["sphere_physical"] and "friction" in randomized_sphere_physical:
        log_randomization_result(
            "Material",
            "sphere",
            "friction",
            initial_values["sphere_physical"]["friction"][0],
            randomized_sphere_physical["friction"][0],
            "",
        )
    if "restitution" in initial_values["sphere_physical"] and "restitution" in randomized_sphere_physical:
        log_randomization_result(
            "Material",
            "sphere",
            "restitution",
            initial_values["sphere_physical"]["restitution"][0],
            randomized_sphere_physical["restitution"][0],
            "",
        )

    log.info("Box (Metal material with MDL):")
    try:
        box_material_randomizer()
        log.info("  Applied: Metal MDL texture + Physics properties")
    except Exception as e:
        log.warning(f"  Metal material randomization failed: {e}")
        log.info("  This is expected if MDL files are not available")

    # run light randomization for all lights in the scenario
    log_randomization_header(
        "LIGHT RANDOMIZATION",
        f"{args.lighting_scenario} scenario with {len(light_randomizers)} lights",
    )

    # Initial light randomization
    for i, randomizer in enumerate(light_randomizers):
        try:
            # Get initial properties
            initial_light_props = randomizer.get_light_properties()

            # Apply randomization
            randomizer()

            # Get new properties
            new_light_props = randomizer.get_light_properties()

            light_name = randomizer.cfg.light_name
            log.info(f"Light {i + 1}: {light_name}")

            # Log changes
            if "intensity" in initial_light_props and "intensity" in new_light_props:
                log_randomization_result(
                    "Light",
                    light_name,
                    "intensity",
                    initial_light_props["intensity"],
                    new_light_props["intensity"],
                    "cd",
                )
            if "color" in initial_light_props and "color" in new_light_props:
                log_randomization_result(
                    "Light",
                    light_name,
                    "color",
                    initial_light_props["color"],
                    new_light_props["color"],
                    "RGB",
                )
            if "position" in initial_light_props and "position" in new_light_props:
                log_randomization_result(
                    "Light",
                    light_name,
                    "position",
                    initial_light_props["position"],
                    new_light_props["position"],
                    "m",
                )

        except Exception as e:
            log.warning(f"  Light {i + 1} randomization failed: {e}")

    # run camera randomization for all cameras in the scenario
    log_randomization_header(
        "CAMERA RANDOMIZATION",
        f"{args.camera_scenario} mode with {len(camera_randomizers)} cameras",
    )

    # Initial camera randomization
    for i, randomizer in enumerate(camera_randomizers):
        try:
            # Get initial properties
            initial_camera_props = randomizer.get_camera_properties()

            # Apply randomization
            randomizer()

            # Get new properties
            new_camera_props = randomizer.get_camera_properties()

            camera_name = randomizer.cfg.camera_name
            log.info(f"Camera {i + 1}: {camera_name}")

            # Log changes
            if "position" in initial_camera_props and "position" in new_camera_props:
                log_randomization_result(
                    "Camera",
                    camera_name,
                    "position",
                    initial_camera_props["position"],
                    new_camera_props["position"],
                    "m",
                )
            if "focal_length" in initial_camera_props and "focal_length" in new_camera_props:
                log_randomization_result(
                    "Camera",
                    camera_name,
                    "focal_length",
                    initial_camera_props["focal_length"],
                    new_camera_props["focal_length"],
                    "cm",
                )
            if "look_at" in initial_camera_props and "look_at" in new_camera_props:
                log_randomization_result(
                    "Camera",
                    camera_name,
                    "look_at",
                    initial_camera_props["look_at"],
                    new_camera_props["look_at"],
                    "m",
                )

        except Exception as e:
            log.warning(f"  Camera {i + 1} randomization failed: {e}")

    # Run simulation for a few steps with video recording
    log_randomization_header("SIMULATION", "Running with periodic re-randomization")

    for step in range(100):
        log.debug(f"Simulation step {step}")
        handler.simulate()
        obs = handler.get_states(mode="tensor")
        obs_saver.add(obs)

        # Apply randomization every 10 steps to show material and lighting changes very frequently
        if step % 10 == 0 and step > 0:
            log.info(f"Step {step}: Re-applying all randomizations")
            try:
                # Randomize objects using unified approach
                for i, randomizer in enumerate(object_randomizers):
                    randomizer()
                    log.info(f"  Applied ObjectRandomizer {i + 1}")

                # Randomize materials
                cube_material_randomizer()
                sphere_material_randomizer()
                box_material_randomizer()
                log.info("  Applied material randomization (3 objects)")

                # Randomize all lights in the scenario
                for randomizer in light_randomizers:
                    randomizer()
                log.info(f"  Applied light randomization ({len(light_randomizers)} lights)")

                # Randomize all cameras in the scenario
                for randomizer in camera_randomizers:
                    randomizer()
                log.info(f"  Applied camera randomization ({len(camera_randomizers)} cameras)")

            except Exception as e:
                log.warning(f"  Randomization failed at step {step}: {e}")

    # Save video and close
    log_randomization_header("COMPLETION", "Saving results and cleanup")
    log.info("Saving video and closing simulation...")
    obs_saver.save()
    handler.close()
    log.info("Domain randomization demo completed successfully!")


def main():
    @configclass
    class Args:
        """Arguments for the domain randomization demo."""

        robot: str = "franka"

        ## Handlers
        sim: Literal["isaacsim"] = "isaacsim"

        ## Lighting scenarios
        lighting_scenario: Literal["default", "indoor_room", "outdoor_scene", "studio", "demo"] = "default"
        """Choose lighting scenario: default, indoor_room, outdoor_scene, studio, or demo (for testing)"""

        ## Camera randomization modes
        camera_scenario: Literal[
            "combined",
            "position_only",
            "orientation_only",
            "look_at_only",
            "intrinsics_only",
            "image_only",
        ] = "combined"
        """Choose camera randomization mode: combined, position_only, orientation_only, look_at_only, intrinsics_only, or image_only"""

        ## Others
        num_envs: int = 1
        headless: bool = False
        seed: int | None = None
        """Random seed for reproducible randomization. If None, uses random seed."""

        def __post_init__(self):
            """Post-initialization configuration."""
            log.info(f"Args: {self}")

    args = tyro.cli(Args)
    """Main function to run the domain randomization demo."""
    log.info("Starting Domain Randomization Demo")
    log.info("This demo showcases:")

    # Object randomization info
    log.info("  - ObjectRandomizer approach:")
    log.info("    * Cube: Grasping target (mass + friction + position + rotation)")
    log.info("    * Sphere: Bouncy object (mass + restitution + varied position)")
    log.info("    * Franka: Robot base (mass + friction + minimal pose adjustment)")

    log.info("  - Advanced material randomization with combined mode:")
    log.info("    * Cube: Wood (MDL + physics)")
    log.info("    * Sphere: Rubber (PBR + physics, high bounce)")
    log.info("    * Box: Metal (MDL + physics)")
    log.info("  - Advanced lighting randomization using LightScenarios:")
    log.info(f"    * Scenario: {args.lighting_scenario}")
    if args.lighting_scenario == "indoor_room":
        log.info("    * 3 lights: ceiling_light, window_light, desk_lamp")
    elif args.lighting_scenario == "outdoor_scene":
        log.info("    * 2 lights: sun_light, sky_light")
    elif args.lighting_scenario == "studio":
        log.info("    * 3 lights: key_light, fill_light, rim_light")
    elif args.lighting_scenario == "demo":
        log.info("    * 3 lights: rainbow_light (COLORS), disco_light (SHADOWS), shadow_light (MORE SHADOWS)")
    else:
        log.info("    * 2 lights: main_light, ambient_light")
    log.info("  - Camera randomization (micro-adjustment mode):")
    log.info(f"    * Mode: {args.camera_scenario}")
    if args.camera_scenario == "combined":
        log.info("    * ALL: position + orientation + intrinsics (small adjustments from current)")
    elif args.camera_scenario == "position_only":
        log.info("    * POSITION: small camera position adjustments")
    elif args.camera_scenario == "orientation_only":
        log.info("    * ORIENTATION: small rotation adjustments (pitch/yaw/roll deltas)")
    elif args.camera_scenario == "look_at_only":
        log.info("    * LOOK-AT: small target point adjustments (where camera looks)")
    elif args.camera_scenario == "intrinsics_only":
        log.info("    * INTRINSICS: focal length, aperture changes")
    elif args.camera_scenario == "image_only":
        log.info("    * IMAGE: resolution and aspect ratio changes")
    else:
        log.info("    * DEFAULT: combined randomization (micro-adjustments)")
    log.info("  - Flexible and extensible configuration system")
    log.info("  - Video recording and comprehensive logging")
    log.info("  - Reproducible results with --seed argument")
    log.info("")
    log.info("Try different lighting scenarios:")
    log.info("  --lighting-scenario demo        # For testing with maximum visual changes")
    log.info("  --lighting-scenario indoor_room # 3-light indoor setup")
    log.info("  --lighting-scenario outdoor_scene # 2-light outdoor setup")
    log.info("  --lighting-scenario studio      # 3-light studio setup")
    log.info("")
    log.info("Try different camera randomization modes (micro-adjustment by default):")
    log.info("  --camera-scenario position_only    # Small position adjustments from current")
    log.info("  --camera-scenario orientation_only # Small rotation adjustments (pitch/yaw/roll deltas)")
    log.info("  --camera-scenario look_at_only     # Small target point adjustments (where camera looks)")
    log.info("  --camera-scenario intrinsics_only  # Only randomize focal length/aperture")
    log.info("  --camera-scenario image_only       # Only randomize resolution/aspect ratio")
    log.info("  --camera-scenario combined         # All micro-adjustments (default)")
    log.info("Note: Camera uses micro-adjustment mode (delta-based) to avoid jarring position changes")

    log.info("For reproducible results:")
    log.info("  --seed 42                       # Use specific seed for reproducibility")
    log.info("  --seed 123                      # Different seed for different random sequences")
    log.info("")

    # Run IsaacSim demo
    run_domain_randomization(args)
    log.info("\nRandomization demo completed! Check the logs above for detailed results.")
    log.info(f"Video saved to: get_started/output/12_domain_randomization_{args.sim}.mp4")


if __name__ == "__main__":
    main()
