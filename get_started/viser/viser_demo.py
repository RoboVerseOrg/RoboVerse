"""Comprehensive Viser visualization demo with multiple control modes.

This unified demo replaces multiple separate demos and supports:
- Static/Dynamic scene visualization
- Joint control
- IK control
- Trajectory playback
- Camera controls

All features can be enabled/disabled via command line arguments.
"""

from __future__ import annotations

from typing import Literal

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass


import rootutils
import torch
import tyro
from loguru import logger as log
from rich.logging import RichHandler

rootutils.setup_root(__file__, pythonpath=True)
log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])


from metasim.constants import PhysicStateType
from metasim.scenario.cameras import PinholeCameraCfg
from metasim.scenario.objects import (
    ArticulationObjCfg,
    PrimitiveCubeCfg,
    PrimitiveSphereCfg,
    RigidObjCfg,
)
from metasim.scenario.scenario import ScenarioCfg
from metasim.utils import configclass
from metasim.utils.hf_util import check_and_download_recursive
from metasim.utils.ik_solver import process_gripper_command, setup_ik_solver
from metasim.utils.obs_utils import ObsSaver
from metasim.utils.setup_util import get_handler
from metasim.utils.state import state_tensor_to_nested


@configclass
class Args:
    """Arguments for the viser demo."""

    robot: str = "franka"
    """Robot to use in the demo."""

    ## Simulator
    sim: Literal[
        "isaacsim",
        "isaacgym",
        "isaaclab",
        "genesis",
        "pybullet",
        "sapien2",
        "sapien3",
        "mujoco",
    ] = "mujoco"
    """Simulator backend to use."""

    num_envs: int = 1
    """Number of parallel environments."""

    headless: bool = False
    """Use viser for visualization, not simulator's viewer."""

    ## Control modes
    dynamic: bool = True
    """Enable dynamic simulation with IK motion."""

    enable_joint_control: bool = False
    """Enable interactive joint control GUI."""

    enable_ik_control: bool = True
    """Enable interactive IK control GUI."""

    enable_trajectory: bool = False
    """Enable trajectory playback GUI."""

    trajectory_path: str | None = None
    """Path to trajectory file to load (e.g., .pkl.gz file). If None, no trajectory is loaded."""

    ## IK solver (only used if dynamic=True)
    solver: Literal["curobo", "pyroki"] = "pyroki"
    """IK solver to use for dynamic motion."""

    ## Recording (only used if dynamic=True)
    save_video: bool = False
    """Save video of dynamic simulation."""

    def __post_init__(self):
        """Post-initialization configuration."""
        log.info(f"Args: {self}")


def extract_states_from_obs(obs, handler, key):
    """Extract states from observation tensor.

    Args:
        obs: TensorState observation
        handler: Simulator handler
        key: "objects" or "robots"

    Returns:
        dict[name] = {"pos": ..., "rot": ..., "dof_pos": ...}
    """
    env_states = state_tensor_to_nested(handler, obs)
    result = {}
    if env_states and len(env_states) > 0:
        state = env_states[0]
        if key in state:
            for name, item in state[key].items():
                state_dict = {}
                if "pos" in item and item["pos"] is not None:
                    state_dict["pos"] = (
                        item["pos"].cpu().numpy().tolist() if hasattr(item["pos"], "cpu") else list(item["pos"])
                    )
                if "rot" in item and item["rot"] is not None:
                    state_dict["rot"] = (
                        item["rot"].cpu().numpy().tolist() if hasattr(item["rot"], "cpu") else list(item["rot"])
                    )
                if "dof_pos" in item and item["dof_pos"] is not None:
                    state_dict["dof_pos"] = item["dof_pos"]
                result[name] = state_dict
    return result


def download_urdf_files(scenario):
    """Download URDF files for all objects and robots in the scenario."""
    urdf_paths = []

    # Collect URDF paths from objects
    for obj in scenario.objects:
        if hasattr(obj, "urdf_path") and obj.urdf_path:
            urdf_paths.append(obj.urdf_path)

    # Collect URDF paths from robots
    for robot in scenario.robots:
        if hasattr(robot, "urdf_path") and robot.urdf_path:
            urdf_paths.append(robot.urdf_path)

    # Download URDF files and all related mesh files recursively
    if urdf_paths:
        log.info(f"Downloading {len(urdf_paths)} URDF files and all related meshes...")
        check_and_download_recursive(urdf_paths, n_processes=16)
        log.info("URDF files and meshes download completed!")


def main():
    args = tyro.cli(Args)

    # ========================================================================
    # Setup Scenario
    # ========================================================================
    scenario = ScenarioCfg(
        robots=[args.robot],
        simulator=args.sim,
        headless=args.headless,
        num_envs=args.num_envs,
    )

    # Add cameras
    scenario.cameras = [
        PinholeCameraCfg(
            name="camera",
            width=1024,
            height=1024,
            pos=(1.5, -1.5, 1.5),
            look_at=(0.0, 0.0, 0.0),
        )
    ]

    # Add objects for demonstration
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
        RigidObjCfg(
            name="bbq_sauce",
            scale=(200, 200, 200),
            physics=PhysicStateType.RIGIDBODY,
            usd_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/bbq_sauce/usd/bbq_sauce.usd",
            urdf_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/bbq_sauce/urdf/bbq_sauce.urdf",
            mjcf_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/bbq_sauce/mjcf/bbq_sauce.xml",
        ),
        ArticulationObjCfg(
            name="box_base",
            fix_base_link=True,
            usd_path="roboverse_data/assets/rlbench/close_box/box_base/usd/box_base.usd",
            urdf_path="roboverse_data/assets/rlbench/close_box/box_base/urdf/box_base_unique.urdf",
            mjcf_path="roboverse_data/assets/rlbench/close_box/box_base/mjcf/box_base_unique.mjcf",
        ),
    ]

    log.info(f"Using simulator: {args.sim}")
    handler = get_handler(scenario)

    # Set initial states
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
                "bbq_sauce": {
                    "pos": torch.tensor([0.7, -0.3, 0.14]),
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
    ]

    handler.set_states(init_states * scenario.num_envs)
    obs = handler.get_states(mode="tensor")

    # Setup IK solver if needed for dynamic mode
    ik_solver = None
    if args.dynamic:
        log.info(f"Using IK solver: {args.solver}")
        robot = scenario.robots[0]
        ik_solver = setup_ik_solver(robot, args.solver)

    # ========================================================================
    # Setup Viser Visualization
    # ========================================================================
    from metasim.utils.viser.viser_util import ViserVisualizer

    # Download URDF files before visualization
    download_urdf_files(scenario)

    # Initialize the viser server
    visualizer = ViserVisualizer(port=8080)
    visualizer.add_grid()
    visualizer.add_frame("/world_frame")

    # Extract states from objects and robots
    default_object_states = extract_states_from_obs(obs, handler, "objects")
    default_robot_states = extract_states_from_obs(obs, handler, "robots")

    # Visualize all objects and robots
    visualizer.visualize_scenario_items(scenario.objects, default_object_states)
    visualizer.visualize_scenario_items(scenario.robots, default_robot_states)

    log.info("Viser has been initialized, visit http://localhost:8080 to view the scene!")

    # Scene info
    scene_info = ["Scene includes:"]
    for obj in scenario.objects:
        scene_info.append(f"  • {obj.name} ({type(obj).__name__})")
    for robot in scenario.robots:
        scene_info.append(f"  • {robot.name} ({type(robot).__name__})")
    log.info("\n".join(scene_info))

    # Enable camera controls
    visualizer.enable_camera_controls(
        initial_position=[1.5, -1.5, 1.5],
        render_width=1024,
        render_height=1024,
        look_at_position=[0, 0, 0],
        initial_fov=71.28,
    )

    # Enable optional control modes
    if args.enable_joint_control:
        visualizer.enable_joint_control()
        log.info("Joint control enabled")

    if args.enable_ik_control:
        robot_config = scenario.robots[0]
        robot_name = robot_config.name
        success = visualizer.setup_ik_solver(robot_name, robot_config, handler, solver=args.solver)
        if success:
            visualizer.enable_ik_control()
            log.info("IK control enabled")
        else:
            log.warning("Failed to setup IK solver for IK control")

    if args.enable_trajectory:
        visualizer.enable_trajectory_playback()
        log.info("Trajectory playback enabled")

        # Load trajectory if path is provided
        if args.trajectory_path:
            import os

            if os.path.exists(args.trajectory_path):
                log.info(f"Loading trajectory from: {args.trajectory_path}")
                success = visualizer.load_trajectory(args.trajectory_path)
                if success:
                    log.info("Trajectory loaded successfully")
                    # Get available trajectories
                    trajectories = visualizer.get_available_trajectories()
                    if trajectories:
                        log.info(f"Available trajectories: {trajectories}")
                        # Auto-set first trajectory
                        robot_name, _ = trajectories[0]
                        visualizer.set_current_trajectory(robot_name, 0)
                        log.info(f"Set trajectory for robot: {robot_name}")
                else:
                    log.warning(f"Failed to load trajectory from {args.trajectory_path}")
            else:
                log.warning(f"Trajectory file not found: {args.trajectory_path}")
        else:
            log.info("No trajectory path provided. Use --trajectory-path to load a trajectory file.")

    # ========================================================================
    # Dynamic Simulation Loop (if enabled)
    # ========================================================================
    if args.dynamic:
        log.info("Starting dynamic simulation with IK motion...")

        # Setup video recording if requested
        obs_saver = None
        if args.save_video:
            obs_saver = ObsSaver(video_path=f"get_started/output/viser_demo_{args.sim}.mp4")
            obs_saver.add(obs)

        robot = scenario.robots[0]

        for step in range(200):
            states = handler.get_states(mode="tensor")

            # Generate target end-effector pose
            if robot.name == "franka":
                x_target = 0.3 + 0.1 * (step / 100)
                y_target = 0.5 - 0.5 * (step / 100)
                z_target = 0.6 - 0.2 * (step / 100)
                ee_pos_target = torch.zeros((args.num_envs, 3), device="cuda:0")
                for i in range(args.num_envs):
                    if i % 3 == 0:
                        ee_pos_target[i] = torch.tensor([x_target, 0.0, 0.6], device="cuda:0")
                    elif i % 3 == 1:
                        ee_pos_target[i] = torch.tensor([0.3, y_target, 0.6], device="cuda:0")
                    else:
                        ee_pos_target[i] = torch.tensor([0.3, 0.0, z_target], device="cuda:0")
                ee_quat_target = torch.tensor(
                    [[0.0, 1.0, 0.0, 0.0]] * args.num_envs,
                    device="cuda:0",
                )
            elif robot.name == "kinova_gen3_robotiq_2f85":
                ee_pos_target = torch.tensor([[0.2 + 0.2 * (step / 100), 0.0, 0.4]], device="cuda:0").repeat(
                    args.num_envs, 1
                )
                ee_quat_target = torch.tensor(
                    [[0.0, 0.0, 1.0, 0.0]] * args.num_envs,
                    device="cuda:0",
                )
            else:
                # Default motion for other robots
                ee_pos_target = torch.tensor([[0.3, 0.0, 0.6]], device="cuda:0").repeat(args.num_envs, 1)
                ee_quat_target = torch.tensor([[0.0, 1.0, 0.0, 0.0]] * args.num_envs, device="cuda:0")

            # Get current robot state for IK seeding
            curr_robot_q = states.robots[robot.name].joint_pos.cuda()

            # Solve IK
            q_solution, ik_succ = ik_solver.solve_ik_batch(ee_pos_target, ee_quat_target, curr_robot_q)

            # Process gripper command (fixed open position)
            gripper_binary = torch.ones(scenario.num_envs, device="cuda:0")  # all open
            gripper_widths = process_gripper_command(gripper_binary, robot, "cuda:0")

            # Compose full joint command
            actions = ik_solver.compose_joint_action(q_solution, gripper_widths, curr_robot_q, return_dict=True)

            handler.set_dof_targets(actions)
            handler.simulate()
            obs = handler.get_states(mode="tensor")

            # Settle for first step
            if step == 0:
                for _ in range(50):
                    handler.simulate()
                    obs = handler.get_states(mode="tensor")

            if obs_saver:
                obs_saver.add(obs)

            # Update visualization
            object_states = extract_states_from_obs(obs, handler, "objects")
            robot_states = extract_states_from_obs(obs, handler, "robots")

            for name, state in object_states.items():
                visualizer.update_item_pose(name, state)
            for name, state in robot_states.items():
                visualizer.update_item_pose(name, state)

            visualizer.refresh_camera_view()

        if obs_saver:
            obs_saver.save()
            log.info(f"Video saved to {obs_saver.video_path}")

        log.info("Dynamic simulation completed!")

    # ========================================================================
    # Print Usage Instructions
    # ========================================================================
    log.info("")
    log.info("=" * 70)
    log.info("Viser Demo Ready! Open http://localhost:8080 in your browser")
    log.info("=" * 70)

    mode_description = "Static Scene" if not args.dynamic else "Dynamic Scene (simulation completed)"
    log.info(f"Mode: {mode_description}")

    log.info("\nEnabled Features:")
    log.info("  • Camera Controls: Always enabled")
    if args.enable_joint_control:
        log.info("  • Joint Control: Use sliders to control robot joints")
    if args.enable_ik_control:
        log.info("  • IK Control: Use target position/orientation controls")
    if args.enable_trajectory:
        log.info("  • Trajectory Playback: Load and play trajectories")

    if not any([args.enable_joint_control, args.enable_ik_control, args.enable_trajectory]):
        log.info("  • No interactive controls enabled")
        log.info("    Tip: Use --enable-joint-control, --enable-ik-control, or --enable-trajectory")

    log.info("\nUseful Commands:")
    log.info("  • Rotate view: Left mouse drag")
    log.info("  • Pan view: Right mouse drag or Shift+Left drag")
    log.info("  • Zoom: Scroll wheel")

    if args.enable_trajectory:
        log.info("\nTo load trajectory:")
        log.info("  visualizer.load_trajectory('path/to/trajectory.pkl.gz')")

    log.info("=" * 70)

    # Keep running
    try:
        while True:
            pass
    except KeyboardInterrupt:
        log.info("\nShutting down...")


if __name__ == "__main__":
    main()
