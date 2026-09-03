"""Direct control of Inspire Hand - Stable version based on random_action_notik.py"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

import tyro
from loguru import logger as log
from rich.logging import RichHandler

log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

import torch

from metasim.scenario.cameras import PinholeCameraCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.utils.obs_utils import ObsSaver
from metasim.utils.setup_util import get_handler, get_robot


@dataclass
class Args:
    """Arguments.

    NOTE: PSIHand robots (psihand_left/psihand_right) have known compatibility issues with IsaacGym.
    Use MuJoCo, Genesis, or other simulators for PSIHand instead.
    """

    robot: str = "inspire_hand_left"  # or "inspire_hand_right, brainco_hand_right/left, psihand_left/right"
    sim: Literal["isaaclab", "isaacgym", "genesis", "pybullet", "mujoco", "sapien2", "sapien3"] = "mujoco"
    num_envs: int = 1
    num_steps: int = 10  # Number of different random poses


def main():
    args = tyro.cli(Args)
    log.info(f"Args: {args}")

    # Get robot configuration
    robot = get_robot(args.robot)

    # Create scenario configuration with custom dt
    scenario = ScenarioCfg(
        robots=[robot],
        cameras=[
            PinholeCameraCfg(
                width=640,
                height=480,
                pos=(0.2, -0.3, 0.2),  # Front-facing camera position
                look_at=(0.0, 0.0, 0.1),  # Looking at hand center
            )
        ],
        simulator=args.sim,
        num_envs=args.num_envs,
    )

    # Set simulation timestep to 0.002 seconds (2ms per step) - MuJoCo default
    # Smaller timestep = more stable physics for complex models with constraints
    scenario.sim_params.dt = 0.002

    # Create handler
    handler = get_handler(scenario)

    # Set initial pose for IsaacGym to lift hand above ground
    # Only set position, not rotation, to avoid NaN issues with some robots
    # MuJoCo uses MJCF-defined pose to avoid constraint violations
    if args.sim == "isaacgym":
        init_states = [
            {
                "robots": {
                    robot.name: {
                        "pos": torch.tensor([0.0, 0.0, 0.05]),
                        "rot": torch.tensor([-0.7071, 0.7071, 0.0, 0.0]),
                    }
                },
                "objects": {},  # Empty objects dict required by IsaacGym handler
            }
        ]
        handler.set_states(init_states)

    # Get initial states
    states = handler.get_states(mode="tensor")

    # Create output directory and initialize ObsSaver
    os.makedirs("get_started/output/dexhands", exist_ok=True)
    obs_saver = ObsSaver(video_path=f"get_started/output/dexhands/1_diff_dex_hand_{args.sim}_{args.robot}.mp4")
    obs_saver.add(states)

    log.info(f"Robot: {robot.name}")
    log.info(f"Number of joints: {len(robot.actuators)}")
    log.info(f"Joint names: {list(robot.actuators.keys())}")

    # Get all joint names from the handler (actual order in simulation)
    all_joint_names_from_handler = handler.get_joint_names(robot.name, sort=True)
    log.info(f"\nJoint names from handler (actual sim order): {all_joint_names_from_handler}")
    log.info(f"Number of joints in handler: {len(all_joint_names_from_handler)}")

    # Identify which joints are actuated (not passive)
    # CRITICAL: Only select joints where fully_actuated is explicitly True
    # This excludes both fully_actuated=False (passive joints) and any None values
    actuated_joints = [name for name in all_joint_names_from_handler if robot.actuators[name].fully_actuated is True]
    log.info(f"\nActuated joints (controllable): {len(actuated_joints)}")
    log.info(f"Actuated joint names: {actuated_joints}")

    # Identify passive joints
    passive_joints = [name for name in all_joint_names_from_handler if robot.actuators[name].fully_actuated is False]
    log.info(f"\nPassive joints (coupled/mimic): {len(passive_joints)}")
    log.info(f"Passive joint names: {passive_joints}")

    # Get joint limits for ONLY actuated joints
    j_names_actuated = actuated_joints
    j_limits = torch.tensor(
        [[robot.joint_limits[name][0], robot.joint_limits[name][1]] for name in j_names_actuated], device="cuda:0"
    )
    j_ranges = j_limits[:, 1] - j_limits[:, 0]
    j_centers = (j_limits[:, 0] + j_limits[:, 1]) / 2

    num_actuators = len(j_names_actuated)  # Only actuated joints
    num_envs = args.num_envs

    log.info("\nStarting random control demo...")
    log.info(f"Will generate {args.num_steps} different random poses")
    log.info("Each pose will be held for 60 simulation steps\n")

    # Run simulation
    for step_i in range(args.num_steps):
        log.info("=" * 60)
        log.info(f"Step {step_i + 1}/{args.num_steps}: Generating new random pose...")

        # Generate random joint positions (only for actuated joints)
        random_offset = (torch.rand((num_envs, num_actuators), device="cuda:0") - 0.5) * 0.6  # 60% of range
        q_actuated = j_centers.unsqueeze(0) + random_offset * j_ranges.unsqueeze(0)
        q_actuated = torch.clamp(q_actuated, j_limits[:, 0].unsqueeze(0), j_limits[:, 1].unsqueeze(0))

        # For MuJoCo with equality constraints, we need to set passive joint targets
        # to match the expected positions from the mimic relationships.
        # This prevents conflicts between actuator control and equality constraints.
        # The key is: set passive joints to their COUPLED values, not arbitrary values.

        actions = []
        for i_env in range(num_envs):
            # Start with actuated joints
            dof_targets = dict(zip(j_names_actuated, q_actuated[i_env].tolist(), strict=False))

            # Calculate passive joint targets based on mimic relationships
            # This is needed for ALL simulators to ensure consistency
            # Thumb: intermediate = pitch * 1.6, distal = pitch * 2.4
            # Fingers: intermediate = proximal * 1.0
            for passive_joint in passive_joints:
                if "thumb_intermediate" in passive_joint:
                    # Find the pitch joint for this hand
                    pitch_joint = passive_joint.replace("intermediate", "proximal_pitch")
                    dof_targets[passive_joint] = dof_targets.get(pitch_joint, 0.0) * 1.6
                elif "thumb_distal" in passive_joint:
                    pitch_joint = passive_joint.replace("distal", "proximal_pitch")
                    dof_targets[passive_joint] = dof_targets.get(pitch_joint, 0.0) * 2.4
                elif "intermediate" in passive_joint:
                    # For index, middle, ring, pinky fingers
                    proximal_joint = passive_joint.replace("intermediate", "proximal")
                    dof_targets[passive_joint] = dof_targets.get(proximal_joint, 0.0) * 1.0
                else:
                    # Fallback to default position
                    dof_targets[passive_joint] = robot.default_joint_positions[passive_joint]

            actions.append({robot.name: {"dof_pos_target": dof_targets}})

        # Log joint positions (only actuated joints)
        log.info("Target joint positions (actuated only):")
        for i, (name, value) in enumerate(zip(j_names_actuated[:3], q_actuated[0][:3].tolist(), strict=False)):
            log.info(f"  {name}: {value:.3f}")
        log.info("  ... (showing first 3 actuated joints)")

        # Hold this pose for 60 steps (~0.24 seconds at dt=0.004)
        # Testing shows joints converge within 40 steps, so 60 is sufficient
        for hold_step in range(60):
            handler.set_dof_targets(actions)
            handler.simulate()

            # Debug: check actual joint positions at key steps
            if hold_step in [0, 30, 59]:  # Check at start, middle, and end
                log.info(f"  Holding pose... step {hold_step}/60")
                # Get current state to see actual joint positions
                current_states = handler.get_states(mode="tensor")
                # Access through robots dict -> RobotState -> joint_pos
                actual_q_all = current_states.robots[robot.name].joint_pos[0]  # All joints

                # Show only actuated joints for comparison
                actuated_indices = [all_joint_names_from_handler.index(name) for name in j_names_actuated]
                actual_q_actuated = actual_q_all[actuated_indices]

                log.info(f"  First 3 actuated joints - Actual: {actual_q_actuated[:3].tolist()}")
                log.info(f"  First 3 actuated joints - Target: {q_actuated[0][:3].tolist()}")

                # Save observation only at the final step of each pose for concise video
                if hold_step == 59:
                    obs_saver.add(current_states)

    log.info("\n" + "=" * 60)
    log.info("Demo completed successfully!")
    log.info("Saving video...")
    obs_saver.save()
    log.info(f"Video saved to: get_started/output/dexhands/1_diff_dex_hand_{args.sim}_{args.robot}.mp4")
    log.info("Closing simulation...")
    handler.close()
    log.info("Done!")


if __name__ == "__main__":
    main()
