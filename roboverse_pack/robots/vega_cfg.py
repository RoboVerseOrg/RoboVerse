from __future__ import annotations

from typing import Literal

from metasim.scenario.robot import BaseActuatorCfg, RobotCfg
from metasim.utils import configclass


@configclass
class VegaCfg(RobotCfg):
    """Configuration for the Vega Humanoid Robot (vega-1).

    The Vega is a full-body humanoid robot with:
    - Mobile base with wheels
    - Torso with 3 DOF
    - Head (fixed)
    - Two 7-DOF arms (left and right)
    - Two 5-finger dexterous hands (left and right)
    - Various sensors (cameras, lidar, IMU, ultrasonic)
    """

    name: str = "vega"
    num_joints: int = 45  # Total movable joints (excluding fixed joints)
    fix_base_link: bool = True  # Humanoid robots typically have fixed base in simulation

    # Asset paths
    urdf_path: str = "roboverse_pack/robots/robots_vega/humanoid/vega_1/vega.urdf"
    usd_path: str = "roboverse_pack/robots/robots_vega/humanoid/vega_1/vega/vega.usd"

    # Physical properties
    enabled_gravity: bool = False  # Disable gravity for default setup

    # ==================== Actuator Configuration ====================
    actuators: dict[str, BaseActuatorCfg] = {
        # Base wheels - continuous joints need higher stiffness for smooth rotation
        "B_wheel_j1": BaseActuatorCfg(velocity_limit=12.0, torque_limit=16.0, stiffness=1e4, damping=1e3),
        "B_wheel_j2": BaseActuatorCfg(velocity_limit=12.0, torque_limit=16.0, stiffness=1e4, damping=1e3),
        "R_wheel_j1": BaseActuatorCfg(velocity_limit=3.0, torque_limit=6.0, stiffness=5e3, damping=500),
        "R_wheel_j2": BaseActuatorCfg(velocity_limit=12.0, torque_limit=16.0, stiffness=1e4, damping=1e3),
        "L_wheel_j1": BaseActuatorCfg(velocity_limit=3.0, torque_limit=6.0, stiffness=5e3, damping=500),
        "L_wheel_j2": BaseActuatorCfg(velocity_limit=12.0, torque_limit=16.0, stiffness=1e4, damping=1e3),
        # Torso - high torque joints need high stiffness
        "torso_j1": BaseActuatorCfg(
            velocity_limit=0.9, torque_limit=700.0, stiffness=1e6, damping=1e5
        ),  # Increased stiffness for stability
        "torso_j2": BaseActuatorCfg(
            velocity_limit=0.9, torque_limit=380.0, stiffness=1e6, damping=1e5
        ),  # Increased stiffness for stability
        "torso_j3": BaseActuatorCfg(
            velocity_limit=0.9, torque_limit=380.0, stiffness=1e6, damping=1e5
        ),  # Increased stiffness for stability
        # # Left arm - progressive stiffness from base to tip
        "L_arm_j1": BaseActuatorCfg(
            velocity_limit=2.4, torque_limit=150.0, stiffness=5e4, damping=5e3
        ),  # Increased for stability
        "L_arm_j2": BaseActuatorCfg(
            velocity_limit=2.4, torque_limit=150.0, stiffness=5e4, damping=5e3
        ),  # Increased for stability
        "L_arm_j3": BaseActuatorCfg(velocity_limit=2.7, torque_limit=80.0, stiffness=2e4, damping=2e3),
        "L_arm_j4": BaseActuatorCfg(velocity_limit=2.7, torque_limit=80.0, stiffness=1e4, damping=1e3),
        "L_arm_j5": BaseActuatorCfg(velocity_limit=2.7, torque_limit=25.0, stiffness=5e3, damping=500),
        "L_arm_j6": BaseActuatorCfg(velocity_limit=2.7, torque_limit=25.0, stiffness=5e3, damping=500),
        "L_arm_j7": BaseActuatorCfg(velocity_limit=2.7, torque_limit=25.0, stiffness=5e3, damping=500),
        # Right arm - progressive stiffness from base to tip
        "R_arm_j1": BaseActuatorCfg(
            velocity_limit=2.4, torque_limit=150.0, stiffness=5e4, damping=5e3
        ),  # Increased for stability
        "R_arm_j2": BaseActuatorCfg(
            velocity_limit=2.4, torque_limit=150.0, stiffness=5e4, damping=5e3
        ),  # Increased for stability
        "R_arm_j3": BaseActuatorCfg(velocity_limit=2.7, torque_limit=80.0, stiffness=2e4, damping=2e3),
        "R_arm_j4": BaseActuatorCfg(velocity_limit=2.7, torque_limit=80.0, stiffness=1e4, damping=1e3),
        "R_arm_j5": BaseActuatorCfg(velocity_limit=2.7, torque_limit=25.0, stiffness=5e3, damping=500),
        "R_arm_j6": BaseActuatorCfg(velocity_limit=2.7, torque_limit=25.0, stiffness=5e3, damping=500),
        "R_arm_j7": BaseActuatorCfg(velocity_limit=2.7, torque_limit=25.0, stiffness=5e3, damping=500),
        # Left hand - Thumb
        "L_th_j0": BaseActuatorCfg(velocity_limit=6.28, torque_limit=1.4, stiffness=300, damping=22),
        "L_th_j1": BaseActuatorCfg(velocity_limit=6.28, torque_limit=1.4, stiffness=300, damping=22),
        "L_th_j2": BaseActuatorCfg(velocity_limit=6.28, torque_limit=1.1, stiffness=260, damping=20),
        # Left hand - Fingers
        "L_ff_j1": BaseActuatorCfg(velocity_limit=6.28, torque_limit=0.9, stiffness=320, damping=22),
        "L_ff_j2": BaseActuatorCfg(velocity_limit=6.28, torque_limit=0.9, stiffness=320, damping=22),
        "L_mf_j1": BaseActuatorCfg(velocity_limit=6.28, torque_limit=0.9, stiffness=320, damping=22),
        "L_mf_j2": BaseActuatorCfg(velocity_limit=6.28, torque_limit=0.9, stiffness=320, damping=22),
        "L_rf_j1": BaseActuatorCfg(velocity_limit=6.28, torque_limit=0.9, stiffness=320, damping=22),
        "L_rf_j2": BaseActuatorCfg(velocity_limit=6.28, torque_limit=0.9, stiffness=320, damping=22),
        "L_lf_j1": BaseActuatorCfg(velocity_limit=6.28, torque_limit=0.9, stiffness=320, damping=22),
        "L_lf_j2": BaseActuatorCfg(velocity_limit=6.28, torque_limit=0.9, stiffness=320, damping=22),
        # Right hand - Thumb
        "R_th_j0": BaseActuatorCfg(velocity_limit=6.28, torque_limit=1.4, stiffness=300, damping=22),
        "R_th_j1": BaseActuatorCfg(velocity_limit=6.28, torque_limit=1.4, stiffness=300, damping=22),
        "R_th_j2": BaseActuatorCfg(velocity_limit=6.28, torque_limit=1.1, stiffness=260, damping=20),
        # Right hand - Fingers
        "R_ff_j1": BaseActuatorCfg(velocity_limit=6.28, torque_limit=0.9, stiffness=320, damping=22),
        "R_ff_j2": BaseActuatorCfg(velocity_limit=6.28, torque_limit=0.9, stiffness=320, damping=22),
        "R_mf_j1": BaseActuatorCfg(velocity_limit=6.28, torque_limit=0.9, stiffness=320, damping=22),
        "R_mf_j2": BaseActuatorCfg(velocity_limit=6.28, torque_limit=0.9, stiffness=320, damping=22),
        "R_rf_j1": BaseActuatorCfg(velocity_limit=6.28, torque_limit=0.9, stiffness=320, damping=22),
        "R_rf_j2": BaseActuatorCfg(velocity_limit=6.28, torque_limit=0.9, stiffness=320, damping=22),
        "R_lf_j1": BaseActuatorCfg(velocity_limit=6.28, torque_limit=0.9, stiffness=320, damping=22),
        "R_lf_j2": BaseActuatorCfg(velocity_limit=6.28, torque_limit=0.9, stiffness=320, damping=22),
    }

    # ==================== Joint Limits ====================
    # Joint angle limits from URDF (in radians)
    joint_limits: dict[str, tuple[float, float]] = {
        # Base wheels - locked at 0.0
        "B_wheel_j1": (0.0, 0.0),
        "B_wheel_j2": (0.0, 0.0),
        "R_wheel_j1": (0.0, 0.0),
        "R_wheel_j2": (0.0, 0.0),
        "L_wheel_j1": (0.0, 0.0),
        "L_wheel_j2": (0.0, 0.0),
        # Torso
        "torso_j1": (0.2, 0.2),
        "torso_j2": (0.5, 1.0),
        "torso_j3": (0.0, 0.0),
        # Left arm
        "L_arm_j1": (-3.071, 3.071),
        "L_arm_j2": (-0.453, 1.553),
        "L_arm_j3": (-3.071, 3.071),
        "L_arm_j4": (-3.071, 0.244),
        "L_arm_j5": (-3.071, 3.071),
        "L_arm_j6": (-1.396, 1.396),
        "L_arm_j7": (-1.378, 1.117),
        # Right arm - locked at 0.0
        "R_arm_j1": (0.0, 0.0),
        "R_arm_j2": (0.0, 0.0),
        "R_arm_j3": (0.0, 0.0),
        "R_arm_j4": (0.0, 0.0),
        "R_arm_j5": (0.0, 0.0),
        "R_arm_j6": (0.0, 0.0),
        "R_arm_j7": (0.0, 0.0),
        # Left hand - Thumb
        "L_th_j0": (-0.0158, 1.605),
        "L_th_j1": (-0.3468, 0.1834),
        "L_th_j2": (-0.4298, 0.2731),
        # Left hand - Fingers
        "L_ff_j1": (-1.0946, 0.2891),
        "L_ff_j2": (-1.2101, 0.3681),
        "L_mf_j1": (-1.0844, 0.2801),
        "L_mf_j2": (-1.2026, 0.3533),
        "L_rf_j1": (-1.0154, 0.2840),
        "L_rf_j2": (-1.1156, 0.3599),
        "L_lf_j1": (-1.0118, 0.2811),
        "L_lf_j2": (-1.1073, 0.4014),
        # Right hand - locked at 0.0
        "R_th_j0": (0.0, 0.0),
        "R_th_j1": (0.0, 0.0),
        "R_th_j2": (0.0, 0.0),
        # Right hand - Fingers (locked at 0.0)
        "R_ff_j1": (0.0, 0.0),
        "R_ff_j2": (0.0, 0.0),
        "R_mf_j1": (0.0, 0.0),
        "R_mf_j2": (0.0, 0.0),
        "R_rf_j1": (0.0, 0.0),
        "R_rf_j2": (0.0, 0.0),
        "R_lf_j1": (0.0, 0.0),
        "R_lf_j2": (0.0, 0.0),
    }

    # ==================== Control Types ====================
    # Default to position control for all joints
    control_type: dict[str, Literal["position", "effort"]] = {
        # Base wheels
        "B_wheel_j1": "position",
        "B_wheel_j2": "position",
        "R_wheel_j1": "position",
        "R_wheel_j2": "position",
        "L_wheel_j1": "position",
        "L_wheel_j2": "position",
        # Torso
        "torso_j1": "position",
        "torso_j2": "position",
        "torso_j3": "position",
        # Head
        # "head_j1": "position",
        # "head_j2": "position",
        # "head_j3": "position",
        # Left arm
        "L_arm_j1": "position",
        "L_arm_j2": "position",
        "L_arm_j3": "position",
        "L_arm_j4": "position",
        "L_arm_j5": "position",
        "L_arm_j6": "position",
        "L_arm_j7": "position",
        # Right arm
        "R_arm_j1": "position",
        "R_arm_j2": "position",
        "R_arm_j3": "position",
        "R_arm_j4": "position",
        "R_arm_j5": "position",
        "R_arm_j6": "position",
        "R_arm_j7": "position",
        # Left hand
        "L_th_j0": "position",
        "L_th_j1": "position",
        "L_th_j2": "position",
        "L_ff_j1": "position",
        "L_ff_j2": "position",
        "L_mf_j1": "position",
        "L_mf_j2": "position",
        "L_rf_j1": "position",
        "L_rf_j2": "position",
        "L_lf_j1": "position",
        "L_lf_j2": "position",
        # Right hand
        "R_th_j0": "position",
        "R_th_j1": "position",
        "R_th_j2": "position",
        "R_ff_j1": "position",
        "R_ff_j2": "position",
        "R_mf_j1": "position",
        "R_mf_j2": "position",
        "R_rf_j1": "position",
        "R_rf_j2": "position",
        "R_lf_j1": "position",
        "R_lf_j2": "position",
    }

    # ==================== End Effector Configuration ====================
    # End effector link names from URDF
    # URDF structure: L_arm_l7 (last movable) -> L_arm_l8 (fixed, with geometry) -> L_ee (fixed, marker) -> L_hand_base (fixed, hand)
    # Since collapse_fixed_joints=False, all fixed links are preserved in body_names
    # L_arm_l8 is the best choice as it:
    #   - Is the last arm link with actual geometry (visual/collision meshes)
    #   - Represents the natural end of the arm structure
    #   - Should reliably exist in body_names
    #   - Is the boundary between arm and hand
    ee_body_name: str = (
        "L_arm_l7"  # Last arm link with geometry (fixed joint, but preserved when collapse_fixed_joints=False)
    )
    # Alternative options:
    # - "L_arm_l7": Last movable arm link (revolute joint), but less ideal as it's before the arm tip geometry
    # - "L_ee": End effector marker link (fixed, lightweight marker only, same position as L_arm_l8)
    # - "L_hand_base": Hand base link (fixed, part of hand structure, not arm)

    # ==================== Gripper Configuration ====================
    # Left hand gripper open/close positions (all left hand finger joints)
    # Order: L_th_j0, L_th_j1, L_th_j2, L_ff_j1, L_ff_j2, L_mf_j1, L_mf_j2, L_rf_j1, L_rf_j2, L_lf_j1, L_lf_j2
    gripper_close_q: list[float] = [
        1.20,  # L_th_j0: thumb abduction (close towards palm)
        -0.30,  # L_th_j1: thumb flexion (negative closes)
        -0.40,  # L_th_j2: thumb tip flexion
        -0.95,  # L_ff_j1: index finger proximal (negative closes)
        -1.05,  # L_ff_j2: index finger distal
        -0.95,  # L_mf_j1: middle finger proximal
        -1.05,  # L_mf_j2: middle finger distal
        -0.90,  # L_rf_j1: ring finger proximal
        -1.00,  # L_rf_j2: ring finger distal
        -0.90,  # L_lf_j1: little finger proximal
        -1.00,  # L_lf_j2: little finger distal
    ]  # Closed hand (approx. 85-90% of lower joint limits)
    gripper_open_q: list[float] = [
        0.20,  # L_th_j0: relaxed abduction
        0.0,  # L_th_j1: thumb flexion open
        0.0,  # L_th_j2: thumb tip open
        0.0,  # L_ff_j1: index finger proximal open
        0.0,  # L_ff_j2: index finger distal open
        0.0,  # L_mf_j1: middle finger proximal open
        0.0,  # L_mf_j2: middle finger distal open
        0.0,  # L_rf_j1: ring finger proximal open
        0.0,  # L_rf_j2: ring finger distal open
        0.0,  # L_lf_j1: little finger proximal open
        0.0,  # L_lf_j2: little finger distal open
    ]  # Open hand (neutral positions)

    # ==================== Default Joint Positions ====================
    # Default home positions (can be customized based on use case)
    default_joint_positions: dict[str, float] = {
        # Base wheels - neutral
        "B_wheel_j1": 0.0,
        "B_wheel_j2": 0.0,
        "R_wheel_j1": 0.0,
        "R_wheel_j2": 0.0,
        "L_wheel_j1": 0.0,
        "L_wheel_j2": 0.0,
        # Torso - upright
        "torso_j1": 0.0,
        "torso_j2": 0.5,
        "torso_j3": 0.0,
        # Head - forward looking
        # "head_j1": 0.0,
        # "head_j2": 0.0,
        # "head_j3": 0.0,
        # Left arm - neutral pose
        "L_arm_j1": 0.0,
        "L_arm_j2": 0.0,
        "L_arm_j3": 0.0,
        "L_arm_j4": 0.0,
        "L_arm_j5": 0.0,
        "L_arm_j6": 0.0,
        "L_arm_j7": 0.0,
        # Right arm - neutral pose
        "R_arm_j1": 0.0,
        "R_arm_j2": 0.0,
        "R_arm_j3": 0.0,
        "R_arm_j4": 0.0,
        "R_arm_j5": 0.0,
        "R_arm_j6": 0.0,
        "R_arm_j7": 0.0,
        # Left hand - open
        "L_th_j0": 0.0,
        "L_th_j1": 0.0,
        "L_th_j2": 0.0,
        "L_ff_j1": 0.0,
        "L_ff_j2": 0.0,
        "L_mf_j1": 0.0,
        "L_mf_j2": 0.0,
        "L_rf_j1": 0.0,
        "L_rf_j2": 0.0,
        "L_lf_j1": 0.0,
        "L_lf_j2": 0.0,
        # Right hand - open
        "R_th_j0": 0.0,
        "R_th_j1": 0.0,
        "R_th_j2": 0.0,
        "R_ff_j1": 0.0,
        "R_ff_j2": 0.0,
        "R_mf_j1": 0.0,
        "R_mf_j2": 0.0,
        "R_rf_j1": 0.0,
        "R_rf_j2": 0.0,
        "R_lf_j1": 0.0,
        "R_lf_j2": 0.0,
    }
