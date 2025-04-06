"""Helper functions for humanoid robots, including h1 and h1_simple_hand."""

import torch

_METASIM_BODY_PREFIX = "metasim_body_"
_METASIM_SITE_PREFIX = "metasim_site_"
# robot_name = "h1"
# robot_name = "h1_simple_hand"


def torso_upright(envstate):
    """Returns projection from z-axes of torso to the z-axes of world."""
    xmat = quaternion_to_matrix(envstate[f"{_METASIM_BODY_PREFIX}torso_link"]["rot"])
    return xmat[2, 2]


def head_height(envstate):
    """Returns the height of the torso."""
    # return envstate[f"{_METASIM_SITE_PREFIX}head"]["pos"][2]  # Good for mujoco
    return (
        envstate["metasim_body_left_shoulder_pitch_link"]["pos"][2]
        + envstate["metasim_body_right_shoulder_pitch_link"]["pos"][2]
    ) / 2  # TODO: isaacgym don't have head site
    # return envstate[f"{_METASIM_BODY_PREFIX}mid360_link"]["pos"][2]  # TODO: isaaclab don't have head site


def left_foot_height(envstate):
    """Returns the height of the left foot."""
    return envstate[f"{_METASIM_SITE_PREFIX}left_foot"]["pos"][2]


def right_foot_height(envstate):
    """Returns the height of the right foot."""
    return envstate[f"{_METASIM_SITE_PREFIX}right_foot"]["pos"][2]


def center_of_mass_position(envstate):
    """Returns position of the center-of-mass."""
    return envstate[f"{_METASIM_BODY_PREFIX}pelvis"]["com"]


def center_of_mass_velocity(envstate):
    """Returns the velocity of the center-of-mass."""
    return envstate[f"{_METASIM_BODY_PREFIX}pelvis"]["vel"]


def center_of_mass_rotation(envstate):
    """Returns the rotation of the center-of-mass."""
    return envstate[f"{_METASIM_BODY_PREFIX}pelvis"]["rot"]


def body_velocity(envstate):
    """Returns the velocity of the torso in local frame."""
    return envstate[f"{_METASIM_SITE_PREFIX}com"]["body_velocimeter"]


def imu_position(envstate):
    """Returns the position of the IMU."""
    return envstate[f"{_METASIM_SITE_PREFIX}imu"]["pos"]


def torso_vertical_orientation(envstate):
    """Returns the z-projection of the torso orientation matrix."""
    xmat = quaternion_to_matrix(envstate[f"{_METASIM_BODY_PREFIX}torso_link"]["rot"])
    return xmat[2, :]


@DeprecationWarning
def joint_angles(envstate):
    """Returns the state without global orientation or position."""
    # Skip the 7 DoFs of the free root joint.
    return "joint_angles not implemented. Used for perception in humanoid_bench."


@DeprecationWarning
def joint_velocities(envstate):
    """Returns the joint velocities."""
    return "joint_velocities not implemented. Used for perception in humanoid_bench."


@DeprecationWarning
def control(envstate):
    """Returns a copy of the control signals for the actuators."""
    return "control not implemented. Used for perception in humanoid_bench."


def actuator_forces(envstate, robot_name: str):
    """Returns a copy of the forces applied by the actuators."""
    return torch.tensor([
        (x if x is not None else 0) for x in envstate[f"{robot_name}"]["dof_torque"].values()
    ])  # envstate[f"{robot_name}"]["dof_torque"]


def left_hand_position(envstate):
    """Returns the position of the left hand."""
    return envstate[f"{_METASIM_SITE_PREFIX}left_hand"]["pos"]


def left_hand_velocity(envstate):
    """Returns the velocity of the left hand."""
    return envstate[f"{_METASIM_BODY_PREFIX}left_hand"]["left_hand_subtreelinvel"]


def left_hand_orientation(envstate):
    """Returns the orientation of the left hand."""
    return envstate[f"{_METASIM_SITE_PREFIX}left_hand"]["rot"]


def right_hand_position(envstate):
    """Returns the position of the right hand."""
    return envstate[f"{_METASIM_SITE_PREFIX}right_hand"]["pos"]


def right_hand_velocity(envstate):
    """Returns the velocity of the right hand."""
    return envstate[f"{_METASIM_BODY_PREFIX}right_hand"]["right_hand_subtreelinvel"]


def right_hand_orientation(envstate):
    """Returns the orientation of the right hand."""
    return envstate[f"{_METASIM_SITE_PREFIX}right_hand"]["rot"]


#######################################################
## Extra functions
#######################################################


def z_height(envstate, robot_name: str):
    """Returns the z-coordinate of the robot's position."""
    return envstate[f"{robot_name}"]["pos"][2]


#######################################################
## Helper functions
#######################################################


def quaternion_to_matrix(quat: torch.Tensor):
    """Converts a quaternion to a rotation matrix."""
    q_w, q_x, q_y, q_z = quat
    R = torch.tensor(
        [
            [1 - 2 * (q_y**2 + q_z**2), 2 * (q_x * q_y - q_w * q_z), 2 * (q_x * q_z + q_w * q_y)],
            [2 * (q_x * q_y + q_w * q_z), 1 - 2 * (q_x**2 + q_z**2), 2 * (q_y * q_z - q_w * q_x)],
            [2 * (q_x * q_z - q_w * q_y), 2 * (q_y * q_z + q_w * q_x), 1 - 2 * (q_x**2 + q_y**2)],
        ],
        dtype=quat.dtype,
    )
    return R
