"""Helper functions for humanoid robots, including h1 and h1_simple_hand."""

import torch


def torso_upright(envstate, robot_name: str):
    """Returns projection from z-axes of torso to the z-axes of world."""
    xmat = quaternion_to_matrix(envstate["robots"][robot_name]["body"]["pelvis"]["rot"])
    return xmat[2, 2]


def head_height(envstate, robot_name: str):
    """Returns the height of the head, actually the neck."""
    raise NotImplementedError("head_height is not implemented for isaacgym and isaaclab")
    # return envstate["robots"][robot_name]["head"]["pos"][2]  # Good for mujoco, but isaacgym and isaaclab don't have head site


def neck_height(envstate, robot_name: str):
    """Returns the height of the neck."""
    # print(envstate["robots"][robot_name].keys())
    # exit()
    return (
        envstate["robots"][robot_name]["body"]["left_shoulder_roll_link"]["pos"][2]
        + envstate["robots"][robot_name]["body"]["right_shoulder_roll_link"]["pos"][2]
    ) / 2


def left_foot_height(envstate, robot_name: str):
    """Returns the height of the left foot."""
    # return envstate[f"{_METASIM_SITE_PREFIX}left_foot"]["pos"][2] # Only for mujoco
    return envstate["robots"][robot_name]["body"]["left_ankle_link"]["pos"][2]


def right_foot_height(envstate, robot_name: str):
    """Returns the height of the right foot."""
    # return envstate[f"{_METASIM_SITE_PREFIX}right_foot"]["pos"][2] # Only for mujoco
    return envstate["robots"][robot_name]["body"]["right_ankle_link"]["pos"][2]


def robot_position(envstate, robot_name: str):
    """Returns position of the robot."""
    return envstate["robots"][robot_name]["pos"]


def robot_velocity(envstate, robot_name: str):
    """Returns the velocity of the robot."""
    return envstate["robots"][robot_name]["vel"]


def robot_rotation(envstate, robot_name: str):
    """Returns the rotation of the robot."""
    return envstate["robots"][robot_name]["rot"]


def torso_vertical_orientation(envstate, robot_name: str):
    """Returns the z-projection of the torso orientation matrix."""
    xmat = quaternion_to_matrix(envstate["robots"][robot_name]["body"]["pelvis"]["rot"])
    return xmat[2, :]


def actuator_forces(envstate, robot_name: str):
    """Returns a copy of the forces applied by the actuators."""
    return (
        torch.tensor([x for x in envstate["robots"][robot_name]["dof_torque"].values()])
        if envstate["robots"][robot_name].get("dof_torque", None) is not None
        else torch.zeros(len(envstate["robots"][robot_name]["dof_pos"]))
    )


def left_hand_position(envstate, robot_name: str):
    """Returns the position of the left hand."""
    # return envstate[f"{_METASIM_SITE_PREFIX}left_hand"]["pos"] # Only for mujoco
    return envstate["robots"][robot_name]["body"]["left_elbow_link"]["pos"]


def left_hand_velocity(envstate, robot_name: str):
    """Returns the velocity of the left hand."""
    # return envstate[f"{_METASIM_BODY_PREFIX}left_hand"]["left_hand_subtreelinvel"] # Only for mujoco
    return envstate["robots"][robot_name]["body"]["left_elbow_link"]["vel"]


def left_hand_orientation(envstate, robot_name: str):
    """Returns the orientation of the left hand."""
    # return envstate[f"{_METASIM_SITE_PREFIX}left_hand"]["rot"] # Only for mujoco
    return envstate["robots"][robot_name]["body"]["left_elbow_link"]["rot"]


def right_hand_position(envstate, robot_name: str):
    """Returns the position of the right hand."""
    # return envstate[f"{_METASIM_SITE_PREFIX}right_hand"]["pos"] # Only for mujoco
    return envstate["robots"][robot_name]["body"]["right_elbow_link"]["pos"]


def right_hand_velocity(envstate, robot_name: str):
    """Returns the velocity of the right hand."""
    # return envstate[f"{_METASIM_BODY_PREFIX}right_hand"]["right_hand_subtreelinvel"] # Only for mujoco
    return envstate["robots"][robot_name]["body"]["right_elbow_link"]["vel"]


def right_hand_orientation(envstate, robot_name: str):
    """Returns the orientation of the right hand."""
    # return envstate[f"{_METASIM_SITE_PREFIX}right_hand"]["rot"] # Only for mujoco
    return envstate["robots"][robot_name]["body"]["right_elbow_link"]["rot"]


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
