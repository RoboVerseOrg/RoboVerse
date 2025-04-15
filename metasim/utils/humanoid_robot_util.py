"""Helper functions for humanoid robots, including h1 and h1_simple_hand."""

import torch


def torso_upright(envstate, robot_name: str):
    """Returns projection from z-axes of torso to the z-axes of world."""
    xmat = quaternion_to_matrix(envstate["robots"][robot_name]["body"]["pelvis"]["rot"])
    return xmat[2, 2]


def torso_upright_tensor(envstate, robot_name: str):
    """Returns projection from z-axes of torso to the z-axes of world."""
    robot_body_name = envstate.robots[robot_name].body_names
    body_id = robot_body_name.index("pelvis")
    xmat = quaternion_to_matrix_tensor(envstate.robots[robot_name].body_state[:, body_id, 3:7])
    return xmat[:, 2, 2]


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


def neck_height_tensor(envstate, robot_name: str):
    """Returns the height of the neck."""
    robot_body_name = envstate.robots[robot_name].body_names
    body_id_l = robot_body_name.index("left_shoulder_roll_link")
    body_id_r = robot_body_name.index("right_shoulder_roll_link")
    body_pos_l = envstate.robots[robot_name].body_state[:, body_id_l, 2]
    body_pos_r = envstate.robots[robot_name].body_state[:, body_id_r, 2]
    return (body_pos_l + body_pos_r) / 2


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


def robot_position_tensor(envstate, robot_name: str):
    """Returns position of the robot."""
    return envstate.robots[robot_name].root_state[:, 0:3]


def robot_velocity(envstate, robot_name: str):
    """Returns the velocity of the robot."""
    return envstate["robots"][robot_name]["vel"]


def robot_velocity_tensor(envstate, robot_name: str):
    """Returns the velocity of the robot."""
    return envstate.robots[robot_name].root_state[:, 7:10]


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


def actuator_forces_tensor(envstate, robot_name: str):
    """Returns a copy of the forces applied by the actuators."""
    return (
        envstate.robots[robot_name].joint_effort_target
        if envstate.robots[robot_name].joint_effort_target is not None
        else torch.zeros_like(envstate.robots[robot_name].joint_pos)
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


def quaternion_to_matrix_tensor(quat: torch.Tensor) -> torch.Tensor:
    """Converts a batch of quaternions to rotation matrices.

    Args:
        quat: Quaternion tensor of shape (..., 4) where last dim is (w,x,y,z)

    Returns:
        Rotation matrix tensor of shape (..., 3, 3)
    """
    # Split quaternion components along last dimension
    q_w = quat[..., 0]
    q_x = quat[..., 1]
    q_y = quat[..., 2]
    q_z = quat[..., 3]

    # Compute common terms
    qx2 = q_x**2
    qy2 = q_y**2
    qz2 = q_z**2
    qxqy = q_x * q_y
    qxqz = q_x * q_z
    qyqz = q_y * q_z
    qwqx = q_w * q_x
    qwqy = q_w * q_y
    qwqz = q_w * q_z

    # Build rotation matrix
    R = torch.stack(
        [
            torch.stack([1 - 2 * (qy2 + qz2), 2 * (qxqy - qwqz), 2 * (qxqz + qwqy)], dim=-1),
            torch.stack([2 * (qxqy + qwqz), 1 - 2 * (qx2 + qz2), 2 * (qyqz - qwqx)], dim=-1),
            torch.stack([2 * (qxqz - qwqy), 2 * (qyqz + qwqx), 1 - 2 * (qx2 + qy2)], dim=-1),
        ],
        dim=-2,
    )

    return R
