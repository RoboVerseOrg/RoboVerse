from __future__ import annotations

import logging
import time

import numpy as np
from pytransform3d import rotations

from metasim.utils.dex_util.constants import OPERATOR2MANO, HandType
from metasim.utils.dex_util.optimizer import Optimizer
from metasim.utils.dex_util.optimizer_utils import LPFilter


class SeqRetargeting:
    """Sequence retargeting class.

    Args:
        optimizer (Optimizer): The optimizer to use.
        has_joint_limits (bool): Whether to use joint limits.
        lp_filter (LPFilter): The low-pass filter to use.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        has_joint_limits=True,
        lp_filter: LPFilter | None = None,
    ):
        self.optimizer = optimizer
        robot = self.optimizer.robot

        # Joint limit
        self.has_joint_limits = has_joint_limits
        joint_limits = np.ones_like(robot.joint_limits)
        joint_limits[:, 0] = -1e4  # a large value is equivalent to no limit
        joint_limits[:, 1] = 1e4
        if has_joint_limits:
            joint_limits[:] = robot.joint_limits[:]
            self.optimizer.set_joint_limit(joint_limits[self.optimizer.idx_pin2target])
        self.joint_limits = joint_limits[self.optimizer.idx_pin2target]

        # Temporal information
        self.last_qpos = joint_limits.mean(1)[self.optimizer.idx_pin2target].astype(np.float32)
        self.accumulated_time = 0
        self.num_retargeting = 0

        # Filter
        self.filter = lp_filter

        # Warm started
        self.is_warm_started = False

    def warm_start(
        self,
        wrist_pos: np.ndarray,
        wrist_quat: np.ndarray,
        hand_type: HandType = HandType.right,
        is_mano_convention: bool = False,
    ):
        """Initialize the wrist joint pose using analytical computation instead of retargeting optimization.

        This function is specifically for position retargeting with the flying robot hand, i.e. has 6D free joint
        You are not expected to use this function for vector retargeting, e.g. when you are working on teleoperation

        Args:
            wrist_pos: position of the hand wrist, typically from human hand pose
            wrist_quat: quaternion of the hand wrist, the same convention as the operator frame definition if not is_mano_convention
            hand_type: hand type, used to determine the operator2mano matrix
            is_mano_convention: whether the wrist_quat is in mano convention
        """
        # This function can only be used when the first joints of robot are free joints

        if len(wrist_pos) != 3:
            raise ValueError(f"Wrist pos: {wrist_pos} is not a 3-dim vector.")
        if len(wrist_quat) != 4:
            raise ValueError(f"Wrist quat: {wrist_quat} is not a 4-dim vector.")

        operator2mano = OPERATOR2MANO[hand_type] if is_mano_convention else np.eye(3)
        robot = self.optimizer.robot
        target_wrist_pose = np.eye(4)
        target_wrist_pose[:3, :3] = rotations.matrix_from_quaternion(wrist_quat) @ operator2mano.T
        target_wrist_pose[:3, 3] = wrist_pos

        name_list = [
            "dummy_x_translation_joint",
            "dummy_y_translation_joint",
            "dummy_z_translation_joint",
            "dummy_x_rotation_joint",
            "dummy_y_rotation_joint",
            "dummy_z_rotation_joint",
        ]
        wrist_link_id = robot.get_joint_parent_child_frames(name_list[5])[1]

        # Set the dummy joints angles to zero
        old_qpos = robot.q0
        new_qpos = old_qpos.copy()
        for num, joint_name in enumerate(self.optimizer.target_joint_names):
            if joint_name in name_list:
                new_qpos[num] = 0

        robot.compute_forward_kinematics(new_qpos)
        root2wrist = robot.get_link_pose_inv(wrist_link_id)
        target_root_pose = target_wrist_pose @ root2wrist

        euler = rotations.euler_from_matrix(target_root_pose[:3, :3], 0, 1, 2, extrinsic=False)
        pose_vec = np.concatenate([target_root_pose[:3, 3], euler])

        # Find the dummy joints
        for num, joint_name in enumerate(self.optimizer.target_joint_names):
            if joint_name in name_list:
                index = name_list.index(joint_name)
                self.last_qpos[num] = pose_vec[index]

        self.is_warm_started = True

    def retarget(self, ref_value, fixed_qpos=None):
        """Retarget the robot to the reference value.

        Args:
            ref_value (np.ndarray): The reference value.
            fixed_qpos (np.ndarray | None): The fixed joint positions.
        """
        if fixed_qpos is None:
            fixed_qpos = np.array([])
        tic = time.perf_counter()

        qpos = self.optimizer.retarget(
            ref_value=ref_value.astype(np.float32),
            fixed_qpos=fixed_qpos.astype(np.float32),
            last_qpos=np.clip(self.last_qpos, self.joint_limits[:, 0], self.joint_limits[:, 1]),
        )
        self.accumulated_time += time.perf_counter() - tic
        self.num_retargeting += 1
        self.last_qpos = qpos
        robot_qpos = np.zeros(self.optimizer.robot.dof)
        robot_qpos[self.optimizer.idx_pin2fixed] = fixed_qpos
        robot_qpos[self.optimizer.idx_pin2target] = qpos

        if self.optimizer.adaptor is not None:
            robot_qpos = self.optimizer.adaptor.forward_qpos(robot_qpos)

        if self.filter is not None:
            robot_qpos = self.filter.next(robot_qpos)
        return robot_qpos

    def set_qpos(self, robot_qpos: np.ndarray):
        """Set the joint positions.

        Args:
            robot_qpos (np.ndarray): The joint positions.
        """
        target_qpos = robot_qpos[self.optimizer.idx_pin2target]
        self.last_qpos = target_qpos

    def get_qpos(self, fixed_qpos: np.ndarray | None = None):
        """Get the joint positions.

        Args:
            fixed_qpos (np.ndarray): The fixed joint positions.
        """
        robot_qpos = np.zeros(self.optimizer.robot.dof)
        robot_qpos[self.optimizer.idx_pin2target] = self.last_qpos
        if fixed_qpos is not None:
            robot_qpos[self.optimizer.idx_pin2fixed] = fixed_qpos
        return robot_qpos

    def verbose(self):
        """Print the retargeting information."""
        min_value = self.optimizer.opt.last_optimum_value()
        logging.info(f"Retargeting {self.num_retargeting} times takes: {self.accumulated_time}s")
        logging.info(f"Last distance: {min_value}")

    def reset(self):
        """Reset the retargeting."""
        self.last_qpos = self.joint_limits.mean(1).astype(np.float32)
        self.num_retargeting = 0
        self.accumulated_time = 0

    @property
    def joint_names(self):
        """Get the joint names of the robot.

        Returns:
            list[str]: List of joint names.
        """
        return self.optimizer.robot.dof_joint_names
