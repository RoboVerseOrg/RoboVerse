from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pinocchio as pin


class RobotWrapper:
    """This class does not take mimic joint into consideration."""

    def __init__(self, urdf_path: str, use_collision=False, use_visual=False):
        # Create robot model and data
        self.model: pin.Model = pin.buildModelFromUrdf(urdf_path)  # type: ignore
        self.data: pin.Data = self.model.createData()  # type: ignore

        if use_visual or use_collision:
            raise NotImplementedError

        self.q0 = pin.neutral(self.model)  # type: ignore
        if self.model.nv != self.model.nq:
            raise NotImplementedError("Can not handle robot with special joint.")

    # -------------------------------------------------------------------------- #
    # Robot property
    # -------------------------------------------------------------------------- #
    @property
    def joint_names(self) -> list[str]:
        """Get the names of all joints in the robot model.

        Returns:
            list[str]: List of all joint names.
        """
        return list(self.model.names)

    @property
    def dof_joint_names(self) -> list[str]:
        """Get the names of joints that have degrees of freedom.

        Returns:
            List[str]: List of joint names that have DOFs.
        """
        nqs = self.model.nqs
        return [name for i, name in enumerate(self.model.names) if nqs[i] > 0]

    @property
    def dof(self) -> int:
        """Get the number of degrees of freedom of the robot.

        Returns:
            int: Number of degrees of freedom.
        """
        return self.model.nq

    @property
    def link_names(self) -> list[str]:
        """Get the names of all links in the robot model.

        Returns:
            list[str]: List of all link names.
        """
        link_names = []
        for i, frame in enumerate(self.model.frames):
            link_names.append(frame.name)
        return link_names

    @property
    def joint_limits(self):
        """Get the joint limits of the robot.

        Returns:
            np.ndarray: Array of shape (n_joints, 2) containing the lower and upper joint limits.
        """
        lower = self.model.lowerPositionLimit
        upper = self.model.upperPositionLimit
        return np.stack([lower, upper], axis=1)

    # -------------------------------------------------------------------------- #
    # Query function
    # -------------------------------------------------------------------------- #
    def get_joint_index(self, name: str):
        """Get the index of a joint by name.

        Args:
            name (str): The name of the joint.

        Returns:
            int: The index of the joint.
        """
        return self.dof_joint_names.index(name)

    def get_link_index(self, name: str):
        """Get the index of a link by name.

        Args:
            name (str): The name of the link.

        Returns:
            int: The index of the link.
        """
        if name not in self.link_names:
            raise ValueError(f"{name} is not a link name. Valid link names: \n{self.link_names}")
        return self.model.getFrameId(name, pin.BODY)  # type: ignore

    def get_joint_parent_child_frames(self, joint_name: str):
        """Get the parent and child frames of a joint.

        Args:
            joint_name (str): The name of the joint.

        Returns:
            tuple[int, int]: The index of the parent and child frames.
        """
        joint_id = self.model.getFrameId(joint_name)
        parent_id = self.model.frames[joint_id].parent
        child_id = -1
        for idx, frame in enumerate(self.model.frames):
            if frame.previousFrame == joint_id:
                child_id = idx
        if child_id == -1:
            raise ValueError(f"Can not find child link of {joint_name}")

        return parent_id, child_id

    # -------------------------------------------------------------------------- #
    # Kinematics function
    # -------------------------------------------------------------------------- #
    def compute_forward_kinematics(self, qpos: npt.NDArray):
        """Compute forward kinematics for the given joint configuration.

        Args:
            qpos (npt.NDArray): Joint configuration vector.
        """
        pin.forwardKinematics(self.model, self.data, qpos)  # type: ignore

    def get_link_pose(self, link_id: int) -> npt.NDArray:
        """Get the pose of a link.

        Args:
            link_id (int): The index of the link.

        Returns:
            npt.NDArray: 4x4 transformation matrix.
        """
        pose: pin.SE3 = pin.updateFramePlacement(self.model, self.data, link_id)  # type: ignore
        return pose.homogeneous

    def get_link_pose_inv(self, link_id: int) -> npt.NDArray:
        """Get the inverse pose of a link.

        Args:
            link_id (int): The index of the link.

        Returns:
            npt.NDArray: 4x4 transformation matrix.
        """
        pose: pin.SE3 = pin.updateFramePlacement(self.model, self.data, link_id)  # type: ignore
        return pose.inverse().homogeneous

    def compute_single_link_local_jacobian(self, qpos, link_id: int) -> npt.NDArray:
        """Compute the Jacobian of a single link.

        Args:
            qpos (npt.NDArray): Joint configuration vector.
            link_id (int): The index of the link.

        Returns:
            npt.NDArray: Jacobian matrix.
        """
        J = pin.computeFrameJacobian(self.model, self.data, qpos, link_id)  # type: ignore
        return J
