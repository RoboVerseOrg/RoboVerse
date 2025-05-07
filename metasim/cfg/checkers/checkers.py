## ruff: noqa: D102

from __future__ import annotations

from dataclasses import MISSING
from typing import Literal

import torch
from loguru import logger as log

from metasim.cfg.objects import BaseObjCfg
from metasim.utils.configclass import configclass
from metasim.utils.math import euler_xyz_from_quat, matrix_from_quat, quat_from_matrix
from metasim.utils.tensor_util import tensor_to_str

from .base_checker import BaseChecker
from .detectors import BaseDetector

try:
    from metasim.sim import BaseSimHandler
except:
    pass


@configclass
class EmptyChecker(BaseChecker):
    """A checker that always returns False."""

    def reset(self, handler: BaseSimHandler, env_ids: list[int] | None = None):
        pass

    def check(self, handler: BaseSimHandler) -> torch.BoolTensor:
        return torch.zeros(handler.num_envs, dtype=torch.bool, device=handler.device)

    def get_debug_viewers(self) -> list[BaseObjCfg]:
        return []


@configclass
class DetectedChecker(BaseChecker):
    """Check if the object with ``obj_name`` is detected by the detector.

    This class should always be used with a detector.
    """

    obj_name: str = MISSING
    """The name of the object to be checked."""
    detector: BaseDetector = MISSING
    """The detector to be used."""
    ignore_if_first_check_success: bool = False
    """If True, the checker will ignore the success of the first check. Default to False."""

    def reset(self, handler: BaseSimHandler, env_ids: list[int] | None = None):
        self._first_check = torch.ones(handler.num_envs, dtype=torch.bool, device=handler.device)  # True
        self._ignore = torch.zeros(handler.num_envs, dtype=torch.bool, device=handler.device)  # False
        self.detector.reset(handler, env_ids=env_ids)

    def check(self, handler: BaseSimHandler) -> torch.BoolTensor:
        success = self.detector.is_detected(handler, self.obj_name)
        if self.ignore_if_first_check_success:
            self._ignore[self._first_check & success] = True
        self._first_check[self._first_check] = False

        success[self._ignore] = False
        return success

    def get_debug_viewers(self) -> list[BaseObjCfg]:
        return self.detector.get_debug_viewers()


@configclass
class JointPosChecker(BaseChecker):
    """Check if the joint with ``joint_name`` of the object with ``obj_name`` has position ``radian_threshold`` units.

    - ``mode`` should be one of "ge", "le". "ge" for greater than or equal to, "le" for less than or equal to.
    - ``radian_threshold`` is the threshold for the joint position.
    """

    obj_name: str = MISSING
    """The name of the object to be checked."""
    joint_name: str = MISSING
    """The name of the joint to be checked."""
    mode: Literal["ge", "le"] = MISSING
    """The mode of the joint position checker. "ge" for greater than or equal to, "le" for less than or equal to."""
    radian_threshold: float = MISSING
    """The threshold for the joint position. (in radian)"""

    def check(self, handler: BaseSimHandler) -> torch.BoolTensor:
        dof_pos = handler.get_dof_pos(self.obj_name, self.joint_name)
        log.debug(f"Joint {self.joint_name} of object {self.obj_name} has position {tensor_to_str(dof_pos)}")
        if self.mode == "ge":
            return dof_pos >= self.radian_threshold
        elif self.mode == "le":
            return dof_pos <= self.radian_threshold
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    def get_debug_viewers(self) -> list[BaseObjCfg]:
        return []


@configclass
class JointPosShiftChecker(BaseChecker):
    """Check if the joint with ``joint_name`` of the object with ``obj_name`` was moved more than ``threshold`` units.

    - ``threshold`` is negative for moving towards the negative direction and positive for moving towards the positive direction.
    """

    obj_name: str = MISSING
    """The name of the object to be checked."""
    joint_name: str = MISSING
    """The name of the joint to be checked."""
    threshold: float = MISSING
    """The threshold for the joint position. (in radian)"""

    def reset(self, handler: BaseSimHandler, env_ids: list[int] | None = None):
        if env_ids is None:
            env_ids = list(range(handler.num_envs))

        if not hasattr(self, "init_joint_pos"):
            self.init_joint_pos = torch.zeros(handler.num_envs, dtype=torch.float32, device=handler.device)

        self.init_joint_pos[env_ids] = handler.get_dof_pos(self.obj_name, self.joint_name, env_ids=env_ids)

    def check(self, handler: BaseSimHandler) -> torch.BoolTensor:
        cur_joint_pos = handler.get_dof_pos(self.obj_name, self.joint_name)
        joint_pos_diff = cur_joint_pos - self.init_joint_pos

        log.debug(f"Joint {self.joint_name} of object {self.obj_name} moved {tensor_to_str(joint_pos_diff)} units")

        if self.threshold > 0:
            return joint_pos_diff >= self.threshold
        else:
            return joint_pos_diff <= self.threshold


@configclass
class RotationShiftChecker(BaseChecker):
    """Check if the object with ``obj_name`` was rotated more than ``radian_threshold`` radians around the given ``axis``.

    - ``radian_threshold`` is negative for clockwise rotations and positive for counter-clockwise rotations.
    - ``radian_threshold`` should be in the range of [-pi, pi].
    - ``axis`` should be one of "x", "y", "z". default is "z".
    """

    ## ref: https://github.com/mees/calvin_env/blob/c7377a6485be43f037f4a0b02e525c8c6e8d24b0/calvin_env/envs/tasks.py#L54
    obj_name: str = MISSING
    """The name of the object to be checked."""
    radian_threshold: float = MISSING
    """The threshold for the rotation. (in radian)"""
    axis: Literal["x", "y", "z"] = "z"
    """The axis to detect the rotation around."""

    def reset(self, handler: BaseSimHandler, env_ids: list[int] | None = None):
        if env_ids is None:
            env_ids = list(range(handler.num_envs))

        if not hasattr(self, "init_quat"):
            self.init_quat = torch.zeros(handler.num_envs, 4, dtype=torch.float32, device=handler.device)

        self.init_quat[env_ids] = handler.get_rot(self.obj_name, env_ids=env_ids)

    def check(self, handler: BaseSimHandler) -> torch.BoolTensor:
        cur_quat = handler.get_rot(self.obj_name)
        init_rot_mat = matrix_from_quat(self.init_quat)
        cur_rot_mat = matrix_from_quat(cur_quat)
        rot_diff = torch.matmul(cur_rot_mat, init_rot_mat.transpose(-1, -2))
        x, y, z = euler_xyz_from_quat(quat_from_matrix(rot_diff))
        v = {"x": x, "y": y, "z": z}[self.axis]

        ## Normalize the rotation angle to be within [-pi, pi]
        v[v > torch.pi] -= 2 * torch.pi
        v[v < -torch.pi] += 2 * torch.pi
        assert ((v >= -torch.pi) & (v <= torch.pi)).all()

        log.debug(f"Object {self.obj_name} rotated {tensor_to_str(v / torch.pi * 180)} degrees around {self.axis}-axis")

        if self.radian_threshold > 0:
            return v >= self.radian_threshold
        else:
            return v <= self.radian_threshold


@configclass
class PositionShiftChecker(BaseChecker):
    """Check if the object with ``obj_name`` was moved more than ``distance`` meters in given ``axis``.

    - ``distance`` is negative for moving towards the negative direction and positive for moving towards the positive direction.
    - ``max_distance`` is the maximum distance the object can move.
    - ``axis`` should be one of "x", "y", "z".
    """

    obj_name: str = MISSING
    """The name of the object to be checked."""
    distance: float = MISSING
    """The threshold for the position shift. (in meters)"""
    bounding_distance: float = 1e2
    """The maximum distance the object can move. (in meters)"""
    axis: Literal["x", "y", "z"] = MISSING
    """The axis to detect the position shift along."""

    def reset(self, handler: BaseSimHandler, env_ids: list[int] | None = None):
        if env_ids is None:
            env_ids = list(range(handler.num_envs))

        if not hasattr(self, "init_pos"):
            self.init_pos = torch.zeros((handler.num_envs, 3), dtype=torch.float32, device=handler.device)

        tmp = handler.get_pos(self.obj_name, env_ids=env_ids)
        assert tmp.shape == (len(env_ids), 3)
        self.init_pos[env_ids] = tmp

    def check(self, handler: BaseSimHandler) -> torch.BoolTensor:
        cur_pos = handler.get_pos(self.obj_name)
        if torch.isnan(cur_pos).any():
            log.debug(f"Object {self.obj_name} moved to nan position")
            return torch.ones(cur_pos.shape[0], dtype=torch.bool, device=handler.device)
        dim = {"x": 0, "y": 1, "z": 2}[self.axis]
        dis_diff = cur_pos - self.init_pos
        dim_diff = dis_diff[:, dim]
        tot_dis = torch.norm(dis_diff, dim=-1)
        log.debug(f"Object {self.obj_name} moved {tensor_to_str(dim_diff)} meters in {self.axis} direction")
        if self.distance > 0:
            return (dim_diff >= self.distance) * (tot_dis <= self.bounding_distance)
        else:
            return dim_diff <= self.distance


################################################################################
## The following checkers are deprecated, we should remove them in the future!
################################################################################


## XXX: this checker is hacky, we should remove it!
@configclass
class _JointPosPercentShiftChecker(BaseChecker):
    """Check if the joint with ``joint_name`` of the object with ``obj_name`` was moved more than ``threshold`` percent.

    - ``threshold`` is negative for moving towards the negative direction and positive for moving towards the positive direction.
    """

    obj_name: str = MISSING
    joint_name: str = MISSING
    threshold: float = MISSING
    percentage_target: float = MISSING
    scale: float = 100.0
    type: Literal["prismatic", "revolute"] = "prismatic"

    def reset(self, handler: BaseSimHandler, env_ids: list[int] | None = None):
        if env_ids is None:
            env_ids = list(range(handler.num_envs))

        if not hasattr(self, "init_joint_pos"):
            self.init_joint_pos = torch.zeros(handler.num_envs, dtype=torch.float32, device=handler.device)

        self.init_joint_pos[env_ids] = handler.get_dof_pos(self.obj_name, self.joint_name, env_ids=env_ids)

    def check(self, handler: BaseSimHandler) -> torch.BoolTensor:
        # Get the current joint position as a tensor.
        cur_joint_pos = handler.get_dof_pos(self.obj_name, self.joint_name)

        # Get the joint limits tensor and joint names list.
        joint_limits = handler.get_joint_limits(self.obj_name, self.joint_name)
        joint_lower_limit = joint_limits[:, 0]
        if self.type == "prismatic":
            joint_upper_limit = joint_limits[:, 1] / self.scale
        elif self.type == "revolute":
            joint_upper_limit = joint_limits[:, 1]
        else:
            raise ValueError(f"Invalid joint type: {self.type}")
        # Compute the joint range and the percentage of the position within that range.
        joint_range = joint_upper_limit - joint_lower_limit
        joint_pos_percentage = (cur_joint_pos - joint_lower_limit) / joint_range

        # Check if the current position is within the desired threshold of the target percentage.
        # Use the bitwise '&' operator for element-wise logical AND on tensors.
        condition = (joint_pos_percentage >= self.percentage_target - self.threshold) & (
            joint_pos_percentage <= self.percentage_target + self.threshold
        )

        log.debug(
            f"joint_pos_percentage {tensor_to_str(joint_pos_percentage)}, percentage_target {self.percentage_target}, threshold {self.threshold}"
        )
        return condition


## FIXME: this function is redundant with PositionShiftChecker, we should remove it!
@configclass
class _PositionShiftCheckerWithTolerance(BaseChecker):
    """Check if the object with ``obj_name`` was moved to ``distance`` meters in given ``axis`` with a tolerance of ``tolerance``.

    - ``distance`` is negative for moving towards the negative direction and positive for moving towards the positive direction.
    - ``max_distance`` is the maximum distance the object can move.
    - ``axis`` should be one of "x", "y", "z".
    """

    obj_name: str = MISSING
    distance: float = MISSING
    bounding_distance: float = 1e2
    tolerance: float = 0.01
    axis: Literal["x", "y", "z"] = MISSING

    def reset(self, handler: BaseSimHandler, env_ids: list[int] | None = None):
        if env_ids is None:
            env_ids = list(range(handler.num_envs))

        if not hasattr(self, "init_pos"):
            self.init_pos = torch.zeros((handler.num_envs, 3), dtype=torch.float32, device=handler.device)

        tmp = handler.get_pos(self.obj_name, env_ids=env_ids)
        assert tmp.shape == (len(env_ids), 3)
        self.init_pos[env_ids] = tmp

    def check(self, handler: BaseSimHandler) -> torch.BoolTensor:
        cur_pos = handler.get_pos(self.obj_name)
        cur_vel = handler.get_vel(self.obj_name)

        if torch.isnan(cur_pos).any():
            log.debug(f"Object {self.obj_name} moved to nan position")
            return torch.ones(cur_pos.shape[0], dtype=torch.bool, device=handler.device)
        dim = {"x": 0, "y": 1, "z": 2}[self.axis]
        dis_diff = cur_pos - self.init_pos
        dim_diff = dis_diff[:, dim]
        tot_dis = torch.norm(dis_diff, dim=-1)
        log.debug(f"Object {self.obj_name} moved {tensor_to_str(dim_diff)} meters in {self.axis} direction")
        # TODO: velocity check
        if self.distance > 0:
            return (
                (dim_diff >= self.distance - self.tolerance)
                * (dim_diff <= self.distance + self.tolerance)
                * (tot_dis <= self.bounding_distance)
            )

        else:
            return dim_diff <= self.distance + self.tolerance


## FIXME: this function is redundant with RotationShiftChecker, we should remove it!
@configclass
class _UpAxisRotationChecker(BaseChecker):
    """Check if the object with ``obj_name`` was rotated away ``target_degree`` degrees from the given ``axis`` (for example,  "z", [0,0,1] ) by more than ``degree_threshold`` degrees.

    - ``degree_threshold`` should be in the range of [0, 180].
    - ``axis`` should be one of "x", "y", "z". default is "z".
    """

    ## ref: https://github.com/mees/calvin_env/blob/c7377a6485be43f037f4a0b02e525c8c6e8d24b0/calvin_env/envs/tasks.py#L54
    obj_name: str = MISSING
    degree_threshold: float = MISSING
    target_degree: float = MISSING
    axis: Literal["x", "y", "z"] = "z"

    def reset(self, handler: BaseSimHandler, env_ids: list[int] | None = None):
        if env_ids is None:
            env_ids = list(range(handler.num_envs))

        if not hasattr(self, "init_quat"):
            self.init_quat = torch.zeros(handler.num_envs, 4, dtype=torch.float32, device=handler.device)

        self.init_quat[env_ids] = handler.get_rot(self.obj_name, env_ids=env_ids)

    def check(self, handler: BaseSimHandler) -> torch.BoolTensor:
        cur_quat = handler.get_rot(self.obj_name)
        cur_rot_mat = matrix_from_quat(cur_quat)

        v = {"x": 0, "y": 1, "z": 2}[self.axis]

        # Get the rotation around the up axis.
        # If cur_rot_mat is batched (e.g. shape [B, 3, 3]), this indexing works over the batch.
        up_axis = cur_rot_mat[..., v, :]

        # Compute the norm (magnitude) of the up_axis vector along the last dimension.
        norm = torch.norm(up_axis, dim=-1)

        # Compute the cosine of the angle using batch division.
        cos_angle = up_axis[..., 1] / norm

        # Calculate the angle in radians then convert to degrees.
        angle = torch.acos(cos_angle)
        angle = angle * 180.0 / torch.pi

        # Compute the absolute difference from the target degree.
        delta_angle = torch.abs(angle - self.target_degree)

        log.debug(
            f"Object {self.obj_name} rotated {angle} degrees away from {self.axis}-axis, the delta is {delta_angle}"
        )

        return delta_angle <= self.degree_threshold


## FIXME: This checker should be removed!
@configclass
class _SlideChecker(BaseChecker):
    def check(self, handler: BaseSimHandler) -> torch.BoolTensor:
        from metasim.utils.humanoid_robot_util import torso_upright

        states = handler.get_states()
        terminated = []
        for state in states:
            if torso_upright(state, handler._robot.name) < 0.6:
                terminated.append(True)
            else:
                terminated.append(False)
        return torch.tensor(terminated)


## FIXME: This checker should be removed!
@configclass
class _WalkChecker(BaseChecker):
    def check(self, handler: BaseSimHandler) -> torch.BoolTensor:
        from metasim.utils.humanoid_robot_util import robot_position_tensor

        states = handler.get_states()
        terminated = robot_position_tensor(states, handler.robot.name)[:, 2] < 0.2
        return terminated


## FIXME: This checker should be removed!
@configclass
class _StandChecker(_WalkChecker):
    pass


## FIXME: This checker should be removed!
@configclass
class _RunChecker(_WalkChecker):
    pass


## FIXME: This checker should be removed!
@configclass
class _CrawlChecker(BaseChecker):
    def check(self, handler: BaseSimHandler) -> torch.BoolTensor:
        states = handler.get_states()
        terminated = [False] * len(states)
        return torch.tensor(terminated)


## FIXME: This checker should be removed!
@configclass
class _HurdleChecker(BaseChecker):
    def check(self, handler: BaseSimHandler) -> torch.BoolTensor:
        states = handler.get_states()
        terminated = [False] * len(states)
        return torch.tensor(terminated)


## FIXME: This checker should be removed!
@configclass
class _MazeChecker(BaseChecker):
    def check(self, handler: BaseSimHandler) -> torch.BoolTensor:
        from metasim.utils.humanoid_robot_util import robot_position

        states = handler.get_states()
        terminated = []
        for state in states:
            if robot_position(state, handler.robot.name)[2] < 0.2:
                terminated.append(True)
            else:
                terminated.append(False)
        return torch.tensor(terminated)


## FIXME: This checker should be removed!
@configclass
class _PoleChecker(BaseChecker):
    def check(self, handler: BaseSimHandler) -> torch.BoolTensor:
        from metasim.utils.humanoid_robot_util import robot_position

        states = handler.get_states()
        terminated = []
        for state in states:
            if robot_position(state, handler.robot.name)[2] < 0.6:
                terminated.append(True)
            else:
                terminated.append(False)
        return torch.tensor(terminated)


## FIXME: This checker should be removed!
@configclass
class _SitChecker(BaseChecker):
    def check(self, handler: BaseSimHandler) -> torch.BoolTensor:
        from metasim.utils.humanoid_robot_util import robot_position

        states = handler.get_states()
        terminated = []
        for state in states:
            if robot_position(state, handler.robot.name)[2] < 0.5:
                terminated.append(True)
            else:
                terminated.append(False)
        return torch.tensor(terminated)


## FIXME: This checker should be removed!
@configclass
class _StairChecker(BaseChecker):
    def check(self, handler: BaseSimHandler) -> torch.BoolTensor:
        from metasim.utils.humanoid_robot_util import torso_upright

        states = handler.get_states()
        terminated = []
        for state in states:
            if torso_upright(state, handler.robot.name) < 0.1:
                terminated.append(True)
            else:
                terminated.append(False)
        return torch.tensor(terminated)


## FIXME: This checker should be removed!
@configclass
class _PushChecker(BaseChecker):
    def check(self, handler: BaseSimHandler) -> torch.BoolTensor:
        states = handler.get_states()
        terminated = []

        for state in states:
            # Get box position
            box_pos = state["object"]["pos"]

            # Get destination position
            dest_pos = state["destination"]["pos"]

            # Calculate distance between box and destination
            dgoal = torch.norm(box_pos - dest_pos)

            # Terminate when dgoal < 0.05 (success)
            if dgoal < 0.05:
                terminated.append(True)
            else:
                terminated.append(False)

        return torch.tensor(terminated)


## FIXME: This checker should be removed!
@configclass
class _CubeChecker(BaseChecker):
    def check(self, handler: BaseSimHandler) -> torch.BoolTensor:
        states = handler.get_states()
        terminated = []

        for state in states:
            # Get the z-coordinate of the robot pelvis
            pelvis_z = state[f"metasim_body_{handler.robot.name}/pelvis"]["com"][2]  # 骨盆 z 坐标

            # Get the z-coordinate of the cube in the left and right hands
            cube_left_z = state["metasim_body_cube_1/cube_1"]["pos"][2]  # z-coordinate of the cube in the left hand
            cube_right_z = state["metasim_body_cube_2/cube_2"]["pos"][2]  # z-coordinate of the cube in the right hand

            # When the pelvis z-coordinate or any cube z-coordinate is less than 0.5, terminate
            if pelvis_z < 0.5 or cube_left_z < 0.5 or cube_right_z < 0.5:
                terminated.append(True)
            else:
                terminated.append(False)

        return torch.tensor(terminated)


## FIXME: This checker should be removed!
@configclass
class _DoorChecker(BaseChecker):
    def check(self, handler: BaseSimHandler) -> torch.BoolTensor:
        from metasim.utils.humanoid_robot_util import robot_position

        states = handler.get_states()
        terminated = []
        for state in states:
            if robot_position(state, handler.robot.name)[2] < 0.58:
                terminated.append(True)
            else:
                terminated.append(False)
        return torch.tensor(terminated)


## FIXME: This checker should be removed!
@configclass
class _PackageChecker(BaseChecker):
    def check(self, handler: BaseSimHandler) -> torch.BoolTensor:
        states = handler.get_states()
        terminated = []
        for state in states:
            package_pos = state["metasim_body_package/package"]["pos"]
            target_pos = state["metasim_body_target/target"]["pos"]
            package_target_dist = torch.norm(package_pos - target_pos)

            if package_target_dist < 0.1:
                terminated.append(True)
            else:
                terminated.append(False)
        return torch.tensor(terminated)


## FIXME: This checker should be removed!
@configclass
class _PowerliftChecker(BaseChecker):
    def check(self, handler: BaseSimHandler) -> torch.BoolTensor:
        from metasim.utils.humanoid_robot_util import robot_position

        states = handler.get_states()
        terminated = []
        for state in states:
            if robot_position(state, handler.robot.name)[2] < 0.2:
                terminated.append(True)
            else:
                terminated.append(False)
        return torch.tensor(terminated)


## FIXME: This checker should be removed!
@configclass
class _SpoonChecker(BaseChecker):
    def check(self, handler: BaseSimHandler) -> torch.BoolTensor:
        states = handler.get_states()
        terminated = [False] * len(states)
        return torch.tensor(terminated)


## FIXME: This checker should be removed!
@configclass
class _HighbarChecker(BaseChecker):
    def check(self, handler: BaseSimHandler) -> torch.BoolTensor:
        states = handler.get_states()
        terminated = [False] * len(states)
        return torch.tensor(terminated)
