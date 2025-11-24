## ruff: noqa: D102

from __future__ import annotations

from dataclasses import MISSING
from typing import Literal

import torch
from loguru import logger as log

from metasim.scenario.objects import BaseObjCfg
from metasim.utils.configclass import configclass
from metasim.utils.math import euler_xyz_from_quat, matrix_from_quat, quat_from_matrix
from metasim.utils.state import TensorState
from metasim.utils.tensor_util import tensor_to_str

from .detectors import BaseDetector
from .util import get_dof_pos, get_pos, get_rot

try:
    from metasim.sim import BaseSimHandler
except:
    pass


@configclass
class EmptyChecker:
    """A checker that always returns False."""

    def reset(self, handler: BaseSimHandler, env_ids: list[int] | None = None):
        pass

    def check(self, handler: BaseSimHandler, states: TensorState) -> torch.BoolTensor:
        return torch.zeros(handler.num_envs, dtype=torch.bool, device=handler.device)

    def get_debug_viewers(self) -> list[BaseObjCfg]:
        return []


@configclass
class DetectedChecker:
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

    def check(self, handler: BaseSimHandler, states: TensorState) -> torch.BoolTensor:
        success = self.detector.is_detected(handler, self.obj_name)
        if self.ignore_if_first_check_success:
            self._ignore[self._first_check & success] = True
        self._first_check[self._first_check] = False

        success[self._ignore] = False
        return success

    def get_debug_viewers(self) -> list[BaseObjCfg]:
        return self.detector.get_debug_viewers()


@configclass
class JointPosChecker:
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

    def check(self, handler: BaseSimHandler, states: TensorState) -> torch.BoolTensor:
        dof_pos = get_dof_pos(handler, self.obj_name, self.joint_name)
        log.debug(f"Joint {self.joint_name} of object {self.obj_name} has position {tensor_to_str(dof_pos)}")
        if self.mode == "ge":
            return dof_pos >= self.radian_threshold
        elif self.mode == "le":
            return dof_pos <= self.radian_threshold
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    def get_debug_viewers(self) -> list[BaseObjCfg]:
        return []

    def reset(self, handler: BaseSimHandler, env_ids: list[int] | None = None):
        pass


@configclass
class JointPosShiftChecker:
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

        self.init_joint_pos[env_ids] = get_dof_pos(handler, self.obj_name, self.joint_name, env_ids=env_ids)

    def check(self, handler: BaseSimHandler, states: TensorState) -> torch.BoolTensor:
        cur_joint_pos = get_dof_pos(handler, self.obj_name, self.joint_name)
        joint_pos_diff = cur_joint_pos - self.init_joint_pos

        log.debug(f"Joint {self.joint_name} of object {self.obj_name} moved {tensor_to_str(joint_pos_diff)} units")

        if self.threshold > 0:
            return joint_pos_diff >= self.threshold
        else:
            return joint_pos_diff <= self.threshold


@configclass
class RotationShiftChecker:
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

        self.init_quat[env_ids] = get_rot(handler, self.obj_name, env_ids=env_ids)

    def check(self, handler: BaseSimHandler, states: TensorState) -> torch.BoolTensor:
        cur_quat = get_rot(handler, self.obj_name)
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
class PositionShiftChecker:
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

        tmp = get_pos(handler, self.obj_name, env_ids=env_ids)
        assert tmp.shape == (len(env_ids), 3)
        self.init_pos[env_ids] = tmp

    def check(self, handler: BaseSimHandler, states: TensorState) -> torch.BoolTensor:
        cur_pos = get_pos(handler, self.obj_name)
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
