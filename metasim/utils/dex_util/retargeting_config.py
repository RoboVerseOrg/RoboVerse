from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from metasim.utils.dex_util import yourdfpy as urdf
from metasim.utils.dex_util.kinematics_adaptor import MimicJointKinematicAdaptor
from metasim.utils.dex_util.optimizer_utils import LPFilter
from metasim.utils.dex_util.robot_wrapper import RobotWrapper
from metasim.utils.dex_util.seq_retarget import SeqRetargeting
from metasim.utils.dex_util.yourdfpy import DUMMY_JOINT_NAMES


@dataclass
class RetargetingConfig:
    """Configuration for retargeting.

    Args:
        type (str): The type of retargeting.
        urdf_path (str): The path to the URDF file.
    """

    type: str
    urdf_path: str

    # Whether to add free joint to the root of the robot. Free joint enable the robot hand move freely in the space
    add_dummy_free_joint: bool = False

    # Source refers to the retargeting input, which usually corresponds to the human hand
    # The joint indices of human hand joint which corresponds to each link in the target_link_names
    target_link_human_indices: np.ndarray | None = None

    # The link on the robot hand which corresponding to the wrist of human hand
    wrist_link_name: str | None = None

    # Position retargeting link names
    target_link_names: list[str] | None = None

    # Vector retargeting link names
    target_joint_names: list[str] | None = None
    target_origin_link_names: list[str] | None = None
    target_task_link_names: list[str] | None = None

    # DexPilot retargeting link names
    finger_tip_link_names: list[str] | None = None

    # Scaling factor for vector retargeting only
    # For example, Allegro is 1.6 times larger than normal human hand, then this scaling factor should be 1.6
    scaling_factor: float = 1.0

    # Optimization parameters
    normal_delta: float = 4e-3
    huber_delta: float = 2e-2

    # DexPilot optimizer parameters
    project_dist: float = 0.03
    escape_dist: float = 0.05

    # Joint limit tag
    has_joint_limits: bool = True

    # Mimic joint tag
    ignore_mimic_joint: bool = False

    # Low pass filter
    low_pass_alpha: float = 0.1

    _TYPE = ["vector", "position", "dexpilot"]
    _DEFAULT_URDF_DIR = "./"

    def __post_init__(self):
        # Retargeting type check
        self.type = self.type.lower()
        if self.type not in self._TYPE:
            raise ValueError(f"Retargeting type must be one of {self._TYPE}")

        # Vector retargeting requires: target_origin_link_names + target_task_link_names
        # Position retargeting requires: target_link_names
        if self.type == "vector":
            if self.target_origin_link_names is None or self.target_task_link_names is None:
                raise ValueError("Vector retargeting requires: target_origin_link_names + target_task_link_names")
            if len(self.target_task_link_names) != len(self.target_origin_link_names):
                raise ValueError("Vector retargeting origin and task links dim mismatch")
            if self.target_link_human_indices is None:
                raise ValueError("Vector retargeting requires: target_link_human_indices")
            if self.target_link_human_indices.shape != (
                2,
                len(self.target_origin_link_names),
            ):
                raise ValueError("Vector retargeting link names and link indices dim mismatch")

        elif self.type == "position":
            if self.target_link_names is None:
                raise ValueError("Position retargeting requires: target_link_names")
            if self.target_link_human_indices is None:
                raise ValueError("Position retargeting requires: target_link_human_indices")
            self.target_link_human_indices = self.target_link_human_indices.squeeze()
            if self.target_link_human_indices.shape != (len(self.target_link_names),):
                raise ValueError("Position retargeting link names and link indices dim mismatch")

        elif self.type == "dexpilot":
            if self.finger_tip_link_names is None or self.wrist_link_name is None:
                raise ValueError("Position retargeting requires: finger_tip_link_names + wrist_link_name")
            if self.target_link_human_indices is not None:
                logging.warning(
                    "Target link human indices is provided in the DexPilot retargeting config, which is uncommon. "
                    "If you do not know exactly how it is used, please leave it to None for default."
                )

        # URDF path check
        urdf_path = Path(self.urdf_path)
        if not urdf_path.is_absolute():
            urdf_path = self._DEFAULT_URDF_DIR / urdf_path
            urdf_path = urdf_path.absolute()
        if not urdf_path.exists():
            raise ValueError(f"URDF path {urdf_path} does not exist")
        self.urdf_path = str(urdf_path)

    @classmethod
    def set_default_urdf_dir(cls, urdf_dir: str | Path):
        """Set the default directory for URDF files.

        Args:
            urdf_dir (Union[str, Path]): The directory to set as default.
        """
        path = Path(urdf_dir)
        if not path.exists():
            raise ValueError(f"URDF dir {urdf_dir} not exists.")
        cls._DEFAULT_URDF_DIR = urdf_dir

    @classmethod
    def load_from_file(cls, config_path: str | Path, override: dict[str, Any] | None = None):
        """Load retargeting configuration from a file.

        Args:
            config_path (Union[str, Path]): The path to the configuration file.
            override (Optional[Dict]): Optional dictionary of overrides.
        """
        path = Path(config_path)
        if not path.is_absolute():
            path = path.absolute()

        with path.open("r") as f:
            yaml_config = yaml.load(f, Loader=yaml.FullLoader)
            cfg = yaml_config["retargeting"]
            return cls.from_dict(cfg, override)

    @classmethod
    def from_dict(cls, cfg: dict[str, Any], override: dict[str, Any] | None = None):
        """Create a RetargetingConfig from a dictionary.

        Args:
            cfg (Dict[str, Any]): The configuration dictionary.
            override (Optional[Dict]): Optional dictionary of overrides.
        """
        if "target_link_human_indices" in cfg:
            cfg["target_link_human_indices"] = np.array(cfg["target_link_human_indices"])
        if override is not None:
            for key, value in override.items():
                cfg[key] = value
        config = RetargetingConfig(**cfg)
        return config

    def build(self) -> SeqRetargeting:
        """Build the retargeting object.

        Returns:
            SeqRetargeting: The retargeting object.
        """
        import tempfile

        from metasim.utils.dex_util.optimizer import (
            DexPilotOptimizer,
            PositionOptimizer,
            VectorOptimizer,
        )

        # Process the URDF with yourdfpy to better find file path
        robot_urdf = urdf.URDF.load(
            self.urdf_path,
            add_dummy_free_joints=self.add_dummy_free_joint,
            build_scene_graph=False,
        )
        urdf_name = self.urdf_path.split(os.path.sep)[-1]
        temp_dir = tempfile.mkdtemp(prefix="dex_retargeting-")
        temp_path = f"{temp_dir}/{urdf_name}"
        robot_urdf.write_xml_file(temp_path)

        # Load pinocchio model
        robot = RobotWrapper(temp_path)

        # Add 6D dummy joint to target joint names so that it will also be optimized
        if self.add_dummy_free_joint and self.target_joint_names is not None:
            self.target_joint_names = DUMMY_JOINT_NAMES + self.target_joint_names
        joint_names = self.target_joint_names if self.target_joint_names is not None else robot.dof_joint_names

        if self.type == "position":
            assert self.target_link_names is not None
            assert self.target_link_human_indices is not None
            optimizer = PositionOptimizer(
                robot,
                joint_names,
                target_link_names=self.target_link_names,
                target_link_human_indices=self.target_link_human_indices,
                norm_delta=self.normal_delta,
                huber_delta=self.huber_delta,
            )
        elif self.type == "vector":
            assert self.target_origin_link_names is not None
            assert self.target_task_link_names is not None
            assert self.target_link_human_indices is not None
            optimizer = VectorOptimizer(
                robot,
                joint_names,
                target_origin_link_names=self.target_origin_link_names,
                target_task_link_names=self.target_task_link_names,
                target_link_human_indices=self.target_link_human_indices,
                scaling=self.scaling_factor,
                norm_delta=self.normal_delta,
                huber_delta=self.huber_delta,
            )
        elif self.type == "dexpilot":
            assert self.finger_tip_link_names is not None
            assert self.wrist_link_name is not None
            optimizer = DexPilotOptimizer(
                robot,
                joint_names,
                finger_tip_link_names=self.finger_tip_link_names,
                wrist_link_name=self.wrist_link_name,
                target_link_human_indices=self.target_link_human_indices,
                scaling=self.scaling_factor,
                project_dist=self.project_dist,
                escape_dist=self.escape_dist,
            )
        else:
            raise RuntimeError()

        if 0 <= self.low_pass_alpha <= 1:
            lp_filter = LPFilter(self.low_pass_alpha)
        else:
            lp_filter = None

        # Parse mimic joints and set kinematics adaptor for optimizer
        has_mimic_joints, source_names, mimic_names, multipliers, offsets = parse_mimic_joint(robot_urdf)
        if has_mimic_joints and not self.ignore_mimic_joint:
            adaptor = MimicJointKinematicAdaptor(
                robot,
                target_joint_names=joint_names,
                source_joint_names=source_names,
                mimic_joint_names=mimic_names,
                multipliers=multipliers,
                offsets=offsets,
            )
            optimizer.set_kinematic_adaptor(adaptor)

        retargeting = SeqRetargeting(
            optimizer,
            has_joint_limits=self.has_joint_limits,
            lp_filter=lp_filter,
        )
        return retargeting


def get_retargeting_config(config_path: str | Path) -> RetargetingConfig:
    """Get the retargeting configuration from a file.

    Args:
        config_path (Union[str, Path]): The path to the configuration file.

    Returns:
        RetargetingConfig: The retargeting configuration.
    """
    config = RetargetingConfig.load_from_file(config_path)
    return config


def parse_mimic_joint(
    robot_urdf: urdf.URDF,
) -> tuple[bool, list[str], list[str], list[float], list[float]]:
    """Parse mimic joints from a URDF robot model.

    Args:
        robot_urdf (urdf.URDF): The URDF robot model.

    Returns:
        Tuple[bool, List[str], List[str], List[float], List[float]]: A tuple containing:
            - bool: Whether mimic joints exist
            - List[str]: Source joint names
            - List[str]: Mimic joint names
            - List[float]: Multipliers
            - List[float]: Offsets
    """
    mimic_joint_names = []
    source_joint_names = []
    multipliers = []
    offsets = []
    for name, joint in robot_urdf.joint_map.items():
        if joint.mimic is not None:
            mimic_joint_names.append(name)
            source_joint_names.append(joint.mimic.joint)
            multipliers.append(joint.mimic.multiplier)
            offsets.append(joint.mimic.offset)

    return (
        len(mimic_joint_names) > 0,
        source_joint_names,
        mimic_joint_names,
        multipliers,
        offsets,
    )
