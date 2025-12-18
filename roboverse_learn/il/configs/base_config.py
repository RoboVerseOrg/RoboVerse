"""Base policy configuration classes for the metasim framework.

This module defines configuration dataclasses for policies in the metasim framework,
including observation, action, and end-effector configurations.
"""

from __future__ import annotations

from dataclasses import field
from typing import Literal

from metasim.utils.configclass import configclass


@configclass
class EndEffectorCfg:
    """Configuration for End Effector control (only used if action_space is 'ee_abs' or 'ee_delta')."""

    rotation_rep: Literal["quaternion", "axis_angle"] = "quaternion"
    gripper_rep: Literal["q_pos", "strength"] = "q_pos"


@configclass
class ObsCfg:
    """Configuration for observations in policies.

    Defines the observation type, dimension, and format.
    """

    obs_type: Literal["joint_pos", "ee", "no_proprio"] = "joint_pos"
    ee_cfg: EndEffectorCfg = EndEffectorCfg()
    obs_dim: int = 0
    obs_padding: int = 0
    obs_keys: list[str] = field(default_factory=lambda: ["head_cam", "agent_pos"])
    obs_dtype: Literal["np", "torch"] = "torch"
    norm_image: bool = True


@configclass
class ActionCfg:
    """Configuration for actions in policies.

    Defines the action type, dimension, and control parameters.
    """

    action_type: Literal["joint_pos", "ee"] = "joint_pos"
    ee_cfg: EndEffectorCfg = EndEffectorCfg()
    action_dim: int = 0
    action_rtype: Literal["np", "torch"] = "torch"
    delta: int = 0  # 0 means absolute control, 1 means delta control.
    action_set_steps: int = 1
    action_chunk_steps: int = 1
    interpolate_chunk: bool = False
    temporal_agg: bool = True


@configclass
class BasePolicyCfg:
    """Base configuration for policy metadata."""

    checkpoint_path: str = ""
    name: str = "Base"
    obs_config: ObsCfg = ObsCfg()
    action_config: ActionCfg = ActionCfg()


@configclass
class DiffusionPolicyCfg(BasePolicyCfg):
    """Configuration for Diffusion Policy metadata.

    Extends the base policy configuration with diffusion-specific settings.
    """

    name: str = "DiffusionPolicy"
    action_config: ActionCfg = ActionCfg(temporal_agg=False)


@configclass
class VLAPolicyCfg(BasePolicyCfg):
    """Configuration for VLAPolicy metadata.

    Extends the base policy configuration with VLAD-specific settings.
    """

    name: str = "VLAPolicy"
    action_config: ActionCfg = ActionCfg(
        action_type="ee",
        delta=1,
        action_dim=7,
        ee_cfg=EndEffectorCfg(rotation_rep="axis_angle", gripper_rep="strength"),
    )

    obs_config: ObsCfg = ObsCfg(
        obs_type="no_proprio",
        norm_image=False,
    )


@configclass
class ACTPolicyCfg(BasePolicyCfg):
    """Configuration for ACTPolicy metadata.

    Extends the base policy configuration with ACT-specific settings.
    """

    name: str = "ACTPolicy"
    action_config: ActionCfg = ActionCfg(
        action_type="joint_pos",
        action_dim=9,
        temporal_agg=True,
        action_chunk_steps=100,
    )

    obs_config: ObsCfg = ObsCfg(
        obs_type="joint_pos",
        obs_dim=14,
        obs_padding=14,
        norm_image=True,
    )
