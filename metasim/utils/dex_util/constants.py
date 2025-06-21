from __future__ import annotations

"""Constants for dex retargeting utilities."""

import enum
from pathlib import Path

import numpy as np

OPERATOR2MANO_RIGHT = np.array([
    [0, 0, -1],
    [-1, 0, 0],
    [0, 1, 0],
])

OPERATOR2MANO_LEFT = np.array([
    [0, 0, -1],
    [1, 0, 0],
    [0, -1, 0],
])


class HandType(enum.Enum):
    """Hand type enumeration."""

    left = "left"
    right = "right"


class RobotName(enum.Enum):
    """Robot name enumeration."""

    allegro = "allegro"
    shadow = "shadow"
    svh = "schunk_svh_hand"
    leap = "leap_hand"
    ability = "ability_hand"
    inspire = "inspire_hand"
    panda = "panda_gripper"
    barrett = "barrett"
    robotiq = "robotiq"


class RetargetingType(enum.Enum):  # noqa: D101
    vector = enum.auto()  # For teleoperation, no finger closing prior
    position = enum.auto()  # For offline data processing, especially hand-object interaction data
    dexpilot = enum.auto()  # For teleoperation, with finger closing prior


ROBOT_NAME_MAP = {
    RobotName.allegro: "allegro_hand",
    RobotName.shadow: "shadow_hand",
    RobotName.svh: "schunk_svh_hand",
    RobotName.leap: "leap_hand",
    RobotName.ability: "ability_hand",
    RobotName.inspire: "inspire_hand",
    RobotName.panda: "panda_gripper",
}

ROBOT_NAMES = list(ROBOT_NAME_MAP.keys())


def get_default_config_path(
    robot_name: RobotName, retargeting_type: RetargetingType, hand_type: HandType
) -> Path | None:
    """Get the default configuration path for a robot and retargeting type.

    Args:
        robot_name: The robot name enum
        retargeting_type: The retargeting type enum
        hand_type: The hand type enum

    Returns:
        Optional[Path]: The path to the configuration file, or None if not found
    """
    config_path = Path(__file__).parent / "configs"
    if retargeting_type is RetargetingType.position:
        config_path = config_path / "offline"
    else:
        config_path = config_path / "teleop"

    robot_name_str = ROBOT_NAME_MAP[robot_name]
    hand_type_str = hand_type.name
    if "gripper" in robot_name_str:  # For gripper robots, only use gripper config file.
        if retargeting_type == RetargetingType.dexpilot:
            config_name = f"{robot_name_str}_dexpilot.yml"
        else:
            config_name = f"{robot_name_str}.yml"
    else:
        if retargeting_type == RetargetingType.dexpilot:
            config_name = f"{robot_name_str}_{hand_type_str}_dexpilot.yml"
        else:
            config_name = f"{robot_name_str}_{hand_type_str}.yml"
    return config_path / config_name


OPERATOR2MANO = {
    HandType.right: OPERATOR2MANO_RIGHT,
    HandType.left: OPERATOR2MANO_LEFT,
}
