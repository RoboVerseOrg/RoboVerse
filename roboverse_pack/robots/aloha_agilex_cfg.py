"""ALOHA-AgileX dual-arm RobotCfg (RoboTwin embodiment).

The URDF + meshes live in RoboTwin's embodiments asset bundle. We expect
them at `~/projects/robotwin/assets/embodiments/aloha-agilex/` or via
the `ROBOTWIN_ASSETS` env var. See
`tools/robotwin_integration/` for the integration harness and
`docs/source/dataset_benchmark/integrations/robotwin.md` for the full
write-up.

The URDF declares 38 DoFs in total (mobile base + sensor mast + dual
6-DoF arms with 2-finger mimic grippers). Per RoboTwin's
`config.yml`, the arm joints are `fl_joint1..6` (left) and
`fr_joint1..6` (right); each gripper is `fX_joint7` with `fX_joint8`
mimicking it. The other DoFs (wheels, suspension, camera mast)
typically stay locked in the RL tasks RoboTwin ships.
"""

from __future__ import annotations

import os
from pathlib import Path

from metasim.scenario.robot import BaseActuatorCfg, RobotCfg
from metasim.utils import configclass


def _resolve_robotwin_assets() -> Path:
    env = os.environ.get("ROBOTWIN_ASSETS")
    if env:
        p = Path(env).expanduser().resolve()
        if p.exists():
            return p
    for c in (Path.home() / "projects" / "robotwin" / "assets",):
        if (c / "embodiments" / "aloha-agilex" / "urdf").exists():
            return c
    raise FileNotFoundError(
        "Could not locate robotwin assets. Clone RoboTwin under ~/projects/robotwin and "
        "download embodiments.zip (220 MB) per `assets/_download.py`, or set ROBOTWIN_ASSETS."
    )


def _aloha_agilex_urdf() -> str:
    return str(_resolve_robotwin_assets() / "embodiments" / "aloha-agilex" / "urdf" / "arx5_description_isaac.urdf")


_ARM_NAMES = [
    "fl_joint1",
    "fl_joint2",
    "fl_joint3",
    "fl_joint4",
    "fl_joint5",
    "fl_joint6",
    "fl_joint7",  # gripper base
    "fl_joint8",  # gripper mimic
    "fr_joint1",
    "fr_joint2",
    "fr_joint3",
    "fr_joint4",
    "fr_joint5",
    "fr_joint6",
    "fr_joint7",
    "fr_joint8",
]


def _arm_actuator(stiffness: float = 1000.0, damping: float = 200.0) -> BaseActuatorCfg:
    return BaseActuatorCfg(stiffness=stiffness, damping=damping, velocity_limit=3.14159)


def _grip_actuator(stiffness: float = 1000.0, damping: float = 200.0) -> BaseActuatorCfg:
    return BaseActuatorCfg(stiffness=stiffness, damping=damping, velocity_limit=0.5, is_ee=True)


@configclass
class AlohaAgilexCfg(RobotCfg):
    """RoboTwin's dual-arm embodiment: arx5 arms on an agilex mobile base."""

    name: str = "aloha_agilex"
    # 16 effective DoFs we control (left/right arm 6 + 2 gripper joints each);
    # the URDF has 38 DoFs but the wheels/mast/sensor mounts stay passive.
    num_joints: int = 16
    fix_base_link: bool = True
    enabled_gravity: bool = True
    enabled_self_collisions: bool = False
    urdf_path: str = ""  # populated in __post_init__

    actuators: dict[str, BaseActuatorCfg] = {n: _arm_actuator() for n in _ARM_NAMES[:6] + _ARM_NAMES[8:14]} | {
        n: _grip_actuator() for n in (_ARM_NAMES[6], _ARM_NAMES[7], _ARM_NAMES[14], _ARM_NAMES[15])
    }

    # The arx5 URDF reports per-joint limits via <limit lower=… upper=…>; we
    # mirror the ones the planner uses (see config.yml + curobo cfg). Empty
    # dict means "trust the URDF" which works for our parity tests.
    joint_limits: dict[str, tuple[float, float]] = {}

    default_joint_positions: dict[str, float] = {n: 0.0 for n in _ARM_NAMES}

    def __post_init__(self):  # type: ignore[override]
        if not self.urdf_path:
            try:
                self.urdf_path = _aloha_agilex_urdf()
            except FileNotFoundError:
                # Leave empty; loading will raise a clear error later.
                self.urdf_path = ""
        return super().__post_init__()
