# Robots Config

The `robot` configuration is the robot used in the simulation, it is inherited from the `ArticulationObjCfg`. We already provide many robots for you, and you can also introduce your own robots into the system by writing a `RobotCfg`:

```python
@configclass
class BaseRobotCfg(ArticulationObjCfg):
    """Base Cfg class for robots."""

    # Articulation
    num_joints: int = MISSING
    actuators: dict[str, BaseActuatorCfg] = {}
    ee_body_name: str | None = None
    fix_base_link: bool = True
    joint_limits: dict[str, tuple[float, float]] = {}
    default_joint_positions: dict[str, float] = {}
    default_position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """
    Joint limits in the format of `{joint_name: (lower_limit, upper_limit)}`.
    Note that different simulators may have different order of joints, so you should not use the order in this dict!
    """

    """Need a more elegant implementation!"""
    gripper_open_q: list[float] = MISSING
    gripper_close_q: list[float] = MISSING

    # cuRobo Configs
    curobo_ref_cfg_name: str = MISSING
    curobo_tcp_rel_pos: tuple[float, float, float] = MISSING
    curobo_tcp_rel_rot: tuple[float, float, float] = MISSING

    # Simulation
    enabled_gravity: bool = True
    """Whether to enable gravity in the simulation."""
    enabled_self_collisions: bool = True
    isaacgym_flip_visual_attachments: bool = True
```

This is the base class for robot configurations. Take the famous franka robot as example:

```python
@configclass
class FrankaCfg(BaseRobotCfg):
    """Cfg for the Franka Emika Panda robot.

    Args:
        BaseRobotCfg (_type_): _description_
    """

    name: str = "franka"
    num_joints: int = 9
    fix_base_link: bool = True
    usd_path: str = "metasim/data/quick_start/robots/franka/usd/franka_v2.usd"
    mjcf_path: str = "metasim/data/quick_start/robots/franka/mjcf/panda.xml"
    # urdf_path: str = "roboverse_data/robots/franka/urdf/panda.urdf"  # work for pybullet and sapien
    urdf_path: str = "metasim/data/quick_start/robots/franka/urdf/franka_panda.urdf"  # work for isaacgym
    enabled_gravity: bool = False
    enabled_self_collisions: bool = False
    actuators: dict[str, BaseActuatorCfg] = {
        "panda_joint1": BaseActuatorCfg(velocity_limit=2.175),
        "panda_joint2": BaseActuatorCfg(velocity_limit=2.175),
        "panda_joint3": BaseActuatorCfg(velocity_limit=2.175),
        "panda_joint4": BaseActuatorCfg(velocity_limit=2.175),
        "panda_joint5": BaseActuatorCfg(velocity_limit=2.61),
        "panda_joint6": BaseActuatorCfg(velocity_limit=2.61),
        "panda_joint7": BaseActuatorCfg(velocity_limit=2.61),
        "panda_finger_joint1": BaseActuatorCfg(velocity_limit=0.2, is_ee=True),
        "panda_finger_joint2": BaseActuatorCfg(velocity_limit=0.2, is_ee=True),
    }
    joint_limits: dict[str, tuple[float, float]] = {
        "panda_joint1": (-2.8973, 2.8973),
        "panda_joint2": (-1.7628, 1.7628),
        "panda_joint3": (-2.8973, 2.8973),
        "panda_joint4": (-3.0718, -0.0698),
        "panda_joint5": (-2.8973, 2.8973),
        "panda_joint6": (-0.0175, 3.7525),
        "panda_joint7": (-2.8973, 2.8973),
        "panda_finger_joint1": (0.0, 0.04),  # 0.0 close, 0.04 open
        "panda_finger_joint2": (0.0, 0.04),  # 0.0 close, 0.04 open
    }
    ee_body_name: str = "panda_hand"

    default_joint_positions: dict[str, float] = {
        "panda_joint1": 0.0,
        "panda_joint2": 0.0,
        "panda_joint3": 0.0,
        "panda_joint4": 0.0,
        "panda_joint5": 0.0,
        "panda_joint6": 0.0,
        "panda_joint7": 0.0,
        "panda_finger_joint1": 0.0,
        "panda_finger_joint2": 0.0,
    }

    gripper_open_q = [0.04, 0.04]
    gripper_close_q = [0.0, 0.0]

    curobo_ref_cfg_name: str = "franka.yml"
    curobo_tcp_rel_pos: tuple[float, float, float] = [0.0, 0.0, 0.10312]
    curobo_tcp_rel_rot: tuple[float, float, float] = [0.0, 0.0, 0.0]
```

This is how you can introduce a new robot into your simulation, as well as the RoboVerse community.
