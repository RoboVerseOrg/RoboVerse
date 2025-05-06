"""This script is used to test the consistency of `set_states` and `get_states`."""

from __future__ import annotations

import math
from typing import Literal

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass


import torch
import tyro
from loguru import logger as log
from rich.logging import RichHandler

from metasim.cfg.objects import ArticulationObjCfg, PrimitiveCubeCfg, PrimitiveSphereCfg, RigidObjCfg
from metasim.cfg.scenario import ScenarioCfg
from metasim.cfg.sensors import PinholeCameraCfg
from metasim.constants import PhysicStateType, SimType
from metasim.utils import configclass
from metasim.utils.setup_util import get_sim_env_class
from metasim.utils.state import state_tensor_to_nested

log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])


@configclass
class Args:
    robot: str = "franka"
    sim: Literal["isaaclab", "isaacgym", "genesis", "pyrep", "pybullet", "sapien", "sapien3", "mujoco", "blender"] = (
        "isaaclab"
    )
    num_envs: int = 1
    headless: bool = True

    def __post_init__(self):
        log.info(f"Args: {self}")


args = tyro.cli(Args)

# initialize scenario
scenario = ScenarioCfg(
    robot=args.robot,
    try_add_table=False,
    sim=args.sim,
    headless=args.headless,
    num_envs=args.num_envs,
)

# add cameras
scenario.cameras = [PinholeCameraCfg(width=1024, height=1024, pos=(1.5, -1.5, 1.5), look_at=(0.0, 0.0, 0.0))]

# add objects
scenario.objects = [
    PrimitiveCubeCfg(
        name="cube",
        size=(0.1, 0.1, 0.1),
        color=[1.0, 0.0, 0.0],
        physics=PhysicStateType.RIGIDBODY,
    ),
    PrimitiveSphereCfg(
        name="sphere",
        radius=0.1,
        color=[0.0, 0.0, 1.0],
        physics=PhysicStateType.RIGIDBODY,
    ),
    RigidObjCfg(
        name="bbq_sauce",
        scale=(2, 2, 2),
        physics=PhysicStateType.RIGIDBODY,
        usd_path="get_started/example_assets/bbq_sauce/usd/bbq_sauce.usd",
        urdf_path="get_started/example_assets/bbq_sauce/urdf/bbq_sauce.urdf",
        mjcf_path="get_started/example_assets/bbq_sauce/mjcf/bbq_sauce.xml",
    ),
    ArticulationObjCfg(
        name="box_base",
        fix_base_link=True,
        usd_path="get_started/example_assets/box_base/usd/box_base.usd",
        urdf_path="get_started/example_assets/box_base/urdf/box_base_unique.urdf",
        mjcf_path="get_started/example_assets/box_base/mjcf/box_base_unique.mjcf",
    ),
]


log.info(f"Using simulator: {args.sim}")
env_class = get_sim_env_class(SimType(args.sim))
env = env_class(scenario)

init_states = [
    {
        "objects": {
            "cube": {
                "pos": torch.tensor([0.3, -0.2, 0.05]),
                "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
            },
            "sphere": {
                "pos": torch.tensor([0.4, -0.6, 0.05]),
                "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
            },
            "bbq_sauce": {
                "pos": torch.tensor([0.7, -0.3, 0.14]),
                "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
            },
            "box_base": {
                "pos": torch.tensor([0.5, 0.2, 0.1]),
                "rot": torch.tensor([0.0, 0.7071, 0.0, 0.7071]),
                "dof_pos": {"box_joint": 0.0},
            },
        },
        "robots": {
            "franka": {
                "pos": torch.tensor([0.0, 0.0, 0.0]),
                "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                "dof_pos": {
                    "panda_joint1": 0.0,
                    "panda_joint2": -0.785398,
                    "panda_joint3": 0.0,
                    "panda_joint4": -2.356194,
                    "panda_joint5": 0.0,
                    "panda_joint6": 1.570796,
                    "panda_joint7": 0.785398,
                    "panda_finger_joint1": 0.04,
                    "panda_finger_joint2": 0.04,
                },
            },
        },
    }
]

env.handler.set_states(init_states)
states = state_tensor_to_nested(env.handler, env.handler.get_states())


def assert_close(a, b, atol=1e-3):
    if isinstance(a, torch.Tensor):
        assert torch.allclose(a, b, atol=atol), f"a: {a} != b: {b}"
    elif isinstance(a, float):
        assert math.isclose(a, b, abs_tol=atol), f"a: {a} != b: {b}"
    else:
        raise ValueError(f"Unsupported type: {type(a)}")


for i in range(args.num_envs):
    assert_close(states[i]["objects"]["cube"]["pos"], init_states[i]["objects"]["cube"]["pos"])
    assert_close(states[i]["objects"]["sphere"]["pos"], init_states[i]["objects"]["sphere"]["pos"])
    assert_close(states[i]["objects"]["bbq_sauce"]["pos"], init_states[i]["objects"]["bbq_sauce"]["pos"])
    assert_close(states[i]["objects"]["box_base"]["pos"], init_states[i]["objects"]["box_base"]["pos"])
    assert_close(states[i]["objects"]["box_base"]["rot"], init_states[i]["objects"]["box_base"]["rot"])
    assert_close(states[i]["robots"]["franka"]["pos"], init_states[i]["robots"]["franka"]["pos"])
    assert_close(states[i]["robots"]["franka"]["rot"], init_states[i]["robots"]["franka"]["rot"])
    assert_close(
        states[i]["objects"]["box_base"]["dof_pos"]["box_joint"],
        init_states[i]["objects"]["box_base"]["dof_pos"]["box_joint"],
    )
    for k in states[i]["robots"]["franka"]["dof_pos"].keys():
        assert_close(
            states[i]["robots"]["franka"]["dof_pos"][k],
            init_states[i]["robots"]["franka"]["dof_pos"][k],
        )
