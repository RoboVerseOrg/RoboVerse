try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

import pytest
import torch
from loguru import logger as log

from metasim.constants import PhysicStateType
from metasim.scenario.objects import ArticulationObjCfg, PrimitiveCubeCfg, PrimitiveSphereCfg, RigidObjCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.sim.sim_context import HandlerContext
from metasim.test.test_utils import assert_close
from metasim.utils.state import state_tensor_to_nested
from roboverse_pack.robots.franka_cfg import FrankaCfg


def _get_test_parameters():
    return [("isaacsim", 1), ("isaacsim", 2), ("mujoco", 1), ("mujoco", 2)]


@pytest.fixture(scope="session")
def simulation_app():
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=False, enable_cameras=True).app
    yield app
    # NOTE: Don't call app.close(), otherwise pytest summary will be skipped!


@pytest.fixture(scope="function", autouse=True)
def isaacsim_context():
    import isaaclab.sim as sim_utils
    import isaacsim.core.utils.stage as stage_utils

    log.debug("Creating new stage")
    stage_utils.create_new_stage()
    log.debug("New stage created")
    sim_cfg = sim_utils.SimulationCfg()
    sim = sim_utils.SimulationContext(sim_cfg)
    sim._app_control_on_stop_handle = None
    yield sim
    log.debug("Stopping simulation")
    sim.clear_all_callbacks()
    sim.clear_instance()
    log.debug("Simulation stopped")


@pytest.mark.parametrize("sim,num_envs", _get_test_parameters())
def test_consistency(simulation_app, sim, num_envs):
    scenario = ScenarioCfg(
        simulator=sim,
        num_envs=num_envs,
        headless=True,
        objects=[
            PrimitiveCubeCfg(
                name="cube", size=(0.1, 0.1, 0.1), color=[1.0, 0.0, 0.0], physics=PhysicStateType.RIGIDBODY
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
                usd_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/bbq_sauce/usd/bbq_sauce.usd",
                urdf_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/bbq_sauce/urdf/bbq_sauce.urdf",
                mjcf_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/bbq_sauce/mjcf/bbq_sauce.xml",
            ),
            ArticulationObjCfg(
                name="box_base",
                fix_base_link=True,
                usd_path="roboverse_data/assets/rlbench/close_box/box_base/usd/box_base.usd",
                urdf_path="roboverse_data/assets/rlbench/close_box/box_base/urdf/box_base_unique.urdf",
                mjcf_path="roboverse_data/assets/rlbench/close_box/box_base/mjcf/box_base_unique.mjcf",
            ),
        ],
        robots=[FrankaCfg()],
    )
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
    ] * num_envs

    with HandlerContext(scenario, simulation_app) as handler:
        handler.set_states(init_states)
        states = state_tensor_to_nested(handler, handler.get_states())
        for i in range(num_envs):
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
