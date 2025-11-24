"""Rest everything follows."""

import pytest
import rootutils
import torch

# from metasim.sim.sim_context import HandlerContext
from loguru import logger as log

from metasim.constants import PhysicStateType
from metasim.scenario.objects import ArticulationObjCfg, PrimitiveCubeCfg, PrimitiveSphereCfg, RigidObjCfg
from metasim.scenario.scenario import ScenarioCfg

rootutils.setup_root(__file__, pythonpath=True)
from metasim.test.test_utils import assert_close, get_test_parameters
from roboverse_pack.robots.franka_cfg import FrankaCfg


@pytest.mark.parametrize("sim,num_envs", get_test_parameters())
def test_consistency(sim, num_envs):
    if sim not in ("sapien3"):
        pytest.skip(f"Skipping simulator {sim} for this test (only sapien3 is supported).")

    # initialize scenario
    scenario = ScenarioCfg(
        robots=[FrankaCfg()],
        headless=True,
        num_envs=num_envs,
        simulator=sim,
    )

    # add objects
    scenario.objects = [
        PrimitiveCubeCfg(
            name="cube",
            size=(0.1, 0.1, 0.1),
            color=[1.0, 0.0, 0.0],
            default_position=[0.3, -0.2, 0.05],
            default_orientation=[1.0, 0.0, 0.0, 0.0],
            physics=PhysicStateType.RIGIDBODY,
        ),
        PrimitiveSphereCfg(
            name="sphere",
            radius=0.1,
            color=[0.0, 0.0, 1.0],
            default_position=[0.4, -0.6, 0.1],
            default_orientation=[1.0, 0.0, 0.0, 0.0],
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="bbq_sauce",
            scale=(2, 2, 2),
            physics=PhysicStateType.RIGIDBODY,
            default_position=[0.7, -0.3, 0.0094],
            default_orientation=[1.0, 0.0, 0.0, 0.0],
            usd_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/bbq_sauce/usd/bbq_sauce.usd",
            urdf_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/bbq_sauce/urdf/bbq_sauce.urdf",
            mjcf_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/bbq_sauce/mjcf/bbq_sauce.xml",
        ),
        ArticulationObjCfg(
            name="box_base",
            fix_base_link=True,
            default_position=[0.5, 0.2, 0.1],
            default_orientation=[0.0, 0.7071, 0.0, 0.7071],
            usd_path="roboverse_data/assets/rlbench/close_box/box_base/usd/box_base.usd",
            urdf_path="roboverse_data/assets/rlbench/close_box/box_base/urdf/box_base_unique.urdf",
            mjcf_path="roboverse_data/assets/rlbench/close_box/box_base/mjcf/box_base_unique.mjcf",
        ),
    ]

    from metasim.utils.setup_util import get_handler

    handler = get_handler(scenario)
    handler.set_dof_targets(
        [
            {
                "franka": {
                    "dof_pos_target": {
                        "panda_joint1": 0.0,
                        "panda_joint2": -0.785398,
                        "panda_joint3": 0.0,
                        "panda_joint4": -2.356194,
                        "panda_joint5": 0.0,
                        "panda_joint6": 1.570796,
                        "panda_joint7": 0.785398,
                        "panda_finger_joint1": 0.04,
                        "panda_finger_joint2": 0.04,
                    }
                }
            }
        ]
        * scenario.num_envs
    )
    # handler.simulate()  # need step once to update the kinematics in sapien
    state = handler.get_states(mode="dict")
    dof_pos = state[0]["robots"]["franka"]["dof_pos"]
    dof_vel = state[0]["robots"]["franka"]["dof_vel"]
    dof_pos_tensor = torch.Tensor([
        dof_pos[joint_name] for joint_name in sorted(handler.get_joint_names("franka", True))
    ])
    dof_vel_tensor = torch.Tensor([
        dof_vel[joint_name] for joint_name in sorted(handler.get_joint_names("franka", True))
    ])
    init_dof_pos_tensor = torch.Tensor([
        scenario.robots[0].default_joint_positions[joint_name]
        for joint_name in sorted(handler.get_joint_names("franka", True))
    ])
    assert_close(dof_pos_tensor, init_dof_pos_tensor, atol=1e-3, message="DoF pos")
    assert_close(dof_vel_tensor, torch.zeros(9), atol=1e-3, message="DoF vel")

    pos = state[0]["robots"]["franka"]["pos"]
    rot = state[0]["robots"]["franka"]["rot"]
    assert_close(pos, torch.Tensor(scenario.robots[0].default_position), atol=1e-3, message="franka pos")
    assert_close(rot, torch.Tensor(scenario.robots[0].default_orientation), atol=1e-3, message="franka rot")

    pos = state[0]["objects"]["cube"]["pos"]
    rot = state[0]["objects"]["cube"]["rot"]
    assert_close(pos, torch.Tensor(scenario.objects[0].default_position), atol=1e-3, message="cube pos")
    assert_close(rot, torch.Tensor(scenario.objects[0].default_orientation), atol=1e-3, message="cube rot")

    pos = state[0]["objects"]["sphere"]["pos"]
    rot = state[0]["objects"]["sphere"]["rot"]
    assert_close(pos, torch.Tensor(scenario.objects[1].default_position), atol=1e-3, message="sphere pos")
    assert_close(rot, torch.Tensor(scenario.objects[1].default_orientation), atol=1e-3, message="sphere rot")

    pos = state[0]["objects"]["bbq_sauce"]["pos"]
    rot = state[0]["objects"]["bbq_sauce"]["rot"]
    assert_close(pos, torch.Tensor(scenario.objects[2].default_position), atol=1e-3, message="bbq_sauce pos")
    assert_close(rot, torch.Tensor(scenario.objects[2].default_orientation), atol=1e-3, message="bbq_sauce rot")

    pos = state[0]["objects"]["box_base"]["pos"]
    rot = state[0]["objects"]["box_base"]["rot"]
    assert_close(pos, torch.Tensor(scenario.objects[3].default_position), atol=1e-3, message="box_base pos")
    assert_close(rot, torch.Tensor(scenario.objects[3].default_orientation), atol=1e-3, message="box_base rot")

    tensor_state = handler.get_states(mode="tensor")
    assert_close(tensor_state.robots["franka"].joint_pos, init_dof_pos_tensor, atol=1e-3, message="tensor DoF pos")
    assert_close(tensor_state.robots["franka"].joint_vel, torch.zeros(9), atol=1e-3, message="tensor DoF vel")


if __name__ == "__main__":
    # 直接运行时，可以指定要测试的模拟器和环境数量
    import sys

    # 默认参数
    sim = "mujoco"
    num_envs = 1

    # 从命令行获取参数
    if len(sys.argv) > 1:
        sim = sys.argv[1]
    if len(sys.argv) > 2:
        num_envs = int(sys.argv[2])

    log.info(f"Testing {sim} with {num_envs} envs...")
    test_consistency(sim, num_envs)
    log.success(f"✅ Test passed for {sim} with {num_envs} envs!")
