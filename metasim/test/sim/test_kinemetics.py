"""Rest everything follows."""

import pytest
import rootutils
import torch

from metasim.constants import PhysicStateType
from metasim.scenario.objects import ArticulationObjCfg, PrimitiveCubeCfg, PrimitiveSphereCfg, RigidObjCfg
from metasim.scenario.scenario import ScenarioCfg

# from metasim.sim.sim_context import HandlerContext

rootutils.setup_root(__file__, pythonpath=True)
from loguru import logger as log

from metasim.test.test_utils import assert_close, get_test_parameters
from roboverse_pack.robots.franka_cfg import FrankaCfg


@pytest.mark.parametrize("sim,num_envs", get_test_parameters())
def test_consistency(sim, num_envs):
    if sim not in ("sapien3",):
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
    handler.simulate()  # need step once to update the kinematics in sapien
    state = handler.get_states(mode="dict")

    if "body" in state[0]["robots"]["franka"].keys():
        hand_pos = state[0]["robots"]["franka"]["body"]["panda_hand"]["pos"]
        hand_rot = state[0]["robots"]["franka"]["body"]["panda_hand"]["rot"]

        expected_hand_pos = torch.Tensor([3.0689e-01, 0.0, 5.9028e-01])
        expected_hand_rot = torch.Tensor([0.0, 1.0000e00, 0.0, 0.0])

        assert_close(hand_pos, expected_hand_pos, atol=1e-3, message="hand pos")
        assert_close(hand_rot, expected_hand_rot, atol=1e-3, message="hand rot")


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
