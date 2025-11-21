"""Rest everything follows."""

# try:
#     import isaacgym
# except ImportError:
#     pass

import pytest
import rootutils

# from metasim.sim.sim_context import HandlerContext
from loguru import logger as log

from metasim.scenario.scenario import ScenarioCfg

rootutils.setup_root(__file__, pythonpath=True)
from metasim.test.test_utils import get_test_parameters
from roboverse_pack.robots.franka_cfg import FrankaCfg


@pytest.mark.parametrize("sim,num_envs", get_test_parameters())
def test_1_robot(sim, num_envs):
    # initialize scenario

    scenario = ScenarioCfg(
        robots=[
            FrankaCfg(
                default_joint_positions={
                    "panda_joint1": 0.0 - 0.1,
                    "panda_joint2": -0.785398 - 0.1,
                    "panda_joint3": 0.0 - 0.1,
                    "panda_joint4": -2.356194 - 0.1,
                    "panda_joint5": 0.0 - 0.1,
                    "panda_joint6": 1.570796 + 0.1,
                    "panda_joint7": 0.785398 + 0.1,
                    "panda_finger_joint1": 0.0,
                    "panda_finger_joint2": 0.0,
                },
                default_position=(0, 0, 1.0),
            )
        ],
        headless=True,
        num_envs=num_envs,
        simulator=sim,
    )
    from metasim.utils.setup_util import get_handler

    handler = get_handler(scenario)
    handler.set_dof_targets(
        [
            {
                "franka": {
                    "dof_pos_target": {
                        "panda_joint1": -3.0,
                        "panda_joint2": +2.0,
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
    states_default = handler.get_states(mode="dict")
    for i in range(10):
        handler.simulate()
    states_after = handler.get_states(mode="dict")
    assert (
        states_after[0]["robots"]["franka"]["dof_pos"]["panda_joint1"]
        >= FrankaCfg().joint_limits["panda_joint1"][0] - 1e-3
    )
    assert (
        states_after[0]["robots"]["franka"]["dof_pos"]["panda_joint2"]
        <= FrankaCfg().joint_limits["panda_joint2"][1] + 1e-3
    )


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
    test_1_robot(sim, num_envs)
    log.success(f"✅ Test passed for {sim} with {num_envs} envs!")
