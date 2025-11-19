"""Rest everything follows."""

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
    if sim not in ["mujoco"]:
        pytest.skip(f"Skipping simulator {sim} for this test.")
    # initialize scenario
    scenario = ScenarioCfg(
        robots=[FrankaCfg(name="franka1"), FrankaCfg(name="franka2")],
        headless=True,
        num_envs=num_envs,
        simulator=sim,
    )
    from metasim.utils.setup_util import get_handler

    handler = get_handler(scenario)
    handler.simulate()
    robot_keys = list(handler.get_states(mode="tensor").robots.keys())
    assert len(robot_keys) == 2 and "franka1" in robot_keys and "franka2" in robot_keys
    assert robot_keys == ["franka1", "franka2"]
    robot_keys_dict = list(handler.get_states(mode="dict")[0]["robots"].keys())
    assert robot_keys_dict == ["franka1", "franka2"]


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
