try:
    import isaacgym  # noqa: F401
except ImportError:
    pass


# from isaaclab.app import AppLauncher

# launch omniverse app
# simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

# import isaacsim.core.utils.stage as stage_utils
# import pytest
# from isaacsim.core.api.simulation_context import SimulationContext
# import isaaclab.sim as sim_utils
# from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
# from isaaclab.markers.config import FRAME_MARKER_CFG, POSITION_GOAL_MARKER_CFG
# from isaaclab.utils.math import random_orientation
# from isaaclab.utils.timer import Timer
import pytest
import rootutils
import torch
from loguru import logger as log

from metasim.constants import PhysicStateType
from metasim.scenario.objects import PrimitiveCubeCfg
from metasim.scenario.scenario import ScenarioCfg

# from metasim.sim.sim_context import HandlerContext

rootutils.setup_root(__file__, pythonpath=True)
from metasim.scenario.simulator_params import SimParamCfg
from metasim.test.test_utils import assert_close, get_test_parameters
from roboverse_pack.robots.franka_cfg import FrankaCfg


@pytest.mark.parametrize("sim,num_envs", get_test_parameters())
def test_consistency(sim, num_envs):
    if sim not in ("sapien3",):
        pytest.skip(f"Skipping simulator {sim} for this test (only sapien3 is supported).")

    scenario = ScenarioCfg(
        simulator=sim,
        num_envs=num_envs,
        headless=True,
        objects=[
            PrimitiveCubeCfg(
                name="cube",
                size=(0.1, 0.1, 0.1),
                color=[1.0, 0.0, 0.0],
                physics=PhysicStateType.RIGIDBODY,
                default_position=[0, 0, 10.0],
            ),
        ],
        robots=[FrankaCfg()],
        sim_params=SimParamCfg(dt=0.001),
        gravity=(0, 0, -1),
        decimation=100,
    )

    from metasim.constants import SimType
    from metasim.utils.setup_util import get_sim_handler_class

    env_class = get_sim_handler_class(SimType(sim))
    env = env_class(scenario)
    env.launch()

    state = env.get_states(mode="dict")
    pos = state[0]["objects"]["cube"]["pos"]
    assert_close(pos, torch.Tensor([0, 0, 10.0]), atol=0.001, message="gravity")

    env.simulate()

    state = env.get_states(mode="dict")
    pos = state[0]["objects"]["cube"]["pos"]
    assert_close(pos, torch.Tensor([0, 0, 9.9950]), atol=0.001, message="gravity")

    env.simulate()

    state = env.get_states(mode="dict")
    pos = state[0]["objects"]["cube"]["pos"]
    assert_close(pos, torch.Tensor([0, 0, 9.9800]), atol=0.001, message="gravity")

    env.simulate()

    state = env.get_states(mode="dict")
    pos = state[0]["objects"]["cube"]["pos"]
    assert_close(pos, torch.Tensor([0, 0, 9.9551]), atol=0.001, message="gravity")

    env.close()


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
