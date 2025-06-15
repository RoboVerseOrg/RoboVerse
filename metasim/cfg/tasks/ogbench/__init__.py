"""OGBench task configurations."""

from .antmaze_navigate_cfg import (
    AntMazeGiantNavigateCfg,
    AntMazeLargeNavigateCfg,
    AntMazeLargeNavigateSingleTaskCfg,
    AntMazeMediumNavigateCfg,
)
from .cube_play_cfg import (
    CubeDoublePlayCfg,
    CubeDoublePlaySingleTaskCfg,
    CubeQuadruplePlayCfg,
    CubeTriplePlayCfg,
)
from .humanoidmaze_navigate_cfg import (
    HumanoidMazeGiantNavigateCfg,
    HumanoidMazeLargeNavigateCfg,
    HumanoidMazeMediumNavigateCfg,
)
from .ogbench_base import OGBenchBaseCfg
from .ogbench_env import OGBenchEnv
from .ogbench_wrapper import OGBenchWrapper

__all__ = [
    # Base classes
    "OGBenchBaseCfg",
    "OGBenchEnv", 
    "OGBenchWrapper",
    # AntMaze tasks
    "AntMazeLargeNavigateCfg",
    "AntMazeLargeNavigateSingleTaskCfg",
    "AntMazeMediumNavigateCfg",
    "AntMazeGiantNavigateCfg",
    # HumanoidMaze tasks
    "HumanoidMazeLargeNavigateCfg",
    "HumanoidMazeMediumNavigateCfg",
    "HumanoidMazeGiantNavigateCfg",
    # Cube tasks
    "CubeDoublePlayCfg",
    "CubeDoublePlaySingleTaskCfg",
    "CubeTriplePlayCfg",
    "CubeQuadruplePlayCfg",
]