"""OGBench task configurations for RoboVerse."""

from .antmaze_navigate_cfg import (
    AntMazeGiantNavigateCfg,
    AntMazeLargeNavigateCfg,
    AntMazeLargeNavigateSingleTaskCfg,
    AntMazeMediumNavigateCfg,
    AntMazeTeleportNavigateCfg,
)
from .antsoccer_cfg import AntSoccerArenaCfg, AntSoccerMediumCfg
from .cube_play_cfg import (
    CubeDoubleCfg,
    CubeDoublePlayCfg,
    CubeDoublePlaySingleTaskCfg,
    CubeQuadrupleCfg,
    CubeQuadruplePlayCfg,
    CubeSingleCfg,
    CubeTripleCfg,
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
from .pointmaze_navigate_cfg import (
    PointMazeGiantNavigateCfg,
    PointMazeLargeNavigateCfg,
    PointMazeMediumNavigateCfg,
    PointMazeTeleportNavigateCfg,
)
from .powderworld_cfg import PowderworldEasyCfg, PowderworldHardCfg, PowderworldMediumCfg
from .puzzle_cfg import Puzzle3x3Cfg, Puzzle4x4Cfg, Puzzle4x5Cfg, Puzzle4x6Cfg
from .scene_cfg import SceneCfg
from .visual_antmaze_cfg import (
    VisualAntMazeGiantCfg,
    VisualAntMazeLargeCfg,
    VisualAntMazeMediumCfg,
    VisualAntMazeTeleportCfg,
)
from .visual_cube_cfg import (
    VisualCubeDoubleCfg,
    VisualCubeQuadrupleCfg,
    VisualCubeSingleCfg,
    VisualCubeTripleCfg,
)
from .visual_humanoidmaze_cfg import (
    VisualHumanoidMazeGiantCfg,
    VisualHumanoidMazeLargeCfg,
    VisualHumanoidMazeMediumCfg,
)
from .visual_puzzle_cfg import (
    VisualPuzzle3x3Cfg,
    VisualPuzzle4x4Cfg,
    VisualPuzzle4x5Cfg,
    VisualPuzzle4x6Cfg,
)
from .visual_scene_cfg import VisualSceneCfg

__all__ = [
    "AntMazeGiantNavigateCfg",
    "AntMazeLargeNavigateCfg",
    "AntMazeLargeNavigateSingleTaskCfg",
    "AntMazeMediumNavigateCfg",
    "AntMazeTeleportNavigateCfg",
    "AntSoccerArenaCfg",
    "AntSoccerMediumCfg",
    "CubeDoubleCfg",
    "CubeDoublePlayCfg",
    "CubeDoublePlaySingleTaskCfg",
    "CubeQuadrupleCfg",
    "CubeQuadruplePlayCfg",
    "CubeSingleCfg",
    "CubeTripleCfg",
    "CubeTriplePlayCfg",
    "HumanoidMazeGiantNavigateCfg",
    "HumanoidMazeLargeNavigateCfg",
    "HumanoidMazeMediumNavigateCfg",
    "OGBenchBaseCfg",
    "OGBenchEnv",
    "OGBenchWrapper",
    "PointMazeGiantNavigateCfg",
    "PointMazeLargeNavigateCfg",
    "PointMazeMediumNavigateCfg",
    "PointMazeTeleportNavigateCfg",
    "PowderworldEasyCfg",
    "PowderworldHardCfg",
    "PowderworldMediumCfg",
    "Puzzle3x3Cfg",
    "Puzzle4x4Cfg",
    "Puzzle4x5Cfg",
    "Puzzle4x6Cfg",
    "SceneCfg",
    "VisualAntMazeGiantCfg",
    "VisualAntMazeLargeCfg",
    "VisualAntMazeMediumCfg",
    "VisualAntMazeTeleportCfg",
    "VisualCubeDoubleCfg",
    "VisualCubeQuadrupleCfg",
    "VisualCubeSingleCfg",
    "VisualCubeTripleCfg",
    "VisualHumanoidMazeGiantCfg",
    "VisualHumanoidMazeLargeCfg",
    "VisualHumanoidMazeMediumCfg",
    "VisualPuzzle3x3Cfg",
    "VisualPuzzle4x4Cfg",
    "VisualPuzzle4x5Cfg",
    "VisualPuzzle4x6Cfg",
    "VisualSceneCfg",
]
