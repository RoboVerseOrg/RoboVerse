from metasim.scenario.grounds import GroundCfg, ObstacleCfg
from metasim.utils import configclass


@configclass
class ObstacleFieldCfg(GroundCfg):
    """Ground filled with rectangular obstacles."""

    width: float = 12.0
    length: float = 12.0
    min_obstacle_size: float = 0.4
    max_obstacle_size: float = 1.0
    num_rectangles: int = 25
    max_height: float = 0.25
    platform_size: float = 1.5

    def __post_init__(self):
        super().__post_init__()

        for key in self.elements.keys():
            self.elements[key].clear()

        self.elements["obstacle"].append(
            ObstacleCfg(
                origin=[0.0, 0.0],
                size=[self.width, self.length],
                rectangle_params=[
                    self.min_obstacle_size,
                    self.max_obstacle_size,
                    self.num_rectangles,
                ],
                max_height=self.max_height,
                platform_size=self.platform_size,
            )
        )
