from metasim.scenario.grounds import GroundCfg, StairCfg
from metasim.utils import configclass


@configclass
class SingleStairCfg(GroundCfg):
    """Ground composed of a single staircase structure."""

    width: float = 10.0
    length: float = 12.0
    step_width: float = 0.35
    step_height: float = 0.12
    platform_size: float = 2.0

    def __post_init__(self):
        super().__post_init__()

        for key in self.elements.keys():
            self.elements[key].clear()

        self.elements["stair"].append(
            StairCfg(
                origin=[0.0, 0.0],
                size=[self.width, self.length],
                step=[self.step_width, self.step_height],
                platform_size=self.platform_size,
            )
        )
