from metasim.scenario.grounds import GroundCfg, SlopeCfg
from metasim.utils import configclass


@configclass
class SingleSlopeCfg(GroundCfg):
    """Ground made of a single slope covering the full area."""

    width: float = 10.0
    length: float = 10.0
    repeat_direction_gap = (1, "row", 5.0)

    def __post_init__(self):
        super().__post_init__()

        for key in self.elements.keys():
            self.elements[key].clear()

        self.elements["slope"].append(
            SlopeCfg(
                origin=[0.0, 0.0],
                size=[self.width, self.length],
                slope=0.2,
                random=False,
                platform_size=2.0,
            )
        )
