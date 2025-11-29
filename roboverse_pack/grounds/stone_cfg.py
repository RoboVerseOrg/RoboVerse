from metasim.scenario.grounds import GroundCfg, StoneCfg
from metasim.utils import configclass


@configclass
class SteppingStoneCfg(GroundCfg):
    """Ground with sparse stepping stones over a height field."""

    width: float = 10.0
    length: float = 14.0
    stone_size: float = 0.7
    stone_spacing: float = 0.5
    max_height: float = 0.2
    platform_size: float = 1.5

    def __post_init__(self):
        super().__post_init__()

        for key in self.elements.keys():
            self.elements[key].clear()

        self.elements["stone"].append(
            StoneCfg(
                origin=[0.0, 0.0],
                size=[self.width, self.length],
                stone_params=[self.stone_size, self.stone_spacing],
                max_height=self.max_height,
                platform_size=self.platform_size,
            )
        )
