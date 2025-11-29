from metasim.scenario.grounds import GroundCfg, PitCfg
from metasim.utils import configclass


@configclass
class SinglePitCfg(GroundCfg):
    """Ground with a centered pit."""

    width: float = 10.0
    length: float = 10.0
    depth: float = 0.8
    platform_size: float = 2.0

    def __post_init__(self):
        super().__post_init__()

        for key in self.elements.keys():
            self.elements[key].clear()

        self.elements["pit"].append(
            PitCfg(
                origin=[0.0, 0.0],
                size=[self.width, self.length],
                depth=self.depth,
                platform_size=self.platform_size,
            )
        )
