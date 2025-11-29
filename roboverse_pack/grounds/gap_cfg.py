from metasim.scenario.grounds import GapCfg, GroundCfg
from metasim.utils import configclass


@configclass
class SingleGapCfg(GroundCfg):
    """Ground with a single traversable gap."""

    width: float = 10.0
    length: float = 10.0
    gap_size: float = 1.0
    platform_size: float = 2.0

    def __post_init__(self):
        super().__post_init__()

        for key in self.elements.keys():
            self.elements[key].clear()

        self.elements["gap"].append(
            GapCfg(
                origin=[0.0, 0.0],
                size=[self.width, self.length],
                gap_size=self.gap_size,
                platform_size=self.platform_size,
            )
        )
