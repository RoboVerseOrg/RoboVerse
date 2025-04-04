## TODO: only export task cfg whose suffix is A/B/C/D + Cfg
from .calvin import *

__all__ = [task for task in dir() if task.endswith("ACfg")]
