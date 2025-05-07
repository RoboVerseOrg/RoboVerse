"""Configuration classes for simulator parameters."""

from __future__ import annotations

from metasim.utils import configclass


@configclass
class SimParamCfg:
    """Simulation parameters cfg.

    This class defines the parameters for the simulator.
    It is important to ensure that each task is configured with appropriate simulation
    parameters to avoid divergence or unexpected results.

    Reference for IsaacGym: https://docs.robotsfan.com/isaacgym/api/python/struct_py.html#isaacgym.gymapi.PhysXParams
    """

    # TODO: Currently supports only Isaac Gym. Add compatibility cfgs for other simulators. Also needed to read from scenario as basic.
    timestep: float = 1.0 / 60.0
    substeps: int = 2
    contact_offset: float = 0.001
    num_position_iterations: int = 8
    num_velocity_iterations: int = 1
    bounce_threshold_velocity: float = 0.2
    friction_offset_threshold: float = 0.001
    friction_correlation_distance: float = 0.0005
