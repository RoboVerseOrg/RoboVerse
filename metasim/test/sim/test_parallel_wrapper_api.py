from __future__ import annotations

from dataclasses import dataclass, field

import pytest
import rootutils
import torch

rootutils.setup_root(__file__, pythonpath=True)

from metasim.queries.base import BaseQueryType
from metasim.sim.base import BaseSimHandler
from metasim.sim.parallel import ParallelSimWrapper
from metasim.types import TensorState


class _OnesQuery(BaseQueryType):
    """Pickle-safe query that returns ones for each worker."""

    def __call__(self):
        # Each worker has num_envs=1; the wrapper should concatenate across workers.
        return torch.ones((1, 1), dtype=torch.float32)


@dataclass
class _Named:
    name: str


@dataclass
class _FakeScenario:
    simulator: str = "mujoco"
    num_envs: int = 2
    headless: bool = True
    decimation: int = 1
    robots: list[_Named] = field(default_factory=lambda: [_Named("robot")])
    objects: list[_Named] = field(default_factory=list)
    cameras: list = field(default_factory=list)
    lights: list = field(default_factory=list)

    def check_assets(self) -> None:
        # Unit tests should never download assets.
        return None


class _FakeHandler(BaseSimHandler):
    """Minimal handler implementation for wrapper API tests."""

    def __init__(self, scenario: _FakeScenario, optional_queries=None):
        super().__init__(scenario, optional_queries)
        self._device = torch.device("cpu")

    @property
    def device(self) -> torch.device:
        return self._device

    def launch(self) -> None:
        return None

    def render(self) -> None:
        return None

    def close(self) -> None:
        return None

    def _set_states(self, states, env_ids=None) -> None:
        return None

    def _get_states(self, env_ids=None) -> TensorState:
        empty = {}
        return TensorState(objects=empty, robots=empty, cameras=empty, extras=self.get_extra())

    def _simulate(self):
        return None

    def _get_joint_names(self, obj_name: str, sort: bool = True) -> list[str]:
        joints = ["b_joint", "a_joint", "c_joint"]
        return sorted(joints) if sort else joints

    def _get_body_names(self, obj_name: str, sort: bool = True) -> list[str]:
        bodies = ["torso", "arm", "leg"]
        return sorted(bodies) if sort else bodies


@pytest.mark.general
def test_parallel_wrapper_forwards_sort_kwargs():
    Wrapped = ParallelSimWrapper(_FakeHandler)
    handler = Wrapped(_FakeScenario())
    try:
        assert handler.get_joint_names("robot", sort=False) == ["b_joint", "a_joint", "c_joint"]
        assert handler.get_joint_names("robot", sort=True) == ["a_joint", "b_joint", "c_joint"]
        assert handler.get_body_names("robot", sort=False) == ["torso", "arm", "leg"]
        assert handler.get_body_names("robot", sort=True) == ["arm", "leg", "torso"]
    finally:
        handler.close()


@pytest.mark.general
def test_parallel_wrapper_propagates_optional_queries_to_workers():
    Wrapped = ParallelSimWrapper(_FakeHandler)
    handler = Wrapped(_FakeScenario(), {"q": _OnesQuery()})
    try:
        handler.launch()
        state = handler.get_states()
        assert "q" in state.extras
        assert isinstance(state.extras["q"], torch.Tensor)
        assert state.extras["q"].shape == (handler.scenario.num_envs, 1)
        assert torch.allclose(state.extras["q"], torch.ones((handler.scenario.num_envs, 1), dtype=torch.float32))
    finally:
        handler.close()
