"""``ParallelSimWrapper`` must forward ``extra_spec`` into its workers.

Each backend's ``_get_states`` calls ``self.get_extra()`` (which iterates
``self.optional_queries``) and stores the result in ``TensorState.extras``.
The parallel workers were constructed as ``base_cls(sub_scenario)`` with NO
``extra_spec``, so every worker's ``optional_queries`` was empty, its
``get_extra()`` returned ``{}``, and ``state.extras`` was always empty for
num_envs>1 — i.e. site / contact / custom queries were silently dead in
exactly the multi-env config used for eval, while single-env handlers
populated them. The fix passes ``extra_spec`` into the worker handler.

This test spawns real worker processes (the only honest way to guard the
``__init__`` construction line); it uses a tiny stub backend + a tensor-valued
query, no simulator. Tensor-valued query results are concatenated across
workers by ``join_tensor_states``; non-tensor results (e.g. ContactForces)
are a separate, documented contract issue and are dropped by the join, not
exercised here.
"""

from __future__ import annotations

import pytest
import torch

from metasim.queries.base import BaseQueryType
from metasim.scenario.scenario import ScenarioCfg
from metasim.sim.parallel import ParallelSimWrapper
from metasim.types import TensorState


class _OnesQuery(BaseQueryType):
    """A tensor-valued query: returns a (1, 3) tensor per (single-env) worker."""

    def __call__(self):
        return torch.ones(1, 3)


class _ExtrasStubBackend:
    """Duck-typed single-env backend that, like the real backends, folds
    ``get_extra()`` into the TensorState it returns. Module-level so spawn /
    forkserver can re-import it in each worker."""

    def __init__(self, scenario, extra_spec=None):
        self.scenario = scenario
        self.num_envs = scenario.num_envs  # 1 inside a worker
        self.optional_queries = extra_spec or {}

    def launch(self):
        for q in self.optional_queries.values():
            q.bind_handler(self)

    def get_extra(self):
        return {name: q() for name, q in self.optional_queries.items()}

    def get_states(self, **kwargs):
        return TensorState(objects={}, robots={}, cameras={}, extras=self.get_extra())

    # --- remaining no-op surface the worker loop / wrapper may call ---
    def set_states(self, states):
        pass

    def simulate(self):
        pass

    def render(self):
        pass

    def refresh_render(self):
        pass

    def set_dof_targets(self, actions):
        pass

    def set_seed(self, seed):
        pass

    def get_joint_names(self, obj_name, sort=True):
        return []

    def get_body_names(self, obj_name, sort=True):
        return []

    @property
    def device(self):
        return "cpu"

    def close(self):
        pass


@pytest.mark.general
def test_parallel_workers_receive_extra_spec_and_populate_extras():
    """A tensor-valued query must survive into ``get_states().extras`` at
    num_envs>1 (was always empty because workers got no extra_spec)."""
    scenario = ScenarioCfg(simulator="mujoco", num_envs=2, headless=True, robots=[])
    wrapped_cls = ParallelSimWrapper(_ExtrasStubBackend)

    handler = wrapped_cls(scenario, {"ones": _OnesQuery()})
    try:
        handler.launch()
        states = handler.get_states(mode="tensor")
        assert isinstance(states, TensorState)
        assert "ones" in states.extras, f"query result missing from extras: {states.extras!r}"
        # one (1,3) per worker -> concatenated to (num_envs, 3)
        assert states.extras["ones"].shape == (2, 3)
        assert torch.allclose(states.extras["ones"], torch.ones(2, 3))
    finally:
        handler.close()
