"""State-cache invalidation must close re-entrant gaps (M7).

The previous ``set_states`` / ``simulate`` invalidated the cache *before*
the underlying ``_set_states`` / ``_simulate`` ran. That meant a
re-entrant ``get_states`` call from inside the mutation (e.g. a backend
hook that snapshots state, or a callback that introspects) could refill
the cache with the *pre-mutation* state, and that stale value would
survive the mutation.

After the fix:

- ``set_states`` and ``simulate`` invalidate *after* their inner call,
  wrapped in ``try/finally`` so the cache is cleared even when the
  mutation raises.
- A new public ``invalidate_state_caches`` exists for code paths that
  mutate sim state outside the public wrappers (randomizers, custom
  scripts).
"""

from __future__ import annotations

from typing import ClassVar

import pytest

from metasim.sim.base import BaseSimHandler
from metasim.types import TensorState


class _RecordingHandler(BaseSimHandler):
    """In-memory handler that lets the test observe cache state during
    a mutation. We bypass BaseSimHandler.__init__ so we don't need a
    real scenario.

    ``_state_value`` is the "actual" simulator state. ``_set_states``
    updates it. ``_get_states`` returns it. The class records the
    cached value visible to a re-entrant ``get_states`` inside
    ``_set_states``.
    """

    seen_during_set: ClassVar[list] = []
    seen_during_sim: ClassVar[list] = []

    def __init__(self):  # noqa: D401 — minimal stub init
        self._tensor_state_cache = None
        self._dict_state_cache = None
        self._state_value = "initial"

    def _get_states(self, env_ids=None):
        # Return as a tensor-cache-compatible payload by short-circuiting
        # the conversion: we cast to TensorState only at the surface to
        # please isinstance checks. The test only inspects cache identity,
        # not the converted contents.
        return _CacheableValue(self._state_value)

    def _set_states(self, states, env_ids=None):
        # Re-entrantly read state mid-mutation. Pre-fix, this would
        # refill the cache with the pre-mutation value and that cached
        # value would survive once we set _state_value below.
        type(self).seen_during_set.append(self._read_cache_directly())
        self._state_value = states.value if hasattr(states, "value") else states

    def _set_dof_targets(self, actions):
        return None

    def _simulate(self):
        type(self).seen_during_sim.append(self._read_cache_directly())
        self._state_value = "after-step"

    def _read_cache_directly(self):
        """Trigger the cache-fill path without going through tensor conversion."""
        if self._tensor_state_cache is None and self._dict_state_cache is None:
            result = self._get_states()
            if isinstance(result, TensorState):
                self._tensor_state_cache = result
            else:
                self._dict_state_cache = result
        if self._tensor_state_cache is not None:
            return self._tensor_state_cache.value
        return self._dict_state_cache.value


def _CacheableValue(value):
    """Build an empty TensorState whose ``extras['value']`` carries the marker.

    Using ``TensorState(objects={}, robots={}, cameras={})`` keeps the
    isinstance(result, TensorState) branch in ``get_states`` selecting the
    tensor cache.
    """
    state = TensorState(objects={}, robots={}, cameras={})
    state.extras["value"] = value
    state.value = value
    return state


@pytest.fixture(autouse=True)
def _reset_class_state():
    _RecordingHandler.seen_during_set = []
    _RecordingHandler.seen_during_sim = []


def _val(state) -> str:
    """Pull the test-marker value out of a TensorState cache payload."""
    return state.value if state is not None else None


@pytest.mark.general
def test_set_states_invalidates_after_mutation():
    """A re-entrant read during _set_states must not poison the cache."""
    h = _RecordingHandler()

    # Prime the cache so it holds "initial".
    assert _val(h.get_states(mode="tensor")) == "initial"
    assert _val(h._tensor_state_cache) == "initial"

    h.set_states(_CacheableValue("new-state"))

    # The re-entrant read inside _set_states saw the (still-pre-mutation)
    # cached value — that is normal, the simulator hasn't applied the
    # change yet at that point.
    assert _RecordingHandler.seen_during_set == ["initial"]

    # The cache MUST have been cleared after _set_states finished, so the
    # next get_states refetches the new state.
    assert h._tensor_state_cache is None
    assert h._dict_state_cache is None
    assert _val(h.get_states(mode="tensor")) == "new-state"


@pytest.mark.general
def test_simulate_invalidates_after_step():
    """Re-entrant reads during _simulate must not survive the step."""
    h = _RecordingHandler()

    assert _val(h.get_states(mode="tensor")) == "initial"
    h.simulate()

    assert _RecordingHandler.seen_during_sim == ["initial"]
    assert h._tensor_state_cache is None
    # After the step, the simulator's state is "after-step".
    assert _val(h.get_states(mode="tensor")) == "after-step"


@pytest.mark.general
def test_set_states_invalidates_even_on_exception():
    """If _set_states raises, the cache must still be invalidated.

    Partial mutations are possible (the sim may have applied some of the
    change before erroring); the safe default is to refetch next time.
    """

    class _Raises(_RecordingHandler):
        def _set_states(self, states, env_ids=None):
            self._state_value = "partial"  # simulate partial application
            raise RuntimeError("boom")

    h = _Raises()
    h.get_states(mode="tensor")  # prime
    assert _val(h._tensor_state_cache) == "initial"

    with pytest.raises(RuntimeError, match="boom"):
        h.set_states(_CacheableValue("never-reaches-here"))

    assert h._tensor_state_cache is None
    assert _val(h.get_states(mode="tensor")) == "partial"


@pytest.mark.general
def test_simulate_invalidates_even_on_exception():
    """Same guarantee for simulate."""

    class _Raises(_RecordingHandler):
        def _simulate(self):
            self._state_value = "half-stepped"
            raise RuntimeError("step-failure")

    h = _Raises()
    h.get_states(mode="tensor")
    assert _val(h._tensor_state_cache) == "initial"

    with pytest.raises(RuntimeError, match="step-failure"):
        h.simulate()

    assert h._tensor_state_cache is None
    assert _val(h.get_states(mode="tensor")) == "half-stepped"


@pytest.mark.general
def test_public_invalidate_clears_both_caches():
    """The new ``invalidate_state_caches`` public method clears both caches.

    This is the documented hook for randomizers and custom code that
    mutates sim state outside the public wrappers.
    """
    h = _RecordingHandler()
    h._tensor_state_cache = "stale-tensor"
    h._dict_state_cache = "stale-dict"

    h.invalidate_state_caches()

    assert h._tensor_state_cache is None
    assert h._dict_state_cache is None


@pytest.mark.general
def test_get_states_idempotent_when_unchanged():
    """Successive get_states returns the same cached object."""
    h = _RecordingHandler()
    a = h.get_states(mode="tensor")
    b = h.get_states(mode="tensor")
    # Same object identity: the cache short-circuits the second call.
    assert a is b
    assert _val(a) == "initial"
    assert _val(h._tensor_state_cache) == "initial"
