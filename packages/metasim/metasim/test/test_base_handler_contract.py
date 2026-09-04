"""``BaseSimHandler``'s abstract surface is consistent (M6).

Before the fix:

- ``_set_states`` / ``_set_dof_targets`` / ``_get_states`` / ``_simulate``
  were ``@abstractmethod`` — must override.
- ``_get_joint_names`` and ``_get_body_names`` had a *commented-out*
  ``# @abstractmethod`` marker but their bodies still ``raise
  NotImplementedError``. The docstrings said "for non-articulation
  objects, return an empty list" — yet calling them on a primitive
  blew up. Stale "someday I'll decorate this" comments + body that
  contradicted the docstring.
- ``render``, ``close``, ``device`` were ``raise NotImplementedError``
  without ``@abstractmethod``. Subclasses that forgot ``close`` silently
  failed only at teardown.

This branch's fix (minimal, no new ``@abstractmethod`` to keep blast
radius small):

- ``_get_joint_names`` and ``_get_body_names`` now return ``[]``,
  matching their docstring contract for non-articulation backends.
- Stale ``# @abstractmethod`` comments removed.
- Class docstring spells out the actual contract surface so future
  contributors don't have to reverse-engineer the rules.

Strictening ``close`` / ``device`` to ``@abstractmethod`` is left for
a follow-up branch since it requires touching every backend stub.
"""

from __future__ import annotations

import inspect

import pytest

from metasim.sim.base import BaseSimHandler


# A minimal concrete subclass — implements only the truly abstract methods.
class _MinimalHandler(BaseSimHandler):
    def _set_states(self, states, env_ids=None):
        return None

    def _set_dof_targets(self, actions):
        return None

    def _get_states(self, env_ids=None):
        return {}

    def _simulate(self):
        return None


def _make_minimal():
    h = _MinimalHandler.__new__(_MinimalHandler)
    h._tensor_state_cache = None
    h._dict_state_cache = None
    h._defer_all_visual_flushes = False
    return h


@pytest.mark.general
def test_minimal_handler_is_concrete():
    """A subclass implementing only the four documented abstracts can
    be instantiated. Prevents accidental abstractmethod creep.
    """
    h = _make_minimal()
    assert isinstance(h, BaseSimHandler)


@pytest.mark.general
def test_get_joint_names_default_returns_empty_list():
    """Pre-fix this raised ``NotImplementedError`` while the docstring
    said non-articulations return an empty list — contradictory.
    """
    h = _make_minimal()
    assert h._get_joint_names("any_obj") == []
    assert h.get_joint_names("any_obj") == []


@pytest.mark.general
def test_get_body_names_default_returns_empty_list():
    h = _make_minimal()
    assert h._get_body_names("any_obj") == []
    assert h.get_body_names("any_obj") == []


@pytest.mark.general
def test_get_joint_names_default_honours_sort_kwarg():
    """Subclasses pass ``sort=False`` to recover insertion order. The
    default must accept that kwarg without TypeError.
    """
    h = _make_minimal()
    assert h._get_joint_names("any_obj", sort=False) == []
    assert h._get_joint_names("any_obj", sort=True) == []


@pytest.mark.general
def test_abstract_method_set_documents_the_contract():
    """The four required methods are the four declared in the contract
    summary in the class docstring. If you add an abstract method,
    update the docstring.
    """
    abstracts = BaseSimHandler.__abstractmethods__
    assert abstracts == frozenset({"_set_states", "_set_dof_targets", "_get_states", "_simulate"}), (
        f"abstract methods changed unexpectedly: {sorted(abstracts)}"
    )


@pytest.mark.general
def test_no_stale_abstractmethod_comments_remain():
    """Lock the regression: ``# @abstractmethod`` placeholder comments
    were the source of M6's "is it abstract or isn't it" confusion.
    """
    src = inspect.getsource(BaseSimHandler)
    # Allow the word inside string literals / docstrings; reject
    # commented-out decorator placeholders specifically.
    bad_lines = [line for line in src.splitlines() if line.strip().startswith("# @abstractmethod")]
    assert bad_lines == [], f"stale # @abstractmethod placeholders: {bad_lines}"


@pytest.mark.general
def test_class_docstring_lists_the_contract():
    """The contract is now in the docstring — make sure it stays there.
    A drive-by edit that wipes the contract section would defeat the
    audit fix.
    """
    doc = BaseSimHandler.__doc__ or ""
    for required_method in ("_set_states", "_set_dof_targets", "_get_states", "_simulate"):
        assert required_method in doc, f"contract summary missing reference to {required_method}"
    assert "Optional" in doc, "contract summary missing the 'optional / defaults' section"


@pytest.mark.general
def test_get_states_subset_does_not_poison_full_cache():
    """``get_states(env_ids=subset)`` must not be cached as the full-env state.

    Backends that honour ``env_ids`` (ParallelHandler, mjx/genesis/newton)
    return a real subset. Previously the first ``get_states`` populated the
    shared cache regardless of ``env_ids``, so a subset request poisoned a later
    full request (it returned the stale subset) and vice-versa. The fix bypasses
    the cache for subset requests.
    """
    from metasim.types import TensorState

    class _EnvIdsHandler(BaseSimHandler):
        """``_get_states`` honours env_ids; marks env count in ``extras``."""

        def __init__(self):
            self._tensor_state_cache = None
            self._dict_state_cache = None
            self.calls: list = []

        def _set_states(self, states, env_ids=None):
            return None

        def _set_dof_targets(self, actions):
            return None

        def _simulate(self):
            return None

        def _get_states(self, env_ids=None):
            self.calls.append(env_ids)
            n = len(env_ids) if env_ids is not None else 99  # 99 == "all envs"
            return TensorState(objects={}, robots={}, cameras={}, extras={"n": n})

    h = _EnvIdsHandler()

    # Subset request returns the subset and is NOT cached.
    sub = h.get_states(env_ids=[0], mode="tensor")
    assert sub.extras["n"] == 1

    # Full request must refetch (env_ids=None), not return the stale subset.
    full = h.get_states(mode="tensor")
    assert full.extras["n"] == 99
    assert h.calls == [[0], None]

    # Full result IS cached now: a second full request does not refetch.
    full2 = h.get_states(mode="tensor")
    assert full2 is full
    assert h.calls == [[0], None]

    # A subset request after a full cache must still fetch a fresh subset,
    # not return the cached full state.
    sub2 = h.get_states(env_ids=[1, 2], mode="tensor")
    assert sub2.extras["n"] == 2
    assert h.calls == [[0], None, [1, 2]]


@pytest.mark.general
def test_get_states_dict_mode_subset_not_cached():
    """Same anti-poisoning contract as the tensor case, for ``mode="dict"``.

    Exercises the dict-mode subset branch (``state_tensor_to_nested`` on the
    subset result) which the tensor-only test above does not cover.
    """
    import torch

    from metasim.types import RobotState, TensorState

    class _DictEnvIdsHandler(BaseSimHandler):
        def __init__(self):
            self._tensor_state_cache = None
            self._dict_state_cache = None
            self.calls: list = []

        def _set_states(self, states, env_ids=None):
            return None

        def _set_dof_targets(self, actions):
            return None

        def _simulate(self):
            return None

        def _get_joint_names(self, obj_name, sort=True):
            return ["j0"]

        def _get_body_names(self, obj_name, sort=True):
            return []

        def _get_states(self, env_ids=None):
            self.calls.append(env_ids)
            n = len(env_ids) if env_ids is not None else 5  # 5 == "all envs"
            robot = RobotState(root_state=torch.zeros(n, 13), joint_pos=torch.zeros(n, 1), joint_vel=torch.zeros(n, 1))
            return TensorState(objects={}, robots={"r0": robot}, cameras={})

    h = _DictEnvIdsHandler()

    sub = h.get_states(env_ids=[0], mode="dict")
    assert len(sub) == 1  # one per-env dict
    full = h.get_states(mode="dict")
    assert len(full) == 5  # full set, not the stale 1-env subset
    assert h.calls == [[0], None]
    sub2 = h.get_states(env_ids=[1, 2], mode="dict")
    assert len(sub2) == 2
    assert h.calls == [[0], None, [1, 2]]


@pytest.mark.general
def test_get_states_env_ids_permutation_is_honoured_on_backends_that_ignore_env_ids():
    """A backend that returns the full batch for any ``env_ids`` used to pass the length check for a
    permutation or duplication of the full batch, so ``get_states(env_ids=[2, 0, 1])`` came back in
    natural order and the caller read env 2's state as env 0's.
    """
    import torch

    from metasim.types import RobotState, TensorState

    class _IgnoresEnvIds(BaseSimHandler):
        num_envs = 3

        def __init__(self):
            self._tensor_state_cache = None
            self._dict_state_cache = None

        def _set_states(self, states, env_ids=None):
            return None

        def _set_dof_targets(self, actions):
            return None

        def _simulate(self):
            return None

        def _get_joint_names(self, obj_name, sort=True):
            return ["j0"]

        def _get_body_names(self, obj_name, sort=True):
            return []

        def _get_states(self, env_ids=None):
            root = torch.zeros(3, 13)
            root[:, 0] = torch.tensor([0.0, 1.0, 2.0])  # x = env index
            return TensorState(objects={}, robots={"r0": RobotState(root_state=root)}, cameras={})

    h = _IgnoresEnvIds()
    permuted = h.get_states(env_ids=[2, 0, 1], mode="tensor")
    assert permuted.robots["r0"].root_state[:, 0].tolist() == [2.0, 0.0, 1.0]
    dup = h.get_states(env_ids=[1, 1], mode="tensor")
    assert dup.robots["r0"].root_state[:, 0].tolist() == [1.0, 1.0]
    natural = h.get_states(env_ids=[0, 1, 2], mode="tensor")
    assert natural.robots["r0"].root_state[:, 0].tolist() == [0.0, 1.0, 2.0]


@pytest.mark.general
def test_get_states_env_ids_on_a_backend_that_honours_them_is_returned_as_is():
    """A backend declaring ``get_states_honours_env_ids`` already returns the requested rows in order;
    slicing them again would permute a permutation (the bug a length-only check invited).
    """
    import torch

    from metasim.types import RobotState, TensorState

    class _Honours(BaseSimHandler):
        num_envs = 3
        get_states_honours_env_ids = True

        def __init__(self):
            self._tensor_state_cache = None
            self._dict_state_cache = None

        def _set_states(self, states, env_ids=None):
            return None

        def _set_dof_targets(self, actions):
            return None

        def _simulate(self):
            return None

        def _get_joint_names(self, obj_name, sort=True):
            return ["j0"]

        def _get_body_names(self, obj_name, sort=True):
            return []

        def _get_states(self, env_ids=None):
            ids = list(range(3)) if env_ids is None else list(env_ids)
            root = torch.zeros(len(ids), 13)
            root[:, 0] = torch.tensor([float(i) for i in ids])
            return TensorState(objects={}, robots={"r0": RobotState(root_state=root)}, cameras={})

    h = _Honours()
    assert h.get_states(env_ids=[2, 0, 1], mode="tensor").robots["r0"].root_state[:, 0].tolist() == [2.0, 0.0, 1.0]
    assert h.get_states(env_ids=[1, 0], mode="tensor").robots["r0"].root_state[:, 0].tolist() == [1.0, 0.0]
