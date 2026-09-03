"""PyRep handler is marked unmaintained until a real port lands (H1).

The original ``PyrepHandler`` diverged from the ``BaseSimHandler``
contract in multiple ways (missing ``env_ids`` params, no ``_simulate``,
references to ``self.task`` and ``self.robot.name`` that don't exist on
the base). Anyone constructing the handler hit ``AttributeError`` deep
inside ``launch()`` with no hint that the handler was broken at the
contract level.

The fix:

- The handler class now imports cleanly without requiring the third-party
  ``pyrep``/``rlbench`` packages (no top-level ``from pyrep import ...``).
- ``__init__`` raises a clear ``NotImplementedError`` explaining the
  situation.
- The abstract-method signatures match ``BaseSimHandler``, so a future
  port can fill in bodies without re-deriving the contract from scratch.
"""

from __future__ import annotations

import inspect

import pytest

from metasim.sim import BaseSimHandler


@pytest.mark.general
def test_pyrep_module_imports_without_external_deps():
    """PyrepHandler must be importable on a machine with no pyrep / rlbench."""
    from metasim.sim.pyrep import PyrepHandler

    assert PyrepHandler is not None


@pytest.mark.general
def test_pyrep_instantiation_raises_clear_error():
    """Constructing the handler raises ``NotImplementedError`` with the
    word "unmaintained" so users have a hint where to look."""
    from metasim.sim.pyrep import PyrepHandler

    with pytest.raises(NotImplementedError, match="unmaintained"):
        PyrepHandler(None, None)


@pytest.mark.general
def test_pyrep_setup_util_dispatch_still_resolves():
    """``SimType.PYREP`` must still resolve via ``get_sim_handler_class``;
    the dispatch table is not what's broken, the handler body is."""
    from metasim.constants import SimType
    from metasim.sim.pyrep import PyrepHandler
    from metasim.utils.setup_util import get_sim_handler_class

    assert get_sim_handler_class(SimType.PYREP) is PyrepHandler


@pytest.mark.general
def test_pyrep_method_signatures_match_base_contract():
    """Future port should not need to re-derive the contract. Spot-check
    each abstract method's keyword-arg surface."""
    from metasim.sim.pyrep import PyrepHandler

    set_states = inspect.signature(PyrepHandler._set_states)
    assert "env_ids" in set_states.parameters, "PyrepHandler._set_states is missing env_ids"

    get_states = inspect.signature(PyrepHandler._get_states)
    assert "env_ids" in get_states.parameters, "PyrepHandler._get_states is missing env_ids"

    # The base class has these as abstract; PyrepHandler must declare them
    # (raising) so the class is not abstract.
    for name in ("_set_states", "_set_dof_targets", "_get_states", "_simulate"):
        assert hasattr(PyrepHandler, name), f"PyrepHandler missing {name}"


@pytest.mark.general
def test_pyrep_handler_subclass_relationship():
    """Sanity: it still subclasses BaseSimHandler so registry consumers
    that ``isinstance`` check are happy."""
    from metasim.sim.pyrep import PyrepHandler

    assert issubclass(PyrepHandler, BaseSimHandler)
