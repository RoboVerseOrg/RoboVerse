API
==============

RoboVerse depends on the standalone ``metasim`` package for the core simulation API. This repository no longer vendors the ``metasim`` source tree; it installs MetaSim from the standalone repository and registers RoboVerse content packages through entry points and ``metasim.toml``.

The reference below is for the installed MetaSim dependency used by RoboVerse.

metasim
-------

.. currentmodule:: metasim

.. autosummary::
    :toctree: metasim

    example
    queries
    scenario
    sim
    task
    test
    utils
    constants
    types
