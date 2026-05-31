"""LIBERO-plus passthrough task pack.

Exposes helpers to build native LIBERO-plus benchmark environments 1:1 from
RoboVerse without re-implementing the tasks. LIBERO-plus shares the LIBERO
BDDL/robosuite env format but expands each suite with thousands of perturbation
variants (~10k total across 7 dimensions). Importing this package
auto-registers every available task under ``LiberoPlus/<suite>__<task>`` (a safe
no-op when LIBERO-plus is not installed).
"""

from __future__ import annotations

from ._passthrough import (
    LIBERO_PLUS_SUITES,
    REFERENCE_SUITE_COUNTS,
    discover_liberoplus_tasks,
    list_liberoplus_tasks,
    make_liberoplus_env,
    make_liberoplus_env_from_bddl,
    register_liberoplus_passthrough,
    suite_task_count,
)

__all__ = [
    "LIBERO_PLUS_SUITES",
    "REFERENCE_SUITE_COUNTS",
    "discover_liberoplus_tasks",
    "list_liberoplus_tasks",
    "make_liberoplus_env",
    "make_liberoplus_env_from_bddl",
    "register_liberoplus_passthrough",
    "suite_task_count",
]
