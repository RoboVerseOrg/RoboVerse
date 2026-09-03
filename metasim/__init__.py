"""All metasim packages."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _dist_version

try:
    __version__ = _dist_version("roboverse-metasim")
except PackageNotFoundError:  # running from a checkout that was never installed
    __version__ = "0.0.0+unknown"


def register_gym_envs() -> None:
    """Discover task modules and register ``RoboVerse/<task>`` with Gymnasium.

    Call this before ``gymnasium.make`` / ``make_vec`` with RoboVerse env ids.
    Sim-only code paths do not need this.
    """
    from metasim.task.gym_registration import register_all_tasks_with_gym

    register_all_tasks_with_gym()
