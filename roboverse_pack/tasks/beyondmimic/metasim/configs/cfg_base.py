# Copyright (c) The BeyondMimic authors (HybridRobotics/whole_body_tracking)
# SPDX-License-Identifier: MIT
#
# Adapted from BeyondMimic / whole_body_tracking (https://github.com/HybridRobotics/whole_body_tracking).
# Changes: MetaSim-native environment/callback config base classes written for the port; no single upstream file corresponds to it.
# Full license: roboverse_pack/tasks/beyondmimic/LICENSE.beyondmimic

from __future__ import annotations

from dataclasses import MISSING
from typing import Callable

from metasim.utils import configclass


@configclass
class CallbacksCfg:
    """Configuration for callbacks."""

    setup: dict = {}
    reset: dict = {}
    pre_step: dict = {}
    post_step: dict = {}
    query: dict = {}


@configclass
class BaseEnvCfg:
    """The base class of environment configuration for legged robots."""

    max_episode_length_s = 10.0
    obs_len_history = 1  # number of past observations to include in the observation
    priv_obs_len_history = 1  # number of past privileged observations to include in the privileged observation
    decimation = 4  # task-level

    callbacks_setup: dict[str, tuple[Callable, dict] | Callable] = {}
    callbacks_reset: dict[str, tuple[Callable, dict] | Callable] = {}
    callbacks_pre_step: dict[str, tuple[Callable, dict] | Callable] = {}
    callbacks_post_step: dict[str, tuple[Callable, dict] | Callable] = {}
    callbacks_query: dict[str, tuple[Callable, dict] | Callable] = MISSING

    def __post_init__(self):

        def _normalize(value) -> dict:
            return {} if value is MISSING else value

        self.callbacks = CallbacksCfg()
        self.callbacks.query = _normalize(self.callbacks_query)
        self.callbacks.setup = _normalize(self.callbacks_setup)
        self.callbacks.reset = _normalize(self.callbacks_reset)
        self.callbacks.pre_step = _normalize(self.callbacks_pre_step)
        self.callbacks.post_step = _normalize(self.callbacks_post_step)

        # Type check for callbacks
        for cb_attr in [
            "setup",
            "reset",
            "pre_step",
            "post_step",
            # "terminate",
            "query",
        ]:
            cb_dict = getattr(self.callbacks, cb_attr)
            for func_name, func_tuple in cb_dict.items():
                if not (
                    callable(func_tuple)
                    or (
                        isinstance(func_tuple, tuple)
                        and len(func_tuple) == 2
                        and (callable(func_tuple[0]) or isinstance(func_tuple[0], object))
                        and isinstance(func_tuple[1], dict)
                    )
                ):
                    raise ValueError(
                        f"Callback {func_name} in {cb_attr} must be a callable or a tuple of (callable, dict)."
                    )
