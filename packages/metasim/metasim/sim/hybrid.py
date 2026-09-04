from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

import torch
from loguru import logger

if TYPE_CHECKING:
    from metasim.scenario.scenario import ScenarioCfg

from metasim.queries.base import BaseQueryType
from metasim.sim.base import BaseSimHandler
from metasim.types import CompatActionInput, TensorState
from metasim.utils.state import state_to_device


def _extract_rgb_frame(tensor_state, view: str):
    camera_state = tensor_state.cameras.get(view)
    if camera_state is None or camera_state.rgb is None:
        return None
    frame = camera_state.rgb[0].detach().cpu().numpy()
    if frame.ndim != 3 or frame.shape[-1] < 3:
        return None
    return frame[..., :3]


def _preserve_physics_extras(physics_states: TensorState, render_states: TensorState) -> dict:
    merged = {}
    if isinstance(getattr(physics_states, "extras", None), dict):
        merged.update(physics_states.extras)
    if isinstance(getattr(render_states, "extras", None), dict):
        for key, value in render_states.extras.items():
            merged.setdefault(key, value)
    return merged


class HybridSimHandler(BaseSimHandler):
    """Hybrid simulation handler that uses one simulator for physics and another for rendering."""

    def __init__(
        self,
        scenario: ScenarioCfg,
        physics_handler: BaseSimHandler,
        render_handler: BaseSimHandler,
        optional_queries: dict[str, BaseQueryType] | None = None,
    ):
        super().__init__(scenario, optional_queries)
        self.physics_handler = physics_handler  # physics simulator
        self.render_handler = render_handler  # render simulator
        # renderers whose refresh_render takes `passes` can render a synced step in one pass
        self._renderer_single_pass = "passes" in inspect.signature(render_handler.refresh_render).parameters

    get_states_honours_env_ids = True  # ``_get_states`` slices both sides to ``env_ids``

    @property
    def set_states_restores_velocities(self) -> bool:  # type: ignore[override]
        """Physics owns the state: whether a restore keeps velocities is the physics handler's property."""
        return bool(getattr(self.physics_handler, "set_states_restores_velocities", False))

    @property
    def set_states_restores_dict_velocities(self) -> bool:  # type: ignore[override]
        return bool(getattr(self.physics_handler, "set_states_restores_dict_velocities", False))

    @property
    def set_states_refreshes(self) -> bool:  # type: ignore[override]
        """``set_states`` pushes the physics state into the render handler; whether that write leaves
        the renderer's frame current is the render handler's property.
        """
        return bool(getattr(self.render_handler, "set_states_refreshes", False))

    def refresh_render(self) -> None:
        """Cameras come from the render handler, so a refresh request goes there."""
        self.render_handler.refresh_render()

    def launch(self, **render_launch_kwargs) -> None:
        """Launch both physics and render simulations.

        Keyword arguments are forwarded to the render handler's ``launch`` (e.g. an
        already-running ``simulation_app`` for Isaac Sim, so a second Kit instance is
        not started inside a process that already hosts one).
        """
        self.physics_handler.launch()
        self.render_handler.launch(**render_launch_kwargs)
        super().launch()

    def render(self) -> None:
        """Render using the render handler."""
        self.render_handler.render()

    def render_frame(self, view: str):
        return _extract_rgb_frame(self.render_handler.get_states(mode="tensor"), view)

    def render_frames(self, views):
        tensor_state = self.render_handler.get_states(mode="tensor")
        return {view: _extract_rgb_frame(tensor_state, view) for view in views}

    def close(self) -> None:
        """Close both physics and render simulations.

        Uses try/finally so the render handler's close — which owns the IsaacSim
        app lifecycle (and its close-time hang watchdog) — always runs even if
        the physics handler's close raises or hangs.
        """
        try:
            self.physics_handler.close()
        finally:
            self.render_handler.close()

    @property
    def physics_dt(self) -> float | None:  # type: ignore[override]
        """Physics owns the time base."""
        return self.physics_handler.physics_dt

    @property
    def env_step_s(self) -> float | None:  # type: ignore[override]
        return self.physics_handler.env_step_s

    def set_seed(self, seed: int) -> None:
        """Forward the reproducibility seed to both wrapped handlers.

        ``BaseSimHandler.set_seed`` seeds Python ``random`` + NumPy + Torch
        globally — once per process, not per handler — so calling it from
        the hybrid handler alone is technically enough today. We still
        forward explicitly: if either wrapped backend later adds a
        backend-specific override (Newton warp RNG, Sapien physics
        noise), hybrid keeps the contract intact without further code
        changes.
        """
        super().set_seed(seed)
        self.physics_handler.set_seed(seed)
        self.render_handler.set_seed(seed)

    def _push_to_renderer(self, states, env_ids=None) -> None:
        """Write ``states`` into the renderer and render one frame.

        Renderers that flush after every write (Isaac Sim: two RTX passes per ``_set_states``) are
        told to hold their flush while the writes land, then asked for a single pass; otherwise a
        hybrid step paid for four render passes where one is enough (22 ms → 9 ms per synced
        state for one 256² camera on an RTX 5090; the frame is the same to within RTX noise).
        """
        rh = self.render_handler
        # the renderer declares the state form it consumes (``_set_states_input_type``); convert
        # deterministically instead of catching a TypeError that could come from anywhere
        states = rh._normalise_set_states_input(states)
        prev = rh._defer_all_visual_flushes
        rh._defer_all_visual_flushes = True
        try:
            rh._set_states(states, env_ids) if env_ids is not None else rh._set_states(states)
        finally:
            rh._defer_all_visual_flushes = prev
        if self._renderer_single_pass:
            rh.refresh_render(passes=1)
        else:
            rh.refresh_render()
        # the renderer was driven through its private API, so its public get_states cache is stale
        rh._invalidate_state_caches()

    def _for_renderer(self, states):
        """Physics states on the render handler's device (MuJoCo is CPU, Isaac Sim / Newton are CUDA)."""
        device = getattr(self.render_handler, "device", None)
        if isinstance(states, TensorState) and device is not None:
            return state_to_device(states, device)
        return states

    def _set_dof_targets(self, actions: CompatActionInput) -> None:
        """Set the dof targets of the robot in the physics handler."""
        self.physics_handler.set_dof_targets(actions)

    def _set_states(self, states: TensorState, env_ids: list[int] | None = None) -> None:
        """Set states in both physics and render handlers."""
        # Normalise per wrapped handler: the base ``set_states`` only normalised
        # against the Hybrid handler's own (``"both"``) input type, so a wrapped
        # ``"dict"`` backend (genesis/sapien3/pybullet) would otherwise receive an
        # un-normalised ``TensorState`` it can't index. ``_normalise_set_states_input``
        # is a no-op for ``"both"`` handlers.
        self.physics_handler._set_states(self.physics_handler._normalise_set_states_input(states), env_ids)
        self.physics_handler._invalidate_state_caches()
        # Pull the physics-resolved tensor state (with body_state filled in by
        # whatever FK the physics handler runs) and forward that to the render
        # handler. Necessary for articulations: render handlers typically have
        # no FK of their own and rely on per-body world transforms from physics.
        # Mirrors what ``_simulate`` does after stepping.
        try:
            # full batch on purpose: renderers index a full-batch TensorState with ``env_ids``;
            # ``ParallelHandler._get_states(env_ids)`` would return only the subset rows
            physics_states = self.physics_handler._get_states()
        except Exception as exc:
            # WHY this catch exists: most render handlers (Blender) can't apply a dict-state
            # robot directly (need body_state from physics FK). When physics _get_states fails
            # we fall through to pass the original dict to render, which then often raises a
            # less actionable error ("cannot apply dof_pos directly"). Surface the underlying
            # reason at WARN level so callers can debug (e.g. EGL thread-affinity issues where
            # physics rendering happens on a non-main thread).
            logger.warning(
                "HybridSimHandler.set_states: physics._get_states raised "
                f"({type(exc).__name__}: {exc}); falling back to dict-state path on render handler."
            )
            physics_states = None
        if physics_states is not None:
            self._push_to_renderer(self._for_renderer(physics_states), env_ids)
        else:
            self.render_handler._set_states(self.render_handler._normalise_set_states_input(states), env_ids)
            self.render_handler._invalidate_state_caches()

    def _get_states(self, env_ids: list[int] | None = None) -> TensorState:
        """Get states from physics handler and camera data from render handler."""
        # Get physics states (robots and objects)
        physics_states = self.physics_handler._get_states(env_ids)

        # Get render states (mainly for camera data)
        render_states = self.render_handler._get_states(env_ids)

        # the two sides may disagree on ``env_ids``: ParallelHandler returns the subset rows while
        # Isaac Sim returns the full batch; a TensorState needs one env count, so slice both
        if env_ids is not None:
            env_ids = list(env_ids)
            physics_states = self.physics_handler._enforce_env_subset(physics_states, env_ids)
            render_states = self.render_handler._enforce_env_subset(render_states, env_ids)

        # Combine states: use physics for robots/objects, render for cameras
        return TensorState(
            objects=physics_states.objects,
            robots=physics_states.robots,
            cameras=render_states.cameras,  # Use camera data from render handler
            extras=_preserve_physics_extras(physics_states, render_states),
        )

    def _simulate(self):
        """Simulate physics and sync render state."""
        # Simulate physics
        self.physics_handler._simulate()
        self.physics_handler._invalidate_state_caches()

        # Get states from physics and sync to render, then render exactly once
        self._push_to_renderer(self._for_renderer(self.physics_handler._get_states()))

    def _get_joint_names(self, obj_name: str, sort: bool = True) -> list[str]:
        """Get joint names from physics handler."""
        return self.physics_handler._get_joint_names(obj_name, sort)

    def _get_body_names(self, obj_name: str, sort: bool = True) -> list[str]:
        """Get body names from physics handler."""
        return self.physics_handler._get_body_names(obj_name, sort)

    @property
    def device(self) -> torch.device:
        """Get device from physics handler."""
        return self.physics_handler.device
