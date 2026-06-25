from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from loguru import logger

if TYPE_CHECKING:
    from metasim.scenario.scenario import ScenarioCfg

from metasim.queries.base import BaseQueryType
from metasim.sim.base import BaseSimHandler
from metasim.types import CompatActionInput, TensorState
from metasim.utils.state import state_tensor_to_nested


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

    def launch(self) -> None:
        """Launch both physics and render simulations."""
        self.physics_handler.launch()
        self.render_handler.launch()
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
        # Pull the physics-resolved tensor state (with body_state filled in by
        # whatever FK the physics handler runs) and forward that to the render
        # handler. Necessary for articulations: render handlers typically have
        # no FK of their own and rely on per-body world transforms from physics.
        # Mirrors what ``_simulate`` does after stepping.
        try:
            physics_states = self.physics_handler._get_states(env_ids)
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
            try:
                self.render_handler._set_states(physics_states, env_ids)
            except TypeError:
                states_nested = state_tensor_to_nested(self.physics_handler, physics_states)
                self.render_handler._set_states(states_nested, env_ids)
        else:
            self.render_handler._set_states(self.render_handler._normalise_set_states_input(states), env_ids)

    def _get_states(self, env_ids: list[int] | None = None) -> TensorState:
        """Get states from physics handler and camera data from render handler."""
        # Get physics states (robots and objects)
        physics_states = self.physics_handler._get_states(env_ids)

        # Get render states (mainly for camera data)
        render_states = self.render_handler._get_states(env_ids)

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

        # Get states from physics and sync to render
        physics_states = self.physics_handler._get_states()
        try:
            self.render_handler._set_states(physics_states)
        except TypeError:
            states_nested = state_tensor_to_nested(self.physics_handler, physics_states)
            self.render_handler._set_states(states_nested)

        # Update render and ensure camera data is refreshed
        self.render_handler.refresh_render()

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
