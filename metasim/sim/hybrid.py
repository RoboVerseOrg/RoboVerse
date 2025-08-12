from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from metasim.cfg.scenario import ScenarioCfg

from metasim.queries.base import BaseQueryType
from metasim.sim.base import BaseSimHandler
from metasim.types import Action, TensorState
from metasim.utils.state import TensorState, state_tensor_to_nested


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

    def close(self) -> None:
        """Close both physics and render simulations."""
        self.physics_handler.close()
        self.render_handler.close()

    def set_dof_targets(self, obj_name: str, actions: list[Action]) -> None:
        """Set the dof targets of the robot in the physics handler."""
        self.physics_handler.set_dof_targets(obj_name, actions)

    def _set_states(self, states: TensorState, env_ids: list[int] | None = None) -> None:
        """Set states in both physics and render handlers."""
        self.physics_handler._set_states(states, env_ids)
        self.render_handler._set_states(states, env_ids)

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
        )

    def _simulate(self):
        """Simulate physics and sync render state."""
        # Simulate physics
        self.physics_handler._simulate()

        # Get states from physics and sync to render
        physics_states = self.physics_handler._get_states()
        states_nested = state_tensor_to_nested(self.physics_handler, physics_states)
        self.render_handler._set_states(states_nested)

        # Update render and ensure camera data is refreshed
        self.render_handler.refresh_render()
        # Also run a simulation step in render handler to update sensors
        self.render_handler._simulate()

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
