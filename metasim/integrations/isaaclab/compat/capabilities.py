from __future__ import annotations

"""Declarative compatibility capabilities for MetaSim ↔ IsaacLab manager-based shim.

This module centralizes backend support checks so registries/managers can answer:
    "Is feature X supported on backend Y?"
in a deterministic way, with structured guidance in strict mode.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class CapabilitySupport:
    supported: bool
    reason: str = ""
    how_to: str | None = None


class CapabilityRegistry:
    """A small, declarative registry of backend capabilities.

    Backends are identified by their MetaSim scenario simulator string (e.g., "mujoco").
    """

    # Capability identifiers (stringly-typed for ease of use across shims).
    OPTIONAL_QUERY_CONTACT_FORCES = "optional_query.contact_forces"
    ISAACSIM_PHYSX_VIEWS = "isaacsim.physx_views"

    def __init__(self) -> None:
        self._support_matrix: dict[str, set[str] | None] = {
            # Contact forces query currently implemented for these backends.
            self.OPTIONAL_QUERY_CONTACT_FORCES: {"isaacgym", "isaacsim", "mujoco", "newton", "mjx", "pybullet"},
            # PhysX views are available when running on the IsaacSim handler (through its IsaacLab scene/assets).
            self.ISAACSIM_PHYSX_VIEWS: {"isaacsim"},
        }
        self._guidance: dict[str, str] = {
            self.OPTIONAL_QUERY_CONTACT_FORCES: (
                "Implement a handler-specific contact force source and wire it into "
                "`metasim/queries/contact_force.py` (and ensure the query can run under multiprocessing)."
            ),
            self.ISAACSIM_PHYSX_VIEWS: (
                "Expose IsaacSim PhysX views (e.g., `root_physx_view`) through the compat scene/assets layer, "
                "or gate the event/term to isaacsim-only and provide a no-op fallback."
            ),
        }

    def check(self, *, capability: str, backend: str | None) -> CapabilitySupport:
        if backend is None:
            backend = "unknown"
        if capability not in self._support_matrix:
            return CapabilitySupport(
                supported=False,
                reason=f"unknown capability '{capability}'.",
                how_to="Add this capability to `CapabilityRegistry._support_matrix` with supported backends and guidance.",
            )

        supported_backends = self._support_matrix[capability]
        if supported_backends is None:
            return CapabilitySupport(supported=True)
        if backend in supported_backends:
            return CapabilitySupport(supported=True)

        return CapabilitySupport(
            supported=False,
            reason=f"capability '{capability}' is not supported on backend '{backend}'.",
            how_to=self._guidance.get(capability),
        )


# Default singleton used by compat registries.
CAPABILITIES = CapabilityRegistry()
