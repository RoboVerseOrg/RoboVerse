"""Camera presets for benchmark runs: the view a visual tier observes, named once and recorded in the result.

Most tasks configure no camera (their demos are joint trajectories), so a run under ``visual`` /
``agent-*`` access has nothing to observe. ``bench run --camera front`` adds a preset camera to the
scenario; the result records which presets were used so two visual-tier runs are comparable only when
they looked from the same place.
"""

from __future__ import annotations

from metasim.scenario.cameras import PinholeCameraCfg

CAMERA_PRESETS: dict[str, PinholeCameraCfg] = {
    # tabletop views for a single-arm robot standing at -x looking towards +x (ManiSkill / RLBench ports);
    # the table top is z = 0 and the workspace is x in [-0.2, 0.3], y in [-0.3, 0.3]
    "front": PinholeCameraCfg(
        name="front",
        width=256,
        height=256,
        pos=(0.9, 0.0, 0.65),
        look_at=(-0.05, 0.0, 0.05),
        data_types=["rgb", "depth"],
    ),
    "side": PinholeCameraCfg(
        name="side",
        width=256,
        height=256,
        pos=(0.0, 0.9, 0.6),
        look_at=(0.0, 0.0, 0.05),
        data_types=["rgb", "depth"],
    ),
    "top": PinholeCameraCfg(
        name="top",
        width=256,
        height=256,
        pos=(0.0, 0.0, 1.2),
        look_at=(0.0, 0.0, 0.0),
        data_types=["rgb", "depth"],
    ),
}
"""Named views. Keys are what ``--camera`` accepts; each is a ``PinholeCameraCfg`` whose ``name`` is the key."""


def camera_preset(name: str) -> PinholeCameraCfg:
    """The preset called ``name``; unknown names list the presets."""
    try:
        return CAMERA_PRESETS[name]
    except KeyError:
        raise ValueError(f"unknown camera preset {name!r}; presets: {sorted(CAMERA_PRESETS)}") from None


def with_cameras(scenario, names: list[str] | tuple[str, ...]):
    """``scenario`` with the preset cameras ``names`` appended (a scenario camera of the same name is an error)."""
    if not names:
        return scenario
    existing = {c.name for c in getattr(scenario, "cameras", []) or []}
    added = []
    for name in names:
        cam = camera_preset(name)
        if cam.name in existing:
            raise ValueError(f"scenario already has a camera named {cam.name!r}")
        existing.add(cam.name)
        added.append(cam)
    return scenario.replace(cameras=[*(scenario.cameras or []), *added])
