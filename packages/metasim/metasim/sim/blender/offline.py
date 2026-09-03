from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import imageio.v3 as iio

from metasim.scenario.scenario import ScenarioCfg
from metasim.types import TensorState

from .blender import BlenderHandler


@dataclass(frozen=True)
class BlenderOfflineRenderCfg:
    output_dir: str | Path
    samples: int | None = None
    device: str = "AUTO"
    cameras: tuple[str, ...] | None = None


def _render_scenario_from_source(scenario: ScenarioCfg, cfg: BlenderOfflineRenderCfg) -> ScenarioCfg:
    render_scenario = copy.deepcopy(scenario)
    render_scenario.simulator = "blender"
    render_scenario.headless = True
    if cfg.cameras is not None:
        wanted = set(cfg.cameras)
        render_scenario.cameras = [camera for camera in render_scenario.cameras if camera.name in wanted]
        missing = sorted(wanted - {camera.name for camera in render_scenario.cameras})
        if missing:
            raise ValueError(f"Unknown camera(s) for Blender offline render: {missing}")
    if cfg.samples is not None:
        render_scenario.render.samples = int(cfg.samples)
    render_scenario.render.device = cfg.device
    return render_scenario


def _write_camera_frames(output_dir: Path, frame_index: int, rendered: TensorState) -> list[Path]:
    written: list[Path] = []
    for camera_name, camera_state in rendered.cameras.items():
        if camera_state.rgb is None:
            continue
        camera_dir = output_dir / camera_name
        camera_dir.mkdir(parents=True, exist_ok=True)
        path = camera_dir / f"frame_{frame_index:06d}.png"
        iio.imwrite(path, camera_state.rgb[0].detach().cpu().numpy())
        written.append(path)
    return written


def render_state_sequence(
    scenario: ScenarioCfg,
    states: Iterable[TensorState],
    cfg: BlenderOfflineRenderCfg,
) -> list[Path]:
    """Render exact one-env TensorState frames in Blender.

    Blender does not run physics or articulation kinematics here. Robot and
    articulated-object poses must already be expressed as body_state transforms.
    """
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    handler = BlenderHandler(_render_scenario_from_source(scenario, cfg))
    written: list[Path] = []
    try:
        handler.launch()
        for frame_index, state in enumerate(states):
            handler.set_states(state)
            rendered = handler.get_states(mode="tensor")
            written.extend(_write_camera_frames(output_dir, frame_index, rendered))
    finally:
        handler.close()
    return written
