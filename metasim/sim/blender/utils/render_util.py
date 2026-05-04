"""Compositor / render-pass plumbing for the Blender backend.

Targets Blender 5.x ``bpy``. Three things differ from 4.x and shape this module:

* ``Scene.use_nodes`` / ``Scene.node_tree`` are gone — the compositor lives
  on a separate ``Scene.compositing_node_group`` NodeTree datablock.
* ``CompositorNodeOutputFile.format.file_format`` only accepts
  ``OPEN_EXR_MULTILAYER``; PNG/EXR are rejected.
* In 5.0 a single output node only writes its first slot, so we use one
  output node per pass and read each layer back via OpenEXR.
"""

from __future__ import annotations

import os

import bpy
import numpy as np
import OpenEXR

# data_type -> (RLayers output socket, file_output_items socket_type, file prefix)
PASS_OUTPUT_SPECS: dict[str, tuple[str, str, str]] = {
    "rgb": ("Image", "RGBA", "rgb"),
    "depth": ("Depth", "FLOAT", "depth"),
    "normal": ("Normal", "VECTOR", "normal"),
    "instance_seg": ("Object Index", "FLOAT", "instance_seg"),
}


def get_view_layer(scene: bpy.types.Scene) -> bpy.types.ViewLayer:
    """Return the active view layer; tolerant to default-name differences."""
    for vl in scene.view_layers:
        if vl.name in ("ViewLayer", "View Layer"):
            return vl
    return scene.view_layers[0]


def enable_passes(scene: bpy.types.Scene, data_types: set[str]) -> None:
    """Enable the render-layer passes required by ``data_types``."""
    vl = get_view_layer(scene)
    vl.use_pass_combined = True
    vl.use_pass_z = "depth" in data_types
    vl.use_pass_normal = "normal" in data_types
    vl.use_pass_object_index = "instance_seg" in data_types


def get_compositor_tree(scene: bpy.types.Scene) -> bpy.types.NodeTree:
    """Get (or create) the scene's compositor node tree (Blender 5.x layout)."""
    tree = scene.compositing_node_group
    if tree is None:
        tree = bpy.data.node_groups.new("Compositor", "CompositorNodeTree")
        scene.compositing_node_group = tree
    return tree


def configure_compositor(
    scene: bpy.types.Scene,
    data_types: set[str],
    output_dir: str,
) -> dict[str, str]:
    """Wire one OPEN_EXR_MULTILAYER ``CompositorNodeOutputFile`` per requested pass.

    Returns a mapping ``data_type -> file_name prefix``. Blender appends the
    current frame number and ``.exr``, so callers should glob ``<prefix>*.exr``.
    """
    tree = get_compositor_tree(scene)
    for node in list(tree.nodes):
        tree.nodes.remove(node)

    rl = tree.nodes.new("CompositorNodeRLayers")
    prefixes: dict[str, str] = {}

    for dtype in ("rgb", "depth", "normal", "instance_seg"):
        if dtype not in data_types:
            continue
        src_socket, sock_type, prefix = PASS_OUTPUT_SPECS[dtype]
        if src_socket not in rl.outputs:
            continue
        out = tree.nodes.new("CompositorNodeOutputFile")
        out.directory = output_dir
        out.file_name = prefix + "_"
        out.format.file_format = "OPEN_EXR_MULTILAYER"
        out.file_output_items.new(sock_type, dtype)
        in_sock = next(i for i in out.inputs if i.name == dtype)
        tree.links.new(rl.outputs[src_socket], in_sock)
        prefixes[dtype] = out.file_name

    return prefixes


def find_rendered_file(output_dir: str, prefix: str) -> str:
    """Locate the .exr Blender wrote for a CompositorNodeOutputFile."""
    candidates = sorted(f for f in os.listdir(output_dir) if f.startswith(prefix) and f.endswith(".exr"))
    if not candidates:
        raise FileNotFoundError(f"No .exr with prefix {prefix!r} in {output_dir}: {os.listdir(output_dir)}")
    return os.path.join(output_dir, candidates[-1])


_CHANNEL_SUFFIX_GROUPS = (("R", "G", "B", "A"), ("R", "G", "B"), ("X", "Y", "Z"), ("V",))


def read_exr_layer(path: str, layer: str) -> np.ndarray:
    """Read a single multilayer-EXR layer and return an HxW(xC) float32 array.

    Blender writes channels as:
      RGBA   -> ``<layer>.R``, ``<layer>.G``, ``<layer>.B``, ``<layer>.A``
      VECTOR -> ``<layer>.X``, ``<layer>.Y``, ``<layer>.Z``
      FLOAT  -> ``<layer>.V``
    """
    f = OpenEXR.InputFile(path)
    header = f.header()
    dw = header["dataWindow"]
    w = dw.max.x - dw.min.x + 1
    h = dw.max.y - dw.min.y + 1
    chan_names = list(header["channels"].keys())

    chosen: list[str] = []
    for suffixes in _CHANNEL_SUFFIX_GROUPS:
        candidate = [f"{layer}.{s}" for s in suffixes]
        if all(c in chan_names for c in candidate):
            chosen = candidate
            break
    if not chosen:
        raise ValueError(f"no recognized channel set for layer {layer!r} in {chan_names}")

    planes = [np.frombuffer(f.channel(c), dtype=np.float32).reshape(h, w) for c in chosen]
    return planes[0] if len(planes) == 1 else np.stack(planes, axis=-1)


def configure_cycles(
    scene: bpy.types.Scene,
    samples: int = 64,
    adaptive_threshold: float = 0.01,
    min_samples: int | None = None,
    max_bounces: int = 8,
    diffuse_bounces: int = 4,
    glossy_bounces: int = 4,
    transmission_bounces: int = 12,
    transparent_bounces: int = 8,
    use_denoising: bool = True,
) -> None:
    """Apply sane Cycles defaults and switch to GPU when possible.

    Tries OPTIX → CUDA → HIP → METAL → ONEAPI in order; the first backend
    that exposes a non-CPU device wins. Logs which backend ended up active
    so users can confirm GPU is being used.

    Quality knobs (used when ``samples`` is bumped above the default) keep
    Cycles from terminating too early on noisy pixels and let it bounce
    enough rays for HDRI-driven scenes to look right.
    """
    from loguru import logger as _log

    scene.render.engine = "CYCLES"
    cycles = scene.cycles
    cycles.samples = int(samples)

    # Adaptive sampling — keep tracing pixels that haven't converged yet.
    cycles.use_adaptive_sampling = True
    cycles.adaptive_threshold = float(adaptive_threshold)
    cycles.adaptive_min_samples = int(min_samples or max(8, samples // 16))

    # Light bounce limits matter under HDRI lighting (esp. metal / glass).
    cycles.max_bounces = int(max_bounces)
    cycles.diffuse_bounces = int(diffuse_bounces)
    cycles.glossy_bounces = int(glossy_bounces)
    cycles.transmission_bounces = int(transmission_bounces)
    cycles.transparent_max_bounces = int(transparent_bounces)

    # Denoiser: prefer Open Image Denoise; fall back gracefully if unavailable.
    cycles.use_denoising = bool(use_denoising)
    if use_denoising:
        try:
            cycles.denoiser = "OPENIMAGEDENOISE"
        except TypeError:
            pass
    prefs = bpy.context.preferences.addons.get("cycles")
    if prefs is None:
        scene.cycles.device = "CPU"
        _log.info("cycles: CPU (cycles addon not loaded)")
        return
    cprefs = prefs.preferences
    original_backend = getattr(cprefs, "compute_device_type", None)
    for backend in ("OPTIX", "CUDA", "HIP", "METAL", "ONEAPI"):
        try:
            cprefs.compute_device_type = backend
        except TypeError:
            continue
        devices = cprefs.get_devices_for_type(backend) if hasattr(cprefs, "get_devices_for_type") else []
        gpu_devs = [d for d in devices if d.type == backend]
        if gpu_devs:
            for d in devices:
                d.use = d.type == backend
            scene.cycles.device = "GPU"
            names = ", ".join(d.name for d in gpu_devs)
            _log.info(f"cycles: GPU via {backend} ({names})")
            return
    if original_backend is not None:
        try:
            cprefs.compute_device_type = original_backend
        except TypeError:
            pass
    scene.cycles.device = "CPU"
    _log.info("cycles: CPU (no GPU backend with devices found)")


def assign_pass_indices(objects: list[bpy.types.Object]) -> dict[int, str]:
    """Assign a unique ``pass_index`` to each object, returning the id->name map.

    ``pass_index`` of 0 is reserved for background / unindexed pixels.
    """
    id2name: dict[int, str] = {}
    for i, obj in enumerate(objects, start=1):
        obj.pass_index = i
        id2name[i] = obj.name
    return id2name
