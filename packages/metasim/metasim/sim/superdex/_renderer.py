"""Offscreen camera rendering for the SuperDex backend.

SuperDex ships no headless image path (its viewers are a desktop debugger, a polyscope window and
an Unreal client), so cameras are rendered here with ``pyrender`` from the same link transforms the
physics engine reports. This gives RGB **and** metric depth, which is what the rest of MetaSim
expects from a ``CameraState``; instance segmentation is not implemented and is rejected loudly at
launch instead of returning an empty tensor.

The renderer is optional: it is only constructed when the scenario declares cameras, and it raises
an actionable ``ImportError`` if ``pyrender`` is missing.
"""

from __future__ import annotations

import os

import numpy as np
from loguru import logger as log

# pyrender's EGL platform needs to be selected before it (PyOpenGL) is first imported. Respect a
# user choice; default to EGL, which is what every other headless renderer in MetaSim assumes.
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

try:
    import pyrender
except ImportError as _exc:  # pragma: no cover - reported with an install hint by the handler
    pyrender = None
    _PYRENDER_IMPORT_ERROR = _exc
else:
    _PYRENDER_IMPORT_ERROR = None


def look_at_pose(eye, target, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """Camera-to-world 4x4 for a camera at ``eye`` looking at ``target`` (OpenGL: looks down -Z, +Y up)."""
    eye = np.asarray(eye, dtype=np.float64)
    forward = np.asarray(target, dtype=np.float64) - eye
    norm = np.linalg.norm(forward)
    if norm < 1e-9:
        raise ValueError("camera pos and look_at coincide")
    forward /= norm
    up = np.asarray(up, dtype=np.float64)
    if abs(np.dot(forward, up / np.linalg.norm(up))) > 0.999:  # looking straight up/down
        up = np.array([0.0, 1.0, 0.0])
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    true_up = np.cross(right, forward)
    pose = np.eye(4)
    pose[:3, 0], pose[:3, 1], pose[:3, 2], pose[:3, 3] = right, true_up, -forward, eye
    return pose


def _has_texture(mesh) -> bool:
    """True when the mesh carries an image texture pyrender can upload (UVs + material image)."""
    import trimesh

    vis = getattr(mesh, "visual", None)
    if not isinstance(vis, trimesh.visual.texture.TextureVisuals) or getattr(vis, "uv", None) is None:
        return False
    mat = getattr(vis, "material", None)
    image = getattr(mat, "image", None) or getattr(mat, "baseColorTexture", None)
    return image is not None and len(vis.uv) == len(mesh.vertices)


def _visual_color(mesh):
    """A flat RGBA (0..1) from a mesh's own material / vertex colours, if it has any."""
    import trimesh

    vis = getattr(mesh, "visual", None)
    try:
        if isinstance(vis, trimesh.visual.texture.TextureVisuals):
            mat = vis.material
            base = getattr(mat, "baseColorFactor", None)
            if base is None:
                base = getattr(mat, "diffuse", None)
            if base is not None:
                rgba = np.asarray(base, dtype=np.float64).reshape(-1)[:4]
                if rgba.max() > 1.0:
                    rgba = rgba / 255.0
                return tuple(float(v) for v in (list(rgba) + [1.0])[:4])
        elif isinstance(vis, trimesh.visual.ColorVisuals) and vis.kind is not None:
            rgba = np.asarray(vis.main_color, dtype=np.float64) / 255.0
            return tuple(float(v) for v in rgba[:4])
    except Exception:  # a broken material must not break the render
        return None
    return None


class OffscreenRenderer:
    """Keeps a ``pyrender.Scene`` in sync with poses the handler pushes in, and renders cameras."""

    def __init__(self) -> None:
        if pyrender is None:
            raise ImportError(
                "SuperDex cameras are rendered with pyrender, which is not installed: "
                "python -m pip install pyrender  (headless rendering also needs EGL, PYOPENGL_PLATFORM=egl)"
            ) from _PYRENDER_IMPORT_ERROR
        self._scene = pyrender.Scene(bg_color=[0.35, 0.35, 0.38, 1.0], ambient_light=[0.35, 0.35, 0.35])
        self._nodes: dict[tuple[str, str], list[tuple[object, np.ndarray]]] = {}
        self._renderers: dict[tuple[int, int], object] = {}
        self._scene.add(
            pyrender.DirectionalLight(color=np.ones(3), intensity=3.0), pose=look_at_pose((1, 1, 3), (0, 0, 0))
        )
        self._scene.add(
            pyrender.DirectionalLight(color=np.ones(3), intensity=1.5), pose=look_at_pose((-2, -1, 2), (0, 0, 0))
        )

    # ------------------------------------------------------------------ scene construction
    def add_ground(self, size: float = 20.0, tile: float = 0.5) -> None:
        """A checkerboard floor at z=0 so renders have a horizon and depth has a floor."""
        import trimesh

        n = int(size / tile)
        tiles = []
        for i in range(n):
            for j in range(n):
                quad = trimesh.creation.box(extents=(tile, tile, 0.002))
                quad.apply_translation(((i - n / 2 + 0.5) * tile, (j - n / 2 + 0.5) * tile, -0.001))
                shade = [150, 150, 158, 255] if (i + j) % 2 else [205, 205, 212, 255]
                quad.visual = trimesh.visual.ColorVisuals(quad, face_colors=np.tile(shade, (len(quad.faces), 1)))
                tiles.append(quad)
        floor = trimesh.util.concatenate(tiles)
        self.add_body("__ground__", "ground", [(floor, np.eye(4), None)])

    def add_body(self, obj_name: str, body_name: str, geoms) -> None:
        """Register the visual geometry of one body: ``geoms`` = iterable of (trimesh, body_from_geom, rgba|None)."""
        import trimesh

        nodes = []
        for mesh, body_from_geom, color in geoms:
            mesh = mesh.copy()
            pose = np.asarray(body_from_geom, dtype=np.float64)
            node = None
            if _has_texture(mesh):
                # A textured visual (OBJ+MTL/PNG, DAE, GLB) keeps its texture: the image travels in
                # ``visual.material`` and pyrender uploads it. A URDF ``<material>`` colour does not
                # override a real texture (it is usually the exporter's default grey).
                try:
                    node = self._scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False), pose=pose)
                except Exception as exc:  # texture upload failed: fall back to a flat colour
                    log.warning(
                        f"[superdex] texture of {obj_name}/{body_name} not uploaded ({exc}); rendering untextured"
                    )
                    node = None
            if node is None:
                if color is None:
                    color = _visual_color(mesh) or (0.75, 0.75, 0.78, 1.0)
                rgba = (np.clip(np.asarray(color, dtype=np.float64), 0, 1) * 255).astype(np.uint8)
                if len(rgba) == 3:
                    rgba = np.append(rgba, 255)
                mesh.visual = trimesh.visual.ColorVisuals(mesh, face_colors=np.tile(rgba, (len(mesh.faces), 1)))
                node = self._scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False), pose=pose)
            nodes.append((node, pose))
        self._nodes[(obj_name, body_name)] = nodes

    # ------------------------------------------------------------------ per-frame updates
    def set_body_pose(self, obj_name: str, body_name: str, world_from_body: np.ndarray) -> None:
        """Move every geometry of a body to ``world_from_body`` (4x4)."""
        for node, body_from_geom in self._nodes.get((obj_name, body_name), ()):
            self._scene.set_pose(node, world_from_body @ body_from_geom)

    def render(self, camera_cfg, pose: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Render one ``PinholeCameraCfg``: returns (rgb uint8 HxWx3, depth float32 HxW in metres, 0 = no hit).

        ``pose`` overrides the camera-to-world transform (mounted cameras); by default it is derived from
        ``pos``/``look_at``.
        """
        width, height = int(camera_cfg.width), int(camera_cfg.height)
        near, far = camera_cfg.clipping_range
        cam = pyrender.PerspectiveCamera(
            yfov=float(np.deg2rad(camera_cfg.vertical_fov)),
            aspectRatio=width / height,
            znear=float(near),
            zfar=float(far),
        )
        if pose is None:
            pose = look_at_pose(camera_cfg.pos, camera_cfg.look_at)
        node = self._scene.add(cam, pose=np.asarray(pose, dtype=np.float64))
        try:
            renderer = self._renderers.get((width, height))
            if renderer is None:
                renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)
                self._renderers[(width, height)] = renderer
            rgb, depth = renderer.render(self._scene)
        finally:
            self._scene.remove_node(node)
        return np.ascontiguousarray(rgb[..., :3]), np.ascontiguousarray(depth, dtype=np.float32)

    def close(self) -> None:
        for renderer in self._renderers.values():
            try:
                renderer.delete()
            except Exception as exc:  # best effort GL teardown
                log.debug(f"[superdex] renderer teardown: {exc}")
        self._renderers.clear()
