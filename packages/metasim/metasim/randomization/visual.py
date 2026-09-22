"""Portable, replayable appearance augmentation for existing render scenes.

Sampling has no renderer dependency and uses independent streams for named targets.
A recipe is the unit of scheduling and replay: worker count and call order do not
change its parameters. World lighting is shared by all environments in a renderer.
The sensor stage is renderer-independent image processing applied after capture.
"""

from __future__ import annotations

import copy
import functools
import hashlib
import json
import math
import numbers
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

SCHEMA_VERSION = 3


def _finite(value):
    return isinstance(value, numbers.Real) and not isinstance(value, bool) and math.isfinite(value)


def _range(value, name, *, minimum=None, maximum=None):
    if not isinstance(value, list | tuple) or len(value) != 2 or not all(_finite(v) for v in value):
        raise ValueError(f"{name} must contain two finite numbers")
    if value[0] > value[1] or (minimum is not None and value[0] < minimum):
        raise ValueError(f"Invalid {name}: {value}")
    if maximum is not None and value[1] > maximum:
        raise ValueError(f"Invalid {name}: {value}")


def _vector(value, name):
    if not isinstance(value, list | tuple) or len(value) != 3 or not all(_finite(v) for v in value):
        raise ValueError(f"{name} must contain three finite numbers")


def _identifier(value, name):
    """Accept any integral scalar (numpy included) except bool; callers store ``int(value)``."""
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def _asset(path):
    if path is None:
        return None
    if not isinstance(path, str) or not path:
        raise ValueError("Asset paths must be nonempty strings")
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Visual randomization asset does not exist: {resolved}")
    return str(resolved)


def _mapping(value, keys, name):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError(f"{name} requires exactly these keys: {sorted(keys)}")


def _targets(value, name):
    if not isinstance(value, dict) or any(not isinstance(k, str) or not k for k in value):
        raise ValueError(f"{name} must map nonempty target names to parameters")


def _hdri(path):
    resolved = _asset(path)
    if resolved is not None and Path(resolved).suffix.lower() not in {".hdr", ".exr"}:
        raise ValueError("HDRI assets must be .hdr or .exr files")
    return resolved


def _texture_channels(textures):
    if textures.get("orm") and (textures.get("roughness") or textures.get("metallic")):
        raise ValueError("A packed orm map replaces separate roughness/metallic maps; supply one or the other")


def _stream_seed(seed, sample_id, variant_id, category, name=""):
    key = json.dumps([1, seed, sample_id, variant_id, category, name], separators=(",", ":"))
    return int.from_bytes(hashlib.blake2b(key.encode(), digest_size=16).digest(), "little")


def color_temperature_to_rgb(kelvin: float) -> tuple[float, float, float]:
    """Linear sRGB of a Planckian radiator, normalized so the brightest channel is 1.

    Uses the CIE 1931 chromaticity fit of the Planckian locus (Kim et al., 2002),
    valid from 1667 K to 25000 K; near 6500 K the result is close to neutral.
    """
    if not _finite(kelvin) or not 1667 <= kelvin <= 25000:
        raise ValueError(f"Color temperature must be within [1667, 25000] K, got {kelvin!r}")
    t = float(kelvin)
    if t <= 4000:
        x = -0.2661239e9 / t**3 - 0.2343589e6 / t**2 + 0.8776956e3 / t + 0.179910
    else:
        x = -3.0258469e9 / t**3 + 2.1070379e6 / t**2 + 0.2226347e3 / t + 0.240390
    if t <= 2222:
        y = -1.1063814 * x**3 - 1.34811020 * x**2 + 2.18555832 * x - 0.20219683
    elif t <= 4000:
        y = -0.9549476 * x**3 - 1.37418593 * x**2 + 2.09137015 * x - 0.16748867
    else:
        y = 3.0817580 * x**3 - 5.87338670 * x**2 + 3.75112997 * x - 0.37001483
    big_x, big_y, big_z = x / y, 1.0, (1 - x - y) / y
    rgb = (
        3.2406 * big_x - 1.5372 * big_y - 0.4986 * big_z,
        -0.9689 * big_x + 1.8758 * big_y + 0.0415 * big_z,
        0.0557 * big_x - 0.2040 * big_y + 1.0570 * big_z,
    )
    rgb = [max(v, 0.0) for v in rgb]
    peak = max(rgb)
    return tuple(v / peak for v in rgb)


def _srgb_to_linear(value):
    return np.where(value <= 0.04045, value / 12.92, ((value + 0.055) / 1.055) ** 2.4)


_SRGB_DECODE = _srgb_to_linear(np.arange(256, dtype=np.float64) / 255.0)


def _linear_to_srgb(value):
    value = np.clip(value, 0.0, 1.0)
    return np.where(value <= 0.0031308, value * 12.92, 1.055 * np.power(value, 1 / 2.4) - 0.055)


def _distort(x, y, coefficients):
    k1, k2, p1, p2 = coefficients
    r2 = x * x + y * y
    radial = 1 + k1 * r2 + k2 * r2 * r2
    return (
        x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x),
        y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y,
    )


def _undistort(xd, yd, coefficients, iterations=12):
    k1, k2, p1, p2 = coefficients
    x, y = xd.copy(), yd.copy()
    for _ in range(iterations):
        r2 = x * x + y * y
        radial = 1 + k1 * r2 + k2 * r2 * r2
        dx = 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        dy = p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        x = (xd - dx) / radial
        y = (yd - dy) / radial
    return x, y


def _sensor_geometry(matrix, height, width, coefficients):
    """Output intrinsics (zoomed so every distorted pixel samples inside the render) and source coordinates.

    The maps depend only on the calibration, which is constant across the frames and
    envs of a variant, so they are memoized; callers must not mutate the arrays.
    """
    key = np.ascontiguousarray(matrix, dtype=np.float64).tobytes()
    return _sensor_geometry_cached(key, int(height), int(width), tuple(float(c) for c in coefficients))


def _fit_geometry(matrix, height, width, coefficients):
    """Smallest output zoom whose distorted pixels all sample inside the render; unclipped maps.

    A pixel is admissible only where the inverse distortion converges (inside the lens'
    radial fold) and lands inside the render. The extremes of a smooth lens map lie on
    the image border, so the search evaluates the border ring and the full map once.
    """
    fx, fy, cx, cy = matrix[0, 0], matrix[1, 1], matrix[0, 2], matrix[1, 2]
    us, vs = np.meshgrid(np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64))
    border = np.zeros((height, width), dtype=bool)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True

    def sample(zoom, mask):
        xd, yd = (us[mask] - cx) / (fx * zoom), (vs[mask] - cy) / (fy * zoom)
        x, y = _undistort(xd, yd, coefficients)
        back_x, back_y = _distort(x, y, coefficients)
        converged = max(np.abs(back_x - xd).max(), np.abs(back_y - yd).max()) < 1e-7
        su, sv = fx * x + cx, fy * y + cy
        inside = (
            su.min() >= -1e-9 and su.max() <= width - 1 + 1e-9 and sv.min() >= -1e-9 and sv.max() <= height - 1 + 1e-9
        )
        return converged and inside, (su, sv, xd, yd)

    zoom = 1.0
    if not sample(1.0, border)[0]:
        low, high = 1.0, 2.0
        while not sample(high, border)[0]:
            low, high = high, high * 2
            if high > 1024:
                raise ValueError(f"distortion {tuple(coefficients)} leaves no usable image at this calibration")
        for _ in range(40):
            middle = 0.5 * (low + high)
            if sample(middle, border)[0]:
                high = middle
            else:
                low = middle
        zoom = high
    everywhere = np.ones_like(border)
    ok, maps = sample(zoom, everywhere)
    if not ok:
        raise ValueError(f"distortion {tuple(coefficients)} is not invertible over the whole image")
    return zoom, tuple(m.reshape(height, width) for m in maps)


@functools.lru_cache(maxsize=16)
def _sensor_geometry_cached(key, height, width, coefficients):
    matrix = np.frombuffer(key, dtype=np.float64).reshape(3, 3)
    zoom, (su, sv, xd, yd) = _fit_geometry(matrix, height, width, coefficients)
    output = matrix.copy()
    output[0, 0], output[1, 1] = matrix[0, 0] * zoom, matrix[1, 1] * zoom
    return output, np.clip(su, 0, width - 1), np.clip(sv, 0, height - 1), xd, yd


def _remap(image, su, sv):
    """Bilinear resampling at source coordinates (OpenCV; maps are clipped to the image)."""
    import cv2

    return cv2.remap(
        image, su.astype(np.float32), sv.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )


def _gaussian_blur(image, sigma):
    if sigma <= 0:
        return image
    radius = math.ceil(3 * sigma)
    if radius >= min(image.shape[:2]):
        raise ValueError(f"blur_sigma {sigma} is too large for a {image.shape[1]}x{image.shape[0]} image")
    taps = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (taps / sigma) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(image, ((radius, radius), (0, 0), (0, 0)), mode="reflect")
    rows = sum(kernel[i] * padded[i : i + image.shape[0]] for i in range(len(kernel)))
    padded = np.pad(rows, ((0, 0), (radius, radius), (0, 0)), mode="reflect")
    return sum(kernel[i] * padded[:, i : i + image.shape[1]] for i in range(len(kernel)))


@dataclass
class TextureSetCfg:
    """One coherent UV texture set; color is sRGB, scalar / normal maps are raw.

    Normal maps use the OpenGL (+Y) tangent-space convention. Scalar maps read the
    red channel; ``orm`` is a glTF-style packed map (R occlusion, G roughness,
    B metallic) that replaces separate roughness/metallic maps. Each selected set
    is applied as a unit.
    """

    base_color: str | None = None
    roughness: str | None = None
    metallic: str | None = None
    normal: str | None = None
    orm: str | None = None


@dataclass
class SurfaceRandomCfg:
    """PBR ranges for one named object, in linear RGB and the metalness workflow."""

    color: tuple[tuple[float, float], ...] = ((0.08, 0.8), (0.08, 0.8), (0.08, 0.8))
    roughness: tuple[float, float] = (0.25, 0.75)
    metallic: tuple[float, float] = (0.0, 0.0)
    textures: tuple[TextureSetCfg, ...] = ()
    color_palette: tuple[tuple[float, float, float], ...] = ()
    """Optional coherent linear-RGB colors, instead of independent channel draws."""
    uv_scale: tuple[tuple[float, float], tuple[float, float]] = ((1.0, 1.0), (1.0, 1.0))
    uv_rotation: tuple[float, float] = (0.0, 0.0)
    ior: tuple[float, float] = (1.5, 1.5)
    uv_projection: Literal["mesh", "box"] = "mesh"
    """``box`` authors metric per-face planar UVs (one tile per metre) on every renderer."""

    def __post_init__(self):
        """Validate physically bounded material parameters."""
        if len(self.color) != 3:
            raise ValueError("SurfaceRandomCfg.color requires three channel ranges")
        for name, value in [("roughness", self.roughness), ("metallic", self.metallic)]:
            _range(value, name, minimum=0, maximum=1)
        for value in self.color:
            _range(value, "color", minimum=0, maximum=1)
        for color in self.color_palette:
            _vector(color, "color_palette")
            for value in color:
                _range((value, value), "color_palette", minimum=0, maximum=1)
        if len(self.uv_scale) != 2:
            raise ValueError("uv_scale requires U and V ranges")
        for value in self.uv_scale:
            _range(value, "uv_scale", minimum=1e-6)
        _range(self.uv_rotation, "uv_rotation")
        _range(self.ior, "ior", minimum=1, maximum=3)
        if self.uv_projection not in ("mesh", "box"):
            raise ValueError(f"uv_projection must be 'mesh' or 'box', got {self.uv_projection!r}")


@dataclass
class LightingRandomCfg:
    """Multipliers relative to original light settings; positions use meters.

    Backend intensity units differ. A multiplier preserves a tuned lighting rig
    without pretending that Blender watts and USD light intensity are equivalent.
    """

    intensity_scale: tuple[float, float] = (0.7, 1.4)
    color: tuple[tuple[float, float], ...] = ((0.8, 1.0), (0.8, 1.0), (0.8, 1.0))
    position_delta: tuple[float, float, float] = (0.15, 0.15, 0.1)
    color_temperature: tuple[float, float] | None = None
    """Kelvin range; when set, the light color is the Planckian color instead of ``color``."""

    def __post_init__(self):
        """Validate nonnegative intensity, jitter and bounded colors."""
        _range(self.intensity_scale, "intensity_scale", minimum=0)
        _vector(self.position_delta, "position_delta")
        if any(v < 0 for v in self.position_delta) or len(self.color) != 3:
            raise ValueError("Light jitter must be nonnegative and color must have three channels")
        for value in self.color:
            _range(value, "light.color", minimum=0, maximum=1)
        if self.color_temperature is not None:
            _range(self.color_temperature, "color_temperature", minimum=1667, maximum=25000)


@dataclass
class EnvironmentRandomCfg:
    """Shared background and illumination, with optional equirectangular HDRIs.

    Strength is relative to an unscaled HDRI (Blender strength 1 / USD intensity
    500). It is a documented backend calibration, not photometric equivalence.
    """

    hdri_paths: tuple[str, ...] = ()
    strength: tuple[float, float] = (0.3, 0.8)
    rotation: tuple[float, float] = (-math.pi, math.pi)
    color: tuple[tuple[float, float], ...] = ((0.15, 0.5), (0.15, 0.5), (0.15, 0.5))

    def __post_init__(self):
        """Validate world lighting ranges."""
        _range(self.strength, "environment.strength", minimum=0)
        _range(self.rotation, "environment.rotation")
        if len(self.color) != 3:
            raise ValueError("Environment color requires three channel ranges")
        for value in self.color:
            _range(value, "environment.color", minimum=0, maximum=1)


@dataclass
class ViewRandomCfg:
    """Camera jitter relative to the original calibration; world-space meters.

    ``f_stop`` above zero enables thin-lens depth of field in the renderer; the
    focus distance is the jittered camera-to-look-at distance times ``focus_scale``.
    """

    position_delta: tuple[float, float, float] = (0.02, 0.02, 0.02)
    look_at_delta: tuple[float, float, float] = (0.01, 0.01, 0.01)
    focal_scale: tuple[float, float] = (0.95, 1.05)
    f_stop: tuple[float, float] = (0.0, 0.0)
    focus_scale: tuple[float, float] = (1.0, 1.0)

    def __post_init__(self):
        """Validate camera jitter and positive focal lengths."""
        for name in ("position_delta", "look_at_delta"):
            value = getattr(self, name)
            _vector(value, name)
            if any(v < 0 for v in value):
                raise ValueError(f"{name} must be nonnegative")
        _range(self.focal_scale, "focal_scale", minimum=1e-6)
        _range(self.f_stop, "f_stop", minimum=0)
        _range(self.focus_scale, "focus_scale", minimum=1e-6)


@dataclass
class SensorRandomCfg:
    """Image-space lens and sensor model applied identically to every renderer's capture.

    The capture is decoded from display-encoded sRGB to linear light, then exposure,
    white balance, OpenCV Brown-Conrady distortion (k1, k2, p1, p2), natural cos^4
    vignetting, Gaussian defocus and heteroscedastic shot/read noise are applied and
    the result is re-encoded. This approximates a camera pipeline on tone-mapped
    output; it is not a radiometric sensor simulation.
    """

    exposure_ev: tuple[float, float] = (-0.3, 0.3)
    auto_exposure: tuple[float, float] | None = None
    """Target mean linear luminance; when set, a gain (clamped to 1/16..16) meets it before ``exposure_ev``."""
    white_balance: tuple[tuple[float, float], tuple[float, float]] = ((0.92, 1.08), (0.92, 1.08))
    """Red and blue gains relative to green."""
    vignetting: tuple[float, float] = (0.0, 0.35)
    distortion: tuple[tuple[float, float], ...] = ((-0.08, 0.02), (-0.005, 0.005), (-0.001, 0.001), (-0.001, 0.001))
    """Ranges of k1, k2, p1, p2 in normalized image coordinates."""
    blur_sigma: tuple[float, float] = (0.0, 0.8)
    shot_noise: tuple[float, float] = (0.0, 0.001)
    read_noise: tuple[float, float] = (0.0, 0.0002)
    """Noise variance is ``shot_noise * value + read_noise`` in linear [0, 1] units."""

    def __post_init__(self):
        """Validate finite, physically bounded sensor ranges."""
        _range(self.exposure_ev, "exposure_ev")
        if len(self.white_balance) != 2 or len(self.distortion) != 4:
            raise ValueError("white_balance needs red/blue gain ranges and distortion needs k1, k2, p1, p2 ranges")
        for value in self.white_balance:
            _range(value, "white_balance", minimum=1e-6)
        for value in self.distortion:
            _range(value, "distortion")
        _range(self.vignetting, "vignetting", minimum=0, maximum=1)
        for name in ("blur_sigma", "shot_noise", "read_noise"):
            _range(getattr(self, name), name, minimum=0)
        if self.auto_exposure is not None:
            _range(self.auto_exposure, "auto_exposure", minimum=1e-4, maximum=1)


@dataclass
class SensorCapture:
    """A processed frame with the OpenCV pinhole calibration that describes it."""

    rgb: np.ndarray
    intrinsics: list[list[float]]
    distortion: list[float]
    """OpenCV order k1, k2, p1, p2, k3; k3 is always zero."""
    model: str = "opencv_pinhole"
    exposure_gain: float = 1.0
    """Linear gain applied by auto exposure (1 when disabled), before the sampled EV offset."""


@dataclass
class VisualRandomizationCfg:
    """Explicit named targets; omitted categories preserve existing appearance.

    Scene geometry / collision changes remain the responsibility of ScenarioCfg
    and SceneRandomizer. This configuration never changes physics state.
    """

    materials: dict[str, SurfaceRandomCfg] = field(default_factory=dict)
    lights: dict[str, LightingRandomCfg] = field(default_factory=dict)
    cameras: dict[str, ViewRandomCfg] = field(default_factory=dict)
    environment: EnvironmentRandomCfg | None = None
    sensors: dict[str, SensorRandomCfg] = field(default_factory=dict)
    """Per-camera image-space processing, keyed by camera name."""


@dataclass
class VisualRecipe:
    """Realized parameters in meters/radians.

    Schema v2 added UV transforms and dielectric IOR; v3 adds light color
    temperature, depth of field, packed ORM maps, box UV projection and the
    per-camera sensor stage. Older recipes remain readable with their original
    appearance defaults.
    """

    seed: int
    sample_id: int
    variant_id: int
    materials: dict = field(default_factory=dict)
    lights: dict = field(default_factory=dict)
    cameras: dict = field(default_factory=dict)
    environment: dict | None = None
    schema_version: int = SCHEMA_VERSION
    sensors: dict = field(default_factory=dict)

    def to_json(self) -> str:
        """Serialize without NaN or machine-dependent object representations."""
        return json.dumps(asdict(self), sort_keys=True, indent=2, allow_nan=False)

    @classmethod
    def from_json(cls, value: str) -> VisualRecipe:
        """Read a recipe and reject unknown versions and malformed values."""
        data = json.loads(value)
        if isinstance(data, dict) and data.get("schema_version") in (1, 2):
            data.setdefault("sensors", {})
        _mapping(data, cls.__dataclass_fields__, "recipe")
        recipe = cls(**data)
        recipe.validate()
        return recipe

    def validate(self):
        """Validate before renderer mutation, including referenced files."""
        if type(self.schema_version) is not int or self.schema_version not in (1, 2, 3):
            raise ValueError(f"Unsupported visual recipe schema: {self.schema_version}")
        version = self.schema_version
        for name in ("seed", "sample_id", "variant_id"):
            _identifier(getattr(self, name), name)
        for name in ("materials", "lights", "cameras", "sensors"):
            _targets(getattr(self, name), name)
        if version < 3 and self.sensors:
            raise ValueError("Sensor parameters require recipe schema version 3")
        for material in self.materials.values():
            keys = ("color", "roughness", "metallic", "textures")
            if version >= 2:
                keys += ("uv_scale", "uv_rotation", "ior")
            if version >= 3:
                keys += ("uv_projection",)
            _mapping(material, keys, "material")
            if version >= 2:
                scale = material["uv_scale"]
                if not isinstance(scale, list | tuple) or len(scale) != 2:
                    raise ValueError("uv_scale requires U and V values")
                for value in scale:
                    _range((value, value), "uv_scale", minimum=1e-6)
                _range((material["uv_rotation"],) * 2, "uv_rotation")
                _range((material["ior"],) * 2, "ior", minimum=1, maximum=3)
            if version >= 3 and material["uv_projection"] not in ("mesh", "box"):
                raise ValueError(f"uv_projection must be 'mesh' or 'box', got {material['uv_projection']!r}")
            _vector(material["color"], "material.color")
            for value in (*material["color"], material["roughness"], material["metallic"]):
                _range((value, value), "material", minimum=0, maximum=1)
            textures = material["textures"]
            texture_keys = tuple(TextureSetCfg.__dataclass_fields__)
            if version < 3:
                texture_keys = tuple(k for k in texture_keys if k != "orm")
            _mapping(textures, texture_keys, "textures")
            _texture_channels(textures)
            for path in textures.values():
                _asset(path)
        for light in self.lights.values():
            keys = ("intensity_scale", "position_delta", "color")
            if version >= 3:
                keys += ("color_temperature",)
            _mapping(light, keys, "light")
            _range((light["intensity_scale"],) * 2, "intensity_scale", minimum=0)
            _vector(light["position_delta"], "light.position_delta")
            _vector(light["color"], "light.color")
            for value in light["color"]:
                _range((value, value), "light.color", minimum=0, maximum=1)
            if version >= 3 and light["color_temperature"] is not None:
                _range((light["color_temperature"],) * 2, "color_temperature", minimum=1667, maximum=25000)
        for camera in self.cameras.values():
            keys = ("position_delta", "look_at_delta", "focal_scale")
            if version >= 3:
                keys += ("f_stop", "focus_scale")
            _mapping(camera, keys, "camera")
            _vector(camera["position_delta"], "camera.position_delta")
            _vector(camera["look_at_delta"], "camera.look_at_delta")
            _range((camera["focal_scale"],) * 2, "camera.focal_scale", minimum=1e-6)
            if version >= 3:
                _range((camera["f_stop"],) * 2, "camera.f_stop", minimum=0)
                _range((camera["focus_scale"],) * 2, "camera.focus_scale", minimum=1e-6)
        for sensor in self.sensors.values():
            keys = (
                "exposure_ev",
                "white_balance",
                "vignetting",
                "distortion",
                "blur_sigma",
                "shot_noise",
                "read_noise",
            )
            if not isinstance(sensor, dict):
                raise ValueError("sensor parameters must be a mapping")
            # auto_exposure joined schema 3 later; recipes written without it stay valid.
            _mapping({k: v for k, v in sensor.items() if k != "auto_exposure"}, keys, "sensor")
            if sensor.get("auto_exposure") is not None:
                _range((sensor["auto_exposure"],) * 2, "sensor.auto_exposure", minimum=1e-4, maximum=1)
            _range((sensor["exposure_ev"],) * 2, "sensor.exposure_ev")
            gains, coefficients = sensor["white_balance"], sensor["distortion"]
            if not isinstance(gains, list | tuple) or len(gains) != 2:
                raise ValueError("sensor.white_balance requires red and blue gains")
            for value in gains:
                _range((value, value), "sensor.white_balance", minimum=1e-6)
            if not isinstance(coefficients, list | tuple) or len(coefficients) != 4:
                raise ValueError("sensor.distortion requires k1, k2, p1, p2")
            for value in coefficients:
                _range((value, value), "sensor.distortion")
            _range((sensor["vignetting"],) * 2, "sensor.vignetting", minimum=0, maximum=1)
            for name in ("blur_sigma", "shot_noise", "read_noise"):
                _range((sensor[name],) * 2, f"sensor.{name}", minimum=0)
        if self.environment is not None:
            env = self.environment
            _mapping(env, ("hdri_path", "strength", "rotation", "color"), "environment")
            _hdri(env["hdri_path"])
            _range((env["strength"],) * 2, "environment.strength", minimum=0)
            _range((env["rotation"],) * 2, "environment.rotation")
            _vector(env["color"], "environment.color")
            for value in env["color"]:
                _range((value, value), "environment.color", minimum=0, maximum=1)

    def apply_sensor(self, camera: str, rgb, intrinsics, *, frame: int = 0) -> SensorCapture:
        """Process one captured frame with this recipe's sensor parameters for ``camera``.

        ``rgb`` is a display-encoded uint8 HxWx3 image and ``intrinsics`` the 3x3
        pinhole matrix the renderer captured with. Noise is seeded by recipe, camera
        and frame index so a replay reproduces the same image. The returned
        calibration describes the processed image in the OpenCV pinhole model.
        """
        self.validate()
        _identifier(frame, "frame")
        frame = int(frame)
        if camera not in self.sensors:
            raise ValueError(f"Recipe has no sensor parameters for camera {camera!r}")
        image = np.asarray(rgb.cpu() if hasattr(rgb, "cpu") else rgb)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 8:
            raise ValueError(f"Expected a uint8 HxWx3 image of at least 8x8 pixels, got {image.dtype} {image.shape}")
        matrix = np.asarray(intrinsics.cpu() if hasattr(intrinsics, "cpu") else intrinsics, dtype=np.float64)
        if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)) or matrix[0, 0] <= 0 or matrix[1, 1] <= 0:
            raise ValueError("intrinsics must be a finite 3x3 matrix with positive focal lengths")
        values = self.sensors[camera]
        height, width = image.shape[:2]
        if not (0 < matrix[0, 2] < width - 1 and 0 < matrix[1, 2] < height - 1):
            raise ValueError(f"principal point {matrix[0, 2], matrix[1, 2]} must lie inside the {width}x{height} image")
        linear = _SRGB_DECODE[image]
        gain = 1.0
        target = values.get("auto_exposure")
        if target is not None:
            luminance = linear @ np.array([0.2126, 0.7152, 0.0722])
            gain = float(np.clip(target / max(float(luminance.mean()), 1e-6), 1 / 16, 16))
        linear = linear * (gain * 2.0 ** values["exposure_ev"])
        linear[..., 0] *= values["white_balance"][0]
        linear[..., 2] *= values["white_balance"][1]
        output, su, sv, xd, yd = _sensor_geometry(matrix, height, width, values["distortion"])
        linear = _remap(linear, su, sv)
        falloff = 1.0 / (1.0 + xd * xd + yd * yd) ** 2
        linear *= (1.0 - values["vignetting"] * (1.0 - falloff))[..., None]
        linear = _gaussian_blur(linear, values["blur_sigma"])
        if values["shot_noise"] > 0 or values["read_noise"] > 0:
            rng = np.random.default_rng(
                _stream_seed(self.seed, self.sample_id, self.variant_id, "sensor", f"{camera}/{frame}")
            )
            deviation = np.sqrt(values["shot_noise"] * np.clip(linear, 0.0, None) + values["read_noise"])
            linear = linear + rng.standard_normal(linear.shape) * deviation
        processed = np.rint(_linear_to_srgb(linear) * 255.0).astype(np.uint8)
        return SensorCapture(
            rgb=processed,
            intrinsics=output.tolist(),
            distortion=[*(float(v) for v in values["distortion"]), 0.0],
            exposure_gain=gain,
        )


class VisualRandomizer:
    """Sample independently of execution order, then apply to a launched renderer.

    Bind once per handler lifetime. Reapplying a recipe uses original light and
    camera baselines, preventing cumulative drift. Sample once per episode for
    temporally consistent videos. One renderer process owns one world/HDRI.
    """

    def __init__(self, cfg: VisualRandomizationCfg, *, seed: int = 0):
        """Create an isolated sampler, validating all asset paths upfront."""
        _identifier(seed, "seed")
        if not isinstance(cfg, VisualRandomizationCfg):
            raise TypeError("cfg must be VisualRandomizationCfg")
        self.cfg = copy.deepcopy(cfg)
        self.seed = int(seed)
        self._adapter = None
        self._handler = None
        # Validate copies as config dataclasses can have been mutated after construction.
        for category, expected in (
            ("materials", SurfaceRandomCfg),
            ("lights", LightingRandomCfg),
            ("cameras", ViewRandomCfg),
            ("sensors", SensorRandomCfg),
        ):
            group = getattr(self.cfg, category)
            _targets(group, category)
            for name, value in group.items():
                if not isinstance(value, expected):
                    raise TypeError(f"{category}.{name} must be {expected.__name__}")
                if not isinstance(name, str) or not name:
                    raise ValueError("Visual targets must be nonempty object names")
                value.__post_init__()
        for material in self.cfg.materials.values():
            for textures in material.textures:
                if not isinstance(textures, TextureSetCfg):
                    raise TypeError("textures must contain TextureSetCfg instances")
                _texture_channels(asdict(textures))
                for key, path in asdict(textures).items():
                    setattr(textures, key, _asset(path))
        if self.cfg.environment is not None:
            if not isinstance(self.cfg.environment, EnvironmentRandomCfg):
                raise TypeError("environment must be EnvironmentRandomCfg")
            self.cfg.environment.__post_init__()
            if any(not isinstance(p, str) for p in self.cfg.environment.hdri_paths):
                raise ValueError("hdri_paths must contain file path strings")
            self.cfg.environment.hdri_paths = tuple(sorted(_hdri(p) for p in self.cfg.environment.hdri_paths))

    def sample(self, *, sample_id: int, variant_id: int = 0) -> VisualRecipe:
        """Sample stable named streams; adding one target never changes another."""
        _identifier(sample_id, "sample_id")
        _identifier(variant_id, "variant_id")
        sample_id, variant_id = int(sample_id), int(variant_id)

        def rng_for(category, name=""):
            return random.Random(_stream_seed(self.seed, sample_id, variant_id, category, name))

        recipe = VisualRecipe(self.seed, sample_id, variant_id)
        for name, cfg in sorted(self.cfg.materials.items()):
            rng = rng_for("material", name)
            recipe.materials[name] = {
                "color": [rng.uniform(*r) for r in cfg.color],
                "roughness": rng.uniform(*cfg.roughness),
                "metallic": rng.uniform(*cfg.metallic),
                "textures": asdict(rng.choice(cfg.textures)) if cfg.textures else asdict(TextureSetCfg()),
            }
            detail_rng = rng_for("material_detail", name)
            recipe.materials[name].update(
                uv_scale=[detail_rng.uniform(*r) for r in cfg.uv_scale],
                uv_rotation=detail_rng.uniform(*cfg.uv_rotation),
                ior=detail_rng.uniform(*cfg.ior),
                uv_projection=cfg.uv_projection,
            )
            if cfg.color_palette:
                recipe.materials[name]["color"] = list(rng_for("material_palette", name).choice(cfg.color_palette))
        for name, cfg in sorted(self.cfg.lights.items()):
            rng = rng_for("light", name)
            recipe.lights[name] = {
                "intensity_scale": rng.uniform(*cfg.intensity_scale),
                "color": [rng.uniform(*r) for r in cfg.color],
                "position_delta": [rng.uniform(-v, v) for v in cfg.position_delta],
                "color_temperature": None,
            }
            if cfg.color_temperature is not None:
                kelvin = rng_for("light_temperature", name).uniform(*cfg.color_temperature)
                recipe.lights[name].update(color_temperature=kelvin, color=list(color_temperature_to_rgb(kelvin)))
        for name, cfg in sorted(self.cfg.cameras.items()):
            rng = rng_for("camera", name)
            optics = rng_for("camera_optics", name)
            recipe.cameras[name] = {
                "position_delta": [rng.uniform(-v, v) for v in cfg.position_delta],
                "look_at_delta": [rng.uniform(-v, v) for v in cfg.look_at_delta],
                "focal_scale": rng.uniform(*cfg.focal_scale),
                "f_stop": optics.uniform(*cfg.f_stop),
                "focus_scale": optics.uniform(*cfg.focus_scale),
            }
        for name, cfg in sorted(self.cfg.sensors.items()):
            rng = rng_for("sensor", name)
            recipe.sensors[name] = {
                "exposure_ev": rng.uniform(*cfg.exposure_ev),
                "white_balance": [rng.uniform(*r) for r in cfg.white_balance],
                "vignetting": rng.uniform(*cfg.vignetting),
                "distortion": [rng.uniform(*r) for r in cfg.distortion],
                "blur_sigma": rng.uniform(*cfg.blur_sigma),
                "shot_noise": rng.uniform(*cfg.shot_noise),
                "read_noise": rng.uniform(*cfg.read_noise),
                "auto_exposure": rng.uniform(*cfg.auto_exposure) if cfg.auto_exposure is not None else None,
            }
        if self.cfg.environment is not None:
            cfg = self.cfg.environment
            rng = rng_for("environment")
            recipe.environment = {
                "hdri_path": rng.choice(cfg.hdri_paths) if cfg.hdri_paths else None,
                "strength": rng.uniform(*cfg.strength),
                "rotation": rng.uniform(*cfg.rotation),
                "color": [rng.uniform(*r) for r in cfg.color],
            }
        return recipe

    def bind_handler(self, handler, *, backend: Literal["blender", "isaacsim"] | None = None):
        """Bind a launched handler or hybrid; reject unsupported backends upfront."""
        from .core.visual_adapter import BlenderVisualAdapter, IsaacSimVisualAdapter

        renderer = getattr(handler, "render_handler", handler)
        backend = backend or renderer.scenario.simulator
        adapters = {"blender": BlenderVisualAdapter, "isaacsim": IsaacSimVisualAdapter}
        if backend not in adapters:
            raise NotImplementedError(f"VisualRandomizer supports blender and isaacsim; got {backend!r}")
        if self._adapter is not None:
            raise RuntimeError("Bind once per VisualRandomizer; create another sampler for another handler")
        self._adapter = adapters[backend](renderer)
        self._handler = handler
        return self

    def apply(self, recipe: VisualRecipe, *, render: bool = True, env_ids: list[int] | None = None):
        """Apply realized values; errors propagate and no physics step is taken.

        Set render=False before replaying the next physics state to avoid an
        extra frame. Adapters invalidate cached images even when rendering is deferred.
        Sensor parameters are not applied here; process captures with
        ``recipe.apply_sensor``.
        """
        if self._adapter is None:
            raise RuntimeError("Call bind_handler() with a launched renderer before apply()")
        recipe.validate()
        count = self._adapter.handler.num_envs
        ids = list(range(count)) if env_ids is None else env_ids
        if not isinstance(ids, list) or any(
            isinstance(i, bool) or not isinstance(i, numbers.Integral) or not 0 <= i < count for i in ids
        ):
            raise ValueError(f"env_ids must be a list of integer indices in [0, {count})")
        ids = [int(i) for i in ids]
        if len(set(ids)) != len(ids):
            raise ValueError("env_ids must not contain duplicates")
        if not ids:
            return
        if (recipe.lights or recipe.environment is not None) and set(ids) != set(range(count)):
            raise ValueError("Lights and environment are shared; apply them to all environments")
        self._adapter.validate(recipe, ids)
        try:
            self._adapter.apply(recipe, ids)
        finally:
            # Even partial edits on a backend error must never leave stale public state.
            self._adapter.invalidate()
            self._handler._invalidate_state_caches()
        if render:
            self._adapter.render()
