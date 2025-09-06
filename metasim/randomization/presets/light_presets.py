"""Light randomization presets for common lighting scenarios."""

from __future__ import annotations

from ..light_randomizer import (
    LightColorRandomCfg,
    LightIntensityRandomCfg,
    LightOrientationRandomCfg,
    LightPositionRandomCfg,
    LightRandomCfg,
)


def kelvin_to_rgb(kelvin: float) -> tuple[float, float, float]:
    """Convert color temperature in Kelvin to RGB values (0-1 range)."""
    # Simplified conversion algorithm
    temp = kelvin / 100

    if temp <= 66:
        red = 1.0
        green = min(1.0, max(0.0, (99.4708025861 * (temp**0.1981) - 161.1195681661) / 255))
        blue = (
            0.0 if temp < 19 else min(1.0, max(0.0, (138.5177312231 * ((temp - 10) ** 0.1981) - 305.0447927307) / 255))
        )
    else:
        red = min(1.0, max(0.0, (329.698727446 * ((temp - 60) ** -0.1332047592)) / 255))
        green = min(1.0, max(0.0, (288.1221695283 * ((temp - 60) ** -0.0755148492)) / 255))
        blue = 1.0

    return (red, green, blue)


class LightProperties:
    """Common light property ranges for different scenarios."""

    # Intensity ranges
    INTENSITY_DIM = (50.0, 200.0)
    INTENSITY_NORMAL = (200.0, 800.0)
    INTENSITY_BRIGHT = (800.0, 2000.0)
    INTENSITY_VERY_BRIGHT = (2000.0, 5000.0)

    # Color ranges (RGB 0-1)
    COLOR_WARM = ((0.6, 1.0), (0.3, 0.8), (0.0, 0.5))
    COLOR_COOL = ((0.2, 0.8), (0.4, 1.0), (0.7, 1.0))
    COLOR_NATURAL = ((0.6, 1.0), (0.6, 1.0), (0.5, 1.0))
    COLOR_FULL = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))

    # Color temperature ranges (in Kelvin)
    TEMP_WARM_WHITE = (2700.0, 3500.0)
    TEMP_COOL_WHITE = (4000.0, 5000.0)
    TEMP_DAYLIGHT = (5500.0, 6500.0)

    # Position ranges (in meters)
    POSITION_SMALL = ((-2.0, 2.0), (-2.0, 2.0), (-1.0, 1.0))
    POSITION_MEDIUM = ((-5.0, 5.0), (-5.0, 5.0), (-3.0, 3.0))
    POSITION_LARGE = ((-8.0, 8.0), (-8.0, 8.0), (-4.0, 4.0))

    # Orientation ranges (in degrees)
    ORIENTATION_SMALL = ((-15.0, 15.0), (-15.0, 15.0), (-15.0, 15.0))
    ORIENTATION_MEDIUM = ((-45.0, 45.0), (-45.0, 45.0), (-45.0, 45.0))
    ORIENTATION_LARGE = ((-90.0, 90.0), (-90.0, 90.0), (-180.0, 180.0))


class LightPresets:
    """Pre-configured light randomization setups for common scenarios."""

    @staticmethod
    def indoor_ambient(light_name: str, randomization_mode: str = "combined") -> LightRandomCfg:
        """Create indoor ambient lighting configuration."""
        return LightRandomCfg(
            light_name=light_name,
            intensity=LightIntensityRandomCfg(
                intensity_range=LightProperties.INTENSITY_BRIGHT, distribution="uniform", enabled=True
            ),
            color=LightColorRandomCfg(color_range=LightProperties.COLOR_WARM, distribution="uniform", enabled=True),
            position=LightPositionRandomCfg(
                position_range=LightProperties.POSITION_MEDIUM,
                relative_to_origin=True,
                distribution="uniform",
                enabled=True,
            ),
            randomization_mode=randomization_mode,
        )

    @staticmethod
    def outdoor_daylight(light_name: str, randomization_mode: str = "combined") -> LightRandomCfg:
        """Create outdoor daylight configuration."""
        return LightRandomCfg(
            light_name=light_name,
            intensity=LightIntensityRandomCfg(
                intensity_range=LightProperties.INTENSITY_VERY_BRIGHT, distribution="uniform", enabled=True
            ),
            color=LightColorRandomCfg(color_range=LightProperties.COLOR_NATURAL, distribution="uniform", enabled=True),
            orientation=LightOrientationRandomCfg(
                angle_range=LightProperties.ORIENTATION_LARGE,
                relative_to_origin=True,
                distribution="uniform",
                enabled=True,
            ),
            randomization_mode=randomization_mode,
        )

    @staticmethod
    def demo_colors(light_name: str, randomization_mode: str = "color_only") -> LightRandomCfg:
        """Demo color randomization for visual feedback."""
        return LightRandomCfg(
            light_name=light_name,
            intensity=LightIntensityRandomCfg(
                intensity_range=(8000.0, 8000.0),  # Fixed bright intensity
                distribution="uniform",
                enabled=True,
            ),
            color=LightColorRandomCfg(color_range=LightProperties.COLOR_FULL, distribution="uniform", enabled=True),
            randomization_mode=randomization_mode,
        )

    @staticmethod
    def demo_positions(light_name: str, randomization_mode: str = "position_only") -> LightRandomCfg:
        """Demo position randomization for shadow feedback."""
        return LightRandomCfg(
            light_name=light_name,
            intensity=LightIntensityRandomCfg(
                intensity_range=(20000.0, 20000.0),  # Fixed ultra-bright intensity
                distribution="uniform",
                enabled=True,
            ),
            color=LightColorRandomCfg(
                color_range=((1.0, 1.0), (1.0, 1.0), (1.0, 1.0)),  # Fixed white
                distribution="uniform",
                enabled=True,
            ),
            position=LightPositionRandomCfg(
                position_range=((-5.0, 5.0), (-5.0, 5.0), (1.5, 4.0)),  # Absolute positions
                relative_to_origin=False,
                distribution="uniform",
                enabled=True,
            ),
            randomization_mode=randomization_mode,
        )


class LightScenarios:
    """Pre-configured multi-light scenarios for complex lighting setups."""

    @staticmethod
    def indoor_room() -> list[LightRandomCfg]:
        """Three-point indoor lighting setup."""
        return [
            LightPresets.indoor_ambient("ceiling_light", "combined"),
            LightPresets.indoor_ambient("window_light", "combined"),
            LightPresets.indoor_ambient("desk_lamp", "combined"),
        ]

    @staticmethod
    def outdoor_scene() -> list[LightRandomCfg]:
        """Outdoor daylight setup with sun and sky lighting."""
        return [
            LightPresets.outdoor_daylight("sun_light", "combined"),
            LightPresets.outdoor_daylight("sky_light", "combined"),
        ]

    @staticmethod
    def three_point_studio() -> list[LightRandomCfg]:
        """Professional three-point studio lighting setup."""
        return [
            LightPresets.outdoor_daylight("key_light", "combined"),
            LightPresets.indoor_ambient("fill_light", "combined"),
            LightPresets.indoor_ambient("rim_light", "combined"),
        ]
