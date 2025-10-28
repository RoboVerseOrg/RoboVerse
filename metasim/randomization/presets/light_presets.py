"""Light randomization presets for different light types and scenarios.

Light Types:

1. DistantLight (directional, parallel rays):
   - No position (infinite distance)
   - Randomize: intensity, color, orientation

2. DomeLight (environment/sky light):
   - No position (wraps entire scene)
   - Randomize: intensity, color

3. SphereLight/CylinderLight/DiskLight (local sources):
   - Has position
   - Randomize: position, intensity, color, orientation (Disk/Cylinder only)

Usage Guidelines:
- Open scenes: Any light type
- Enclosed rooms: Use local lights (Sphere/Disk/Cylinder) positioned inside
"""

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


class LightIntensityRanges:
    """Common intensity ranges for different scenarios (in candela or nits)."""

    # General ranges
    DIM = (50.0, 200.0)
    NORMAL = (200.0, 800.0)
    BRIGHT = (800.0, 2000.0)
    VERY_BRIGHT = (2000.0, 5000.0)
    ULTRA_BRIGHT = (5000.0, 15000.0)

    # Scene-specific ranges
    OUTDOOR_SUN = (10000.0, 50000.0)  # Direct sunlight
    INDOOR_AMBIENT = (500.0, 2000.0)  # Indoor ceiling lights
    INDOOR_TASK = (1000.0, 3000.0)  # Desk lamps, reading lights
    STUDIO = (2000.0, 8000.0)  # Studio lighting

    # Enclosed room ranges (need higher intensity due to wall absorption)
    ENCLOSED_ROOM_DISTANT = (5000.0, 15000.0)  # DistantLight in enclosed room
    ENCLOSED_ROOM_SPHERE = (3000.0, 10000.0)  # SphereLight with is_global=True


class LightColorRanges:
    """Common color ranges for different lighting scenarios (RGB 0-1)."""

    # RGB-based
    WARM = ((0.9, 1.0), (0.7, 0.9), (0.4, 0.7))  # Warm indoor lighting
    COOL = ((0.7, 0.9), (0.8, 1.0), (0.9, 1.0))  # Cool/clinical lighting
    NATURAL = ((0.95, 1.0), (0.95, 1.0), (0.9, 1.0))  # Natural daylight
    FULL_SPECTRUM = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))  # Full color range (for demos)

    # Temperature-based (in Kelvin)
    CANDLE = (1500.0, 2000.0)  # Candle light
    WARM_WHITE = (2700.0, 3500.0)  # Incandescent bulbs
    NEUTRAL_WHITE = (3500.0, 4500.0)  # Fluorescent lights
    COOL_WHITE = (4500.0, 5500.0)  # Cool office lighting
    DAYLIGHT = (5500.0, 6500.0)  # Natural daylight
    OVERCAST_SKY = (6500.0, 8000.0)  # Cloudy day


class LightPositionRanges:
    """Position randomization ranges (in meters, relative to origin)."""

    # Relative offsets (for position randomization around initial position)
    SMALL_OFFSET = ((-1.0, 1.0), (-1.0, 1.0), (-0.5, 0.5))
    MEDIUM_OFFSET = ((-2.0, 2.0), (-2.0, 2.0), (-1.0, 1.0))
    LARGE_OFFSET = ((-4.0, 4.0), (-4.0, 4.0), (-2.0, 2.0))

    # Absolute positions (for randomization within bounded space)
    TABLETOP = ((-1.0, 1.0), (-1.0, 1.0), (0.5, 2.0))  # Above a table
    ROOM_SMALL = ((-3.0, 3.0), (-3.0, 3.0), (1.0, 4.0))  # Small room interior
    ROOM_MEDIUM = ((-5.0, 5.0), (-5.0, 5.0), (1.5, 4.5))  # Medium room interior


class LightOrientationRanges:
    """Orientation randomization ranges (in degrees)."""

    SMALL = ((-15.0, 15.0), (-15.0, 15.0), (-15.0, 15.0))
    MEDIUM = ((-45.0, 45.0), (-45.0, 45.0), (-45.0, 45.0))
    LARGE = ((-90.0, 90.0), (-90.0, 90.0), (-180.0, 180.0))


class LightPresets:
    """Pre-configured light randomization presets organized by light type and scenario."""

    @staticmethod
    def distant_outdoor_sun(light_name: str, randomization_mode: str = "combined") -> LightRandomCfg:
        """DistantLight preset for outdoor sun lighting.

        Strong directional light with natural daylight colors.
        Orientation randomization simulates different times of day.
        """
        return LightRandomCfg(
            light_name=light_name,
            intensity=LightIntensityRandomCfg(
                intensity_range=LightIntensityRanges.OUTDOOR_SUN, distribution="uniform", enabled=True
            ),
            color=LightColorRandomCfg(
                temperature_range=LightColorRanges.DAYLIGHT,
                use_temperature=True,
                distribution="uniform",
                enabled=True,
            ),
            orientation=LightOrientationRandomCfg(
                angle_range=LightOrientationRanges.LARGE,
                relative_to_origin=True,
                distribution="uniform",
                enabled=True,
            ),
            randomization_mode=randomization_mode,
        )

    @staticmethod
    def distant_indoor_sun(light_name: str, randomization_mode: str = "combined") -> LightRandomCfg:
        """DistantLight preset for indoor scenes with window lighting.

        Moderate directional light simulating sunlight through windows.
        """
        return LightRandomCfg(
            light_name=light_name,
            intensity=LightIntensityRandomCfg(
                intensity_range=LightIntensityRanges.STUDIO, distribution="uniform", enabled=True
            ),
            color=LightColorRandomCfg(
                temperature_range=LightColorRanges.DAYLIGHT,
                use_temperature=True,
                distribution="uniform",
                enabled=True,
            ),
            orientation=LightOrientationRandomCfg(
                angle_range=LightOrientationRanges.MEDIUM,
                relative_to_origin=True,
                distribution="uniform",
                enabled=True,
            ),
            randomization_mode=randomization_mode,
        )

    @staticmethod
    def distant_enclosed_room(light_name: str, randomization_mode: str = "combined") -> LightRandomCfg:
        """DistantLight preset for enclosed rooms with walls.

        High intensity to compensate for enclosed environment.
        is_global=True ensures light penetrates walls.
        """
        return LightRandomCfg(
            light_name=light_name,
            intensity=LightIntensityRandomCfg(
                intensity_range=LightIntensityRanges.ENCLOSED_ROOM_DISTANT, distribution="uniform", enabled=True
            ),
            color=LightColorRandomCfg(color_range=LightColorRanges.NATURAL, distribution="uniform", enabled=True),
            orientation=LightOrientationRandomCfg(
                angle_range=LightOrientationRanges.MEDIUM,
                relative_to_origin=True,
                distribution="uniform",
                enabled=True,
            ),
            randomization_mode=randomization_mode,
        )

    @staticmethod
    def sphere_ceiling_light(light_name: str, randomization_mode: str = "combined") -> LightRandomCfg:
        """SphereLight preset for ceiling-mounted lights in open scenes.

        Moderate intensity with warm indoor colors.
        Position randomization keeps light near ceiling.
        Use with is_global=False (default) for realistic wall blocking.
        """
        return LightRandomCfg(
            light_name=light_name,
            intensity=LightIntensityRandomCfg(
                intensity_range=LightIntensityRanges.INDOOR_AMBIENT, distribution="uniform", enabled=True
            ),
            color=LightColorRandomCfg(
                temperature_range=LightColorRanges.WARM_WHITE,
                use_temperature=True,
                distribution="uniform",
                enabled=True,
            ),
            position=LightPositionRandomCfg(
                position_range=LightPositionRanges.MEDIUM_OFFSET,
                relative_to_origin=True,
                distribution="uniform",
                enabled=True,
            ),
            randomization_mode=randomization_mode,
        )

    @staticmethod
    def sphere_task_light(light_name: str, randomization_mode: str = "combined") -> LightRandomCfg:
        """SphereLight preset for task lighting (desk lamps, reading lights).

        Higher intensity with small position variation.
        Use with is_global=False (default) for realistic shadows.
        """
        return LightRandomCfg(
            light_name=light_name,
            intensity=LightIntensityRandomCfg(
                intensity_range=LightIntensityRanges.INDOOR_TASK, distribution="uniform", enabled=True
            ),
            color=LightColorRandomCfg(
                temperature_range=LightColorRanges.NEUTRAL_WHITE,
                use_temperature=True,
                distribution="uniform",
                enabled=True,
            ),
            position=LightPositionRandomCfg(
                position_range=LightPositionRanges.SMALL_OFFSET,
                relative_to_origin=True,
                distribution="uniform",
                enabled=True,
            ),
            randomization_mode=randomization_mode,
        )

    @staticmethod
    def sphere_enclosed_room_global(light_name: str, randomization_mode: str = "combined") -> LightRandomCfg:
        """SphereLight preset for enclosed rooms (with is_global=True).

        High intensity with intensity and color randomization ONLY.
        NO position randomization (lights stay at fixed positions).
        Use with is_global=True to penetrate walls.

        This is the recommended preset for enclosed room scenarios.
        """
        return LightRandomCfg(
            light_name=light_name,
            intensity=LightIntensityRandomCfg(
                intensity_range=LightIntensityRanges.ENCLOSED_ROOM_SPHERE, distribution="uniform", enabled=True
            ),
            color=LightColorRandomCfg(color_range=LightColorRanges.NATURAL, distribution="uniform", enabled=True),
            # NO position randomization for enclosed rooms!
            position=None,
            randomization_mode=randomization_mode,
        )

    @staticmethod
    def sphere_enclosed_room_local(light_name: str, randomization_mode: str = "combined") -> LightRandomCfg:
        """SphereLight preset for enclosed rooms (with is_global=False).

        Moderate intensity with small position offsets to stay inside room.
        Use with is_global=False for realistic wall blocking.

        WARNING: Ensure initial position + offset stays within room bounds!
        """
        return LightRandomCfg(
            light_name=light_name,
            intensity=LightIntensityRandomCfg(
                intensity_range=LightIntensityRanges.STUDIO, distribution="uniform", enabled=True
            ),
            color=LightColorRandomCfg(
                temperature_range=LightColorRanges.WARM_WHITE,
                use_temperature=True,
                distribution="uniform",
                enabled=True,
            ),
            position=LightPositionRandomCfg(
                position_range=LightPositionRanges.SMALL_OFFSET,  # Small offsets to avoid walls
                relative_to_origin=True,
                distribution="uniform",
                enabled=True,
            ),
            randomization_mode=randomization_mode,
        )

    @staticmethod
    def dome_ambient(light_name: str, randomization_mode: str = "combined") -> LightRandomCfg:
        """DomeLight preset for ambient sky lighting.

        Provides uniform ambient lighting from all directions.
        Always penetrates walls (is_global=True).
        """
        return LightRandomCfg(
            light_name=light_name,
            intensity=LightIntensityRandomCfg(
                intensity_range=LightIntensityRanges.INDOOR_AMBIENT, distribution="uniform", enabled=True
            ),
            color=LightColorRandomCfg(
                temperature_range=LightColorRanges.OVERCAST_SKY,
                use_temperature=True,
                distribution="uniform",
                enabled=True,
            ),
            randomization_mode=randomization_mode,
        )

    @staticmethod
    def demo_intensity_only(
        light_name: str, intensity_range: tuple[float, float] = (1000.0, 10000.0)
    ) -> LightRandomCfg:
        """Demo preset: Only randomize intensity (keep color and position fixed).

        Useful for testing intensity effects.
        """
        return LightRandomCfg(
            light_name=light_name,
            intensity=LightIntensityRandomCfg(intensity_range=intensity_range, distribution="uniform", enabled=True),
            randomization_mode="intensity_only",
        )

    @staticmethod
    def demo_color_only(light_name: str) -> LightRandomCfg:
        """Demo preset: Only randomize color (keep intensity and position fixed).

        Useful for testing color effects with full spectrum randomization.
        """
        return LightRandomCfg(
            light_name=light_name,
            color=LightColorRandomCfg(color_range=LightColorRanges.FULL_SPECTRUM, distribution="uniform", enabled=True),
            randomization_mode="color_only",
        )

    @staticmethod
    def demo_position_only(light_name: str, position_range: tuple = LightPositionRanges.ROOM_MEDIUM) -> LightRandomCfg:
        """Demo preset: Only randomize position (keep intensity and color fixed).

        Useful for testing position effects and shadows.
        Only applicable to SphereLight/CylinderLight/DiskLight.
        """
        return LightRandomCfg(
            light_name=light_name,
            position=LightPositionRandomCfg(
                position_range=position_range, relative_to_origin=False, distribution="uniform", enabled=True
            ),
            randomization_mode="position_only",
        )


class LightScenarios:
    """Pre-configured multi-light scenarios for common setups."""

    @staticmethod
    def three_point_studio() -> list[LightRandomCfg]:
        """Professional three-point studio lighting setup.

        For open scenes or studios.
        """
        return [
            LightPresets.distant_outdoor_sun("key_light", "combined"),
            LightPresets.sphere_ceiling_light("fill_light", "combined"),
            LightPresets.sphere_task_light("rim_light", "combined"),
        ]

    @staticmethod
    def enclosed_room_basic() -> list[LightRandomCfg]:
        """Basic enclosed room lighting with 1 distant + 3 sphere lights.

        All lights use is_global=True to penetrate walls.
        DistantLight: Main directional lighting
        SphereLights: Fill lighting from different positions
        """
        return [
            LightPresets.distant_enclosed_room("main_light", "combined"),
            LightPresets.sphere_enclosed_room_global("fill_light", "combined"),
            LightPresets.sphere_enclosed_room_global("back_light", "combined"),
            LightPresets.sphere_enclosed_room_global("table_light", "combined"),
        ]

    @staticmethod
    def outdoor_daylight() -> list[LightRandomCfg]:
        """Outdoor scene with sun and ambient sky lighting."""
        return [
            LightPresets.distant_outdoor_sun("sun_light", "combined"),
            LightPresets.dome_ambient("sky_light", "combined"),
        ]
