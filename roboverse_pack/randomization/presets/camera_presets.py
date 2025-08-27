"""Presets for camera domain randomization."""

from __future__ import annotations

from ..camera_randomizer import (
    CameraImageRandomCfg,
    CameraIntrinsicsRandomCfg,
    CameraLookAtRandomCfg,
    CameraOrientationRandomCfg,
    CameraPositionRandomCfg,
    CameraRandomCfg,
)


class CameraProperties:
    """Common camera property ranges for different scenarios."""

    # Position ranges (in meters) - for absolute positioning mode
    POSITION_CLOSE = ((-1.0, 1.0), (-1.0, 1.0), (0.8, 1.5))  # Close-up view
    POSITION_MEDIUM = ((-2.0, 2.0), (-2.0, 2.0), (1.0, 2.5))  # Medium distance
    POSITION_FAR = ((-3.0, 3.0), (-3.0, 3.0), (1.5, 3.5))  # Far view
    POSITION_EXTREME = ((-4.0, 4.0), (-4.0, 4.0), (0.5, 4.5))  # Extreme variation

    # Position delta ranges (in meters) - for micro-adjustment mode (默认)
    DELTA_TINY = ((-0.1, 0.1), (-0.1, 0.1), (-0.1, 0.1))  # Very small movements
    DELTA_SMALL = ((-0.3, 0.3), (-0.3, 0.3), (-0.2, 0.2))  # Small movements
    DELTA_MEDIUM = ((-0.5, 0.5), (-0.5, 0.5), (-0.3, 0.3))  # Medium movements
    DELTA_LARGE = ((-1.0, 1.0), (-1.0, 1.0), (-0.5, 0.5))  # Large movements

    # Look-at ranges (in meters) - for absolute look-at mode
    LOOKAT_CENTER = ((-0.1, 0.1), (-0.1, 0.1), (-0.1, 0.1))  # Focus on center
    LOOKAT_OBJECT = ((-0.5, 0.5), (-0.5, 0.5), (-0.2, 0.3))  # Focus on objects
    LOOKAT_WORKSPACE = ((-1.0, 1.0), (-1.0, 1.0), (-0.5, 0.5))  # Workspace area
    LOOKAT_WIDE = ((-2.0, 2.0), (-2.0, 2.0), (-1.0, 1.0))  # Wide area

    # Look-at delta ranges (in meters) - for micro-adjustment mode (默认)
    LOOKAT_DELTA_TINY = ((-0.05, 0.05), (-0.05, 0.05), (-0.05, 0.05))  # Very small direction changes
    LOOKAT_DELTA_SMALL = ((-0.2, 0.2), (-0.2, 0.2), (-0.1, 0.1))  # Small direction changes
    LOOKAT_DELTA_MEDIUM = ((-0.5, 0.5), (-0.5, 0.5), (-0.3, 0.3))  # Medium direction changes
    LOOKAT_DELTA_LARGE = ((-1.0, 1.0), (-1.0, 1.0), (-0.5, 0.5))  # Large direction changes

    # Rotation delta ranges (in degrees) - for orientation micro-adjustments
    ROTATION_DELTA_TINY = ((-2, 2), (-2, 2), (-1, 1))  # Very small rotations
    ROTATION_DELTA_SMALL = ((-5, 5), (-5, 5), (-2, 2))  # Small rotations
    ROTATION_DELTA_MEDIUM = ((-10, 10), (-10, 10), (-5, 5))  # Medium rotations
    ROTATION_DELTA_LARGE = ((-20, 20), (-20, 20), (-10, 10))  # Large rotations

    # Spherical coordinates (radius in meters, angles in degrees)
    SPHERICAL_CLOSE = ((0.8, 1.5), (-45, 45), (10, 80))  # Close orbit
    SPHERICAL_MEDIUM = ((1.5, 2.5), (-90, 90), (15, 75))  # Medium orbit
    SPHERICAL_FAR = ((2.0, 4.0), (-180, 180), (5, 85))  # Far orbit
    SPHERICAL_FULL = ((1.0, 3.0), (-180, 180), (-30, 90))  # Full sphere

    # FOV ranges (in degrees)
    FOV_NARROW = (30, 50)  # Telephoto
    FOV_NORMAL = (45, 75)  # Normal lens
    FOV_WIDE = (60, 90)  # Wide angle
    FOV_EXTREME = (20, 110)  # Extreme variation

    # Focal length ranges (in cm)
    FOCAL_SHORT = (12, 24)  # Wide angle
    FOCAL_NORMAL = (24, 50)  # Normal
    FOCAL_LONG = (50, 100)  # Telephoto
    FOCAL_EXTREME = (8, 120)  # Extreme variation

    # Aperture ranges (in cm)
    APERTURE_NARROW = (15, 25)  # Narrow aperture
    APERTURE_NORMAL = (20, 30)  # Normal aperture
    APERTURE_WIDE = (25, 40)  # Wide aperture

    # Focus distance ranges (in m)
    FOCUS_CLOSE = (0.1, 1.0)  # Close focus
    FOCUS_MEDIUM = (1.0, 10.0)  # Medium focus
    FOCUS_FAR = (10.0, 100.0)  # Far focus
    FOCUS_INFINITE = (100.0, 1000.0)  # Near infinity

    # Clipping ranges (near, far) in meters
    CLIPPING_CLOSE = ((0.01, 0.1), (1.0, 10.0))  # Close range
    CLIPPING_NORMAL = ((0.05, 0.5), (10.0, 100.0))  # Normal range
    CLIPPING_FAR = ((0.1, 1.0), (100.0, 1000.0))  # Far range

    # Image property ranges
    ASPECT_RATIO_WIDE = (0.5, 2.0)  # Wide variation
    ASPECT_RATIO_NORMAL = (0.8, 1.2)  # Normal variation
    RESOLUTION_LOW = ((256, 512), (256, 512))  # Low resolution range
    RESOLUTION_NORMAL = ((512, 1024), (512, 1024))  # Normal resolution range
    RESOLUTION_HIGH = ((1024, 2048), (1024, 2048))  # High resolution range


class CameraPresets:
    """Predefined camera randomization configurations for common use cases."""

    @staticmethod
    def surveillance_camera(camera_name: str, randomization_mode: str = "combined") -> CameraRandomCfg:
        """Surveillance/security camera setup with micro-adjustments."""
        return CameraRandomCfg(
            camera_name=camera_name,
            position=CameraPositionRandomCfg(
                # Default: micro-adjustment mode
                delta_range=CameraProperties.DELTA_SMALL,
                use_delta=True,
                # Alternative: absolute positioning
                position_range=CameraProperties.POSITION_MEDIUM,
                distribution="uniform",
                enabled=True,
            ),
            orientation=CameraOrientationRandomCfg(
                # Small rotation adjustments
                rotation_delta=CameraProperties.ROTATION_DELTA_SMALL,
                distribution="uniform",
                enabled=True,
            ),
            look_at=CameraLookAtRandomCfg(
                # Default: micro-adjustment mode
                look_at_delta=CameraProperties.LOOKAT_DELTA_SMALL,
                use_delta=True,
                # Alternative: absolute look-at
                look_at_range=CameraProperties.LOOKAT_WORKSPACE,
                distribution="uniform",
                enabled=True,
            ),
            intrinsics=CameraIntrinsicsRandomCfg(
                fov_range=CameraProperties.FOV_NORMAL, use_fov=True, distribution="uniform", enabled=True
            ),
            randomization_mode=randomization_mode,
        )

    @staticmethod
    def handheld_camera(camera_name: str, randomization_mode: str = "combined") -> CameraRandomCfg:
        """Handheld/mobile camera with natural movement patterns."""
        return CameraRandomCfg(
            camera_name=camera_name,
            position=CameraPositionRandomCfg(
                position_range=CameraProperties.POSITION_CLOSE,
                distribution="gaussian",  # More natural movement
                enabled=True,
            ),
            orientation=CameraOrientationRandomCfg(
                # Natural handheld shake
                rotation_delta=CameraProperties.ROTATION_DELTA_MEDIUM,
                distribution="gaussian",
                enabled=True,
            ),
            look_at=CameraLookAtRandomCfg(
                look_at_range=CameraProperties.LOOKAT_OBJECT, distribution="gaussian", enabled=True
            ),
            intrinsics=CameraIntrinsicsRandomCfg(
                fov_range=CameraProperties.FOV_NORMAL, use_fov=True, distribution="uniform", enabled=True
            ),
            randomization_mode=randomization_mode,
        )

    @staticmethod
    def robotic_camera(camera_name: str, randomization_mode: str = "combined") -> CameraRandomCfg:
        """Robot-mounted camera with precise positioning."""
        return CameraRandomCfg(
            camera_name=camera_name,
            position=CameraPositionRandomCfg(
                position_range=CameraProperties.POSITION_CLOSE, distribution="uniform", enabled=True
            ),
            orientation=CameraOrientationRandomCfg(
                # Precise small adjustments
                rotation_delta=CameraProperties.ROTATION_DELTA_TINY,
                distribution="uniform",
                enabled=True,
            ),
            look_at=CameraLookAtRandomCfg(
                spherical_range=CameraProperties.SPHERICAL_CLOSE,
                use_spherical=True,
                distribution="uniform",
                enabled=True,
            ),
            intrinsics=CameraIntrinsicsRandomCfg(
                focal_length_range=CameraProperties.FOCAL_NORMAL,
                horizontal_aperture_range=CameraProperties.APERTURE_NORMAL,
                focus_distance_range=CameraProperties.FOCUS_MEDIUM,
                clipping_range=CameraProperties.CLIPPING_NORMAL,
                distribution="uniform",
                enabled=True,
            ),
            image=CameraImageRandomCfg(
                aspect_ratio_range=CameraProperties.ASPECT_RATIO_NORMAL,
                width_range=CameraProperties.RESOLUTION_NORMAL[0],
                height_range=CameraProperties.RESOLUTION_NORMAL[1],
                use_aspect_ratio=True,
                distribution="uniform",
                enabled=True,
            ),
            randomization_mode=randomization_mode,
        )

    @staticmethod
    def surveillance_camera_absolute(camera_name: str, randomization_mode: str = "combined") -> CameraRandomCfg:
        """Surveillance camera with absolute positioning (original behavior)."""
        return CameraRandomCfg(
            camera_name=camera_name,
            position=CameraPositionRandomCfg(
                # Absolute positioning mode
                position_range=CameraProperties.POSITION_MEDIUM,
                use_delta=False,
                distribution="uniform",
                enabled=True,
            ),
            orientation=CameraOrientationRandomCfg(
                # Medium rotation adjustments
                rotation_delta=CameraProperties.ROTATION_DELTA_MEDIUM,
                distribution="uniform",
                enabled=True,
            ),
            look_at=CameraLookAtRandomCfg(
                # Absolute look-at mode
                look_at_range=CameraProperties.LOOKAT_WORKSPACE,
                use_delta=False,
                distribution="uniform",
                enabled=True,
            ),
            intrinsics=CameraIntrinsicsRandomCfg(
                fov_range=CameraProperties.FOV_NORMAL, use_fov=True, distribution="uniform", enabled=True
            ),
            randomization_mode=randomization_mode,
        )

    @staticmethod
    def drone_camera(camera_name: str, randomization_mode: str = "combined") -> CameraRandomCfg:
        """Drone/aerial camera with high viewpoints."""
        return CameraRandomCfg(
            camera_name=camera_name,
            position=CameraPositionRandomCfg(
                position_range=CameraProperties.POSITION_FAR, distribution="uniform", enabled=True
            ),
            orientation=CameraOrientationRandomCfg(
                # Flying camera movements
                rotation_delta=CameraProperties.ROTATION_DELTA_LARGE,
                distribution="uniform",
                enabled=True,
            ),
            look_at=CameraLookAtRandomCfg(
                spherical_range=CameraProperties.SPHERICAL_MEDIUM,
                use_spherical=True,
                distribution="uniform",
                enabled=True,
            ),
            intrinsics=CameraIntrinsicsRandomCfg(
                fov_range=CameraProperties.FOV_WIDE, use_fov=True, distribution="uniform", enabled=True
            ),
            randomization_mode=randomization_mode,
        )

    @staticmethod
    def cinema_camera(camera_name: str, randomization_mode: str = "combined") -> CameraRandomCfg:
        """Cinematic camera with dramatic angles and focal lengths."""
        return CameraRandomCfg(
            camera_name=camera_name,
            position=CameraPositionRandomCfg(
                position_range=CameraProperties.POSITION_EXTREME, distribution="uniform", enabled=True
            ),
            orientation=CameraOrientationRandomCfg(
                # Dramatic camera movements
                rotation_delta=CameraProperties.ROTATION_DELTA_LARGE,
                distribution="uniform",
                enabled=True,
            ),
            look_at=CameraLookAtRandomCfg(
                spherical_range=CameraProperties.SPHERICAL_FULL,
                use_spherical=True,
                distribution="uniform",
                enabled=True,
            ),
            intrinsics=CameraIntrinsicsRandomCfg(
                focal_length_range=CameraProperties.FOCAL_EXTREME,
                horizontal_aperture_range=CameraProperties.APERTURE_WIDE,
                distribution="uniform",
                enabled=True,
            ),
            randomization_mode=randomization_mode,
        )

    @staticmethod
    def inspection_camera(camera_name: str, randomization_mode: str = "combined") -> CameraRandomCfg:
        """Industrial inspection camera with close-up details."""
        return CameraRandomCfg(
            camera_name=camera_name,
            position=CameraPositionRandomCfg(
                position_range=CameraProperties.POSITION_CLOSE, distribution="uniform", enabled=True
            ),
            orientation=CameraOrientationRandomCfg(
                # Precise inspection movements
                rotation_delta=CameraProperties.ROTATION_DELTA_TINY,
                distribution="gaussian",
                enabled=True,
            ),
            look_at=CameraLookAtRandomCfg(
                look_at_range=CameraProperties.LOOKAT_CENTER, distribution="gaussian", enabled=True
            ),
            intrinsics=CameraIntrinsicsRandomCfg(
                focal_length_range=CameraProperties.FOCAL_LONG,
                horizontal_aperture_range=CameraProperties.APERTURE_NARROW,
                distribution="uniform",
                enabled=True,
            ),
            randomization_mode=randomization_mode,
        )

    @staticmethod
    def demo_camera(camera_name: str, randomization_mode: str = "combined") -> CameraRandomCfg:
        """Demonstration camera with maximum variation for testing."""
        return CameraRandomCfg(
            camera_name=camera_name,
            position=CameraPositionRandomCfg(
                position_range=CameraProperties.POSITION_EXTREME, distribution="uniform", enabled=True
            ),
            orientation=CameraOrientationRandomCfg(
                # Maximum rotation variation for testing
                rotation_delta=CameraProperties.ROTATION_DELTA_LARGE,
                distribution="uniform",
                enabled=True,
            ),
            look_at=CameraLookAtRandomCfg(
                spherical_range=CameraProperties.SPHERICAL_FULL,
                use_spherical=True,
                distribution="uniform",
                enabled=True,
            ),
            intrinsics=CameraIntrinsicsRandomCfg(
                fov_range=CameraProperties.FOV_EXTREME, use_fov=True, distribution="uniform", enabled=True
            ),
            randomization_mode=randomization_mode,
        )


class CameraScenarios:
    """Complete camera scenarios with multiple cameras."""

    @staticmethod
    def multi_view_setup() -> list[CameraRandomCfg]:
        """Multi-camera setup with different viewpoints."""
        return [
            CameraPresets.surveillance_camera("main_camera"),
            CameraPresets.handheld_camera("side_camera"),
            CameraPresets.drone_camera("top_camera"),
        ]

    @staticmethod
    def stereo_setup() -> list[CameraRandomCfg]:
        """Stereo camera pair with synchronized variations."""
        return [CameraPresets.robotic_camera("left_camera"), CameraPresets.robotic_camera("right_camera")]

    @staticmethod
    def production_setup() -> list[CameraRandomCfg]:
        """Production environment with multiple specialized cameras."""
        return [
            CameraPresets.surveillance_camera("overview_camera"),
            CameraPresets.robotic_camera("arm_camera"),
            CameraPresets.inspection_camera("detail_camera"),
        ]
