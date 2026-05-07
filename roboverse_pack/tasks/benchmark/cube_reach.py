from __future__ import annotations

from roboverse_pack.benchmark.spec import (
    BenchmarkCameraPreset,
    BenchmarkObjectSpec,
    BenchmarkRobotTeleopProfile,
    BenchmarkSceneSpec,
    BenchmarkTaskSpec,
)

CUBE_HALF_EXTENTS_CM = (2.25, 2.25, 2.25)
LEFT_CUBE_REACH_POS_CM = (-12.0, -70.0, 22.25)

OPENARM_BIMANUAL_WUJI_PROFILE = BenchmarkRobotTeleopProfile(
    robot="openarm_bimanual_wuji",
    control_joint_count=54,
    joint_slices={
        "left_arm": (0, 7),
        "right_arm": (7, 14),
        "left_hand": (14, 34),
        "right_hand": (34, 54),
    },
    body_names={
        "left_ee": "openarm_left_link7",
        "right_ee": "openarm_right_link7",
        "left_palm": "left_palm_link",
        "right_palm": "right_palm_link",
    },
)


def build_cube_reach_spec() -> BenchmarkTaskSpec:
    """Build the first native benchmark task migrated from BiDexBench."""
    return BenchmarkTaskSpec(
        name="benchmark.cube_reach",
        family="precision",
        description="Reach a small tabletop cube with either hand.",
        scene=BenchmarkSceneSpec(
            objects=(
                BenchmarkObjectSpec(
                    name="cube",
                    kind="cube",
                    pos_cm=LEFT_CUBE_REACH_POS_CM,
                    half_extents_cm=CUBE_HALF_EXTENTS_CM,
                    movable=True,
                ),
            ),
            camera_presets=(
                BenchmarkCameraPreset(
                    name="overview",
                    width=1920,
                    height=1080,
                    pos=(1.20, 0.0, 0.95),
                    look_at=(0.0, -0.55, 0.24),
                ),
                BenchmarkCameraPreset(
                    name="chest",
                    width=1920,
                    height=1080,
                    pos=(0.0, -1.05, 0.55),
                    look_at=(0.0, -0.55, 0.25),
                ),
            ),
            physics_hints={"cube_half_extents_cm": CUBE_HALF_EXTENTS_CM},
        ),
        supported_simulators=("isaacsim", "mujoco"),
        supported_robots=("openarm_bimanual_wuji",),
        default_robot="openarm_bimanual_wuji",
        robot_profiles={"openarm_bimanual_wuji": OPENARM_BIMANUAL_WUJI_PROFILE},
        aliases=("cube_reach",),
    )


BENCHMARK_TASK_SPECS = (build_cube_reach_spec(),)
