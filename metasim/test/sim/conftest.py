"""Sim test suite registration for shared handler utilities."""

from __future__ import annotations

from metasim.constants import PhysicStateType
from metasim.scenario.objects import ArticulationObjCfg, PrimitiveCubeCfg, PrimitiveSphereCfg, RigidObjCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.scenario.simulator_params import SimParamCfg
from metasim.test.conftest import _SUPPORTED_SIMS, register_shared_suite
from roboverse_pack.robots.franka_cfg import FrankaCfg


def get_state_consistency_scenario(sim: str, num_envs: int) -> ScenarioCfg:
    """Create the standard scenario configuration for state consistency tests."""
    if sim not in _SUPPORTED_SIMS:
        raise ValueError(f"Unsupported simulator '{sim}' for state consistency tests")

    return ScenarioCfg(
        simulator=sim,
        num_envs=num_envs,
        headless=True,
        objects=[
            PrimitiveCubeCfg(
                name="cube",
                size=(0.1, 0.1, 0.1),
                color=[1.0, 0.0, 0.0],
                physics=PhysicStateType.RIGIDBODY,
            ),
            PrimitiveSphereCfg(
                name="sphere",
                radius=0.1,
                color=[0.0, 0.0, 1.0],
                physics=PhysicStateType.RIGIDBODY,
            ),
            RigidObjCfg(
                name="bbq_sauce",
                scale=(2, 2, 2),
                physics=PhysicStateType.RIGIDBODY,
                usd_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/bbq_sauce/usd/bbq_sauce.usd",
                urdf_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/bbq_sauce/urdf/bbq_sauce.urdf",
                mjcf_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/bbq_sauce/mjcf/bbq_sauce.xml",
            ),
            ArticulationObjCfg(
                name="box_base",
                fix_base_link=True,
                usd_path="roboverse_data/assets/rlbench/close_box/box_base/usd/box_base.usd",
                urdf_path="roboverse_data/assets/rlbench/close_box/box_base/urdf/box_base_unique.urdf",
                mjcf_path="roboverse_data/assets/rlbench/close_box/box_base/mjcf/box_base_unique.mjcf",
            ),
        ],
        robots=[FrankaCfg()],
    )


def get_gravity_scenario(sim: str, num_envs: int) -> ScenarioCfg:
    """Create scenario configuration for gravity tests."""
    if sim not in _SUPPORTED_SIMS:
        raise ValueError(f"Unsupported simulator '{sim}' for gravity tests")

    return ScenarioCfg(
        simulator=sim,
        num_envs=num_envs,
        headless=True,
        objects=[
            PrimitiveCubeCfg(
                name="cube",
                size=(0.1, 0.1, 0.1),
                color=[1.0, 0.0, 0.0],
                physics=PhysicStateType.RIGIDBODY,
                default_position=[0, 0, 10.0],
            ),
        ],
        robots=[FrankaCfg()],
        sim_params=SimParamCfg(dt=0.001),
        gravity=(0, 0, -1),
        decimation=100,
    )


def get_init_pose_scenario(sim: str, num_envs: int) -> ScenarioCfg:
    """Create scenario configuration for init pose tests."""
    if sim not in _SUPPORTED_SIMS:
        raise ValueError(f"Unsupported simulator '{sim}' for init pose tests")

    return ScenarioCfg(
        robots=[FrankaCfg()],
        headless=True,
        num_envs=num_envs,
        simulator=sim,
        objects=[
            PrimitiveCubeCfg(
                name="cube",
                size=(0.1, 0.1, 0.1),
                color=[1.0, 0.0, 0.0],
                default_position=[0.3, -0.2, 0.05],
                default_orientation=[1.0, 0.0, 0.0, 0.0],
                physics=PhysicStateType.RIGIDBODY,
            ),
            PrimitiveSphereCfg(
                name="sphere",
                radius=0.1,
                color=[0.0, 0.0, 1.0],
                default_position=[0.4, -0.6, 0.1],
                default_orientation=[1.0, 0.0, 0.0, 0.0],
                physics=PhysicStateType.RIGIDBODY,
            ),
            RigidObjCfg(
                name="bbq_sauce",
                scale=(2, 2, 2),
                physics=PhysicStateType.RIGIDBODY,
                default_position=[0.7, -0.3, 0.0094],
                default_orientation=[1.0, 0.0, 0.0, 0.0],
                usd_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/bbq_sauce/usd/bbq_sauce.usd",
                urdf_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/bbq_sauce/urdf/bbq_sauce.urdf",
                mjcf_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/bbq_sauce/mjcf/bbq_sauce.xml",
            ),
            ArticulationObjCfg(
                name="box_base",
                fix_base_link=True,
                default_position=[0.5, 0.2, 0.1],
                default_orientation=[0.0, 0.7071, 0.0, 0.7071],
                usd_path="roboverse_data/assets/rlbench/close_box/box_base/usd/box_base.usd",
                urdf_path="roboverse_data/assets/rlbench/close_box/box_base/urdf/box_base_unique.urdf",
                mjcf_path="roboverse_data/assets/rlbench/close_box/box_base/mjcf/box_base_unique.mjcf",
            ),
        ],
    )


def get_kinematics_scenario(sim: str, num_envs: int) -> ScenarioCfg:
    """Create scenario configuration for kinematics tests."""
    if sim not in _SUPPORTED_SIMS:
        raise ValueError(f"Unsupported simulator '{sim}' for kinematics tests")

    return ScenarioCfg(
        robots=[FrankaCfg()],
        headless=True,
        num_envs=num_envs,
        simulator=sim,
        objects=[
            PrimitiveCubeCfg(
                name="cube",
                size=(0.1, 0.1, 0.1),
                color=[1.0, 0.0, 0.0],
                default_position=[0.3, -0.2, 0.05],
                default_orientation=[1.0, 0.0, 0.0, 0.0],
                physics=PhysicStateType.RIGIDBODY,
            ),
            PrimitiveSphereCfg(
                name="sphere",
                radius=0.1,
                color=[0.0, 0.0, 1.0],
                default_position=[0.4, -0.6, 0.1],
                default_orientation=[1.0, 0.0, 0.0, 0.0],
                physics=PhysicStateType.RIGIDBODY,
            ),
            RigidObjCfg(
                name="bbq_sauce",
                scale=(2, 2, 2),
                physics=PhysicStateType.RIGIDBODY,
                default_position=[0.7, -0.3, 0.0094],
                default_orientation=[1.0, 0.0, 0.0, 0.0],
                usd_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/bbq_sauce/usd/bbq_sauce.usd",
                urdf_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/bbq_sauce/urdf/bbq_sauce.urdf",
                mjcf_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/bbq_sauce/mjcf/bbq_sauce.xml",
            ),
            ArticulationObjCfg(
                name="box_base",
                fix_base_link=True,
                default_position=[0.5, 0.2, 0.1],
                default_orientation=[0.0, 0.7071, 0.0, 0.7071],
                usd_path="roboverse_data/assets/rlbench/close_box/box_base/usd/box_base.usd",
                urdf_path="roboverse_data/assets/rlbench/close_box/box_base/urdf/box_base_unique.urdf",
                mjcf_path="roboverse_data/assets/rlbench/close_box/box_base/mjcf/box_base_unique.mjcf",
            ),
        ],
    )


# Register suites with the shared handler machinery in metasim/test/conftest.py
register_shared_suite("metasim.test.sim.test_state_consistency", get_state_consistency_scenario)
register_shared_suite("metasim.test.sim.test_gravity", get_gravity_scenario)
register_shared_suite("metasim.test.sim.test_init_pose", get_init_pose_scenario)
register_shared_suite("metasim.test.sim.test_kinemetics", get_kinematics_scenario)

# Register new test suites (reuse existing scenarios that match requirements)
register_shared_suite("metasim.test.sim.test_handler_lifecycle", get_init_pose_scenario)
register_shared_suite("metasim.test.sim.test_parallel_handler", get_init_pose_scenario)
register_shared_suite("metasim.test.sim.test_state_modes", get_init_pose_scenario)
register_shared_suite("metasim.test.sim.test_dof_control", get_init_pose_scenario)
