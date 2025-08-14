import logoru as log
import numpy as np
from base import BaseTaskWrapper

from metasim.constants import PhysicStateType
from metasim.scenario.cameras import PinholeCameraCfg
from metasim.scenario.objects import ArticulationObjCfg, PrimitiveCubeCfg, PrimitiveSphereCfg, RigidObjCfg
from metasim.scenario.scenario import ScenarioCfg

if __name__ == "__main__":
    # initialize scenario
    scenario = ScenarioCfg(
        robots=["franka"],
        try_add_table=False,
        sim="mujoco",
        headless=True,
        num_envs=1,
    )

    # add cameras
    scenario.cameras = [PinholeCameraCfg(width=1024, height=1024, pos=(1.5, -1.5, 1.5), look_at=(0.0, 0.0, 0.0))]

    # add objects
    scenario.objects = [
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
            usd_path="../../metasim/example/example_assets/bbq_sauce/usd/bbq_sauce.usd",
            urdf_path="../../metasim/example/example_assets/bbq_sauce/urdf/bbq_sauce.urdf",
            mjcf_path="../../metasim/example/example_assets/bbq_sauce/mjcf/bbq_sauce.xml",
        ),
        ArticulationObjCfg(
            name="box_base",
            fix_base_link=True,
            usd_path="../../metasim/example/example_assets/box_base/usd/box_base.usd",
            urdf_path="../../metasim/example/example_assets/box_base/urdf/box_base_unique.urdf",
            mjcf_path="../../metasim/example/example_assets/box_base/mjcf/box_base_unique.mjcf",
        ),
    ]

    task = BaseTaskWrapper(scenario)

    for i in range(10):
        tmp = task.step(np.zeros(9))
        log.info(tmp)
