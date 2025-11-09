from __future__ import annotations

from metasim.constants import PhysicStateType
from metasim.example.example_pack.tasks.checkers.checkers import DetectedChecker
from metasim.example.example_pack.tasks.checkers.detectors import RelativeBboxDetector
from metasim.scenario.objects import PrimitiveSphereCfg, RigidObjCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.registry import register_task

from .maniskill_base import ManiskillBaseTask


@register_task("maniskill.place_sphere", "place_sphere")
class PlaceSphereTask(ManiskillBaseTask):
    """The place-sphere task from ManiSkill.
    .. Description:
    ### 📦 Source Metadata (from ManiSkill)
    ### title:
    PlaceSphere
    ### group:
    Maniskill
    ### description:
    Place the sphere into the shallow bin.
    ### randomizations:
    - The position of the bin and the sphere are randomized: The bin is initialized in [0, 0.1] x [-0.1, 0.1], and the sphere is initialized in [-0.1, -0.05] x [-0.1, 0.1]
    ### success:
    - The sphere is placed on the top of the bin. The robot remains static and the gripper is not closed at the end state.
    ### badges:
    - dense
    - sparse
    ### official_url:
    https://maniskill.readthedocs.io/en/latest/tasks/table_top_gripper/index.html#placesphere-v1
    ### poster_url:
    (none).
    ---
    ### video_url:
    place_sphere.mp4
    ### platforms:
    - mujuco
    - isaaclab
    ### notes:
    (none)
    """  # noqa: D205

    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="bin",
                usd_path="roboverse_data/assets/maniskill/PlaceSphere/bin/usd/bin.usd",
                mjcf_path="roboverse_data/assets/maniskill/PlaceSphere/bin/mjcf/PlaceSphere_bin.xml",
                urdf_path="roboverse_data/assets/maniskill/PlaceSphere/bin/urdf/PlaceSphere_bin.urdf",
                physics=PhysicStateType.RIGIDBODY,
            ),
            PrimitiveSphereCfg(
                name="sphere",
                radius=0.03,
                mass=0.001,
                physics=PhysicStateType.RIGIDBODY,
                color=[1.0, 0.0, 0.0],
            ),
        ]
    )

    episode_length = 250
    checker = DetectedChecker(
        obj_name="sphere",
        detector=RelativeBboxDetector(
            base_obj_name="bin",
            relative_pos=(0.0, 0.0, 0.04),
            relative_quat=(1.0, 0.0, 0.0, 0.0),
            checker_lower=(-0.02, -0.02, -0.02),
            checker_upper=(0.02, 0.02, 0.02),
            ignore_base_ori=True,
        ),
    )
    traj_filepath = "roboverse_data/trajs/maniskill/place_sphere/v2/franka_v2.pkl.gz"
