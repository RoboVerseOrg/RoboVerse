"""The base class and derived classes for the pick up single EGAD object task from ManiSkill."""

from maniskill_base import ManiskillBaseTask

from metasim.constants import PhysicStateType
from metasim.example.example_pack.tasks.checkers.checkers import PositionShiftChecker
from metasim.scenario.objects import RigidObjCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.registry import register_task


@register_task("maniskill.pick_single_egad_base", "pick_single_egad_base")
class _PickSingleEgadBaseTask(ManiskillBaseTask):
    """The pickup single EGAD object task from ManiSkill.

    The robot is tasked to pick up an EGAD object.
    Note that the checker is not same as the original one (checking if the cube is near the target position).
    The current one checks if the cube is lifted up 7.5 cm.
    This class should be derived to specify the exact configuration (asset path and demo path) of the task.
    """

    max_episode_steps = 250
    checker = PositionShiftChecker(
        obj_name="obj",
        distance=0.075,
        axis="z",
    )


@register_task("maniskill.pick_single_egad_a100", "pick_single_egad_a100")
class PickSingleEgadA100Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/A10_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/A10_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-A10_0_v2.pkl"


@register_task("maniskill.pick_single_egad_a110", "pick_single_egad_a110")
class PickSingleEgadA110Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/A11_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/A11_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-A11_0_v2.pkl"


@register_task("maniskill.pick_single_egad_a130", "pick_single_egad_a130")
class PickSingleEgadA130Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/A13_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/A13_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-A13_0_v2.pkl"


@register_task("maniskill.pick_single_egad_a140", "pick_single_egad_a140")
class PickSingleEgadA140Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/A14_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/A14_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-A14_0_v2.pkl"


@register_task("maniskill.pick_single_egad_a160", "pick_single_egad_a160")
class PickSingleEgadA160Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/A16_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/A16_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-A16_0_v2.pkl"


@register_task("maniskill.pick_single_egad_a161", "pick_single_egad_a161")
class PickSingleEgadA161Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/A16_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/A16_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-A16_1_v2.pkl"


@register_task("maniskill.pick_single_egad_a180", "pick_single_egad_a180")
class PickSingleEgadA180Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/A18_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/A18_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-A18_0_v2.pkl"


@register_task("maniskill.pick_single_egad_a190", "pick_single_egad_a190")
class PickSingleEgadA190Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/A19_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/A19_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-A19_0_v2.pkl"


@register_task("maniskill.pick_single_egad_a200", "pick_single_egad_a200")
class PickSingleEgadA200Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/A20_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/A20_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-A20_0_v2.pkl"


@register_task("maniskill.pick_single_egad_a210", "pick_single_egad_a210")
class PickSingleEgadA210Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/A21_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/A21_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-A21_0_v2.pkl"


@register_task("maniskill.pick_single_egad_a220", "pick_single_egad_a220")
class PickSingleEgadA220Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/A22_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/A22_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-A22_0_v2.pkl"


@register_task("maniskill.pick_single_egad_a240", "pick_single_egad_a240")
class PickSingleEgadA240Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/A24_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/A24_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-A24_0_v2.pkl"


@register_task("maniskill.pick_single_egad_b100", "pick_single_egad_b100")
class PickSingleEgadB100Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/B10_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/B10_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-B10_0_v2.pkl"


@register_task("maniskill.pick_single_egad_b101", "pick_single_egad_b101")
class PickSingleEgadB101Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/B10_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/B10_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-B10_1_v2.pkl"


@register_task("maniskill.pick_single_egad_b102", "pick_single_egad_b102")
class PickSingleEgadB102Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/B10_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/B10_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-B10_2_v2.pkl"


@register_task("maniskill.pick_single_egad_b103", "pick_single_egad_b103")
class PickSingleEgadB103Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/B10_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/B10_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-B10_3_v2.pkl"


@register_task("maniskill.pick_single_egad_b111", "pick_single_egad_b111")
class PickSingleEgadB111Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/B11_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/B11_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-B11_1_v2.pkl"


@register_task("maniskill.pick_single_egad_b112", "pick_single_egad_b112")
class PickSingleEgadB112Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/B11_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/B11_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-B11_2_v2.pkl"


@register_task("maniskill.pick_single_egad_b113", "pick_single_egad_b113")
class PickSingleEgadB113Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/B11_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/B11_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-B11_3_v2.pkl"


@register_task("maniskill.pick_single_egad_b121", "pick_single_egad_b121")
class PickSingleEgadB121Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/B12_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/B12_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-B12_1_v2.pkl"


@register_task("maniskill.pick_single_egad_b130", "pick_single_egad_b130")
class PickSingleEgadB130Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/B13_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/B13_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-B13_0_v2.pkl"


@register_task("maniskill.pick_single_egad_b131", "pick_single_egad_b131")
class PickSingleEgadB131Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/B13_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/B13_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-B13_1_v2.pkl"


@register_task("maniskill.pick_single_egad_b132", "pick_single_egad_b132")
class PickSingleEgadB132Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/B13_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/B13_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-B13_2_v2.pkl"


@register_task("maniskill.pick_single_egad_b133", "pick_single_egad_b133")
class PickSingleEgadB133Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/B13_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/B13_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-B13_3_v2.pkl"


@register_task("maniskill.pick_single_egad_b140", "pick_single_egad_b140")
class PickSingleEgadB140Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/B14_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/B14_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-B14_0_v2.pkl"


@register_task("maniskill.pick_single_egad_b141", "pick_single_egad_b141")
class PickSingleEgadB141Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/B14_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/B14_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-B14_1_v2.pkl"


@register_task("maniskill.pick_single_egad_b142", "pick_single_egad_b142")
class PickSingleEgadB142Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/B14_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/B14_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-B14_2_v2.pkl"


@register_task("maniskill.pick_single_egad_b143", "pick_single_egad_b143")
class PickSingleEgadB143Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/B14_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/B14_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-B14_3_v2.pkl"


@register_task("maniskill.pick_single_egad_b150", "pick_single_egad_b150")
class PickSingleEgadB150Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/B15_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/B15_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-B15_0_v2.pkl"


@register_task("maniskill.pick_single_egad_b151", "pick_single_egad_b151")
class PickSingleEgadB151Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/B15_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/B15_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-B15_1_v2.pkl"


@register_task("maniskill.pick_single_egad_b152", "pick_single_egad_b152")
class PickSingleEgadB152Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/B15_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/B15_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-B15_2_v2.pkl"


@register_task("maniskill.pick_single_egad_b153", "pick_single_egad_b153")
class PickSingleEgadB153Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/B15_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/B15_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-B15_3_v2.pkl"


@register_task("maniskill.pick_single_egad_b161", "pick_single_egad_b161")
class PickSingleEgadB161Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/B16_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/B16_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-B16_1_v2.pkl"


@register_task("maniskill.pick_single_egad_b162", "pick_single_egad_b162")
class PickSingleEgadB162Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/B16_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/B16_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-B16_2_v2.pkl"


@register_task("maniskill.pick_single_egad_b163", "pick_single_egad_b163")
class PickSingleEgadB163Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/B16_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/B16_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-B16_3_v2.pkl"


@register_task("maniskill.pick_single_egad_b170", "pick_single_egad_b170")
class PickSingleEgadB170Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/B17_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/B17_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-B17_0_v2.pkl"


@register_task("maniskill.pick_single_egad_b171", "pick_single_egad_b171")
class PickSingleEgadB171Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/B17_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/B17_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-B17_1_v2.pkl"


@register_task("maniskill.pick_single_egad_b172", "pick_single_egad_b172")
class PickSingleEgadB172Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/B17_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/B17_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-B17_2_v2.pkl"


@register_task("maniskill.pick_single_egad_b173", "pick_single_egad_b173")
class PickSingleEgadB173Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/B17_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/B17_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-B17_3_v2.pkl"


@register_task("maniskill.pick_single_egad_b180", "pick_single_egad_b180")
class PickSingleEgadB180Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/B18_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/B18_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-B18_0_v2.pkl"


@register_task("maniskill.pick_single_egad_b190", "pick_single_egad_b190")
class PickSingleEgadB190Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/B19_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/B19_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-B19_0_v2.pkl"


@register_task("maniskill.pick_single_egad_b192", "pick_single_egad_b192")
class PickSingleEgadB192Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/B19_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/B19_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-B19_2_v2.pkl"


@register_task("maniskill.pick_single_egad_b193", "pick_single_egad_b193")
class PickSingleEgadB193Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/B19_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/B19_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-B19_3_v2.pkl"


@register_task("maniskill.pick_single_egad_b200", "pick_single_egad_b200")
class PickSingleEgadB200Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/B20_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/B20_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-B20_0_v2.pkl"


@register_task("maniskill.pick_single_egad_b201", "pick_single_egad_b201")
class PickSingleEgadB201Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/B20_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/B20_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-B20_1_v2.pkl"


@register_task("maniskill.pick_single_egad_b202", "pick_single_egad_b202")
class PickSingleEgadB202Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/B20_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/B20_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-B20_2_v2.pkl"


@register_task("maniskill.pick_single_egad_b210", "pick_single_egad_b210")
class PickSingleEgadB210Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/B21_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/B21_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-B21_0_v2.pkl"


@register_task("maniskill.pick_single_egad_b211", "pick_single_egad_b211")
class PickSingleEgadB211Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/B21_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/B21_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-B21_1_v2.pkl"


@register_task("maniskill.pick_single_egad_b212", "pick_single_egad_b212")
class PickSingleEgadB212Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/B21_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/B21_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-B21_2_v2.pkl"


@register_task("maniskill.pick_single_egad_b213", "pick_single_egad_b213")
class PickSingleEgadB213Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/B21_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/B21_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-B21_3_v2.pkl"


@register_task("maniskill.pick_single_egad_b220", "pick_single_egad_b220")
class PickSingleEgadB220Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/B22_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/B22_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-B22_0_v2.pkl"


@register_task("maniskill.pick_single_egad_b221", "pick_single_egad_b221")
class PickSingleEgadB221Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/B22_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/B22_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-B22_1_v2.pkl"


@register_task("maniskill.pick_single_egad_b222", "pick_single_egad_b222")
class PickSingleEgadB222Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/B22_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/B22_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-B22_2_v2.pkl"


@register_task("maniskill.pick_single_egad_b223", "pick_single_egad_b223")
class PickSingleEgadB223Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/B22_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/B22_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-B22_3_v2.pkl"


@register_task("maniskill.pick_single_egad_b231", "pick_single_egad_b231")
class PickSingleEgadB231Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/B23_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/B23_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-B23_1_v2.pkl"


@register_task("maniskill.pick_single_egad_b232", "pick_single_egad_b232")
class PickSingleEgadB232Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/B23_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/B23_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-B23_2_v2.pkl"


@register_task("maniskill.pick_single_egad_b233", "pick_single_egad_b233")
class PickSingleEgadB233Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/B23_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/B23_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-B23_3_v2.pkl"


@register_task("maniskill.pick_single_egad_b240", "pick_single_egad_b240")
class PickSingleEgadB240Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/B24_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/B24_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-B24_0_v2.pkl"


@register_task("maniskill.pick_single_egad_b241", "pick_single_egad_b241")
class PickSingleEgadB241Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/B24_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/B24_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-B24_1_v2.pkl"


@register_task("maniskill.pick_single_egad_b242", "pick_single_egad_b242")
class PickSingleEgadB242Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/B24_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/B24_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-B24_2_v2.pkl"


@register_task("maniskill.pick_single_egad_b243", "pick_single_egad_b243")
class PickSingleEgadB243Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/B24_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/B24_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-B24_3_v2.pkl"


@register_task("maniskill.pick_single_egad_b250", "pick_single_egad_b250")
class PickSingleEgadB250Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/B25_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/B25_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-B25_0_v2.pkl"


@register_task("maniskill.pick_single_egad_b251", "pick_single_egad_b251")
class PickSingleEgadB251Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/B25_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/B25_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-B25_1_v2.pkl"


@register_task("maniskill.pick_single_egad_b252", "pick_single_egad_b252")
class PickSingleEgadB252Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/B25_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/B25_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-B25_2_v2.pkl"


@register_task("maniskill.pick_single_egad_b253", "pick_single_egad_b253")
class PickSingleEgadB253Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/B25_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/B25_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-B25_3_v2.pkl"


@register_task("maniskill.pick_single_egad_c100", "pick_single_egad_c100")
class PickSingleEgadC100Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/C10_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/C10_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-C10_0_v2.pkl"


@register_task("maniskill.pick_single_egad_c101", "pick_single_egad_c101")
class PickSingleEgadC101Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/C10_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/C10_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-C10_1_v2.pkl"


@register_task("maniskill.pick_single_egad_c102", "pick_single_egad_c102")
class PickSingleEgadC102Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/C10_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/C10_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-C10_2_v2.pkl"


@register_task("maniskill.pick_single_egad_c103", "pick_single_egad_c103")
class PickSingleEgadC103Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/C10_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/C10_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-C10_3_v2.pkl"


@register_task("maniskill.pick_single_egad_c110", "pick_single_egad_c110")
class PickSingleEgadC110Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/C11_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/C11_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-C11_0_v2.pkl"


@register_task("maniskill.pick_single_egad_c111", "pick_single_egad_c111")
class PickSingleEgadC111Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/C11_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/C11_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-C11_1_v2.pkl"


@register_task("maniskill.pick_single_egad_c113", "pick_single_egad_c113")
class PickSingleEgadC113Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/C11_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/C11_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-C11_3_v2.pkl"


@register_task("maniskill.pick_single_egad_c120", "pick_single_egad_c120")
class PickSingleEgadC120Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/C12_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/C12_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-C12_0_v2.pkl"


@register_task("maniskill.pick_single_egad_c121", "pick_single_egad_c121")
class PickSingleEgadC121Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/C12_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/C12_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-C12_1_v2.pkl"


@register_task("maniskill.pick_single_egad_c122", "pick_single_egad_c122")
class PickSingleEgadC122Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/C12_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/C12_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-C12_2_v2.pkl"


@register_task("maniskill.pick_single_egad_c123", "pick_single_egad_c123")
class PickSingleEgadC123Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/C12_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/C12_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-C12_3_v2.pkl"


@register_task("maniskill.pick_single_egad_c130", "pick_single_egad_c130")
class PickSingleEgadC130Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/C13_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/C13_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-C13_0_v2.pkl"


@register_task("maniskill.pick_single_egad_c131", "pick_single_egad_c131")
class PickSingleEgadC131Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/C13_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/C13_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-C13_1_v2.pkl"


@register_task("maniskill.pick_single_egad_c132", "pick_single_egad_c132")
class PickSingleEgadC132Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/C13_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/C13_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-C13_2_v2.pkl"


@register_task("maniskill.pick_single_egad_c133", "pick_single_egad_c133")
class PickSingleEgadC133Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/C13_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/C13_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-C13_3_v2.pkl"


@register_task("maniskill.pick_single_egad_c140", "pick_single_egad_c140")
class PickSingleEgadC140Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/C14_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/C14_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-C14_0_v2.pkl"


@register_task("maniskill.pick_single_egad_c142", "pick_single_egad_c142")
class PickSingleEgadC142Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/C14_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/C14_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-C14_2_v2.pkl"


@register_task("maniskill.pick_single_egad_c143", "pick_single_egad_c143")
class PickSingleEgadC143Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/C14_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/C14_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-C14_3_v2.pkl"


@register_task("maniskill.pick_single_egad_c150", "pick_single_egad_c150")
class PickSingleEgadC150Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/C15_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/C15_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-C15_0_v2.pkl"


@register_task("maniskill.pick_single_egad_c151", "pick_single_egad_c151")
class PickSingleEgadC151Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/C15_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/C15_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-C15_1_v2.pkl"


@register_task("maniskill.pick_single_egad_c152", "pick_single_egad_c152")
class PickSingleEgadC152Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/C15_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/C15_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-C15_2_v2.pkl"


@register_task("maniskill.pick_single_egad_c153", "pick_single_egad_c153")
class PickSingleEgadC153Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/C15_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/C15_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-C15_3_v2.pkl"


@register_task("maniskill.pick_single_egad_c161", "pick_single_egad_c161")
class PickSingleEgadC161Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/C16_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/C16_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-C16_1_v2.pkl"


@register_task("maniskill.pick_single_egad_c162", "pick_single_egad_c162")
class PickSingleEgadC162Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/C16_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/C16_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-C16_2_v2.pkl"


@register_task("maniskill.pick_single_egad_c163", "pick_single_egad_c163")
class PickSingleEgadC163Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/C16_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/C16_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-C16_3_v2.pkl"


@register_task("maniskill.pick_single_egad_c170", "pick_single_egad_c170")
class PickSingleEgadC170Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/C17_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/C17_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-C17_0_v2.pkl"


@register_task("maniskill.pick_single_egad_c171", "pick_single_egad_c171")
class PickSingleEgadC171Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/C17_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/C17_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-C17_1_v2.pkl"


@register_task("maniskill.pick_single_egad_c172", "pick_single_egad_c172")
class PickSingleEgadC172Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/C17_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/C17_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-C17_2_v2.pkl"


@register_task("maniskill.pick_single_egad_c173", "pick_single_egad_c173")
class PickSingleEgadC173Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/C17_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/C17_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-C17_3_v2.pkl"


@register_task("maniskill.pick_single_egad_c180", "pick_single_egad_c180")
class PickSingleEgadC180Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/C18_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/C18_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-C18_0_v2.pkl"


@register_task("maniskill.pick_single_egad_c181", "pick_single_egad_c181")
class PickSingleEgadC181Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/C18_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/C18_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-C18_1_v2.pkl"


@register_task("maniskill.pick_single_egad_c182", "pick_single_egad_c182")
class PickSingleEgadC182Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/C18_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/C18_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-C18_2_v2.pkl"


@register_task("maniskill.pick_single_egad_c183", "pick_single_egad_c183")
class PickSingleEgadC183Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/C18_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/C18_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-C18_3_v2.pkl"


@register_task("maniskill.pick_single_egad_c190", "pick_single_egad_c190")
class PickSingleEgadC190Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/C19_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/C19_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-C19_0_v2.pkl"


@register_task("maniskill.pick_single_egad_c191", "pick_single_egad_c191")
class PickSingleEgadC191Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/C19_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/C19_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-C19_1_v2.pkl"


@register_task("maniskill.pick_single_egad_c192", "pick_single_egad_c192")
class PickSingleEgadC192Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/C19_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/C19_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-C19_2_v2.pkl"


@register_task("maniskill.pick_single_egad_c193", "pick_single_egad_c193")
class PickSingleEgadC193Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/C19_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/C19_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-C19_3_v2.pkl"


@register_task("maniskill.pick_single_egad_c200", "pick_single_egad_c200")
class PickSingleEgadC200Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/C20_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/C20_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-C20_0_v2.pkl"


@register_task("maniskill.pick_single_egad_c201", "pick_single_egad_c201")
class PickSingleEgadC201Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/C20_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/C20_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-C20_1_v2.pkl"


@register_task("maniskill.pick_single_egad_c202", "pick_single_egad_c202")
class PickSingleEgadC202Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/C20_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/C20_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-C20_2_v2.pkl"


@register_task("maniskill.pick_single_egad_c203", "pick_single_egad_c203")
class PickSingleEgadC203Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/C20_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/C20_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-C20_3_v2.pkl"


@register_task("maniskill.pick_single_egad_c210", "pick_single_egad_c210")
class PickSingleEgadC210Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/C21_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/C21_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-C21_0_v2.pkl"


@register_task("maniskill.pick_single_egad_c211", "pick_single_egad_c211")
class PickSingleEgadC211Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/C21_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/C21_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-C21_1_v2.pkl"


@register_task("maniskill.pick_single_egad_c212", "pick_single_egad_c212")
class PickSingleEgadC212Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/C21_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/C21_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-C21_2_v2.pkl"


@register_task("maniskill.pick_single_egad_c213", "pick_single_egad_c213")
class PickSingleEgadC213Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/C21_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/C21_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-C21_3_v2.pkl"


@register_task("maniskill.pick_single_egad_c220", "pick_single_egad_c220")
class PickSingleEgadC220Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/C22_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/C22_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-C22_0_v2.pkl"


@register_task("maniskill.pick_single_egad_c221", "pick_single_egad_c221")
class PickSingleEgadC221Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/C22_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/C22_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-C22_1_v2.pkl"


@register_task("maniskill.pick_single_egad_c223", "pick_single_egad_c223")
class PickSingleEgadC223Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/C22_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/C22_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-C22_3_v2.pkl"


@register_task("maniskill.pick_single_egad_c230", "pick_single_egad_c230")
class PickSingleEgadC230Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/C23_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/C23_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-C23_0_v2.pkl"


@register_task("maniskill.pick_single_egad_c231", "pick_single_egad_c231")
class PickSingleEgadC231Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/C23_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/C23_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-C23_1_v2.pkl"


@register_task("maniskill.pick_single_egad_c232", "pick_single_egad_c232")
class PickSingleEgadC232Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/C23_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/C23_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-C23_2_v2.pkl"


@register_task("maniskill.pick_single_egad_c233", "pick_single_egad_c233")
class PickSingleEgadC233Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/C23_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/C23_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-C23_3_v2.pkl"


@register_task("maniskill.pick_single_egad_c240", "pick_single_egad_c240")
class PickSingleEgadC240Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/C24_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/C24_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-C24_0_v2.pkl"


@register_task("maniskill.pick_single_egad_c241", "pick_single_egad_c241")
class PickSingleEgadC241Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/C24_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/C24_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-C24_1_v2.pkl"


@register_task("maniskill.pick_single_egad_c242", "pick_single_egad_c242")
class PickSingleEgadC242Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/C24_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/C24_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-C24_2_v2.pkl"


@register_task("maniskill.pick_single_egad_c243", "pick_single_egad_c243")
class PickSingleEgadC243Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/C24_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/C24_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-C24_3_v2.pkl"


@register_task("maniskill.pick_single_egad_c250", "pick_single_egad_c250")
class PickSingleEgadC250Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/C25_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/C25_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-C25_0_v2.pkl"


@register_task("maniskill.pick_single_egad_c251", "pick_single_egad_c251")
class PickSingleEgadC251Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/C25_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/C25_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-C25_1_v2.pkl"


@register_task("maniskill.pick_single_egad_c252", "pick_single_egad_c252")
class PickSingleEgadC252Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/C25_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/C25_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-C25_2_v2.pkl"


@register_task("maniskill.pick_single_egad_c253", "pick_single_egad_c253")
class PickSingleEgadC253Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/C25_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/C25_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-C25_3_v2.pkl"


@register_task("maniskill.pick_single_egad_d100", "pick_single_egad_d100")
class PickSingleEgadD100Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/D10_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/D10_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-D10_0_v2.pkl"


@register_task("maniskill.pick_single_egad_d101", "pick_single_egad_d101")
class PickSingleEgadD101Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/D10_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/D10_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-D10_1_v2.pkl"


@register_task("maniskill.pick_single_egad_d102", "pick_single_egad_d102")
class PickSingleEgadD102Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/D10_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/D10_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-D10_2_v2.pkl"


@register_task("maniskill.pick_single_egad_d103", "pick_single_egad_d103")
class PickSingleEgadD103Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/D10_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/D10_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-D10_3_v2.pkl"


@register_task("maniskill.pick_single_egad_d110", "pick_single_egad_d110")
class PickSingleEgadD110Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/D11_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/D11_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-D11_0_v2.pkl"


@register_task("maniskill.pick_single_egad_d111", "pick_single_egad_d111")
class PickSingleEgadD111Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/D11_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/D11_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-D11_1_v2.pkl"


@register_task("maniskill.pick_single_egad_d112", "pick_single_egad_d112")
class PickSingleEgadD112Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/D11_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/D11_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-D11_2_v2.pkl"


@register_task("maniskill.pick_single_egad_d113", "pick_single_egad_d113")
class PickSingleEgadD113Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/D11_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/D11_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-D11_3_v2.pkl"


@register_task("maniskill.pick_single_egad_d121", "pick_single_egad_d121")
class PickSingleEgadD121Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/D12_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/D12_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-D12_1_v2.pkl"


@register_task("maniskill.pick_single_egad_d122", "pick_single_egad_d122")
class PickSingleEgadD122Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/D12_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/D12_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-D12_2_v2.pkl"


@register_task("maniskill.pick_single_egad_d130", "pick_single_egad_d130")
class PickSingleEgadD130Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/D13_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/D13_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-D13_0_v2.pkl"


@register_task("maniskill.pick_single_egad_d131", "pick_single_egad_d131")
class PickSingleEgadD131Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/D13_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/D13_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-D13_1_v2.pkl"


@register_task("maniskill.pick_single_egad_d132", "pick_single_egad_d132")
class PickSingleEgadD132Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/D13_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/D13_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-D13_2_v2.pkl"


@register_task("maniskill.pick_single_egad_d133", "pick_single_egad_d133")
class PickSingleEgadD133Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/D13_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/D13_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-D13_3_v2.pkl"


@register_task("maniskill.pick_single_egad_d141", "pick_single_egad_d141")
class PickSingleEgadD141Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/D14_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/D14_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-D14_1_v2.pkl"


@register_task("maniskill.pick_single_egad_d142", "pick_single_egad_d142")
class PickSingleEgadD142Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/D14_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/D14_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-D14_2_v2.pkl"


@register_task("maniskill.pick_single_egad_d150", "pick_single_egad_d150")
class PickSingleEgadD150Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/D15_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/D15_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-D15_0_v2.pkl"


@register_task("maniskill.pick_single_egad_d151", "pick_single_egad_d151")
class PickSingleEgadD151Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/D15_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/D15_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-D15_1_v2.pkl"


@register_task("maniskill.pick_single_egad_d152", "pick_single_egad_d152")
class PickSingleEgadD152Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/D15_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/D15_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-D15_2_v2.pkl"


@register_task("maniskill.pick_single_egad_d153", "pick_single_egad_d153")
class PickSingleEgadD153Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/D15_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/D15_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-D15_3_v2.pkl"


@register_task("maniskill.pick_single_egad_d160", "pick_single_egad_d160")
class PickSingleEgadD160Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/D16_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/D16_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-D16_0_v2.pkl"


@register_task("maniskill.pick_single_egad_d161", "pick_single_egad_d161")
class PickSingleEgadD161Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/D16_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/D16_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-D16_1_v2.pkl"


@register_task("maniskill.pick_single_egad_d162", "pick_single_egad_d162")
class PickSingleEgadD162Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/D16_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/D16_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-D16_2_v2.pkl"


@register_task("maniskill.pick_single_egad_d163", "pick_single_egad_d163")
class PickSingleEgadD163Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/D16_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/D16_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-D16_3_v2.pkl"


@register_task("maniskill.pick_single_egad_d170", "pick_single_egad_d170")
class PickSingleEgadD170Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/D17_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/D17_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-D17_0_v2.pkl"


@register_task("maniskill.pick_single_egad_d171", "pick_single_egad_d171")
class PickSingleEgadD171Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/D17_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/D17_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-D17_1_v2.pkl"


@register_task("maniskill.pick_single_egad_d172", "pick_single_egad_d172")
class PickSingleEgadD172Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/D17_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/D17_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-D17_2_v2.pkl"


@register_task("maniskill.pick_single_egad_d180", "pick_single_egad_d180")
class PickSingleEgadD180Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/D18_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/D18_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-D18_0_v2.pkl"


@register_task("maniskill.pick_single_egad_d181", "pick_single_egad_d181")
class PickSingleEgadD181Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/D18_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/D18_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-D18_1_v2.pkl"


@register_task("maniskill.pick_single_egad_d182", "pick_single_egad_d182")
class PickSingleEgadD182Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/D18_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/D18_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-D18_2_v2.pkl"


@register_task("maniskill.pick_single_egad_d183", "pick_single_egad_d183")
class PickSingleEgadD183Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/D18_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/D18_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-D18_3_v2.pkl"


@register_task("maniskill.pick_single_egad_d190", "pick_single_egad_d190")
class PickSingleEgadD190Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/D19_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/D19_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-D19_0_v2.pkl"


@register_task("maniskill.pick_single_egad_d191", "pick_single_egad_d191")
class PickSingleEgadD191Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/D19_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/D19_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-D19_1_v2.pkl"


@register_task("maniskill.pick_single_egad_d193", "pick_single_egad_d193")
class PickSingleEgadD193Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/D19_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/D19_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-D19_3_v2.pkl"


@register_task("maniskill.pick_single_egad_d200", "pick_single_egad_d200")
class PickSingleEgadD200Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/D20_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/D20_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-D20_0_v2.pkl"


@register_task("maniskill.pick_single_egad_d201", "pick_single_egad_d201")
class PickSingleEgadD201Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/D20_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/D20_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-D20_1_v2.pkl"


@register_task("maniskill.pick_single_egad_d202", "pick_single_egad_d202")
class PickSingleEgadD202Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/D20_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/D20_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-D20_2_v2.pkl"


@register_task("maniskill.pick_single_egad_d203", "pick_single_egad_d203")
class PickSingleEgadD203Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/D20_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/D20_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-D20_3_v2.pkl"


@register_task("maniskill.pick_single_egad_d210", "pick_single_egad_d210")
class PickSingleEgadD210Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/D21_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/D21_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-D21_0_v2.pkl"


@register_task("maniskill.pick_single_egad_d211", "pick_single_egad_d211")
class PickSingleEgadD211Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/D21_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/D21_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-D21_1_v2.pkl"


@register_task("maniskill.pick_single_egad_d212", "pick_single_egad_d212")
class PickSingleEgadD212Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/D21_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/D21_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-D21_2_v2.pkl"


@register_task("maniskill.pick_single_egad_d213", "pick_single_egad_d213")
class PickSingleEgadD213Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/D21_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/D21_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-D21_3_v2.pkl"


@register_task("maniskill.pick_single_egad_d220", "pick_single_egad_d220")
class PickSingleEgadD220Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/D22_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/D22_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-D22_0_v2.pkl"


@register_task("maniskill.pick_single_egad_d221", "pick_single_egad_d221")
class PickSingleEgadD221Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/D22_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/D22_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-D22_1_v2.pkl"


@register_task("maniskill.pick_single_egad_d222", "pick_single_egad_d222")
class PickSingleEgadD222Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/D22_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/D22_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-D22_2_v2.pkl"


@register_task("maniskill.pick_single_egad_d223", "pick_single_egad_d223")
class PickSingleEgadD223Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/D22_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/D22_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-D22_3_v2.pkl"


@register_task("maniskill.pick_single_egad_d230", "pick_single_egad_d230")
class PickSingleEgadD230Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/D23_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/D23_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-D23_0_v2.pkl"


@register_task("maniskill.pick_single_egad_d231", "pick_single_egad_d231")
class PickSingleEgadD231Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/D23_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/D23_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-D23_1_v2.pkl"


@register_task("maniskill.pick_single_egad_d232", "pick_single_egad_d232")
class PickSingleEgadD232Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/D23_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/D23_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-D23_2_v2.pkl"


@register_task("maniskill.pick_single_egad_d233", "pick_single_egad_d233")
class PickSingleEgadD233Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/D23_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/D23_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-D23_3_v2.pkl"


@register_task("maniskill.pick_single_egad_d240", "pick_single_egad_d240")
class PickSingleEgadD240Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/D24_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/D24_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-D24_0_v2.pkl"


@register_task("maniskill.pick_single_egad_d241", "pick_single_egad_d241")
class PickSingleEgadD241Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/D24_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/D24_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-D24_1_v2.pkl"


@register_task("maniskill.pick_single_egad_d242", "pick_single_egad_d242")
class PickSingleEgadD242Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/D24_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/D24_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-D24_2_v2.pkl"


@register_task("maniskill.pick_single_egad_d243", "pick_single_egad_d243")
class PickSingleEgadD243Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/D24_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/D24_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-D24_3_v2.pkl"


@register_task("maniskill.pick_single_egad_d250", "pick_single_egad_d250")
class PickSingleEgadD250Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/D25_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/D25_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-D25_0_v2.pkl"


@register_task("maniskill.pick_single_egad_d251", "pick_single_egad_d251")
class PickSingleEgadD251Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/D25_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/D25_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-D25_1_v2.pkl"


@register_task("maniskill.pick_single_egad_d252", "pick_single_egad_d252")
class PickSingleEgadD252Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/D25_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/D25_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-D25_2_v2.pkl"


@register_task("maniskill.pick_single_egad_d253", "pick_single_egad_d253")
class PickSingleEgadD253Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/D25_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/D25_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-D25_3_v2.pkl"


@register_task("maniskill.pick_single_egad_e100", "pick_single_egad_e100")
class PickSingleEgadE100Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/E10_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/E10_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-E10_0_v2.pkl"


@register_task("maniskill.pick_single_egad_e101", "pick_single_egad_e101")
class PickSingleEgadE101Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/E10_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/E10_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-E10_1_v2.pkl"


@register_task("maniskill.pick_single_egad_e102", "pick_single_egad_e102")
class PickSingleEgadE102Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/E10_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/E10_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-E10_2_v2.pkl"


@register_task("maniskill.pick_single_egad_e103", "pick_single_egad_e103")
class PickSingleEgadE103Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/E10_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/E10_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-E10_3_v2.pkl"


@register_task("maniskill.pick_single_egad_e111", "pick_single_egad_e111")
class PickSingleEgadE111Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/E11_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/E11_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-E11_1_v2.pkl"


@register_task("maniskill.pick_single_egad_e112", "pick_single_egad_e112")
class PickSingleEgadE112Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/E11_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/E11_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-E11_2_v2.pkl"


@register_task("maniskill.pick_single_egad_e113", "pick_single_egad_e113")
class PickSingleEgadE113Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/E11_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/E11_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-E11_3_v2.pkl"


@register_task("maniskill.pick_single_egad_e120", "pick_single_egad_e120")
class PickSingleEgadE120Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/E12_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/E12_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-E12_0_v2.pkl"


@register_task("maniskill.pick_single_egad_e121", "pick_single_egad_e121")
class PickSingleEgadE121Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/E12_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/E12_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-E12_1_v2.pkl"


@register_task("maniskill.pick_single_egad_e122", "pick_single_egad_e122")
class PickSingleEgadE122Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/E12_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/E12_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-E12_2_v2.pkl"


@register_task("maniskill.pick_single_egad_e123", "pick_single_egad_e123")
class PickSingleEgadE123Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/E12_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/E12_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-E12_3_v2.pkl"


@register_task("maniskill.pick_single_egad_e131", "pick_single_egad_e131")
class PickSingleEgadE131Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/E13_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/E13_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-E13_1_v2.pkl"


@register_task("maniskill.pick_single_egad_e132", "pick_single_egad_e132")
class PickSingleEgadE132Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/E13_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/E13_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-E13_2_v2.pkl"


@register_task("maniskill.pick_single_egad_e133", "pick_single_egad_e133")
class PickSingleEgadE133Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/E13_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/E13_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-E13_3_v2.pkl"


@register_task("maniskill.pick_single_egad_e140", "pick_single_egad_e140")
class PickSingleEgadE140Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/E14_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/E14_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-E14_0_v2.pkl"


@register_task("maniskill.pick_single_egad_e141", "pick_single_egad_e141")
class PickSingleEgadE141Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/E14_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/E14_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-E14_1_v2.pkl"


@register_task("maniskill.pick_single_egad_e142", "pick_single_egad_e142")
class PickSingleEgadE142Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/E14_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/E14_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-E14_2_v2.pkl"


@register_task("maniskill.pick_single_egad_e143", "pick_single_egad_e143")
class PickSingleEgadE143Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/E14_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/E14_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-E14_3_v2.pkl"


@register_task("maniskill.pick_single_egad_e150", "pick_single_egad_e150")
class PickSingleEgadE150Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/E15_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/E15_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-E15_0_v2.pkl"


@register_task("maniskill.pick_single_egad_e151", "pick_single_egad_e151")
class PickSingleEgadE151Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/E15_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/E15_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-E15_1_v2.pkl"


@register_task("maniskill.pick_single_egad_e152", "pick_single_egad_e152")
class PickSingleEgadE152Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/E15_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/E15_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-E15_2_v2.pkl"


@register_task("maniskill.pick_single_egad_e153", "pick_single_egad_e153")
class PickSingleEgadE153Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/E15_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/E15_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-E15_3_v2.pkl"


@register_task("maniskill.pick_single_egad_e160", "pick_single_egad_e160")
class PickSingleEgadE160Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/E16_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/E16_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-E16_0_v2.pkl"


@register_task("maniskill.pick_single_egad_e161", "pick_single_egad_e161")
class PickSingleEgadE161Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/E16_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/E16_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-E16_1_v2.pkl"


@register_task("maniskill.pick_single_egad_e162", "pick_single_egad_e162")
class PickSingleEgadE162Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/E16_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/E16_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-E16_2_v2.pkl"


@register_task("maniskill.pick_single_egad_e163", "pick_single_egad_e163")
class PickSingleEgadE163Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/E16_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/E16_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-E16_3_v2.pkl"


@register_task("maniskill.pick_single_egad_e170", "pick_single_egad_e170")
class PickSingleEgadE170Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/E17_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/E17_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-E17_0_v2.pkl"


@register_task("maniskill.pick_single_egad_e171", "pick_single_egad_e171")
class PickSingleEgadE171Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/E17_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/E17_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-E17_1_v2.pkl"


@register_task("maniskill.pick_single_egad_e172", "pick_single_egad_e172")
class PickSingleEgadE172Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/E17_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/E17_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-E17_2_v2.pkl"


@register_task("maniskill.pick_single_egad_e181", "pick_single_egad_e181")
class PickSingleEgadE181Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/E18_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/E18_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-E18_1_v2.pkl"


@register_task("maniskill.pick_single_egad_e182", "pick_single_egad_e182")
class PickSingleEgadE182Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/E18_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/E18_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-E18_2_v2.pkl"


@register_task("maniskill.pick_single_egad_e190", "pick_single_egad_e190")
class PickSingleEgadE190Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/E19_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/E19_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-E19_0_v2.pkl"


@register_task("maniskill.pick_single_egad_e191", "pick_single_egad_e191")
class PickSingleEgadE191Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/E19_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/E19_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-E19_1_v2.pkl"


@register_task("maniskill.pick_single_egad_e192", "pick_single_egad_e192")
class PickSingleEgadE192Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/E19_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/E19_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-E19_2_v2.pkl"


@register_task("maniskill.pick_single_egad_e193", "pick_single_egad_e193")
class PickSingleEgadE193Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/E19_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/E19_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-E19_3_v2.pkl"


@register_task("maniskill.pick_single_egad_e200", "pick_single_egad_e200")
class PickSingleEgadE200Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/E20_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/E20_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-E20_0_v2.pkl"


@register_task("maniskill.pick_single_egad_e201", "pick_single_egad_e201")
class PickSingleEgadE201Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/E20_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/E20_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-E20_1_v2.pkl"


@register_task("maniskill.pick_single_egad_e202", "pick_single_egad_e202")
class PickSingleEgadE202Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/E20_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/E20_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-E20_2_v2.pkl"


@register_task("maniskill.pick_single_egad_e210", "pick_single_egad_e210")
class PickSingleEgadE210Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/E21_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/E21_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-E21_0_v2.pkl"


@register_task("maniskill.pick_single_egad_e211", "pick_single_egad_e211")
class PickSingleEgadE211Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/E21_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/E21_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-E21_1_v2.pkl"


@register_task("maniskill.pick_single_egad_e212", "pick_single_egad_e212")
class PickSingleEgadE212Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/E21_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/E21_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-E21_2_v2.pkl"


@register_task("maniskill.pick_single_egad_e213", "pick_single_egad_e213")
class PickSingleEgadE213Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/E21_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/E21_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-E21_3_v2.pkl"


@register_task("maniskill.pick_single_egad_e220", "pick_single_egad_e220")
class PickSingleEgadE220Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/E22_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/E22_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-E22_0_v2.pkl"


@register_task("maniskill.pick_single_egad_e221", "pick_single_egad_e221")
class PickSingleEgadE221Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/E22_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/E22_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-E22_1_v2.pkl"


@register_task("maniskill.pick_single_egad_e222", "pick_single_egad_e222")
class PickSingleEgadE222Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/E22_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/E22_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-E22_2_v2.pkl"


@register_task("maniskill.pick_single_egad_e223", "pick_single_egad_e223")
class PickSingleEgadE223Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/E22_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/E22_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-E22_3_v2.pkl"


@register_task("maniskill.pick_single_egad_e230", "pick_single_egad_e230")
class PickSingleEgadE230Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/E23_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/E23_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-E23_0_v2.pkl"


@register_task("maniskill.pick_single_egad_e231", "pick_single_egad_e231")
class PickSingleEgadE231Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/E23_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/E23_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-E23_1_v2.pkl"


@register_task("maniskill.pick_single_egad_e232", "pick_single_egad_e232")
class PickSingleEgadE232Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/E23_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/E23_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-E23_2_v2.pkl"


@register_task("maniskill.pick_single_egad_e233", "pick_single_egad_e233")
class PickSingleEgadE233Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/E23_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/E23_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-E23_3_v2.pkl"


@register_task("maniskill.pick_single_egad_e240", "pick_single_egad_e240")
class PickSingleEgadE240Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/E24_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/E24_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-E24_0_v2.pkl"


@register_task("maniskill.pick_single_egad_e241", "pick_single_egad_e241")
class PickSingleEgadE241Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/E24_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/E24_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-E24_1_v2.pkl"


@register_task("maniskill.pick_single_egad_e242", "pick_single_egad_e242")
class PickSingleEgadE242Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/E24_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/E24_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-E24_2_v2.pkl"


@register_task("maniskill.pick_single_egad_e243", "pick_single_egad_e243")
class PickSingleEgadE243Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/E24_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/E24_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-E24_3_v2.pkl"


@register_task("maniskill.pick_single_egad_e250", "pick_single_egad_e250")
class PickSingleEgadE250Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/E25_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/E25_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-E25_0_v2.pkl"


@register_task("maniskill.pick_single_egad_e251", "pick_single_egad_e251")
class PickSingleEgadE251Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/E25_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/E25_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-E25_1_v2.pkl"


@register_task("maniskill.pick_single_egad_e252", "pick_single_egad_e252")
class PickSingleEgadE252Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/E25_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/E25_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-E25_2_v2.pkl"


@register_task("maniskill.pick_single_egad_e253", "pick_single_egad_e253")
class PickSingleEgadE253Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/E25_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/E25_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-E25_3_v2.pkl"


@register_task("maniskill.pick_single_egad_f100", "pick_single_egad_f100")
class PickSingleEgadF100Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/F10_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/F10_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-F10_0_v2.pkl"


@register_task("maniskill.pick_single_egad_f101", "pick_single_egad_f101")
class PickSingleEgadF101Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/F10_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/F10_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-F10_1_v2.pkl"


@register_task("maniskill.pick_single_egad_f103", "pick_single_egad_f103")
class PickSingleEgadF103Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/F10_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/F10_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-F10_3_v2.pkl"


@register_task("maniskill.pick_single_egad_f110", "pick_single_egad_f110")
class PickSingleEgadF110Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/F11_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/F11_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-F11_0_v2.pkl"


@register_task("maniskill.pick_single_egad_f111", "pick_single_egad_f111")
class PickSingleEgadF111Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/F11_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/F11_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-F11_1_v2.pkl"


@register_task("maniskill.pick_single_egad_f112", "pick_single_egad_f112")
class PickSingleEgadF112Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/F11_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/F11_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-F11_2_v2.pkl"


@register_task("maniskill.pick_single_egad_f113", "pick_single_egad_f113")
class PickSingleEgadF113Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/F11_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/F11_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-F11_3_v2.pkl"


@register_task("maniskill.pick_single_egad_f121", "pick_single_egad_f121")
class PickSingleEgadF121Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/F12_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/F12_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-F12_1_v2.pkl"


@register_task("maniskill.pick_single_egad_f122", "pick_single_egad_f122")
class PickSingleEgadF122Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/F12_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/F12_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-F12_2_v2.pkl"


@register_task("maniskill.pick_single_egad_f130", "pick_single_egad_f130")
class PickSingleEgadF130Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/F13_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/F13_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-F13_0_v2.pkl"


@register_task("maniskill.pick_single_egad_f131", "pick_single_egad_f131")
class PickSingleEgadF131Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/F13_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/F13_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-F13_1_v2.pkl"


@register_task("maniskill.pick_single_egad_f132", "pick_single_egad_f132")
class PickSingleEgadF132Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/F13_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/F13_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-F13_2_v2.pkl"


@register_task("maniskill.pick_single_egad_f133", "pick_single_egad_f133")
class PickSingleEgadF133Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/F13_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/F13_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-F13_3_v2.pkl"


@register_task("maniskill.pick_single_egad_f140", "pick_single_egad_f140")
class PickSingleEgadF140Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/F14_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/F14_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-F14_0_v2.pkl"


@register_task("maniskill.pick_single_egad_f142", "pick_single_egad_f142")
class PickSingleEgadF142Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/F14_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/F14_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-F14_2_v2.pkl"


@register_task("maniskill.pick_single_egad_f143", "pick_single_egad_f143")
class PickSingleEgadF143Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/F14_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/F14_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-F14_3_v2.pkl"


@register_task("maniskill.pick_single_egad_f150", "pick_single_egad_f150")
class PickSingleEgadF150Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/F15_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/F15_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-F15_0_v2.pkl"


@register_task("maniskill.pick_single_egad_f151", "pick_single_egad_f151")
class PickSingleEgadF151Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/F15_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/F15_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-F15_1_v2.pkl"


@register_task("maniskill.pick_single_egad_f152", "pick_single_egad_f152")
class PickSingleEgadF152Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/F15_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/F15_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-F15_2_v2.pkl"


@register_task("maniskill.pick_single_egad_f153", "pick_single_egad_f153")
class PickSingleEgadF153Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/F15_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/F15_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-F15_3_v2.pkl"


@register_task("maniskill.pick_single_egad_f160", "pick_single_egad_f160")
class PickSingleEgadF160Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/F16_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/F16_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-F16_0_v2.pkl"


@register_task("maniskill.pick_single_egad_f161", "pick_single_egad_f161")
class PickSingleEgadF161Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/F16_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/F16_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-F16_1_v2.pkl"


@register_task("maniskill.pick_single_egad_f162", "pick_single_egad_f162")
class PickSingleEgadF162Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/F16_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/F16_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-F16_2_v2.pkl"


@register_task("maniskill.pick_single_egad_f163", "pick_single_egad_f163")
class PickSingleEgadF163Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/F16_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/F16_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-F16_3_v2.pkl"


@register_task("maniskill.pick_single_egad_f170", "pick_single_egad_f170")
class PickSingleEgadF170Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/F17_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/F17_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-F17_0_v2.pkl"


@register_task("maniskill.pick_single_egad_f171", "pick_single_egad_f171")
class PickSingleEgadF171Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/F17_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/F17_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-F17_1_v2.pkl"


@register_task("maniskill.pick_single_egad_f172", "pick_single_egad_f172")
class PickSingleEgadF172Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/F17_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/F17_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-F17_2_v2.pkl"


@register_task("maniskill.pick_single_egad_f173", "pick_single_egad_f173")
class PickSingleEgadF173Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/F17_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/F17_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-F17_3_v2.pkl"


@register_task("maniskill.pick_single_egad_f180", "pick_single_egad_f180")
class PickSingleEgadF180Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/F18_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/F18_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-F18_0_v2.pkl"


@register_task("maniskill.pick_single_egad_f181", "pick_single_egad_f181")
class PickSingleEgadF181Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/F18_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/F18_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-F18_1_v2.pkl"


@register_task("maniskill.pick_single_egad_f182", "pick_single_egad_f182")
class PickSingleEgadF182Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/F18_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/F18_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-F18_2_v2.pkl"


@register_task("maniskill.pick_single_egad_f183", "pick_single_egad_f183")
class PickSingleEgadF183Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/F18_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/F18_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-F18_3_v2.pkl"


@register_task("maniskill.pick_single_egad_f190", "pick_single_egad_f190")
class PickSingleEgadF190Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/F19_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/F19_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-F19_0_v2.pkl"


@register_task("maniskill.pick_single_egad_f191", "pick_single_egad_f191")
class PickSingleEgadF191Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/F19_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/F19_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-F19_1_v2.pkl"


@register_task("maniskill.pick_single_egad_f192", "pick_single_egad_f192")
class PickSingleEgadF192Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/F19_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/F19_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-F19_2_v2.pkl"


@register_task("maniskill.pick_single_egad_f193", "pick_single_egad_f193")
class PickSingleEgadF193Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/F19_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/F19_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-F19_3_v2.pkl"


@register_task("maniskill.pick_single_egad_f200", "pick_single_egad_f200")
class PickSingleEgadF200Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/F20_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/F20_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-F20_0_v2.pkl"


@register_task("maniskill.pick_single_egad_f202", "pick_single_egad_f202")
class PickSingleEgadF202Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/F20_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/F20_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-F20_2_v2.pkl"


@register_task("maniskill.pick_single_egad_f203", "pick_single_egad_f203")
class PickSingleEgadF203Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/F20_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/F20_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-F20_3_v2.pkl"


@register_task("maniskill.pick_single_egad_f210", "pick_single_egad_f210")
class PickSingleEgadF210Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/F21_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/F21_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-F21_0_v2.pkl"


@register_task("maniskill.pick_single_egad_f211", "pick_single_egad_f211")
class PickSingleEgadF211Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/F21_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/F21_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-F21_1_v2.pkl"


@register_task("maniskill.pick_single_egad_f212", "pick_single_egad_f212")
class PickSingleEgadF212Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/F21_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/F21_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-F21_2_v2.pkl"


@register_task("maniskill.pick_single_egad_f213", "pick_single_egad_f213")
class PickSingleEgadF213Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/F21_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/F21_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-F21_3_v2.pkl"


@register_task("maniskill.pick_single_egad_f220", "pick_single_egad_f220")
class PickSingleEgadF220Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/F22_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/F22_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-F22_0_v2.pkl"


@register_task("maniskill.pick_single_egad_f221", "pick_single_egad_f221")
class PickSingleEgadF221Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/F22_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/F22_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-F22_1_v2.pkl"


@register_task("maniskill.pick_single_egad_f222", "pick_single_egad_f222")
class PickSingleEgadF222Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/F22_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/F22_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-F22_2_v2.pkl"


@register_task("maniskill.pick_single_egad_f223", "pick_single_egad_f223")
class PickSingleEgadF223Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/F22_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/F22_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-F22_3_v2.pkl"


@register_task("maniskill.pick_single_egad_f230", "pick_single_egad_f230")
class PickSingleEgadF230Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/F23_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/F23_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-F23_0_v2.pkl"


@register_task("maniskill.pick_single_egad_f231", "pick_single_egad_f231")
class PickSingleEgadF231Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/F23_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/F23_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-F23_1_v2.pkl"


@register_task("maniskill.pick_single_egad_f232", "pick_single_egad_f232")
class PickSingleEgadF232Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/F23_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/F23_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-F23_2_v2.pkl"


@register_task("maniskill.pick_single_egad_f233", "pick_single_egad_f233")
class PickSingleEgadF233Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/F23_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/F23_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-F23_3_v2.pkl"


@register_task("maniskill.pick_single_egad_f240", "pick_single_egad_f240")
class PickSingleEgadF240Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/F24_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/F24_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-F24_0_v2.pkl"


@register_task("maniskill.pick_single_egad_f241", "pick_single_egad_f241")
class PickSingleEgadF241Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/F24_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/F24_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-F24_1_v2.pkl"


@register_task("maniskill.pick_single_egad_f242", "pick_single_egad_f242")
class PickSingleEgadF242Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/F24_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/F24_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-F24_2_v2.pkl"


@register_task("maniskill.pick_single_egad_f243", "pick_single_egad_f243")
class PickSingleEgadF243Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/F24_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/F24_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-F24_3_v2.pkl"


@register_task("maniskill.pick_single_egad_f250", "pick_single_egad_f250")
class PickSingleEgadF250Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/F25_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/F25_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-F25_0_v2.pkl"


@register_task("maniskill.pick_single_egad_f251", "pick_single_egad_f251")
class PickSingleEgadF251Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/F25_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/F25_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-F25_1_v2.pkl"


@register_task("maniskill.pick_single_egad_f252", "pick_single_egad_f252")
class PickSingleEgadF252Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/F25_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/F25_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-F25_2_v2.pkl"


@register_task("maniskill.pick_single_egad_f253", "pick_single_egad_f253")
class PickSingleEgadF253Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/F25_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/F25_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-F25_3_v2.pkl"


@register_task("maniskill.pick_single_egad_g100", "pick_single_egad_g100")
class PickSingleEgadG100Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/G10_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/G10_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-G10_0_v2.pkl"


@register_task("maniskill.pick_single_egad_g101", "pick_single_egad_g101")
class PickSingleEgadG101Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/G10_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/G10_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-G10_1_v2.pkl"


@register_task("maniskill.pick_single_egad_g102", "pick_single_egad_g102")
class PickSingleEgadG102Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/G10_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/G10_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-G10_2_v2.pkl"


@register_task("maniskill.pick_single_egad_g103", "pick_single_egad_g103")
class PickSingleEgadG103Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/G10_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/G10_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-G10_3_v2.pkl"


@register_task("maniskill.pick_single_egad_g110", "pick_single_egad_g110")
class PickSingleEgadG110Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/G11_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/G11_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-G11_0_v2.pkl"


@register_task("maniskill.pick_single_egad_g111", "pick_single_egad_g111")
class PickSingleEgadG111Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/G11_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/G11_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-G11_1_v2.pkl"


@register_task("maniskill.pick_single_egad_g112", "pick_single_egad_g112")
class PickSingleEgadG112Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/G11_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/G11_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-G11_2_v2.pkl"


@register_task("maniskill.pick_single_egad_g113", "pick_single_egad_g113")
class PickSingleEgadG113Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/G11_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/G11_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-G11_3_v2.pkl"


@register_task("maniskill.pick_single_egad_g120", "pick_single_egad_g120")
class PickSingleEgadG120Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/G12_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/G12_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-G12_0_v2.pkl"


@register_task("maniskill.pick_single_egad_g122", "pick_single_egad_g122")
class PickSingleEgadG122Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/G12_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/G12_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-G12_2_v2.pkl"


@register_task("maniskill.pick_single_egad_g123", "pick_single_egad_g123")
class PickSingleEgadG123Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/G12_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/G12_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-G12_3_v2.pkl"


@register_task("maniskill.pick_single_egad_g130", "pick_single_egad_g130")
class PickSingleEgadG130Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/G13_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/G13_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-G13_0_v2.pkl"


@register_task("maniskill.pick_single_egad_g131", "pick_single_egad_g131")
class PickSingleEgadG131Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/G13_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/G13_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-G13_1_v2.pkl"


@register_task("maniskill.pick_single_egad_g132", "pick_single_egad_g132")
class PickSingleEgadG132Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/G13_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/G13_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-G13_2_v2.pkl"


@register_task("maniskill.pick_single_egad_g133", "pick_single_egad_g133")
class PickSingleEgadG133Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/G13_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/G13_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-G13_3_v2.pkl"


@register_task("maniskill.pick_single_egad_g140", "pick_single_egad_g140")
class PickSingleEgadG140Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/G14_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/G14_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-G14_0_v2.pkl"


@register_task("maniskill.pick_single_egad_g141", "pick_single_egad_g141")
class PickSingleEgadG141Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/G14_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/G14_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-G14_1_v2.pkl"


@register_task("maniskill.pick_single_egad_g142", "pick_single_egad_g142")
class PickSingleEgadG142Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/G14_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/G14_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-G14_2_v2.pkl"


@register_task("maniskill.pick_single_egad_g143", "pick_single_egad_g143")
class PickSingleEgadG143Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/G14_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/G14_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-G14_3_v2.pkl"


@register_task("maniskill.pick_single_egad_g150", "pick_single_egad_g150")
class PickSingleEgadG150Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/G15_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/G15_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-G15_0_v2.pkl"


@register_task("maniskill.pick_single_egad_g151", "pick_single_egad_g151")
class PickSingleEgadG151Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/G15_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/G15_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-G15_1_v2.pkl"


@register_task("maniskill.pick_single_egad_g152", "pick_single_egad_g152")
class PickSingleEgadG152Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/G15_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/G15_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-G15_2_v2.pkl"


@register_task("maniskill.pick_single_egad_g160", "pick_single_egad_g160")
class PickSingleEgadG160Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/G16_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/G16_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-G16_0_v2.pkl"


@register_task("maniskill.pick_single_egad_g161", "pick_single_egad_g161")
class PickSingleEgadG161Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/G16_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/G16_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-G16_1_v2.pkl"


@register_task("maniskill.pick_single_egad_g162", "pick_single_egad_g162")
class PickSingleEgadG162Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/G16_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/G16_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-G16_2_v2.pkl"


@register_task("maniskill.pick_single_egad_g163", "pick_single_egad_g163")
class PickSingleEgadG163Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/G16_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/G16_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-G16_3_v2.pkl"


@register_task("maniskill.pick_single_egad_g170", "pick_single_egad_g170")
class PickSingleEgadG170Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/G17_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/G17_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-G17_0_v2.pkl"


@register_task("maniskill.pick_single_egad_g171", "pick_single_egad_g171")
class PickSingleEgadG171Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/G17_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/G17_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-G17_1_v2.pkl"


@register_task("maniskill.pick_single_egad_g172", "pick_single_egad_g172")
class PickSingleEgadG172Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/G17_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/G17_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-G17_2_v2.pkl"


@register_task("maniskill.pick_single_egad_g173", "pick_single_egad_g173")
class PickSingleEgadG173Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/G17_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/G17_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-G17_3_v2.pkl"


@register_task("maniskill.pick_single_egad_g181", "pick_single_egad_g181")
class PickSingleEgadG181Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/G18_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/G18_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-G18_1_v2.pkl"


@register_task("maniskill.pick_single_egad_g182", "pick_single_egad_g182")
class PickSingleEgadG182Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/G18_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/G18_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-G18_2_v2.pkl"


@register_task("maniskill.pick_single_egad_g183", "pick_single_egad_g183")
class PickSingleEgadG183Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/G18_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/G18_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-G18_3_v2.pkl"


@register_task("maniskill.pick_single_egad_g191", "pick_single_egad_g191")
class PickSingleEgadG191Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/G19_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/G19_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-G19_1_v2.pkl"


@register_task("maniskill.pick_single_egad_g192", "pick_single_egad_g192")
class PickSingleEgadG192Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/G19_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/G19_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-G19_2_v2.pkl"


@register_task("maniskill.pick_single_egad_g193", "pick_single_egad_g193")
class PickSingleEgadG193Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/G19_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/G19_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-G19_3_v2.pkl"


@register_task("maniskill.pick_single_egad_g200", "pick_single_egad_g200")
class PickSingleEgadG200Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/G20_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/G20_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-G20_0_v2.pkl"


@register_task("maniskill.pick_single_egad_g201", "pick_single_egad_g201")
class PickSingleEgadG201Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/G20_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/G20_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-G20_1_v2.pkl"


@register_task("maniskill.pick_single_egad_g202", "pick_single_egad_g202")
class PickSingleEgadG202Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/G20_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/G20_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-G20_2_v2.pkl"


@register_task("maniskill.pick_single_egad_g203", "pick_single_egad_g203")
class PickSingleEgadG203Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/G20_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/G20_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-G20_3_v2.pkl"


@register_task("maniskill.pick_single_egad_g210", "pick_single_egad_g210")
class PickSingleEgadG210Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/G21_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/G21_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-G21_0_v2.pkl"


@register_task("maniskill.pick_single_egad_g211", "pick_single_egad_g211")
class PickSingleEgadG211Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/G21_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/G21_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-G21_1_v2.pkl"


@register_task("maniskill.pick_single_egad_g213", "pick_single_egad_g213")
class PickSingleEgadG213Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/G21_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/G21_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-G21_3_v2.pkl"


@register_task("maniskill.pick_single_egad_g220", "pick_single_egad_g220")
class PickSingleEgadG220Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/G22_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/G22_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-G22_0_v2.pkl"


@register_task("maniskill.pick_single_egad_g221", "pick_single_egad_g221")
class PickSingleEgadG221Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/G22_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/G22_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-G22_1_v2.pkl"


@register_task("maniskill.pick_single_egad_g222", "pick_single_egad_g222")
class PickSingleEgadG222Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/G22_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/G22_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-G22_2_v2.pkl"


@register_task("maniskill.pick_single_egad_g223", "pick_single_egad_g223")
class PickSingleEgadG223Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/G22_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/G22_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-G22_3_v2.pkl"


@register_task("maniskill.pick_single_egad_g230", "pick_single_egad_g230")
class PickSingleEgadG230Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/G23_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/G23_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-G23_0_v2.pkl"


@register_task("maniskill.pick_single_egad_g231", "pick_single_egad_g231")
class PickSingleEgadG231Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/G23_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/G23_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-G23_1_v2.pkl"


@register_task("maniskill.pick_single_egad_g233", "pick_single_egad_g233")
class PickSingleEgadG233Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/G23_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/G23_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-G23_3_v2.pkl"


@register_task("maniskill.pick_single_egad_g240", "pick_single_egad_g240")
class PickSingleEgadG240Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/G24_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/G24_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-G24_0_v2.pkl"


@register_task("maniskill.pick_single_egad_g241", "pick_single_egad_g241")
class PickSingleEgadG241Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/G24_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/G24_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-G24_1_v2.pkl"


@register_task("maniskill.pick_single_egad_g242", "pick_single_egad_g242")
class PickSingleEgadG242Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/G24_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/G24_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-G24_2_v2.pkl"


@register_task("maniskill.pick_single_egad_g243", "pick_single_egad_g243")
class PickSingleEgadG243Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/G24_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/G24_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-G24_3_v2.pkl"


@register_task("maniskill.pick_single_egad_g250", "pick_single_egad_g250")
class PickSingleEgadG250Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/G25_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/G25_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-G25_0_v2.pkl"


@register_task("maniskill.pick_single_egad_g251", "pick_single_egad_g251")
class PickSingleEgadG251Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/G25_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/G25_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-G25_1_v2.pkl"


@register_task("maniskill.pick_single_egad_g252", "pick_single_egad_g252")
class PickSingleEgadG252Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/G25_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/G25_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-G25_2_v2.pkl"


@register_task("maniskill.pick_single_egad_g253", "pick_single_egad_g253")
class PickSingleEgadG253Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/G25_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/G25_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-G25_3_v2.pkl"


@register_task("maniskill.pick_single_egad_h100", "pick_single_egad_h100")
class PickSingleEgadH100Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/H10_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/H10_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-H10_0_v2.pkl"


@register_task("maniskill.pick_single_egad_h101", "pick_single_egad_h101")
class PickSingleEgadH101Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/H10_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/H10_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-H10_1_v2.pkl"


@register_task("maniskill.pick_single_egad_h102", "pick_single_egad_h102")
class PickSingleEgadH102Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/H10_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/H10_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-H10_2_v2.pkl"


@register_task("maniskill.pick_single_egad_h103", "pick_single_egad_h103")
class PickSingleEgadH103Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/H10_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/H10_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-H10_3_v2.pkl"


@register_task("maniskill.pick_single_egad_h110", "pick_single_egad_h110")
class PickSingleEgadH110Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/H11_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/H11_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-H11_0_v2.pkl"


@register_task("maniskill.pick_single_egad_h111", "pick_single_egad_h111")
class PickSingleEgadH111Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/H11_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/H11_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-H11_1_v2.pkl"


@register_task("maniskill.pick_single_egad_h112", "pick_single_egad_h112")
class PickSingleEgadH112Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/H11_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/H11_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-H11_2_v2.pkl"


@register_task("maniskill.pick_single_egad_h113", "pick_single_egad_h113")
class PickSingleEgadH113Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/H11_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/H11_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-H11_3_v2.pkl"


@register_task("maniskill.pick_single_egad_h120", "pick_single_egad_h120")
class PickSingleEgadH120Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/H12_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/H12_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-H12_0_v2.pkl"


@register_task("maniskill.pick_single_egad_h121", "pick_single_egad_h121")
class PickSingleEgadH121Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/H12_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/H12_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-H12_1_v2.pkl"


@register_task("maniskill.pick_single_egad_h122", "pick_single_egad_h122")
class PickSingleEgadH122Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/H12_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/H12_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-H12_2_v2.pkl"


@register_task("maniskill.pick_single_egad_h123", "pick_single_egad_h123")
class PickSingleEgadH123Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/H12_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/H12_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-H12_3_v2.pkl"


@register_task("maniskill.pick_single_egad_h130", "pick_single_egad_h130")
class PickSingleEgadH130Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/H13_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/H13_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-H13_0_v2.pkl"


@register_task("maniskill.pick_single_egad_h131", "pick_single_egad_h131")
class PickSingleEgadH131Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/H13_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/H13_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-H13_1_v2.pkl"


@register_task("maniskill.pick_single_egad_h132", "pick_single_egad_h132")
class PickSingleEgadH132Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/H13_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/H13_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-H13_2_v2.pkl"


@register_task("maniskill.pick_single_egad_h140", "pick_single_egad_h140")
class PickSingleEgadH140Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/H14_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/H14_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-H14_0_v2.pkl"


@register_task("maniskill.pick_single_egad_h141", "pick_single_egad_h141")
class PickSingleEgadH141Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/H14_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/H14_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-H14_1_v2.pkl"


@register_task("maniskill.pick_single_egad_h142", "pick_single_egad_h142")
class PickSingleEgadH142Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/H14_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/H14_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-H14_2_v2.pkl"


@register_task("maniskill.pick_single_egad_h143", "pick_single_egad_h143")
class PickSingleEgadH143Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/H14_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/H14_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-H14_3_v2.pkl"


@register_task("maniskill.pick_single_egad_h150", "pick_single_egad_h150")
class PickSingleEgadH150Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/H15_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/H15_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-H15_0_v2.pkl"


@register_task("maniskill.pick_single_egad_h151", "pick_single_egad_h151")
class PickSingleEgadH151Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/H15_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/H15_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-H15_1_v2.pkl"


@register_task("maniskill.pick_single_egad_h152", "pick_single_egad_h152")
class PickSingleEgadH152Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/H15_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/H15_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-H15_2_v2.pkl"


@register_task("maniskill.pick_single_egad_h153", "pick_single_egad_h153")
class PickSingleEgadH153Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/H15_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/H15_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-H15_3_v2.pkl"


@register_task("maniskill.pick_single_egad_h160", "pick_single_egad_h160")
class PickSingleEgadH160Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/H16_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/H16_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-H16_0_v2.pkl"


@register_task("maniskill.pick_single_egad_h161", "pick_single_egad_h161")
class PickSingleEgadH161Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/H16_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/H16_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-H16_1_v2.pkl"


@register_task("maniskill.pick_single_egad_h162", "pick_single_egad_h162")
class PickSingleEgadH162Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/H16_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/H16_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-H16_2_v2.pkl"


@register_task("maniskill.pick_single_egad_h163", "pick_single_egad_h163")
class PickSingleEgadH163Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/H16_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/H16_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-H16_3_v2.pkl"


@register_task("maniskill.pick_single_egad_h170", "pick_single_egad_h170")
class PickSingleEgadH170Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/H17_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/H17_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-H17_0_v2.pkl"


@register_task("maniskill.pick_single_egad_h171", "pick_single_egad_h171")
class PickSingleEgadH171Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/H17_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/H17_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-H17_1_v2.pkl"


@register_task("maniskill.pick_single_egad_h172", "pick_single_egad_h172")
class PickSingleEgadH172Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/H17_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/H17_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-H17_2_v2.pkl"


@register_task("maniskill.pick_single_egad_h173", "pick_single_egad_h173")
class PickSingleEgadH173Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/H17_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/H17_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-H17_3_v2.pkl"


@register_task("maniskill.pick_single_egad_h181", "pick_single_egad_h181")
class PickSingleEgadH181Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/H18_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/H18_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-H18_1_v2.pkl"


@register_task("maniskill.pick_single_egad_h182", "pick_single_egad_h182")
class PickSingleEgadH182Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/H18_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/H18_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-H18_2_v2.pkl"


@register_task("maniskill.pick_single_egad_h183", "pick_single_egad_h183")
class PickSingleEgadH183Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/H18_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/H18_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-H18_3_v2.pkl"


@register_task("maniskill.pick_single_egad_h190", "pick_single_egad_h190")
class PickSingleEgadH190Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/H19_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/H19_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-H19_0_v2.pkl"


@register_task("maniskill.pick_single_egad_h191", "pick_single_egad_h191")
class PickSingleEgadH191Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/H19_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/H19_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-H19_1_v2.pkl"


@register_task("maniskill.pick_single_egad_h192", "pick_single_egad_h192")
class PickSingleEgadH192Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/H19_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/H19_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-H19_2_v2.pkl"


@register_task("maniskill.pick_single_egad_h193", "pick_single_egad_h193")
class PickSingleEgadH193Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/H19_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/H19_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-H19_3_v2.pkl"


@register_task("maniskill.pick_single_egad_h200", "pick_single_egad_h200")
class PickSingleEgadH200Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/H20_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/H20_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-H20_0_v2.pkl"


@register_task("maniskill.pick_single_egad_h201", "pick_single_egad_h201")
class PickSingleEgadH201Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/H20_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/H20_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-H20_1_v2.pkl"


@register_task("maniskill.pick_single_egad_h202", "pick_single_egad_h202")
class PickSingleEgadH202Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/H20_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/H20_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-H20_2_v2.pkl"


@register_task("maniskill.pick_single_egad_h203", "pick_single_egad_h203")
class PickSingleEgadH203Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/H20_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/H20_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-H20_3_v2.pkl"


@register_task("maniskill.pick_single_egad_h210", "pick_single_egad_h210")
class PickSingleEgadH210Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/H21_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/H21_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-H21_0_v2.pkl"


@register_task("maniskill.pick_single_egad_h211", "pick_single_egad_h211")
class PickSingleEgadH211Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/H21_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/H21_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-H21_1_v2.pkl"


@register_task("maniskill.pick_single_egad_h212", "pick_single_egad_h212")
class PickSingleEgadH212Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/H21_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/H21_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-H21_2_v2.pkl"


@register_task("maniskill.pick_single_egad_h220", "pick_single_egad_h220")
class PickSingleEgadH220Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/H22_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/H22_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-H22_0_v2.pkl"


@register_task("maniskill.pick_single_egad_h221", "pick_single_egad_h221")
class PickSingleEgadH221Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/H22_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/H22_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-H22_1_v2.pkl"


@register_task("maniskill.pick_single_egad_h222", "pick_single_egad_h222")
class PickSingleEgadH222Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/H22_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/H22_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-H22_2_v2.pkl"


@register_task("maniskill.pick_single_egad_h223", "pick_single_egad_h223")
class PickSingleEgadH223Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/H22_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/H22_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-H22_3_v2.pkl"


@register_task("maniskill.pick_single_egad_h230", "pick_single_egad_h230")
class PickSingleEgadH230Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/H23_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/H23_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-H23_0_v2.pkl"


@register_task("maniskill.pick_single_egad_h231", "pick_single_egad_h231")
class PickSingleEgadH231Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/H23_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/H23_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-H23_1_v2.pkl"


@register_task("maniskill.pick_single_egad_h240", "pick_single_egad_h240")
class PickSingleEgadH240Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/H24_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/H24_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-H24_0_v2.pkl"


@register_task("maniskill.pick_single_egad_h241", "pick_single_egad_h241")
class PickSingleEgadH241Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/H24_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/H24_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-H24_1_v2.pkl"


@register_task("maniskill.pick_single_egad_h242", "pick_single_egad_h242")
class PickSingleEgadH242Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/H24_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/H24_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-H24_2_v2.pkl"


@register_task("maniskill.pick_single_egad_h243", "pick_single_egad_h243")
class PickSingleEgadH243Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/H24_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/H24_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-H24_3_v2.pkl"


@register_task("maniskill.pick_single_egad_h250", "pick_single_egad_h250")
class PickSingleEgadH250Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/H25_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/H25_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-H25_0_v2.pkl"


@register_task("maniskill.pick_single_egad_h251", "pick_single_egad_h251")
class PickSingleEgadH251Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/H25_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/H25_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-H25_1_v2.pkl"


@register_task("maniskill.pick_single_egad_h252", "pick_single_egad_h252")
class PickSingleEgadH252Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/H25_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/H25_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-H25_2_v2.pkl"


@register_task("maniskill.pick_single_egad_h253", "pick_single_egad_h253")
class PickSingleEgadH253Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/H25_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/H25_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-H25_3_v2.pkl"


@register_task("maniskill.pick_single_egad_i070", "pick_single_egad_i070")
class PickSingleEgadI070Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/I07_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/I07_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I07_0_v2.pkl"


@register_task("maniskill.pick_single_egad_i071", "pick_single_egad_i071")
class PickSingleEgadI071Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/I07_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/I07_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I07_1_v2.pkl"


@register_task("maniskill.pick_single_egad_i072", "pick_single_egad_i072")
class PickSingleEgadI072Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/I07_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/I07_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I07_2_v2.pkl"


@register_task("maniskill.pick_single_egad_i073", "pick_single_egad_i073")
class PickSingleEgadI073Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/I07_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/I07_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I07_3_v2.pkl"


@register_task("maniskill.pick_single_egad_i080", "pick_single_egad_i080")
class PickSingleEgadI080Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/I08_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/I08_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I08_0_v2.pkl"


@register_task("maniskill.pick_single_egad_i081", "pick_single_egad_i081")
class PickSingleEgadI081Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/I08_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/I08_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I08_1_v2.pkl"


@register_task("maniskill.pick_single_egad_i083", "pick_single_egad_i083")
class PickSingleEgadI083Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/I08_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/I08_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I08_3_v2.pkl"


@register_task("maniskill.pick_single_egad_i090", "pick_single_egad_i090")
class PickSingleEgadI090Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/I09_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/I09_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I09_0_v2.pkl"


@register_task("maniskill.pick_single_egad_i091", "pick_single_egad_i091")
class PickSingleEgadI091Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/I09_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/I09_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I09_1_v2.pkl"


@register_task("maniskill.pick_single_egad_i092", "pick_single_egad_i092")
class PickSingleEgadI092Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/I09_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/I09_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I09_2_v2.pkl"


@register_task("maniskill.pick_single_egad_i102", "pick_single_egad_i102")
class PickSingleEgadI102Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/I10_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/I10_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I10_2_v2.pkl"


@register_task("maniskill.pick_single_egad_i103", "pick_single_egad_i103")
class PickSingleEgadI103Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/I10_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/I10_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I10_3_v2.pkl"


@register_task("maniskill.pick_single_egad_i110", "pick_single_egad_i110")
class PickSingleEgadI110Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/I11_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/I11_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I11_0_v2.pkl"


@register_task("maniskill.pick_single_egad_i111", "pick_single_egad_i111")
class PickSingleEgadI111Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/I11_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/I11_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I11_1_v2.pkl"


@register_task("maniskill.pick_single_egad_i112", "pick_single_egad_i112")
class PickSingleEgadI112Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/I11_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/I11_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I11_2_v2.pkl"


@register_task("maniskill.pick_single_egad_i113", "pick_single_egad_i113")
class PickSingleEgadI113Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/I11_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/I11_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I11_3_v2.pkl"


@register_task("maniskill.pick_single_egad_i120", "pick_single_egad_i120")
class PickSingleEgadI120Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/I12_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/I12_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I12_0_v2.pkl"


@register_task("maniskill.pick_single_egad_i121", "pick_single_egad_i121")
class PickSingleEgadI121Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/I12_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/I12_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I12_1_v2.pkl"


@register_task("maniskill.pick_single_egad_i122", "pick_single_egad_i122")
class PickSingleEgadI122Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/I12_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/I12_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I12_2_v2.pkl"


@register_task("maniskill.pick_single_egad_i123", "pick_single_egad_i123")
class PickSingleEgadI123Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/I12_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/I12_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I12_3_v2.pkl"


@register_task("maniskill.pick_single_egad_i130", "pick_single_egad_i130")
class PickSingleEgadI130Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/I13_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/I13_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I13_0_v2.pkl"


@register_task("maniskill.pick_single_egad_i131", "pick_single_egad_i131")
class PickSingleEgadI131Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/I13_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/I13_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I13_1_v2.pkl"


@register_task("maniskill.pick_single_egad_i132", "pick_single_egad_i132")
class PickSingleEgadI132Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/I13_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/I13_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I13_2_v2.pkl"


@register_task("maniskill.pick_single_egad_i133", "pick_single_egad_i133")
class PickSingleEgadI133Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/I13_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/I13_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I13_3_v2.pkl"


@register_task("maniskill.pick_single_egad_i140", "pick_single_egad_i140")
class PickSingleEgadI140Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/I14_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/I14_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I14_0_v2.pkl"


@register_task("maniskill.pick_single_egad_i141", "pick_single_egad_i141")
class PickSingleEgadI141Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/I14_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/I14_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I14_1_v2.pkl"


@register_task("maniskill.pick_single_egad_i142", "pick_single_egad_i142")
class PickSingleEgadI142Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/I14_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/I14_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I14_2_v2.pkl"


@register_task("maniskill.pick_single_egad_i143", "pick_single_egad_i143")
class PickSingleEgadI143Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/I14_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/I14_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I14_3_v2.pkl"


@register_task("maniskill.pick_single_egad_i150", "pick_single_egad_i150")
class PickSingleEgadI150Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/I15_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/I15_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I15_0_v2.pkl"


@register_task("maniskill.pick_single_egad_i151", "pick_single_egad_i151")
class PickSingleEgadI151Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/I15_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/I15_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I15_1_v2.pkl"


@register_task("maniskill.pick_single_egad_i152", "pick_single_egad_i152")
class PickSingleEgadI152Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/I15_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/I15_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I15_2_v2.pkl"


@register_task("maniskill.pick_single_egad_i153", "pick_single_egad_i153")
class PickSingleEgadI153Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/I15_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/I15_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I15_3_v2.pkl"


@register_task("maniskill.pick_single_egad_i160", "pick_single_egad_i160")
class PickSingleEgadI160Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/I16_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/I16_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I16_0_v2.pkl"


@register_task("maniskill.pick_single_egad_i161", "pick_single_egad_i161")
class PickSingleEgadI161Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/I16_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/I16_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I16_1_v2.pkl"


@register_task("maniskill.pick_single_egad_i162", "pick_single_egad_i162")
class PickSingleEgadI162Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/I16_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/I16_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I16_2_v2.pkl"


@register_task("maniskill.pick_single_egad_i163", "pick_single_egad_i163")
class PickSingleEgadI163Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/I16_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/I16_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I16_3_v2.pkl"


@register_task("maniskill.pick_single_egad_i170", "pick_single_egad_i170")
class PickSingleEgadI170Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/I17_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/I17_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I17_0_v2.pkl"


@register_task("maniskill.pick_single_egad_i171", "pick_single_egad_i171")
class PickSingleEgadI171Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/I17_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/I17_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I17_1_v2.pkl"


@register_task("maniskill.pick_single_egad_i172", "pick_single_egad_i172")
class PickSingleEgadI172Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/I17_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/I17_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I17_2_v2.pkl"


@register_task("maniskill.pick_single_egad_i173", "pick_single_egad_i173")
class PickSingleEgadI173Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/I17_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/I17_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I17_3_v2.pkl"


@register_task("maniskill.pick_single_egad_i180", "pick_single_egad_i180")
class PickSingleEgadI180Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/I18_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/I18_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I18_0_v2.pkl"


@register_task("maniskill.pick_single_egad_i181", "pick_single_egad_i181")
class PickSingleEgadI181Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/I18_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/I18_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I18_1_v2.pkl"


@register_task("maniskill.pick_single_egad_i182", "pick_single_egad_i182")
class PickSingleEgadI182Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/I18_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/I18_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I18_2_v2.pkl"


@register_task("maniskill.pick_single_egad_i183", "pick_single_egad_i183")
class PickSingleEgadI183Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/I18_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/I18_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I18_3_v2.pkl"


@register_task("maniskill.pick_single_egad_i190", "pick_single_egad_i190")
class PickSingleEgadI190Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/I19_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/I19_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I19_0_v2.pkl"


@register_task("maniskill.pick_single_egad_i191", "pick_single_egad_i191")
class PickSingleEgadI191Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/I19_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/I19_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I19_1_v2.pkl"


@register_task("maniskill.pick_single_egad_i192", "pick_single_egad_i192")
class PickSingleEgadI192Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/I19_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/I19_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I19_2_v2.pkl"


@register_task("maniskill.pick_single_egad_i200", "pick_single_egad_i200")
class PickSingleEgadI200Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/I20_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/I20_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I20_0_v2.pkl"


@register_task("maniskill.pick_single_egad_i201", "pick_single_egad_i201")
class PickSingleEgadI201Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/I20_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/I20_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I20_1_v2.pkl"


@register_task("maniskill.pick_single_egad_i203", "pick_single_egad_i203")
class PickSingleEgadI203Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/I20_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/I20_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I20_3_v2.pkl"


@register_task("maniskill.pick_single_egad_i210", "pick_single_egad_i210")
class PickSingleEgadI210Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/I21_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/I21_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I21_0_v2.pkl"


@register_task("maniskill.pick_single_egad_i211", "pick_single_egad_i211")
class PickSingleEgadI211Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/I21_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/I21_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I21_1_v2.pkl"


@register_task("maniskill.pick_single_egad_i213", "pick_single_egad_i213")
class PickSingleEgadI213Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/I21_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/I21_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I21_3_v2.pkl"


@register_task("maniskill.pick_single_egad_i220", "pick_single_egad_i220")
class PickSingleEgadI220Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/I22_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/I22_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I22_0_v2.pkl"


@register_task("maniskill.pick_single_egad_i221", "pick_single_egad_i221")
class PickSingleEgadI221Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/I22_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/I22_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I22_1_v2.pkl"


@register_task("maniskill.pick_single_egad_i223", "pick_single_egad_i223")
class PickSingleEgadI223Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/I22_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/I22_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I22_3_v2.pkl"


@register_task("maniskill.pick_single_egad_i230", "pick_single_egad_i230")
class PickSingleEgadI230Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/I23_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/I23_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I23_0_v2.pkl"


@register_task("maniskill.pick_single_egad_i232", "pick_single_egad_i232")
class PickSingleEgadI232Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/I23_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/I23_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I23_2_v2.pkl"


@register_task("maniskill.pick_single_egad_i233", "pick_single_egad_i233")
class PickSingleEgadI233Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/I23_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/I23_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I23_3_v2.pkl"


@register_task("maniskill.pick_single_egad_i240", "pick_single_egad_i240")
class PickSingleEgadI240Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/I24_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/I24_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I24_0_v2.pkl"


@register_task("maniskill.pick_single_egad_i241", "pick_single_egad_i241")
class PickSingleEgadI241Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/I24_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/I24_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I24_1_v2.pkl"


@register_task("maniskill.pick_single_egad_i242", "pick_single_egad_i242")
class PickSingleEgadI242Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/I24_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/I24_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I24_2_v2.pkl"


@register_task("maniskill.pick_single_egad_i243", "pick_single_egad_i243")
class PickSingleEgadI243Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/I24_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/I24_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I24_3_v2.pkl"


@register_task("maniskill.pick_single_egad_i250", "pick_single_egad_i250")
class PickSingleEgadI250Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/I25_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/I25_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I25_0_v2.pkl"


@register_task("maniskill.pick_single_egad_i251", "pick_single_egad_i251")
class PickSingleEgadI251Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/I25_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/I25_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I25_1_v2.pkl"


@register_task("maniskill.pick_single_egad_i252", "pick_single_egad_i252")
class PickSingleEgadI252Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/I25_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/I25_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I25_2_v2.pkl"


@register_task("maniskill.pick_single_egad_i253", "pick_single_egad_i253")
class PickSingleEgadI253Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/I25_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/I25_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I25_3_v2.pkl"


@register_task("maniskill.pick_single_egad_j070", "pick_single_egad_j070")
class PickSingleEgadJ070Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/J07_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/J07_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J07_0_v2.pkl"


@register_task("maniskill.pick_single_egad_j071", "pick_single_egad_j071")
class PickSingleEgadJ071Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/J07_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/J07_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J07_1_v2.pkl"


@register_task("maniskill.pick_single_egad_j072", "pick_single_egad_j072")
class PickSingleEgadJ072Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/J07_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/J07_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J07_2_v2.pkl"


@register_task("maniskill.pick_single_egad_j073", "pick_single_egad_j073")
class PickSingleEgadJ073Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/J07_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/J07_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J07_3_v2.pkl"


@register_task("maniskill.pick_single_egad_j080", "pick_single_egad_j080")
class PickSingleEgadJ080Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/J08_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/J08_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J08_0_v2.pkl"


@register_task("maniskill.pick_single_egad_j082", "pick_single_egad_j082")
class PickSingleEgadJ082Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/J08_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/J08_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J08_2_v2.pkl"


@register_task("maniskill.pick_single_egad_j083", "pick_single_egad_j083")
class PickSingleEgadJ083Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/J08_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/J08_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J08_3_v2.pkl"


@register_task("maniskill.pick_single_egad_j090", "pick_single_egad_j090")
class PickSingleEgadJ090Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/J09_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/J09_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J09_0_v2.pkl"


@register_task("maniskill.pick_single_egad_j091", "pick_single_egad_j091")
class PickSingleEgadJ091Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/J09_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/J09_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J09_1_v2.pkl"


@register_task("maniskill.pick_single_egad_j092", "pick_single_egad_j092")
class PickSingleEgadJ092Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/J09_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/J09_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J09_2_v2.pkl"


@register_task("maniskill.pick_single_egad_j100", "pick_single_egad_j100")
class PickSingleEgadJ100Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/J10_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/J10_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J10_0_v2.pkl"


@register_task("maniskill.pick_single_egad_j101", "pick_single_egad_j101")
class PickSingleEgadJ101Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/J10_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/J10_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J10_1_v2.pkl"


@register_task("maniskill.pick_single_egad_j102", "pick_single_egad_j102")
class PickSingleEgadJ102Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/J10_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/J10_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J10_2_v2.pkl"


@register_task("maniskill.pick_single_egad_j103", "pick_single_egad_j103")
class PickSingleEgadJ103Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/J10_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/J10_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J10_3_v2.pkl"


@register_task("maniskill.pick_single_egad_j110", "pick_single_egad_j110")
class PickSingleEgadJ110Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/J11_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/J11_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J11_0_v2.pkl"


@register_task("maniskill.pick_single_egad_j111", "pick_single_egad_j111")
class PickSingleEgadJ111Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/J11_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/J11_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J11_1_v2.pkl"


@register_task("maniskill.pick_single_egad_j112", "pick_single_egad_j112")
class PickSingleEgadJ112Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/J11_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/J11_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J11_2_v2.pkl"


@register_task("maniskill.pick_single_egad_j113", "pick_single_egad_j113")
class PickSingleEgadJ113Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/J11_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/J11_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J11_3_v2.pkl"


@register_task("maniskill.pick_single_egad_j120", "pick_single_egad_j120")
class PickSingleEgadJ120Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/J12_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/J12_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J12_0_v2.pkl"


@register_task("maniskill.pick_single_egad_j121", "pick_single_egad_j121")
class PickSingleEgadJ121Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/J12_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/J12_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J12_1_v2.pkl"


@register_task("maniskill.pick_single_egad_j122", "pick_single_egad_j122")
class PickSingleEgadJ122Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/J12_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/J12_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J12_2_v2.pkl"


@register_task("maniskill.pick_single_egad_j123", "pick_single_egad_j123")
class PickSingleEgadJ123Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/J12_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/J12_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J12_3_v2.pkl"


@register_task("maniskill.pick_single_egad_j130", "pick_single_egad_j130")
class PickSingleEgadJ130Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/J13_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/J13_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J13_0_v2.pkl"


@register_task("maniskill.pick_single_egad_j131", "pick_single_egad_j131")
class PickSingleEgadJ131Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/J13_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/J13_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J13_1_v2.pkl"


@register_task("maniskill.pick_single_egad_j132", "pick_single_egad_j132")
class PickSingleEgadJ132Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/J13_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/J13_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J13_2_v2.pkl"


@register_task("maniskill.pick_single_egad_j133", "pick_single_egad_j133")
class PickSingleEgadJ133Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/J13_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/J13_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J13_3_v2.pkl"


@register_task("maniskill.pick_single_egad_j140", "pick_single_egad_j140")
class PickSingleEgadJ140Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/J14_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/J14_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J14_0_v2.pkl"


@register_task("maniskill.pick_single_egad_j141", "pick_single_egad_j141")
class PickSingleEgadJ141Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/J14_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/J14_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J14_1_v2.pkl"


@register_task("maniskill.pick_single_egad_j142", "pick_single_egad_j142")
class PickSingleEgadJ142Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/J14_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/J14_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J14_2_v2.pkl"


@register_task("maniskill.pick_single_egad_j143", "pick_single_egad_j143")
class PickSingleEgadJ143Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/J14_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/J14_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J14_3_v2.pkl"


@register_task("maniskill.pick_single_egad_j150", "pick_single_egad_j150")
class PickSingleEgadJ150Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/J15_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/J15_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J15_0_v2.pkl"


@register_task("maniskill.pick_single_egad_j151", "pick_single_egad_j151")
class PickSingleEgadJ151Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/J15_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/J15_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J15_1_v2.pkl"


@register_task("maniskill.pick_single_egad_j152", "pick_single_egad_j152")
class PickSingleEgadJ152Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/J15_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/J15_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J15_2_v2.pkl"


@register_task("maniskill.pick_single_egad_j153", "pick_single_egad_j153")
class PickSingleEgadJ153Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/J15_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/J15_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J15_3_v2.pkl"


@register_task("maniskill.pick_single_egad_j160", "pick_single_egad_j160")
class PickSingleEgadJ160Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/J16_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/J16_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J16_0_v2.pkl"


@register_task("maniskill.pick_single_egad_j162", "pick_single_egad_j162")
class PickSingleEgadJ162Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/J16_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/J16_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J16_2_v2.pkl"


@register_task("maniskill.pick_single_egad_j163", "pick_single_egad_j163")
class PickSingleEgadJ163Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/J16_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/J16_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J16_3_v2.pkl"


@register_task("maniskill.pick_single_egad_j170", "pick_single_egad_j170")
class PickSingleEgadJ170Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/J17_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/J17_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J17_0_v2.pkl"


@register_task("maniskill.pick_single_egad_j171", "pick_single_egad_j171")
class PickSingleEgadJ171Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/J17_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/J17_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J17_1_v2.pkl"


@register_task("maniskill.pick_single_egad_j172", "pick_single_egad_j172")
class PickSingleEgadJ172Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/J17_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/J17_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J17_2_v2.pkl"


@register_task("maniskill.pick_single_egad_j173", "pick_single_egad_j173")
class PickSingleEgadJ173Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/J17_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/J17_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J17_3_v2.pkl"


@register_task("maniskill.pick_single_egad_j180", "pick_single_egad_j180")
class PickSingleEgadJ180Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/J18_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/J18_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J18_0_v2.pkl"


@register_task("maniskill.pick_single_egad_j181", "pick_single_egad_j181")
class PickSingleEgadJ181Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/J18_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/J18_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J18_1_v2.pkl"


@register_task("maniskill.pick_single_egad_j182", "pick_single_egad_j182")
class PickSingleEgadJ182Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/J18_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/J18_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J18_2_v2.pkl"


@register_task("maniskill.pick_single_egad_j183", "pick_single_egad_j183")
class PickSingleEgadJ183Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/J18_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/J18_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J18_3_v2.pkl"


@register_task("maniskill.pick_single_egad_j190", "pick_single_egad_j190")
class PickSingleEgadJ190Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/J19_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/J19_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J19_0_v2.pkl"


@register_task("maniskill.pick_single_egad_j191", "pick_single_egad_j191")
class PickSingleEgadJ191Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/J19_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/J19_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J19_1_v2.pkl"


@register_task("maniskill.pick_single_egad_j192", "pick_single_egad_j192")
class PickSingleEgadJ192Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/J19_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/J19_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J19_2_v2.pkl"


@register_task("maniskill.pick_single_egad_j193", "pick_single_egad_j193")
class PickSingleEgadJ193Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/J19_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/J19_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J19_3_v2.pkl"


@register_task("maniskill.pick_single_egad_j200", "pick_single_egad_j200")
class PickSingleEgadJ200Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/J20_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/J20_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J20_0_v2.pkl"


@register_task("maniskill.pick_single_egad_j201", "pick_single_egad_j201")
class PickSingleEgadJ201Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/J20_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/J20_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J20_1_v2.pkl"


@register_task("maniskill.pick_single_egad_j202", "pick_single_egad_j202")
class PickSingleEgadJ202Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/J20_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/J20_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J20_2_v2.pkl"


@register_task("maniskill.pick_single_egad_j203", "pick_single_egad_j203")
class PickSingleEgadJ203Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/J20_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/J20_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J20_3_v2.pkl"


@register_task("maniskill.pick_single_egad_j210", "pick_single_egad_j210")
class PickSingleEgadJ210Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/J21_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/J21_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J21_0_v2.pkl"


@register_task("maniskill.pick_single_egad_j211", "pick_single_egad_j211")
class PickSingleEgadJ211Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/J21_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/J21_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J21_1_v2.pkl"


@register_task("maniskill.pick_single_egad_j212", "pick_single_egad_j212")
class PickSingleEgadJ212Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/J21_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/J21_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J21_2_v2.pkl"


@register_task("maniskill.pick_single_egad_j213", "pick_single_egad_j213")
class PickSingleEgadJ213Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/J21_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/J21_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J21_3_v2.pkl"


@register_task("maniskill.pick_single_egad_j220", "pick_single_egad_j220")
class PickSingleEgadJ220Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/J22_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/J22_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J22_0_v2.pkl"


@register_task("maniskill.pick_single_egad_j221", "pick_single_egad_j221")
class PickSingleEgadJ221Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/J22_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/J22_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J22_1_v2.pkl"


@register_task("maniskill.pick_single_egad_j222", "pick_single_egad_j222")
class PickSingleEgadJ222Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/J22_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/J22_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J22_2_v2.pkl"


@register_task("maniskill.pick_single_egad_j223", "pick_single_egad_j223")
class PickSingleEgadJ223Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/J22_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/J22_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J22_3_v2.pkl"


@register_task("maniskill.pick_single_egad_j230", "pick_single_egad_j230")
class PickSingleEgadJ230Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/J23_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/J23_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J23_0_v2.pkl"


@register_task("maniskill.pick_single_egad_j231", "pick_single_egad_j231")
class PickSingleEgadJ231Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/J23_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/J23_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J23_1_v2.pkl"


@register_task("maniskill.pick_single_egad_j232", "pick_single_egad_j232")
class PickSingleEgadJ232Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/J23_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/J23_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J23_2_v2.pkl"


@register_task("maniskill.pick_single_egad_j233", "pick_single_egad_j233")
class PickSingleEgadJ233Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/J23_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/J23_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J23_3_v2.pkl"


@register_task("maniskill.pick_single_egad_j240", "pick_single_egad_j240")
class PickSingleEgadJ240Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/J24_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/J24_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J24_0_v2.pkl"


@register_task("maniskill.pick_single_egad_j241", "pick_single_egad_j241")
class PickSingleEgadJ241Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/J24_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/J24_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J24_1_v2.pkl"


@register_task("maniskill.pick_single_egad_j242", "pick_single_egad_j242")
class PickSingleEgadJ242Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/J24_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/J24_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J24_2_v2.pkl"


@register_task("maniskill.pick_single_egad_j243", "pick_single_egad_j243")
class PickSingleEgadJ243Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/J24_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/J24_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J24_3_v2.pkl"


@register_task("maniskill.pick_single_egad_j250", "pick_single_egad_j250")
class PickSingleEgadJ250Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/J25_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/J25_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J25_0_v2.pkl"


@register_task("maniskill.pick_single_egad_j251", "pick_single_egad_j251")
class PickSingleEgadJ251Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/J25_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/J25_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J25_1_v2.pkl"


@register_task("maniskill.pick_single_egad_j252", "pick_single_egad_j252")
class PickSingleEgadJ252Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/J25_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/J25_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J25_2_v2.pkl"


@register_task("maniskill.pick_single_egad_j253", "pick_single_egad_j253")
class PickSingleEgadJ253Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/J25_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/J25_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J25_3_v2.pkl"


@register_task("maniskill.pick_single_egad_k070", "pick_single_egad_k070")
class PickSingleEgadK070Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/K07_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/K07_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K07_0_v2.pkl"


@register_task("maniskill.pick_single_egad_k071", "pick_single_egad_k071")
class PickSingleEgadK071Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/K07_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/K07_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K07_1_v2.pkl"


@register_task("maniskill.pick_single_egad_k072", "pick_single_egad_k072")
class PickSingleEgadK072Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/K07_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/K07_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K07_2_v2.pkl"


@register_task("maniskill.pick_single_egad_k073", "pick_single_egad_k073")
class PickSingleEgadK073Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/K07_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/K07_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K07_3_v2.pkl"


@register_task("maniskill.pick_single_egad_k080", "pick_single_egad_k080")
class PickSingleEgadK080Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/K08_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/K08_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K08_0_v2.pkl"


@register_task("maniskill.pick_single_egad_k081", "pick_single_egad_k081")
class PickSingleEgadK081Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/K08_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/K08_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K08_1_v2.pkl"


@register_task("maniskill.pick_single_egad_k082", "pick_single_egad_k082")
class PickSingleEgadK082Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/K08_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/K08_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K08_2_v2.pkl"


@register_task("maniskill.pick_single_egad_k083", "pick_single_egad_k083")
class PickSingleEgadK083Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/K08_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/K08_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K08_3_v2.pkl"


@register_task("maniskill.pick_single_egad_k090", "pick_single_egad_k090")
class PickSingleEgadK090Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/K09_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/K09_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K09_0_v2.pkl"


@register_task("maniskill.pick_single_egad_k092", "pick_single_egad_k092")
class PickSingleEgadK092Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/K09_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/K09_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K09_2_v2.pkl"


@register_task("maniskill.pick_single_egad_k093", "pick_single_egad_k093")
class PickSingleEgadK093Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/K09_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/K09_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K09_3_v2.pkl"


@register_task("maniskill.pick_single_egad_k100", "pick_single_egad_k100")
class PickSingleEgadK100Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/K10_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/K10_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K10_0_v2.pkl"


@register_task("maniskill.pick_single_egad_k101", "pick_single_egad_k101")
class PickSingleEgadK101Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/K10_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/K10_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K10_1_v2.pkl"


@register_task("maniskill.pick_single_egad_k102", "pick_single_egad_k102")
class PickSingleEgadK102Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/K10_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/K10_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K10_2_v2.pkl"


@register_task("maniskill.pick_single_egad_k103", "pick_single_egad_k103")
class PickSingleEgadK103Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/K10_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/K10_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K10_3_v2.pkl"


@register_task("maniskill.pick_single_egad_k110", "pick_single_egad_k110")
class PickSingleEgadK110Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/K11_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/K11_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K11_0_v2.pkl"


@register_task("maniskill.pick_single_egad_k111", "pick_single_egad_k111")
class PickSingleEgadK111Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/K11_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/K11_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K11_1_v2.pkl"


@register_task("maniskill.pick_single_egad_k112", "pick_single_egad_k112")
class PickSingleEgadK112Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/K11_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/K11_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K11_2_v2.pkl"


@register_task("maniskill.pick_single_egad_k113", "pick_single_egad_k113")
class PickSingleEgadK113Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/K11_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/K11_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K11_3_v2.pkl"


@register_task("maniskill.pick_single_egad_k120", "pick_single_egad_k120")
class PickSingleEgadK120Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/K12_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/K12_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K12_0_v2.pkl"


@register_task("maniskill.pick_single_egad_k121", "pick_single_egad_k121")
class PickSingleEgadK121Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/K12_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/K12_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K12_1_v2.pkl"


@register_task("maniskill.pick_single_egad_k122", "pick_single_egad_k122")
class PickSingleEgadK122Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/K12_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/K12_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K12_2_v2.pkl"


@register_task("maniskill.pick_single_egad_k123", "pick_single_egad_k123")
class PickSingleEgadK123Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/K12_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/K12_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K12_3_v2.pkl"


@register_task("maniskill.pick_single_egad_k130", "pick_single_egad_k130")
class PickSingleEgadK130Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/K13_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/K13_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K13_0_v2.pkl"


@register_task("maniskill.pick_single_egad_k132", "pick_single_egad_k132")
class PickSingleEgadK132Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/K13_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/K13_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K13_2_v2.pkl"


@register_task("maniskill.pick_single_egad_k140", "pick_single_egad_k140")
class PickSingleEgadK140Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/K14_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/K14_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K14_0_v2.pkl"


@register_task("maniskill.pick_single_egad_k142", "pick_single_egad_k142")
class PickSingleEgadK142Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/K14_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/K14_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K14_2_v2.pkl"


@register_task("maniskill.pick_single_egad_k143", "pick_single_egad_k143")
class PickSingleEgadK143Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/K14_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/K14_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K14_3_v2.pkl"


@register_task("maniskill.pick_single_egad_k150", "pick_single_egad_k150")
class PickSingleEgadK150Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/K15_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/K15_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K15_0_v2.pkl"


@register_task("maniskill.pick_single_egad_k151", "pick_single_egad_k151")
class PickSingleEgadK151Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/K15_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/K15_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K15_1_v2.pkl"


@register_task("maniskill.pick_single_egad_k152", "pick_single_egad_k152")
class PickSingleEgadK152Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/K15_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/K15_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K15_2_v2.pkl"


@register_task("maniskill.pick_single_egad_k153", "pick_single_egad_k153")
class PickSingleEgadK153Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/K15_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/K15_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K15_3_v2.pkl"


@register_task("maniskill.pick_single_egad_k160", "pick_single_egad_k160")
class PickSingleEgadK160Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/K16_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/K16_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K16_0_v2.pkl"


@register_task("maniskill.pick_single_egad_k161", "pick_single_egad_k161")
class PickSingleEgadK161Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/K16_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/K16_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K16_1_v2.pkl"


@register_task("maniskill.pick_single_egad_k163", "pick_single_egad_k163")
class PickSingleEgadK163Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/K16_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/K16_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K16_3_v2.pkl"


@register_task("maniskill.pick_single_egad_k170", "pick_single_egad_k170")
class PickSingleEgadK170Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/K17_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/K17_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K17_0_v2.pkl"


@register_task("maniskill.pick_single_egad_k171", "pick_single_egad_k171")
class PickSingleEgadK171Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/K17_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/K17_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K17_1_v2.pkl"


@register_task("maniskill.pick_single_egad_k172", "pick_single_egad_k172")
class PickSingleEgadK172Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/K17_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/K17_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K17_2_v2.pkl"


@register_task("maniskill.pick_single_egad_k173", "pick_single_egad_k173")
class PickSingleEgadK173Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/K17_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/K17_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K17_3_v2.pkl"


@register_task("maniskill.pick_single_egad_k180", "pick_single_egad_k180")
class PickSingleEgadK180Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/K18_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/K18_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K18_0_v2.pkl"


@register_task("maniskill.pick_single_egad_k181", "pick_single_egad_k181")
class PickSingleEgadK181Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/K18_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/K18_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K18_1_v2.pkl"


@register_task("maniskill.pick_single_egad_k182", "pick_single_egad_k182")
class PickSingleEgadK182Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/K18_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/K18_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K18_2_v2.pkl"


@register_task("maniskill.pick_single_egad_k183", "pick_single_egad_k183")
class PickSingleEgadK183Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/K18_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/K18_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K18_3_v2.pkl"


@register_task("maniskill.pick_single_egad_k190", "pick_single_egad_k190")
class PickSingleEgadK190Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/K19_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/K19_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K19_0_v2.pkl"


@register_task("maniskill.pick_single_egad_k191", "pick_single_egad_k191")
class PickSingleEgadK191Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/K19_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/K19_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K19_1_v2.pkl"


@register_task("maniskill.pick_single_egad_k192", "pick_single_egad_k192")
class PickSingleEgadK192Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/K19_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/K19_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K19_2_v2.pkl"


@register_task("maniskill.pick_single_egad_k193", "pick_single_egad_k193")
class PickSingleEgadK193Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/K19_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/K19_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K19_3_v2.pkl"


@register_task("maniskill.pick_single_egad_k200", "pick_single_egad_k200")
class PickSingleEgadK200Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/K20_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/K20_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K20_0_v2.pkl"


@register_task("maniskill.pick_single_egad_k201", "pick_single_egad_k201")
class PickSingleEgadK201Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/K20_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/K20_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K20_1_v2.pkl"


@register_task("maniskill.pick_single_egad_k202", "pick_single_egad_k202")
class PickSingleEgadK202Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/K20_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/K20_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K20_2_v2.pkl"


@register_task("maniskill.pick_single_egad_k203", "pick_single_egad_k203")
class PickSingleEgadK203Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/K20_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/K20_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K20_3_v2.pkl"


@register_task("maniskill.pick_single_egad_k210", "pick_single_egad_k210")
class PickSingleEgadK210Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/K21_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/K21_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K21_0_v2.pkl"


@register_task("maniskill.pick_single_egad_k211", "pick_single_egad_k211")
class PickSingleEgadK211Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/K21_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/K21_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K21_1_v2.pkl"


@register_task("maniskill.pick_single_egad_k212", "pick_single_egad_k212")
class PickSingleEgadK212Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/K21_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/K21_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K21_2_v2.pkl"


@register_task("maniskill.pick_single_egad_k213", "pick_single_egad_k213")
class PickSingleEgadK213Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/K21_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/K21_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K21_3_v2.pkl"


@register_task("maniskill.pick_single_egad_k220", "pick_single_egad_k220")
class PickSingleEgadK220Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/K22_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/K22_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K22_0_v2.pkl"


@register_task("maniskill.pick_single_egad_k221", "pick_single_egad_k221")
class PickSingleEgadK221Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/K22_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/K22_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K22_1_v2.pkl"


@register_task("maniskill.pick_single_egad_k222", "pick_single_egad_k222")
class PickSingleEgadK222Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/K22_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/K22_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K22_2_v2.pkl"


@register_task("maniskill.pick_single_egad_k223", "pick_single_egad_k223")
class PickSingleEgadK223Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/K22_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/K22_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K22_3_v2.pkl"


@register_task("maniskill.pick_single_egad_k230", "pick_single_egad_k230")
class PickSingleEgadK230Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/K23_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/K23_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K23_0_v2.pkl"


@register_task("maniskill.pick_single_egad_k231", "pick_single_egad_k231")
class PickSingleEgadK231Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/K23_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/K23_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K23_1_v2.pkl"


@register_task("maniskill.pick_single_egad_k232", "pick_single_egad_k232")
class PickSingleEgadK232Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/K23_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/K23_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K23_2_v2.pkl"


@register_task("maniskill.pick_single_egad_k233", "pick_single_egad_k233")
class PickSingleEgadK233Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/K23_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/K23_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K23_3_v2.pkl"


@register_task("maniskill.pick_single_egad_k240", "pick_single_egad_k240")
class PickSingleEgadK240Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/K24_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/K24_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K24_0_v2.pkl"


@register_task("maniskill.pick_single_egad_k241", "pick_single_egad_k241")
class PickSingleEgadK241Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/K24_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/K24_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K24_1_v2.pkl"


@register_task("maniskill.pick_single_egad_k242", "pick_single_egad_k242")
class PickSingleEgadK242Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/K24_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/K24_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K24_2_v2.pkl"


@register_task("maniskill.pick_single_egad_k243", "pick_single_egad_k243")
class PickSingleEgadK243Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/K24_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/K24_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K24_3_v2.pkl"


@register_task("maniskill.pick_single_egad_k250", "pick_single_egad_k250")
class PickSingleEgadK250Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/K25_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/K25_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K25_0_v2.pkl"


@register_task("maniskill.pick_single_egad_k251", "pick_single_egad_k251")
class PickSingleEgadK251Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/K25_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/K25_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K25_1_v2.pkl"


@register_task("maniskill.pick_single_egad_k252", "pick_single_egad_k252")
class PickSingleEgadK252Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/K25_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/K25_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K25_2_v2.pkl"


@register_task("maniskill.pick_single_egad_k253", "pick_single_egad_k253")
class PickSingleEgadK253Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/K25_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/K25_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K25_3_v2.pkl"


@register_task("maniskill.pick_single_egad_l070", "pick_single_egad_l070")
class PickSingleEgadL070Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/L07_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/L07_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L07_0_v2.pkl"


@register_task("maniskill.pick_single_egad_l071", "pick_single_egad_l071")
class PickSingleEgadL071Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/L07_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/L07_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L07_1_v2.pkl"


@register_task("maniskill.pick_single_egad_l072", "pick_single_egad_l072")
class PickSingleEgadL072Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/L07_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/L07_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L07_2_v2.pkl"


@register_task("maniskill.pick_single_egad_l073", "pick_single_egad_l073")
class PickSingleEgadL073Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/L07_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/L07_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L07_3_v2.pkl"


@register_task("maniskill.pick_single_egad_l080", "pick_single_egad_l080")
class PickSingleEgadL080Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/L08_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/L08_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L08_0_v2.pkl"


@register_task("maniskill.pick_single_egad_l081", "pick_single_egad_l081")
class PickSingleEgadL081Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/L08_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/L08_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L08_1_v2.pkl"


@register_task("maniskill.pick_single_egad_l082", "pick_single_egad_l082")
class PickSingleEgadL082Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/L08_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/L08_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L08_2_v2.pkl"


@register_task("maniskill.pick_single_egad_l083", "pick_single_egad_l083")
class PickSingleEgadL083Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/L08_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/L08_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L08_3_v2.pkl"


@register_task("maniskill.pick_single_egad_l090", "pick_single_egad_l090")
class PickSingleEgadL090Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/L09_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/L09_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L09_0_v2.pkl"


@register_task("maniskill.pick_single_egad_l091", "pick_single_egad_l091")
class PickSingleEgadL091Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/L09_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/L09_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L09_1_v2.pkl"


@register_task("maniskill.pick_single_egad_l092", "pick_single_egad_l092")
class PickSingleEgadL092Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/L09_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/L09_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L09_2_v2.pkl"


@register_task("maniskill.pick_single_egad_l093", "pick_single_egad_l093")
class PickSingleEgadL093Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/L09_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/L09_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L09_3_v2.pkl"


@register_task("maniskill.pick_single_egad_l100", "pick_single_egad_l100")
class PickSingleEgadL100Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/L10_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/L10_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L10_0_v2.pkl"


@register_task("maniskill.pick_single_egad_l101", "pick_single_egad_l101")
class PickSingleEgadL101Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/L10_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/L10_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L10_1_v2.pkl"


@register_task("maniskill.pick_single_egad_l102", "pick_single_egad_l102")
class PickSingleEgadL102Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/L10_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/L10_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L10_2_v2.pkl"


@register_task("maniskill.pick_single_egad_l110", "pick_single_egad_l110")
class PickSingleEgadL110Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/L11_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/L11_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L11_0_v2.pkl"


@register_task("maniskill.pick_single_egad_l111", "pick_single_egad_l111")
class PickSingleEgadL111Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/L11_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/L11_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L11_1_v2.pkl"


@register_task("maniskill.pick_single_egad_l112", "pick_single_egad_l112")
class PickSingleEgadL112Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/L11_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/L11_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L11_2_v2.pkl"


@register_task("maniskill.pick_single_egad_l113", "pick_single_egad_l113")
class PickSingleEgadL113Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/L11_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/L11_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L11_3_v2.pkl"


@register_task("maniskill.pick_single_egad_l120", "pick_single_egad_l120")
class PickSingleEgadL120Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/L12_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/L12_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L12_0_v2.pkl"


@register_task("maniskill.pick_single_egad_l121", "pick_single_egad_l121")
class PickSingleEgadL121Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/L12_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/L12_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L12_1_v2.pkl"


@register_task("maniskill.pick_single_egad_l122", "pick_single_egad_l122")
class PickSingleEgadL122Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/L12_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/L12_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L12_2_v2.pkl"


@register_task("maniskill.pick_single_egad_l123", "pick_single_egad_l123")
class PickSingleEgadL123Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/L12_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/L12_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L12_3_v2.pkl"


@register_task("maniskill.pick_single_egad_l130", "pick_single_egad_l130")
class PickSingleEgadL130Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/L13_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/L13_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L13_0_v2.pkl"


@register_task("maniskill.pick_single_egad_l131", "pick_single_egad_l131")
class PickSingleEgadL131Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/L13_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/L13_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L13_1_v2.pkl"


@register_task("maniskill.pick_single_egad_l132", "pick_single_egad_l132")
class PickSingleEgadL132Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/L13_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/L13_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L13_2_v2.pkl"


@register_task("maniskill.pick_single_egad_l133", "pick_single_egad_l133")
class PickSingleEgadL133Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/L13_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/L13_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L13_3_v2.pkl"


@register_task("maniskill.pick_single_egad_l141", "pick_single_egad_l141")
class PickSingleEgadL141Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/L14_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/L14_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L14_1_v2.pkl"


@register_task("maniskill.pick_single_egad_l142", "pick_single_egad_l142")
class PickSingleEgadL142Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/L14_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/L14_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L14_2_v2.pkl"


@register_task("maniskill.pick_single_egad_l143", "pick_single_egad_l143")
class PickSingleEgadL143Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/L14_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/L14_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L14_3_v2.pkl"


@register_task("maniskill.pick_single_egad_l150", "pick_single_egad_l150")
class PickSingleEgadL150Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/L15_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/L15_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L15_0_v2.pkl"


@register_task("maniskill.pick_single_egad_l151", "pick_single_egad_l151")
class PickSingleEgadL151Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/L15_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/L15_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L15_1_v2.pkl"


@register_task("maniskill.pick_single_egad_l153", "pick_single_egad_l153")
class PickSingleEgadL153Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/L15_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/L15_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L15_3_v2.pkl"


@register_task("maniskill.pick_single_egad_l160", "pick_single_egad_l160")
class PickSingleEgadL160Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/L16_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/L16_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L16_0_v2.pkl"


@register_task("maniskill.pick_single_egad_l161", "pick_single_egad_l161")
class PickSingleEgadL161Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/L16_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/L16_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L16_1_v2.pkl"


@register_task("maniskill.pick_single_egad_l162", "pick_single_egad_l162")
class PickSingleEgadL162Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/L16_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/L16_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L16_2_v2.pkl"


@register_task("maniskill.pick_single_egad_l163", "pick_single_egad_l163")
class PickSingleEgadL163Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/L16_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/L16_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L16_3_v2.pkl"


@register_task("maniskill.pick_single_egad_l171", "pick_single_egad_l171")
class PickSingleEgadL171Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/L17_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/L17_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L17_1_v2.pkl"


@register_task("maniskill.pick_single_egad_l172", "pick_single_egad_l172")
class PickSingleEgadL172Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/L17_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/L17_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L17_2_v2.pkl"


@register_task("maniskill.pick_single_egad_l173", "pick_single_egad_l173")
class PickSingleEgadL173Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/L17_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/L17_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L17_3_v2.pkl"


@register_task("maniskill.pick_single_egad_l180", "pick_single_egad_l180")
class PickSingleEgadL180Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/L18_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/L18_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L18_0_v2.pkl"


@register_task("maniskill.pick_single_egad_l181", "pick_single_egad_l181")
class PickSingleEgadL181Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/L18_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/L18_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L18_1_v2.pkl"


@register_task("maniskill.pick_single_egad_l182", "pick_single_egad_l182")
class PickSingleEgadL182Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/L18_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/L18_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L18_2_v2.pkl"


@register_task("maniskill.pick_single_egad_l183", "pick_single_egad_l183")
class PickSingleEgadL183Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/L18_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/L18_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L18_3_v2.pkl"


@register_task("maniskill.pick_single_egad_l191", "pick_single_egad_l191")
class PickSingleEgadL191Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/L19_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/L19_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L19_1_v2.pkl"


@register_task("maniskill.pick_single_egad_l192", "pick_single_egad_l192")
class PickSingleEgadL192Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/L19_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/L19_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L19_2_v2.pkl"


@register_task("maniskill.pick_single_egad_l193", "pick_single_egad_l193")
class PickSingleEgadL193Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/L19_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/L19_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L19_3_v2.pkl"


@register_task("maniskill.pick_single_egad_l200", "pick_single_egad_l200")
class PickSingleEgadL200Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/L20_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/L20_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L20_0_v2.pkl"


@register_task("maniskill.pick_single_egad_l201", "pick_single_egad_l201")
class PickSingleEgadL201Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/L20_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/L20_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L20_1_v2.pkl"


@register_task("maniskill.pick_single_egad_l202", "pick_single_egad_l202")
class PickSingleEgadL202Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/L20_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/L20_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L20_2_v2.pkl"


@register_task("maniskill.pick_single_egad_l203", "pick_single_egad_l203")
class PickSingleEgadL203Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/L20_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/L20_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L20_3_v2.pkl"


@register_task("maniskill.pick_single_egad_l210", "pick_single_egad_l210")
class PickSingleEgadL210Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/L21_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/L21_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L21_0_v2.pkl"


@register_task("maniskill.pick_single_egad_l211", "pick_single_egad_l211")
class PickSingleEgadL211Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/L21_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/L21_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L21_1_v2.pkl"


@register_task("maniskill.pick_single_egad_l212", "pick_single_egad_l212")
class PickSingleEgadL212Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/L21_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/L21_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L21_2_v2.pkl"


@register_task("maniskill.pick_single_egad_l213", "pick_single_egad_l213")
class PickSingleEgadL213Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/L21_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/L21_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L21_3_v2.pkl"


@register_task("maniskill.pick_single_egad_l220", "pick_single_egad_l220")
class PickSingleEgadL220Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/L22_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/L22_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L22_0_v2.pkl"


@register_task("maniskill.pick_single_egad_l221", "pick_single_egad_l221")
class PickSingleEgadL221Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/L22_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/L22_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L22_1_v2.pkl"


@register_task("maniskill.pick_single_egad_l222", "pick_single_egad_l222")
class PickSingleEgadL222Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/L22_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/L22_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L22_2_v2.pkl"


@register_task("maniskill.pick_single_egad_l223", "pick_single_egad_l223")
class PickSingleEgadL223Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/L22_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/L22_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L22_3_v2.pkl"


@register_task("maniskill.pick_single_egad_l230", "pick_single_egad_l230")
class PickSingleEgadL230Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/L23_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/L23_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L23_0_v2.pkl"


@register_task("maniskill.pick_single_egad_l231", "pick_single_egad_l231")
class PickSingleEgadL231Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/L23_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/L23_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L23_1_v2.pkl"


@register_task("maniskill.pick_single_egad_l232", "pick_single_egad_l232")
class PickSingleEgadL232Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/L23_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/L23_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L23_2_v2.pkl"


@register_task("maniskill.pick_single_egad_l233", "pick_single_egad_l233")
class PickSingleEgadL233Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/L23_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/L23_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L23_3_v2.pkl"


@register_task("maniskill.pick_single_egad_l240", "pick_single_egad_l240")
class PickSingleEgadL240Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/L24_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/L24_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L24_0_v2.pkl"


@register_task("maniskill.pick_single_egad_l241", "pick_single_egad_l241")
class PickSingleEgadL241Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/L24_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/L24_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L24_1_v2.pkl"


@register_task("maniskill.pick_single_egad_l243", "pick_single_egad_l243")
class PickSingleEgadL243Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/L24_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/L24_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L24_3_v2.pkl"


@register_task("maniskill.pick_single_egad_l250", "pick_single_egad_l250")
class PickSingleEgadL250Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/L25_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/L25_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L25_0_v2.pkl"


@register_task("maniskill.pick_single_egad_l251", "pick_single_egad_l251")
class PickSingleEgadL251Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/L25_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/L25_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L25_1_v2.pkl"


@register_task("maniskill.pick_single_egad_l252", "pick_single_egad_l252")
class PickSingleEgadL252Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/L25_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/L25_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L25_2_v2.pkl"


@register_task("maniskill.pick_single_egad_l253", "pick_single_egad_l253")
class PickSingleEgadL253Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/L25_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/L25_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L25_3_v2.pkl"


@register_task("maniskill.pick_single_egad_m051", "pick_single_egad_m051")
class PickSingleEgadM051Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/M05_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/M05_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M05_1_v2.pkl"


@register_task("maniskill.pick_single_egad_m052", "pick_single_egad_m052")
class PickSingleEgadM052Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/M05_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/M05_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M05_2_v2.pkl"


@register_task("maniskill.pick_single_egad_m053", "pick_single_egad_m053")
class PickSingleEgadM053Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/M05_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/M05_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M05_3_v2.pkl"


@register_task("maniskill.pick_single_egad_m061", "pick_single_egad_m061")
class PickSingleEgadM061Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/M06_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/M06_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M06_1_v2.pkl"


@register_task("maniskill.pick_single_egad_m062", "pick_single_egad_m062")
class PickSingleEgadM062Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/M06_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/M06_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M06_2_v2.pkl"


@register_task("maniskill.pick_single_egad_m063", "pick_single_egad_m063")
class PickSingleEgadM063Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/M06_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/M06_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M06_3_v2.pkl"


@register_task("maniskill.pick_single_egad_m070", "pick_single_egad_m070")
class PickSingleEgadM070Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/M07_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/M07_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M07_0_v2.pkl"


@register_task("maniskill.pick_single_egad_m071", "pick_single_egad_m071")
class PickSingleEgadM071Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/M07_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/M07_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M07_1_v2.pkl"


@register_task("maniskill.pick_single_egad_m073", "pick_single_egad_m073")
class PickSingleEgadM073Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/M07_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/M07_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M07_3_v2.pkl"


@register_task("maniskill.pick_single_egad_m080", "pick_single_egad_m080")
class PickSingleEgadM080Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/M08_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/M08_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M08_0_v2.pkl"


@register_task("maniskill.pick_single_egad_m082", "pick_single_egad_m082")
class PickSingleEgadM082Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/M08_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/M08_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M08_2_v2.pkl"


@register_task("maniskill.pick_single_egad_m083", "pick_single_egad_m083")
class PickSingleEgadM083Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/M08_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/M08_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M08_3_v2.pkl"


@register_task("maniskill.pick_single_egad_m090", "pick_single_egad_m090")
class PickSingleEgadM090Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/M09_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/M09_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M09_0_v2.pkl"


@register_task("maniskill.pick_single_egad_m091", "pick_single_egad_m091")
class PickSingleEgadM091Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/M09_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/M09_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M09_1_v2.pkl"


@register_task("maniskill.pick_single_egad_m092", "pick_single_egad_m092")
class PickSingleEgadM092Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/M09_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/M09_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M09_2_v2.pkl"


@register_task("maniskill.pick_single_egad_m093", "pick_single_egad_m093")
class PickSingleEgadM093Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/M09_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/M09_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M09_3_v2.pkl"


@register_task("maniskill.pick_single_egad_m100", "pick_single_egad_m100")
class PickSingleEgadM100Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/M10_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/M10_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M10_0_v2.pkl"


@register_task("maniskill.pick_single_egad_m101", "pick_single_egad_m101")
class PickSingleEgadM101Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/M10_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/M10_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M10_1_v2.pkl"


@register_task("maniskill.pick_single_egad_m102", "pick_single_egad_m102")
class PickSingleEgadM102Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/M10_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/M10_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M10_2_v2.pkl"


@register_task("maniskill.pick_single_egad_m103", "pick_single_egad_m103")
class PickSingleEgadM103Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/M10_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/M10_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M10_3_v2.pkl"


@register_task("maniskill.pick_single_egad_m110", "pick_single_egad_m110")
class PickSingleEgadM110Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/M11_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/M11_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M11_0_v2.pkl"


@register_task("maniskill.pick_single_egad_m111", "pick_single_egad_m111")
class PickSingleEgadM111Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/M11_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/M11_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M11_1_v2.pkl"


@register_task("maniskill.pick_single_egad_m112", "pick_single_egad_m112")
class PickSingleEgadM112Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/M11_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/M11_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M11_2_v2.pkl"


@register_task("maniskill.pick_single_egad_m113", "pick_single_egad_m113")
class PickSingleEgadM113Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/M11_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/M11_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M11_3_v2.pkl"


@register_task("maniskill.pick_single_egad_m120", "pick_single_egad_m120")
class PickSingleEgadM120Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/M12_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/M12_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M12_0_v2.pkl"


@register_task("maniskill.pick_single_egad_m121", "pick_single_egad_m121")
class PickSingleEgadM121Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/M12_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/M12_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M12_1_v2.pkl"


@register_task("maniskill.pick_single_egad_m122", "pick_single_egad_m122")
class PickSingleEgadM122Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/M12_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/M12_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M12_2_v2.pkl"


@register_task("maniskill.pick_single_egad_m123", "pick_single_egad_m123")
class PickSingleEgadM123Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/M12_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/M12_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M12_3_v2.pkl"


@register_task("maniskill.pick_single_egad_m130", "pick_single_egad_m130")
class PickSingleEgadM130Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/M13_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/M13_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M13_0_v2.pkl"


@register_task("maniskill.pick_single_egad_m131", "pick_single_egad_m131")
class PickSingleEgadM131Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/M13_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/M13_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M13_1_v2.pkl"


@register_task("maniskill.pick_single_egad_m132", "pick_single_egad_m132")
class PickSingleEgadM132Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/M13_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/M13_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M13_2_v2.pkl"


@register_task("maniskill.pick_single_egad_m133", "pick_single_egad_m133")
class PickSingleEgadM133Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/M13_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/M13_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M13_3_v2.pkl"


@register_task("maniskill.pick_single_egad_m140", "pick_single_egad_m140")
class PickSingleEgadM140Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/M14_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/M14_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M14_0_v2.pkl"


@register_task("maniskill.pick_single_egad_m141", "pick_single_egad_m141")
class PickSingleEgadM141Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/M14_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/M14_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M14_1_v2.pkl"


@register_task("maniskill.pick_single_egad_m142", "pick_single_egad_m142")
class PickSingleEgadM142Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/M14_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/M14_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M14_2_v2.pkl"


@register_task("maniskill.pick_single_egad_m143", "pick_single_egad_m143")
class PickSingleEgadM143Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/M14_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/M14_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M14_3_v2.pkl"


@register_task("maniskill.pick_single_egad_m150", "pick_single_egad_m150")
class PickSingleEgadM150Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/M15_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/M15_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M15_0_v2.pkl"


@register_task("maniskill.pick_single_egad_m151", "pick_single_egad_m151")
class PickSingleEgadM151Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/M15_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/M15_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M15_1_v2.pkl"


@register_task("maniskill.pick_single_egad_m152", "pick_single_egad_m152")
class PickSingleEgadM152Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/M15_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/M15_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M15_2_v2.pkl"


@register_task("maniskill.pick_single_egad_m153", "pick_single_egad_m153")
class PickSingleEgadM153Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/M15_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/M15_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M15_3_v2.pkl"


@register_task("maniskill.pick_single_egad_m160", "pick_single_egad_m160")
class PickSingleEgadM160Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/M16_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/M16_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M16_0_v2.pkl"


@register_task("maniskill.pick_single_egad_m161", "pick_single_egad_m161")
class PickSingleEgadM161Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/M16_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/M16_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M16_1_v2.pkl"


@register_task("maniskill.pick_single_egad_m162", "pick_single_egad_m162")
class PickSingleEgadM162Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/M16_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/M16_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M16_2_v2.pkl"


@register_task("maniskill.pick_single_egad_m163", "pick_single_egad_m163")
class PickSingleEgadM163Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/M16_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/M16_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M16_3_v2.pkl"


@register_task("maniskill.pick_single_egad_m171", "pick_single_egad_m171")
class PickSingleEgadM171Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/M17_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/M17_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M17_1_v2.pkl"


@register_task("maniskill.pick_single_egad_m172", "pick_single_egad_m172")
class PickSingleEgadM172Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/M17_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/M17_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M17_2_v2.pkl"


@register_task("maniskill.pick_single_egad_m173", "pick_single_egad_m173")
class PickSingleEgadM173Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/M17_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/M17_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M17_3_v2.pkl"


@register_task("maniskill.pick_single_egad_m180", "pick_single_egad_m180")
class PickSingleEgadM180Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/M18_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/M18_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M18_0_v2.pkl"


@register_task("maniskill.pick_single_egad_m181", "pick_single_egad_m181")
class PickSingleEgadM181Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/M18_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/M18_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M18_1_v2.pkl"


@register_task("maniskill.pick_single_egad_m182", "pick_single_egad_m182")
class PickSingleEgadM182Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/M18_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/M18_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M18_2_v2.pkl"


@register_task("maniskill.pick_single_egad_m183", "pick_single_egad_m183")
class PickSingleEgadM183Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/M18_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/M18_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M18_3_v2.pkl"


@register_task("maniskill.pick_single_egad_m190", "pick_single_egad_m190")
class PickSingleEgadM190Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/M19_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/M19_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M19_0_v2.pkl"


@register_task("maniskill.pick_single_egad_m191", "pick_single_egad_m191")
class PickSingleEgadM191Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/M19_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/M19_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M19_1_v2.pkl"


@register_task("maniskill.pick_single_egad_m193", "pick_single_egad_m193")
class PickSingleEgadM193Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/M19_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/M19_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M19_3_v2.pkl"


@register_task("maniskill.pick_single_egad_m200", "pick_single_egad_m200")
class PickSingleEgadM200Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/M20_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/M20_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M20_0_v2.pkl"


@register_task("maniskill.pick_single_egad_m201", "pick_single_egad_m201")
class PickSingleEgadM201Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/M20_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/M20_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M20_1_v2.pkl"


@register_task("maniskill.pick_single_egad_m202", "pick_single_egad_m202")
class PickSingleEgadM202Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/M20_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/M20_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M20_2_v2.pkl"


@register_task("maniskill.pick_single_egad_m203", "pick_single_egad_m203")
class PickSingleEgadM203Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/M20_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/M20_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M20_3_v2.pkl"


@register_task("maniskill.pick_single_egad_m210", "pick_single_egad_m210")
class PickSingleEgadM210Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/M21_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/M21_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M21_0_v2.pkl"


@register_task("maniskill.pick_single_egad_m211", "pick_single_egad_m211")
class PickSingleEgadM211Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/M21_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/M21_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M21_1_v2.pkl"


@register_task("maniskill.pick_single_egad_m213", "pick_single_egad_m213")
class PickSingleEgadM213Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/M21_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/M21_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M21_3_v2.pkl"


@register_task("maniskill.pick_single_egad_m221", "pick_single_egad_m221")
class PickSingleEgadM221Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/M22_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/M22_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M22_1_v2.pkl"


@register_task("maniskill.pick_single_egad_m222", "pick_single_egad_m222")
class PickSingleEgadM222Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/M22_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/M22_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M22_2_v2.pkl"


@register_task("maniskill.pick_single_egad_m223", "pick_single_egad_m223")
class PickSingleEgadM223Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/M22_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/M22_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M22_3_v2.pkl"


@register_task("maniskill.pick_single_egad_m230", "pick_single_egad_m230")
class PickSingleEgadM230Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/M23_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/M23_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M23_0_v2.pkl"


@register_task("maniskill.pick_single_egad_m231", "pick_single_egad_m231")
class PickSingleEgadM231Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/M23_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/M23_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M23_1_v2.pkl"


@register_task("maniskill.pick_single_egad_m232", "pick_single_egad_m232")
class PickSingleEgadM232Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/M23_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/M23_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M23_2_v2.pkl"


@register_task("maniskill.pick_single_egad_m233", "pick_single_egad_m233")
class PickSingleEgadM233Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/M23_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/M23_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M23_3_v2.pkl"


@register_task("maniskill.pick_single_egad_m240", "pick_single_egad_m240")
class PickSingleEgadM240Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/M24_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/M24_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M24_0_v2.pkl"


@register_task("maniskill.pick_single_egad_m241", "pick_single_egad_m241")
class PickSingleEgadM241Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/M24_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/M24_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M24_1_v2.pkl"


@register_task("maniskill.pick_single_egad_m242", "pick_single_egad_m242")
class PickSingleEgadM242Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/M24_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/M24_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M24_2_v2.pkl"


@register_task("maniskill.pick_single_egad_m243", "pick_single_egad_m243")
class PickSingleEgadM243Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/M24_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/M24_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M24_3_v2.pkl"


@register_task("maniskill.pick_single_egad_m250", "pick_single_egad_m250")
class PickSingleEgadM250Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/M25_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/M25_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M25_0_v2.pkl"


@register_task("maniskill.pick_single_egad_m251", "pick_single_egad_m251")
class PickSingleEgadM251Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/M25_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/M25_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M25_1_v2.pkl"


@register_task("maniskill.pick_single_egad_m252", "pick_single_egad_m252")
class PickSingleEgadM252Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/M25_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/M25_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M25_2_v2.pkl"


@register_task("maniskill.pick_single_egad_m253", "pick_single_egad_m253")
class PickSingleEgadM253Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/M25_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/M25_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M25_3_v2.pkl"


@register_task("maniskill.pick_single_egad_n050", "pick_single_egad_n050")
class PickSingleEgadN050Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/N05_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/N05_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N05_0_v2.pkl"


@register_task("maniskill.pick_single_egad_n051", "pick_single_egad_n051")
class PickSingleEgadN051Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/N05_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/N05_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N05_1_v2.pkl"


@register_task("maniskill.pick_single_egad_n052", "pick_single_egad_n052")
class PickSingleEgadN052Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/N05_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/N05_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N05_2_v2.pkl"


@register_task("maniskill.pick_single_egad_n060", "pick_single_egad_n060")
class PickSingleEgadN060Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/N06_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/N06_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N06_0_v2.pkl"


@register_task("maniskill.pick_single_egad_n061", "pick_single_egad_n061")
class PickSingleEgadN061Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/N06_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/N06_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N06_1_v2.pkl"


@register_task("maniskill.pick_single_egad_n062", "pick_single_egad_n062")
class PickSingleEgadN062Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/N06_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/N06_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N06_2_v2.pkl"


@register_task("maniskill.pick_single_egad_n063", "pick_single_egad_n063")
class PickSingleEgadN063Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/N06_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/N06_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N06_3_v2.pkl"


@register_task("maniskill.pick_single_egad_n070", "pick_single_egad_n070")
class PickSingleEgadN070Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/N07_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/N07_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N07_0_v2.pkl"


@register_task("maniskill.pick_single_egad_n071", "pick_single_egad_n071")
class PickSingleEgadN071Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/N07_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/N07_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N07_1_v2.pkl"


@register_task("maniskill.pick_single_egad_n072", "pick_single_egad_n072")
class PickSingleEgadN072Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/N07_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/N07_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N07_2_v2.pkl"


@register_task("maniskill.pick_single_egad_n073", "pick_single_egad_n073")
class PickSingleEgadN073Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/N07_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/N07_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N07_3_v2.pkl"


@register_task("maniskill.pick_single_egad_n080", "pick_single_egad_n080")
class PickSingleEgadN080Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/N08_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/N08_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N08_0_v2.pkl"


@register_task("maniskill.pick_single_egad_n081", "pick_single_egad_n081")
class PickSingleEgadN081Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/N08_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/N08_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N08_1_v2.pkl"


@register_task("maniskill.pick_single_egad_n083", "pick_single_egad_n083")
class PickSingleEgadN083Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/N08_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/N08_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N08_3_v2.pkl"


@register_task("maniskill.pick_single_egad_n090", "pick_single_egad_n090")
class PickSingleEgadN090Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/N09_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/N09_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N09_0_v2.pkl"


@register_task("maniskill.pick_single_egad_n091", "pick_single_egad_n091")
class PickSingleEgadN091Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/N09_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/N09_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N09_1_v2.pkl"


@register_task("maniskill.pick_single_egad_n092", "pick_single_egad_n092")
class PickSingleEgadN092Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/N09_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/N09_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N09_2_v2.pkl"


@register_task("maniskill.pick_single_egad_n093", "pick_single_egad_n093")
class PickSingleEgadN093Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/N09_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/N09_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N09_3_v2.pkl"


@register_task("maniskill.pick_single_egad_n100", "pick_single_egad_n100")
class PickSingleEgadN100Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/N10_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/N10_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N10_0_v2.pkl"


@register_task("maniskill.pick_single_egad_n101", "pick_single_egad_n101")
class PickSingleEgadN101Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/N10_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/N10_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N10_1_v2.pkl"


@register_task("maniskill.pick_single_egad_n103", "pick_single_egad_n103")
class PickSingleEgadN103Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/N10_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/N10_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N10_3_v2.pkl"


@register_task("maniskill.pick_single_egad_n110", "pick_single_egad_n110")
class PickSingleEgadN110Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/N11_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/N11_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N11_0_v2.pkl"


@register_task("maniskill.pick_single_egad_n111", "pick_single_egad_n111")
class PickSingleEgadN111Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/N11_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/N11_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N11_1_v2.pkl"


@register_task("maniskill.pick_single_egad_n112", "pick_single_egad_n112")
class PickSingleEgadN112Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/N11_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/N11_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N11_2_v2.pkl"


@register_task("maniskill.pick_single_egad_n113", "pick_single_egad_n113")
class PickSingleEgadN113Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/N11_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/N11_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N11_3_v2.pkl"


@register_task("maniskill.pick_single_egad_n120", "pick_single_egad_n120")
class PickSingleEgadN120Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/N12_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/N12_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N12_0_v2.pkl"


@register_task("maniskill.pick_single_egad_n121", "pick_single_egad_n121")
class PickSingleEgadN121Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/N12_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/N12_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N12_1_v2.pkl"


@register_task("maniskill.pick_single_egad_n123", "pick_single_egad_n123")
class PickSingleEgadN123Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/N12_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/N12_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N12_3_v2.pkl"


@register_task("maniskill.pick_single_egad_n130", "pick_single_egad_n130")
class PickSingleEgadN130Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/N13_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/N13_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N13_0_v2.pkl"


@register_task("maniskill.pick_single_egad_n131", "pick_single_egad_n131")
class PickSingleEgadN131Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/N13_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/N13_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N13_1_v2.pkl"


@register_task("maniskill.pick_single_egad_n133", "pick_single_egad_n133")
class PickSingleEgadN133Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/N13_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/N13_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N13_3_v2.pkl"


@register_task("maniskill.pick_single_egad_n143", "pick_single_egad_n143")
class PickSingleEgadN143Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/N14_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/N14_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N14_3_v2.pkl"


@register_task("maniskill.pick_single_egad_n150", "pick_single_egad_n150")
class PickSingleEgadN150Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/N15_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/N15_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N15_0_v2.pkl"


@register_task("maniskill.pick_single_egad_n151", "pick_single_egad_n151")
class PickSingleEgadN151Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/N15_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/N15_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N15_1_v2.pkl"


@register_task("maniskill.pick_single_egad_n152", "pick_single_egad_n152")
class PickSingleEgadN152Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/N15_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/N15_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N15_2_v2.pkl"


@register_task("maniskill.pick_single_egad_n153", "pick_single_egad_n153")
class PickSingleEgadN153Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/N15_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/N15_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N15_3_v2.pkl"


@register_task("maniskill.pick_single_egad_n160", "pick_single_egad_n160")
class PickSingleEgadN160Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/N16_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/N16_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N16_0_v2.pkl"


@register_task("maniskill.pick_single_egad_n161", "pick_single_egad_n161")
class PickSingleEgadN161Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/N16_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/N16_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N16_1_v2.pkl"


@register_task("maniskill.pick_single_egad_n162", "pick_single_egad_n162")
class PickSingleEgadN162Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/N16_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/N16_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N16_2_v2.pkl"


@register_task("maniskill.pick_single_egad_n163", "pick_single_egad_n163")
class PickSingleEgadN163Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/N16_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/N16_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N16_3_v2.pkl"


@register_task("maniskill.pick_single_egad_n170", "pick_single_egad_n170")
class PickSingleEgadN170Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/N17_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/N17_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N17_0_v2.pkl"


@register_task("maniskill.pick_single_egad_n171", "pick_single_egad_n171")
class PickSingleEgadN171Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/N17_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/N17_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N17_1_v2.pkl"


@register_task("maniskill.pick_single_egad_n172", "pick_single_egad_n172")
class PickSingleEgadN172Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/N17_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/N17_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N17_2_v2.pkl"


@register_task("maniskill.pick_single_egad_n173", "pick_single_egad_n173")
class PickSingleEgadN173Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/N17_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/N17_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N17_3_v2.pkl"


@register_task("maniskill.pick_single_egad_n180", "pick_single_egad_n180")
class PickSingleEgadN180Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/N18_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/N18_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N18_0_v2.pkl"


@register_task("maniskill.pick_single_egad_n181", "pick_single_egad_n181")
class PickSingleEgadN181Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/N18_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/N18_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N18_1_v2.pkl"


@register_task("maniskill.pick_single_egad_n182", "pick_single_egad_n182")
class PickSingleEgadN182Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/N18_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/N18_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N18_2_v2.pkl"


@register_task("maniskill.pick_single_egad_n183", "pick_single_egad_n183")
class PickSingleEgadN183Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/N18_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/N18_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N18_3_v2.pkl"


@register_task("maniskill.pick_single_egad_n190", "pick_single_egad_n190")
class PickSingleEgadN190Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/N19_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/N19_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N19_0_v2.pkl"


@register_task("maniskill.pick_single_egad_n191", "pick_single_egad_n191")
class PickSingleEgadN191Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/N19_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/N19_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N19_1_v2.pkl"


@register_task("maniskill.pick_single_egad_n192", "pick_single_egad_n192")
class PickSingleEgadN192Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/N19_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/N19_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N19_2_v2.pkl"


@register_task("maniskill.pick_single_egad_n193", "pick_single_egad_n193")
class PickSingleEgadN193Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/N19_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/N19_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N19_3_v2.pkl"


@register_task("maniskill.pick_single_egad_n200", "pick_single_egad_n200")
class PickSingleEgadN200Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/N20_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/N20_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N20_0_v2.pkl"


@register_task("maniskill.pick_single_egad_n201", "pick_single_egad_n201")
class PickSingleEgadN201Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/N20_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/N20_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N20_1_v2.pkl"


@register_task("maniskill.pick_single_egad_n202", "pick_single_egad_n202")
class PickSingleEgadN202Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/N20_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/N20_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N20_2_v2.pkl"


@register_task("maniskill.pick_single_egad_n203", "pick_single_egad_n203")
class PickSingleEgadN203Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/N20_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/N20_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N20_3_v2.pkl"


@register_task("maniskill.pick_single_egad_n210", "pick_single_egad_n210")
class PickSingleEgadN210Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/N21_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/N21_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N21_0_v2.pkl"


@register_task("maniskill.pick_single_egad_n211", "pick_single_egad_n211")
class PickSingleEgadN211Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/N21_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/N21_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N21_1_v2.pkl"


@register_task("maniskill.pick_single_egad_n212", "pick_single_egad_n212")
class PickSingleEgadN212Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/N21_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/N21_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N21_2_v2.pkl"


@register_task("maniskill.pick_single_egad_n213", "pick_single_egad_n213")
class PickSingleEgadN213Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/N21_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/N21_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N21_3_v2.pkl"


@register_task("maniskill.pick_single_egad_n220", "pick_single_egad_n220")
class PickSingleEgadN220Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/N22_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/N22_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N22_0_v2.pkl"


@register_task("maniskill.pick_single_egad_n222", "pick_single_egad_n222")
class PickSingleEgadN222Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/N22_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/N22_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N22_2_v2.pkl"


@register_task("maniskill.pick_single_egad_n223", "pick_single_egad_n223")
class PickSingleEgadN223Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/N22_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/N22_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N22_3_v2.pkl"


@register_task("maniskill.pick_single_egad_n230", "pick_single_egad_n230")
class PickSingleEgadN230Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/N23_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/N23_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N23_0_v2.pkl"


@register_task("maniskill.pick_single_egad_n231", "pick_single_egad_n231")
class PickSingleEgadN231Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/N23_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/N23_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N23_1_v2.pkl"


@register_task("maniskill.pick_single_egad_n233", "pick_single_egad_n233")
class PickSingleEgadN233Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/N23_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/N23_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N23_3_v2.pkl"


@register_task("maniskill.pick_single_egad_n240", "pick_single_egad_n240")
class PickSingleEgadN240Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/N24_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/N24_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N24_0_v2.pkl"


@register_task("maniskill.pick_single_egad_n241", "pick_single_egad_n241")
class PickSingleEgadN241Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/N24_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/N24_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N24_1_v2.pkl"


@register_task("maniskill.pick_single_egad_n242", "pick_single_egad_n242")
class PickSingleEgadN242Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/N24_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/N24_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N24_2_v2.pkl"


@register_task("maniskill.pick_single_egad_n243", "pick_single_egad_n243")
class PickSingleEgadN243Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/N24_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/N24_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N24_3_v2.pkl"


@register_task("maniskill.pick_single_egad_n250", "pick_single_egad_n250")
class PickSingleEgadN250Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/N25_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/N25_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N25_0_v2.pkl"


@register_task("maniskill.pick_single_egad_n251", "pick_single_egad_n251")
class PickSingleEgadN251Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/N25_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/N25_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N25_1_v2.pkl"


@register_task("maniskill.pick_single_egad_n252", "pick_single_egad_n252")
class PickSingleEgadN252Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/N25_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/N25_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N25_2_v2.pkl"


@register_task("maniskill.pick_single_egad_n253", "pick_single_egad_n253")
class PickSingleEgadN253Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/N25_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/N25_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N25_3_v2.pkl"


@register_task("maniskill.pick_single_egad_o050", "pick_single_egad_o050")
class PickSingleEgadO050Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/O05_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/O05_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O05_0_v2.pkl"


@register_task("maniskill.pick_single_egad_o051", "pick_single_egad_o051")
class PickSingleEgadO051Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/O05_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/O05_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O05_1_v2.pkl"


@register_task("maniskill.pick_single_egad_o053", "pick_single_egad_o053")
class PickSingleEgadO053Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/O05_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/O05_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O05_3_v2.pkl"


@register_task("maniskill.pick_single_egad_o060", "pick_single_egad_o060")
class PickSingleEgadO060Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/O06_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/O06_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O06_0_v2.pkl"


@register_task("maniskill.pick_single_egad_o061", "pick_single_egad_o061")
class PickSingleEgadO061Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/O06_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/O06_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O06_1_v2.pkl"


@register_task("maniskill.pick_single_egad_o062", "pick_single_egad_o062")
class PickSingleEgadO062Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/O06_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/O06_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O06_2_v2.pkl"


@register_task("maniskill.pick_single_egad_o063", "pick_single_egad_o063")
class PickSingleEgadO063Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/O06_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/O06_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O06_3_v2.pkl"


@register_task("maniskill.pick_single_egad_o070", "pick_single_egad_o070")
class PickSingleEgadO070Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/O07_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/O07_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O07_0_v2.pkl"


@register_task("maniskill.pick_single_egad_o072", "pick_single_egad_o072")
class PickSingleEgadO072Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/O07_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/O07_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O07_2_v2.pkl"


@register_task("maniskill.pick_single_egad_o073", "pick_single_egad_o073")
class PickSingleEgadO073Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/O07_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/O07_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O07_3_v2.pkl"


@register_task("maniskill.pick_single_egad_o080", "pick_single_egad_o080")
class PickSingleEgadO080Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/O08_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/O08_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O08_0_v2.pkl"


@register_task("maniskill.pick_single_egad_o081", "pick_single_egad_o081")
class PickSingleEgadO081Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/O08_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/O08_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O08_1_v2.pkl"


@register_task("maniskill.pick_single_egad_o082", "pick_single_egad_o082")
class PickSingleEgadO082Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/O08_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/O08_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O08_2_v2.pkl"


@register_task("maniskill.pick_single_egad_o083", "pick_single_egad_o083")
class PickSingleEgadO083Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/O08_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/O08_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O08_3_v2.pkl"


@register_task("maniskill.pick_single_egad_o090", "pick_single_egad_o090")
class PickSingleEgadO090Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/O09_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/O09_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O09_0_v2.pkl"


@register_task("maniskill.pick_single_egad_o091", "pick_single_egad_o091")
class PickSingleEgadO091Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/O09_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/O09_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O09_1_v2.pkl"


@register_task("maniskill.pick_single_egad_o092", "pick_single_egad_o092")
class PickSingleEgadO092Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/O09_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/O09_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O09_2_v2.pkl"


@register_task("maniskill.pick_single_egad_o093", "pick_single_egad_o093")
class PickSingleEgadO093Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/O09_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/O09_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O09_3_v2.pkl"


@register_task("maniskill.pick_single_egad_o100", "pick_single_egad_o100")
class PickSingleEgadO100Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/O10_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/O10_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O10_0_v2.pkl"


@register_task("maniskill.pick_single_egad_o101", "pick_single_egad_o101")
class PickSingleEgadO101Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/O10_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/O10_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O10_1_v2.pkl"


@register_task("maniskill.pick_single_egad_o102", "pick_single_egad_o102")
class PickSingleEgadO102Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/O10_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/O10_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O10_2_v2.pkl"


@register_task("maniskill.pick_single_egad_o103", "pick_single_egad_o103")
class PickSingleEgadO103Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/O10_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/O10_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O10_3_v2.pkl"


@register_task("maniskill.pick_single_egad_o110", "pick_single_egad_o110")
class PickSingleEgadO110Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/O11_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/O11_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O11_0_v2.pkl"


@register_task("maniskill.pick_single_egad_o112", "pick_single_egad_o112")
class PickSingleEgadO112Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/O11_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/O11_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O11_2_v2.pkl"


@register_task("maniskill.pick_single_egad_o113", "pick_single_egad_o113")
class PickSingleEgadO113Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/O11_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/O11_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O11_3_v2.pkl"


@register_task("maniskill.pick_single_egad_o120", "pick_single_egad_o120")
class PickSingleEgadO120Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/O12_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/O12_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O12_0_v2.pkl"


@register_task("maniskill.pick_single_egad_o121", "pick_single_egad_o121")
class PickSingleEgadO121Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/O12_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/O12_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O12_1_v2.pkl"


@register_task("maniskill.pick_single_egad_o122", "pick_single_egad_o122")
class PickSingleEgadO122Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/O12_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/O12_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O12_2_v2.pkl"


@register_task("maniskill.pick_single_egad_o123", "pick_single_egad_o123")
class PickSingleEgadO123Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/O12_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/O12_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O12_3_v2.pkl"


@register_task("maniskill.pick_single_egad_o130", "pick_single_egad_o130")
class PickSingleEgadO130Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/O13_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/O13_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O13_0_v2.pkl"


@register_task("maniskill.pick_single_egad_o131", "pick_single_egad_o131")
class PickSingleEgadO131Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/O13_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/O13_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O13_1_v2.pkl"


@register_task("maniskill.pick_single_egad_o132", "pick_single_egad_o132")
class PickSingleEgadO132Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/O13_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/O13_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O13_2_v2.pkl"


@register_task("maniskill.pick_single_egad_o133", "pick_single_egad_o133")
class PickSingleEgadO133Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/O13_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/O13_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O13_3_v2.pkl"


@register_task("maniskill.pick_single_egad_o140", "pick_single_egad_o140")
class PickSingleEgadO140Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/O14_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/O14_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O14_0_v2.pkl"


@register_task("maniskill.pick_single_egad_o141", "pick_single_egad_o141")
class PickSingleEgadO141Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/O14_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/O14_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O14_1_v2.pkl"


@register_task("maniskill.pick_single_egad_o142", "pick_single_egad_o142")
class PickSingleEgadO142Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/O14_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/O14_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O14_2_v2.pkl"


@register_task("maniskill.pick_single_egad_o150", "pick_single_egad_o150")
class PickSingleEgadO150Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/O15_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/O15_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O15_0_v2.pkl"


@register_task("maniskill.pick_single_egad_o151", "pick_single_egad_o151")
class PickSingleEgadO151Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/O15_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/O15_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O15_1_v2.pkl"


@register_task("maniskill.pick_single_egad_o152", "pick_single_egad_o152")
class PickSingleEgadO152Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/O15_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/O15_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O15_2_v2.pkl"


@register_task("maniskill.pick_single_egad_o153", "pick_single_egad_o153")
class PickSingleEgadO153Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/O15_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/O15_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O15_3_v2.pkl"


@register_task("maniskill.pick_single_egad_o160", "pick_single_egad_o160")
class PickSingleEgadO160Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/O16_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/O16_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O16_0_v2.pkl"


@register_task("maniskill.pick_single_egad_o162", "pick_single_egad_o162")
class PickSingleEgadO162Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/O16_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/O16_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O16_2_v2.pkl"


@register_task("maniskill.pick_single_egad_o163", "pick_single_egad_o163")
class PickSingleEgadO163Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/O16_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/O16_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O16_3_v2.pkl"


@register_task("maniskill.pick_single_egad_o170", "pick_single_egad_o170")
class PickSingleEgadO170Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/O17_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/O17_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O17_0_v2.pkl"


@register_task("maniskill.pick_single_egad_o171", "pick_single_egad_o171")
class PickSingleEgadO171Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/O17_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/O17_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O17_1_v2.pkl"


@register_task("maniskill.pick_single_egad_o172", "pick_single_egad_o172")
class PickSingleEgadO172Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/O17_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/O17_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O17_2_v2.pkl"


@register_task("maniskill.pick_single_egad_o173", "pick_single_egad_o173")
class PickSingleEgadO173Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/O17_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/O17_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O17_3_v2.pkl"


@register_task("maniskill.pick_single_egad_o180", "pick_single_egad_o180")
class PickSingleEgadO180Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/O18_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/O18_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O18_0_v2.pkl"


@register_task("maniskill.pick_single_egad_o181", "pick_single_egad_o181")
class PickSingleEgadO181Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/O18_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/O18_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O18_1_v2.pkl"


@register_task("maniskill.pick_single_egad_o182", "pick_single_egad_o182")
class PickSingleEgadO182Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/O18_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/O18_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O18_2_v2.pkl"


@register_task("maniskill.pick_single_egad_o183", "pick_single_egad_o183")
class PickSingleEgadO183Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/O18_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/O18_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O18_3_v2.pkl"


@register_task("maniskill.pick_single_egad_o190", "pick_single_egad_o190")
class PickSingleEgadO190Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/O19_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/O19_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O19_0_v2.pkl"


@register_task("maniskill.pick_single_egad_o191", "pick_single_egad_o191")
class PickSingleEgadO191Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/O19_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/O19_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O19_1_v2.pkl"


@register_task("maniskill.pick_single_egad_o192", "pick_single_egad_o192")
class PickSingleEgadO192Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/O19_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/O19_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O19_2_v2.pkl"


@register_task("maniskill.pick_single_egad_o193", "pick_single_egad_o193")
class PickSingleEgadO193Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/O19_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/O19_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O19_3_v2.pkl"


@register_task("maniskill.pick_single_egad_o200", "pick_single_egad_o200")
class PickSingleEgadO200Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/O20_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/O20_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O20_0_v2.pkl"


@register_task("maniskill.pick_single_egad_o201", "pick_single_egad_o201")
class PickSingleEgadO201Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/O20_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/O20_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O20_1_v2.pkl"


@register_task("maniskill.pick_single_egad_o202", "pick_single_egad_o202")
class PickSingleEgadO202Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/O20_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/O20_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O20_2_v2.pkl"


@register_task("maniskill.pick_single_egad_o203", "pick_single_egad_o203")
class PickSingleEgadO203Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/O20_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/O20_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O20_3_v2.pkl"


@register_task("maniskill.pick_single_egad_o211", "pick_single_egad_o211")
class PickSingleEgadO211Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/O21_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/O21_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O21_1_v2.pkl"


@register_task("maniskill.pick_single_egad_o212", "pick_single_egad_o212")
class PickSingleEgadO212Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/O21_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/O21_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O21_2_v2.pkl"


@register_task("maniskill.pick_single_egad_o213", "pick_single_egad_o213")
class PickSingleEgadO213Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/O21_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/O21_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O21_3_v2.pkl"


@register_task("maniskill.pick_single_egad_o220", "pick_single_egad_o220")
class PickSingleEgadO220Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/O22_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/O22_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O22_0_v2.pkl"


@register_task("maniskill.pick_single_egad_o221", "pick_single_egad_o221")
class PickSingleEgadO221Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/O22_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/O22_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O22_1_v2.pkl"


@register_task("maniskill.pick_single_egad_o222", "pick_single_egad_o222")
class PickSingleEgadO222Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/O22_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/O22_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O22_2_v2.pkl"


@register_task("maniskill.pick_single_egad_o223", "pick_single_egad_o223")
class PickSingleEgadO223Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/O22_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/O22_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O22_3_v2.pkl"


@register_task("maniskill.pick_single_egad_o230", "pick_single_egad_o230")
class PickSingleEgadO230Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/O23_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/O23_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O23_0_v2.pkl"


@register_task("maniskill.pick_single_egad_o231", "pick_single_egad_o231")
class PickSingleEgadO231Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/O23_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/O23_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O23_1_v2.pkl"


@register_task("maniskill.pick_single_egad_o232", "pick_single_egad_o232")
class PickSingleEgadO232Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/O23_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/O23_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O23_2_v2.pkl"


@register_task("maniskill.pick_single_egad_o233", "pick_single_egad_o233")
class PickSingleEgadO233Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/O23_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/O23_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O23_3_v2.pkl"


@register_task("maniskill.pick_single_egad_o240", "pick_single_egad_o240")
class PickSingleEgadO240Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/O24_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/O24_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O24_0_v2.pkl"


@register_task("maniskill.pick_single_egad_o241", "pick_single_egad_o241")
class PickSingleEgadO241Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/O24_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/O24_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O24_1_v2.pkl"


@register_task("maniskill.pick_single_egad_o242", "pick_single_egad_o242")
class PickSingleEgadO242Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/O24_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/O24_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O24_2_v2.pkl"


@register_task("maniskill.pick_single_egad_o243", "pick_single_egad_o243")
class PickSingleEgadO243Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/O24_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/O24_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O24_3_v2.pkl"


@register_task("maniskill.pick_single_egad_o250", "pick_single_egad_o250")
class PickSingleEgadO250Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/O25_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/O25_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O25_0_v2.pkl"


@register_task("maniskill.pick_single_egad_o251", "pick_single_egad_o251")
class PickSingleEgadO251Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/O25_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/O25_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O25_1_v2.pkl"


@register_task("maniskill.pick_single_egad_o252", "pick_single_egad_o252")
class PickSingleEgadO252Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/O25_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/O25_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O25_2_v2.pkl"


@register_task("maniskill.pick_single_egad_o253", "pick_single_egad_o253")
class PickSingleEgadO253Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/O25_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/O25_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O25_3_v2.pkl"


@register_task("maniskill.pick_single_egad_p050", "pick_single_egad_p050")
class PickSingleEgadP050Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/P05_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/P05_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P05_0_v2.pkl"


@register_task("maniskill.pick_single_egad_p051", "pick_single_egad_p051")
class PickSingleEgadP051Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/P05_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/P05_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P05_1_v2.pkl"


@register_task("maniskill.pick_single_egad_p052", "pick_single_egad_p052")
class PickSingleEgadP052Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/P05_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/P05_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P05_2_v2.pkl"


@register_task("maniskill.pick_single_egad_p053", "pick_single_egad_p053")
class PickSingleEgadP053Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/P05_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/P05_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P05_3_v2.pkl"


@register_task("maniskill.pick_single_egad_p060", "pick_single_egad_p060")
class PickSingleEgadP060Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/P06_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/P06_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P06_0_v2.pkl"


@register_task("maniskill.pick_single_egad_p061", "pick_single_egad_p061")
class PickSingleEgadP061Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/P06_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/P06_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P06_1_v2.pkl"


@register_task("maniskill.pick_single_egad_p062", "pick_single_egad_p062")
class PickSingleEgadP062Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/P06_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/P06_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P06_2_v2.pkl"


@register_task("maniskill.pick_single_egad_p070", "pick_single_egad_p070")
class PickSingleEgadP070Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/P07_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/P07_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P07_0_v2.pkl"


@register_task("maniskill.pick_single_egad_p071", "pick_single_egad_p071")
class PickSingleEgadP071Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/P07_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/P07_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P07_1_v2.pkl"


@register_task("maniskill.pick_single_egad_p072", "pick_single_egad_p072")
class PickSingleEgadP072Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/P07_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/P07_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P07_2_v2.pkl"


@register_task("maniskill.pick_single_egad_p080", "pick_single_egad_p080")
class PickSingleEgadP080Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/P08_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/P08_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P08_0_v2.pkl"


@register_task("maniskill.pick_single_egad_p081", "pick_single_egad_p081")
class PickSingleEgadP081Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/P08_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/P08_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P08_1_v2.pkl"


@register_task("maniskill.pick_single_egad_p083", "pick_single_egad_p083")
class PickSingleEgadP083Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/P08_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/P08_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P08_3_v2.pkl"


@register_task("maniskill.pick_single_egad_p090", "pick_single_egad_p090")
class PickSingleEgadP090Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/P09_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/P09_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P09_0_v2.pkl"


@register_task("maniskill.pick_single_egad_p091", "pick_single_egad_p091")
class PickSingleEgadP091Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/P09_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/P09_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P09_1_v2.pkl"


@register_task("maniskill.pick_single_egad_p092", "pick_single_egad_p092")
class PickSingleEgadP092Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/P09_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/P09_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P09_2_v2.pkl"


@register_task("maniskill.pick_single_egad_p093", "pick_single_egad_p093")
class PickSingleEgadP093Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/P09_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/P09_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P09_3_v2.pkl"


@register_task("maniskill.pick_single_egad_p100", "pick_single_egad_p100")
class PickSingleEgadP100Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/P10_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/P10_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P10_0_v2.pkl"


@register_task("maniskill.pick_single_egad_p101", "pick_single_egad_p101")
class PickSingleEgadP101Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/P10_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/P10_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P10_1_v2.pkl"


@register_task("maniskill.pick_single_egad_p102", "pick_single_egad_p102")
class PickSingleEgadP102Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/P10_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/P10_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P10_2_v2.pkl"


@register_task("maniskill.pick_single_egad_p103", "pick_single_egad_p103")
class PickSingleEgadP103Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/P10_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/P10_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P10_3_v2.pkl"


@register_task("maniskill.pick_single_egad_p110", "pick_single_egad_p110")
class PickSingleEgadP110Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/P11_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/P11_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P11_0_v2.pkl"


@register_task("maniskill.pick_single_egad_p111", "pick_single_egad_p111")
class PickSingleEgadP111Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/P11_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/P11_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P11_1_v2.pkl"


@register_task("maniskill.pick_single_egad_p112", "pick_single_egad_p112")
class PickSingleEgadP112Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/P11_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/P11_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P11_2_v2.pkl"


@register_task("maniskill.pick_single_egad_p113", "pick_single_egad_p113")
class PickSingleEgadP113Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/P11_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/P11_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P11_3_v2.pkl"


@register_task("maniskill.pick_single_egad_p120", "pick_single_egad_p120")
class PickSingleEgadP120Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/P12_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/P12_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P12_0_v2.pkl"


@register_task("maniskill.pick_single_egad_p121", "pick_single_egad_p121")
class PickSingleEgadP121Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/P12_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/P12_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P12_1_v2.pkl"


@register_task("maniskill.pick_single_egad_p122", "pick_single_egad_p122")
class PickSingleEgadP122Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/P12_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/P12_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P12_2_v2.pkl"


@register_task("maniskill.pick_single_egad_p123", "pick_single_egad_p123")
class PickSingleEgadP123Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/P12_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/P12_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P12_3_v2.pkl"


@register_task("maniskill.pick_single_egad_p130", "pick_single_egad_p130")
class PickSingleEgadP130Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/P13_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/P13_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P13_0_v2.pkl"


@register_task("maniskill.pick_single_egad_p131", "pick_single_egad_p131")
class PickSingleEgadP131Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/P13_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/P13_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P13_1_v2.pkl"


@register_task("maniskill.pick_single_egad_p132", "pick_single_egad_p132")
class PickSingleEgadP132Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/P13_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/P13_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P13_2_v2.pkl"


@register_task("maniskill.pick_single_egad_p133", "pick_single_egad_p133")
class PickSingleEgadP133Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/P13_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/P13_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P13_3_v2.pkl"


@register_task("maniskill.pick_single_egad_p140", "pick_single_egad_p140")
class PickSingleEgadP140Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/P14_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/P14_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P14_0_v2.pkl"


@register_task("maniskill.pick_single_egad_p141", "pick_single_egad_p141")
class PickSingleEgadP141Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/P14_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/P14_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P14_1_v2.pkl"


@register_task("maniskill.pick_single_egad_p142", "pick_single_egad_p142")
class PickSingleEgadP142Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/P14_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/P14_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P14_2_v2.pkl"


@register_task("maniskill.pick_single_egad_p143", "pick_single_egad_p143")
class PickSingleEgadP143Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/P14_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/P14_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P14_3_v2.pkl"


@register_task("maniskill.pick_single_egad_p150", "pick_single_egad_p150")
class PickSingleEgadP150Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/P15_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/P15_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P15_0_v2.pkl"


@register_task("maniskill.pick_single_egad_p151", "pick_single_egad_p151")
class PickSingleEgadP151Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/P15_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/P15_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P15_1_v2.pkl"


@register_task("maniskill.pick_single_egad_p152", "pick_single_egad_p152")
class PickSingleEgadP152Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/P15_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/P15_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P15_2_v2.pkl"


@register_task("maniskill.pick_single_egad_p153", "pick_single_egad_p153")
class PickSingleEgadP153Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/P15_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/P15_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P15_3_v2.pkl"


@register_task("maniskill.pick_single_egad_p160", "pick_single_egad_p160")
class PickSingleEgadP160Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/P16_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/P16_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P16_0_v2.pkl"


@register_task("maniskill.pick_single_egad_p161", "pick_single_egad_p161")
class PickSingleEgadP161Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/P16_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/P16_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P16_1_v2.pkl"


@register_task("maniskill.pick_single_egad_p162", "pick_single_egad_p162")
class PickSingleEgadP162Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/P16_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/P16_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P16_2_v2.pkl"


@register_task("maniskill.pick_single_egad_p163", "pick_single_egad_p163")
class PickSingleEgadP163Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/P16_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/P16_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P16_3_v2.pkl"


@register_task("maniskill.pick_single_egad_p170", "pick_single_egad_p170")
class PickSingleEgadP170Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/P17_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/P17_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P17_0_v2.pkl"


@register_task("maniskill.pick_single_egad_p171", "pick_single_egad_p171")
class PickSingleEgadP171Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/P17_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/P17_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P17_1_v2.pkl"


@register_task("maniskill.pick_single_egad_p172", "pick_single_egad_p172")
class PickSingleEgadP172Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/P17_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/P17_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P17_2_v2.pkl"


@register_task("maniskill.pick_single_egad_p173", "pick_single_egad_p173")
class PickSingleEgadP173Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/P17_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/P17_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P17_3_v2.pkl"


@register_task("maniskill.pick_single_egad_p180", "pick_single_egad_p180")
class PickSingleEgadP180Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/P18_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/P18_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P18_0_v2.pkl"


@register_task("maniskill.pick_single_egad_p181", "pick_single_egad_p181")
class PickSingleEgadP181Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/P18_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/P18_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P18_1_v2.pkl"


@register_task("maniskill.pick_single_egad_p182", "pick_single_egad_p182")
class PickSingleEgadP182Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/P18_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/P18_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P18_2_v2.pkl"


@register_task("maniskill.pick_single_egad_p183", "pick_single_egad_p183")
class PickSingleEgadP183Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/P18_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/P18_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P18_3_v2.pkl"


@register_task("maniskill.pick_single_egad_p190", "pick_single_egad_p190")
class PickSingleEgadP190Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/P19_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/P19_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P19_0_v2.pkl"


@register_task("maniskill.pick_single_egad_p191", "pick_single_egad_p191")
class PickSingleEgadP191Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/P19_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/P19_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P19_1_v2.pkl"


@register_task("maniskill.pick_single_egad_p192", "pick_single_egad_p192")
class PickSingleEgadP192Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/P19_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/P19_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P19_2_v2.pkl"


@register_task("maniskill.pick_single_egad_p200", "pick_single_egad_p200")
class PickSingleEgadP200Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/P20_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/P20_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P20_0_v2.pkl"


@register_task("maniskill.pick_single_egad_p202", "pick_single_egad_p202")
class PickSingleEgadP202Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/P20_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/P20_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P20_2_v2.pkl"


@register_task("maniskill.pick_single_egad_p203", "pick_single_egad_p203")
class PickSingleEgadP203Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/P20_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/P20_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P20_3_v2.pkl"


@register_task("maniskill.pick_single_egad_p211", "pick_single_egad_p211")
class PickSingleEgadP211Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/P21_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/P21_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P21_1_v2.pkl"


@register_task("maniskill.pick_single_egad_p212", "pick_single_egad_p212")
class PickSingleEgadP212Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/P21_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/P21_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P21_2_v2.pkl"


@register_task("maniskill.pick_single_egad_p213", "pick_single_egad_p213")
class PickSingleEgadP213Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/P21_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/P21_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P21_3_v2.pkl"


@register_task("maniskill.pick_single_egad_p220", "pick_single_egad_p220")
class PickSingleEgadP220Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/P22_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/P22_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P22_0_v2.pkl"


@register_task("maniskill.pick_single_egad_p221", "pick_single_egad_p221")
class PickSingleEgadP221Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/P22_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/P22_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P22_1_v2.pkl"


@register_task("maniskill.pick_single_egad_p222", "pick_single_egad_p222")
class PickSingleEgadP222Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/P22_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/P22_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P22_2_v2.pkl"


@register_task("maniskill.pick_single_egad_p223", "pick_single_egad_p223")
class PickSingleEgadP223Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/P22_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/P22_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P22_3_v2.pkl"


@register_task("maniskill.pick_single_egad_p230", "pick_single_egad_p230")
class PickSingleEgadP230Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/P23_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/P23_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P23_0_v2.pkl"


@register_task("maniskill.pick_single_egad_p231", "pick_single_egad_p231")
class PickSingleEgadP231Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/P23_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/P23_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P23_1_v2.pkl"


@register_task("maniskill.pick_single_egad_p232", "pick_single_egad_p232")
class PickSingleEgadP232Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/P23_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/P23_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P23_2_v2.pkl"


@register_task("maniskill.pick_single_egad_p233", "pick_single_egad_p233")
class PickSingleEgadP233Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/P23_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/P23_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P23_3_v2.pkl"


@register_task("maniskill.pick_single_egad_p240", "pick_single_egad_p240")
class PickSingleEgadP240Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/P24_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/P24_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P24_0_v2.pkl"


@register_task("maniskill.pick_single_egad_p241", "pick_single_egad_p241")
class PickSingleEgadP241Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/P24_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/P24_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P24_1_v2.pkl"


@register_task("maniskill.pick_single_egad_p242", "pick_single_egad_p242")
class PickSingleEgadP242Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/P24_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/P24_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P24_2_v2.pkl"


@register_task("maniskill.pick_single_egad_p243", "pick_single_egad_p243")
class PickSingleEgadP243Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/P24_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/P24_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P24_3_v2.pkl"


@register_task("maniskill.pick_single_egad_p250", "pick_single_egad_p250")
class PickSingleEgadP250Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/P25_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/P25_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P25_0_v2.pkl"


@register_task("maniskill.pick_single_egad_p251", "pick_single_egad_p251")
class PickSingleEgadP251Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/P25_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/P25_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P25_1_v2.pkl"


@register_task("maniskill.pick_single_egad_p252", "pick_single_egad_p252")
class PickSingleEgadP252Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/P25_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/P25_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P25_2_v2.pkl"


@register_task("maniskill.pick_single_egad_p253", "pick_single_egad_p253")
class PickSingleEgadP253Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/P25_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/P25_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P25_3_v2.pkl"


@register_task("maniskill.pick_single_egad_q050", "pick_single_egad_q050")
class PickSingleEgadQ050Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/Q05_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q05_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q05_0_v2.pkl"


@register_task("maniskill.pick_single_egad_q051", "pick_single_egad_q051")
class PickSingleEgadQ051Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/Q05_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q05_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q05_1_v2.pkl"


@register_task("maniskill.pick_single_egad_q052", "pick_single_egad_q052")
class PickSingleEgadQ052Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/Q05_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q05_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q05_2_v2.pkl"


@register_task("maniskill.pick_single_egad_q053", "pick_single_egad_q053")
class PickSingleEgadQ053Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/Q05_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q05_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q05_3_v2.pkl"


@register_task("maniskill.pick_single_egad_q062", "pick_single_egad_q062")
class PickSingleEgadQ062Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/Q06_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q06_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q06_2_v2.pkl"


@register_task("maniskill.pick_single_egad_q063", "pick_single_egad_q063")
class PickSingleEgadQ063Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/Q06_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q06_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q06_3_v2.pkl"


@register_task("maniskill.pick_single_egad_q070", "pick_single_egad_q070")
class PickSingleEgadQ070Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/Q07_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q07_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q07_0_v2.pkl"


@register_task("maniskill.pick_single_egad_q072", "pick_single_egad_q072")
class PickSingleEgadQ072Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/Q07_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q07_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q07_2_v2.pkl"


@register_task("maniskill.pick_single_egad_q073", "pick_single_egad_q073")
class PickSingleEgadQ073Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/Q07_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q07_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q07_3_v2.pkl"


@register_task("maniskill.pick_single_egad_q080", "pick_single_egad_q080")
class PickSingleEgadQ080Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/Q08_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q08_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q08_0_v2.pkl"


@register_task("maniskill.pick_single_egad_q081", "pick_single_egad_q081")
class PickSingleEgadQ081Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/Q08_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q08_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q08_1_v2.pkl"


@register_task("maniskill.pick_single_egad_q083", "pick_single_egad_q083")
class PickSingleEgadQ083Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/Q08_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q08_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q08_3_v2.pkl"


@register_task("maniskill.pick_single_egad_q090", "pick_single_egad_q090")
class PickSingleEgadQ090Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/Q09_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q09_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q09_0_v2.pkl"


@register_task("maniskill.pick_single_egad_q091", "pick_single_egad_q091")
class PickSingleEgadQ091Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/Q09_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q09_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q09_1_v2.pkl"


@register_task("maniskill.pick_single_egad_q092", "pick_single_egad_q092")
class PickSingleEgadQ092Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/Q09_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q09_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q09_2_v2.pkl"


@register_task("maniskill.pick_single_egad_q093", "pick_single_egad_q093")
class PickSingleEgadQ093Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/Q09_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q09_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q09_3_v2.pkl"


@register_task("maniskill.pick_single_egad_q100", "pick_single_egad_q100")
class PickSingleEgadQ100Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/Q10_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q10_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q10_0_v2.pkl"


@register_task("maniskill.pick_single_egad_q101", "pick_single_egad_q101")
class PickSingleEgadQ101Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/Q10_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q10_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q10_1_v2.pkl"


@register_task("maniskill.pick_single_egad_q102", "pick_single_egad_q102")
class PickSingleEgadQ102Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/Q10_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q10_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q10_2_v2.pkl"


@register_task("maniskill.pick_single_egad_q103", "pick_single_egad_q103")
class PickSingleEgadQ103Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/Q10_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q10_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q10_3_v2.pkl"


@register_task("maniskill.pick_single_egad_q110", "pick_single_egad_q110")
class PickSingleEgadQ110Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/Q11_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q11_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q11_0_v2.pkl"


@register_task("maniskill.pick_single_egad_q111", "pick_single_egad_q111")
class PickSingleEgadQ111Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/Q11_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q11_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q11_1_v2.pkl"


@register_task("maniskill.pick_single_egad_q112", "pick_single_egad_q112")
class PickSingleEgadQ112Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/Q11_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q11_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q11_2_v2.pkl"


@register_task("maniskill.pick_single_egad_q113", "pick_single_egad_q113")
class PickSingleEgadQ113Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/Q11_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q11_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q11_3_v2.pkl"


@register_task("maniskill.pick_single_egad_q120", "pick_single_egad_q120")
class PickSingleEgadQ120Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/Q12_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q12_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q12_0_v2.pkl"


@register_task("maniskill.pick_single_egad_q122", "pick_single_egad_q122")
class PickSingleEgadQ122Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/Q12_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q12_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q12_2_v2.pkl"


@register_task("maniskill.pick_single_egad_q123", "pick_single_egad_q123")
class PickSingleEgadQ123Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/Q12_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q12_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q12_3_v2.pkl"


@register_task("maniskill.pick_single_egad_q130", "pick_single_egad_q130")
class PickSingleEgadQ130Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/Q13_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q13_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q13_0_v2.pkl"


@register_task("maniskill.pick_single_egad_q131", "pick_single_egad_q131")
class PickSingleEgadQ131Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/Q13_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q13_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q13_1_v2.pkl"


@register_task("maniskill.pick_single_egad_q140", "pick_single_egad_q140")
class PickSingleEgadQ140Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/Q14_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q14_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q14_0_v2.pkl"


@register_task("maniskill.pick_single_egad_q141", "pick_single_egad_q141")
class PickSingleEgadQ141Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/Q14_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q14_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q14_1_v2.pkl"


@register_task("maniskill.pick_single_egad_q142", "pick_single_egad_q142")
class PickSingleEgadQ142Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/Q14_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q14_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q14_2_v2.pkl"


@register_task("maniskill.pick_single_egad_q143", "pick_single_egad_q143")
class PickSingleEgadQ143Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/Q14_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q14_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q14_3_v2.pkl"


@register_task("maniskill.pick_single_egad_q150", "pick_single_egad_q150")
class PickSingleEgadQ150Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/Q15_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q15_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q15_0_v2.pkl"


@register_task("maniskill.pick_single_egad_q151", "pick_single_egad_q151")
class PickSingleEgadQ151Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/Q15_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q15_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q15_1_v2.pkl"


@register_task("maniskill.pick_single_egad_q152", "pick_single_egad_q152")
class PickSingleEgadQ152Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/Q15_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q15_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q15_2_v2.pkl"


@register_task("maniskill.pick_single_egad_q153", "pick_single_egad_q153")
class PickSingleEgadQ153Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/Q15_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q15_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q15_3_v2.pkl"


@register_task("maniskill.pick_single_egad_q160", "pick_single_egad_q160")
class PickSingleEgadQ160Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/Q16_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q16_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q16_0_v2.pkl"


@register_task("maniskill.pick_single_egad_q161", "pick_single_egad_q161")
class PickSingleEgadQ161Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/Q16_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q16_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q16_1_v2.pkl"


@register_task("maniskill.pick_single_egad_q162", "pick_single_egad_q162")
class PickSingleEgadQ162Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/Q16_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q16_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q16_2_v2.pkl"


@register_task("maniskill.pick_single_egad_q163", "pick_single_egad_q163")
class PickSingleEgadQ163Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/Q16_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q16_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q16_3_v2.pkl"


@register_task("maniskill.pick_single_egad_q170", "pick_single_egad_q170")
class PickSingleEgadQ170Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/Q17_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q17_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q17_0_v2.pkl"


@register_task("maniskill.pick_single_egad_q171", "pick_single_egad_q171")
class PickSingleEgadQ171Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/Q17_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q17_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q17_1_v2.pkl"


@register_task("maniskill.pick_single_egad_q180", "pick_single_egad_q180")
class PickSingleEgadQ180Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/Q18_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q18_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q18_0_v2.pkl"


@register_task("maniskill.pick_single_egad_q181", "pick_single_egad_q181")
class PickSingleEgadQ181Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/Q18_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q18_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q18_1_v2.pkl"


@register_task("maniskill.pick_single_egad_q182", "pick_single_egad_q182")
class PickSingleEgadQ182Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/Q18_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q18_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q18_2_v2.pkl"


@register_task("maniskill.pick_single_egad_q183", "pick_single_egad_q183")
class PickSingleEgadQ183Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/Q18_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q18_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q18_3_v2.pkl"


@register_task("maniskill.pick_single_egad_q190", "pick_single_egad_q190")
class PickSingleEgadQ190Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/Q19_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q19_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q19_0_v2.pkl"


@register_task("maniskill.pick_single_egad_q191", "pick_single_egad_q191")
class PickSingleEgadQ191Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/Q19_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q19_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q19_1_v2.pkl"


@register_task("maniskill.pick_single_egad_q192", "pick_single_egad_q192")
class PickSingleEgadQ192Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/Q19_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q19_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q19_2_v2.pkl"


@register_task("maniskill.pick_single_egad_q193", "pick_single_egad_q193")
class PickSingleEgadQ193Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/Q19_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q19_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q19_3_v2.pkl"


@register_task("maniskill.pick_single_egad_q200", "pick_single_egad_q200")
class PickSingleEgadQ200Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/Q20_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q20_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q20_0_v2.pkl"


@register_task("maniskill.pick_single_egad_q202", "pick_single_egad_q202")
class PickSingleEgadQ202Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/Q20_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q20_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q20_2_v2.pkl"


@register_task("maniskill.pick_single_egad_q203", "pick_single_egad_q203")
class PickSingleEgadQ203Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/Q20_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q20_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q20_3_v2.pkl"


@register_task("maniskill.pick_single_egad_q210", "pick_single_egad_q210")
class PickSingleEgadQ210Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/Q21_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q21_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q21_0_v2.pkl"


@register_task("maniskill.pick_single_egad_q211", "pick_single_egad_q211")
class PickSingleEgadQ211Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/Q21_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q21_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q21_1_v2.pkl"


@register_task("maniskill.pick_single_egad_q212", "pick_single_egad_q212")
class PickSingleEgadQ212Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/Q21_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q21_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q21_2_v2.pkl"


@register_task("maniskill.pick_single_egad_q213", "pick_single_egad_q213")
class PickSingleEgadQ213Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/Q21_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q21_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q21_3_v2.pkl"


@register_task("maniskill.pick_single_egad_q220", "pick_single_egad_q220")
class PickSingleEgadQ220Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/Q22_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q22_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q22_0_v2.pkl"


@register_task("maniskill.pick_single_egad_q221", "pick_single_egad_q221")
class PickSingleEgadQ221Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/Q22_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q22_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q22_1_v2.pkl"


@register_task("maniskill.pick_single_egad_q222", "pick_single_egad_q222")
class PickSingleEgadQ222Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/Q22_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q22_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q22_2_v2.pkl"


@register_task("maniskill.pick_single_egad_q223", "pick_single_egad_q223")
class PickSingleEgadQ223Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/Q22_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q22_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q22_3_v2.pkl"


@register_task("maniskill.pick_single_egad_q230", "pick_single_egad_q230")
class PickSingleEgadQ230Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/Q23_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q23_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q23_0_v2.pkl"


@register_task("maniskill.pick_single_egad_q231", "pick_single_egad_q231")
class PickSingleEgadQ231Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/Q23_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q23_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q23_1_v2.pkl"


@register_task("maniskill.pick_single_egad_q232", "pick_single_egad_q232")
class PickSingleEgadQ232Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/Q23_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q23_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q23_2_v2.pkl"


@register_task("maniskill.pick_single_egad_q240", "pick_single_egad_q240")
class PickSingleEgadQ240Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/Q24_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q24_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q24_0_v2.pkl"


@register_task("maniskill.pick_single_egad_q241", "pick_single_egad_q241")
class PickSingleEgadQ241Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/Q24_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q24_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q24_1_v2.pkl"


@register_task("maniskill.pick_single_egad_q242", "pick_single_egad_q242")
class PickSingleEgadQ242Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/Q24_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q24_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q24_2_v2.pkl"


@register_task("maniskill.pick_single_egad_q243", "pick_single_egad_q243")
class PickSingleEgadQ243Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/Q24_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q24_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q24_3_v2.pkl"


@register_task("maniskill.pick_single_egad_q250", "pick_single_egad_q250")
class PickSingleEgadQ250Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/Q25_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q25_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q25_0_v2.pkl"


@register_task("maniskill.pick_single_egad_q251", "pick_single_egad_q251")
class PickSingleEgadQ251Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/Q25_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q25_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q25_1_v2.pkl"


@register_task("maniskill.pick_single_egad_q253", "pick_single_egad_q253")
class PickSingleEgadQ253Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/Q25_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q25_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q25_3_v2.pkl"


@register_task("maniskill.pick_single_egad_r050", "pick_single_egad_r050")
class PickSingleEgadR050Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/R05_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/R05_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R05_0_v2.pkl"


@register_task("maniskill.pick_single_egad_r052", "pick_single_egad_r052")
class PickSingleEgadR052Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/R05_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/R05_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R05_2_v2.pkl"


@register_task("maniskill.pick_single_egad_r053", "pick_single_egad_r053")
class PickSingleEgadR053Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/R05_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/R05_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R05_3_v2.pkl"


@register_task("maniskill.pick_single_egad_r060", "pick_single_egad_r060")
class PickSingleEgadR060Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/R06_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/R06_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R06_0_v2.pkl"


@register_task("maniskill.pick_single_egad_r061", "pick_single_egad_r061")
class PickSingleEgadR061Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/R06_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/R06_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R06_1_v2.pkl"


@register_task("maniskill.pick_single_egad_r062", "pick_single_egad_r062")
class PickSingleEgadR062Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/R06_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/R06_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R06_2_v2.pkl"


@register_task("maniskill.pick_single_egad_r063", "pick_single_egad_r063")
class PickSingleEgadR063Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/R06_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/R06_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R06_3_v2.pkl"


@register_task("maniskill.pick_single_egad_r070", "pick_single_egad_r070")
class PickSingleEgadR070Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/R07_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/R07_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R07_0_v2.pkl"


@register_task("maniskill.pick_single_egad_r071", "pick_single_egad_r071")
class PickSingleEgadR071Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/R07_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/R07_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R07_1_v2.pkl"


@register_task("maniskill.pick_single_egad_r073", "pick_single_egad_r073")
class PickSingleEgadR073Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/R07_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/R07_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R07_3_v2.pkl"


@register_task("maniskill.pick_single_egad_r081", "pick_single_egad_r081")
class PickSingleEgadR081Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/R08_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/R08_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R08_1_v2.pkl"


@register_task("maniskill.pick_single_egad_r082", "pick_single_egad_r082")
class PickSingleEgadR082Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/R08_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/R08_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R08_2_v2.pkl"


@register_task("maniskill.pick_single_egad_r083", "pick_single_egad_r083")
class PickSingleEgadR083Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/R08_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/R08_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R08_3_v2.pkl"


@register_task("maniskill.pick_single_egad_r090", "pick_single_egad_r090")
class PickSingleEgadR090Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/R09_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/R09_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R09_0_v2.pkl"


@register_task("maniskill.pick_single_egad_r093", "pick_single_egad_r093")
class PickSingleEgadR093Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/R09_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/R09_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R09_3_v2.pkl"


@register_task("maniskill.pick_single_egad_r100", "pick_single_egad_r100")
class PickSingleEgadR100Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/R10_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/R10_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R10_0_v2.pkl"


@register_task("maniskill.pick_single_egad_r101", "pick_single_egad_r101")
class PickSingleEgadR101Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/R10_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/R10_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R10_1_v2.pkl"


@register_task("maniskill.pick_single_egad_r103", "pick_single_egad_r103")
class PickSingleEgadR103Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/R10_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/R10_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R10_3_v2.pkl"


@register_task("maniskill.pick_single_egad_r110", "pick_single_egad_r110")
class PickSingleEgadR110Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/R11_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/R11_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R11_0_v2.pkl"


@register_task("maniskill.pick_single_egad_r111", "pick_single_egad_r111")
class PickSingleEgadR111Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/R11_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/R11_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R11_1_v2.pkl"


@register_task("maniskill.pick_single_egad_r112", "pick_single_egad_r112")
class PickSingleEgadR112Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/R11_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/R11_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R11_2_v2.pkl"


@register_task("maniskill.pick_single_egad_r113", "pick_single_egad_r113")
class PickSingleEgadR113Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/R11_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/R11_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R11_3_v2.pkl"


@register_task("maniskill.pick_single_egad_r121", "pick_single_egad_r121")
class PickSingleEgadR121Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/R12_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/R12_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R12_1_v2.pkl"


@register_task("maniskill.pick_single_egad_r122", "pick_single_egad_r122")
class PickSingleEgadR122Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/R12_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/R12_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R12_2_v2.pkl"


@register_task("maniskill.pick_single_egad_r123", "pick_single_egad_r123")
class PickSingleEgadR123Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/R12_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/R12_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R12_3_v2.pkl"


@register_task("maniskill.pick_single_egad_r130", "pick_single_egad_r130")
class PickSingleEgadR130Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/R13_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/R13_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R13_0_v2.pkl"


@register_task("maniskill.pick_single_egad_r131", "pick_single_egad_r131")
class PickSingleEgadR131Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/R13_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/R13_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R13_1_v2.pkl"


@register_task("maniskill.pick_single_egad_r133", "pick_single_egad_r133")
class PickSingleEgadR133Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/R13_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/R13_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R13_3_v2.pkl"


@register_task("maniskill.pick_single_egad_r140", "pick_single_egad_r140")
class PickSingleEgadR140Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/R14_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/R14_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R14_0_v2.pkl"


@register_task("maniskill.pick_single_egad_r141", "pick_single_egad_r141")
class PickSingleEgadR141Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/R14_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/R14_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R14_1_v2.pkl"


@register_task("maniskill.pick_single_egad_r143", "pick_single_egad_r143")
class PickSingleEgadR143Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/R14_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/R14_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R14_3_v2.pkl"


@register_task("maniskill.pick_single_egad_r150", "pick_single_egad_r150")
class PickSingleEgadR150Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/R15_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/R15_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R15_0_v2.pkl"


@register_task("maniskill.pick_single_egad_r151", "pick_single_egad_r151")
class PickSingleEgadR151Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/R15_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/R15_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R15_1_v2.pkl"


@register_task("maniskill.pick_single_egad_r152", "pick_single_egad_r152")
class PickSingleEgadR152Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/R15_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/R15_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R15_2_v2.pkl"


@register_task("maniskill.pick_single_egad_r153", "pick_single_egad_r153")
class PickSingleEgadR153Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/R15_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/R15_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R15_3_v2.pkl"


@register_task("maniskill.pick_single_egad_r161", "pick_single_egad_r161")
class PickSingleEgadR161Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/R16_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/R16_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R16_1_v2.pkl"


@register_task("maniskill.pick_single_egad_r162", "pick_single_egad_r162")
class PickSingleEgadR162Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/R16_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/R16_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R16_2_v2.pkl"


@register_task("maniskill.pick_single_egad_r170", "pick_single_egad_r170")
class PickSingleEgadR170Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/R17_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/R17_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R17_0_v2.pkl"


@register_task("maniskill.pick_single_egad_r171", "pick_single_egad_r171")
class PickSingleEgadR171Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/R17_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/R17_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R17_1_v2.pkl"


@register_task("maniskill.pick_single_egad_r172", "pick_single_egad_r172")
class PickSingleEgadR172Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/R17_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/R17_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R17_2_v2.pkl"


@register_task("maniskill.pick_single_egad_r173", "pick_single_egad_r173")
class PickSingleEgadR173Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/R17_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/R17_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R17_3_v2.pkl"


@register_task("maniskill.pick_single_egad_r180", "pick_single_egad_r180")
class PickSingleEgadR180Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/R18_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/R18_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R18_0_v2.pkl"


@register_task("maniskill.pick_single_egad_r181", "pick_single_egad_r181")
class PickSingleEgadR181Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/R18_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/R18_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R18_1_v2.pkl"


@register_task("maniskill.pick_single_egad_r182", "pick_single_egad_r182")
class PickSingleEgadR182Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/R18_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/R18_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R18_2_v2.pkl"


@register_task("maniskill.pick_single_egad_r191", "pick_single_egad_r191")
class PickSingleEgadR191Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/R19_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/R19_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R19_1_v2.pkl"


@register_task("maniskill.pick_single_egad_r193", "pick_single_egad_r193")
class PickSingleEgadR193Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/R19_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/R19_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R19_3_v2.pkl"


@register_task("maniskill.pick_single_egad_r200", "pick_single_egad_r200")
class PickSingleEgadR200Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/R20_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/R20_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R20_0_v2.pkl"


@register_task("maniskill.pick_single_egad_r201", "pick_single_egad_r201")
class PickSingleEgadR201Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/R20_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/R20_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R20_1_v2.pkl"


@register_task("maniskill.pick_single_egad_r202", "pick_single_egad_r202")
class PickSingleEgadR202Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/R20_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/R20_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R20_2_v2.pkl"


@register_task("maniskill.pick_single_egad_r203", "pick_single_egad_r203")
class PickSingleEgadR203Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/R20_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/R20_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R20_3_v2.pkl"


@register_task("maniskill.pick_single_egad_r210", "pick_single_egad_r210")
class PickSingleEgadR210Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/R21_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/R21_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R21_0_v2.pkl"


@register_task("maniskill.pick_single_egad_r211", "pick_single_egad_r211")
class PickSingleEgadR211Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/R21_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/R21_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R21_1_v2.pkl"


@register_task("maniskill.pick_single_egad_r212", "pick_single_egad_r212")
class PickSingleEgadR212Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/R21_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/R21_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R21_2_v2.pkl"


@register_task("maniskill.pick_single_egad_r213", "pick_single_egad_r213")
class PickSingleEgadR213Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/R21_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/R21_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R21_3_v2.pkl"


@register_task("maniskill.pick_single_egad_r220", "pick_single_egad_r220")
class PickSingleEgadR220Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/R22_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/R22_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R22_0_v2.pkl"


@register_task("maniskill.pick_single_egad_r221", "pick_single_egad_r221")
class PickSingleEgadR221Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/R22_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/R22_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R22_1_v2.pkl"


@register_task("maniskill.pick_single_egad_r222", "pick_single_egad_r222")
class PickSingleEgadR222Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/R22_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/R22_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R22_2_v2.pkl"


@register_task("maniskill.pick_single_egad_r223", "pick_single_egad_r223")
class PickSingleEgadR223Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/R22_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/R22_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R22_3_v2.pkl"


@register_task("maniskill.pick_single_egad_r230", "pick_single_egad_r230")
class PickSingleEgadR230Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/R23_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/R23_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R23_0_v2.pkl"


@register_task("maniskill.pick_single_egad_r231", "pick_single_egad_r231")
class PickSingleEgadR231Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/R23_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/R23_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R23_1_v2.pkl"


@register_task("maniskill.pick_single_egad_r232", "pick_single_egad_r232")
class PickSingleEgadR232Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/R23_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/R23_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R23_2_v2.pkl"


@register_task("maniskill.pick_single_egad_r233", "pick_single_egad_r233")
class PickSingleEgadR233Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/R23_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/R23_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R23_3_v2.pkl"


@register_task("maniskill.pick_single_egad_r240", "pick_single_egad_r240")
class PickSingleEgadR240Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/R24_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/R24_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R24_0_v2.pkl"


@register_task("maniskill.pick_single_egad_r241", "pick_single_egad_r241")
class PickSingleEgadR241Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/R24_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/R24_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R24_1_v2.pkl"


@register_task("maniskill.pick_single_egad_r242", "pick_single_egad_r242")
class PickSingleEgadR242Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/R24_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/R24_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R24_2_v2.pkl"


@register_task("maniskill.pick_single_egad_r243", "pick_single_egad_r243")
class PickSingleEgadR243Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/R24_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/R24_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R24_3_v2.pkl"


@register_task("maniskill.pick_single_egad_r250", "pick_single_egad_r250")
class PickSingleEgadR250Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/R25_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/R25_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R25_0_v2.pkl"


@register_task("maniskill.pick_single_egad_r251", "pick_single_egad_r251")
class PickSingleEgadR251Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/R25_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/R25_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R25_1_v2.pkl"


@register_task("maniskill.pick_single_egad_r252", "pick_single_egad_r252")
class PickSingleEgadR252Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/R25_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/R25_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R25_2_v2.pkl"


@register_task("maniskill.pick_single_egad_r253", "pick_single_egad_r253")
class PickSingleEgadR253Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/R25_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/R25_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R25_3_v2.pkl"


@register_task("maniskill.pick_single_egad_s040", "pick_single_egad_s040")
class PickSingleEgadS040Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/S04_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/S04_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S04_0_v2.pkl"


@register_task("maniskill.pick_single_egad_s041", "pick_single_egad_s041")
class PickSingleEgadS041Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/S04_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/S04_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S04_1_v2.pkl"


@register_task("maniskill.pick_single_egad_s043", "pick_single_egad_s043")
class PickSingleEgadS043Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/S04_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/S04_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S04_3_v2.pkl"


@register_task("maniskill.pick_single_egad_s051", "pick_single_egad_s051")
class PickSingleEgadS051Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/S05_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/S05_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S05_1_v2.pkl"


@register_task("maniskill.pick_single_egad_s052", "pick_single_egad_s052")
class PickSingleEgadS052Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/S05_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/S05_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S05_2_v2.pkl"


@register_task("maniskill.pick_single_egad_s053", "pick_single_egad_s053")
class PickSingleEgadS053Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/S05_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/S05_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S05_3_v2.pkl"


@register_task("maniskill.pick_single_egad_s060", "pick_single_egad_s060")
class PickSingleEgadS060Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/S06_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/S06_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S06_0_v2.pkl"


@register_task("maniskill.pick_single_egad_s061", "pick_single_egad_s061")
class PickSingleEgadS061Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/S06_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/S06_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S06_1_v2.pkl"


@register_task("maniskill.pick_single_egad_s062", "pick_single_egad_s062")
class PickSingleEgadS062Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/S06_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/S06_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S06_2_v2.pkl"


@register_task("maniskill.pick_single_egad_s070", "pick_single_egad_s070")
class PickSingleEgadS070Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/S07_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/S07_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S07_0_v2.pkl"


@register_task("maniskill.pick_single_egad_s071", "pick_single_egad_s071")
class PickSingleEgadS071Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/S07_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/S07_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S07_1_v2.pkl"


@register_task("maniskill.pick_single_egad_s072", "pick_single_egad_s072")
class PickSingleEgadS072Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/S07_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/S07_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S07_2_v2.pkl"


@register_task("maniskill.pick_single_egad_s080", "pick_single_egad_s080")
class PickSingleEgadS080Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/S08_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/S08_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S08_0_v2.pkl"


@register_task("maniskill.pick_single_egad_s081", "pick_single_egad_s081")
class PickSingleEgadS081Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/S08_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/S08_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S08_1_v2.pkl"


@register_task("maniskill.pick_single_egad_s082", "pick_single_egad_s082")
class PickSingleEgadS082Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/S08_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/S08_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S08_2_v2.pkl"


@register_task("maniskill.pick_single_egad_s091", "pick_single_egad_s091")
class PickSingleEgadS091Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/S09_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/S09_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S09_1_v2.pkl"


@register_task("maniskill.pick_single_egad_s092", "pick_single_egad_s092")
class PickSingleEgadS092Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/S09_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/S09_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S09_2_v2.pkl"


@register_task("maniskill.pick_single_egad_s101", "pick_single_egad_s101")
class PickSingleEgadS101Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/S10_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/S10_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S10_1_v2.pkl"


@register_task("maniskill.pick_single_egad_s102", "pick_single_egad_s102")
class PickSingleEgadS102Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/S10_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/S10_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S10_2_v2.pkl"


@register_task("maniskill.pick_single_egad_s103", "pick_single_egad_s103")
class PickSingleEgadS103Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/S10_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/S10_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S10_3_v2.pkl"


@register_task("maniskill.pick_single_egad_s110", "pick_single_egad_s110")
class PickSingleEgadS110Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/S11_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/S11_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S11_0_v2.pkl"


@register_task("maniskill.pick_single_egad_s111", "pick_single_egad_s111")
class PickSingleEgadS111Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/S11_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/S11_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S11_1_v2.pkl"


@register_task("maniskill.pick_single_egad_s113", "pick_single_egad_s113")
class PickSingleEgadS113Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/S11_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/S11_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S11_3_v2.pkl"


@register_task("maniskill.pick_single_egad_s120", "pick_single_egad_s120")
class PickSingleEgadS120Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/S12_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/S12_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S12_0_v2.pkl"


@register_task("maniskill.pick_single_egad_s121", "pick_single_egad_s121")
class PickSingleEgadS121Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/S12_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/S12_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S12_1_v2.pkl"


@register_task("maniskill.pick_single_egad_s122", "pick_single_egad_s122")
class PickSingleEgadS122Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/S12_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/S12_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S12_2_v2.pkl"


@register_task("maniskill.pick_single_egad_s123", "pick_single_egad_s123")
class PickSingleEgadS123Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/S12_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/S12_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S12_3_v2.pkl"


@register_task("maniskill.pick_single_egad_s130", "pick_single_egad_s130")
class PickSingleEgadS130Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/S13_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/S13_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S13_0_v2.pkl"


@register_task("maniskill.pick_single_egad_s131", "pick_single_egad_s131")
class PickSingleEgadS131Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/S13_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/S13_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S13_1_v2.pkl"


@register_task("maniskill.pick_single_egad_s132", "pick_single_egad_s132")
class PickSingleEgadS132Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/S13_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/S13_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S13_2_v2.pkl"


@register_task("maniskill.pick_single_egad_s133", "pick_single_egad_s133")
class PickSingleEgadS133Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/S13_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/S13_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S13_3_v2.pkl"


@register_task("maniskill.pick_single_egad_s140", "pick_single_egad_s140")
class PickSingleEgadS140Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/S14_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/S14_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S14_0_v2.pkl"


@register_task("maniskill.pick_single_egad_s141", "pick_single_egad_s141")
class PickSingleEgadS141Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/S14_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/S14_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S14_1_v2.pkl"


@register_task("maniskill.pick_single_egad_s142", "pick_single_egad_s142")
class PickSingleEgadS142Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/S14_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/S14_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S14_2_v2.pkl"


@register_task("maniskill.pick_single_egad_s150", "pick_single_egad_s150")
class PickSingleEgadS150Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/S15_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/S15_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S15_0_v2.pkl"


@register_task("maniskill.pick_single_egad_s151", "pick_single_egad_s151")
class PickSingleEgadS151Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/S15_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/S15_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S15_1_v2.pkl"


@register_task("maniskill.pick_single_egad_s153", "pick_single_egad_s153")
class PickSingleEgadS153Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/S15_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/S15_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S15_3_v2.pkl"


@register_task("maniskill.pick_single_egad_s160", "pick_single_egad_s160")
class PickSingleEgadS160Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/S16_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/S16_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S16_0_v2.pkl"


@register_task("maniskill.pick_single_egad_s161", "pick_single_egad_s161")
class PickSingleEgadS161Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/S16_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/S16_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S16_1_v2.pkl"


@register_task("maniskill.pick_single_egad_s163", "pick_single_egad_s163")
class PickSingleEgadS163Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/S16_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/S16_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S16_3_v2.pkl"


@register_task("maniskill.pick_single_egad_s170", "pick_single_egad_s170")
class PickSingleEgadS170Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/S17_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/S17_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S17_0_v2.pkl"


@register_task("maniskill.pick_single_egad_s171", "pick_single_egad_s171")
class PickSingleEgadS171Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/S17_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/S17_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S17_1_v2.pkl"


@register_task("maniskill.pick_single_egad_s172", "pick_single_egad_s172")
class PickSingleEgadS172Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/S17_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/S17_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S17_2_v2.pkl"


@register_task("maniskill.pick_single_egad_s180", "pick_single_egad_s180")
class PickSingleEgadS180Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/S18_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/S18_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S18_0_v2.pkl"


@register_task("maniskill.pick_single_egad_s181", "pick_single_egad_s181")
class PickSingleEgadS181Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/S18_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/S18_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S18_1_v2.pkl"


@register_task("maniskill.pick_single_egad_s183", "pick_single_egad_s183")
class PickSingleEgadS183Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/S18_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/S18_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S18_3_v2.pkl"


@register_task("maniskill.pick_single_egad_s190", "pick_single_egad_s190")
class PickSingleEgadS190Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/S19_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/S19_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S19_0_v2.pkl"


@register_task("maniskill.pick_single_egad_s191", "pick_single_egad_s191")
class PickSingleEgadS191Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/S19_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/S19_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S19_1_v2.pkl"


@register_task("maniskill.pick_single_egad_s192", "pick_single_egad_s192")
class PickSingleEgadS192Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/S19_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/S19_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S19_2_v2.pkl"


@register_task("maniskill.pick_single_egad_s193", "pick_single_egad_s193")
class PickSingleEgadS193Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/S19_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/S19_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S19_3_v2.pkl"


@register_task("maniskill.pick_single_egad_s200", "pick_single_egad_s200")
class PickSingleEgadS200Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/S20_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/S20_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S20_0_v2.pkl"


@register_task("maniskill.pick_single_egad_s201", "pick_single_egad_s201")
class PickSingleEgadS201Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/S20_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/S20_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S20_1_v2.pkl"


@register_task("maniskill.pick_single_egad_s202", "pick_single_egad_s202")
class PickSingleEgadS202Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/S20_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/S20_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S20_2_v2.pkl"


@register_task("maniskill.pick_single_egad_s203", "pick_single_egad_s203")
class PickSingleEgadS203Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/S20_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/S20_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S20_3_v2.pkl"


@register_task("maniskill.pick_single_egad_s210", "pick_single_egad_s210")
class PickSingleEgadS210Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/S21_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/S21_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S21_0_v2.pkl"


@register_task("maniskill.pick_single_egad_s211", "pick_single_egad_s211")
class PickSingleEgadS211Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/S21_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/S21_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S21_1_v2.pkl"


@register_task("maniskill.pick_single_egad_s212", "pick_single_egad_s212")
class PickSingleEgadS212Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/S21_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/S21_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S21_2_v2.pkl"


@register_task("maniskill.pick_single_egad_s213", "pick_single_egad_s213")
class PickSingleEgadS213Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/S21_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/S21_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S21_3_v2.pkl"


@register_task("maniskill.pick_single_egad_s220", "pick_single_egad_s220")
class PickSingleEgadS220Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/S22_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/S22_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S22_0_v2.pkl"


@register_task("maniskill.pick_single_egad_s221", "pick_single_egad_s221")
class PickSingleEgadS221Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/S22_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/S22_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S22_1_v2.pkl"


@register_task("maniskill.pick_single_egad_s222", "pick_single_egad_s222")
class PickSingleEgadS222Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/S22_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/S22_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S22_2_v2.pkl"


@register_task("maniskill.pick_single_egad_s223", "pick_single_egad_s223")
class PickSingleEgadS223Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/S22_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/S22_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S22_3_v2.pkl"


@register_task("maniskill.pick_single_egad_s230", "pick_single_egad_s230")
class PickSingleEgadS230Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/S23_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/S23_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S23_0_v2.pkl"


@register_task("maniskill.pick_single_egad_s231", "pick_single_egad_s231")
class PickSingleEgadS231Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/S23_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/S23_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S23_1_v2.pkl"


@register_task("maniskill.pick_single_egad_s232", "pick_single_egad_s232")
class PickSingleEgadS232Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/S23_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/S23_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S23_2_v2.pkl"


@register_task("maniskill.pick_single_egad_s233", "pick_single_egad_s233")
class PickSingleEgadS233Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/S23_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/S23_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S23_3_v2.pkl"


@register_task("maniskill.pick_single_egad_s240", "pick_single_egad_s240")
class PickSingleEgadS240Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/S24_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/S24_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S24_0_v2.pkl"


@register_task("maniskill.pick_single_egad_s241", "pick_single_egad_s241")
class PickSingleEgadS241Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/S24_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/S24_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S24_1_v2.pkl"


@register_task("maniskill.pick_single_egad_s242", "pick_single_egad_s242")
class PickSingleEgadS242Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/S24_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/S24_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S24_2_v2.pkl"


@register_task("maniskill.pick_single_egad_s243", "pick_single_egad_s243")
class PickSingleEgadS243Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/S24_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/S24_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S24_3_v2.pkl"


@register_task("maniskill.pick_single_egad_s250", "pick_single_egad_s250")
class PickSingleEgadS250Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/S25_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/S25_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S25_0_v2.pkl"


@register_task("maniskill.pick_single_egad_s251", "pick_single_egad_s251")
class PickSingleEgadS251Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/S25_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/S25_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S25_1_v2.pkl"


@register_task("maniskill.pick_single_egad_s252", "pick_single_egad_s252")
class PickSingleEgadS252Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/S25_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/S25_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S25_2_v2.pkl"


@register_task("maniskill.pick_single_egad_s253", "pick_single_egad_s253")
class PickSingleEgadS253Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/S25_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/S25_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S25_3_v2.pkl"


@register_task("maniskill.pick_single_egad_t041", "pick_single_egad_t041")
class PickSingleEgadT041Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/T04_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/T04_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T04_1_v2.pkl"


@register_task("maniskill.pick_single_egad_t043", "pick_single_egad_t043")
class PickSingleEgadT043Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/T04_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/T04_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T04_3_v2.pkl"


@register_task("maniskill.pick_single_egad_t050", "pick_single_egad_t050")
class PickSingleEgadT050Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/T05_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/T05_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T05_0_v2.pkl"


@register_task("maniskill.pick_single_egad_t051", "pick_single_egad_t051")
class PickSingleEgadT051Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/T05_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/T05_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T05_1_v2.pkl"


@register_task("maniskill.pick_single_egad_t052", "pick_single_egad_t052")
class PickSingleEgadT052Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/T05_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/T05_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T05_2_v2.pkl"


@register_task("maniskill.pick_single_egad_t053", "pick_single_egad_t053")
class PickSingleEgadT053Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/T05_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/T05_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T05_3_v2.pkl"


@register_task("maniskill.pick_single_egad_t060", "pick_single_egad_t060")
class PickSingleEgadT060Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/T06_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/T06_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T06_0_v2.pkl"


@register_task("maniskill.pick_single_egad_t061", "pick_single_egad_t061")
class PickSingleEgadT061Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/T06_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/T06_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T06_1_v2.pkl"


@register_task("maniskill.pick_single_egad_t062", "pick_single_egad_t062")
class PickSingleEgadT062Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/T06_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/T06_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T06_2_v2.pkl"


@register_task("maniskill.pick_single_egad_t063", "pick_single_egad_t063")
class PickSingleEgadT063Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/T06_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/T06_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T06_3_v2.pkl"


@register_task("maniskill.pick_single_egad_t070", "pick_single_egad_t070")
class PickSingleEgadT070Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/T07_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/T07_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T07_0_v2.pkl"


@register_task("maniskill.pick_single_egad_t071", "pick_single_egad_t071")
class PickSingleEgadT071Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/T07_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/T07_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T07_1_v2.pkl"


@register_task("maniskill.pick_single_egad_t072", "pick_single_egad_t072")
class PickSingleEgadT072Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/T07_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/T07_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T07_2_v2.pkl"


@register_task("maniskill.pick_single_egad_t073", "pick_single_egad_t073")
class PickSingleEgadT073Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/T07_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/T07_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T07_3_v2.pkl"


@register_task("maniskill.pick_single_egad_t080", "pick_single_egad_t080")
class PickSingleEgadT080Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/T08_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/T08_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T08_0_v2.pkl"


@register_task("maniskill.pick_single_egad_t081", "pick_single_egad_t081")
class PickSingleEgadT081Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/T08_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/T08_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T08_1_v2.pkl"


@register_task("maniskill.pick_single_egad_t082", "pick_single_egad_t082")
class PickSingleEgadT082Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/T08_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/T08_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T08_2_v2.pkl"


@register_task("maniskill.pick_single_egad_t083", "pick_single_egad_t083")
class PickSingleEgadT083Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/T08_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/T08_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T08_3_v2.pkl"


@register_task("maniskill.pick_single_egad_t090", "pick_single_egad_t090")
class PickSingleEgadT090Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/T09_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/T09_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T09_0_v2.pkl"


@register_task("maniskill.pick_single_egad_t091", "pick_single_egad_t091")
class PickSingleEgadT091Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/T09_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/T09_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T09_1_v2.pkl"


@register_task("maniskill.pick_single_egad_t100", "pick_single_egad_t100")
class PickSingleEgadT100Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/T10_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/T10_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T10_0_v2.pkl"


@register_task("maniskill.pick_single_egad_t102", "pick_single_egad_t102")
class PickSingleEgadT102Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/T10_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/T10_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T10_2_v2.pkl"


@register_task("maniskill.pick_single_egad_t103", "pick_single_egad_t103")
class PickSingleEgadT103Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/T10_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/T10_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T10_3_v2.pkl"


@register_task("maniskill.pick_single_egad_t110", "pick_single_egad_t110")
class PickSingleEgadT110Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/T11_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/T11_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T11_0_v2.pkl"


@register_task("maniskill.pick_single_egad_t111", "pick_single_egad_t111")
class PickSingleEgadT111Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/T11_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/T11_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T11_1_v2.pkl"


@register_task("maniskill.pick_single_egad_t112", "pick_single_egad_t112")
class PickSingleEgadT112Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/T11_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/T11_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T11_2_v2.pkl"


@register_task("maniskill.pick_single_egad_t120", "pick_single_egad_t120")
class PickSingleEgadT120Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/T12_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/T12_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T12_0_v2.pkl"


@register_task("maniskill.pick_single_egad_t121", "pick_single_egad_t121")
class PickSingleEgadT121Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/T12_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/T12_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T12_1_v2.pkl"


@register_task("maniskill.pick_single_egad_t122", "pick_single_egad_t122")
class PickSingleEgadT122Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/T12_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/T12_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T12_2_v2.pkl"


@register_task("maniskill.pick_single_egad_t123", "pick_single_egad_t123")
class PickSingleEgadT123Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/T12_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/T12_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T12_3_v2.pkl"


@register_task("maniskill.pick_single_egad_t130", "pick_single_egad_t130")
class PickSingleEgadT130Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/T13_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/T13_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T13_0_v2.pkl"


@register_task("maniskill.pick_single_egad_t131", "pick_single_egad_t131")
class PickSingleEgadT131Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/T13_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/T13_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T13_1_v2.pkl"


@register_task("maniskill.pick_single_egad_t132", "pick_single_egad_t132")
class PickSingleEgadT132Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/T13_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/T13_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T13_2_v2.pkl"


@register_task("maniskill.pick_single_egad_t140", "pick_single_egad_t140")
class PickSingleEgadT140Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/T14_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/T14_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T14_0_v2.pkl"


@register_task("maniskill.pick_single_egad_t141", "pick_single_egad_t141")
class PickSingleEgadT141Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/T14_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/T14_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T14_1_v2.pkl"


@register_task("maniskill.pick_single_egad_t143", "pick_single_egad_t143")
class PickSingleEgadT143Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/T14_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/T14_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T14_3_v2.pkl"


@register_task("maniskill.pick_single_egad_t151", "pick_single_egad_t151")
class PickSingleEgadT151Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/T15_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/T15_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T15_1_v2.pkl"


@register_task("maniskill.pick_single_egad_t152", "pick_single_egad_t152")
class PickSingleEgadT152Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/T15_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/T15_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T15_2_v2.pkl"


@register_task("maniskill.pick_single_egad_t153", "pick_single_egad_t153")
class PickSingleEgadT153Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/T15_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/T15_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T15_3_v2.pkl"


@register_task("maniskill.pick_single_egad_t160", "pick_single_egad_t160")
class PickSingleEgadT160Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/T16_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/T16_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T16_0_v2.pkl"


@register_task("maniskill.pick_single_egad_t161", "pick_single_egad_t161")
class PickSingleEgadT161Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/T16_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/T16_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T16_1_v2.pkl"


@register_task("maniskill.pick_single_egad_t163", "pick_single_egad_t163")
class PickSingleEgadT163Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/T16_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/T16_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T16_3_v2.pkl"


@register_task("maniskill.pick_single_egad_t171", "pick_single_egad_t171")
class PickSingleEgadT171Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/T17_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/T17_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T17_1_v2.pkl"


@register_task("maniskill.pick_single_egad_t172", "pick_single_egad_t172")
class PickSingleEgadT172Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/T17_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/T17_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T17_2_v2.pkl"


@register_task("maniskill.pick_single_egad_t173", "pick_single_egad_t173")
class PickSingleEgadT173Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/T17_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/T17_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T17_3_v2.pkl"


@register_task("maniskill.pick_single_egad_t180", "pick_single_egad_t180")
class PickSingleEgadT180Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/T18_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/T18_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T18_0_v2.pkl"


@register_task("maniskill.pick_single_egad_t181", "pick_single_egad_t181")
class PickSingleEgadT181Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/T18_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/T18_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T18_1_v2.pkl"


@register_task("maniskill.pick_single_egad_t182", "pick_single_egad_t182")
class PickSingleEgadT182Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/T18_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/T18_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T18_2_v2.pkl"


@register_task("maniskill.pick_single_egad_t183", "pick_single_egad_t183")
class PickSingleEgadT183Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/T18_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/T18_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T18_3_v2.pkl"


@register_task("maniskill.pick_single_egad_t190", "pick_single_egad_t190")
class PickSingleEgadT190Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/T19_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/T19_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T19_0_v2.pkl"


@register_task("maniskill.pick_single_egad_t191", "pick_single_egad_t191")
class PickSingleEgadT191Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/T19_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/T19_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T19_1_v2.pkl"


@register_task("maniskill.pick_single_egad_t192", "pick_single_egad_t192")
class PickSingleEgadT192Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/T19_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/T19_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T19_2_v2.pkl"


@register_task("maniskill.pick_single_egad_t193", "pick_single_egad_t193")
class PickSingleEgadT193Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/T19_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/T19_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T19_3_v2.pkl"


@register_task("maniskill.pick_single_egad_t200", "pick_single_egad_t200")
class PickSingleEgadT200Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/T20_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/T20_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T20_0_v2.pkl"


@register_task("maniskill.pick_single_egad_t201", "pick_single_egad_t201")
class PickSingleEgadT201Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/T20_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/T20_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T20_1_v2.pkl"


@register_task("maniskill.pick_single_egad_t202", "pick_single_egad_t202")
class PickSingleEgadT202Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/T20_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/T20_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T20_2_v2.pkl"


@register_task("maniskill.pick_single_egad_t203", "pick_single_egad_t203")
class PickSingleEgadT203Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/T20_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/T20_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T20_3_v2.pkl"


@register_task("maniskill.pick_single_egad_t210", "pick_single_egad_t210")
class PickSingleEgadT210Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/T21_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/T21_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T21_0_v2.pkl"


@register_task("maniskill.pick_single_egad_t211", "pick_single_egad_t211")
class PickSingleEgadT211Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/T21_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/T21_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T21_1_v2.pkl"


@register_task("maniskill.pick_single_egad_t212", "pick_single_egad_t212")
class PickSingleEgadT212Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/T21_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/T21_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T21_2_v2.pkl"


@register_task("maniskill.pick_single_egad_t213", "pick_single_egad_t213")
class PickSingleEgadT213Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/T21_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/T21_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T21_3_v2.pkl"


@register_task("maniskill.pick_single_egad_t220", "pick_single_egad_t220")
class PickSingleEgadT220Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/T22_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/T22_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T22_0_v2.pkl"


@register_task("maniskill.pick_single_egad_t221", "pick_single_egad_t221")
class PickSingleEgadT221Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/T22_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/T22_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T22_1_v2.pkl"


@register_task("maniskill.pick_single_egad_t222", "pick_single_egad_t222")
class PickSingleEgadT222Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/T22_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/T22_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T22_2_v2.pkl"


@register_task("maniskill.pick_single_egad_t223", "pick_single_egad_t223")
class PickSingleEgadT223Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/T22_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/T22_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T22_3_v2.pkl"


@register_task("maniskill.pick_single_egad_t230", "pick_single_egad_t230")
class PickSingleEgadT230Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/T23_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/T23_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T23_0_v2.pkl"


@register_task("maniskill.pick_single_egad_t231", "pick_single_egad_t231")
class PickSingleEgadT231Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/T23_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/T23_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T23_1_v2.pkl"


@register_task("maniskill.pick_single_egad_t232", "pick_single_egad_t232")
class PickSingleEgadT232Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/T23_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/T23_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T23_2_v2.pkl"


@register_task("maniskill.pick_single_egad_t233", "pick_single_egad_t233")
class PickSingleEgadT233Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/T23_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/T23_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T23_3_v2.pkl"


@register_task("maniskill.pick_single_egad_t240", "pick_single_egad_t240")
class PickSingleEgadT240Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/T24_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/T24_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T24_0_v2.pkl"


@register_task("maniskill.pick_single_egad_t241", "pick_single_egad_t241")
class PickSingleEgadT241Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/T24_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/T24_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T24_1_v2.pkl"


@register_task("maniskill.pick_single_egad_t242", "pick_single_egad_t242")
class PickSingleEgadT242Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/T24_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/T24_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T24_2_v2.pkl"


@register_task("maniskill.pick_single_egad_t243", "pick_single_egad_t243")
class PickSingleEgadT243Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/T24_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/T24_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T24_3_v2.pkl"


@register_task("maniskill.pick_single_egad_t251", "pick_single_egad_t251")
class PickSingleEgadT251Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/T25_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/T25_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T25_1_v2.pkl"


@register_task("maniskill.pick_single_egad_t252", "pick_single_egad_t252")
class PickSingleEgadT252Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/T25_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/T25_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T25_2_v2.pkl"


@register_task("maniskill.pick_single_egad_t253", "pick_single_egad_t253")
class PickSingleEgadT253Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/T25_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/T25_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T25_3_v2.pkl"


@register_task("maniskill.pick_single_egad_u020", "pick_single_egad_u020")
class PickSingleEgadU020Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/U02_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/U02_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U02_0_v2.pkl"


@register_task("maniskill.pick_single_egad_u023", "pick_single_egad_u023")
class PickSingleEgadU023Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/U02_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/U02_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U02_3_v2.pkl"


@register_task("maniskill.pick_single_egad_u030", "pick_single_egad_u030")
class PickSingleEgadU030Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/U03_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/U03_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U03_0_v2.pkl"


@register_task("maniskill.pick_single_egad_u031", "pick_single_egad_u031")
class PickSingleEgadU031Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/U03_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/U03_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U03_1_v2.pkl"


@register_task("maniskill.pick_single_egad_u033", "pick_single_egad_u033")
class PickSingleEgadU033Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/U03_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/U03_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U03_3_v2.pkl"


@register_task("maniskill.pick_single_egad_u040", "pick_single_egad_u040")
class PickSingleEgadU040Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/U04_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/U04_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U04_0_v2.pkl"


@register_task("maniskill.pick_single_egad_u043", "pick_single_egad_u043")
class PickSingleEgadU043Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/U04_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/U04_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U04_3_v2.pkl"


@register_task("maniskill.pick_single_egad_u050", "pick_single_egad_u050")
class PickSingleEgadU050Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/U05_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/U05_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U05_0_v2.pkl"


@register_task("maniskill.pick_single_egad_u052", "pick_single_egad_u052")
class PickSingleEgadU052Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/U05_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/U05_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U05_2_v2.pkl"


@register_task("maniskill.pick_single_egad_u060", "pick_single_egad_u060")
class PickSingleEgadU060Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/U06_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/U06_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U06_0_v2.pkl"


@register_task("maniskill.pick_single_egad_u061", "pick_single_egad_u061")
class PickSingleEgadU061Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/U06_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/U06_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U06_1_v2.pkl"


@register_task("maniskill.pick_single_egad_u063", "pick_single_egad_u063")
class PickSingleEgadU063Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/U06_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/U06_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U06_3_v2.pkl"


@register_task("maniskill.pick_single_egad_u070", "pick_single_egad_u070")
class PickSingleEgadU070Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/U07_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/U07_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U07_0_v2.pkl"


@register_task("maniskill.pick_single_egad_u081", "pick_single_egad_u081")
class PickSingleEgadU081Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/U08_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/U08_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U08_1_v2.pkl"


@register_task("maniskill.pick_single_egad_u083", "pick_single_egad_u083")
class PickSingleEgadU083Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/U08_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/U08_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U08_3_v2.pkl"


@register_task("maniskill.pick_single_egad_u090", "pick_single_egad_u090")
class PickSingleEgadU090Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/U09_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/U09_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U09_0_v2.pkl"


@register_task("maniskill.pick_single_egad_u093", "pick_single_egad_u093")
class PickSingleEgadU093Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/U09_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/U09_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U09_3_v2.pkl"


@register_task("maniskill.pick_single_egad_u100", "pick_single_egad_u100")
class PickSingleEgadU100Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/U10_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/U10_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U10_0_v2.pkl"


@register_task("maniskill.pick_single_egad_u101", "pick_single_egad_u101")
class PickSingleEgadU101Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/U10_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/U10_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U10_1_v2.pkl"


@register_task("maniskill.pick_single_egad_u103", "pick_single_egad_u103")
class PickSingleEgadU103Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/U10_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/U10_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U10_3_v2.pkl"


@register_task("maniskill.pick_single_egad_u111", "pick_single_egad_u111")
class PickSingleEgadU111Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/U11_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/U11_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U11_1_v2.pkl"


@register_task("maniskill.pick_single_egad_u112", "pick_single_egad_u112")
class PickSingleEgadU112Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/U11_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/U11_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U11_2_v2.pkl"


@register_task("maniskill.pick_single_egad_u113", "pick_single_egad_u113")
class PickSingleEgadU113Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/U11_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/U11_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U11_3_v2.pkl"


@register_task("maniskill.pick_single_egad_u122", "pick_single_egad_u122")
class PickSingleEgadU122Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/U12_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/U12_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U12_2_v2.pkl"


@register_task("maniskill.pick_single_egad_u123", "pick_single_egad_u123")
class PickSingleEgadU123Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/U12_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/U12_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U12_3_v2.pkl"


@register_task("maniskill.pick_single_egad_u130", "pick_single_egad_u130")
class PickSingleEgadU130Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/U13_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/U13_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U13_0_v2.pkl"


@register_task("maniskill.pick_single_egad_u131", "pick_single_egad_u131")
class PickSingleEgadU131Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/U13_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/U13_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U13_1_v2.pkl"


@register_task("maniskill.pick_single_egad_u132", "pick_single_egad_u132")
class PickSingleEgadU132Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/U13_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/U13_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U13_2_v2.pkl"


@register_task("maniskill.pick_single_egad_u140", "pick_single_egad_u140")
class PickSingleEgadU140Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/U14_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/U14_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U14_0_v2.pkl"


@register_task("maniskill.pick_single_egad_u141", "pick_single_egad_u141")
class PickSingleEgadU141Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/U14_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/U14_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U14_1_v2.pkl"


@register_task("maniskill.pick_single_egad_u143", "pick_single_egad_u143")
class PickSingleEgadU143Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/U14_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/U14_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U14_3_v2.pkl"


@register_task("maniskill.pick_single_egad_u150", "pick_single_egad_u150")
class PickSingleEgadU150Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/U15_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/U15_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U15_0_v2.pkl"


@register_task("maniskill.pick_single_egad_u151", "pick_single_egad_u151")
class PickSingleEgadU151Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/U15_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/U15_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U15_1_v2.pkl"


@register_task("maniskill.pick_single_egad_u152", "pick_single_egad_u152")
class PickSingleEgadU152Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/U15_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/U15_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U15_2_v2.pkl"


@register_task("maniskill.pick_single_egad_u160", "pick_single_egad_u160")
class PickSingleEgadU160Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/U16_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/U16_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U16_0_v2.pkl"


@register_task("maniskill.pick_single_egad_u161", "pick_single_egad_u161")
class PickSingleEgadU161Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/U16_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/U16_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U16_1_v2.pkl"


@register_task("maniskill.pick_single_egad_u162", "pick_single_egad_u162")
class PickSingleEgadU162Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/U16_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/U16_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U16_2_v2.pkl"


@register_task("maniskill.pick_single_egad_u163", "pick_single_egad_u163")
class PickSingleEgadU163Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/U16_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/U16_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U16_3_v2.pkl"


@register_task("maniskill.pick_single_egad_u170", "pick_single_egad_u170")
class PickSingleEgadU170Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/U17_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/U17_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U17_0_v2.pkl"


@register_task("maniskill.pick_single_egad_u171", "pick_single_egad_u171")
class PickSingleEgadU171Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/U17_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/U17_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U17_1_v2.pkl"


@register_task("maniskill.pick_single_egad_u172", "pick_single_egad_u172")
class PickSingleEgadU172Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/U17_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/U17_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U17_2_v2.pkl"


@register_task("maniskill.pick_single_egad_u173", "pick_single_egad_u173")
class PickSingleEgadU173Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/U17_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/U17_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U17_3_v2.pkl"


@register_task("maniskill.pick_single_egad_u181", "pick_single_egad_u181")
class PickSingleEgadU181Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/U18_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/U18_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U18_1_v2.pkl"


@register_task("maniskill.pick_single_egad_u182", "pick_single_egad_u182")
class PickSingleEgadU182Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/U18_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/U18_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U18_2_v2.pkl"


@register_task("maniskill.pick_single_egad_u183", "pick_single_egad_u183")
class PickSingleEgadU183Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/U18_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/U18_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U18_3_v2.pkl"


@register_task("maniskill.pick_single_egad_u190", "pick_single_egad_u190")
class PickSingleEgadU190Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/U19_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/U19_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U19_0_v2.pkl"


@register_task("maniskill.pick_single_egad_u191", "pick_single_egad_u191")
class PickSingleEgadU191Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/U19_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/U19_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U19_1_v2.pkl"


@register_task("maniskill.pick_single_egad_u192", "pick_single_egad_u192")
class PickSingleEgadU192Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/U19_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/U19_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U19_2_v2.pkl"


@register_task("maniskill.pick_single_egad_u193", "pick_single_egad_u193")
class PickSingleEgadU193Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/U19_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/U19_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U19_3_v2.pkl"


@register_task("maniskill.pick_single_egad_u200", "pick_single_egad_u200")
class PickSingleEgadU200Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/U20_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/U20_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U20_0_v2.pkl"


@register_task("maniskill.pick_single_egad_u201", "pick_single_egad_u201")
class PickSingleEgadU201Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/U20_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/U20_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U20_1_v2.pkl"


@register_task("maniskill.pick_single_egad_u202", "pick_single_egad_u202")
class PickSingleEgadU202Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/U20_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/U20_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U20_2_v2.pkl"


@register_task("maniskill.pick_single_egad_u203", "pick_single_egad_u203")
class PickSingleEgadU203Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/U20_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/U20_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U20_3_v2.pkl"


@register_task("maniskill.pick_single_egad_u210", "pick_single_egad_u210")
class PickSingleEgadU210Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/U21_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/U21_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U21_0_v2.pkl"


@register_task("maniskill.pick_single_egad_u211", "pick_single_egad_u211")
class PickSingleEgadU211Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/U21_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/U21_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U21_1_v2.pkl"


@register_task("maniskill.pick_single_egad_u212", "pick_single_egad_u212")
class PickSingleEgadU212Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/U21_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/U21_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U21_2_v2.pkl"


@register_task("maniskill.pick_single_egad_u213", "pick_single_egad_u213")
class PickSingleEgadU213Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/U21_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/U21_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U21_3_v2.pkl"


@register_task("maniskill.pick_single_egad_u221", "pick_single_egad_u221")
class PickSingleEgadU221Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/U22_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/U22_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U22_1_v2.pkl"


@register_task("maniskill.pick_single_egad_u222", "pick_single_egad_u222")
class PickSingleEgadU222Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/U22_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/U22_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U22_2_v2.pkl"


@register_task("maniskill.pick_single_egad_u230", "pick_single_egad_u230")
class PickSingleEgadU230Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/U23_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/U23_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U23_0_v2.pkl"


@register_task("maniskill.pick_single_egad_u231", "pick_single_egad_u231")
class PickSingleEgadU231Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/U23_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/U23_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U23_1_v2.pkl"


@register_task("maniskill.pick_single_egad_u233", "pick_single_egad_u233")
class PickSingleEgadU233Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/U23_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/U23_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U23_3_v2.pkl"


@register_task("maniskill.pick_single_egad_u241", "pick_single_egad_u241")
class PickSingleEgadU241Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/U24_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/U24_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U24_1_v2.pkl"


@register_task("maniskill.pick_single_egad_u242", "pick_single_egad_u242")
class PickSingleEgadU242Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/U24_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/U24_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U24_2_v2.pkl"


@register_task("maniskill.pick_single_egad_u243", "pick_single_egad_u243")
class PickSingleEgadU243Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/U24_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/U24_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U24_3_v2.pkl"


@register_task("maniskill.pick_single_egad_u250", "pick_single_egad_u250")
class PickSingleEgadU250Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/U25_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/U25_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U25_0_v2.pkl"


@register_task("maniskill.pick_single_egad_u251", "pick_single_egad_u251")
class PickSingleEgadU251Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/U25_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/U25_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U25_1_v2.pkl"


@register_task("maniskill.pick_single_egad_u252", "pick_single_egad_u252")
class PickSingleEgadU252Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/U25_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/U25_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U25_2_v2.pkl"


@register_task("maniskill.pick_single_egad_v020", "pick_single_egad_v020")
class PickSingleEgadV020Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/V02_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/V02_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V02_0_v2.pkl"


@register_task("maniskill.pick_single_egad_v021", "pick_single_egad_v021")
class PickSingleEgadV021Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/V02_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/V02_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V02_1_v2.pkl"


@register_task("maniskill.pick_single_egad_v022", "pick_single_egad_v022")
class PickSingleEgadV022Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/V02_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/V02_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V02_2_v2.pkl"


@register_task("maniskill.pick_single_egad_v023", "pick_single_egad_v023")
class PickSingleEgadV023Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/V02_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/V02_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V02_3_v2.pkl"


@register_task("maniskill.pick_single_egad_v031", "pick_single_egad_v031")
class PickSingleEgadV031Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/V03_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/V03_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V03_1_v2.pkl"


@register_task("maniskill.pick_single_egad_v033", "pick_single_egad_v033")
class PickSingleEgadV033Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/V03_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/V03_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V03_3_v2.pkl"


@register_task("maniskill.pick_single_egad_v041", "pick_single_egad_v041")
class PickSingleEgadV041Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/V04_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/V04_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V04_1_v2.pkl"


@register_task("maniskill.pick_single_egad_v042", "pick_single_egad_v042")
class PickSingleEgadV042Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/V04_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/V04_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V04_2_v2.pkl"


@register_task("maniskill.pick_single_egad_v050", "pick_single_egad_v050")
class PickSingleEgadV050Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/V05_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/V05_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V05_0_v2.pkl"


@register_task("maniskill.pick_single_egad_v052", "pick_single_egad_v052")
class PickSingleEgadV052Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/V05_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/V05_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V05_2_v2.pkl"


@register_task("maniskill.pick_single_egad_v053", "pick_single_egad_v053")
class PickSingleEgadV053Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/V05_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/V05_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V05_3_v2.pkl"


@register_task("maniskill.pick_single_egad_v060", "pick_single_egad_v060")
class PickSingleEgadV060Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/V06_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/V06_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V06_0_v2.pkl"


@register_task("maniskill.pick_single_egad_v061", "pick_single_egad_v061")
class PickSingleEgadV061Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/V06_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/V06_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V06_1_v2.pkl"


@register_task("maniskill.pick_single_egad_v063", "pick_single_egad_v063")
class PickSingleEgadV063Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/V06_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/V06_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V06_3_v2.pkl"


@register_task("maniskill.pick_single_egad_v070", "pick_single_egad_v070")
class PickSingleEgadV070Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/V07_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/V07_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V07_0_v2.pkl"


@register_task("maniskill.pick_single_egad_v072", "pick_single_egad_v072")
class PickSingleEgadV072Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/V07_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/V07_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V07_2_v2.pkl"


@register_task("maniskill.pick_single_egad_v073", "pick_single_egad_v073")
class PickSingleEgadV073Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/V07_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/V07_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V07_3_v2.pkl"


@register_task("maniskill.pick_single_egad_v080", "pick_single_egad_v080")
class PickSingleEgadV080Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/V08_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/V08_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V08_0_v2.pkl"


@register_task("maniskill.pick_single_egad_v082", "pick_single_egad_v082")
class PickSingleEgadV082Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/V08_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/V08_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V08_2_v2.pkl"


@register_task("maniskill.pick_single_egad_v092", "pick_single_egad_v092")
class PickSingleEgadV092Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/V09_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/V09_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V09_2_v2.pkl"


@register_task("maniskill.pick_single_egad_v100", "pick_single_egad_v100")
class PickSingleEgadV100Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/V10_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/V10_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V10_0_v2.pkl"


@register_task("maniskill.pick_single_egad_v101", "pick_single_egad_v101")
class PickSingleEgadV101Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/V10_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/V10_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V10_1_v2.pkl"


@register_task("maniskill.pick_single_egad_v102", "pick_single_egad_v102")
class PickSingleEgadV102Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/V10_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/V10_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V10_2_v2.pkl"


@register_task("maniskill.pick_single_egad_v103", "pick_single_egad_v103")
class PickSingleEgadV103Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/V10_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/V10_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V10_3_v2.pkl"


@register_task("maniskill.pick_single_egad_v110", "pick_single_egad_v110")
class PickSingleEgadV110Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/V11_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/V11_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V11_0_v2.pkl"


@register_task("maniskill.pick_single_egad_v111", "pick_single_egad_v111")
class PickSingleEgadV111Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/V11_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/V11_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V11_1_v2.pkl"


@register_task("maniskill.pick_single_egad_v112", "pick_single_egad_v112")
class PickSingleEgadV112Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/V11_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/V11_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V11_2_v2.pkl"


@register_task("maniskill.pick_single_egad_v113", "pick_single_egad_v113")
class PickSingleEgadV113Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/V11_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/V11_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V11_3_v2.pkl"


@register_task("maniskill.pick_single_egad_v121", "pick_single_egad_v121")
class PickSingleEgadV121Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/V12_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/V12_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V12_1_v2.pkl"


@register_task("maniskill.pick_single_egad_v122", "pick_single_egad_v122")
class PickSingleEgadV122Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/V12_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/V12_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V12_2_v2.pkl"


@register_task("maniskill.pick_single_egad_v123", "pick_single_egad_v123")
class PickSingleEgadV123Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/V12_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/V12_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V12_3_v2.pkl"


@register_task("maniskill.pick_single_egad_v130", "pick_single_egad_v130")
class PickSingleEgadV130Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/V13_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/V13_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V13_0_v2.pkl"


@register_task("maniskill.pick_single_egad_v131", "pick_single_egad_v131")
class PickSingleEgadV131Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/V13_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/V13_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V13_1_v2.pkl"


@register_task("maniskill.pick_single_egad_v132", "pick_single_egad_v132")
class PickSingleEgadV132Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/V13_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/V13_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V13_2_v2.pkl"


@register_task("maniskill.pick_single_egad_v133", "pick_single_egad_v133")
class PickSingleEgadV133Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/V13_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/V13_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V13_3_v2.pkl"


@register_task("maniskill.pick_single_egad_v140", "pick_single_egad_v140")
class PickSingleEgadV140Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/V14_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/V14_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V14_0_v2.pkl"


@register_task("maniskill.pick_single_egad_v141", "pick_single_egad_v141")
class PickSingleEgadV141Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/V14_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/V14_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V14_1_v2.pkl"


@register_task("maniskill.pick_single_egad_v150", "pick_single_egad_v150")
class PickSingleEgadV150Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/V15_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/V15_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V15_0_v2.pkl"


@register_task("maniskill.pick_single_egad_v151", "pick_single_egad_v151")
class PickSingleEgadV151Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/V15_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/V15_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V15_1_v2.pkl"


@register_task("maniskill.pick_single_egad_v153", "pick_single_egad_v153")
class PickSingleEgadV153Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/V15_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/V15_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V15_3_v2.pkl"


@register_task("maniskill.pick_single_egad_v160", "pick_single_egad_v160")
class PickSingleEgadV160Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/V16_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/V16_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V16_0_v2.pkl"


@register_task("maniskill.pick_single_egad_v162", "pick_single_egad_v162")
class PickSingleEgadV162Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/V16_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/V16_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V16_2_v2.pkl"


@register_task("maniskill.pick_single_egad_v171", "pick_single_egad_v171")
class PickSingleEgadV171Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/V17_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/V17_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V17_1_v2.pkl"


@register_task("maniskill.pick_single_egad_v172", "pick_single_egad_v172")
class PickSingleEgadV172Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/V17_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/V17_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V17_2_v2.pkl"


@register_task("maniskill.pick_single_egad_v173", "pick_single_egad_v173")
class PickSingleEgadV173Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/V17_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/V17_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V17_3_v2.pkl"


@register_task("maniskill.pick_single_egad_v181", "pick_single_egad_v181")
class PickSingleEgadV181Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/V18_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/V18_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V18_1_v2.pkl"


@register_task("maniskill.pick_single_egad_v182", "pick_single_egad_v182")
class PickSingleEgadV182Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/V18_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/V18_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V18_2_v2.pkl"


@register_task("maniskill.pick_single_egad_v190", "pick_single_egad_v190")
class PickSingleEgadV190Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/V19_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/V19_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V19_0_v2.pkl"


@register_task("maniskill.pick_single_egad_v191", "pick_single_egad_v191")
class PickSingleEgadV191Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/V19_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/V19_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V19_1_v2.pkl"


@register_task("maniskill.pick_single_egad_v192", "pick_single_egad_v192")
class PickSingleEgadV192Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/V19_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/V19_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V19_2_v2.pkl"


@register_task("maniskill.pick_single_egad_v193", "pick_single_egad_v193")
class PickSingleEgadV193Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/V19_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/V19_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V19_3_v2.pkl"


@register_task("maniskill.pick_single_egad_v200", "pick_single_egad_v200")
class PickSingleEgadV200Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/V20_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/V20_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V20_0_v2.pkl"


@register_task("maniskill.pick_single_egad_v201", "pick_single_egad_v201")
class PickSingleEgadV201Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/V20_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/V20_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V20_1_v2.pkl"


@register_task("maniskill.pick_single_egad_v202", "pick_single_egad_v202")
class PickSingleEgadV202Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/V20_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/V20_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V20_2_v2.pkl"


@register_task("maniskill.pick_single_egad_v203", "pick_single_egad_v203")
class PickSingleEgadV203Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/V20_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/V20_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V20_3_v2.pkl"


@register_task("maniskill.pick_single_egad_v210", "pick_single_egad_v210")
class PickSingleEgadV210Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/V21_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/V21_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V21_0_v2.pkl"


@register_task("maniskill.pick_single_egad_v211", "pick_single_egad_v211")
class PickSingleEgadV211Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/V21_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/V21_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V21_1_v2.pkl"


@register_task("maniskill.pick_single_egad_v213", "pick_single_egad_v213")
class PickSingleEgadV213Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/V21_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/V21_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V21_3_v2.pkl"


@register_task("maniskill.pick_single_egad_v221", "pick_single_egad_v221")
class PickSingleEgadV221Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/V22_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/V22_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V22_1_v2.pkl"


@register_task("maniskill.pick_single_egad_v222", "pick_single_egad_v222")
class PickSingleEgadV222Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/V22_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/V22_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V22_2_v2.pkl"


@register_task("maniskill.pick_single_egad_v223", "pick_single_egad_v223")
class PickSingleEgadV223Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/V22_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/V22_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V22_3_v2.pkl"


@register_task("maniskill.pick_single_egad_v230", "pick_single_egad_v230")
class PickSingleEgadV230Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/V23_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/V23_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V23_0_v2.pkl"


@register_task("maniskill.pick_single_egad_v232", "pick_single_egad_v232")
class PickSingleEgadV232Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/V23_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/V23_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V23_2_v2.pkl"


@register_task("maniskill.pick_single_egad_v233", "pick_single_egad_v233")
class PickSingleEgadV233Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/V23_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/V23_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V23_3_v2.pkl"


@register_task("maniskill.pick_single_egad_v240", "pick_single_egad_v240")
class PickSingleEgadV240Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/V24_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/V24_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V24_0_v2.pkl"


@register_task("maniskill.pick_single_egad_v241", "pick_single_egad_v241")
class PickSingleEgadV241Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/V24_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/V24_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V24_1_v2.pkl"


@register_task("maniskill.pick_single_egad_v242", "pick_single_egad_v242")
class PickSingleEgadV242Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/V24_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/V24_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V24_2_v2.pkl"


@register_task("maniskill.pick_single_egad_v250", "pick_single_egad_v250")
class PickSingleEgadV250Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/V25_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/V25_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V25_0_v2.pkl"


@register_task("maniskill.pick_single_egad_v251", "pick_single_egad_v251")
class PickSingleEgadV251Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/V25_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/V25_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V25_1_v2.pkl"


@register_task("maniskill.pick_single_egad_v252", "pick_single_egad_v252")
class PickSingleEgadV252Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/V25_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/V25_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V25_2_v2.pkl"


@register_task("maniskill.pick_single_egad_v253", "pick_single_egad_v253")
class PickSingleEgadV253Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/V25_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/V25_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V25_3_v2.pkl"


@register_task("maniskill.pick_single_egad_w020", "pick_single_egad_w020")
class PickSingleEgadW020Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/W02_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/W02_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W02_0_v2.pkl"


@register_task("maniskill.pick_single_egad_w030", "pick_single_egad_w030")
class PickSingleEgadW030Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/W03_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/W03_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W03_0_v2.pkl"


@register_task("maniskill.pick_single_egad_w033", "pick_single_egad_w033")
class PickSingleEgadW033Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/W03_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/W03_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W03_3_v2.pkl"


@register_task("maniskill.pick_single_egad_w040", "pick_single_egad_w040")
class PickSingleEgadW040Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/W04_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/W04_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W04_0_v2.pkl"


@register_task("maniskill.pick_single_egad_w041", "pick_single_egad_w041")
class PickSingleEgadW041Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/W04_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/W04_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W04_1_v2.pkl"


@register_task("maniskill.pick_single_egad_w042", "pick_single_egad_w042")
class PickSingleEgadW042Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/W04_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/W04_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W04_2_v2.pkl"


@register_task("maniskill.pick_single_egad_w050", "pick_single_egad_w050")
class PickSingleEgadW050Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/W05_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/W05_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W05_0_v2.pkl"


@register_task("maniskill.pick_single_egad_w052", "pick_single_egad_w052")
class PickSingleEgadW052Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/W05_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/W05_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W05_2_v2.pkl"


@register_task("maniskill.pick_single_egad_w053", "pick_single_egad_w053")
class PickSingleEgadW053Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/W05_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/W05_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W05_3_v2.pkl"


@register_task("maniskill.pick_single_egad_w060", "pick_single_egad_w060")
class PickSingleEgadW060Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/W06_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/W06_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W06_0_v2.pkl"


@register_task("maniskill.pick_single_egad_w062", "pick_single_egad_w062")
class PickSingleEgadW062Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/W06_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/W06_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W06_2_v2.pkl"


@register_task("maniskill.pick_single_egad_w063", "pick_single_egad_w063")
class PickSingleEgadW063Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/W06_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/W06_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W06_3_v2.pkl"


@register_task("maniskill.pick_single_egad_w070", "pick_single_egad_w070")
class PickSingleEgadW070Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/W07_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/W07_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W07_0_v2.pkl"


@register_task("maniskill.pick_single_egad_w071", "pick_single_egad_w071")
class PickSingleEgadW071Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/W07_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/W07_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W07_1_v2.pkl"


@register_task("maniskill.pick_single_egad_w073", "pick_single_egad_w073")
class PickSingleEgadW073Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/W07_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/W07_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W07_3_v2.pkl"


@register_task("maniskill.pick_single_egad_w081", "pick_single_egad_w081")
class PickSingleEgadW081Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/W08_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/W08_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W08_1_v2.pkl"


@register_task("maniskill.pick_single_egad_w082", "pick_single_egad_w082")
class PickSingleEgadW082Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/W08_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/W08_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W08_2_v2.pkl"


@register_task("maniskill.pick_single_egad_w083", "pick_single_egad_w083")
class PickSingleEgadW083Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/W08_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/W08_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W08_3_v2.pkl"


@register_task("maniskill.pick_single_egad_w090", "pick_single_egad_w090")
class PickSingleEgadW090Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/W09_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/W09_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W09_0_v2.pkl"


@register_task("maniskill.pick_single_egad_w091", "pick_single_egad_w091")
class PickSingleEgadW091Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/W09_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/W09_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W09_1_v2.pkl"


@register_task("maniskill.pick_single_egad_w092", "pick_single_egad_w092")
class PickSingleEgadW092Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/W09_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/W09_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W09_2_v2.pkl"


@register_task("maniskill.pick_single_egad_w093", "pick_single_egad_w093")
class PickSingleEgadW093Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/W09_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/W09_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W09_3_v2.pkl"


@register_task("maniskill.pick_single_egad_w101", "pick_single_egad_w101")
class PickSingleEgadW101Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/W10_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/W10_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W10_1_v2.pkl"


@register_task("maniskill.pick_single_egad_w110", "pick_single_egad_w110")
class PickSingleEgadW110Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/W11_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/W11_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W11_0_v2.pkl"


@register_task("maniskill.pick_single_egad_w111", "pick_single_egad_w111")
class PickSingleEgadW111Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/W11_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/W11_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W11_1_v2.pkl"


@register_task("maniskill.pick_single_egad_w113", "pick_single_egad_w113")
class PickSingleEgadW113Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/W11_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/W11_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W11_3_v2.pkl"


@register_task("maniskill.pick_single_egad_w121", "pick_single_egad_w121")
class PickSingleEgadW121Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/W12_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/W12_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W12_1_v2.pkl"


@register_task("maniskill.pick_single_egad_w122", "pick_single_egad_w122")
class PickSingleEgadW122Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/W12_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/W12_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W12_2_v2.pkl"


@register_task("maniskill.pick_single_egad_w123", "pick_single_egad_w123")
class PickSingleEgadW123Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/W12_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/W12_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W12_3_v2.pkl"


@register_task("maniskill.pick_single_egad_w130", "pick_single_egad_w130")
class PickSingleEgadW130Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/W13_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/W13_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W13_0_v2.pkl"


@register_task("maniskill.pick_single_egad_w131", "pick_single_egad_w131")
class PickSingleEgadW131Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/W13_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/W13_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W13_1_v2.pkl"


@register_task("maniskill.pick_single_egad_w132", "pick_single_egad_w132")
class PickSingleEgadW132Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/W13_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/W13_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W13_2_v2.pkl"


@register_task("maniskill.pick_single_egad_w133", "pick_single_egad_w133")
class PickSingleEgadW133Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/W13_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/W13_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W13_3_v2.pkl"


@register_task("maniskill.pick_single_egad_w140", "pick_single_egad_w140")
class PickSingleEgadW140Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/W14_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/W14_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W14_0_v2.pkl"


@register_task("maniskill.pick_single_egad_w143", "pick_single_egad_w143")
class PickSingleEgadW143Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/W14_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/W14_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W14_3_v2.pkl"


@register_task("maniskill.pick_single_egad_w150", "pick_single_egad_w150")
class PickSingleEgadW150Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/W15_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/W15_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W15_0_v2.pkl"


@register_task("maniskill.pick_single_egad_w151", "pick_single_egad_w151")
class PickSingleEgadW151Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/W15_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/W15_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W15_1_v2.pkl"


@register_task("maniskill.pick_single_egad_w153", "pick_single_egad_w153")
class PickSingleEgadW153Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/W15_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/W15_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W15_3_v2.pkl"


@register_task("maniskill.pick_single_egad_w160", "pick_single_egad_w160")
class PickSingleEgadW160Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/W16_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/W16_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W16_0_v2.pkl"


@register_task("maniskill.pick_single_egad_w161", "pick_single_egad_w161")
class PickSingleEgadW161Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/W16_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/W16_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W16_1_v2.pkl"


@register_task("maniskill.pick_single_egad_w162", "pick_single_egad_w162")
class PickSingleEgadW162Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/W16_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/W16_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W16_2_v2.pkl"


@register_task("maniskill.pick_single_egad_w163", "pick_single_egad_w163")
class PickSingleEgadW163Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/W16_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/W16_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W16_3_v2.pkl"


@register_task("maniskill.pick_single_egad_w170", "pick_single_egad_w170")
class PickSingleEgadW170Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/W17_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/W17_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W17_0_v2.pkl"


@register_task("maniskill.pick_single_egad_w171", "pick_single_egad_w171")
class PickSingleEgadW171Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/W17_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/W17_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W17_1_v2.pkl"


@register_task("maniskill.pick_single_egad_w172", "pick_single_egad_w172")
class PickSingleEgadW172Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/W17_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/W17_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W17_2_v2.pkl"


@register_task("maniskill.pick_single_egad_w173", "pick_single_egad_w173")
class PickSingleEgadW173Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/W17_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/W17_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W17_3_v2.pkl"


@register_task("maniskill.pick_single_egad_w180", "pick_single_egad_w180")
class PickSingleEgadW180Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/W18_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/W18_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W18_0_v2.pkl"


@register_task("maniskill.pick_single_egad_w182", "pick_single_egad_w182")
class PickSingleEgadW182Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/W18_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/W18_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W18_2_v2.pkl"


@register_task("maniskill.pick_single_egad_w183", "pick_single_egad_w183")
class PickSingleEgadW183Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/W18_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/W18_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W18_3_v2.pkl"


@register_task("maniskill.pick_single_egad_w190", "pick_single_egad_w190")
class PickSingleEgadW190Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/W19_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/W19_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W19_0_v2.pkl"


@register_task("maniskill.pick_single_egad_w192", "pick_single_egad_w192")
class PickSingleEgadW192Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/W19_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/W19_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W19_2_v2.pkl"


@register_task("maniskill.pick_single_egad_w193", "pick_single_egad_w193")
class PickSingleEgadW193Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/W19_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/W19_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W19_3_v2.pkl"


@register_task("maniskill.pick_single_egad_w200", "pick_single_egad_w200")
class PickSingleEgadW200Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/W20_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/W20_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W20_0_v2.pkl"


@register_task("maniskill.pick_single_egad_w201", "pick_single_egad_w201")
class PickSingleEgadW201Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/W20_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/W20_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W20_1_v2.pkl"


@register_task("maniskill.pick_single_egad_w202", "pick_single_egad_w202")
class PickSingleEgadW202Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/W20_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/W20_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W20_2_v2.pkl"


@register_task("maniskill.pick_single_egad_w203", "pick_single_egad_w203")
class PickSingleEgadW203Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/W20_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/W20_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W20_3_v2.pkl"


@register_task("maniskill.pick_single_egad_w210", "pick_single_egad_w210")
class PickSingleEgadW210Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/W21_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/W21_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W21_0_v2.pkl"


@register_task("maniskill.pick_single_egad_w211", "pick_single_egad_w211")
class PickSingleEgadW211Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/W21_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/W21_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W21_1_v2.pkl"


@register_task("maniskill.pick_single_egad_w220", "pick_single_egad_w220")
class PickSingleEgadW220Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/W22_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/W22_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W22_0_v2.pkl"


@register_task("maniskill.pick_single_egad_w221", "pick_single_egad_w221")
class PickSingleEgadW221Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/W22_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/W22_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W22_1_v2.pkl"


@register_task("maniskill.pick_single_egad_w223", "pick_single_egad_w223")
class PickSingleEgadW223Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/W22_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/W22_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W22_3_v2.pkl"


@register_task("maniskill.pick_single_egad_w230", "pick_single_egad_w230")
class PickSingleEgadW230Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/W23_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/W23_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W23_0_v2.pkl"


@register_task("maniskill.pick_single_egad_w231", "pick_single_egad_w231")
class PickSingleEgadW231Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/W23_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/W23_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W23_1_v2.pkl"


@register_task("maniskill.pick_single_egad_w232", "pick_single_egad_w232")
class PickSingleEgadW232Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/W23_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/W23_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W23_2_v2.pkl"


@register_task("maniskill.pick_single_egad_w233", "pick_single_egad_w233")
class PickSingleEgadW233Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/W23_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/W23_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W23_3_v2.pkl"


@register_task("maniskill.pick_single_egad_w240", "pick_single_egad_w240")
class PickSingleEgadW240Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/W24_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/W24_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W24_0_v2.pkl"


@register_task("maniskill.pick_single_egad_w241", "pick_single_egad_w241")
class PickSingleEgadW241Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/W24_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/W24_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W24_1_v2.pkl"


@register_task("maniskill.pick_single_egad_w242", "pick_single_egad_w242")
class PickSingleEgadW242Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/W24_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/W24_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W24_2_v2.pkl"


@register_task("maniskill.pick_single_egad_w243", "pick_single_egad_w243")
class PickSingleEgadW243Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/W24_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/W24_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W24_3_v2.pkl"


@register_task("maniskill.pick_single_egad_w250", "pick_single_egad_w250")
class PickSingleEgadW250Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/W25_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/W25_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W25_0_v2.pkl"


@register_task("maniskill.pick_single_egad_w251", "pick_single_egad_w251")
class PickSingleEgadW251Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/W25_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/W25_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W25_1_v2.pkl"


@register_task("maniskill.pick_single_egad_w252", "pick_single_egad_w252")
class PickSingleEgadW252Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/W25_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/W25_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W25_2_v2.pkl"


@register_task("maniskill.pick_single_egad_w253", "pick_single_egad_w253")
class PickSingleEgadW253Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/W25_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/W25_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W25_3_v2.pkl"


@register_task("maniskill.pick_single_egad_x000", "pick_single_egad_x000")
class PickSingleEgadX000Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/X00_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/X00_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X00_0_v2.pkl"


@register_task("maniskill.pick_single_egad_x010", "pick_single_egad_x010")
class PickSingleEgadX010Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/X01_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/X01_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X01_0_v2.pkl"


@register_task("maniskill.pick_single_egad_x011", "pick_single_egad_x011")
class PickSingleEgadX011Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/X01_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/X01_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X01_1_v2.pkl"


@register_task("maniskill.pick_single_egad_x012", "pick_single_egad_x012")
class PickSingleEgadX012Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/X01_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/X01_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X01_2_v2.pkl"


@register_task("maniskill.pick_single_egad_x013", "pick_single_egad_x013")
class PickSingleEgadX013Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/X01_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/X01_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X01_3_v2.pkl"


@register_task("maniskill.pick_single_egad_x020", "pick_single_egad_x020")
class PickSingleEgadX020Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/X02_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/X02_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X02_0_v2.pkl"


@register_task("maniskill.pick_single_egad_x021", "pick_single_egad_x021")
class PickSingleEgadX021Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/X02_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/X02_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X02_1_v2.pkl"


@register_task("maniskill.pick_single_egad_x022", "pick_single_egad_x022")
class PickSingleEgadX022Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/X02_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/X02_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X02_2_v2.pkl"


@register_task("maniskill.pick_single_egad_x023", "pick_single_egad_x023")
class PickSingleEgadX023Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/X02_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/X02_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X02_3_v2.pkl"


@register_task("maniskill.pick_single_egad_x030", "pick_single_egad_x030")
class PickSingleEgadX030Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/X03_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/X03_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X03_0_v2.pkl"


@register_task("maniskill.pick_single_egad_x032", "pick_single_egad_x032")
class PickSingleEgadX032Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/X03_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/X03_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X03_2_v2.pkl"


@register_task("maniskill.pick_single_egad_x033", "pick_single_egad_x033")
class PickSingleEgadX033Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/X03_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/X03_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X03_3_v2.pkl"


@register_task("maniskill.pick_single_egad_x040", "pick_single_egad_x040")
class PickSingleEgadX040Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/X04_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/X04_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X04_0_v2.pkl"


@register_task("maniskill.pick_single_egad_x041", "pick_single_egad_x041")
class PickSingleEgadX041Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/X04_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/X04_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X04_1_v2.pkl"


@register_task("maniskill.pick_single_egad_x042", "pick_single_egad_x042")
class PickSingleEgadX042Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/X04_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/X04_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X04_2_v2.pkl"


@register_task("maniskill.pick_single_egad_x050", "pick_single_egad_x050")
class PickSingleEgadX050Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/X05_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/X05_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X05_0_v2.pkl"


@register_task("maniskill.pick_single_egad_x052", "pick_single_egad_x052")
class PickSingleEgadX052Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/X05_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/X05_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X05_2_v2.pkl"


@register_task("maniskill.pick_single_egad_x060", "pick_single_egad_x060")
class PickSingleEgadX060Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/X06_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/X06_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X06_0_v2.pkl"


@register_task("maniskill.pick_single_egad_x063", "pick_single_egad_x063")
class PickSingleEgadX063Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/X06_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/X06_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X06_3_v2.pkl"


@register_task("maniskill.pick_single_egad_x070", "pick_single_egad_x070")
class PickSingleEgadX070Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/X07_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/X07_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X07_0_v2.pkl"


@register_task("maniskill.pick_single_egad_x071", "pick_single_egad_x071")
class PickSingleEgadX071Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/X07_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/X07_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X07_1_v2.pkl"


@register_task("maniskill.pick_single_egad_x072", "pick_single_egad_x072")
class PickSingleEgadX072Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/X07_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/X07_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X07_2_v2.pkl"


@register_task("maniskill.pick_single_egad_x081", "pick_single_egad_x081")
class PickSingleEgadX081Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/X08_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/X08_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X08_1_v2.pkl"


@register_task("maniskill.pick_single_egad_x082", "pick_single_egad_x082")
class PickSingleEgadX082Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/X08_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/X08_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X08_2_v2.pkl"


@register_task("maniskill.pick_single_egad_x090", "pick_single_egad_x090")
class PickSingleEgadX090Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/X09_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/X09_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X09_0_v2.pkl"


@register_task("maniskill.pick_single_egad_x091", "pick_single_egad_x091")
class PickSingleEgadX091Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/X09_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/X09_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X09_1_v2.pkl"


@register_task("maniskill.pick_single_egad_x092", "pick_single_egad_x092")
class PickSingleEgadX092Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/X09_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/X09_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X09_2_v2.pkl"


@register_task("maniskill.pick_single_egad_x093", "pick_single_egad_x093")
class PickSingleEgadX093Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/X09_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/X09_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X09_3_v2.pkl"


@register_task("maniskill.pick_single_egad_x100", "pick_single_egad_x100")
class PickSingleEgadX100Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/X10_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/X10_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X10_0_v2.pkl"


@register_task("maniskill.pick_single_egad_x102", "pick_single_egad_x102")
class PickSingleEgadX102Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/X10_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/X10_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X10_2_v2.pkl"


@register_task("maniskill.pick_single_egad_x103", "pick_single_egad_x103")
class PickSingleEgadX103Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/X10_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/X10_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X10_3_v2.pkl"


@register_task("maniskill.pick_single_egad_x110", "pick_single_egad_x110")
class PickSingleEgadX110Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/X11_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/X11_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X11_0_v2.pkl"


@register_task("maniskill.pick_single_egad_x111", "pick_single_egad_x111")
class PickSingleEgadX111Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/X11_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/X11_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X11_1_v2.pkl"


@register_task("maniskill.pick_single_egad_x112", "pick_single_egad_x112")
class PickSingleEgadX112Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/X11_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/X11_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X11_2_v2.pkl"


@register_task("maniskill.pick_single_egad_x113", "pick_single_egad_x113")
class PickSingleEgadX113Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/X11_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/X11_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X11_3_v2.pkl"


@register_task("maniskill.pick_single_egad_x120", "pick_single_egad_x120")
class PickSingleEgadX120Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/X12_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/X12_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X12_0_v2.pkl"


@register_task("maniskill.pick_single_egad_x121", "pick_single_egad_x121")
class PickSingleEgadX121Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/X12_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/X12_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X12_1_v2.pkl"


@register_task("maniskill.pick_single_egad_x122", "pick_single_egad_x122")
class PickSingleEgadX122Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/X12_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/X12_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X12_2_v2.pkl"


@register_task("maniskill.pick_single_egad_x123", "pick_single_egad_x123")
class PickSingleEgadX123Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/X12_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/X12_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X12_3_v2.pkl"


@register_task("maniskill.pick_single_egad_x130", "pick_single_egad_x130")
class PickSingleEgadX130Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/X13_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/X13_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X13_0_v2.pkl"


@register_task("maniskill.pick_single_egad_x131", "pick_single_egad_x131")
class PickSingleEgadX131Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/X13_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/X13_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X13_1_v2.pkl"


@register_task("maniskill.pick_single_egad_x132", "pick_single_egad_x132")
class PickSingleEgadX132Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/X13_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/X13_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X13_2_v2.pkl"


@register_task("maniskill.pick_single_egad_x140", "pick_single_egad_x140")
class PickSingleEgadX140Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/X14_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/X14_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X14_0_v2.pkl"


@register_task("maniskill.pick_single_egad_x141", "pick_single_egad_x141")
class PickSingleEgadX141Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/X14_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/X14_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X14_1_v2.pkl"


@register_task("maniskill.pick_single_egad_x142", "pick_single_egad_x142")
class PickSingleEgadX142Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/X14_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/X14_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X14_2_v2.pkl"


@register_task("maniskill.pick_single_egad_x143", "pick_single_egad_x143")
class PickSingleEgadX143Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/X14_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/X14_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X14_3_v2.pkl"


@register_task("maniskill.pick_single_egad_x150", "pick_single_egad_x150")
class PickSingleEgadX150Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/X15_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/X15_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X15_0_v2.pkl"


@register_task("maniskill.pick_single_egad_x151", "pick_single_egad_x151")
class PickSingleEgadX151Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/X15_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/X15_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X15_1_v2.pkl"


@register_task("maniskill.pick_single_egad_x152", "pick_single_egad_x152")
class PickSingleEgadX152Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/X15_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/X15_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X15_2_v2.pkl"


@register_task("maniskill.pick_single_egad_x153", "pick_single_egad_x153")
class PickSingleEgadX153Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/X15_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/X15_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X15_3_v2.pkl"


@register_task("maniskill.pick_single_egad_x161", "pick_single_egad_x161")
class PickSingleEgadX161Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/X16_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/X16_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X16_1_v2.pkl"


@register_task("maniskill.pick_single_egad_x170", "pick_single_egad_x170")
class PickSingleEgadX170Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/X17_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/X17_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X17_0_v2.pkl"


@register_task("maniskill.pick_single_egad_x171", "pick_single_egad_x171")
class PickSingleEgadX171Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/X17_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/X17_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X17_1_v2.pkl"


@register_task("maniskill.pick_single_egad_x172", "pick_single_egad_x172")
class PickSingleEgadX172Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/X17_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/X17_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X17_2_v2.pkl"


@register_task("maniskill.pick_single_egad_x173", "pick_single_egad_x173")
class PickSingleEgadX173Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/X17_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/X17_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X17_3_v2.pkl"


@register_task("maniskill.pick_single_egad_x181", "pick_single_egad_x181")
class PickSingleEgadX181Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/X18_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/X18_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X18_1_v2.pkl"


@register_task("maniskill.pick_single_egad_x182", "pick_single_egad_x182")
class PickSingleEgadX182Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/X18_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/X18_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X18_2_v2.pkl"


@register_task("maniskill.pick_single_egad_x190", "pick_single_egad_x190")
class PickSingleEgadX190Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/X19_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/X19_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X19_0_v2.pkl"


@register_task("maniskill.pick_single_egad_x191", "pick_single_egad_x191")
class PickSingleEgadX191Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/X19_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/X19_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X19_1_v2.pkl"


@register_task("maniskill.pick_single_egad_x192", "pick_single_egad_x192")
class PickSingleEgadX192Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/X19_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/X19_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X19_2_v2.pkl"


@register_task("maniskill.pick_single_egad_x200", "pick_single_egad_x200")
class PickSingleEgadX200Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/X20_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/X20_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X20_0_v2.pkl"


@register_task("maniskill.pick_single_egad_x201", "pick_single_egad_x201")
class PickSingleEgadX201Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/X20_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/X20_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X20_1_v2.pkl"


@register_task("maniskill.pick_single_egad_x202", "pick_single_egad_x202")
class PickSingleEgadX202Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/X20_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/X20_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X20_2_v2.pkl"


@register_task("maniskill.pick_single_egad_x210", "pick_single_egad_x210")
class PickSingleEgadX210Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/X21_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/X21_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X21_0_v2.pkl"


@register_task("maniskill.pick_single_egad_x211", "pick_single_egad_x211")
class PickSingleEgadX211Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/X21_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/X21_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X21_1_v2.pkl"


@register_task("maniskill.pick_single_egad_x212", "pick_single_egad_x212")
class PickSingleEgadX212Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/X21_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/X21_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X21_2_v2.pkl"


@register_task("maniskill.pick_single_egad_x213", "pick_single_egad_x213")
class PickSingleEgadX213Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/X21_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/X21_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X21_3_v2.pkl"


@register_task("maniskill.pick_single_egad_x220", "pick_single_egad_x220")
class PickSingleEgadX220Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/X22_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/X22_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X22_0_v2.pkl"


@register_task("maniskill.pick_single_egad_x222", "pick_single_egad_x222")
class PickSingleEgadX222Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/X22_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/X22_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X22_2_v2.pkl"


@register_task("maniskill.pick_single_egad_x223", "pick_single_egad_x223")
class PickSingleEgadX223Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/X22_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/X22_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X22_3_v2.pkl"


@register_task("maniskill.pick_single_egad_x231", "pick_single_egad_x231")
class PickSingleEgadX231Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/X23_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/X23_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X23_1_v2.pkl"


@register_task("maniskill.pick_single_egad_x232", "pick_single_egad_x232")
class PickSingleEgadX232Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/X23_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/X23_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X23_2_v2.pkl"


@register_task("maniskill.pick_single_egad_x233", "pick_single_egad_x233")
class PickSingleEgadX233Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/X23_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/X23_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X23_3_v2.pkl"


@register_task("maniskill.pick_single_egad_x240", "pick_single_egad_x240")
class PickSingleEgadX240Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/X24_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/X24_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X24_0_v2.pkl"


@register_task("maniskill.pick_single_egad_x241", "pick_single_egad_x241")
class PickSingleEgadX241Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/X24_1_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/X24_1.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X24_1_v2.pkl"


@register_task("maniskill.pick_single_egad_x242", "pick_single_egad_x242")
class PickSingleEgadX242Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/X24_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/X24_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X24_2_v2.pkl"


@register_task("maniskill.pick_single_egad_x243", "pick_single_egad_x243")
class PickSingleEgadX243Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/X24_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/X24_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X24_3_v2.pkl"


@register_task("maniskill.pick_single_egad_x250", "pick_single_egad_x250")
class PickSingleEgadX250Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/X25_0_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/X25_0.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X25_0_v2.pkl"


@register_task("maniskill.pick_single_egad_x252", "pick_single_egad_x252")
class PickSingleEgadX252Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/X25_2_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/X25_2.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X25_2_v2.pkl"


@register_task("maniskill.pick_single_egad_x253", "pick_single_egad_x253")
class PickSingleEgadX253Task(_PickSingleEgadBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/egad/usd/X25_3_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/egad/urdf/X25_3.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X25_3_v2.pkl"


"""The base class and derived classes for the pick up single EGAD object task from ManiSkill."""
