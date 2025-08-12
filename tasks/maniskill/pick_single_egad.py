"""The base class and derived classes for the pick up single EGAD object task from ManiSkill."""

from metasim.cfg.checkers import PositionShiftChecker
from metasim.cfg.objects import RigidObjCfg
from metasim.constants import PhysicStateType
from metasim.utils import configclass

from .maniskill_task_cfg import ManiskillTaskCfg


@configclass
class _PickSingleEgadBaseCfg(ManiskillTaskCfg):
    """The pickup single EGAD object task from ManiSkill.

    The robot is tasked to pick up an EGAD object.
    Note that the checker is not same as the original one (checking if the cube is near the target position).
    The current one checks if the cube is lifted up 7.5 cm.
    This class should be derived to specify the exact configuration (asset path and demo path) of the task.
    """

    episode_length = 250
    checker = PositionShiftChecker(
        obj_name="obj",
        distance=0.075,
        axis="z",
    )


@configclass
class PickSingleEgadA100Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/A10_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/A10_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-A10_0_v2.pkl"


@configclass
class PickSingleEgadA110Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/A11_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/A11_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-A11_0_v2.pkl"


@configclass
class PickSingleEgadA130Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/A13_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/A13_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-A13_0_v2.pkl"


@configclass
class PickSingleEgadA140Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/A14_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/A14_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-A14_0_v2.pkl"


@configclass
class PickSingleEgadA160Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/A16_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/A16_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-A16_0_v2.pkl"


@configclass
class PickSingleEgadA161Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/A16_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/A16_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-A16_1_v2.pkl"


@configclass
class PickSingleEgadA180Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/A18_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/A18_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-A18_0_v2.pkl"


@configclass
class PickSingleEgadA190Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/A19_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/A19_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-A19_0_v2.pkl"


@configclass
class PickSingleEgadA200Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/A20_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/A20_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-A20_0_v2.pkl"


@configclass
class PickSingleEgadA210Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/A21_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/A21_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-A21_0_v2.pkl"


@configclass
class PickSingleEgadA220Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/A22_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/A22_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-A22_0_v2.pkl"


@configclass
class PickSingleEgadA240Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/A24_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/A24_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-A24_0_v2.pkl"


@configclass
class PickSingleEgadB100Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/B10_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/B10_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-B10_0_v2.pkl"


@configclass
class PickSingleEgadB101Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/B10_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/B10_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-B10_1_v2.pkl"


@configclass
class PickSingleEgadB102Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/B10_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/B10_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-B10_2_v2.pkl"


@configclass
class PickSingleEgadB103Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/B10_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/B10_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-B10_3_v2.pkl"


@configclass
class PickSingleEgadB111Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/B11_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/B11_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-B11_1_v2.pkl"


@configclass
class PickSingleEgadB112Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/B11_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/B11_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-B11_2_v2.pkl"


@configclass
class PickSingleEgadB113Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/B11_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/B11_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-B11_3_v2.pkl"


@configclass
class PickSingleEgadB121Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/B12_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/B12_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-B12_1_v2.pkl"


@configclass
class PickSingleEgadB130Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/B13_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/B13_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-B13_0_v2.pkl"


@configclass
class PickSingleEgadB131Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/B13_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/B13_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-B13_1_v2.pkl"


@configclass
class PickSingleEgadB132Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/B13_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/B13_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-B13_2_v2.pkl"


@configclass
class PickSingleEgadB133Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/B13_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/B13_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-B13_3_v2.pkl"


@configclass
class PickSingleEgadB140Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/B14_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/B14_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-B14_0_v2.pkl"


@configclass
class PickSingleEgadB141Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/B14_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/B14_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-B14_1_v2.pkl"


@configclass
class PickSingleEgadB142Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/B14_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/B14_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-B14_2_v2.pkl"


@configclass
class PickSingleEgadB143Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/B14_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/B14_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-B14_3_v2.pkl"


@configclass
class PickSingleEgadB150Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/B15_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/B15_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-B15_0_v2.pkl"


@configclass
class PickSingleEgadB151Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/B15_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/B15_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-B15_1_v2.pkl"


@configclass
class PickSingleEgadB152Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/B15_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/B15_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-B15_2_v2.pkl"


@configclass
class PickSingleEgadB153Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/B15_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/B15_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-B15_3_v2.pkl"


@configclass
class PickSingleEgadB161Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/B16_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/B16_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-B16_1_v2.pkl"


@configclass
class PickSingleEgadB162Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/B16_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/B16_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-B16_2_v2.pkl"


@configclass
class PickSingleEgadB163Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/B16_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/B16_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-B16_3_v2.pkl"


@configclass
class PickSingleEgadB170Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/B17_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/B17_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-B17_0_v2.pkl"


@configclass
class PickSingleEgadB171Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/B17_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/B17_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-B17_1_v2.pkl"


@configclass
class PickSingleEgadB172Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/B17_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/B17_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-B17_2_v2.pkl"


@configclass
class PickSingleEgadB173Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/B17_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/B17_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-B17_3_v2.pkl"


@configclass
class PickSingleEgadB180Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/B18_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/B18_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-B18_0_v2.pkl"


@configclass
class PickSingleEgadB190Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/B19_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/B19_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-B19_0_v2.pkl"


@configclass
class PickSingleEgadB192Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/B19_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/B19_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-B19_2_v2.pkl"


@configclass
class PickSingleEgadB193Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/B19_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/B19_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-B19_3_v2.pkl"


@configclass
class PickSingleEgadB200Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/B20_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/B20_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-B20_0_v2.pkl"


@configclass
class PickSingleEgadB201Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/B20_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/B20_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-B20_1_v2.pkl"


@configclass
class PickSingleEgadB202Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/B20_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/B20_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-B20_2_v2.pkl"


@configclass
class PickSingleEgadB210Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/B21_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/B21_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-B21_0_v2.pkl"


@configclass
class PickSingleEgadB211Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/B21_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/B21_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-B21_1_v2.pkl"


@configclass
class PickSingleEgadB212Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/B21_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/B21_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-B21_2_v2.pkl"


@configclass
class PickSingleEgadB213Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/B21_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/B21_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-B21_3_v2.pkl"


@configclass
class PickSingleEgadB220Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/B22_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/B22_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-B22_0_v2.pkl"


@configclass
class PickSingleEgadB221Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/B22_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/B22_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-B22_1_v2.pkl"


@configclass
class PickSingleEgadB222Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/B22_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/B22_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-B22_2_v2.pkl"


@configclass
class PickSingleEgadB223Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/B22_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/B22_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-B22_3_v2.pkl"


@configclass
class PickSingleEgadB231Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/B23_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/B23_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-B23_1_v2.pkl"


@configclass
class PickSingleEgadB232Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/B23_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/B23_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-B23_2_v2.pkl"


@configclass
class PickSingleEgadB233Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/B23_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/B23_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-B23_3_v2.pkl"


@configclass
class PickSingleEgadB240Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/B24_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/B24_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-B24_0_v2.pkl"


@configclass
class PickSingleEgadB241Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/B24_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/B24_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-B24_1_v2.pkl"


@configclass
class PickSingleEgadB242Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/B24_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/B24_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-B24_2_v2.pkl"


@configclass
class PickSingleEgadB243Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/B24_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/B24_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-B24_3_v2.pkl"


@configclass
class PickSingleEgadB250Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/B25_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/B25_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-B25_0_v2.pkl"


@configclass
class PickSingleEgadB251Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/B25_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/B25_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-B25_1_v2.pkl"


@configclass
class PickSingleEgadB252Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/B25_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/B25_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-B25_2_v2.pkl"


@configclass
class PickSingleEgadB253Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/B25_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/B25_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-B25_3_v2.pkl"


@configclass
class PickSingleEgadC100Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/C10_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/C10_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-C10_0_v2.pkl"


@configclass
class PickSingleEgadC101Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/C10_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/C10_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-C10_1_v2.pkl"


@configclass
class PickSingleEgadC102Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/C10_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/C10_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-C10_2_v2.pkl"


@configclass
class PickSingleEgadC103Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/C10_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/C10_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-C10_3_v2.pkl"


@configclass
class PickSingleEgadC110Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/C11_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/C11_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-C11_0_v2.pkl"


@configclass
class PickSingleEgadC111Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/C11_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/C11_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-C11_1_v2.pkl"


@configclass
class PickSingleEgadC113Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/C11_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/C11_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-C11_3_v2.pkl"


@configclass
class PickSingleEgadC120Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/C12_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/C12_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-C12_0_v2.pkl"


@configclass
class PickSingleEgadC121Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/C12_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/C12_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-C12_1_v2.pkl"


@configclass
class PickSingleEgadC122Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/C12_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/C12_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-C12_2_v2.pkl"


@configclass
class PickSingleEgadC123Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/C12_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/C12_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-C12_3_v2.pkl"


@configclass
class PickSingleEgadC130Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/C13_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/C13_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-C13_0_v2.pkl"


@configclass
class PickSingleEgadC131Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/C13_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/C13_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-C13_1_v2.pkl"


@configclass
class PickSingleEgadC132Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/C13_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/C13_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-C13_2_v2.pkl"


@configclass
class PickSingleEgadC133Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/C13_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/C13_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-C13_3_v2.pkl"


@configclass
class PickSingleEgadC140Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/C14_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/C14_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-C14_0_v2.pkl"


@configclass
class PickSingleEgadC142Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/C14_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/C14_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-C14_2_v2.pkl"


@configclass
class PickSingleEgadC143Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/C14_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/C14_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-C14_3_v2.pkl"


@configclass
class PickSingleEgadC150Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/C15_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/C15_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-C15_0_v2.pkl"


@configclass
class PickSingleEgadC151Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/C15_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/C15_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-C15_1_v2.pkl"


@configclass
class PickSingleEgadC152Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/C15_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/C15_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-C15_2_v2.pkl"


@configclass
class PickSingleEgadC153Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/C15_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/C15_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-C15_3_v2.pkl"


@configclass
class PickSingleEgadC161Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/C16_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/C16_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-C16_1_v2.pkl"


@configclass
class PickSingleEgadC162Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/C16_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/C16_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-C16_2_v2.pkl"


@configclass
class PickSingleEgadC163Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/C16_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/C16_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-C16_3_v2.pkl"


@configclass
class PickSingleEgadC170Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/C17_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/C17_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-C17_0_v2.pkl"


@configclass
class PickSingleEgadC171Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/C17_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/C17_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-C17_1_v2.pkl"


@configclass
class PickSingleEgadC172Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/C17_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/C17_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-C17_2_v2.pkl"


@configclass
class PickSingleEgadC173Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/C17_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/C17_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-C17_3_v2.pkl"


@configclass
class PickSingleEgadC180Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/C18_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/C18_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-C18_0_v2.pkl"


@configclass
class PickSingleEgadC181Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/C18_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/C18_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-C18_1_v2.pkl"


@configclass
class PickSingleEgadC182Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/C18_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/C18_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-C18_2_v2.pkl"


@configclass
class PickSingleEgadC183Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/C18_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/C18_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-C18_3_v2.pkl"


@configclass
class PickSingleEgadC190Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/C19_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/C19_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-C19_0_v2.pkl"


@configclass
class PickSingleEgadC191Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/C19_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/C19_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-C19_1_v2.pkl"


@configclass
class PickSingleEgadC192Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/C19_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/C19_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-C19_2_v2.pkl"


@configclass
class PickSingleEgadC193Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/C19_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/C19_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-C19_3_v2.pkl"


@configclass
class PickSingleEgadC200Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/C20_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/C20_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-C20_0_v2.pkl"


@configclass
class PickSingleEgadC201Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/C20_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/C20_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-C20_1_v2.pkl"


@configclass
class PickSingleEgadC202Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/C20_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/C20_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-C20_2_v2.pkl"


@configclass
class PickSingleEgadC203Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/C20_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/C20_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-C20_3_v2.pkl"


@configclass
class PickSingleEgadC210Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/C21_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/C21_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-C21_0_v2.pkl"


@configclass
class PickSingleEgadC211Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/C21_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/C21_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-C21_1_v2.pkl"


@configclass
class PickSingleEgadC212Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/C21_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/C21_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-C21_2_v2.pkl"


@configclass
class PickSingleEgadC213Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/C21_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/C21_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-C21_3_v2.pkl"


@configclass
class PickSingleEgadC220Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/C22_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/C22_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-C22_0_v2.pkl"


@configclass
class PickSingleEgadC221Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/C22_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/C22_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-C22_1_v2.pkl"


@configclass
class PickSingleEgadC223Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/C22_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/C22_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-C22_3_v2.pkl"


@configclass
class PickSingleEgadC230Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/C23_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/C23_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-C23_0_v2.pkl"


@configclass
class PickSingleEgadC231Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/C23_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/C23_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-C23_1_v2.pkl"


@configclass
class PickSingleEgadC232Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/C23_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/C23_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-C23_2_v2.pkl"


@configclass
class PickSingleEgadC233Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/C23_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/C23_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-C23_3_v2.pkl"


@configclass
class PickSingleEgadC240Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/C24_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/C24_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-C24_0_v2.pkl"


@configclass
class PickSingleEgadC241Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/C24_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/C24_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-C24_1_v2.pkl"


@configclass
class PickSingleEgadC242Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/C24_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/C24_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-C24_2_v2.pkl"


@configclass
class PickSingleEgadC243Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/C24_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/C24_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-C24_3_v2.pkl"


@configclass
class PickSingleEgadC250Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/C25_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/C25_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-C25_0_v2.pkl"


@configclass
class PickSingleEgadC251Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/C25_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/C25_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-C25_1_v2.pkl"


@configclass
class PickSingleEgadC252Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/C25_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/C25_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-C25_2_v2.pkl"


@configclass
class PickSingleEgadC253Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/C25_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/C25_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-C25_3_v2.pkl"


@configclass
class PickSingleEgadD100Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/D10_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/D10_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-D10_0_v2.pkl"


@configclass
class PickSingleEgadD101Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/D10_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/D10_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-D10_1_v2.pkl"


@configclass
class PickSingleEgadD102Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/D10_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/D10_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-D10_2_v2.pkl"


@configclass
class PickSingleEgadD103Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/D10_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/D10_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-D10_3_v2.pkl"


@configclass
class PickSingleEgadD110Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/D11_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/D11_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-D11_0_v2.pkl"


@configclass
class PickSingleEgadD111Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/D11_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/D11_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-D11_1_v2.pkl"


@configclass
class PickSingleEgadD112Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/D11_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/D11_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-D11_2_v2.pkl"


@configclass
class PickSingleEgadD113Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/D11_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/D11_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-D11_3_v2.pkl"


@configclass
class PickSingleEgadD121Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/D12_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/D12_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-D12_1_v2.pkl"


@configclass
class PickSingleEgadD122Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/D12_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/D12_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-D12_2_v2.pkl"


@configclass
class PickSingleEgadD130Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/D13_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/D13_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-D13_0_v2.pkl"


@configclass
class PickSingleEgadD131Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/D13_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/D13_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-D13_1_v2.pkl"


@configclass
class PickSingleEgadD132Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/D13_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/D13_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-D13_2_v2.pkl"


@configclass
class PickSingleEgadD133Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/D13_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/D13_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-D13_3_v2.pkl"


@configclass
class PickSingleEgadD141Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/D14_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/D14_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-D14_1_v2.pkl"


@configclass
class PickSingleEgadD142Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/D14_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/D14_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-D14_2_v2.pkl"


@configclass
class PickSingleEgadD150Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/D15_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/D15_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-D15_0_v2.pkl"


@configclass
class PickSingleEgadD151Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/D15_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/D15_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-D15_1_v2.pkl"


@configclass
class PickSingleEgadD152Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/D15_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/D15_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-D15_2_v2.pkl"


@configclass
class PickSingleEgadD153Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/D15_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/D15_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-D15_3_v2.pkl"


@configclass
class PickSingleEgadD160Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/D16_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/D16_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-D16_0_v2.pkl"


@configclass
class PickSingleEgadD161Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/D16_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/D16_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-D16_1_v2.pkl"


@configclass
class PickSingleEgadD162Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/D16_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/D16_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-D16_2_v2.pkl"


@configclass
class PickSingleEgadD163Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/D16_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/D16_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-D16_3_v2.pkl"


@configclass
class PickSingleEgadD170Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/D17_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/D17_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-D17_0_v2.pkl"


@configclass
class PickSingleEgadD171Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/D17_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/D17_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-D17_1_v2.pkl"


@configclass
class PickSingleEgadD172Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/D17_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/D17_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-D17_2_v2.pkl"


@configclass
class PickSingleEgadD180Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/D18_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/D18_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-D18_0_v2.pkl"


@configclass
class PickSingleEgadD181Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/D18_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/D18_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-D18_1_v2.pkl"


@configclass
class PickSingleEgadD182Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/D18_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/D18_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-D18_2_v2.pkl"


@configclass
class PickSingleEgadD183Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/D18_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/D18_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-D18_3_v2.pkl"


@configclass
class PickSingleEgadD190Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/D19_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/D19_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-D19_0_v2.pkl"


@configclass
class PickSingleEgadD191Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/D19_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/D19_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-D19_1_v2.pkl"


@configclass
class PickSingleEgadD193Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/D19_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/D19_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-D19_3_v2.pkl"


@configclass
class PickSingleEgadD200Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/D20_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/D20_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-D20_0_v2.pkl"


@configclass
class PickSingleEgadD201Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/D20_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/D20_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-D20_1_v2.pkl"


@configclass
class PickSingleEgadD202Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/D20_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/D20_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-D20_2_v2.pkl"


@configclass
class PickSingleEgadD203Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/D20_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/D20_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-D20_3_v2.pkl"


@configclass
class PickSingleEgadD210Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/D21_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/D21_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-D21_0_v2.pkl"


@configclass
class PickSingleEgadD211Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/D21_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/D21_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-D21_1_v2.pkl"


@configclass
class PickSingleEgadD212Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/D21_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/D21_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-D21_2_v2.pkl"


@configclass
class PickSingleEgadD213Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/D21_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/D21_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-D21_3_v2.pkl"


@configclass
class PickSingleEgadD220Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/D22_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/D22_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-D22_0_v2.pkl"


@configclass
class PickSingleEgadD221Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/D22_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/D22_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-D22_1_v2.pkl"


@configclass
class PickSingleEgadD222Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/D22_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/D22_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-D22_2_v2.pkl"


@configclass
class PickSingleEgadD223Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/D22_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/D22_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-D22_3_v2.pkl"


@configclass
class PickSingleEgadD230Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/D23_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/D23_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-D23_0_v2.pkl"


@configclass
class PickSingleEgadD231Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/D23_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/D23_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-D23_1_v2.pkl"


@configclass
class PickSingleEgadD232Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/D23_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/D23_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-D23_2_v2.pkl"


@configclass
class PickSingleEgadD233Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/D23_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/D23_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-D23_3_v2.pkl"


@configclass
class PickSingleEgadD240Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/D24_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/D24_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-D24_0_v2.pkl"


@configclass
class PickSingleEgadD241Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/D24_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/D24_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-D24_1_v2.pkl"


@configclass
class PickSingleEgadD242Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/D24_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/D24_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-D24_2_v2.pkl"


@configclass
class PickSingleEgadD243Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/D24_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/D24_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-D24_3_v2.pkl"


@configclass
class PickSingleEgadD250Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/D25_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/D25_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-D25_0_v2.pkl"


@configclass
class PickSingleEgadD251Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/D25_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/D25_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-D25_1_v2.pkl"


@configclass
class PickSingleEgadD252Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/D25_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/D25_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-D25_2_v2.pkl"


@configclass
class PickSingleEgadD253Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/D25_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/D25_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-D25_3_v2.pkl"


@configclass
class PickSingleEgadE100Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/E10_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/E10_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-E10_0_v2.pkl"


@configclass
class PickSingleEgadE101Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/E10_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/E10_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-E10_1_v2.pkl"


@configclass
class PickSingleEgadE102Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/E10_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/E10_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-E10_2_v2.pkl"


@configclass
class PickSingleEgadE103Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/E10_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/E10_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-E10_3_v2.pkl"


@configclass
class PickSingleEgadE111Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/E11_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/E11_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-E11_1_v2.pkl"


@configclass
class PickSingleEgadE112Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/E11_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/E11_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-E11_2_v2.pkl"


@configclass
class PickSingleEgadE113Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/E11_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/E11_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-E11_3_v2.pkl"


@configclass
class PickSingleEgadE120Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/E12_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/E12_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-E12_0_v2.pkl"


@configclass
class PickSingleEgadE121Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/E12_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/E12_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-E12_1_v2.pkl"


@configclass
class PickSingleEgadE122Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/E12_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/E12_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-E12_2_v2.pkl"


@configclass
class PickSingleEgadE123Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/E12_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/E12_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-E12_3_v2.pkl"


@configclass
class PickSingleEgadE131Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/E13_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/E13_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-E13_1_v2.pkl"


@configclass
class PickSingleEgadE132Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/E13_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/E13_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-E13_2_v2.pkl"


@configclass
class PickSingleEgadE133Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/E13_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/E13_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-E13_3_v2.pkl"


@configclass
class PickSingleEgadE140Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/E14_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/E14_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-E14_0_v2.pkl"


@configclass
class PickSingleEgadE141Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/E14_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/E14_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-E14_1_v2.pkl"


@configclass
class PickSingleEgadE142Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/E14_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/E14_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-E14_2_v2.pkl"


@configclass
class PickSingleEgadE143Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/E14_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/E14_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-E14_3_v2.pkl"


@configclass
class PickSingleEgadE150Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/E15_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/E15_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-E15_0_v2.pkl"


@configclass
class PickSingleEgadE151Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/E15_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/E15_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-E15_1_v2.pkl"


@configclass
class PickSingleEgadE152Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/E15_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/E15_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-E15_2_v2.pkl"


@configclass
class PickSingleEgadE153Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/E15_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/E15_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-E15_3_v2.pkl"


@configclass
class PickSingleEgadE160Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/E16_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/E16_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-E16_0_v2.pkl"


@configclass
class PickSingleEgadE161Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/E16_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/E16_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-E16_1_v2.pkl"


@configclass
class PickSingleEgadE162Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/E16_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/E16_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-E16_2_v2.pkl"


@configclass
class PickSingleEgadE163Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/E16_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/E16_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-E16_3_v2.pkl"


@configclass
class PickSingleEgadE170Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/E17_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/E17_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-E17_0_v2.pkl"


@configclass
class PickSingleEgadE171Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/E17_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/E17_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-E17_1_v2.pkl"


@configclass
class PickSingleEgadE172Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/E17_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/E17_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-E17_2_v2.pkl"


@configclass
class PickSingleEgadE181Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/E18_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/E18_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-E18_1_v2.pkl"


@configclass
class PickSingleEgadE182Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/E18_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/E18_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-E18_2_v2.pkl"


@configclass
class PickSingleEgadE190Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/E19_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/E19_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-E19_0_v2.pkl"


@configclass
class PickSingleEgadE191Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/E19_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/E19_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-E19_1_v2.pkl"


@configclass
class PickSingleEgadE192Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/E19_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/E19_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-E19_2_v2.pkl"


@configclass
class PickSingleEgadE193Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/E19_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/E19_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-E19_3_v2.pkl"


@configclass
class PickSingleEgadE200Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/E20_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/E20_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-E20_0_v2.pkl"


@configclass
class PickSingleEgadE201Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/E20_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/E20_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-E20_1_v2.pkl"


@configclass
class PickSingleEgadE202Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/E20_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/E20_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-E20_2_v2.pkl"


@configclass
class PickSingleEgadE210Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/E21_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/E21_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-E21_0_v2.pkl"


@configclass
class PickSingleEgadE211Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/E21_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/E21_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-E21_1_v2.pkl"


@configclass
class PickSingleEgadE212Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/E21_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/E21_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-E21_2_v2.pkl"


@configclass
class PickSingleEgadE213Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/E21_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/E21_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-E21_3_v2.pkl"


@configclass
class PickSingleEgadE220Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/E22_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/E22_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-E22_0_v2.pkl"


@configclass
class PickSingleEgadE221Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/E22_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/E22_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-E22_1_v2.pkl"


@configclass
class PickSingleEgadE222Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/E22_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/E22_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-E22_2_v2.pkl"


@configclass
class PickSingleEgadE223Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/E22_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/E22_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-E22_3_v2.pkl"


@configclass
class PickSingleEgadE230Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/E23_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/E23_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-E23_0_v2.pkl"


@configclass
class PickSingleEgadE231Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/E23_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/E23_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-E23_1_v2.pkl"


@configclass
class PickSingleEgadE232Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/E23_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/E23_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-E23_2_v2.pkl"


@configclass
class PickSingleEgadE233Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/E23_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/E23_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-E23_3_v2.pkl"


@configclass
class PickSingleEgadE240Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/E24_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/E24_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-E24_0_v2.pkl"


@configclass
class PickSingleEgadE241Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/E24_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/E24_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-E24_1_v2.pkl"


@configclass
class PickSingleEgadE242Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/E24_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/E24_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-E24_2_v2.pkl"


@configclass
class PickSingleEgadE243Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/E24_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/E24_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-E24_3_v2.pkl"


@configclass
class PickSingleEgadE250Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/E25_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/E25_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-E25_0_v2.pkl"


@configclass
class PickSingleEgadE251Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/E25_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/E25_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-E25_1_v2.pkl"


@configclass
class PickSingleEgadE252Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/E25_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/E25_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-E25_2_v2.pkl"


@configclass
class PickSingleEgadE253Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/E25_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/E25_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-E25_3_v2.pkl"


@configclass
class PickSingleEgadF100Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/F10_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/F10_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-F10_0_v2.pkl"


@configclass
class PickSingleEgadF101Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/F10_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/F10_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-F10_1_v2.pkl"


@configclass
class PickSingleEgadF103Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/F10_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/F10_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-F10_3_v2.pkl"


@configclass
class PickSingleEgadF110Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/F11_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/F11_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-F11_0_v2.pkl"


@configclass
class PickSingleEgadF111Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/F11_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/F11_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-F11_1_v2.pkl"


@configclass
class PickSingleEgadF112Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/F11_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/F11_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-F11_2_v2.pkl"


@configclass
class PickSingleEgadF113Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/F11_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/F11_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-F11_3_v2.pkl"


@configclass
class PickSingleEgadF121Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/F12_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/F12_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-F12_1_v2.pkl"


@configclass
class PickSingleEgadF122Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/F12_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/F12_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-F12_2_v2.pkl"


@configclass
class PickSingleEgadF130Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/F13_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/F13_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-F13_0_v2.pkl"


@configclass
class PickSingleEgadF131Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/F13_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/F13_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-F13_1_v2.pkl"


@configclass
class PickSingleEgadF132Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/F13_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/F13_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-F13_2_v2.pkl"


@configclass
class PickSingleEgadF133Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/F13_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/F13_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-F13_3_v2.pkl"


@configclass
class PickSingleEgadF140Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/F14_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/F14_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-F14_0_v2.pkl"


@configclass
class PickSingleEgadF142Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/F14_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/F14_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-F14_2_v2.pkl"


@configclass
class PickSingleEgadF143Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/F14_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/F14_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-F14_3_v2.pkl"


@configclass
class PickSingleEgadF150Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/F15_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/F15_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-F15_0_v2.pkl"


@configclass
class PickSingleEgadF151Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/F15_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/F15_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-F15_1_v2.pkl"


@configclass
class PickSingleEgadF152Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/F15_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/F15_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-F15_2_v2.pkl"


@configclass
class PickSingleEgadF153Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/F15_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/F15_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-F15_3_v2.pkl"


@configclass
class PickSingleEgadF160Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/F16_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/F16_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-F16_0_v2.pkl"


@configclass
class PickSingleEgadF161Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/F16_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/F16_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-F16_1_v2.pkl"


@configclass
class PickSingleEgadF162Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/F16_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/F16_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-F16_2_v2.pkl"


@configclass
class PickSingleEgadF163Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/F16_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/F16_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-F16_3_v2.pkl"


@configclass
class PickSingleEgadF170Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/F17_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/F17_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-F17_0_v2.pkl"


@configclass
class PickSingleEgadF171Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/F17_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/F17_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-F17_1_v2.pkl"


@configclass
class PickSingleEgadF172Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/F17_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/F17_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-F17_2_v2.pkl"


@configclass
class PickSingleEgadF173Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/F17_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/F17_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-F17_3_v2.pkl"


@configclass
class PickSingleEgadF180Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/F18_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/F18_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-F18_0_v2.pkl"


@configclass
class PickSingleEgadF181Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/F18_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/F18_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-F18_1_v2.pkl"


@configclass
class PickSingleEgadF182Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/F18_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/F18_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-F18_2_v2.pkl"


@configclass
class PickSingleEgadF183Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/F18_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/F18_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-F18_3_v2.pkl"


@configclass
class PickSingleEgadF190Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/F19_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/F19_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-F19_0_v2.pkl"


@configclass
class PickSingleEgadF191Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/F19_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/F19_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-F19_1_v2.pkl"


@configclass
class PickSingleEgadF192Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/F19_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/F19_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-F19_2_v2.pkl"


@configclass
class PickSingleEgadF193Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/F19_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/F19_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-F19_3_v2.pkl"


@configclass
class PickSingleEgadF200Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/F20_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/F20_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-F20_0_v2.pkl"


@configclass
class PickSingleEgadF202Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/F20_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/F20_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-F20_2_v2.pkl"


@configclass
class PickSingleEgadF203Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/F20_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/F20_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-F20_3_v2.pkl"


@configclass
class PickSingleEgadF210Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/F21_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/F21_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-F21_0_v2.pkl"


@configclass
class PickSingleEgadF211Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/F21_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/F21_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-F21_1_v2.pkl"


@configclass
class PickSingleEgadF212Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/F21_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/F21_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-F21_2_v2.pkl"


@configclass
class PickSingleEgadF213Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/F21_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/F21_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-F21_3_v2.pkl"


@configclass
class PickSingleEgadF220Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/F22_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/F22_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-F22_0_v2.pkl"


@configclass
class PickSingleEgadF221Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/F22_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/F22_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-F22_1_v2.pkl"


@configclass
class PickSingleEgadF222Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/F22_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/F22_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-F22_2_v2.pkl"


@configclass
class PickSingleEgadF223Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/F22_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/F22_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-F22_3_v2.pkl"


@configclass
class PickSingleEgadF230Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/F23_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/F23_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-F23_0_v2.pkl"


@configclass
class PickSingleEgadF231Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/F23_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/F23_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-F23_1_v2.pkl"


@configclass
class PickSingleEgadF232Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/F23_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/F23_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-F23_2_v2.pkl"


@configclass
class PickSingleEgadF233Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/F23_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/F23_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-F23_3_v2.pkl"


@configclass
class PickSingleEgadF240Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/F24_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/F24_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-F24_0_v2.pkl"


@configclass
class PickSingleEgadF241Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/F24_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/F24_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-F24_1_v2.pkl"


@configclass
class PickSingleEgadF242Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/F24_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/F24_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-F24_2_v2.pkl"


@configclass
class PickSingleEgadF243Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/F24_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/F24_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-F24_3_v2.pkl"


@configclass
class PickSingleEgadF250Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/F25_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/F25_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-F25_0_v2.pkl"


@configclass
class PickSingleEgadF251Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/F25_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/F25_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-F25_1_v2.pkl"


@configclass
class PickSingleEgadF252Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/F25_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/F25_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-F25_2_v2.pkl"


@configclass
class PickSingleEgadF253Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/F25_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/F25_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-F25_3_v2.pkl"


@configclass
class PickSingleEgadG100Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/G10_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/G10_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-G10_0_v2.pkl"


@configclass
class PickSingleEgadG101Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/G10_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/G10_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-G10_1_v2.pkl"


@configclass
class PickSingleEgadG102Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/G10_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/G10_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-G10_2_v2.pkl"


@configclass
class PickSingleEgadG103Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/G10_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/G10_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-G10_3_v2.pkl"


@configclass
class PickSingleEgadG110Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/G11_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/G11_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-G11_0_v2.pkl"


@configclass
class PickSingleEgadG111Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/G11_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/G11_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-G11_1_v2.pkl"


@configclass
class PickSingleEgadG112Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/G11_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/G11_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-G11_2_v2.pkl"


@configclass
class PickSingleEgadG113Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/G11_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/G11_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-G11_3_v2.pkl"


@configclass
class PickSingleEgadG120Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/G12_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/G12_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-G12_0_v2.pkl"


@configclass
class PickSingleEgadG122Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/G12_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/G12_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-G12_2_v2.pkl"


@configclass
class PickSingleEgadG123Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/G12_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/G12_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-G12_3_v2.pkl"


@configclass
class PickSingleEgadG130Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/G13_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/G13_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-G13_0_v2.pkl"


@configclass
class PickSingleEgadG131Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/G13_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/G13_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-G13_1_v2.pkl"


@configclass
class PickSingleEgadG132Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/G13_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/G13_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-G13_2_v2.pkl"


@configclass
class PickSingleEgadG133Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/G13_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/G13_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-G13_3_v2.pkl"


@configclass
class PickSingleEgadG140Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/G14_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/G14_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-G14_0_v2.pkl"


@configclass
class PickSingleEgadG141Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/G14_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/G14_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-G14_1_v2.pkl"


@configclass
class PickSingleEgadG142Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/G14_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/G14_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-G14_2_v2.pkl"


@configclass
class PickSingleEgadG143Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/G14_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/G14_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-G14_3_v2.pkl"


@configclass
class PickSingleEgadG150Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/G15_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/G15_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-G15_0_v2.pkl"


@configclass
class PickSingleEgadG151Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/G15_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/G15_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-G15_1_v2.pkl"


@configclass
class PickSingleEgadG152Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/G15_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/G15_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-G15_2_v2.pkl"


@configclass
class PickSingleEgadG160Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/G16_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/G16_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-G16_0_v2.pkl"


@configclass
class PickSingleEgadG161Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/G16_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/G16_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-G16_1_v2.pkl"


@configclass
class PickSingleEgadG162Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/G16_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/G16_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-G16_2_v2.pkl"


@configclass
class PickSingleEgadG163Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/G16_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/G16_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-G16_3_v2.pkl"


@configclass
class PickSingleEgadG170Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/G17_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/G17_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-G17_0_v2.pkl"


@configclass
class PickSingleEgadG171Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/G17_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/G17_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-G17_1_v2.pkl"


@configclass
class PickSingleEgadG172Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/G17_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/G17_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-G17_2_v2.pkl"


@configclass
class PickSingleEgadG173Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/G17_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/G17_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-G17_3_v2.pkl"


@configclass
class PickSingleEgadG181Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/G18_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/G18_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-G18_1_v2.pkl"


@configclass
class PickSingleEgadG182Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/G18_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/G18_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-G18_2_v2.pkl"


@configclass
class PickSingleEgadG183Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/G18_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/G18_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-G18_3_v2.pkl"


@configclass
class PickSingleEgadG191Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/G19_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/G19_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-G19_1_v2.pkl"


@configclass
class PickSingleEgadG192Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/G19_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/G19_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-G19_2_v2.pkl"


@configclass
class PickSingleEgadG193Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/G19_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/G19_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-G19_3_v2.pkl"


@configclass
class PickSingleEgadG200Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/G20_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/G20_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-G20_0_v2.pkl"


@configclass
class PickSingleEgadG201Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/G20_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/G20_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-G20_1_v2.pkl"


@configclass
class PickSingleEgadG202Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/G20_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/G20_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-G20_2_v2.pkl"


@configclass
class PickSingleEgadG203Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/G20_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/G20_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-G20_3_v2.pkl"


@configclass
class PickSingleEgadG210Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/G21_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/G21_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-G21_0_v2.pkl"


@configclass
class PickSingleEgadG211Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/G21_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/G21_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-G21_1_v2.pkl"


@configclass
class PickSingleEgadG213Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/G21_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/G21_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-G21_3_v2.pkl"


@configclass
class PickSingleEgadG220Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/G22_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/G22_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-G22_0_v2.pkl"


@configclass
class PickSingleEgadG221Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/G22_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/G22_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-G22_1_v2.pkl"


@configclass
class PickSingleEgadG222Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/G22_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/G22_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-G22_2_v2.pkl"


@configclass
class PickSingleEgadG223Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/G22_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/G22_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-G22_3_v2.pkl"


@configclass
class PickSingleEgadG230Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/G23_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/G23_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-G23_0_v2.pkl"


@configclass
class PickSingleEgadG231Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/G23_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/G23_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-G23_1_v2.pkl"


@configclass
class PickSingleEgadG233Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/G23_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/G23_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-G23_3_v2.pkl"


@configclass
class PickSingleEgadG240Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/G24_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/G24_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-G24_0_v2.pkl"


@configclass
class PickSingleEgadG241Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/G24_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/G24_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-G24_1_v2.pkl"


@configclass
class PickSingleEgadG242Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/G24_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/G24_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-G24_2_v2.pkl"


@configclass
class PickSingleEgadG243Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/G24_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/G24_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-G24_3_v2.pkl"


@configclass
class PickSingleEgadG250Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/G25_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/G25_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-G25_0_v2.pkl"


@configclass
class PickSingleEgadG251Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/G25_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/G25_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-G25_1_v2.pkl"


@configclass
class PickSingleEgadG252Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/G25_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/G25_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-G25_2_v2.pkl"


@configclass
class PickSingleEgadG253Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/G25_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/G25_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-G25_3_v2.pkl"


@configclass
class PickSingleEgadH100Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/H10_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/H10_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-H10_0_v2.pkl"


@configclass
class PickSingleEgadH101Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/H10_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/H10_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-H10_1_v2.pkl"


@configclass
class PickSingleEgadH102Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/H10_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/H10_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-H10_2_v2.pkl"


@configclass
class PickSingleEgadH103Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/H10_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/H10_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-H10_3_v2.pkl"


@configclass
class PickSingleEgadH110Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/H11_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/H11_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-H11_0_v2.pkl"


@configclass
class PickSingleEgadH111Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/H11_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/H11_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-H11_1_v2.pkl"


@configclass
class PickSingleEgadH112Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/H11_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/H11_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-H11_2_v2.pkl"


@configclass
class PickSingleEgadH113Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/H11_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/H11_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-H11_3_v2.pkl"


@configclass
class PickSingleEgadH120Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/H12_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/H12_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-H12_0_v2.pkl"


@configclass
class PickSingleEgadH121Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/H12_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/H12_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-H12_1_v2.pkl"


@configclass
class PickSingleEgadH122Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/H12_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/H12_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-H12_2_v2.pkl"


@configclass
class PickSingleEgadH123Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/H12_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/H12_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-H12_3_v2.pkl"


@configclass
class PickSingleEgadH130Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/H13_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/H13_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-H13_0_v2.pkl"


@configclass
class PickSingleEgadH131Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/H13_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/H13_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-H13_1_v2.pkl"


@configclass
class PickSingleEgadH132Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/H13_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/H13_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-H13_2_v2.pkl"


@configclass
class PickSingleEgadH140Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/H14_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/H14_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-H14_0_v2.pkl"


@configclass
class PickSingleEgadH141Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/H14_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/H14_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-H14_1_v2.pkl"


@configclass
class PickSingleEgadH142Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/H14_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/H14_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-H14_2_v2.pkl"


@configclass
class PickSingleEgadH143Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/H14_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/H14_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-H14_3_v2.pkl"


@configclass
class PickSingleEgadH150Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/H15_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/H15_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-H15_0_v2.pkl"


@configclass
class PickSingleEgadH151Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/H15_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/H15_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-H15_1_v2.pkl"


@configclass
class PickSingleEgadH152Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/H15_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/H15_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-H15_2_v2.pkl"


@configclass
class PickSingleEgadH153Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/H15_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/H15_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-H15_3_v2.pkl"


@configclass
class PickSingleEgadH160Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/H16_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/H16_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-H16_0_v2.pkl"


@configclass
class PickSingleEgadH161Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/H16_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/H16_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-H16_1_v2.pkl"


@configclass
class PickSingleEgadH162Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/H16_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/H16_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-H16_2_v2.pkl"


@configclass
class PickSingleEgadH163Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/H16_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/H16_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-H16_3_v2.pkl"


@configclass
class PickSingleEgadH170Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/H17_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/H17_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-H17_0_v2.pkl"


@configclass
class PickSingleEgadH171Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/H17_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/H17_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-H17_1_v2.pkl"


@configclass
class PickSingleEgadH172Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/H17_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/H17_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-H17_2_v2.pkl"


@configclass
class PickSingleEgadH173Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/H17_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/H17_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-H17_3_v2.pkl"


@configclass
class PickSingleEgadH181Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/H18_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/H18_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-H18_1_v2.pkl"


@configclass
class PickSingleEgadH182Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/H18_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/H18_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-H18_2_v2.pkl"


@configclass
class PickSingleEgadH183Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/H18_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/H18_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-H18_3_v2.pkl"


@configclass
class PickSingleEgadH190Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/H19_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/H19_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-H19_0_v2.pkl"


@configclass
class PickSingleEgadH191Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/H19_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/H19_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-H19_1_v2.pkl"


@configclass
class PickSingleEgadH192Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/H19_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/H19_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-H19_2_v2.pkl"


@configclass
class PickSingleEgadH193Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/H19_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/H19_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-H19_3_v2.pkl"


@configclass
class PickSingleEgadH200Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/H20_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/H20_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-H20_0_v2.pkl"


@configclass
class PickSingleEgadH201Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/H20_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/H20_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-H20_1_v2.pkl"


@configclass
class PickSingleEgadH202Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/H20_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/H20_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-H20_2_v2.pkl"


@configclass
class PickSingleEgadH203Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/H20_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/H20_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-H20_3_v2.pkl"


@configclass
class PickSingleEgadH210Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/H21_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/H21_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-H21_0_v2.pkl"


@configclass
class PickSingleEgadH211Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/H21_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/H21_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-H21_1_v2.pkl"


@configclass
class PickSingleEgadH212Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/H21_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/H21_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-H21_2_v2.pkl"


@configclass
class PickSingleEgadH220Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/H22_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/H22_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-H22_0_v2.pkl"


@configclass
class PickSingleEgadH221Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/H22_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/H22_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-H22_1_v2.pkl"


@configclass
class PickSingleEgadH222Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/H22_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/H22_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-H22_2_v2.pkl"


@configclass
class PickSingleEgadH223Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/H22_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/H22_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-H22_3_v2.pkl"


@configclass
class PickSingleEgadH230Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/H23_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/H23_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-H23_0_v2.pkl"


@configclass
class PickSingleEgadH231Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/H23_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/H23_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-H23_1_v2.pkl"


@configclass
class PickSingleEgadH240Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/H24_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/H24_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-H24_0_v2.pkl"


@configclass
class PickSingleEgadH241Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/H24_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/H24_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-H24_1_v2.pkl"


@configclass
class PickSingleEgadH242Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/H24_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/H24_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-H24_2_v2.pkl"


@configclass
class PickSingleEgadH243Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/H24_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/H24_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-H24_3_v2.pkl"


@configclass
class PickSingleEgadH250Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/H25_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/H25_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-H25_0_v2.pkl"


@configclass
class PickSingleEgadH251Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/H25_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/H25_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-H25_1_v2.pkl"


@configclass
class PickSingleEgadH252Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/H25_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/H25_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-H25_2_v2.pkl"


@configclass
class PickSingleEgadH253Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/H25_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/H25_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-H25_3_v2.pkl"


@configclass
class PickSingleEgadI070Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/I07_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/I07_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I07_0_v2.pkl"


@configclass
class PickSingleEgadI071Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/I07_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/I07_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I07_1_v2.pkl"


@configclass
class PickSingleEgadI072Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/I07_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/I07_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I07_2_v2.pkl"


@configclass
class PickSingleEgadI073Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/I07_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/I07_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I07_3_v2.pkl"


@configclass
class PickSingleEgadI080Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/I08_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/I08_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I08_0_v2.pkl"


@configclass
class PickSingleEgadI081Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/I08_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/I08_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I08_1_v2.pkl"


@configclass
class PickSingleEgadI083Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/I08_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/I08_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I08_3_v2.pkl"


@configclass
class PickSingleEgadI090Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/I09_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/I09_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I09_0_v2.pkl"


@configclass
class PickSingleEgadI091Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/I09_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/I09_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I09_1_v2.pkl"


@configclass
class PickSingleEgadI092Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/I09_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/I09_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I09_2_v2.pkl"


@configclass
class PickSingleEgadI102Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/I10_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/I10_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I10_2_v2.pkl"


@configclass
class PickSingleEgadI103Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/I10_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/I10_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I10_3_v2.pkl"


@configclass
class PickSingleEgadI110Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/I11_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/I11_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I11_0_v2.pkl"


@configclass
class PickSingleEgadI111Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/I11_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/I11_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I11_1_v2.pkl"


@configclass
class PickSingleEgadI112Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/I11_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/I11_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I11_2_v2.pkl"


@configclass
class PickSingleEgadI113Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/I11_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/I11_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I11_3_v2.pkl"


@configclass
class PickSingleEgadI120Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/I12_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/I12_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I12_0_v2.pkl"


@configclass
class PickSingleEgadI121Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/I12_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/I12_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I12_1_v2.pkl"


@configclass
class PickSingleEgadI122Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/I12_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/I12_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I12_2_v2.pkl"


@configclass
class PickSingleEgadI123Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/I12_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/I12_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I12_3_v2.pkl"


@configclass
class PickSingleEgadI130Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/I13_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/I13_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I13_0_v2.pkl"


@configclass
class PickSingleEgadI131Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/I13_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/I13_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I13_1_v2.pkl"


@configclass
class PickSingleEgadI132Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/I13_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/I13_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I13_2_v2.pkl"


@configclass
class PickSingleEgadI133Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/I13_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/I13_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I13_3_v2.pkl"


@configclass
class PickSingleEgadI140Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/I14_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/I14_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I14_0_v2.pkl"


@configclass
class PickSingleEgadI141Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/I14_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/I14_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I14_1_v2.pkl"


@configclass
class PickSingleEgadI142Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/I14_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/I14_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I14_2_v2.pkl"


@configclass
class PickSingleEgadI143Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/I14_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/I14_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I14_3_v2.pkl"


@configclass
class PickSingleEgadI150Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/I15_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/I15_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I15_0_v2.pkl"


@configclass
class PickSingleEgadI151Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/I15_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/I15_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I15_1_v2.pkl"


@configclass
class PickSingleEgadI152Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/I15_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/I15_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I15_2_v2.pkl"


@configclass
class PickSingleEgadI153Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/I15_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/I15_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I15_3_v2.pkl"


@configclass
class PickSingleEgadI160Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/I16_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/I16_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I16_0_v2.pkl"


@configclass
class PickSingleEgadI161Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/I16_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/I16_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I16_1_v2.pkl"


@configclass
class PickSingleEgadI162Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/I16_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/I16_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I16_2_v2.pkl"


@configclass
class PickSingleEgadI163Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/I16_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/I16_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I16_3_v2.pkl"


@configclass
class PickSingleEgadI170Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/I17_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/I17_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I17_0_v2.pkl"


@configclass
class PickSingleEgadI171Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/I17_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/I17_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I17_1_v2.pkl"


@configclass
class PickSingleEgadI172Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/I17_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/I17_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I17_2_v2.pkl"


@configclass
class PickSingleEgadI173Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/I17_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/I17_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I17_3_v2.pkl"


@configclass
class PickSingleEgadI180Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/I18_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/I18_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I18_0_v2.pkl"


@configclass
class PickSingleEgadI181Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/I18_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/I18_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I18_1_v2.pkl"


@configclass
class PickSingleEgadI182Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/I18_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/I18_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I18_2_v2.pkl"


@configclass
class PickSingleEgadI183Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/I18_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/I18_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I18_3_v2.pkl"


@configclass
class PickSingleEgadI190Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/I19_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/I19_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I19_0_v2.pkl"


@configclass
class PickSingleEgadI191Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/I19_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/I19_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I19_1_v2.pkl"


@configclass
class PickSingleEgadI192Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/I19_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/I19_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I19_2_v2.pkl"


@configclass
class PickSingleEgadI200Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/I20_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/I20_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I20_0_v2.pkl"


@configclass
class PickSingleEgadI201Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/I20_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/I20_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I20_1_v2.pkl"


@configclass
class PickSingleEgadI203Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/I20_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/I20_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I20_3_v2.pkl"


@configclass
class PickSingleEgadI210Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/I21_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/I21_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I21_0_v2.pkl"


@configclass
class PickSingleEgadI211Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/I21_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/I21_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I21_1_v2.pkl"


@configclass
class PickSingleEgadI213Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/I21_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/I21_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I21_3_v2.pkl"


@configclass
class PickSingleEgadI220Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/I22_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/I22_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I22_0_v2.pkl"


@configclass
class PickSingleEgadI221Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/I22_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/I22_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I22_1_v2.pkl"


@configclass
class PickSingleEgadI223Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/I22_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/I22_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I22_3_v2.pkl"


@configclass
class PickSingleEgadI230Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/I23_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/I23_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I23_0_v2.pkl"


@configclass
class PickSingleEgadI232Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/I23_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/I23_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I23_2_v2.pkl"


@configclass
class PickSingleEgadI233Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/I23_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/I23_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I23_3_v2.pkl"


@configclass
class PickSingleEgadI240Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/I24_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/I24_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I24_0_v2.pkl"


@configclass
class PickSingleEgadI241Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/I24_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/I24_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I24_1_v2.pkl"


@configclass
class PickSingleEgadI242Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/I24_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/I24_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I24_2_v2.pkl"


@configclass
class PickSingleEgadI243Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/I24_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/I24_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I24_3_v2.pkl"


@configclass
class PickSingleEgadI250Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/I25_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/I25_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I25_0_v2.pkl"


@configclass
class PickSingleEgadI251Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/I25_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/I25_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I25_1_v2.pkl"


@configclass
class PickSingleEgadI252Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/I25_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/I25_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I25_2_v2.pkl"


@configclass
class PickSingleEgadI253Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/I25_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/I25_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-I25_3_v2.pkl"


@configclass
class PickSingleEgadJ070Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/J07_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/J07_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J07_0_v2.pkl"


@configclass
class PickSingleEgadJ071Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/J07_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/J07_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J07_1_v2.pkl"


@configclass
class PickSingleEgadJ072Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/J07_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/J07_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J07_2_v2.pkl"


@configclass
class PickSingleEgadJ073Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/J07_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/J07_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J07_3_v2.pkl"


@configclass
class PickSingleEgadJ080Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/J08_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/J08_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J08_0_v2.pkl"


@configclass
class PickSingleEgadJ082Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/J08_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/J08_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J08_2_v2.pkl"


@configclass
class PickSingleEgadJ083Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/J08_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/J08_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J08_3_v2.pkl"


@configclass
class PickSingleEgadJ090Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/J09_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/J09_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J09_0_v2.pkl"


@configclass
class PickSingleEgadJ091Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/J09_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/J09_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J09_1_v2.pkl"


@configclass
class PickSingleEgadJ092Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/J09_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/J09_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J09_2_v2.pkl"


@configclass
class PickSingleEgadJ100Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/J10_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/J10_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J10_0_v2.pkl"


@configclass
class PickSingleEgadJ101Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/J10_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/J10_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J10_1_v2.pkl"


@configclass
class PickSingleEgadJ102Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/J10_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/J10_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J10_2_v2.pkl"


@configclass
class PickSingleEgadJ103Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/J10_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/J10_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J10_3_v2.pkl"


@configclass
class PickSingleEgadJ110Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/J11_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/J11_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J11_0_v2.pkl"


@configclass
class PickSingleEgadJ111Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/J11_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/J11_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J11_1_v2.pkl"


@configclass
class PickSingleEgadJ112Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/J11_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/J11_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J11_2_v2.pkl"


@configclass
class PickSingleEgadJ113Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/J11_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/J11_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J11_3_v2.pkl"


@configclass
class PickSingleEgadJ120Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/J12_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/J12_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J12_0_v2.pkl"


@configclass
class PickSingleEgadJ121Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/J12_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/J12_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J12_1_v2.pkl"


@configclass
class PickSingleEgadJ122Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/J12_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/J12_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J12_2_v2.pkl"


@configclass
class PickSingleEgadJ123Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/J12_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/J12_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J12_3_v2.pkl"


@configclass
class PickSingleEgadJ130Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/J13_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/J13_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J13_0_v2.pkl"


@configclass
class PickSingleEgadJ131Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/J13_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/J13_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J13_1_v2.pkl"


@configclass
class PickSingleEgadJ132Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/J13_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/J13_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J13_2_v2.pkl"


@configclass
class PickSingleEgadJ133Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/J13_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/J13_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J13_3_v2.pkl"


@configclass
class PickSingleEgadJ140Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/J14_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/J14_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J14_0_v2.pkl"


@configclass
class PickSingleEgadJ141Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/J14_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/J14_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J14_1_v2.pkl"


@configclass
class PickSingleEgadJ142Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/J14_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/J14_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J14_2_v2.pkl"


@configclass
class PickSingleEgadJ143Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/J14_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/J14_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J14_3_v2.pkl"


@configclass
class PickSingleEgadJ150Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/J15_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/J15_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J15_0_v2.pkl"


@configclass
class PickSingleEgadJ151Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/J15_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/J15_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J15_1_v2.pkl"


@configclass
class PickSingleEgadJ152Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/J15_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/J15_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J15_2_v2.pkl"


@configclass
class PickSingleEgadJ153Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/J15_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/J15_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J15_3_v2.pkl"


@configclass
class PickSingleEgadJ160Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/J16_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/J16_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J16_0_v2.pkl"


@configclass
class PickSingleEgadJ162Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/J16_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/J16_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J16_2_v2.pkl"


@configclass
class PickSingleEgadJ163Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/J16_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/J16_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J16_3_v2.pkl"


@configclass
class PickSingleEgadJ170Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/J17_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/J17_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J17_0_v2.pkl"


@configclass
class PickSingleEgadJ171Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/J17_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/J17_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J17_1_v2.pkl"


@configclass
class PickSingleEgadJ172Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/J17_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/J17_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J17_2_v2.pkl"


@configclass
class PickSingleEgadJ173Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/J17_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/J17_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J17_3_v2.pkl"


@configclass
class PickSingleEgadJ180Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/J18_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/J18_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J18_0_v2.pkl"


@configclass
class PickSingleEgadJ181Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/J18_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/J18_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J18_1_v2.pkl"


@configclass
class PickSingleEgadJ182Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/J18_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/J18_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J18_2_v2.pkl"


@configclass
class PickSingleEgadJ183Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/J18_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/J18_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J18_3_v2.pkl"


@configclass
class PickSingleEgadJ190Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/J19_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/J19_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J19_0_v2.pkl"


@configclass
class PickSingleEgadJ191Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/J19_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/J19_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J19_1_v2.pkl"


@configclass
class PickSingleEgadJ192Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/J19_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/J19_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J19_2_v2.pkl"


@configclass
class PickSingleEgadJ193Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/J19_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/J19_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J19_3_v2.pkl"


@configclass
class PickSingleEgadJ200Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/J20_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/J20_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J20_0_v2.pkl"


@configclass
class PickSingleEgadJ201Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/J20_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/J20_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J20_1_v2.pkl"


@configclass
class PickSingleEgadJ202Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/J20_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/J20_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J20_2_v2.pkl"


@configclass
class PickSingleEgadJ203Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/J20_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/J20_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J20_3_v2.pkl"


@configclass
class PickSingleEgadJ210Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/J21_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/J21_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J21_0_v2.pkl"


@configclass
class PickSingleEgadJ211Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/J21_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/J21_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J21_1_v2.pkl"


@configclass
class PickSingleEgadJ212Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/J21_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/J21_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J21_2_v2.pkl"


@configclass
class PickSingleEgadJ213Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/J21_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/J21_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J21_3_v2.pkl"


@configclass
class PickSingleEgadJ220Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/J22_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/J22_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J22_0_v2.pkl"


@configclass
class PickSingleEgadJ221Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/J22_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/J22_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J22_1_v2.pkl"


@configclass
class PickSingleEgadJ222Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/J22_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/J22_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J22_2_v2.pkl"


@configclass
class PickSingleEgadJ223Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/J22_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/J22_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J22_3_v2.pkl"


@configclass
class PickSingleEgadJ230Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/J23_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/J23_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J23_0_v2.pkl"


@configclass
class PickSingleEgadJ231Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/J23_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/J23_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J23_1_v2.pkl"


@configclass
class PickSingleEgadJ232Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/J23_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/J23_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J23_2_v2.pkl"


@configclass
class PickSingleEgadJ233Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/J23_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/J23_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J23_3_v2.pkl"


@configclass
class PickSingleEgadJ240Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/J24_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/J24_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J24_0_v2.pkl"


@configclass
class PickSingleEgadJ241Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/J24_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/J24_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J24_1_v2.pkl"


@configclass
class PickSingleEgadJ242Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/J24_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/J24_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J24_2_v2.pkl"


@configclass
class PickSingleEgadJ243Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/J24_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/J24_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J24_3_v2.pkl"


@configclass
class PickSingleEgadJ250Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/J25_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/J25_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J25_0_v2.pkl"


@configclass
class PickSingleEgadJ251Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/J25_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/J25_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J25_1_v2.pkl"


@configclass
class PickSingleEgadJ252Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/J25_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/J25_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J25_2_v2.pkl"


@configclass
class PickSingleEgadJ253Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/J25_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/J25_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-J25_3_v2.pkl"


@configclass
class PickSingleEgadK070Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/K07_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/K07_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K07_0_v2.pkl"


@configclass
class PickSingleEgadK071Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/K07_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/K07_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K07_1_v2.pkl"


@configclass
class PickSingleEgadK072Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/K07_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/K07_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K07_2_v2.pkl"


@configclass
class PickSingleEgadK073Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/K07_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/K07_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K07_3_v2.pkl"


@configclass
class PickSingleEgadK080Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/K08_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/K08_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K08_0_v2.pkl"


@configclass
class PickSingleEgadK081Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/K08_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/K08_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K08_1_v2.pkl"


@configclass
class PickSingleEgadK082Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/K08_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/K08_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K08_2_v2.pkl"


@configclass
class PickSingleEgadK083Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/K08_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/K08_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K08_3_v2.pkl"


@configclass
class PickSingleEgadK090Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/K09_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/K09_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K09_0_v2.pkl"


@configclass
class PickSingleEgadK092Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/K09_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/K09_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K09_2_v2.pkl"


@configclass
class PickSingleEgadK093Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/K09_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/K09_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K09_3_v2.pkl"


@configclass
class PickSingleEgadK100Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/K10_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/K10_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K10_0_v2.pkl"


@configclass
class PickSingleEgadK101Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/K10_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/K10_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K10_1_v2.pkl"


@configclass
class PickSingleEgadK102Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/K10_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/K10_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K10_2_v2.pkl"


@configclass
class PickSingleEgadK103Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/K10_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/K10_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K10_3_v2.pkl"


@configclass
class PickSingleEgadK110Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/K11_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/K11_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K11_0_v2.pkl"


@configclass
class PickSingleEgadK111Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/K11_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/K11_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K11_1_v2.pkl"


@configclass
class PickSingleEgadK112Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/K11_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/K11_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K11_2_v2.pkl"


@configclass
class PickSingleEgadK113Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/K11_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/K11_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K11_3_v2.pkl"


@configclass
class PickSingleEgadK120Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/K12_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/K12_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K12_0_v2.pkl"


@configclass
class PickSingleEgadK121Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/K12_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/K12_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K12_1_v2.pkl"


@configclass
class PickSingleEgadK122Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/K12_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/K12_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K12_2_v2.pkl"


@configclass
class PickSingleEgadK123Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/K12_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/K12_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K12_3_v2.pkl"


@configclass
class PickSingleEgadK130Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/K13_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/K13_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K13_0_v2.pkl"


@configclass
class PickSingleEgadK132Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/K13_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/K13_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K13_2_v2.pkl"


@configclass
class PickSingleEgadK140Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/K14_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/K14_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K14_0_v2.pkl"


@configclass
class PickSingleEgadK142Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/K14_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/K14_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K14_2_v2.pkl"


@configclass
class PickSingleEgadK143Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/K14_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/K14_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K14_3_v2.pkl"


@configclass
class PickSingleEgadK150Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/K15_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/K15_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K15_0_v2.pkl"


@configclass
class PickSingleEgadK151Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/K15_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/K15_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K15_1_v2.pkl"


@configclass
class PickSingleEgadK152Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/K15_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/K15_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K15_2_v2.pkl"


@configclass
class PickSingleEgadK153Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/K15_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/K15_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K15_3_v2.pkl"


@configclass
class PickSingleEgadK160Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/K16_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/K16_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K16_0_v2.pkl"


@configclass
class PickSingleEgadK161Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/K16_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/K16_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K16_1_v2.pkl"


@configclass
class PickSingleEgadK163Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/K16_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/K16_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K16_3_v2.pkl"


@configclass
class PickSingleEgadK170Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/K17_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/K17_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K17_0_v2.pkl"


@configclass
class PickSingleEgadK171Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/K17_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/K17_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K17_1_v2.pkl"


@configclass
class PickSingleEgadK172Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/K17_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/K17_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K17_2_v2.pkl"


@configclass
class PickSingleEgadK173Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/K17_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/K17_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K17_3_v2.pkl"


@configclass
class PickSingleEgadK180Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/K18_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/K18_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K18_0_v2.pkl"


@configclass
class PickSingleEgadK181Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/K18_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/K18_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K18_1_v2.pkl"


@configclass
class PickSingleEgadK182Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/K18_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/K18_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K18_2_v2.pkl"


@configclass
class PickSingleEgadK183Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/K18_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/K18_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K18_3_v2.pkl"


@configclass
class PickSingleEgadK190Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/K19_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/K19_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K19_0_v2.pkl"


@configclass
class PickSingleEgadK191Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/K19_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/K19_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K19_1_v2.pkl"


@configclass
class PickSingleEgadK192Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/K19_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/K19_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K19_2_v2.pkl"


@configclass
class PickSingleEgadK193Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/K19_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/K19_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K19_3_v2.pkl"


@configclass
class PickSingleEgadK200Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/K20_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/K20_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K20_0_v2.pkl"


@configclass
class PickSingleEgadK201Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/K20_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/K20_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K20_1_v2.pkl"


@configclass
class PickSingleEgadK202Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/K20_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/K20_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K20_2_v2.pkl"


@configclass
class PickSingleEgadK203Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/K20_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/K20_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K20_3_v2.pkl"


@configclass
class PickSingleEgadK210Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/K21_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/K21_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K21_0_v2.pkl"


@configclass
class PickSingleEgadK211Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/K21_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/K21_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K21_1_v2.pkl"


@configclass
class PickSingleEgadK212Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/K21_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/K21_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K21_2_v2.pkl"


@configclass
class PickSingleEgadK213Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/K21_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/K21_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K21_3_v2.pkl"


@configclass
class PickSingleEgadK220Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/K22_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/K22_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K22_0_v2.pkl"


@configclass
class PickSingleEgadK221Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/K22_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/K22_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K22_1_v2.pkl"


@configclass
class PickSingleEgadK222Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/K22_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/K22_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K22_2_v2.pkl"


@configclass
class PickSingleEgadK223Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/K22_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/K22_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K22_3_v2.pkl"


@configclass
class PickSingleEgadK230Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/K23_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/K23_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K23_0_v2.pkl"


@configclass
class PickSingleEgadK231Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/K23_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/K23_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K23_1_v2.pkl"


@configclass
class PickSingleEgadK232Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/K23_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/K23_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K23_2_v2.pkl"


@configclass
class PickSingleEgadK233Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/K23_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/K23_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K23_3_v2.pkl"


@configclass
class PickSingleEgadK240Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/K24_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/K24_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K24_0_v2.pkl"


@configclass
class PickSingleEgadK241Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/K24_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/K24_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K24_1_v2.pkl"


@configclass
class PickSingleEgadK242Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/K24_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/K24_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K24_2_v2.pkl"


@configclass
class PickSingleEgadK243Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/K24_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/K24_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K24_3_v2.pkl"


@configclass
class PickSingleEgadK250Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/K25_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/K25_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K25_0_v2.pkl"


@configclass
class PickSingleEgadK251Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/K25_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/K25_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K25_1_v2.pkl"


@configclass
class PickSingleEgadK252Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/K25_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/K25_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K25_2_v2.pkl"


@configclass
class PickSingleEgadK253Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/K25_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/K25_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-K25_3_v2.pkl"


@configclass
class PickSingleEgadL070Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/L07_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/L07_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L07_0_v2.pkl"


@configclass
class PickSingleEgadL071Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/L07_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/L07_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L07_1_v2.pkl"


@configclass
class PickSingleEgadL072Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/L07_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/L07_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L07_2_v2.pkl"


@configclass
class PickSingleEgadL073Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/L07_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/L07_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L07_3_v2.pkl"


@configclass
class PickSingleEgadL080Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/L08_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/L08_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L08_0_v2.pkl"


@configclass
class PickSingleEgadL081Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/L08_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/L08_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L08_1_v2.pkl"


@configclass
class PickSingleEgadL082Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/L08_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/L08_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L08_2_v2.pkl"


@configclass
class PickSingleEgadL083Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/L08_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/L08_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L08_3_v2.pkl"


@configclass
class PickSingleEgadL090Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/L09_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/L09_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L09_0_v2.pkl"


@configclass
class PickSingleEgadL091Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/L09_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/L09_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L09_1_v2.pkl"


@configclass
class PickSingleEgadL092Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/L09_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/L09_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L09_2_v2.pkl"


@configclass
class PickSingleEgadL093Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/L09_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/L09_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L09_3_v2.pkl"


@configclass
class PickSingleEgadL100Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/L10_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/L10_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L10_0_v2.pkl"


@configclass
class PickSingleEgadL101Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/L10_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/L10_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L10_1_v2.pkl"


@configclass
class PickSingleEgadL102Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/L10_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/L10_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L10_2_v2.pkl"


@configclass
class PickSingleEgadL110Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/L11_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/L11_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L11_0_v2.pkl"


@configclass
class PickSingleEgadL111Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/L11_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/L11_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L11_1_v2.pkl"


@configclass
class PickSingleEgadL112Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/L11_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/L11_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L11_2_v2.pkl"


@configclass
class PickSingleEgadL113Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/L11_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/L11_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L11_3_v2.pkl"


@configclass
class PickSingleEgadL120Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/L12_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/L12_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L12_0_v2.pkl"


@configclass
class PickSingleEgadL121Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/L12_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/L12_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L12_1_v2.pkl"


@configclass
class PickSingleEgadL122Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/L12_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/L12_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L12_2_v2.pkl"


@configclass
class PickSingleEgadL123Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/L12_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/L12_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L12_3_v2.pkl"


@configclass
class PickSingleEgadL130Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/L13_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/L13_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L13_0_v2.pkl"


@configclass
class PickSingleEgadL131Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/L13_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/L13_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L13_1_v2.pkl"


@configclass
class PickSingleEgadL132Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/L13_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/L13_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L13_2_v2.pkl"


@configclass
class PickSingleEgadL133Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/L13_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/L13_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L13_3_v2.pkl"


@configclass
class PickSingleEgadL141Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/L14_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/L14_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L14_1_v2.pkl"


@configclass
class PickSingleEgadL142Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/L14_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/L14_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L14_2_v2.pkl"


@configclass
class PickSingleEgadL143Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/L14_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/L14_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L14_3_v2.pkl"


@configclass
class PickSingleEgadL150Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/L15_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/L15_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L15_0_v2.pkl"


@configclass
class PickSingleEgadL151Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/L15_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/L15_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L15_1_v2.pkl"


@configclass
class PickSingleEgadL153Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/L15_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/L15_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L15_3_v2.pkl"


@configclass
class PickSingleEgadL160Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/L16_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/L16_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L16_0_v2.pkl"


@configclass
class PickSingleEgadL161Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/L16_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/L16_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L16_1_v2.pkl"


@configclass
class PickSingleEgadL162Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/L16_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/L16_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L16_2_v2.pkl"


@configclass
class PickSingleEgadL163Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/L16_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/L16_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L16_3_v2.pkl"


@configclass
class PickSingleEgadL171Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/L17_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/L17_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L17_1_v2.pkl"


@configclass
class PickSingleEgadL172Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/L17_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/L17_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L17_2_v2.pkl"


@configclass
class PickSingleEgadL173Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/L17_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/L17_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L17_3_v2.pkl"


@configclass
class PickSingleEgadL180Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/L18_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/L18_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L18_0_v2.pkl"


@configclass
class PickSingleEgadL181Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/L18_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/L18_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L18_1_v2.pkl"


@configclass
class PickSingleEgadL182Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/L18_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/L18_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L18_2_v2.pkl"


@configclass
class PickSingleEgadL183Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/L18_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/L18_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L18_3_v2.pkl"


@configclass
class PickSingleEgadL191Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/L19_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/L19_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L19_1_v2.pkl"


@configclass
class PickSingleEgadL192Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/L19_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/L19_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L19_2_v2.pkl"


@configclass
class PickSingleEgadL193Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/L19_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/L19_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L19_3_v2.pkl"


@configclass
class PickSingleEgadL200Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/L20_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/L20_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L20_0_v2.pkl"


@configclass
class PickSingleEgadL201Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/L20_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/L20_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L20_1_v2.pkl"


@configclass
class PickSingleEgadL202Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/L20_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/L20_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L20_2_v2.pkl"


@configclass
class PickSingleEgadL203Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/L20_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/L20_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L20_3_v2.pkl"


@configclass
class PickSingleEgadL210Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/L21_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/L21_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L21_0_v2.pkl"


@configclass
class PickSingleEgadL211Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/L21_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/L21_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L21_1_v2.pkl"


@configclass
class PickSingleEgadL212Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/L21_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/L21_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L21_2_v2.pkl"


@configclass
class PickSingleEgadL213Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/L21_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/L21_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L21_3_v2.pkl"


@configclass
class PickSingleEgadL220Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/L22_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/L22_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L22_0_v2.pkl"


@configclass
class PickSingleEgadL221Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/L22_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/L22_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L22_1_v2.pkl"


@configclass
class PickSingleEgadL222Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/L22_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/L22_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L22_2_v2.pkl"


@configclass
class PickSingleEgadL223Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/L22_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/L22_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L22_3_v2.pkl"


@configclass
class PickSingleEgadL230Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/L23_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/L23_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L23_0_v2.pkl"


@configclass
class PickSingleEgadL231Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/L23_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/L23_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L23_1_v2.pkl"


@configclass
class PickSingleEgadL232Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/L23_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/L23_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L23_2_v2.pkl"


@configclass
class PickSingleEgadL233Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/L23_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/L23_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L23_3_v2.pkl"


@configclass
class PickSingleEgadL240Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/L24_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/L24_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L24_0_v2.pkl"


@configclass
class PickSingleEgadL241Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/L24_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/L24_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L24_1_v2.pkl"


@configclass
class PickSingleEgadL243Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/L24_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/L24_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L24_3_v2.pkl"


@configclass
class PickSingleEgadL250Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/L25_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/L25_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L25_0_v2.pkl"


@configclass
class PickSingleEgadL251Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/L25_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/L25_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L25_1_v2.pkl"


@configclass
class PickSingleEgadL252Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/L25_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/L25_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L25_2_v2.pkl"


@configclass
class PickSingleEgadL253Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/L25_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/L25_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-L25_3_v2.pkl"


@configclass
class PickSingleEgadM051Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/M05_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/M05_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M05_1_v2.pkl"


@configclass
class PickSingleEgadM052Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/M05_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/M05_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M05_2_v2.pkl"


@configclass
class PickSingleEgadM053Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/M05_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/M05_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M05_3_v2.pkl"


@configclass
class PickSingleEgadM061Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/M06_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/M06_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M06_1_v2.pkl"


@configclass
class PickSingleEgadM062Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/M06_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/M06_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M06_2_v2.pkl"


@configclass
class PickSingleEgadM063Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/M06_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/M06_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M06_3_v2.pkl"


@configclass
class PickSingleEgadM070Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/M07_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/M07_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M07_0_v2.pkl"


@configclass
class PickSingleEgadM071Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/M07_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/M07_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M07_1_v2.pkl"


@configclass
class PickSingleEgadM073Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/M07_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/M07_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M07_3_v2.pkl"


@configclass
class PickSingleEgadM080Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/M08_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/M08_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M08_0_v2.pkl"


@configclass
class PickSingleEgadM082Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/M08_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/M08_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M08_2_v2.pkl"


@configclass
class PickSingleEgadM083Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/M08_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/M08_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M08_3_v2.pkl"


@configclass
class PickSingleEgadM090Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/M09_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/M09_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M09_0_v2.pkl"


@configclass
class PickSingleEgadM091Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/M09_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/M09_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M09_1_v2.pkl"


@configclass
class PickSingleEgadM092Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/M09_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/M09_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M09_2_v2.pkl"


@configclass
class PickSingleEgadM093Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/M09_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/M09_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M09_3_v2.pkl"


@configclass
class PickSingleEgadM100Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/M10_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/M10_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M10_0_v2.pkl"


@configclass
class PickSingleEgadM101Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/M10_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/M10_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M10_1_v2.pkl"


@configclass
class PickSingleEgadM102Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/M10_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/M10_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M10_2_v2.pkl"


@configclass
class PickSingleEgadM103Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/M10_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/M10_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M10_3_v2.pkl"


@configclass
class PickSingleEgadM110Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/M11_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/M11_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M11_0_v2.pkl"


@configclass
class PickSingleEgadM111Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/M11_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/M11_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M11_1_v2.pkl"


@configclass
class PickSingleEgadM112Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/M11_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/M11_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M11_2_v2.pkl"


@configclass
class PickSingleEgadM113Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/M11_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/M11_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M11_3_v2.pkl"


@configclass
class PickSingleEgadM120Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/M12_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/M12_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M12_0_v2.pkl"


@configclass
class PickSingleEgadM121Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/M12_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/M12_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M12_1_v2.pkl"


@configclass
class PickSingleEgadM122Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/M12_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/M12_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M12_2_v2.pkl"


@configclass
class PickSingleEgadM123Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/M12_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/M12_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M12_3_v2.pkl"


@configclass
class PickSingleEgadM130Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/M13_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/M13_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M13_0_v2.pkl"


@configclass
class PickSingleEgadM131Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/M13_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/M13_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M13_1_v2.pkl"


@configclass
class PickSingleEgadM132Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/M13_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/M13_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M13_2_v2.pkl"


@configclass
class PickSingleEgadM133Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/M13_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/M13_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M13_3_v2.pkl"


@configclass
class PickSingleEgadM140Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/M14_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/M14_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M14_0_v2.pkl"


@configclass
class PickSingleEgadM141Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/M14_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/M14_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M14_1_v2.pkl"


@configclass
class PickSingleEgadM142Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/M14_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/M14_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M14_2_v2.pkl"


@configclass
class PickSingleEgadM143Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/M14_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/M14_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M14_3_v2.pkl"


@configclass
class PickSingleEgadM150Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/M15_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/M15_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M15_0_v2.pkl"


@configclass
class PickSingleEgadM151Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/M15_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/M15_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M15_1_v2.pkl"


@configclass
class PickSingleEgadM152Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/M15_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/M15_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M15_2_v2.pkl"


@configclass
class PickSingleEgadM153Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/M15_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/M15_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M15_3_v2.pkl"


@configclass
class PickSingleEgadM160Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/M16_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/M16_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M16_0_v2.pkl"


@configclass
class PickSingleEgadM161Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/M16_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/M16_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M16_1_v2.pkl"


@configclass
class PickSingleEgadM162Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/M16_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/M16_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M16_2_v2.pkl"


@configclass
class PickSingleEgadM163Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/M16_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/M16_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M16_3_v2.pkl"


@configclass
class PickSingleEgadM171Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/M17_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/M17_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M17_1_v2.pkl"


@configclass
class PickSingleEgadM172Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/M17_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/M17_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M17_2_v2.pkl"


@configclass
class PickSingleEgadM173Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/M17_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/M17_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M17_3_v2.pkl"


@configclass
class PickSingleEgadM180Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/M18_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/M18_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M18_0_v2.pkl"


@configclass
class PickSingleEgadM181Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/M18_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/M18_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M18_1_v2.pkl"


@configclass
class PickSingleEgadM182Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/M18_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/M18_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M18_2_v2.pkl"


@configclass
class PickSingleEgadM183Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/M18_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/M18_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M18_3_v2.pkl"


@configclass
class PickSingleEgadM190Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/M19_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/M19_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M19_0_v2.pkl"


@configclass
class PickSingleEgadM191Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/M19_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/M19_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M19_1_v2.pkl"


@configclass
class PickSingleEgadM193Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/M19_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/M19_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M19_3_v2.pkl"


@configclass
class PickSingleEgadM200Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/M20_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/M20_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M20_0_v2.pkl"


@configclass
class PickSingleEgadM201Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/M20_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/M20_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M20_1_v2.pkl"


@configclass
class PickSingleEgadM202Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/M20_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/M20_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M20_2_v2.pkl"


@configclass
class PickSingleEgadM203Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/M20_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/M20_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M20_3_v2.pkl"


@configclass
class PickSingleEgadM210Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/M21_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/M21_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M21_0_v2.pkl"


@configclass
class PickSingleEgadM211Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/M21_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/M21_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M21_1_v2.pkl"


@configclass
class PickSingleEgadM213Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/M21_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/M21_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M21_3_v2.pkl"


@configclass
class PickSingleEgadM221Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/M22_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/M22_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M22_1_v2.pkl"


@configclass
class PickSingleEgadM222Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/M22_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/M22_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M22_2_v2.pkl"


@configclass
class PickSingleEgadM223Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/M22_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/M22_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M22_3_v2.pkl"


@configclass
class PickSingleEgadM230Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/M23_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/M23_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M23_0_v2.pkl"


@configclass
class PickSingleEgadM231Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/M23_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/M23_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M23_1_v2.pkl"


@configclass
class PickSingleEgadM232Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/M23_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/M23_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M23_2_v2.pkl"


@configclass
class PickSingleEgadM233Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/M23_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/M23_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M23_3_v2.pkl"


@configclass
class PickSingleEgadM240Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/M24_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/M24_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M24_0_v2.pkl"


@configclass
class PickSingleEgadM241Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/M24_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/M24_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M24_1_v2.pkl"


@configclass
class PickSingleEgadM242Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/M24_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/M24_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M24_2_v2.pkl"


@configclass
class PickSingleEgadM243Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/M24_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/M24_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M24_3_v2.pkl"


@configclass
class PickSingleEgadM250Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/M25_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/M25_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M25_0_v2.pkl"


@configclass
class PickSingleEgadM251Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/M25_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/M25_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M25_1_v2.pkl"


@configclass
class PickSingleEgadM252Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/M25_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/M25_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M25_2_v2.pkl"


@configclass
class PickSingleEgadM253Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/M25_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/M25_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-M25_3_v2.pkl"


@configclass
class PickSingleEgadN050Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/N05_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/N05_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N05_0_v2.pkl"


@configclass
class PickSingleEgadN051Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/N05_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/N05_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N05_1_v2.pkl"


@configclass
class PickSingleEgadN052Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/N05_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/N05_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N05_2_v2.pkl"


@configclass
class PickSingleEgadN060Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/N06_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/N06_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N06_0_v2.pkl"


@configclass
class PickSingleEgadN061Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/N06_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/N06_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N06_1_v2.pkl"


@configclass
class PickSingleEgadN062Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/N06_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/N06_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N06_2_v2.pkl"


@configclass
class PickSingleEgadN063Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/N06_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/N06_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N06_3_v2.pkl"


@configclass
class PickSingleEgadN070Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/N07_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/N07_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N07_0_v2.pkl"


@configclass
class PickSingleEgadN071Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/N07_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/N07_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N07_1_v2.pkl"


@configclass
class PickSingleEgadN072Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/N07_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/N07_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N07_2_v2.pkl"


@configclass
class PickSingleEgadN073Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/N07_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/N07_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N07_3_v2.pkl"


@configclass
class PickSingleEgadN080Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/N08_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/N08_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N08_0_v2.pkl"


@configclass
class PickSingleEgadN081Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/N08_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/N08_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N08_1_v2.pkl"


@configclass
class PickSingleEgadN083Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/N08_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/N08_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N08_3_v2.pkl"


@configclass
class PickSingleEgadN090Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/N09_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/N09_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N09_0_v2.pkl"


@configclass
class PickSingleEgadN091Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/N09_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/N09_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N09_1_v2.pkl"


@configclass
class PickSingleEgadN092Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/N09_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/N09_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N09_2_v2.pkl"


@configclass
class PickSingleEgadN093Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/N09_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/N09_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N09_3_v2.pkl"


@configclass
class PickSingleEgadN100Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/N10_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/N10_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N10_0_v2.pkl"


@configclass
class PickSingleEgadN101Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/N10_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/N10_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N10_1_v2.pkl"


@configclass
class PickSingleEgadN103Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/N10_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/N10_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N10_3_v2.pkl"


@configclass
class PickSingleEgadN110Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/N11_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/N11_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N11_0_v2.pkl"


@configclass
class PickSingleEgadN111Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/N11_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/N11_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N11_1_v2.pkl"


@configclass
class PickSingleEgadN112Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/N11_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/N11_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N11_2_v2.pkl"


@configclass
class PickSingleEgadN113Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/N11_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/N11_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N11_3_v2.pkl"


@configclass
class PickSingleEgadN120Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/N12_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/N12_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N12_0_v2.pkl"


@configclass
class PickSingleEgadN121Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/N12_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/N12_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N12_1_v2.pkl"


@configclass
class PickSingleEgadN123Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/N12_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/N12_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N12_3_v2.pkl"


@configclass
class PickSingleEgadN130Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/N13_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/N13_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N13_0_v2.pkl"


@configclass
class PickSingleEgadN131Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/N13_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/N13_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N13_1_v2.pkl"


@configclass
class PickSingleEgadN133Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/N13_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/N13_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N13_3_v2.pkl"


@configclass
class PickSingleEgadN143Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/N14_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/N14_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N14_3_v2.pkl"


@configclass
class PickSingleEgadN150Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/N15_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/N15_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N15_0_v2.pkl"


@configclass
class PickSingleEgadN151Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/N15_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/N15_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N15_1_v2.pkl"


@configclass
class PickSingleEgadN152Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/N15_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/N15_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N15_2_v2.pkl"


@configclass
class PickSingleEgadN153Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/N15_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/N15_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N15_3_v2.pkl"


@configclass
class PickSingleEgadN160Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/N16_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/N16_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N16_0_v2.pkl"


@configclass
class PickSingleEgadN161Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/N16_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/N16_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N16_1_v2.pkl"


@configclass
class PickSingleEgadN162Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/N16_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/N16_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N16_2_v2.pkl"


@configclass
class PickSingleEgadN163Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/N16_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/N16_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N16_3_v2.pkl"


@configclass
class PickSingleEgadN170Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/N17_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/N17_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N17_0_v2.pkl"


@configclass
class PickSingleEgadN171Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/N17_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/N17_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N17_1_v2.pkl"


@configclass
class PickSingleEgadN172Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/N17_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/N17_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N17_2_v2.pkl"


@configclass
class PickSingleEgadN173Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/N17_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/N17_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N17_3_v2.pkl"


@configclass
class PickSingleEgadN180Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/N18_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/N18_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N18_0_v2.pkl"


@configclass
class PickSingleEgadN181Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/N18_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/N18_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N18_1_v2.pkl"


@configclass
class PickSingleEgadN182Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/N18_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/N18_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N18_2_v2.pkl"


@configclass
class PickSingleEgadN183Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/N18_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/N18_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N18_3_v2.pkl"


@configclass
class PickSingleEgadN190Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/N19_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/N19_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N19_0_v2.pkl"


@configclass
class PickSingleEgadN191Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/N19_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/N19_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N19_1_v2.pkl"


@configclass
class PickSingleEgadN192Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/N19_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/N19_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N19_2_v2.pkl"


@configclass
class PickSingleEgadN193Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/N19_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/N19_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N19_3_v2.pkl"


@configclass
class PickSingleEgadN200Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/N20_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/N20_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N20_0_v2.pkl"


@configclass
class PickSingleEgadN201Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/N20_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/N20_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N20_1_v2.pkl"


@configclass
class PickSingleEgadN202Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/N20_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/N20_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N20_2_v2.pkl"


@configclass
class PickSingleEgadN203Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/N20_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/N20_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N20_3_v2.pkl"


@configclass
class PickSingleEgadN210Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/N21_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/N21_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N21_0_v2.pkl"


@configclass
class PickSingleEgadN211Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/N21_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/N21_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N21_1_v2.pkl"


@configclass
class PickSingleEgadN212Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/N21_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/N21_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N21_2_v2.pkl"


@configclass
class PickSingleEgadN213Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/N21_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/N21_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N21_3_v2.pkl"


@configclass
class PickSingleEgadN220Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/N22_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/N22_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N22_0_v2.pkl"


@configclass
class PickSingleEgadN222Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/N22_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/N22_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N22_2_v2.pkl"


@configclass
class PickSingleEgadN223Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/N22_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/N22_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N22_3_v2.pkl"


@configclass
class PickSingleEgadN230Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/N23_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/N23_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N23_0_v2.pkl"


@configclass
class PickSingleEgadN231Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/N23_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/N23_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N23_1_v2.pkl"


@configclass
class PickSingleEgadN233Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/N23_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/N23_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N23_3_v2.pkl"


@configclass
class PickSingleEgadN240Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/N24_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/N24_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N24_0_v2.pkl"


@configclass
class PickSingleEgadN241Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/N24_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/N24_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N24_1_v2.pkl"


@configclass
class PickSingleEgadN242Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/N24_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/N24_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N24_2_v2.pkl"


@configclass
class PickSingleEgadN243Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/N24_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/N24_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N24_3_v2.pkl"


@configclass
class PickSingleEgadN250Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/N25_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/N25_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N25_0_v2.pkl"


@configclass
class PickSingleEgadN251Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/N25_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/N25_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N25_1_v2.pkl"


@configclass
class PickSingleEgadN252Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/N25_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/N25_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N25_2_v2.pkl"


@configclass
class PickSingleEgadN253Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/N25_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/N25_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-N25_3_v2.pkl"


@configclass
class PickSingleEgadO050Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/O05_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/O05_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O05_0_v2.pkl"


@configclass
class PickSingleEgadO051Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/O05_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/O05_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O05_1_v2.pkl"


@configclass
class PickSingleEgadO053Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/O05_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/O05_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O05_3_v2.pkl"


@configclass
class PickSingleEgadO060Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/O06_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/O06_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O06_0_v2.pkl"


@configclass
class PickSingleEgadO061Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/O06_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/O06_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O06_1_v2.pkl"


@configclass
class PickSingleEgadO062Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/O06_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/O06_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O06_2_v2.pkl"


@configclass
class PickSingleEgadO063Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/O06_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/O06_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O06_3_v2.pkl"


@configclass
class PickSingleEgadO070Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/O07_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/O07_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O07_0_v2.pkl"


@configclass
class PickSingleEgadO072Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/O07_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/O07_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O07_2_v2.pkl"


@configclass
class PickSingleEgadO073Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/O07_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/O07_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O07_3_v2.pkl"


@configclass
class PickSingleEgadO080Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/O08_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/O08_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O08_0_v2.pkl"


@configclass
class PickSingleEgadO081Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/O08_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/O08_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O08_1_v2.pkl"


@configclass
class PickSingleEgadO082Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/O08_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/O08_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O08_2_v2.pkl"


@configclass
class PickSingleEgadO083Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/O08_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/O08_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O08_3_v2.pkl"


@configclass
class PickSingleEgadO090Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/O09_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/O09_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O09_0_v2.pkl"


@configclass
class PickSingleEgadO091Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/O09_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/O09_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O09_1_v2.pkl"


@configclass
class PickSingleEgadO092Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/O09_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/O09_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O09_2_v2.pkl"


@configclass
class PickSingleEgadO093Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/O09_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/O09_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O09_3_v2.pkl"


@configclass
class PickSingleEgadO100Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/O10_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/O10_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O10_0_v2.pkl"


@configclass
class PickSingleEgadO101Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/O10_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/O10_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O10_1_v2.pkl"


@configclass
class PickSingleEgadO102Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/O10_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/O10_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O10_2_v2.pkl"


@configclass
class PickSingleEgadO103Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/O10_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/O10_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O10_3_v2.pkl"


@configclass
class PickSingleEgadO110Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/O11_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/O11_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O11_0_v2.pkl"


@configclass
class PickSingleEgadO112Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/O11_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/O11_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O11_2_v2.pkl"


@configclass
class PickSingleEgadO113Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/O11_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/O11_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O11_3_v2.pkl"


@configclass
class PickSingleEgadO120Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/O12_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/O12_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O12_0_v2.pkl"


@configclass
class PickSingleEgadO121Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/O12_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/O12_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O12_1_v2.pkl"


@configclass
class PickSingleEgadO122Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/O12_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/O12_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O12_2_v2.pkl"


@configclass
class PickSingleEgadO123Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/O12_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/O12_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O12_3_v2.pkl"


@configclass
class PickSingleEgadO130Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/O13_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/O13_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O13_0_v2.pkl"


@configclass
class PickSingleEgadO131Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/O13_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/O13_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O13_1_v2.pkl"


@configclass
class PickSingleEgadO132Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/O13_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/O13_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O13_2_v2.pkl"


@configclass
class PickSingleEgadO133Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/O13_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/O13_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O13_3_v2.pkl"


@configclass
class PickSingleEgadO140Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/O14_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/O14_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O14_0_v2.pkl"


@configclass
class PickSingleEgadO141Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/O14_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/O14_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O14_1_v2.pkl"


@configclass
class PickSingleEgadO142Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/O14_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/O14_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O14_2_v2.pkl"


@configclass
class PickSingleEgadO150Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/O15_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/O15_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O15_0_v2.pkl"


@configclass
class PickSingleEgadO151Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/O15_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/O15_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O15_1_v2.pkl"


@configclass
class PickSingleEgadO152Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/O15_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/O15_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O15_2_v2.pkl"


@configclass
class PickSingleEgadO153Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/O15_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/O15_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O15_3_v2.pkl"


@configclass
class PickSingleEgadO160Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/O16_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/O16_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O16_0_v2.pkl"


@configclass
class PickSingleEgadO162Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/O16_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/O16_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O16_2_v2.pkl"


@configclass
class PickSingleEgadO163Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/O16_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/O16_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O16_3_v2.pkl"


@configclass
class PickSingleEgadO170Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/O17_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/O17_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O17_0_v2.pkl"


@configclass
class PickSingleEgadO171Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/O17_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/O17_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O17_1_v2.pkl"


@configclass
class PickSingleEgadO172Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/O17_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/O17_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O17_2_v2.pkl"


@configclass
class PickSingleEgadO173Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/O17_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/O17_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O17_3_v2.pkl"


@configclass
class PickSingleEgadO180Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/O18_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/O18_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O18_0_v2.pkl"


@configclass
class PickSingleEgadO181Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/O18_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/O18_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O18_1_v2.pkl"


@configclass
class PickSingleEgadO182Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/O18_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/O18_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O18_2_v2.pkl"


@configclass
class PickSingleEgadO183Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/O18_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/O18_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O18_3_v2.pkl"


@configclass
class PickSingleEgadO190Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/O19_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/O19_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O19_0_v2.pkl"


@configclass
class PickSingleEgadO191Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/O19_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/O19_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O19_1_v2.pkl"


@configclass
class PickSingleEgadO192Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/O19_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/O19_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O19_2_v2.pkl"


@configclass
class PickSingleEgadO193Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/O19_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/O19_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O19_3_v2.pkl"


@configclass
class PickSingleEgadO200Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/O20_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/O20_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O20_0_v2.pkl"


@configclass
class PickSingleEgadO201Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/O20_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/O20_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O20_1_v2.pkl"


@configclass
class PickSingleEgadO202Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/O20_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/O20_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O20_2_v2.pkl"


@configclass
class PickSingleEgadO203Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/O20_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/O20_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O20_3_v2.pkl"


@configclass
class PickSingleEgadO211Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/O21_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/O21_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O21_1_v2.pkl"


@configclass
class PickSingleEgadO212Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/O21_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/O21_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O21_2_v2.pkl"


@configclass
class PickSingleEgadO213Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/O21_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/O21_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O21_3_v2.pkl"


@configclass
class PickSingleEgadO220Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/O22_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/O22_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O22_0_v2.pkl"


@configclass
class PickSingleEgadO221Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/O22_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/O22_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O22_1_v2.pkl"


@configclass
class PickSingleEgadO222Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/O22_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/O22_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O22_2_v2.pkl"


@configclass
class PickSingleEgadO223Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/O22_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/O22_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O22_3_v2.pkl"


@configclass
class PickSingleEgadO230Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/O23_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/O23_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O23_0_v2.pkl"


@configclass
class PickSingleEgadO231Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/O23_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/O23_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O23_1_v2.pkl"


@configclass
class PickSingleEgadO232Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/O23_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/O23_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O23_2_v2.pkl"


@configclass
class PickSingleEgadO233Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/O23_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/O23_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O23_3_v2.pkl"


@configclass
class PickSingleEgadO240Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/O24_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/O24_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O24_0_v2.pkl"


@configclass
class PickSingleEgadO241Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/O24_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/O24_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O24_1_v2.pkl"


@configclass
class PickSingleEgadO242Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/O24_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/O24_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O24_2_v2.pkl"


@configclass
class PickSingleEgadO243Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/O24_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/O24_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O24_3_v2.pkl"


@configclass
class PickSingleEgadO250Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/O25_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/O25_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O25_0_v2.pkl"


@configclass
class PickSingleEgadO251Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/O25_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/O25_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O25_1_v2.pkl"


@configclass
class PickSingleEgadO252Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/O25_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/O25_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O25_2_v2.pkl"


@configclass
class PickSingleEgadO253Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/O25_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/O25_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-O25_3_v2.pkl"


@configclass
class PickSingleEgadP050Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/P05_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/P05_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P05_0_v2.pkl"


@configclass
class PickSingleEgadP051Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/P05_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/P05_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P05_1_v2.pkl"


@configclass
class PickSingleEgadP052Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/P05_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/P05_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P05_2_v2.pkl"


@configclass
class PickSingleEgadP053Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/P05_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/P05_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P05_3_v2.pkl"


@configclass
class PickSingleEgadP060Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/P06_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/P06_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P06_0_v2.pkl"


@configclass
class PickSingleEgadP061Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/P06_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/P06_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P06_1_v2.pkl"


@configclass
class PickSingleEgadP062Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/P06_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/P06_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P06_2_v2.pkl"


@configclass
class PickSingleEgadP070Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/P07_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/P07_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P07_0_v2.pkl"


@configclass
class PickSingleEgadP071Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/P07_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/P07_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P07_1_v2.pkl"


@configclass
class PickSingleEgadP072Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/P07_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/P07_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P07_2_v2.pkl"


@configclass
class PickSingleEgadP080Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/P08_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/P08_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P08_0_v2.pkl"


@configclass
class PickSingleEgadP081Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/P08_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/P08_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P08_1_v2.pkl"


@configclass
class PickSingleEgadP083Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/P08_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/P08_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P08_3_v2.pkl"


@configclass
class PickSingleEgadP090Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/P09_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/P09_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P09_0_v2.pkl"


@configclass
class PickSingleEgadP091Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/P09_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/P09_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P09_1_v2.pkl"


@configclass
class PickSingleEgadP092Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/P09_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/P09_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P09_2_v2.pkl"


@configclass
class PickSingleEgadP093Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/P09_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/P09_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P09_3_v2.pkl"


@configclass
class PickSingleEgadP100Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/P10_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/P10_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P10_0_v2.pkl"


@configclass
class PickSingleEgadP101Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/P10_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/P10_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P10_1_v2.pkl"


@configclass
class PickSingleEgadP102Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/P10_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/P10_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P10_2_v2.pkl"


@configclass
class PickSingleEgadP103Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/P10_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/P10_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P10_3_v2.pkl"


@configclass
class PickSingleEgadP110Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/P11_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/P11_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P11_0_v2.pkl"


@configclass
class PickSingleEgadP111Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/P11_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/P11_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P11_1_v2.pkl"


@configclass
class PickSingleEgadP112Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/P11_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/P11_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P11_2_v2.pkl"


@configclass
class PickSingleEgadP113Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/P11_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/P11_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P11_3_v2.pkl"


@configclass
class PickSingleEgadP120Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/P12_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/P12_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P12_0_v2.pkl"


@configclass
class PickSingleEgadP121Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/P12_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/P12_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P12_1_v2.pkl"


@configclass
class PickSingleEgadP122Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/P12_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/P12_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P12_2_v2.pkl"


@configclass
class PickSingleEgadP123Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/P12_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/P12_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P12_3_v2.pkl"


@configclass
class PickSingleEgadP130Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/P13_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/P13_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P13_0_v2.pkl"


@configclass
class PickSingleEgadP131Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/P13_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/P13_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P13_1_v2.pkl"


@configclass
class PickSingleEgadP132Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/P13_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/P13_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P13_2_v2.pkl"


@configclass
class PickSingleEgadP133Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/P13_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/P13_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P13_3_v2.pkl"


@configclass
class PickSingleEgadP140Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/P14_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/P14_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P14_0_v2.pkl"


@configclass
class PickSingleEgadP141Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/P14_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/P14_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P14_1_v2.pkl"


@configclass
class PickSingleEgadP142Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/P14_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/P14_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P14_2_v2.pkl"


@configclass
class PickSingleEgadP143Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/P14_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/P14_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P14_3_v2.pkl"


@configclass
class PickSingleEgadP150Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/P15_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/P15_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P15_0_v2.pkl"


@configclass
class PickSingleEgadP151Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/P15_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/P15_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P15_1_v2.pkl"


@configclass
class PickSingleEgadP152Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/P15_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/P15_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P15_2_v2.pkl"


@configclass
class PickSingleEgadP153Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/P15_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/P15_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P15_3_v2.pkl"


@configclass
class PickSingleEgadP160Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/P16_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/P16_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P16_0_v2.pkl"


@configclass
class PickSingleEgadP161Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/P16_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/P16_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P16_1_v2.pkl"


@configclass
class PickSingleEgadP162Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/P16_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/P16_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P16_2_v2.pkl"


@configclass
class PickSingleEgadP163Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/P16_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/P16_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P16_3_v2.pkl"


@configclass
class PickSingleEgadP170Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/P17_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/P17_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P17_0_v2.pkl"


@configclass
class PickSingleEgadP171Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/P17_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/P17_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P17_1_v2.pkl"


@configclass
class PickSingleEgadP172Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/P17_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/P17_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P17_2_v2.pkl"


@configclass
class PickSingleEgadP173Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/P17_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/P17_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P17_3_v2.pkl"


@configclass
class PickSingleEgadP180Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/P18_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/P18_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P18_0_v2.pkl"


@configclass
class PickSingleEgadP181Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/P18_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/P18_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P18_1_v2.pkl"


@configclass
class PickSingleEgadP182Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/P18_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/P18_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P18_2_v2.pkl"


@configclass
class PickSingleEgadP183Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/P18_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/P18_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P18_3_v2.pkl"


@configclass
class PickSingleEgadP190Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/P19_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/P19_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P19_0_v2.pkl"


@configclass
class PickSingleEgadP191Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/P19_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/P19_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P19_1_v2.pkl"


@configclass
class PickSingleEgadP192Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/P19_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/P19_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P19_2_v2.pkl"


@configclass
class PickSingleEgadP200Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/P20_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/P20_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P20_0_v2.pkl"


@configclass
class PickSingleEgadP202Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/P20_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/P20_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P20_2_v2.pkl"


@configclass
class PickSingleEgadP203Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/P20_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/P20_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P20_3_v2.pkl"


@configclass
class PickSingleEgadP211Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/P21_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/P21_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P21_1_v2.pkl"


@configclass
class PickSingleEgadP212Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/P21_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/P21_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P21_2_v2.pkl"


@configclass
class PickSingleEgadP213Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/P21_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/P21_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P21_3_v2.pkl"


@configclass
class PickSingleEgadP220Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/P22_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/P22_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P22_0_v2.pkl"


@configclass
class PickSingleEgadP221Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/P22_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/P22_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P22_1_v2.pkl"


@configclass
class PickSingleEgadP222Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/P22_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/P22_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P22_2_v2.pkl"


@configclass
class PickSingleEgadP223Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/P22_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/P22_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P22_3_v2.pkl"


@configclass
class PickSingleEgadP230Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/P23_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/P23_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P23_0_v2.pkl"


@configclass
class PickSingleEgadP231Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/P23_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/P23_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P23_1_v2.pkl"


@configclass
class PickSingleEgadP232Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/P23_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/P23_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P23_2_v2.pkl"


@configclass
class PickSingleEgadP233Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/P23_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/P23_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P23_3_v2.pkl"


@configclass
class PickSingleEgadP240Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/P24_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/P24_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P24_0_v2.pkl"


@configclass
class PickSingleEgadP241Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/P24_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/P24_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P24_1_v2.pkl"


@configclass
class PickSingleEgadP242Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/P24_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/P24_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P24_2_v2.pkl"


@configclass
class PickSingleEgadP243Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/P24_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/P24_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P24_3_v2.pkl"


@configclass
class PickSingleEgadP250Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/P25_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/P25_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P25_0_v2.pkl"


@configclass
class PickSingleEgadP251Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/P25_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/P25_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P25_1_v2.pkl"


@configclass
class PickSingleEgadP252Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/P25_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/P25_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P25_2_v2.pkl"


@configclass
class PickSingleEgadP253Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/P25_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/P25_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-P25_3_v2.pkl"


@configclass
class PickSingleEgadQ050Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/Q05_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q05_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q05_0_v2.pkl"


@configclass
class PickSingleEgadQ051Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/Q05_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q05_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q05_1_v2.pkl"


@configclass
class PickSingleEgadQ052Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/Q05_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q05_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q05_2_v2.pkl"


@configclass
class PickSingleEgadQ053Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/Q05_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q05_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q05_3_v2.pkl"


@configclass
class PickSingleEgadQ062Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/Q06_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q06_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q06_2_v2.pkl"


@configclass
class PickSingleEgadQ063Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/Q06_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q06_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q06_3_v2.pkl"


@configclass
class PickSingleEgadQ070Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/Q07_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q07_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q07_0_v2.pkl"


@configclass
class PickSingleEgadQ072Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/Q07_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q07_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q07_2_v2.pkl"


@configclass
class PickSingleEgadQ073Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/Q07_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q07_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q07_3_v2.pkl"


@configclass
class PickSingleEgadQ080Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/Q08_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q08_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q08_0_v2.pkl"


@configclass
class PickSingleEgadQ081Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/Q08_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q08_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q08_1_v2.pkl"


@configclass
class PickSingleEgadQ083Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/Q08_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q08_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q08_3_v2.pkl"


@configclass
class PickSingleEgadQ090Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/Q09_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q09_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q09_0_v2.pkl"


@configclass
class PickSingleEgadQ091Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/Q09_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q09_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q09_1_v2.pkl"


@configclass
class PickSingleEgadQ092Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/Q09_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q09_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q09_2_v2.pkl"


@configclass
class PickSingleEgadQ093Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/Q09_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q09_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q09_3_v2.pkl"


@configclass
class PickSingleEgadQ100Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/Q10_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q10_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q10_0_v2.pkl"


@configclass
class PickSingleEgadQ101Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/Q10_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q10_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q10_1_v2.pkl"


@configclass
class PickSingleEgadQ102Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/Q10_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q10_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q10_2_v2.pkl"


@configclass
class PickSingleEgadQ103Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/Q10_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q10_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q10_3_v2.pkl"


@configclass
class PickSingleEgadQ110Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/Q11_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q11_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q11_0_v2.pkl"


@configclass
class PickSingleEgadQ111Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/Q11_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q11_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q11_1_v2.pkl"


@configclass
class PickSingleEgadQ112Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/Q11_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q11_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q11_2_v2.pkl"


@configclass
class PickSingleEgadQ113Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/Q11_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q11_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q11_3_v2.pkl"


@configclass
class PickSingleEgadQ120Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/Q12_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q12_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q12_0_v2.pkl"


@configclass
class PickSingleEgadQ122Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/Q12_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q12_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q12_2_v2.pkl"


@configclass
class PickSingleEgadQ123Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/Q12_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q12_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q12_3_v2.pkl"


@configclass
class PickSingleEgadQ130Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/Q13_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q13_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q13_0_v2.pkl"


@configclass
class PickSingleEgadQ131Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/Q13_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q13_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q13_1_v2.pkl"


@configclass
class PickSingleEgadQ140Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/Q14_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q14_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q14_0_v2.pkl"


@configclass
class PickSingleEgadQ141Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/Q14_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q14_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q14_1_v2.pkl"


@configclass
class PickSingleEgadQ142Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/Q14_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q14_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q14_2_v2.pkl"


@configclass
class PickSingleEgadQ143Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/Q14_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q14_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q14_3_v2.pkl"


@configclass
class PickSingleEgadQ150Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/Q15_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q15_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q15_0_v2.pkl"


@configclass
class PickSingleEgadQ151Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/Q15_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q15_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q15_1_v2.pkl"


@configclass
class PickSingleEgadQ152Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/Q15_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q15_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q15_2_v2.pkl"


@configclass
class PickSingleEgadQ153Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/Q15_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q15_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q15_3_v2.pkl"


@configclass
class PickSingleEgadQ160Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/Q16_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q16_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q16_0_v2.pkl"


@configclass
class PickSingleEgadQ161Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/Q16_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q16_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q16_1_v2.pkl"


@configclass
class PickSingleEgadQ162Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/Q16_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q16_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q16_2_v2.pkl"


@configclass
class PickSingleEgadQ163Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/Q16_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q16_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q16_3_v2.pkl"


@configclass
class PickSingleEgadQ170Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/Q17_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q17_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q17_0_v2.pkl"


@configclass
class PickSingleEgadQ171Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/Q17_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q17_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q17_1_v2.pkl"


@configclass
class PickSingleEgadQ180Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/Q18_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q18_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q18_0_v2.pkl"


@configclass
class PickSingleEgadQ181Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/Q18_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q18_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q18_1_v2.pkl"


@configclass
class PickSingleEgadQ182Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/Q18_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q18_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q18_2_v2.pkl"


@configclass
class PickSingleEgadQ183Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/Q18_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q18_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q18_3_v2.pkl"


@configclass
class PickSingleEgadQ190Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/Q19_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q19_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q19_0_v2.pkl"


@configclass
class PickSingleEgadQ191Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/Q19_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q19_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q19_1_v2.pkl"


@configclass
class PickSingleEgadQ192Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/Q19_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q19_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q19_2_v2.pkl"


@configclass
class PickSingleEgadQ193Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/Q19_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q19_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q19_3_v2.pkl"


@configclass
class PickSingleEgadQ200Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/Q20_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q20_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q20_0_v2.pkl"


@configclass
class PickSingleEgadQ202Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/Q20_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q20_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q20_2_v2.pkl"


@configclass
class PickSingleEgadQ203Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/Q20_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q20_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q20_3_v2.pkl"


@configclass
class PickSingleEgadQ210Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/Q21_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q21_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q21_0_v2.pkl"


@configclass
class PickSingleEgadQ211Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/Q21_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q21_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q21_1_v2.pkl"


@configclass
class PickSingleEgadQ212Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/Q21_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q21_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q21_2_v2.pkl"


@configclass
class PickSingleEgadQ213Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/Q21_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q21_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q21_3_v2.pkl"


@configclass
class PickSingleEgadQ220Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/Q22_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q22_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q22_0_v2.pkl"


@configclass
class PickSingleEgadQ221Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/Q22_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q22_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q22_1_v2.pkl"


@configclass
class PickSingleEgadQ222Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/Q22_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q22_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q22_2_v2.pkl"


@configclass
class PickSingleEgadQ223Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/Q22_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q22_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q22_3_v2.pkl"


@configclass
class PickSingleEgadQ230Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/Q23_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q23_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q23_0_v2.pkl"


@configclass
class PickSingleEgadQ231Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/Q23_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q23_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q23_1_v2.pkl"


@configclass
class PickSingleEgadQ232Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/Q23_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q23_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q23_2_v2.pkl"


@configclass
class PickSingleEgadQ240Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/Q24_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q24_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q24_0_v2.pkl"


@configclass
class PickSingleEgadQ241Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/Q24_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q24_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q24_1_v2.pkl"


@configclass
class PickSingleEgadQ242Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/Q24_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q24_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q24_2_v2.pkl"


@configclass
class PickSingleEgadQ243Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/Q24_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q24_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q24_3_v2.pkl"


@configclass
class PickSingleEgadQ250Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/Q25_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q25_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q25_0_v2.pkl"


@configclass
class PickSingleEgadQ251Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/Q25_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q25_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q25_1_v2.pkl"


@configclass
class PickSingleEgadQ253Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/Q25_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/Q25_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-Q25_3_v2.pkl"


@configclass
class PickSingleEgadR050Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/R05_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/R05_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R05_0_v2.pkl"


@configclass
class PickSingleEgadR052Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/R05_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/R05_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R05_2_v2.pkl"


@configclass
class PickSingleEgadR053Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/R05_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/R05_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R05_3_v2.pkl"


@configclass
class PickSingleEgadR060Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/R06_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/R06_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R06_0_v2.pkl"


@configclass
class PickSingleEgadR061Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/R06_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/R06_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R06_1_v2.pkl"


@configclass
class PickSingleEgadR062Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/R06_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/R06_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R06_2_v2.pkl"


@configclass
class PickSingleEgadR063Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/R06_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/R06_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R06_3_v2.pkl"


@configclass
class PickSingleEgadR070Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/R07_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/R07_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R07_0_v2.pkl"


@configclass
class PickSingleEgadR071Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/R07_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/R07_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R07_1_v2.pkl"


@configclass
class PickSingleEgadR073Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/R07_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/R07_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R07_3_v2.pkl"


@configclass
class PickSingleEgadR081Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/R08_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/R08_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R08_1_v2.pkl"


@configclass
class PickSingleEgadR082Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/R08_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/R08_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R08_2_v2.pkl"


@configclass
class PickSingleEgadR083Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/R08_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/R08_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R08_3_v2.pkl"


@configclass
class PickSingleEgadR090Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/R09_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/R09_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R09_0_v2.pkl"


@configclass
class PickSingleEgadR093Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/R09_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/R09_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R09_3_v2.pkl"


@configclass
class PickSingleEgadR100Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/R10_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/R10_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R10_0_v2.pkl"


@configclass
class PickSingleEgadR101Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/R10_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/R10_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R10_1_v2.pkl"


@configclass
class PickSingleEgadR103Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/R10_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/R10_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R10_3_v2.pkl"


@configclass
class PickSingleEgadR110Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/R11_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/R11_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R11_0_v2.pkl"


@configclass
class PickSingleEgadR111Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/R11_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/R11_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R11_1_v2.pkl"


@configclass
class PickSingleEgadR112Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/R11_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/R11_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R11_2_v2.pkl"


@configclass
class PickSingleEgadR113Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/R11_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/R11_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R11_3_v2.pkl"


@configclass
class PickSingleEgadR121Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/R12_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/R12_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R12_1_v2.pkl"


@configclass
class PickSingleEgadR122Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/R12_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/R12_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R12_2_v2.pkl"


@configclass
class PickSingleEgadR123Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/R12_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/R12_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R12_3_v2.pkl"


@configclass
class PickSingleEgadR130Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/R13_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/R13_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R13_0_v2.pkl"


@configclass
class PickSingleEgadR131Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/R13_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/R13_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R13_1_v2.pkl"


@configclass
class PickSingleEgadR133Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/R13_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/R13_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R13_3_v2.pkl"


@configclass
class PickSingleEgadR140Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/R14_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/R14_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R14_0_v2.pkl"


@configclass
class PickSingleEgadR141Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/R14_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/R14_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R14_1_v2.pkl"


@configclass
class PickSingleEgadR143Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/R14_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/R14_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R14_3_v2.pkl"


@configclass
class PickSingleEgadR150Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/R15_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/R15_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R15_0_v2.pkl"


@configclass
class PickSingleEgadR151Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/R15_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/R15_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R15_1_v2.pkl"


@configclass
class PickSingleEgadR152Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/R15_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/R15_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R15_2_v2.pkl"


@configclass
class PickSingleEgadR153Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/R15_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/R15_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R15_3_v2.pkl"


@configclass
class PickSingleEgadR161Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/R16_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/R16_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R16_1_v2.pkl"


@configclass
class PickSingleEgadR162Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/R16_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/R16_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R16_2_v2.pkl"


@configclass
class PickSingleEgadR170Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/R17_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/R17_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R17_0_v2.pkl"


@configclass
class PickSingleEgadR171Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/R17_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/R17_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R17_1_v2.pkl"


@configclass
class PickSingleEgadR172Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/R17_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/R17_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R17_2_v2.pkl"


@configclass
class PickSingleEgadR173Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/R17_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/R17_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R17_3_v2.pkl"


@configclass
class PickSingleEgadR180Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/R18_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/R18_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R18_0_v2.pkl"


@configclass
class PickSingleEgadR181Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/R18_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/R18_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R18_1_v2.pkl"


@configclass
class PickSingleEgadR182Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/R18_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/R18_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R18_2_v2.pkl"


@configclass
class PickSingleEgadR191Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/R19_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/R19_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R19_1_v2.pkl"


@configclass
class PickSingleEgadR193Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/R19_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/R19_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R19_3_v2.pkl"


@configclass
class PickSingleEgadR200Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/R20_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/R20_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R20_0_v2.pkl"


@configclass
class PickSingleEgadR201Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/R20_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/R20_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R20_1_v2.pkl"


@configclass
class PickSingleEgadR202Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/R20_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/R20_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R20_2_v2.pkl"


@configclass
class PickSingleEgadR203Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/R20_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/R20_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R20_3_v2.pkl"


@configclass
class PickSingleEgadR210Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/R21_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/R21_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R21_0_v2.pkl"


@configclass
class PickSingleEgadR211Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/R21_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/R21_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R21_1_v2.pkl"


@configclass
class PickSingleEgadR212Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/R21_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/R21_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R21_2_v2.pkl"


@configclass
class PickSingleEgadR213Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/R21_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/R21_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R21_3_v2.pkl"


@configclass
class PickSingleEgadR220Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/R22_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/R22_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R22_0_v2.pkl"


@configclass
class PickSingleEgadR221Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/R22_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/R22_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R22_1_v2.pkl"


@configclass
class PickSingleEgadR222Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/R22_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/R22_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R22_2_v2.pkl"


@configclass
class PickSingleEgadR223Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/R22_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/R22_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R22_3_v2.pkl"


@configclass
class PickSingleEgadR230Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/R23_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/R23_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R23_0_v2.pkl"


@configclass
class PickSingleEgadR231Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/R23_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/R23_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R23_1_v2.pkl"


@configclass
class PickSingleEgadR232Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/R23_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/R23_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R23_2_v2.pkl"


@configclass
class PickSingleEgadR233Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/R23_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/R23_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R23_3_v2.pkl"


@configclass
class PickSingleEgadR240Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/R24_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/R24_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R24_0_v2.pkl"


@configclass
class PickSingleEgadR241Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/R24_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/R24_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R24_1_v2.pkl"


@configclass
class PickSingleEgadR242Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/R24_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/R24_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R24_2_v2.pkl"


@configclass
class PickSingleEgadR243Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/R24_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/R24_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R24_3_v2.pkl"


@configclass
class PickSingleEgadR250Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/R25_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/R25_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R25_0_v2.pkl"


@configclass
class PickSingleEgadR251Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/R25_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/R25_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R25_1_v2.pkl"


@configclass
class PickSingleEgadR252Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/R25_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/R25_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R25_2_v2.pkl"


@configclass
class PickSingleEgadR253Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/R25_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/R25_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-R25_3_v2.pkl"


@configclass
class PickSingleEgadS040Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/S04_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/S04_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S04_0_v2.pkl"


@configclass
class PickSingleEgadS041Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/S04_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/S04_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S04_1_v2.pkl"


@configclass
class PickSingleEgadS043Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/S04_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/S04_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S04_3_v2.pkl"


@configclass
class PickSingleEgadS051Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/S05_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/S05_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S05_1_v2.pkl"


@configclass
class PickSingleEgadS052Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/S05_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/S05_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S05_2_v2.pkl"


@configclass
class PickSingleEgadS053Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/S05_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/S05_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S05_3_v2.pkl"


@configclass
class PickSingleEgadS060Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/S06_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/S06_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S06_0_v2.pkl"


@configclass
class PickSingleEgadS061Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/S06_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/S06_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S06_1_v2.pkl"


@configclass
class PickSingleEgadS062Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/S06_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/S06_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S06_2_v2.pkl"


@configclass
class PickSingleEgadS070Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/S07_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/S07_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S07_0_v2.pkl"


@configclass
class PickSingleEgadS071Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/S07_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/S07_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S07_1_v2.pkl"


@configclass
class PickSingleEgadS072Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/S07_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/S07_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S07_2_v2.pkl"


@configclass
class PickSingleEgadS080Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/S08_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/S08_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S08_0_v2.pkl"


@configclass
class PickSingleEgadS081Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/S08_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/S08_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S08_1_v2.pkl"


@configclass
class PickSingleEgadS082Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/S08_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/S08_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S08_2_v2.pkl"


@configclass
class PickSingleEgadS091Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/S09_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/S09_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S09_1_v2.pkl"


@configclass
class PickSingleEgadS092Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/S09_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/S09_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S09_2_v2.pkl"


@configclass
class PickSingleEgadS101Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/S10_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/S10_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S10_1_v2.pkl"


@configclass
class PickSingleEgadS102Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/S10_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/S10_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S10_2_v2.pkl"


@configclass
class PickSingleEgadS103Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/S10_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/S10_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S10_3_v2.pkl"


@configclass
class PickSingleEgadS110Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/S11_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/S11_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S11_0_v2.pkl"


@configclass
class PickSingleEgadS111Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/S11_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/S11_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S11_1_v2.pkl"


@configclass
class PickSingleEgadS113Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/S11_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/S11_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S11_3_v2.pkl"


@configclass
class PickSingleEgadS120Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/S12_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/S12_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S12_0_v2.pkl"


@configclass
class PickSingleEgadS121Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/S12_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/S12_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S12_1_v2.pkl"


@configclass
class PickSingleEgadS122Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/S12_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/S12_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S12_2_v2.pkl"


@configclass
class PickSingleEgadS123Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/S12_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/S12_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S12_3_v2.pkl"


@configclass
class PickSingleEgadS130Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/S13_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/S13_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S13_0_v2.pkl"


@configclass
class PickSingleEgadS131Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/S13_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/S13_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S13_1_v2.pkl"


@configclass
class PickSingleEgadS132Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/S13_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/S13_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S13_2_v2.pkl"


@configclass
class PickSingleEgadS133Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/S13_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/S13_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S13_3_v2.pkl"


@configclass
class PickSingleEgadS140Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/S14_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/S14_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S14_0_v2.pkl"


@configclass
class PickSingleEgadS141Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/S14_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/S14_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S14_1_v2.pkl"


@configclass
class PickSingleEgadS142Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/S14_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/S14_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S14_2_v2.pkl"


@configclass
class PickSingleEgadS150Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/S15_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/S15_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S15_0_v2.pkl"


@configclass
class PickSingleEgadS151Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/S15_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/S15_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S15_1_v2.pkl"


@configclass
class PickSingleEgadS153Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/S15_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/S15_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S15_3_v2.pkl"


@configclass
class PickSingleEgadS160Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/S16_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/S16_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S16_0_v2.pkl"


@configclass
class PickSingleEgadS161Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/S16_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/S16_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S16_1_v2.pkl"


@configclass
class PickSingleEgadS163Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/S16_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/S16_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S16_3_v2.pkl"


@configclass
class PickSingleEgadS170Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/S17_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/S17_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S17_0_v2.pkl"


@configclass
class PickSingleEgadS171Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/S17_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/S17_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S17_1_v2.pkl"


@configclass
class PickSingleEgadS172Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/S17_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/S17_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S17_2_v2.pkl"


@configclass
class PickSingleEgadS180Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/S18_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/S18_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S18_0_v2.pkl"


@configclass
class PickSingleEgadS181Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/S18_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/S18_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S18_1_v2.pkl"


@configclass
class PickSingleEgadS183Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/S18_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/S18_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S18_3_v2.pkl"


@configclass
class PickSingleEgadS190Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/S19_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/S19_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S19_0_v2.pkl"


@configclass
class PickSingleEgadS191Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/S19_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/S19_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S19_1_v2.pkl"


@configclass
class PickSingleEgadS192Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/S19_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/S19_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S19_2_v2.pkl"


@configclass
class PickSingleEgadS193Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/S19_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/S19_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S19_3_v2.pkl"


@configclass
class PickSingleEgadS200Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/S20_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/S20_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S20_0_v2.pkl"


@configclass
class PickSingleEgadS201Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/S20_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/S20_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S20_1_v2.pkl"


@configclass
class PickSingleEgadS202Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/S20_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/S20_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S20_2_v2.pkl"


@configclass
class PickSingleEgadS203Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/S20_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/S20_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S20_3_v2.pkl"


@configclass
class PickSingleEgadS210Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/S21_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/S21_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S21_0_v2.pkl"


@configclass
class PickSingleEgadS211Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/S21_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/S21_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S21_1_v2.pkl"


@configclass
class PickSingleEgadS212Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/S21_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/S21_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S21_2_v2.pkl"


@configclass
class PickSingleEgadS213Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/S21_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/S21_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S21_3_v2.pkl"


@configclass
class PickSingleEgadS220Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/S22_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/S22_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S22_0_v2.pkl"


@configclass
class PickSingleEgadS221Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/S22_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/S22_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S22_1_v2.pkl"


@configclass
class PickSingleEgadS222Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/S22_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/S22_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S22_2_v2.pkl"


@configclass
class PickSingleEgadS223Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/S22_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/S22_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S22_3_v2.pkl"


@configclass
class PickSingleEgadS230Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/S23_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/S23_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S23_0_v2.pkl"


@configclass
class PickSingleEgadS231Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/S23_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/S23_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S23_1_v2.pkl"


@configclass
class PickSingleEgadS232Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/S23_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/S23_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S23_2_v2.pkl"


@configclass
class PickSingleEgadS233Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/S23_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/S23_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S23_3_v2.pkl"


@configclass
class PickSingleEgadS240Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/S24_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/S24_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S24_0_v2.pkl"


@configclass
class PickSingleEgadS241Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/S24_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/S24_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S24_1_v2.pkl"


@configclass
class PickSingleEgadS242Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/S24_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/S24_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S24_2_v2.pkl"


@configclass
class PickSingleEgadS243Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/S24_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/S24_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S24_3_v2.pkl"


@configclass
class PickSingleEgadS250Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/S25_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/S25_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S25_0_v2.pkl"


@configclass
class PickSingleEgadS251Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/S25_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/S25_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S25_1_v2.pkl"


@configclass
class PickSingleEgadS252Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/S25_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/S25_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S25_2_v2.pkl"


@configclass
class PickSingleEgadS253Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/S25_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/S25_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-S25_3_v2.pkl"


@configclass
class PickSingleEgadT041Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/T04_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/T04_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T04_1_v2.pkl"


@configclass
class PickSingleEgadT043Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/T04_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/T04_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T04_3_v2.pkl"


@configclass
class PickSingleEgadT050Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/T05_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/T05_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T05_0_v2.pkl"


@configclass
class PickSingleEgadT051Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/T05_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/T05_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T05_1_v2.pkl"


@configclass
class PickSingleEgadT052Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/T05_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/T05_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T05_2_v2.pkl"


@configclass
class PickSingleEgadT053Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/T05_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/T05_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T05_3_v2.pkl"


@configclass
class PickSingleEgadT060Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/T06_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/T06_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T06_0_v2.pkl"


@configclass
class PickSingleEgadT061Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/T06_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/T06_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T06_1_v2.pkl"


@configclass
class PickSingleEgadT062Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/T06_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/T06_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T06_2_v2.pkl"


@configclass
class PickSingleEgadT063Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/T06_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/T06_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T06_3_v2.pkl"


@configclass
class PickSingleEgadT070Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/T07_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/T07_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T07_0_v2.pkl"


@configclass
class PickSingleEgadT071Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/T07_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/T07_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T07_1_v2.pkl"


@configclass
class PickSingleEgadT072Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/T07_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/T07_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T07_2_v2.pkl"


@configclass
class PickSingleEgadT073Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/T07_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/T07_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T07_3_v2.pkl"


@configclass
class PickSingleEgadT080Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/T08_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/T08_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T08_0_v2.pkl"


@configclass
class PickSingleEgadT081Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/T08_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/T08_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T08_1_v2.pkl"


@configclass
class PickSingleEgadT082Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/T08_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/T08_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T08_2_v2.pkl"


@configclass
class PickSingleEgadT083Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/T08_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/T08_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T08_3_v2.pkl"


@configclass
class PickSingleEgadT090Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/T09_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/T09_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T09_0_v2.pkl"


@configclass
class PickSingleEgadT091Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/T09_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/T09_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T09_1_v2.pkl"


@configclass
class PickSingleEgadT100Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/T10_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/T10_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T10_0_v2.pkl"


@configclass
class PickSingleEgadT102Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/T10_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/T10_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T10_2_v2.pkl"


@configclass
class PickSingleEgadT103Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/T10_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/T10_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T10_3_v2.pkl"


@configclass
class PickSingleEgadT110Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/T11_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/T11_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T11_0_v2.pkl"


@configclass
class PickSingleEgadT111Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/T11_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/T11_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T11_1_v2.pkl"


@configclass
class PickSingleEgadT112Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/T11_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/T11_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T11_2_v2.pkl"


@configclass
class PickSingleEgadT120Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/T12_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/T12_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T12_0_v2.pkl"


@configclass
class PickSingleEgadT121Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/T12_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/T12_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T12_1_v2.pkl"


@configclass
class PickSingleEgadT122Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/T12_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/T12_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T12_2_v2.pkl"


@configclass
class PickSingleEgadT123Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/T12_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/T12_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T12_3_v2.pkl"


@configclass
class PickSingleEgadT130Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/T13_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/T13_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T13_0_v2.pkl"


@configclass
class PickSingleEgadT131Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/T13_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/T13_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T13_1_v2.pkl"


@configclass
class PickSingleEgadT132Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/T13_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/T13_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T13_2_v2.pkl"


@configclass
class PickSingleEgadT140Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/T14_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/T14_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T14_0_v2.pkl"


@configclass
class PickSingleEgadT141Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/T14_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/T14_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T14_1_v2.pkl"


@configclass
class PickSingleEgadT143Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/T14_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/T14_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T14_3_v2.pkl"


@configclass
class PickSingleEgadT151Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/T15_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/T15_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T15_1_v2.pkl"


@configclass
class PickSingleEgadT152Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/T15_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/T15_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T15_2_v2.pkl"


@configclass
class PickSingleEgadT153Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/T15_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/T15_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T15_3_v2.pkl"


@configclass
class PickSingleEgadT160Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/T16_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/T16_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T16_0_v2.pkl"


@configclass
class PickSingleEgadT161Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/T16_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/T16_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T16_1_v2.pkl"


@configclass
class PickSingleEgadT163Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/T16_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/T16_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T16_3_v2.pkl"


@configclass
class PickSingleEgadT171Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/T17_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/T17_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T17_1_v2.pkl"


@configclass
class PickSingleEgadT172Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/T17_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/T17_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T17_2_v2.pkl"


@configclass
class PickSingleEgadT173Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/T17_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/T17_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T17_3_v2.pkl"


@configclass
class PickSingleEgadT180Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/T18_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/T18_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T18_0_v2.pkl"


@configclass
class PickSingleEgadT181Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/T18_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/T18_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T18_1_v2.pkl"


@configclass
class PickSingleEgadT182Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/T18_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/T18_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T18_2_v2.pkl"


@configclass
class PickSingleEgadT183Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/T18_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/T18_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T18_3_v2.pkl"


@configclass
class PickSingleEgadT190Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/T19_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/T19_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T19_0_v2.pkl"


@configclass
class PickSingleEgadT191Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/T19_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/T19_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T19_1_v2.pkl"


@configclass
class PickSingleEgadT192Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/T19_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/T19_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T19_2_v2.pkl"


@configclass
class PickSingleEgadT193Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/T19_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/T19_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T19_3_v2.pkl"


@configclass
class PickSingleEgadT200Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/T20_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/T20_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T20_0_v2.pkl"


@configclass
class PickSingleEgadT201Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/T20_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/T20_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T20_1_v2.pkl"


@configclass
class PickSingleEgadT202Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/T20_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/T20_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T20_2_v2.pkl"


@configclass
class PickSingleEgadT203Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/T20_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/T20_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T20_3_v2.pkl"


@configclass
class PickSingleEgadT210Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/T21_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/T21_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T21_0_v2.pkl"


@configclass
class PickSingleEgadT211Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/T21_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/T21_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T21_1_v2.pkl"


@configclass
class PickSingleEgadT212Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/T21_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/T21_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T21_2_v2.pkl"


@configclass
class PickSingleEgadT213Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/T21_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/T21_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T21_3_v2.pkl"


@configclass
class PickSingleEgadT220Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/T22_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/T22_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T22_0_v2.pkl"


@configclass
class PickSingleEgadT221Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/T22_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/T22_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T22_1_v2.pkl"


@configclass
class PickSingleEgadT222Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/T22_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/T22_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T22_2_v2.pkl"


@configclass
class PickSingleEgadT223Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/T22_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/T22_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T22_3_v2.pkl"


@configclass
class PickSingleEgadT230Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/T23_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/T23_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T23_0_v2.pkl"


@configclass
class PickSingleEgadT231Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/T23_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/T23_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T23_1_v2.pkl"


@configclass
class PickSingleEgadT232Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/T23_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/T23_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T23_2_v2.pkl"


@configclass
class PickSingleEgadT233Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/T23_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/T23_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T23_3_v2.pkl"


@configclass
class PickSingleEgadT240Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/T24_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/T24_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T24_0_v2.pkl"


@configclass
class PickSingleEgadT241Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/T24_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/T24_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T24_1_v2.pkl"


@configclass
class PickSingleEgadT242Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/T24_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/T24_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T24_2_v2.pkl"


@configclass
class PickSingleEgadT243Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/T24_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/T24_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T24_3_v2.pkl"


@configclass
class PickSingleEgadT251Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/T25_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/T25_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T25_1_v2.pkl"


@configclass
class PickSingleEgadT252Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/T25_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/T25_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T25_2_v2.pkl"


@configclass
class PickSingleEgadT253Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/T25_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/T25_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-T25_3_v2.pkl"


@configclass
class PickSingleEgadU020Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/U02_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/U02_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U02_0_v2.pkl"


@configclass
class PickSingleEgadU023Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/U02_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/U02_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U02_3_v2.pkl"


@configclass
class PickSingleEgadU030Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/U03_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/U03_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U03_0_v2.pkl"


@configclass
class PickSingleEgadU031Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/U03_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/U03_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U03_1_v2.pkl"


@configclass
class PickSingleEgadU033Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/U03_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/U03_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U03_3_v2.pkl"


@configclass
class PickSingleEgadU040Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/U04_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/U04_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U04_0_v2.pkl"


@configclass
class PickSingleEgadU043Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/U04_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/U04_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U04_3_v2.pkl"


@configclass
class PickSingleEgadU050Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/U05_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/U05_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U05_0_v2.pkl"


@configclass
class PickSingleEgadU052Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/U05_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/U05_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U05_2_v2.pkl"


@configclass
class PickSingleEgadU060Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/U06_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/U06_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U06_0_v2.pkl"


@configclass
class PickSingleEgadU061Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/U06_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/U06_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U06_1_v2.pkl"


@configclass
class PickSingleEgadU063Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/U06_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/U06_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U06_3_v2.pkl"


@configclass
class PickSingleEgadU070Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/U07_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/U07_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U07_0_v2.pkl"


@configclass
class PickSingleEgadU081Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/U08_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/U08_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U08_1_v2.pkl"


@configclass
class PickSingleEgadU083Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/U08_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/U08_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U08_3_v2.pkl"


@configclass
class PickSingleEgadU090Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/U09_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/U09_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U09_0_v2.pkl"


@configclass
class PickSingleEgadU093Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/U09_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/U09_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U09_3_v2.pkl"


@configclass
class PickSingleEgadU100Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/U10_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/U10_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U10_0_v2.pkl"


@configclass
class PickSingleEgadU101Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/U10_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/U10_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U10_1_v2.pkl"


@configclass
class PickSingleEgadU103Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/U10_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/U10_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U10_3_v2.pkl"


@configclass
class PickSingleEgadU111Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/U11_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/U11_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U11_1_v2.pkl"


@configclass
class PickSingleEgadU112Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/U11_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/U11_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U11_2_v2.pkl"


@configclass
class PickSingleEgadU113Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/U11_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/U11_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U11_3_v2.pkl"


@configclass
class PickSingleEgadU122Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/U12_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/U12_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U12_2_v2.pkl"


@configclass
class PickSingleEgadU123Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/U12_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/U12_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U12_3_v2.pkl"


@configclass
class PickSingleEgadU130Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/U13_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/U13_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U13_0_v2.pkl"


@configclass
class PickSingleEgadU131Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/U13_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/U13_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U13_1_v2.pkl"


@configclass
class PickSingleEgadU132Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/U13_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/U13_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U13_2_v2.pkl"


@configclass
class PickSingleEgadU140Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/U14_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/U14_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U14_0_v2.pkl"


@configclass
class PickSingleEgadU141Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/U14_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/U14_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U14_1_v2.pkl"


@configclass
class PickSingleEgadU143Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/U14_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/U14_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U14_3_v2.pkl"


@configclass
class PickSingleEgadU150Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/U15_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/U15_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U15_0_v2.pkl"


@configclass
class PickSingleEgadU151Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/U15_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/U15_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U15_1_v2.pkl"


@configclass
class PickSingleEgadU152Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/U15_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/U15_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U15_2_v2.pkl"


@configclass
class PickSingleEgadU160Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/U16_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/U16_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U16_0_v2.pkl"


@configclass
class PickSingleEgadU161Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/U16_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/U16_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U16_1_v2.pkl"


@configclass
class PickSingleEgadU162Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/U16_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/U16_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U16_2_v2.pkl"


@configclass
class PickSingleEgadU163Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/U16_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/U16_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U16_3_v2.pkl"


@configclass
class PickSingleEgadU170Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/U17_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/U17_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U17_0_v2.pkl"


@configclass
class PickSingleEgadU171Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/U17_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/U17_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U17_1_v2.pkl"


@configclass
class PickSingleEgadU172Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/U17_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/U17_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U17_2_v2.pkl"


@configclass
class PickSingleEgadU173Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/U17_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/U17_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U17_3_v2.pkl"


@configclass
class PickSingleEgadU181Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/U18_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/U18_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U18_1_v2.pkl"


@configclass
class PickSingleEgadU182Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/U18_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/U18_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U18_2_v2.pkl"


@configclass
class PickSingleEgadU183Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/U18_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/U18_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U18_3_v2.pkl"


@configclass
class PickSingleEgadU190Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/U19_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/U19_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U19_0_v2.pkl"


@configclass
class PickSingleEgadU191Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/U19_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/U19_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U19_1_v2.pkl"


@configclass
class PickSingleEgadU192Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/U19_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/U19_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U19_2_v2.pkl"


@configclass
class PickSingleEgadU193Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/U19_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/U19_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U19_3_v2.pkl"


@configclass
class PickSingleEgadU200Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/U20_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/U20_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U20_0_v2.pkl"


@configclass
class PickSingleEgadU201Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/U20_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/U20_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U20_1_v2.pkl"


@configclass
class PickSingleEgadU202Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/U20_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/U20_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U20_2_v2.pkl"


@configclass
class PickSingleEgadU203Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/U20_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/U20_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U20_3_v2.pkl"


@configclass
class PickSingleEgadU210Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/U21_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/U21_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U21_0_v2.pkl"


@configclass
class PickSingleEgadU211Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/U21_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/U21_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U21_1_v2.pkl"


@configclass
class PickSingleEgadU212Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/U21_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/U21_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U21_2_v2.pkl"


@configclass
class PickSingleEgadU213Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/U21_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/U21_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U21_3_v2.pkl"


@configclass
class PickSingleEgadU221Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/U22_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/U22_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U22_1_v2.pkl"


@configclass
class PickSingleEgadU222Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/U22_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/U22_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U22_2_v2.pkl"


@configclass
class PickSingleEgadU230Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/U23_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/U23_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U23_0_v2.pkl"


@configclass
class PickSingleEgadU231Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/U23_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/U23_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U23_1_v2.pkl"


@configclass
class PickSingleEgadU233Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/U23_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/U23_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U23_3_v2.pkl"


@configclass
class PickSingleEgadU241Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/U24_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/U24_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U24_1_v2.pkl"


@configclass
class PickSingleEgadU242Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/U24_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/U24_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U24_2_v2.pkl"


@configclass
class PickSingleEgadU243Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/U24_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/U24_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U24_3_v2.pkl"


@configclass
class PickSingleEgadU250Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/U25_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/U25_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U25_0_v2.pkl"


@configclass
class PickSingleEgadU251Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/U25_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/U25_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U25_1_v2.pkl"


@configclass
class PickSingleEgadU252Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/U25_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/U25_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-U25_2_v2.pkl"


@configclass
class PickSingleEgadV020Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/V02_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/V02_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V02_0_v2.pkl"


@configclass
class PickSingleEgadV021Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/V02_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/V02_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V02_1_v2.pkl"


@configclass
class PickSingleEgadV022Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/V02_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/V02_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V02_2_v2.pkl"


@configclass
class PickSingleEgadV023Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/V02_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/V02_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V02_3_v2.pkl"


@configclass
class PickSingleEgadV031Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/V03_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/V03_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V03_1_v2.pkl"


@configclass
class PickSingleEgadV033Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/V03_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/V03_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V03_3_v2.pkl"


@configclass
class PickSingleEgadV041Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/V04_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/V04_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V04_1_v2.pkl"


@configclass
class PickSingleEgadV042Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/V04_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/V04_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V04_2_v2.pkl"


@configclass
class PickSingleEgadV050Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/V05_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/V05_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V05_0_v2.pkl"


@configclass
class PickSingleEgadV052Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/V05_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/V05_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V05_2_v2.pkl"


@configclass
class PickSingleEgadV053Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/V05_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/V05_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V05_3_v2.pkl"


@configclass
class PickSingleEgadV060Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/V06_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/V06_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V06_0_v2.pkl"


@configclass
class PickSingleEgadV061Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/V06_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/V06_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V06_1_v2.pkl"


@configclass
class PickSingleEgadV063Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/V06_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/V06_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V06_3_v2.pkl"


@configclass
class PickSingleEgadV070Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/V07_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/V07_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V07_0_v2.pkl"


@configclass
class PickSingleEgadV072Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/V07_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/V07_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V07_2_v2.pkl"


@configclass
class PickSingleEgadV073Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/V07_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/V07_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V07_3_v2.pkl"


@configclass
class PickSingleEgadV080Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/V08_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/V08_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V08_0_v2.pkl"


@configclass
class PickSingleEgadV082Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/V08_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/V08_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V08_2_v2.pkl"


@configclass
class PickSingleEgadV092Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/V09_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/V09_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V09_2_v2.pkl"


@configclass
class PickSingleEgadV100Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/V10_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/V10_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V10_0_v2.pkl"


@configclass
class PickSingleEgadV101Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/V10_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/V10_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V10_1_v2.pkl"


@configclass
class PickSingleEgadV102Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/V10_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/V10_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V10_2_v2.pkl"


@configclass
class PickSingleEgadV103Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/V10_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/V10_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V10_3_v2.pkl"


@configclass
class PickSingleEgadV110Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/V11_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/V11_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V11_0_v2.pkl"


@configclass
class PickSingleEgadV111Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/V11_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/V11_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V11_1_v2.pkl"


@configclass
class PickSingleEgadV112Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/V11_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/V11_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V11_2_v2.pkl"


@configclass
class PickSingleEgadV113Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/V11_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/V11_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V11_3_v2.pkl"


@configclass
class PickSingleEgadV121Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/V12_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/V12_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V12_1_v2.pkl"


@configclass
class PickSingleEgadV122Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/V12_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/V12_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V12_2_v2.pkl"


@configclass
class PickSingleEgadV123Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/V12_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/V12_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V12_3_v2.pkl"


@configclass
class PickSingleEgadV130Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/V13_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/V13_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V13_0_v2.pkl"


@configclass
class PickSingleEgadV131Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/V13_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/V13_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V13_1_v2.pkl"


@configclass
class PickSingleEgadV132Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/V13_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/V13_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V13_2_v2.pkl"


@configclass
class PickSingleEgadV133Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/V13_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/V13_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V13_3_v2.pkl"


@configclass
class PickSingleEgadV140Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/V14_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/V14_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V14_0_v2.pkl"


@configclass
class PickSingleEgadV141Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/V14_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/V14_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V14_1_v2.pkl"


@configclass
class PickSingleEgadV150Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/V15_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/V15_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V15_0_v2.pkl"


@configclass
class PickSingleEgadV151Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/V15_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/V15_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V15_1_v2.pkl"


@configclass
class PickSingleEgadV153Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/V15_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/V15_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V15_3_v2.pkl"


@configclass
class PickSingleEgadV160Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/V16_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/V16_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V16_0_v2.pkl"


@configclass
class PickSingleEgadV162Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/V16_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/V16_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V16_2_v2.pkl"


@configclass
class PickSingleEgadV171Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/V17_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/V17_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V17_1_v2.pkl"


@configclass
class PickSingleEgadV172Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/V17_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/V17_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V17_2_v2.pkl"


@configclass
class PickSingleEgadV173Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/V17_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/V17_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V17_3_v2.pkl"


@configclass
class PickSingleEgadV181Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/V18_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/V18_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V18_1_v2.pkl"


@configclass
class PickSingleEgadV182Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/V18_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/V18_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V18_2_v2.pkl"


@configclass
class PickSingleEgadV190Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/V19_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/V19_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V19_0_v2.pkl"


@configclass
class PickSingleEgadV191Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/V19_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/V19_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V19_1_v2.pkl"


@configclass
class PickSingleEgadV192Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/V19_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/V19_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V19_2_v2.pkl"


@configclass
class PickSingleEgadV193Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/V19_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/V19_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V19_3_v2.pkl"


@configclass
class PickSingleEgadV200Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/V20_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/V20_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V20_0_v2.pkl"


@configclass
class PickSingleEgadV201Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/V20_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/V20_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V20_1_v2.pkl"


@configclass
class PickSingleEgadV202Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/V20_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/V20_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V20_2_v2.pkl"


@configclass
class PickSingleEgadV203Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/V20_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/V20_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V20_3_v2.pkl"


@configclass
class PickSingleEgadV210Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/V21_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/V21_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V21_0_v2.pkl"


@configclass
class PickSingleEgadV211Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/V21_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/V21_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V21_1_v2.pkl"


@configclass
class PickSingleEgadV213Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/V21_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/V21_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V21_3_v2.pkl"


@configclass
class PickSingleEgadV221Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/V22_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/V22_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V22_1_v2.pkl"


@configclass
class PickSingleEgadV222Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/V22_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/V22_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V22_2_v2.pkl"


@configclass
class PickSingleEgadV223Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/V22_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/V22_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V22_3_v2.pkl"


@configclass
class PickSingleEgadV230Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/V23_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/V23_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V23_0_v2.pkl"


@configclass
class PickSingleEgadV232Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/V23_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/V23_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V23_2_v2.pkl"


@configclass
class PickSingleEgadV233Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/V23_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/V23_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V23_3_v2.pkl"


@configclass
class PickSingleEgadV240Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/V24_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/V24_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V24_0_v2.pkl"


@configclass
class PickSingleEgadV241Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/V24_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/V24_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V24_1_v2.pkl"


@configclass
class PickSingleEgadV242Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/V24_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/V24_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V24_2_v2.pkl"


@configclass
class PickSingleEgadV250Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/V25_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/V25_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V25_0_v2.pkl"


@configclass
class PickSingleEgadV251Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/V25_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/V25_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V25_1_v2.pkl"


@configclass
class PickSingleEgadV252Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/V25_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/V25_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V25_2_v2.pkl"


@configclass
class PickSingleEgadV253Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/V25_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/V25_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-V25_3_v2.pkl"


@configclass
class PickSingleEgadW020Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/W02_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/W02_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W02_0_v2.pkl"


@configclass
class PickSingleEgadW030Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/W03_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/W03_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W03_0_v2.pkl"


@configclass
class PickSingleEgadW033Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/W03_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/W03_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W03_3_v2.pkl"


@configclass
class PickSingleEgadW040Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/W04_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/W04_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W04_0_v2.pkl"


@configclass
class PickSingleEgadW041Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/W04_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/W04_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W04_1_v2.pkl"


@configclass
class PickSingleEgadW042Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/W04_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/W04_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W04_2_v2.pkl"


@configclass
class PickSingleEgadW050Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/W05_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/W05_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W05_0_v2.pkl"


@configclass
class PickSingleEgadW052Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/W05_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/W05_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W05_2_v2.pkl"


@configclass
class PickSingleEgadW053Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/W05_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/W05_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W05_3_v2.pkl"


@configclass
class PickSingleEgadW060Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/W06_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/W06_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W06_0_v2.pkl"


@configclass
class PickSingleEgadW062Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/W06_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/W06_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W06_2_v2.pkl"


@configclass
class PickSingleEgadW063Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/W06_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/W06_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W06_3_v2.pkl"


@configclass
class PickSingleEgadW070Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/W07_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/W07_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W07_0_v2.pkl"


@configclass
class PickSingleEgadW071Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/W07_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/W07_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W07_1_v2.pkl"


@configclass
class PickSingleEgadW073Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/W07_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/W07_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W07_3_v2.pkl"


@configclass
class PickSingleEgadW081Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/W08_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/W08_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W08_1_v2.pkl"


@configclass
class PickSingleEgadW082Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/W08_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/W08_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W08_2_v2.pkl"


@configclass
class PickSingleEgadW083Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/W08_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/W08_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W08_3_v2.pkl"


@configclass
class PickSingleEgadW090Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/W09_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/W09_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W09_0_v2.pkl"


@configclass
class PickSingleEgadW091Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/W09_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/W09_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W09_1_v2.pkl"


@configclass
class PickSingleEgadW092Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/W09_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/W09_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W09_2_v2.pkl"


@configclass
class PickSingleEgadW093Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/W09_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/W09_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W09_3_v2.pkl"


@configclass
class PickSingleEgadW101Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/W10_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/W10_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W10_1_v2.pkl"


@configclass
class PickSingleEgadW110Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/W11_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/W11_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W11_0_v2.pkl"


@configclass
class PickSingleEgadW111Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/W11_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/W11_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W11_1_v2.pkl"


@configclass
class PickSingleEgadW113Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/W11_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/W11_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W11_3_v2.pkl"


@configclass
class PickSingleEgadW121Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/W12_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/W12_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W12_1_v2.pkl"


@configclass
class PickSingleEgadW122Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/W12_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/W12_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W12_2_v2.pkl"


@configclass
class PickSingleEgadW123Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/W12_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/W12_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W12_3_v2.pkl"


@configclass
class PickSingleEgadW130Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/W13_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/W13_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W13_0_v2.pkl"


@configclass
class PickSingleEgadW131Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/W13_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/W13_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W13_1_v2.pkl"


@configclass
class PickSingleEgadW132Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/W13_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/W13_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W13_2_v2.pkl"


@configclass
class PickSingleEgadW133Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/W13_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/W13_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W13_3_v2.pkl"


@configclass
class PickSingleEgadW140Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/W14_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/W14_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W14_0_v2.pkl"


@configclass
class PickSingleEgadW143Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/W14_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/W14_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W14_3_v2.pkl"


@configclass
class PickSingleEgadW150Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/W15_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/W15_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W15_0_v2.pkl"


@configclass
class PickSingleEgadW151Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/W15_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/W15_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W15_1_v2.pkl"


@configclass
class PickSingleEgadW153Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/W15_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/W15_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W15_3_v2.pkl"


@configclass
class PickSingleEgadW160Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/W16_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/W16_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W16_0_v2.pkl"


@configclass
class PickSingleEgadW161Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/W16_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/W16_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W16_1_v2.pkl"


@configclass
class PickSingleEgadW162Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/W16_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/W16_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W16_2_v2.pkl"


@configclass
class PickSingleEgadW163Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/W16_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/W16_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W16_3_v2.pkl"


@configclass
class PickSingleEgadW170Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/W17_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/W17_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W17_0_v2.pkl"


@configclass
class PickSingleEgadW171Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/W17_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/W17_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W17_1_v2.pkl"


@configclass
class PickSingleEgadW172Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/W17_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/W17_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W17_2_v2.pkl"


@configclass
class PickSingleEgadW173Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/W17_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/W17_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W17_3_v2.pkl"


@configclass
class PickSingleEgadW180Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/W18_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/W18_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W18_0_v2.pkl"


@configclass
class PickSingleEgadW182Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/W18_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/W18_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W18_2_v2.pkl"


@configclass
class PickSingleEgadW183Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/W18_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/W18_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W18_3_v2.pkl"


@configclass
class PickSingleEgadW190Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/W19_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/W19_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W19_0_v2.pkl"


@configclass
class PickSingleEgadW192Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/W19_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/W19_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W19_2_v2.pkl"


@configclass
class PickSingleEgadW193Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/W19_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/W19_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W19_3_v2.pkl"


@configclass
class PickSingleEgadW200Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/W20_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/W20_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W20_0_v2.pkl"


@configclass
class PickSingleEgadW201Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/W20_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/W20_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W20_1_v2.pkl"


@configclass
class PickSingleEgadW202Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/W20_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/W20_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W20_2_v2.pkl"


@configclass
class PickSingleEgadW203Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/W20_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/W20_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W20_3_v2.pkl"


@configclass
class PickSingleEgadW210Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/W21_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/W21_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W21_0_v2.pkl"


@configclass
class PickSingleEgadW211Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/W21_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/W21_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W21_1_v2.pkl"


@configclass
class PickSingleEgadW220Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/W22_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/W22_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W22_0_v2.pkl"


@configclass
class PickSingleEgadW221Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/W22_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/W22_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W22_1_v2.pkl"


@configclass
class PickSingleEgadW223Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/W22_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/W22_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W22_3_v2.pkl"


@configclass
class PickSingleEgadW230Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/W23_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/W23_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W23_0_v2.pkl"


@configclass
class PickSingleEgadW231Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/W23_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/W23_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W23_1_v2.pkl"


@configclass
class PickSingleEgadW232Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/W23_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/W23_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W23_2_v2.pkl"


@configclass
class PickSingleEgadW233Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/W23_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/W23_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W23_3_v2.pkl"


@configclass
class PickSingleEgadW240Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/W24_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/W24_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W24_0_v2.pkl"


@configclass
class PickSingleEgadW241Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/W24_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/W24_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W24_1_v2.pkl"


@configclass
class PickSingleEgadW242Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/W24_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/W24_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W24_2_v2.pkl"


@configclass
class PickSingleEgadW243Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/W24_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/W24_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W24_3_v2.pkl"


@configclass
class PickSingleEgadW250Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/W25_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/W25_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W25_0_v2.pkl"


@configclass
class PickSingleEgadW251Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/W25_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/W25_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W25_1_v2.pkl"


@configclass
class PickSingleEgadW252Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/W25_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/W25_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W25_2_v2.pkl"


@configclass
class PickSingleEgadW253Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/W25_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/W25_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-W25_3_v2.pkl"


@configclass
class PickSingleEgadX000Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/X00_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/X00_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X00_0_v2.pkl"


@configclass
class PickSingleEgadX010Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/X01_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/X01_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X01_0_v2.pkl"


@configclass
class PickSingleEgadX011Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/X01_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/X01_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X01_1_v2.pkl"


@configclass
class PickSingleEgadX012Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/X01_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/X01_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X01_2_v2.pkl"


@configclass
class PickSingleEgadX013Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/X01_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/X01_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X01_3_v2.pkl"


@configclass
class PickSingleEgadX020Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/X02_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/X02_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X02_0_v2.pkl"


@configclass
class PickSingleEgadX021Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/X02_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/X02_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X02_1_v2.pkl"


@configclass
class PickSingleEgadX022Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/X02_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/X02_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X02_2_v2.pkl"


@configclass
class PickSingleEgadX023Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/X02_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/X02_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X02_3_v2.pkl"


@configclass
class PickSingleEgadX030Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/X03_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/X03_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X03_0_v2.pkl"


@configclass
class PickSingleEgadX032Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/X03_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/X03_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X03_2_v2.pkl"


@configclass
class PickSingleEgadX033Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/X03_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/X03_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X03_3_v2.pkl"


@configclass
class PickSingleEgadX040Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/X04_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/X04_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X04_0_v2.pkl"


@configclass
class PickSingleEgadX041Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/X04_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/X04_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X04_1_v2.pkl"


@configclass
class PickSingleEgadX042Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/X04_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/X04_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X04_2_v2.pkl"


@configclass
class PickSingleEgadX050Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/X05_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/X05_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X05_0_v2.pkl"


@configclass
class PickSingleEgadX052Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/X05_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/X05_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X05_2_v2.pkl"


@configclass
class PickSingleEgadX060Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/X06_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/X06_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X06_0_v2.pkl"


@configclass
class PickSingleEgadX063Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/X06_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/X06_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X06_3_v2.pkl"


@configclass
class PickSingleEgadX070Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/X07_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/X07_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X07_0_v2.pkl"


@configclass
class PickSingleEgadX071Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/X07_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/X07_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X07_1_v2.pkl"


@configclass
class PickSingleEgadX072Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/X07_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/X07_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X07_2_v2.pkl"


@configclass
class PickSingleEgadX081Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/X08_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/X08_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X08_1_v2.pkl"


@configclass
class PickSingleEgadX082Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/X08_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/X08_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X08_2_v2.pkl"


@configclass
class PickSingleEgadX090Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/X09_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/X09_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X09_0_v2.pkl"


@configclass
class PickSingleEgadX091Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/X09_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/X09_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X09_1_v2.pkl"


@configclass
class PickSingleEgadX092Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/X09_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/X09_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X09_2_v2.pkl"


@configclass
class PickSingleEgadX093Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/X09_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/X09_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X09_3_v2.pkl"


@configclass
class PickSingleEgadX100Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/X10_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/X10_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X10_0_v2.pkl"


@configclass
class PickSingleEgadX102Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/X10_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/X10_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X10_2_v2.pkl"


@configclass
class PickSingleEgadX103Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/X10_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/X10_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X10_3_v2.pkl"


@configclass
class PickSingleEgadX110Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/X11_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/X11_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X11_0_v2.pkl"


@configclass
class PickSingleEgadX111Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/X11_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/X11_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X11_1_v2.pkl"


@configclass
class PickSingleEgadX112Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/X11_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/X11_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X11_2_v2.pkl"


@configclass
class PickSingleEgadX113Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/X11_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/X11_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X11_3_v2.pkl"


@configclass
class PickSingleEgadX120Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/X12_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/X12_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X12_0_v2.pkl"


@configclass
class PickSingleEgadX121Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/X12_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/X12_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X12_1_v2.pkl"


@configclass
class PickSingleEgadX122Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/X12_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/X12_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X12_2_v2.pkl"


@configclass
class PickSingleEgadX123Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/X12_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/X12_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X12_3_v2.pkl"


@configclass
class PickSingleEgadX130Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/X13_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/X13_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X13_0_v2.pkl"


@configclass
class PickSingleEgadX131Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/X13_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/X13_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X13_1_v2.pkl"


@configclass
class PickSingleEgadX132Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/X13_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/X13_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X13_2_v2.pkl"


@configclass
class PickSingleEgadX140Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/X14_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/X14_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X14_0_v2.pkl"


@configclass
class PickSingleEgadX141Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/X14_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/X14_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X14_1_v2.pkl"


@configclass
class PickSingleEgadX142Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/X14_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/X14_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X14_2_v2.pkl"


@configclass
class PickSingleEgadX143Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/X14_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/X14_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X14_3_v2.pkl"


@configclass
class PickSingleEgadX150Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/X15_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/X15_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X15_0_v2.pkl"


@configclass
class PickSingleEgadX151Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/X15_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/X15_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X15_1_v2.pkl"


@configclass
class PickSingleEgadX152Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/X15_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/X15_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X15_2_v2.pkl"


@configclass
class PickSingleEgadX153Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/X15_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/X15_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X15_3_v2.pkl"


@configclass
class PickSingleEgadX161Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/X16_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/X16_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X16_1_v2.pkl"


@configclass
class PickSingleEgadX170Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/X17_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/X17_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X17_0_v2.pkl"


@configclass
class PickSingleEgadX171Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/X17_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/X17_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X17_1_v2.pkl"


@configclass
class PickSingleEgadX172Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/X17_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/X17_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X17_2_v2.pkl"


@configclass
class PickSingleEgadX173Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/X17_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/X17_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X17_3_v2.pkl"


@configclass
class PickSingleEgadX181Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/X18_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/X18_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X18_1_v2.pkl"


@configclass
class PickSingleEgadX182Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/X18_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/X18_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X18_2_v2.pkl"


@configclass
class PickSingleEgadX190Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/X19_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/X19_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X19_0_v2.pkl"


@configclass
class PickSingleEgadX191Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/X19_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/X19_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X19_1_v2.pkl"


@configclass
class PickSingleEgadX192Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/X19_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/X19_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X19_2_v2.pkl"


@configclass
class PickSingleEgadX200Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/X20_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/X20_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X20_0_v2.pkl"


@configclass
class PickSingleEgadX201Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/X20_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/X20_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X20_1_v2.pkl"


@configclass
class PickSingleEgadX202Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/X20_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/X20_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X20_2_v2.pkl"


@configclass
class PickSingleEgadX210Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/X21_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/X21_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X21_0_v2.pkl"


@configclass
class PickSingleEgadX211Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/X21_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/X21_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X21_1_v2.pkl"


@configclass
class PickSingleEgadX212Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/X21_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/X21_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X21_2_v2.pkl"


@configclass
class PickSingleEgadX213Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/X21_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/X21_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X21_3_v2.pkl"


@configclass
class PickSingleEgadX220Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/X22_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/X22_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X22_0_v2.pkl"


@configclass
class PickSingleEgadX222Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/X22_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/X22_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X22_2_v2.pkl"


@configclass
class PickSingleEgadX223Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/X22_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/X22_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X22_3_v2.pkl"


@configclass
class PickSingleEgadX231Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/X23_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/X23_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X23_1_v2.pkl"


@configclass
class PickSingleEgadX232Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/X23_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/X23_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X23_2_v2.pkl"


@configclass
class PickSingleEgadX233Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/X23_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/X23_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X23_3_v2.pkl"


@configclass
class PickSingleEgadX240Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/X24_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/X24_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X24_0_v2.pkl"


@configclass
class PickSingleEgadX241Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/X24_1_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/X24_1.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X24_1_v2.pkl"


@configclass
class PickSingleEgadX242Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/X24_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/X24_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X24_2_v2.pkl"


@configclass
class PickSingleEgadX243Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/X24_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/X24_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X24_3_v2.pkl"


@configclass
class PickSingleEgadX250Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/X25_0_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/X25_0.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X25_0_v2.pkl"


@configclass
class PickSingleEgadX252Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/X25_2_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/X25_2.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X25_2_v2.pkl"


@configclass
class PickSingleEgadX253Cfg(_PickSingleEgadBaseCfg):
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/egad/usd/X25_3_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/egad/urdf/X25_3.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_egad/trajectory-franka-X25_3_v2.pkl"
