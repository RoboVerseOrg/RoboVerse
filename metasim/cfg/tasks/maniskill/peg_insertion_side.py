"""The base class and derived classes for the peg-insertion task from ManiSkill."""

from metasim.cfg.checkers import DetectedChecker, RelativeBboxDetector
from metasim.cfg.objects import RigidObjCfg
from metasim.constants import PhysicStateType
from metasim.utils import configclass

from .maniskill_task_cfg import ManiskillTaskCfg


@configclass
class _PegInsertionSideBaseCfg(ManiskillTaskCfg):
    """The peg-insertion task from ManiSkill.

    The robot is tasked to pick up a peg on the ground and insert it into the hold of a fixed block.
    Note that in this implementation, the hole on the base is slightly enlarged to reduce the difficulty due to the dynamics gap.
    This class should be derived to specify the exact configuration (asset paths and demo path) of the task.
    """

    episode_length = 250
    checker = DetectedChecker(
        obj_name="stick",
        detector=RelativeBboxDetector(
            base_obj_name="box",
            relative_quat=[1, 0, 0, 0],
            relative_pos=[-0.05, 0, 0],
            checker_lower=[-0.05, -0.1, -0.1],
            checker_upper=[0.05, 0.1, 0.1],
        ),
    )


@configclass
class PegInsertionSide363Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_363.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_363.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-363_v2.pkl"


@configclass
class PegInsertionSide976Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_976.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_976.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-976_v2.pkl"


@configclass
class PegInsertionSide458Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_458.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_458.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-458_v2.pkl"


@configclass
class PegInsertionSide268Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_268.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_268.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-268_v2.pkl"


@configclass
class PegInsertionSide419Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_419.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_419.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-419_v2.pkl"


@configclass
class PegInsertionSide744Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_744.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_744.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-744_v2.pkl"


@configclass
class PegInsertionSide461Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_461.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_461.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-461_v2.pkl"


@configclass
class PegInsertionSide885Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_885.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_885.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-885_v2.pkl"


@configclass
class PegInsertionSide249Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_249.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_249.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-249_v2.pkl"


@configclass
class PegInsertionSide957Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_957.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_957.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-957_v2.pkl"


@configclass
class PegInsertionSide18Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_18.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_18.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-18_v2.pkl"


@configclass
class PegInsertionSide372Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_372.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_372.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-372_v2.pkl"


@configclass
class PegInsertionSide473Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_473.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_473.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-473_v2.pkl"


@configclass
class PegInsertionSide495Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_495.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_495.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-495_v2.pkl"


@configclass
class PegInsertionSide557Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_557.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_557.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-557_v2.pkl"


@configclass
class PegInsertionSide601Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_601.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_601.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-601_v2.pkl"


@configclass
class PegInsertionSide170Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_170.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_170.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-170_v2.pkl"


@configclass
class PegInsertionSide705Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_705.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_705.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-705_v2.pkl"


@configclass
class PegInsertionSide683Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_683.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_683.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-683_v2.pkl"


@configclass
class PegInsertionSide590Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_590.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_590.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-590_v2.pkl"


@configclass
class PegInsertionSide263Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_263.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_263.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-263_v2.pkl"


@configclass
class PegInsertionSide544Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_544.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_544.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-544_v2.pkl"


@configclass
class PegInsertionSide476Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_476.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_476.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-476_v2.pkl"


@configclass
class PegInsertionSide40Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_40.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_40.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-40_v2.pkl"


@configclass
class PegInsertionSide227Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_227.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_227.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-227_v2.pkl"


@configclass
class PegInsertionSide77Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_77.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_77.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-77_v2.pkl"


@configclass
class PegInsertionSide471Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_471.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_471.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-471_v2.pkl"


@configclass
class PegInsertionSide915Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_915.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_915.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-915_v2.pkl"


@configclass
class PegInsertionSide122Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_122.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_122.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-122_v2.pkl"


@configclass
class PegInsertionSide42Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_42.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_42.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-42_v2.pkl"


@configclass
class PegInsertionSide216Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_216.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_216.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-216_v2.pkl"


@configclass
class PegInsertionSide830Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_830.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_830.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-830_v2.pkl"


@configclass
class PegInsertionSide609Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_609.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_609.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-609_v2.pkl"


@configclass
class PegInsertionSide291Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_291.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_291.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-291_v2.pkl"


@configclass
class PegInsertionSide277Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_277.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_277.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-277_v2.pkl"


@configclass
class PegInsertionSide980Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_980.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_980.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-980_v2.pkl"


@configclass
class PegInsertionSide504Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_504.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_504.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-504_v2.pkl"


@configclass
class PegInsertionSide710Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_710.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_710.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-710_v2.pkl"


@configclass
class PegInsertionSide490Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_490.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_490.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-490_v2.pkl"


@configclass
class PegInsertionSide577Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_577.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_577.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-577_v2.pkl"


@configclass
class PegInsertionSide378Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_378.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_378.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-378_v2.pkl"


@configclass
class PegInsertionSide149Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_149.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_149.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-149_v2.pkl"


@configclass
class PegInsertionSide187Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_187.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_187.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-187_v2.pkl"


@configclass
class PegInsertionSide220Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_220.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_220.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-220_v2.pkl"


@configclass
class PegInsertionSide304Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_304.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_304.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-304_v2.pkl"


@configclass
class PegInsertionSide194Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_194.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_194.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-194_v2.pkl"


@configclass
class PegInsertionSide997Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_997.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_997.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-997_v2.pkl"


@configclass
class PegInsertionSide441Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_441.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_441.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-441_v2.pkl"


@configclass
class PegInsertionSide563Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_563.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_563.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-563_v2.pkl"


@configclass
class PegInsertionSide564Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_564.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_564.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-564_v2.pkl"


@configclass
class PegInsertionSide450Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_450.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_450.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-450_v2.pkl"


@configclass
class PegInsertionSide370Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_370.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_370.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-370_v2.pkl"


@configclass
class PegInsertionSide243Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_243.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_243.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-243_v2.pkl"


@configclass
class PegInsertionSide426Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_426.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_426.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-426_v2.pkl"


@configclass
class PegInsertionSide58Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_58.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_58.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-58_v2.pkl"


@configclass
class PegInsertionSide311Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_311.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_311.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-311_v2.pkl"


@configclass
class PegInsertionSide92Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_92.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_92.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-92_v2.pkl"


@configclass
class PegInsertionSide673Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_673.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_673.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-673_v2.pkl"


@configclass
class PegInsertionSide494Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_494.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_494.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-494_v2.pkl"


@configclass
class PegInsertionSide664Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_664.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_664.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-664_v2.pkl"


@configclass
class PegInsertionSide825Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_825.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_825.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-825_v2.pkl"


@configclass
class PegInsertionSide106Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_106.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_106.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-106_v2.pkl"


@configclass
class PegInsertionSide199Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_199.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_199.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-199_v2.pkl"


@configclass
class PegInsertionSide31Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_31.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_31.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-31_v2.pkl"


@configclass
class PegInsertionSide492Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_492.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_492.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-492_v2.pkl"


@configclass
class PegInsertionSide574Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_574.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_574.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-574_v2.pkl"


@configclass
class PegInsertionSide491Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_491.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_491.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-491_v2.pkl"


@configclass
class PegInsertionSide914Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_914.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_914.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-914_v2.pkl"


@configclass
class PegInsertionSide480Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_480.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_480.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-480_v2.pkl"


@configclass
class PegInsertionSide283Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_283.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_283.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-283_v2.pkl"


@configclass
class PegInsertionSide588Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_588.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_588.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-588_v2.pkl"


@configclass
class PegInsertionSide375Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_375.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_375.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-375_v2.pkl"


@configclass
class PegInsertionSide778Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_778.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_778.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-778_v2.pkl"


@configclass
class PegInsertionSide361Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_361.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_361.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-361_v2.pkl"


@configclass
class PegInsertionSide502Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_502.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_502.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-502_v2.pkl"


@configclass
class PegInsertionSide196Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_196.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_196.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-196_v2.pkl"


@configclass
class PegInsertionSide652Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_652.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_652.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-652_v2.pkl"


@configclass
class PegInsertionSide169Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_169.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_169.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-169_v2.pkl"


@configclass
class PegInsertionSide120Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_120.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_120.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-120_v2.pkl"


@configclass
class PegInsertionSide302Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_302.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_302.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-302_v2.pkl"


@configclass
class PegInsertionSide966Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_966.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_966.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-966_v2.pkl"


@configclass
class PegInsertionSide562Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_562.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_562.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-562_v2.pkl"


@configclass
class PegInsertionSide136Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_136.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_136.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-136_v2.pkl"


@configclass
class PegInsertionSide126Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_126.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_126.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-126_v2.pkl"


@configclass
class PegInsertionSide603Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_603.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_603.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-603_v2.pkl"


@configclass
class PegInsertionSide153Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_153.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_153.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-153_v2.pkl"


@configclass
class PegInsertionSide405Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_405.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_405.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-405_v2.pkl"


@configclass
class PegInsertionSide486Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_486.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_486.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-486_v2.pkl"


@configclass
class PegInsertionSide167Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_167.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_167.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-167_v2.pkl"


@configclass
class PegInsertionSide177Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_177.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_177.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-177_v2.pkl"


@configclass
class PegInsertionSide907Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_907.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_907.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-907_v2.pkl"


@configclass
class PegInsertionSide454Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_454.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_454.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-454_v2.pkl"


@configclass
class PegInsertionSide390Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_390.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_390.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-390_v2.pkl"


@configclass
class PegInsertionSide67Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_67.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_67.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-67_v2.pkl"


@configclass
class PegInsertionSide422Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_422.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_422.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-422_v2.pkl"


@configclass
class PegInsertionSide904Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_904.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_904.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-904_v2.pkl"


@configclass
class PegInsertionSide139Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_139.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_139.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-139_v2.pkl"


@configclass
class PegInsertionSide894Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_894.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_894.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-894_v2.pkl"


@configclass
class PegInsertionSide856Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_856.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_856.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-856_v2.pkl"


@configclass
class PegInsertionSide558Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_558.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_558.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-558_v2.pkl"


@configclass
class PegInsertionSide517Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_517.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_517.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-517_v2.pkl"


@configclass
class PegInsertionSide532Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_532.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_532.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-532_v2.pkl"


@configclass
class PegInsertionSide668Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_668.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_668.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-668_v2.pkl"


@configclass
class PegInsertionSide847Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_847.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_847.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-847_v2.pkl"


@configclass
class PegInsertionSide937Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_937.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_937.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-937_v2.pkl"


@configclass
class PegInsertionSide217Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_217.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_217.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-217_v2.pkl"


@configclass
class PegInsertionSide926Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_926.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_926.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-926_v2.pkl"


@configclass
class PegInsertionSide414Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_414.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_414.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-414_v2.pkl"


@configclass
class PegInsertionSide852Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_852.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_852.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-852_v2.pkl"


@configclass
class PegInsertionSide210Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_210.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_210.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-210_v2.pkl"


@configclass
class PegInsertionSide981Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_981.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_981.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-981_v2.pkl"


@configclass
class PegInsertionSide135Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_135.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_135.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-135_v2.pkl"


@configclass
class PegInsertionSide351Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_351.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_351.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-351_v2.pkl"


@configclass
class PegInsertionSide462Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_462.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_462.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-462_v2.pkl"


@configclass
class PegInsertionSide699Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_699.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_699.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-699_v2.pkl"


@configclass
class PegInsertionSide152Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_152.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_152.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-152_v2.pkl"


@configclass
class PegInsertionSide665Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_665.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_665.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-665_v2.pkl"


@configclass
class PegInsertionSide855Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_855.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_855.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-855_v2.pkl"


@configclass
class PegInsertionSide500Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_500.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_500.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-500_v2.pkl"


@configclass
class PegInsertionSide692Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_692.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_692.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-692_v2.pkl"


@configclass
class PegInsertionSide246Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_246.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_246.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-246_v2.pkl"


@configclass
class PegInsertionSide162Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_162.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_162.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-162_v2.pkl"


@configclass
class PegInsertionSide7Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_7.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_7.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-7_v2.pkl"


@configclass
class PegInsertionSide159Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_159.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_159.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-159_v2.pkl"


@configclass
class PegInsertionSide171Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_171.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_171.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-171_v2.pkl"


@configclass
class PegInsertionSide848Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_848.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_848.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-848_v2.pkl"


@configclass
class PegInsertionSide138Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_138.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_138.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-138_v2.pkl"


@configclass
class PegInsertionSide523Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_523.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_523.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-523_v2.pkl"


@configclass
class PegInsertionSide96Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_96.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_96.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-96_v2.pkl"


@configclass
class PegInsertionSide784Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_784.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_784.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-784_v2.pkl"


@configclass
class PegInsertionSide677Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_677.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_677.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-677_v2.pkl"


@configclass
class PegInsertionSide951Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_951.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_951.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-951_v2.pkl"


@configclass
class PegInsertionSide413Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_413.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_413.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-413_v2.pkl"


@configclass
class PegInsertionSide691Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_691.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_691.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-691_v2.pkl"


@configclass
class PegInsertionSide916Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_916.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_916.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-916_v2.pkl"


@configclass
class PegInsertionSide266Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_266.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_266.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-266_v2.pkl"


@configclass
class PegInsertionSide925Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_925.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_925.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-925_v2.pkl"


@configclass
class PegInsertionSide29Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_29.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_29.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-29_v2.pkl"


@configclass
class PegInsertionSide73Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_73.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_73.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-73_v2.pkl"


@configclass
class PegInsertionSide44Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_44.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_44.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-44_v2.pkl"


@configclass
class PegInsertionSide913Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_913.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_913.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-913_v2.pkl"


@configclass
class PegInsertionSide575Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_575.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_575.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-575_v2.pkl"


@configclass
class PegInsertionSide342Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_342.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_342.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-342_v2.pkl"


@configclass
class PegInsertionSide658Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_658.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_658.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-658_v2.pkl"


@configclass
class PegInsertionSide611Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_611.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_611.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-611_v2.pkl"


@configclass
class PegInsertionSide437Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_437.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_437.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-437_v2.pkl"


@configclass
class PegInsertionSide191Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_191.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_191.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-191_v2.pkl"


@configclass
class PegInsertionSide506Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_506.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_506.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-506_v2.pkl"


@configclass
class PegInsertionSide213Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_213.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_213.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-213_v2.pkl"


@configclass
class PegInsertionSide824Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_824.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_824.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-824_v2.pkl"


@configclass
class PegInsertionSide85Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_85.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_85.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-85_v2.pkl"


@configclass
class PegInsertionSide547Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_547.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_547.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-547_v2.pkl"


@configclass
class PegInsertionSide654Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_654.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_654.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-654_v2.pkl"


@configclass
class PegInsertionSide218Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_218.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_218.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-218_v2.pkl"


@configclass
class PegInsertionSide902Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_902.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_902.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-902_v2.pkl"


@configclass
class PegInsertionSide337Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_337.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_337.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-337_v2.pkl"


@configclass
class PegInsertionSide674Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_674.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_674.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-674_v2.pkl"


@configclass
class PegInsertionSide546Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_546.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_546.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-546_v2.pkl"


@configclass
class PegInsertionSide146Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_146.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_146.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-146_v2.pkl"


@configclass
class PegInsertionSide145Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_145.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_145.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-145_v2.pkl"


@configclass
class PegInsertionSide893Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_893.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_893.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-893_v2.pkl"


@configclass
class PegInsertionSide616Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_616.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_616.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-616_v2.pkl"


@configclass
class PegInsertionSide891Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_891.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_891.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-891_v2.pkl"


@configclass
class PegInsertionSide795Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_795.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_795.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-795_v2.pkl"


@configclass
class PegInsertionSide68Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_68.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_68.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-68_v2.pkl"


@configclass
class PegInsertionSide207Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_207.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_207.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-207_v2.pkl"


@configclass
class PegInsertionSide610Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_610.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_610.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-610_v2.pkl"


@configclass
class PegInsertionSide972Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_972.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_972.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-972_v2.pkl"


@configclass
class PegInsertionSide870Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_870.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_870.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-870_v2.pkl"


@configclass
class PegInsertionSide301Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_301.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_301.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-301_v2.pkl"


@configclass
class PegInsertionSide226Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_226.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_226.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-226_v2.pkl"


@configclass
class PegInsertionSide785Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_785.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_785.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-785_v2.pkl"


@configclass
class PegInsertionSide513Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_513.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_513.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-513_v2.pkl"


@configclass
class PegInsertionSide154Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_154.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_154.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-154_v2.pkl"


@configclass
class PegInsertionSide804Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_804.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_804.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-804_v2.pkl"


@configclass
class PegInsertionSide764Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_764.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_764.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-764_v2.pkl"


@configclass
class PegInsertionSide938Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_938.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_938.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-938_v2.pkl"


@configclass
class PegInsertionSide822Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_822.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_822.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-822_v2.pkl"


@configclass
class PegInsertionSide223Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_223.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_223.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-223_v2.pkl"


@configclass
class PegInsertionSide978Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_978.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_978.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-978_v2.pkl"


@configclass
class PegInsertionSide359Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_359.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_359.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-359_v2.pkl"


@configclass
class PegInsertionSide551Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_551.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_551.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-551_v2.pkl"


@configclass
class PegInsertionSide46Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_46.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_46.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-46_v2.pkl"


@configclass
class PegInsertionSide983Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_983.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_983.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-983_v2.pkl"


@configclass
class PegInsertionSide66Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_66.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_66.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-66_v2.pkl"


@configclass
class PegInsertionSide469Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_469.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_469.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-469_v2.pkl"


@configclass
class PegInsertionSide436Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_436.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_436.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-436_v2.pkl"


@configclass
class PegInsertionSide933Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_933.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_933.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-933_v2.pkl"


@configclass
class PegInsertionSide130Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_130.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_130.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-130_v2.pkl"


@configclass
class PegInsertionSide765Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_765.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_765.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-765_v2.pkl"


@configclass
class PegInsertionSide329Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_329.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_329.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-329_v2.pkl"


@configclass
class PegInsertionSide686Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_686.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_686.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-686_v2.pkl"


@configclass
class PegInsertionSide179Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_179.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_179.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-179_v2.pkl"


@configclass
class PegInsertionSide357Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_357.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_357.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-357_v2.pkl"


@configclass
class PegInsertionSide742Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_742.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_742.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-742_v2.pkl"


@configclass
class PegInsertionSide322Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_322.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_322.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-322_v2.pkl"


@configclass
class PegInsertionSide531Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_531.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_531.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-531_v2.pkl"


@configclass
class PegInsertionSide688Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_688.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_688.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-688_v2.pkl"


@configclass
class PegInsertionSide725Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_725.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_725.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-725_v2.pkl"


@configclass
class PegInsertionSide713Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_713.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_713.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-713_v2.pkl"


@configclass
class PegInsertionSide369Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_369.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_369.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-369_v2.pkl"


@configclass
class PegInsertionSide90Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_90.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_90.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-90_v2.pkl"


@configclass
class PegInsertionSide381Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_381.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_381.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-381_v2.pkl"


@configclass
class PegInsertionSide211Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_211.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_211.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-211_v2.pkl"


@configclass
class PegInsertionSide594Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_594.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_594.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-594_v2.pkl"


@configclass
class PegInsertionSide384Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_384.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_384.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-384_v2.pkl"


@configclass
class PegInsertionSide195Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_195.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_195.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-195_v2.pkl"


@configclass
class PegInsertionSide964Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_964.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_964.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-964_v2.pkl"


@configclass
class PegInsertionSide289Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_289.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_289.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-289_v2.pkl"


@configclass
class PegInsertionSide873Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_873.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_873.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-873_v2.pkl"


@configclass
class PegInsertionSide892Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_892.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_892.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-892_v2.pkl"


@configclass
class PegInsertionSide445Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_445.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_445.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-445_v2.pkl"


@configclass
class PegInsertionSide189Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_189.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_189.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-189_v2.pkl"


@configclass
class PegInsertionSide219Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_219.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_219.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-219_v2.pkl"


@configclass
class PegInsertionSide368Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_368.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_368.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-368_v2.pkl"


@configclass
class PegInsertionSide457Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_457.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_457.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-457_v2.pkl"


@configclass
class PegInsertionSide57Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_57.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_57.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-57_v2.pkl"


@configclass
class PegInsertionSide939Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_939.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_939.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-939_v2.pkl"


@configclass
class PegInsertionSide102Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_102.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_102.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-102_v2.pkl"


@configclass
class PegInsertionSide49Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_49.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_49.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-49_v2.pkl"


@configclass
class PegInsertionSide982Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_982.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_982.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-982_v2.pkl"


@configclass
class PegInsertionSide716Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_716.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_716.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-716_v2.pkl"


@configclass
class PegInsertionSide791Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_791.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_791.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-791_v2.pkl"


@configclass
class PegInsertionSide832Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_832.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_832.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-832_v2.pkl"


@configclass
class PegInsertionSide895Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_895.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_895.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-895_v2.pkl"


@configclass
class PegInsertionSide897Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_897.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_897.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-897_v2.pkl"


@configclass
class PegInsertionSide95Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_95.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_95.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-95_v2.pkl"


@configclass
class PegInsertionSide607Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_607.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_607.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-607_v2.pkl"


@configclass
class PegInsertionSide912Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_912.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_912.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-912_v2.pkl"


@configclass
class PegInsertionSide183Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_183.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_183.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-183_v2.pkl"


@configclass
class PegInsertionSide560Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_560.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_560.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-560_v2.pkl"


@configclass
class PegInsertionSide116Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_116.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_116.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-116_v2.pkl"


@configclass
class PegInsertionSide796Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_796.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_796.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-796_v2.pkl"


@configclass
class PegInsertionSide201Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_201.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_201.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-201_v2.pkl"


@configclass
class PegInsertionSide539Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_539.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_539.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-539_v2.pkl"


@configclass
class PegInsertionSide151Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_151.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_151.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-151_v2.pkl"


@configclass
class PegInsertionSide477Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_477.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_477.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-477_v2.pkl"


@configclass
class PegInsertionSide928Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_928.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_928.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-928_v2.pkl"


@configclass
class PegInsertionSide879Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_879.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_879.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-879_v2.pkl"


@configclass
class PegInsertionSide520Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_520.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_520.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-520_v2.pkl"


@configclass
class PegInsertionSide354Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_354.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_354.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-354_v2.pkl"


@configclass
class PegInsertionSide335Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_335.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_335.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-335_v2.pkl"


@configclass
class PegInsertionSide595Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_595.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_595.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-595_v2.pkl"


@configclass
class PegInsertionSide806Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_806.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_806.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-806_v2.pkl"


@configclass
class PegInsertionSide140Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_140.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_140.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-140_v2.pkl"


@configclass
class PegInsertionSide43Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_43.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_43.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-43_v2.pkl"


@configclass
class PegInsertionSide729Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_729.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_729.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-729_v2.pkl"


@configclass
class PegInsertionSide671Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_671.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_671.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-671_v2.pkl"


@configclass
class PegInsertionSide814Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_814.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_814.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-814_v2.pkl"


@configclass
class PegInsertionSide503Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_503.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_503.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-503_v2.pkl"


@configclass
class PegInsertionSide452Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_452.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_452.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-452_v2.pkl"


@configclass
class PegInsertionSide844Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_844.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_844.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-844_v2.pkl"


@configclass
class PegInsertionSide161Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_161.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_161.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-161_v2.pkl"


@configclass
class PegInsertionSide394Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_394.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_394.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-394_v2.pkl"


@configclass
class PegInsertionSide343Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_343.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_343.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-343_v2.pkl"


@configclass
class PegInsertionSide878Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_878.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_878.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-878_v2.pkl"


@configclass
class PegInsertionSide882Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_882.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_882.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-882_v2.pkl"


@configclass
class PegInsertionSide815Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_815.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_815.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-815_v2.pkl"


@configclass
class PegInsertionSide945Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_945.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_945.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-945_v2.pkl"


@configclass
class PegInsertionSide703Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_703.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_703.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-703_v2.pkl"


@configclass
class PegInsertionSide818Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_818.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_818.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-818_v2.pkl"


@configclass
class PegInsertionSide451Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_451.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_451.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-451_v2.pkl"


@configclass
class PegInsertionSide969Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_969.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_969.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-969_v2.pkl"


@configclass
class PegInsertionSide47Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_47.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_47.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-47_v2.pkl"


@configclass
class PegInsertionSide320Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_320.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_320.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-320_v2.pkl"


@configclass
class PegInsertionSide467Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_467.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_467.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-467_v2.pkl"


@configclass
class PegInsertionSide632Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_632.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_632.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-632_v2.pkl"


@configclass
class PegInsertionSide954Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_954.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_954.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-954_v2.pkl"


@configclass
class PegInsertionSide947Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_947.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_947.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-947_v2.pkl"


@configclass
class PegInsertionSide485Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_485.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_485.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-485_v2.pkl"


@configclass
class PegInsertionSide579Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_579.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_579.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-579_v2.pkl"


@configclass
class PegInsertionSide474Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_474.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_474.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-474_v2.pkl"


@configclass
class PegInsertionSide775Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_775.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_775.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-775_v2.pkl"


@configclass
class PegInsertionSide294Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_294.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_294.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-294_v2.pkl"


@configclass
class PegInsertionSide963Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_963.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_963.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-963_v2.pkl"


@configclass
class PegInsertionSide858Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_858.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_858.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-858_v2.pkl"


@configclass
class PegInsertionSide760Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_760.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_760.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-760_v2.pkl"


@configclass
class PegInsertionSide942Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_942.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_942.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-942_v2.pkl"


@configclass
class PegInsertionSide316Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_316.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_316.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-316_v2.pkl"


@configclass
class PegInsertionSide331Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_331.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_331.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-331_v2.pkl"


@configclass
class PegInsertionSide753Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_753.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_753.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-753_v2.pkl"


@configclass
class PegInsertionSide783Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_783.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_783.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-783_v2.pkl"


@configclass
class PegInsertionSide528Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_528.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_528.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-528_v2.pkl"


@configclass
class PegInsertionSide566Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_566.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_566.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-566_v2.pkl"


@configclass
class PegInsertionSide84Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_84.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_84.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-84_v2.pkl"


@configclass
class PegInsertionSide255Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_255.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_255.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-255_v2.pkl"


@configclass
class PegInsertionSide344Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_344.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_344.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-344_v2.pkl"


@configclass
class PegInsertionSide507Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_507.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_507.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-507_v2.pkl"


@configclass
class PegInsertionSide412Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_412.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_412.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-412_v2.pkl"


@configclass
class PegInsertionSide559Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_559.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_559.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-559_v2.pkl"


@configclass
class PegInsertionSide247Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_247.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_247.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-247_v2.pkl"


@configclass
class PegInsertionSide3Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_3.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_3.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-3_v2.pkl"


@configclass
class PegInsertionSide935Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_935.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_935.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-935_v2.pkl"


@configclass
class PegInsertionSide941Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_941.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_941.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-941_v2.pkl"


@configclass
class PegInsertionSide896Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_896.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_896.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-896_v2.pkl"


@configclass
class PegInsertionSide543Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_543.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_543.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-543_v2.pkl"


@configclass
class PegInsertionSide0Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_0.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_0.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-0_v2.pkl"


@configclass
class PegInsertionSide222Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_222.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_222.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-222_v2.pkl"


@configclass
class PegInsertionSide137Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_137.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_137.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-137_v2.pkl"


@configclass
class PegInsertionSide842Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_842.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_842.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-842_v2.pkl"


@configclass
class PegInsertionSide379Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_379.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_379.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-379_v2.pkl"


@configclass
class PegInsertionSide62Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_62.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_62.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-62_v2.pkl"


@configclass
class PegInsertionSide955Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_955.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_955.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-955_v2.pkl"


@configclass
class PegInsertionSide440Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_440.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_440.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-440_v2.pkl"


@configclass
class PegInsertionSide240Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_240.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_240.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-240_v2.pkl"


@configclass
class PegInsertionSide883Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_883.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_883.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-883_v2.pkl"


@configclass
class PegInsertionSide585Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_585.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_585.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-585_v2.pkl"


@configclass
class PegInsertionSide860Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_860.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_860.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-860_v2.pkl"


@configclass
class PegInsertionSide684Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_684.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_684.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-684_v2.pkl"


@configclass
class PegInsertionSide510Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_510.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_510.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-510_v2.pkl"


@configclass
class PegInsertionSide694Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_694.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_694.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-694_v2.pkl"


@configclass
class PegInsertionSide113Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_113.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_113.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-113_v2.pkl"


@configclass
class PegInsertionSide323Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_323.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_323.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-323_v2.pkl"


@configclass
class PegInsertionSide862Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_862.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_862.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-862_v2.pkl"


@configclass
class PegInsertionSide239Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_239.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_239.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-239_v2.pkl"


@configclass
class PegInsertionSide111Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_111.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_111.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-111_v2.pkl"


@configclass
class PegInsertionSide33Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_33.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_33.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-33_v2.pkl"


@configclass
class PegInsertionSide448Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_448.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_448.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-448_v2.pkl"


@configclass
class PegInsertionSide994Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_994.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_994.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-994_v2.pkl"


@configclass
class PegInsertionSide333Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_333.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_333.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-333_v2.pkl"


@configclass
class PegInsertionSide127Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_127.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_127.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-127_v2.pkl"


@configclass
class PegInsertionSide54Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_54.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_54.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-54_v2.pkl"


@configclass
class PegInsertionSide576Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_576.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_576.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-576_v2.pkl"


@configclass
class PegInsertionSide583Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_583.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_583.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-583_v2.pkl"


@configclass
class PegInsertionSide900Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_900.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_900.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-900_v2.pkl"


@configclass
class PegInsertionSide91Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_91.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_91.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-91_v2.pkl"


@configclass
class PegInsertionSide992Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_992.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_992.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-992_v2.pkl"


@configclass
class PegInsertionSide374Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_374.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_374.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-374_v2.pkl"


@configclass
class PegInsertionSide846Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_846.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_846.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-846_v2.pkl"


@configclass
class PegInsertionSide990Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_990.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_990.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-990_v2.pkl"


@configclass
class PegInsertionSide516Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_516.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_516.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-516_v2.pkl"


@configclass
class PegInsertionSide875Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_875.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_875.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-875_v2.pkl"


@configclass
class PegInsertionSide676Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_676.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_676.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-676_v2.pkl"


@configclass
class PegInsertionSide423Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_423.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_423.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-423_v2.pkl"


@configclass
class PegInsertionSide297Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_297.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_297.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-297_v2.pkl"


@configclass
class PegInsertionSide269Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_269.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_269.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-269_v2.pkl"


@configclass
class PegInsertionSide158Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_158.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_158.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-158_v2.pkl"


@configclass
class PegInsertionSide110Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_110.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_110.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-110_v2.pkl"


@configclass
class PegInsertionSide286Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_286.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_286.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-286_v2.pkl"


@configclass
class PegInsertionSide636Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_636.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_636.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-636_v2.pkl"


@configclass
class PegInsertionSide176Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_176.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_176.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-176_v2.pkl"


@configclass
class PegInsertionSide144Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_144.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_144.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-144_v2.pkl"


@configclass
class PegInsertionSide142Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_142.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_142.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-142_v2.pkl"


@configclass
class PegInsertionSide640Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_640.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_640.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-640_v2.pkl"


@configclass
class PegInsertionSide773Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_773.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_773.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-773_v2.pkl"


@configclass
class PegInsertionSide732Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_732.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_732.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-732_v2.pkl"


@configclass
class PegInsertionSide919Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_919.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_919.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-919_v2.pkl"


@configclass
class PegInsertionSide619Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_619.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_619.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-619_v2.pkl"


@configclass
class PegInsertionSide349Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_349.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_349.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-349_v2.pkl"


@configclass
class PegInsertionSide373Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_373.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_373.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-373_v2.pkl"


@configclass
class PegInsertionSide133Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_133.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_133.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-133_v2.pkl"


@configclass
class PegInsertionSide622Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_622.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_622.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-622_v2.pkl"


@configclass
class PegInsertionSide69Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_69.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_69.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-69_v2.pkl"


@configclass
class PegInsertionSide173Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_173.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_173.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-173_v2.pkl"


@configclass
class PegInsertionSide861Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_861.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_861.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-861_v2.pkl"


@configclass
class PegInsertionSide572Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_572.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_572.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-572_v2.pkl"


@configclass
class PegInsertionSide697Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_697.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_697.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-697_v2.pkl"


@configclass
class PegInsertionSide180Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_180.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_180.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-180_v2.pkl"


@configclass
class PegInsertionSide290Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_290.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_290.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-290_v2.pkl"


@configclass
class PegInsertionSide910Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_910.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_910.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-910_v2.pkl"


@configclass
class PegInsertionSide604Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_604.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_604.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-604_v2.pkl"


@configclass
class PegInsertionSide488Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_488.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_488.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-488_v2.pkl"


@configclass
class PegInsertionSide237Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_237.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_237.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-237_v2.pkl"


@configclass
class PegInsertionSide94Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_94.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_94.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-94_v2.pkl"


@configclass
class PegInsertionSide525Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_525.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_525.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-525_v2.pkl"


@configclass
class PegInsertionSide927Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_927.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_927.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-927_v2.pkl"


@configclass
class PegInsertionSide353Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_353.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_353.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-353_v2.pkl"


@configclass
class PegInsertionSide752Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_752.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_752.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-752_v2.pkl"


@configclass
class PegInsertionSide550Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_550.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_550.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-550_v2.pkl"


@configclass
class PegInsertionSide535Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_535.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_535.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-535_v2.pkl"


@configclass
class PegInsertionSide769Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_769.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_769.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-769_v2.pkl"


@configclass
class PegInsertionSide184Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_184.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_184.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-184_v2.pkl"


@configclass
class PegInsertionSide292Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_292.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_292.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-292_v2.pkl"


@configclass
class PegInsertionSide687Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_687.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_687.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-687_v2.pkl"


@configclass
class PegInsertionSide618Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_618.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_618.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-618_v2.pkl"


@configclass
class PegInsertionSide132Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_132.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_132.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-132_v2.pkl"


@configclass
class PegInsertionSide511Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_511.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_511.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-511_v2.pkl"


@configclass
class PegInsertionSide800Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_800.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_800.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-800_v2.pkl"


@configclass
class PegInsertionSide975Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_975.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_975.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-975_v2.pkl"


@configclass
class PegInsertionSide663Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_663.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_663.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-663_v2.pkl"


@configclass
class PegInsertionSide456Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_456.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_456.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-456_v2.pkl"


@configclass
class PegInsertionSide392Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_392.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_392.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-392_v2.pkl"


@configclass
class PegInsertionSide871Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_871.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_871.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-871_v2.pkl"


@configclass
class PegInsertionSide833Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_833.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_833.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-833_v2.pkl"


@configclass
class PegInsertionSide163Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_163.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_163.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-163_v2.pkl"


@configclass
class PegInsertionSide306Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_306.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_306.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-306_v2.pkl"


@configclass
class PegInsertionSide740Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_740.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_740.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-740_v2.pkl"


@configclass
class PegInsertionSide816Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_816.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_816.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-816_v2.pkl"


@configclass
class PegInsertionSide430Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_430.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_430.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-430_v2.pkl"


@configclass
class PegInsertionSide968Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_968.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_968.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-968_v2.pkl"


@configclass
class PegInsertionSide38Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_38.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_38.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-38_v2.pkl"


@configclass
class PegInsertionSide656Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_656.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_656.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-656_v2.pkl"


@configclass
class PegInsertionSide751Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_751.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_751.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-751_v2.pkl"


@configclass
class PegInsertionSide630Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_630.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_630.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-630_v2.pkl"


@configclass
class PegInsertionSide401Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_401.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_401.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-401_v2.pkl"


@configclass
class PegInsertionSide726Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_726.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_726.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-726_v2.pkl"


@configclass
class PegInsertionSide155Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_155.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_155.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-155_v2.pkl"


@configclass
class PegInsertionSide360Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_360.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_360.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-360_v2.pkl"


@configclass
class PegInsertionSide807Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_807.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_807.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-807_v2.pkl"


@configclass
class PegInsertionSide11Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_11.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_11.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-11_v2.pkl"


@configclass
class PegInsertionSide738Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_738.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_738.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-738_v2.pkl"


@configclass
class PegInsertionSide906Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_906.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_906.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-906_v2.pkl"


@configclass
class PegInsertionSide690Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_690.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_690.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-690_v2.pkl"


@configclass
class PegInsertionSide888Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_888.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_888.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-888_v2.pkl"


@configclass
class PegInsertionSide330Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_330.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_330.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-330_v2.pkl"


@configclass
class PegInsertionSide708Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_708.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_708.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-708_v2.pkl"


@configclass
class PegInsertionSide459Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_459.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_459.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-459_v2.pkl"


@configclass
class PegInsertionSide287Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_287.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_287.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-287_v2.pkl"


@configclass
class PegInsertionSide984Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_984.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_984.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-984_v2.pkl"


@configclass
class PegInsertionSide917Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_917.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_917.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-917_v2.pkl"


@configclass
class PegInsertionSide717Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_717.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_717.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-717_v2.pkl"


@configclass
class PegInsertionSide720Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_720.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_720.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-720_v2.pkl"


@configclass
class PegInsertionSide481Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_481.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_481.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-481_v2.pkl"


@configclass
class PegInsertionSide114Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_114.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_114.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-114_v2.pkl"


@configclass
class PegInsertionSide295Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_295.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_295.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-295_v2.pkl"


@configclass
class PegInsertionSide884Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_884.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_884.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-884_v2.pkl"


@configclass
class PegInsertionSide435Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_435.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_435.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-435_v2.pkl"


@configclass
class PegInsertionSide837Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_837.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_837.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-837_v2.pkl"


@configclass
class PegInsertionSide613Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_613.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_613.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-613_v2.pkl"


@configclass
class PegInsertionSide86Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_86.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_86.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-86_v2.pkl"


@configclass
class PegInsertionSide10Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_10.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_10.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-10_v2.pkl"


@configclass
class PegInsertionSide489Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_489.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_489.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-489_v2.pkl"


@configclass
class PegInsertionSide63Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_63.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_63.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-63_v2.pkl"


@configclass
class PegInsertionSide131Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_131.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_131.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-131_v2.pkl"


@configclass
class PegInsertionSide208Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_208.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_208.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-208_v2.pkl"


@configclass
class PegInsertionSide803Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_803.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_803.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-803_v2.pkl"


@configclass
class PegInsertionSide647Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_647.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_647.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-647_v2.pkl"


@configclass
class PegInsertionSide398Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_398.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_398.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-398_v2.pkl"


@configclass
class PegInsertionSide164Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_164.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_164.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-164_v2.pkl"


@configclass
class PegInsertionSide105Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_105.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_105.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-105_v2.pkl"


@configclass
class PegInsertionSide962Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_962.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_962.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-962_v2.pkl"


@configclass
class PegInsertionSide252Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_252.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_252.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-252_v2.pkl"


@configclass
class PegInsertionSide801Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_801.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_801.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-801_v2.pkl"


@configclass
class PegInsertionSide881Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_881.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_881.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-881_v2.pkl"


@configclass
class PegInsertionSide388Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_388.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_388.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-388_v2.pkl"


@configclass
class PegInsertionSide646Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_646.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_646.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-646_v2.pkl"


@configclass
class PegInsertionSide735Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_735.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_735.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-735_v2.pkl"


@configclass
class PegInsertionSide898Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_898.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_898.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-898_v2.pkl"


@configclass
class PegInsertionSide479Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_479.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_479.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-479_v2.pkl"


@configclass
class PegInsertionSide293Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_293.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_293.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-293_v2.pkl"


@configclass
class PegInsertionSide946Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_946.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_946.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-946_v2.pkl"


@configclass
class PegInsertionSide756Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_756.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_756.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-756_v2.pkl"


@configclass
class PegInsertionSide32Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_32.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_32.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-32_v2.pkl"


@configclass
class PegInsertionSide28Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_28.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_28.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-28_v2.pkl"


@configclass
class PegInsertionSide298Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_298.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_298.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-298_v2.pkl"


@configclass
class PegInsertionSide591Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_591.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_591.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-591_v2.pkl"


@configclass
class PegInsertionSide386Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_386.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_386.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-386_v2.pkl"


@configclass
class PegInsertionSide780Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_780.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_780.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-780_v2.pkl"


@configclass
class PegInsertionSide165Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_165.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_165.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-165_v2.pkl"


@configclass
class PegInsertionSide273Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_273.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_273.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-273_v2.pkl"


@configclass
class PegInsertionSide835Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_835.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_835.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-835_v2.pkl"


@configclass
class PegInsertionSide463Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_463.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_463.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-463_v2.pkl"


@configclass
class PegInsertionSide261Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_261.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_261.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-261_v2.pkl"


@configclass
class PegInsertionSide442Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_442.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_442.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-442_v2.pkl"


@configclass
class PegInsertionSide332Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_332.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_332.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-332_v2.pkl"


@configclass
class PegInsertionSide13Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_13.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_13.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-13_v2.pkl"


@configclass
class PegInsertionSide256Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_256.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_256.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-256_v2.pkl"


@configclass
class PegInsertionSide899Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_899.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_899.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-899_v2.pkl"


@configclass
class PegInsertionSide959Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_959.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_959.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-959_v2.pkl"


@configclass
class PegInsertionSide107Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_107.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_107.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-107_v2.pkl"


@configclass
class PegInsertionSide706Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_706.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_706.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-706_v2.pkl"


@configclass
class PegInsertionSide758Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_758.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_758.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-758_v2.pkl"


@configclass
class PegInsertionSide777Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_777.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_777.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-777_v2.pkl"


@configclass
class PegInsertionSide23Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_23.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_23.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-23_v2.pkl"


@configclass
class PegInsertionSide288Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_288.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_288.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-288_v2.pkl"


@configclass
class PegInsertionSide829Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_829.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_829.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-829_v2.pkl"


@configclass
class PegInsertionSide823Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_823.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_823.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-823_v2.pkl"


@configclass
class PegInsertionSide267Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_267.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_267.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-267_v2.pkl"


@configclass
class PegInsertionSide395Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_395.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_395.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-395_v2.pkl"


@configclass
class PegInsertionSide411Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_411.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_411.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-411_v2.pkl"


@configclass
class PegInsertionSide839Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_839.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_839.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-839_v2.pkl"


@configclass
class PegInsertionSide518Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_518.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_518.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-518_v2.pkl"


@configclass
class PegInsertionSide828Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_828.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_828.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-828_v2.pkl"


@configclass
class PegInsertionSide197Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_197.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_197.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-197_v2.pkl"


@configclass
class PegInsertionSide364Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_364.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_364.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-364_v2.pkl"


@configclass
class PegInsertionSide406Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_406.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_406.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-406_v2.pkl"


@configclass
class PegInsertionSide642Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_642.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_642.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-642_v2.pkl"


@configclass
class PegInsertionSide83Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_83.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_83.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-83_v2.pkl"


@configclass
class PegInsertionSide921Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_921.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_921.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-921_v2.pkl"


@configclass
class PegInsertionSide877Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_877.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_877.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-877_v2.pkl"


@configclass
class PegInsertionSide540Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_540.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_540.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-540_v2.pkl"


@configclass
class PegInsertionSide698Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_698.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_698.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-698_v2.pkl"


@configclass
class PegInsertionSide470Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_470.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_470.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-470_v2.pkl"


@configclass
class PegInsertionSide356Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_356.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_356.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-356_v2.pkl"


@configclass
class PegInsertionSide190Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_190.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_190.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-190_v2.pkl"


@configclass
class PegInsertionSide160Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_160.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_160.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-160_v2.pkl"


@configclass
class PegInsertionSide953Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_953.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_953.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-953_v2.pkl"


@configclass
class PegInsertionSide657Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_657.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_657.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-657_v2.pkl"


@configclass
class PegInsertionSide193Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_193.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_193.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-193_v2.pkl"


@configclass
class PegInsertionSide987Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_987.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_987.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-987_v2.pkl"


@configclass
class PegInsertionSide282Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_282.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_282.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-282_v2.pkl"


@configclass
class PegInsertionSide112Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_112.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_112.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-112_v2.pkl"


@configclass
class PegInsertionSide911Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_911.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_911.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-911_v2.pkl"


@configclass
class PegInsertionSide759Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_759.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_759.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-759_v2.pkl"


@configclass
class PegInsertionSide203Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_203.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_203.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-203_v2.pkl"


@configclass
class PegInsertionSide76Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_76.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_76.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-76_v2.pkl"


@configclass
class PegInsertionSide631Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_631.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_631.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-631_v2.pkl"


@configclass
class PegInsertionSide944Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_944.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_944.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-944_v2.pkl"


@configclass
class PegInsertionSide41Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_41.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_41.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-41_v2.pkl"


@configclass
class PegInsertionSide232Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_232.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_232.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-232_v2.pkl"


@configclass
class PegInsertionSide794Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_794.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_794.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-794_v2.pkl"


@configclass
class PegInsertionSide859Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_859.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_859.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-859_v2.pkl"


@configclass
class PegInsertionSide60Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_60.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_60.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-60_v2.pkl"


@configclass
class PegInsertionSide78Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_78.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_78.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-78_v2.pkl"


@configclass
class PegInsertionSide548Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_548.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_548.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-548_v2.pkl"


@configclass
class PegInsertionSide150Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_150.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_150.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-150_v2.pkl"


@configclass
class PegInsertionSide88Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_88.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_88.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-88_v2.pkl"


@configclass
class PegInsertionSide221Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_221.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_221.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-221_v2.pkl"


@configclass
class PegInsertionSide407Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_407.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_407.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-407_v2.pkl"


@configclass
class PegInsertionSide988Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_988.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_988.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-988_v2.pkl"


@configclass
class PegInsertionSide64Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_64.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_64.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-64_v2.pkl"


@configclass
class PegInsertionSide355Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_355.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_355.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-355_v2.pkl"


@configclass
class PegInsertionSide443Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_443.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_443.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-443_v2.pkl"


@configclass
class PegInsertionSide56Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_56.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_56.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-56_v2.pkl"


@configclass
class PegInsertionSide967Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_967.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_967.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-967_v2.pkl"


@configclass
class PegInsertionSide258Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_258.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_258.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-258_v2.pkl"


@configclass
class PegInsertionSide634Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_634.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_634.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-634_v2.pkl"


@configclass
class PegInsertionSide866Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_866.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_866.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-866_v2.pkl"


@configclass
class PegInsertionSide228Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_228.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_228.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-228_v2.pkl"


@configclass
class PegInsertionSide397Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_397.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_397.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-397_v2.pkl"


@configclass
class PegInsertionSide552Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_552.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_552.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-552_v2.pkl"


@configclass
class PegInsertionSide168Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_168.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_168.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-168_v2.pkl"


@configclass
class PegInsertionSide497Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_497.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_497.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-497_v2.pkl"


@configclass
class PegInsertionSide362Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_362.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_362.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-362_v2.pkl"


@configclass
class PegInsertionSide695Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_695.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_695.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-695_v2.pkl"


@configclass
class PegInsertionSide857Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_857.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_857.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-857_v2.pkl"


@configclass
class PegInsertionSide728Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_728.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_728.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-728_v2.pkl"


@configclass
class PegInsertionSide59Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_59.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_59.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-59_v2.pkl"


@configclass
class PegInsertionSide93Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_93.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_93.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-93_v2.pkl"


@configclass
class PegInsertionSide129Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_129.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_129.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-129_v2.pkl"


@configclass
class PegInsertionSide157Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_157.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_157.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-157_v2.pkl"


@configclass
class PegInsertionSide338Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_338.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_338.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-338_v2.pkl"


@configclass
class PegInsertionSide648Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_648.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_648.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-648_v2.pkl"


@configclass
class PegInsertionSide693Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_693.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_693.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-693_v2.pkl"


@configclass
class PegInsertionSide313Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_313.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_313.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-313_v2.pkl"


@configclass
class PegInsertionSide743Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_743.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_743.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-743_v2.pkl"


@configclass
class PegInsertionSide100Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_100.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_100.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-100_v2.pkl"


@configclass
class PegInsertionSide45Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_45.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_45.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-45_v2.pkl"


@configclass
class PegInsertionSide529Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_529.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_529.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-529_v2.pkl"


@configclass
class PegInsertionSide812Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_812.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_812.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-812_v2.pkl"


@configclass
class PegInsertionSide234Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_234.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_234.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-234_v2.pkl"


@configclass
class PegInsertionSide421Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_421.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_421.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-421_v2.pkl"


@configclass
class PegInsertionSide838Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_838.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_838.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-838_v2.pkl"


@configclass
class PegInsertionSide30Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_30.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_30.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-30_v2.pkl"


@configclass
class PegInsertionSide864Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_864.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_864.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-864_v2.pkl"


@configclass
class PegInsertionSide484Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_484.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_484.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-484_v2.pkl"


@configclass
class PegInsertionSide15Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_15.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_15.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-15_v2.pkl"


@configclass
class PegInsertionSide521Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_521.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_521.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-521_v2.pkl"


@configclass
class PegInsertionSide475Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_475.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_475.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-475_v2.pkl"


@configclass
class PegInsertionSide918Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_918.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_918.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-918_v2.pkl"


@configclass
class PegInsertionSide667Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_667.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_667.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-667_v2.pkl"


@configclass
class PegInsertionSide580Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_580.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_580.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-580_v2.pkl"


@configclass
class PegInsertionSide409Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_409.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_409.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-409_v2.pkl"


@configclass
class PegInsertionSide453Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_453.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_453.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-453_v2.pkl"


@configclass
class PegInsertionSide148Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_148.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_148.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-148_v2.pkl"


@configclass
class PegInsertionSide556Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_556.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_556.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-556_v2.pkl"


@configclass
class PegInsertionSide553Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_553.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_553.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-553_v2.pkl"


@configclass
class PegInsertionSide626Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_626.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_626.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-626_v2.pkl"


@configclass
class PegInsertionSide865Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_865.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_865.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-865_v2.pkl"


@configclass
class PegInsertionSide961Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_961.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_961.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-961_v2.pkl"


@configclass
class PegInsertionSide281Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_281.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_281.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-281_v2.pkl"


@configclass
class PegInsertionSide874Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_874.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_874.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-874_v2.pkl"


@configclass
class PegInsertionSide385Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_385.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_385.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-385_v2.pkl"


@configclass
class PegInsertionSide496Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_496.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_496.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-496_v2.pkl"


@configclass
class PegInsertionSide235Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_235.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_235.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-235_v2.pkl"


@configclass
class PegInsertionSide308Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_308.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_308.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-308_v2.pkl"


@configclass
class PegInsertionSide836Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_836.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_836.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-836_v2.pkl"


@configclass
class PegInsertionSide582Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_582.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_582.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-582_v2.pkl"


@configclass
class PegInsertionSide98Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_98.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_98.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-98_v2.pkl"


@configclass
class PegInsertionSide383Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_383.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_383.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-383_v2.pkl"


@configclass
class PegInsertionSide438Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_438.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_438.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-438_v2.pkl"


@configclass
class PegInsertionSide714Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_714.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_714.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-714_v2.pkl"


@configclass
class PegInsertionSide12Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_12.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_12.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-12_v2.pkl"


@configclass
class PegInsertionSide99Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_99.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_99.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-99_v2.pkl"


@configclass
class PegInsertionSide17Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_17.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_17.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-17_v2.pkl"


@configclass
class PegInsertionSide124Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_124.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_124.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-124_v2.pkl"


@configclass
class PegInsertionSide813Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_813.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_813.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-813_v2.pkl"


@configclass
class PegInsertionSide317Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_317.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_317.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-317_v2.pkl"


@configclass
class PegInsertionSide943Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_943.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_943.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-943_v2.pkl"


@configclass
class PegInsertionSide231Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_231.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_231.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-231_v2.pkl"


@configclass
class PegInsertionSide662Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_662.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_662.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-662_v2.pkl"


@configclass
class PegInsertionSide89Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_89.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_89.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-89_v2.pkl"


@configclass
class PegInsertionSide820Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_820.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_820.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-820_v2.pkl"


@configclass
class PegInsertionSide934Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_934.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_934.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-934_v2.pkl"


@configclass
class PegInsertionSide460Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_460.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_460.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-460_v2.pkl"


@configclass
class PegInsertionSide296Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_296.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_296.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-296_v2.pkl"


@configclass
class PegInsertionSide432Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_432.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_432.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-432_v2.pkl"


@configclass
class PegInsertionSide447Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_447.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_447.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-447_v2.pkl"


@configclass
class PegInsertionSide280Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_280.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_280.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-280_v2.pkl"


@configclass
class PegInsertionSide417Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_417.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_417.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-417_v2.pkl"


@configclass
class PegInsertionSide514Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_514.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_514.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-514_v2.pkl"


@configclass
class PegInsertionSide175Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_175.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_175.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-175_v2.pkl"


@configclass
class PegInsertionSide253Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_253.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_253.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-253_v2.pkl"


@configclass
class PegInsertionSide242Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_242.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_242.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-242_v2.pkl"


@configclass
class PegInsertionSide229Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_229.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_229.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-229_v2.pkl"


@configclass
class PegInsertionSide985Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_985.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_985.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-985_v2.pkl"


@configclass
class PegInsertionSide782Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_782.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_782.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-782_v2.pkl"


@configclass
class PegInsertionSide790Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_790.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_790.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-790_v2.pkl"


@configclass
class PegInsertionSide763Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_763.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_763.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-763_v2.pkl"


@configclass
class PegInsertionSide334Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_334.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_334.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-334_v2.pkl"


@configclass
class PegInsertionSide624Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_624.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_624.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-624_v2.pkl"


@configclass
class PegInsertionSide766Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_766.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_766.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-766_v2.pkl"


@configclass
class PegInsertionSide74Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_74.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_74.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-74_v2.pkl"


@configclass
class PegInsertionSide787Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_787.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_787.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-787_v2.pkl"


@configclass
class PegInsertionSide487Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_487.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_487.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-487_v2.pkl"


@configclass
class PegInsertionSide749Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_749.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_749.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-749_v2.pkl"


@configclass
class PegInsertionSide793Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_793.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_793.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-793_v2.pkl"


@configclass
class PegInsertionSide404Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_404.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_404.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-404_v2.pkl"


@configclass
class PegInsertionSide225Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_225.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_225.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-225_v2.pkl"


@configclass
class PegInsertionSide434Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_434.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_434.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-434_v2.pkl"


@configclass
class PegInsertionSide909Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_909.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_909.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-909_v2.pkl"


@configclass
class PegInsertionSide715Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_715.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_715.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-715_v2.pkl"


@configclass
class PegInsertionSide230Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_230.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_230.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-230_v2.pkl"


@configclass
class PegInsertionSide809Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_809.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_809.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-809_v2.pkl"


@configclass
class PegInsertionSide325Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_325.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_325.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-325_v2.pkl"


@configclass
class PegInsertionSide36Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_36.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_36.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-36_v2.pkl"


@configclass
class PegInsertionSide589Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_589.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_589.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-589_v2.pkl"


@configclass
class PegInsertionSide204Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_204.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_204.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-204_v2.pkl"


@configclass
class PegInsertionSide680Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_680.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_680.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-680_v2.pkl"


@configclass
class PegInsertionSide746Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_746.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_746.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-746_v2.pkl"


@configclass
class PegInsertionSide2Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_2.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_2.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-2_v2.pkl"


@configclass
class PegInsertionSide259Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_259.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_259.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-259_v2.pkl"


@configclass
class PegInsertionSide641Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_641.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_641.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-641_v2.pkl"


@configclass
class PegInsertionSide285Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_285.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_285.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-285_v2.pkl"


@configclass
class PegInsertionSide649Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_649.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_649.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-649_v2.pkl"


@configclass
class PegInsertionSide251Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_251.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_251.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-251_v2.pkl"


@configclass
class PegInsertionSide371Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_371.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_371.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-371_v2.pkl"


@configclass
class PegInsertionSide675Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_675.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_675.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-675_v2.pkl"


@configclass
class PegInsertionSide299Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_299.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_299.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-299_v2.pkl"


@configclass
class PegInsertionSide755Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_755.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_755.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-755_v2.pkl"


@configclass
class PegInsertionSide730Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_730.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_730.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-730_v2.pkl"


@configclass
class PegInsertionSide819Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_819.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_819.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-819_v2.pkl"


@configclass
class PegInsertionSide639Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_639.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_639.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-639_v2.pkl"


@configclass
class PegInsertionSide387Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_387.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_387.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-387_v2.pkl"


@configclass
class PegInsertionSide166Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_166.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_166.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-166_v2.pkl"


@configclass
class PegInsertionSide747Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_747.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_747.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-747_v2.pkl"


@configclass
class PegInsertionSide887Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_887.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_887.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-887_v2.pkl"


@configclass
class PegInsertionSide416Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_416.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_416.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-416_v2.pkl"


@configclass
class PegInsertionSide670Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_670.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_670.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-670_v2.pkl"


@configclass
class PegInsertionSide841Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_841.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_841.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-841_v2.pkl"


@configclass
class PegInsertionSide328Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_328.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_328.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-328_v2.pkl"


@configclass
class PegInsertionSide644Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_644.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_644.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-644_v2.pkl"


@configclass
class PegInsertionSide272Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_272.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_272.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-272_v2.pkl"


@configclass
class PegInsertionSide719Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_719.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_719.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-719_v2.pkl"


@configclass
class PegInsertionSide305Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_305.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_305.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-305_v2.pkl"


@configclass
class PegInsertionSide950Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_950.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_950.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-950_v2.pkl"


@configclass
class PegInsertionSide876Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_876.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_876.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-876_v2.pkl"


@configclass
class PegInsertionSide593Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_593.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_593.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-593_v2.pkl"


@configclass
class PegInsertionSide788Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_788.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_788.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-788_v2.pkl"


@configclass
class PegInsertionSide431Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_431.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_431.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-431_v2.pkl"


@configclass
class PegInsertionSide9Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_9.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_9.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-9_v2.pkl"


@configclass
class PegInsertionSide628Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_628.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_628.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-628_v2.pkl"


@configclass
class PegInsertionSide930Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_930.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_930.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-930_v2.pkl"


@configclass
class PegInsertionSide472Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_472.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_472.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-472_v2.pkl"


@configclass
class PegInsertionSide52Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_52.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_52.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-52_v2.pkl"


@configclass
class PegInsertionSide125Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_125.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_125.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-125_v2.pkl"


@configclass
class PegInsertionSide973Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_973.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_973.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-973_v2.pkl"


@configclass
class PegInsertionSide637Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_637.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_637.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-637_v2.pkl"


@configclass
class PegInsertionSide346Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_346.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_346.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-346_v2.pkl"


@configclass
class PegInsertionSide932Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_932.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_932.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-932_v2.pkl"


@configclass
class PegInsertionSide71Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_71.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_71.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-71_v2.pkl"


@configclass
class PegInsertionSide653Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_653.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_653.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-653_v2.pkl"


@configclass
class PegInsertionSide65Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_65.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_65.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-65_v2.pkl"


@configclass
class PegInsertionSide779Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_779.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_779.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-779_v2.pkl"


@configclass
class PegInsertionSide185Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_185.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_185.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-185_v2.pkl"


@configclass
class PegInsertionSide172Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_172.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_172.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-172_v2.pkl"


@configclass
class PegInsertionSide505Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_505.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_505.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-505_v2.pkl"


@configclass
class PegInsertionSide82Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_82.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_82.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-82_v2.pkl"


@configclass
class PegInsertionSide723Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_723.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_723.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-723_v2.pkl"


@configclass
class PegInsertionSide811Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_811.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_811.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-811_v2.pkl"


@configclass
class PegInsertionSide704Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_704.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_704.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-704_v2.pkl"


@configclass
class PegInsertionSide568Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_568.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_568.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-568_v2.pkl"


@configclass
class PegInsertionSide327Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_327.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_327.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-327_v2.pkl"


@configclass
class PegInsertionSide97Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_97.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_97.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-97_v2.pkl"


@configclass
class PegInsertionSide156Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_156.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_156.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-156_v2.pkl"


@configclass
class PegInsertionSide770Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_770.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_770.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-770_v2.pkl"


@configclass
class PegInsertionSide284Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_284.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_284.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-284_v2.pkl"


@configclass
class PegInsertionSide672Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_672.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_672.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-672_v2.pkl"


@configclass
class PegInsertionSide478Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_478.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_478.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-478_v2.pkl"


@configclass
class PegInsertionSide178Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_178.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_178.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-178_v2.pkl"


@configclass
class PegInsertionSide400Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_400.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_400.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-400_v2.pkl"


@configclass
class PegInsertionSide428Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_428.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_428.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-428_v2.pkl"


@configclass
class PegInsertionSide991Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_991.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_991.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-991_v2.pkl"


@configclass
class PegInsertionSide561Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_561.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_561.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-561_v2.pkl"


@configclass
class PegInsertionSide745Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_745.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_745.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-745_v2.pkl"


@configclass
class PegInsertionSide198Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_198.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_198.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-198_v2.pkl"


@configclass
class PegInsertionSide826Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_826.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_826.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-826_v2.pkl"


@configclass
class PegInsertionSide625Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_625.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_625.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-625_v2.pkl"


@configclass
class PegInsertionSide614Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_614.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_614.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-614_v2.pkl"


@configclass
class PegInsertionSide712Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_712.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_712.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-712_v2.pkl"


@configclass
class PegInsertionSide123Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_123.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_123.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-123_v2.pkl"


@configclass
class PegInsertionSide236Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_236.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_236.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-236_v2.pkl"


@configclass
class PegInsertionSide850Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_850.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_850.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-850_v2.pkl"


@configclass
class PegInsertionSide276Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_276.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_276.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-276_v2.pkl"


@configclass
class PegInsertionSide702Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_702.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_702.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-702_v2.pkl"


@configclass
class PegInsertionSide402Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_402.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_402.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-402_v2.pkl"


@configclass
class PegInsertionSide274Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_274.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_274.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-274_v2.pkl"


@configclass
class PegInsertionSide53Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_53.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_53.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-53_v2.pkl"


@configclass
class PegInsertionSide541Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_541.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_541.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-541_v2.pkl"


@configclass
class PegInsertionSide799Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_799.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_799.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-799_v2.pkl"


@configclass
class PegInsertionSide26Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_26.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_26.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-26_v2.pkl"


@configclass
class PegInsertionSide206Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_206.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_206.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-206_v2.pkl"


@configclass
class PegInsertionSide774Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_774.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_774.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-774_v2.pkl"


@configclass
class PegInsertionSide399Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_399.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_399.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-399_v2.pkl"


@configclass
class PegInsertionSide854Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_854.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_854.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-854_v2.pkl"


@configclass
class PegInsertionSide834Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_834.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_834.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-834_v2.pkl"


@configclass
class PegInsertionSide643Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_643.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_643.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-643_v2.pkl"


@configclass
class PegInsertionSide265Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_265.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_265.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-265_v2.pkl"


@configclass
class PegInsertionSide20Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_20.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_20.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-20_v2.pkl"


@configclass
class PegInsertionSide103Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_103.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_103.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-103_v2.pkl"


@configclass
class PegInsertionSide270Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_270.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_270.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-270_v2.pkl"


@configclass
class PegInsertionSide224Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_224.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_224.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-224_v2.pkl"


@configclass
class PegInsertionSide567Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_567.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_567.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-567_v2.pkl"


@configclass
class PegInsertionSide455Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_455.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_455.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-455_v2.pkl"


@configclass
class PegInsertionSide739Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_739.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_739.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-739_v2.pkl"


@configclass
class PegInsertionSide498Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_498.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_498.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-498_v2.pkl"


@configclass
class PegInsertionSide108Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_108.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_108.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-108_v2.pkl"


@configclass
class PegInsertionSide711Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_711.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_711.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-711_v2.pkl"


@configclass
class PegInsertionSide599Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_599.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_599.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-599_v2.pkl"


@configclass
class PegInsertionSide669Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_669.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_669.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-669_v2.pkl"


@configclass
class PegInsertionSide278Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_278.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_278.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-278_v2.pkl"


@configclass
class PegInsertionSide908Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_908.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_908.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-908_v2.pkl"


@configclass
class PegInsertionSide970Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_970.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_970.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-970_v2.pkl"


@configclass
class PegInsertionSide482Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_482.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_482.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-482_v2.pkl"


@configclass
class PegInsertionSide996Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_996.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_996.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-996_v2.pkl"


@configclass
class PegInsertionSide901Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_901.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_901.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-901_v2.pkl"


@configclass
class PegInsertionSide554Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_554.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_554.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-554_v2.pkl"


@configclass
class PegInsertionSide118Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_118.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_118.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-118_v2.pkl"


@configclass
class PegInsertionSide570Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_570.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_570.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-570_v2.pkl"


@configclass
class PegInsertionSide817Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_817.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_817.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-817_v2.pkl"


@configclass
class PegInsertionSide244Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_244.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_244.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-244_v2.pkl"


@configclass
class PegInsertionSide638Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_638.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_638.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-638_v2.pkl"


@configclass
class PegInsertionSide620Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_620.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_620.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-620_v2.pkl"


@configclass
class PegInsertionSide449Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_449.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_449.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-449_v2.pkl"


@configclass
class PegInsertionSide425Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_425.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_425.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-425_v2.pkl"


@configclass
class PegInsertionSide924Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_924.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_924.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-924_v2.pkl"


@configclass
class PegInsertionSide781Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_781.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_781.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-781_v2.pkl"


@configclass
class PegInsertionSide205Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_205.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_205.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-205_v2.pkl"


@configclass
class PegInsertionSide318Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_318.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_318.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-318_v2.pkl"


@configclass
class PegInsertionSide326Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_326.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_326.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-326_v2.pkl"


@configclass
class PegInsertionSide309Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_309.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_309.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-309_v2.pkl"


@configclass
class PegInsertionSide721Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_721.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_721.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-721_v2.pkl"


@configclass
class PegInsertionSide209Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_209.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_209.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-209_v2.pkl"


@configclass
class PegInsertionSide960Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_960.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_960.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-960_v2.pkl"


@configclass
class PegInsertionSide889Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_889.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_889.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-889_v2.pkl"


@configclass
class PegInsertionSide347Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_347.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_347.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-347_v2.pkl"


@configclass
class PegInsertionSide853Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_853.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_853.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-853_v2.pkl"


@configclass
class PegInsertionSide827Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_827.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_827.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-827_v2.pkl"


@configclass
class PegInsertionSide393Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_393.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_393.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-393_v2.pkl"


@configclass
class PegInsertionSide651Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_651.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_651.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-651_v2.pkl"


@configclass
class PegInsertionSide512Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_512.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_512.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-512_v2.pkl"


@configclass
class PegInsertionSide186Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_186.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_186.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-186_v2.pkl"


@configclass
class PegInsertionSide538Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_538.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_538.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-538_v2.pkl"


@configclass
class PegInsertionSide48Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_48.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_48.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-48_v2.pkl"


@configclass
class PegInsertionSide380Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_380.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_380.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-380_v2.pkl"


@configclass
class PegInsertionSide573Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_573.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_573.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-573_v2.pkl"


@configclass
class PegInsertionSide754Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_754.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_754.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-754_v2.pkl"


@configclass
class PegInsertionSide681Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_681.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_681.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-681_v2.pkl"


@configclass
class PegInsertionSide802Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_802.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_802.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-802_v2.pkl"


@configclass
class PegInsertionSide920Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_920.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_920.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-920_v2.pkl"


@configclass
class PegInsertionSide768Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_768.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_768.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-768_v2.pkl"


@configclass
class PegInsertionSide666Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_666.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_666.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-666_v2.pkl"


@configclass
class PegInsertionSide974Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_974.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_974.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-974_v2.pkl"


@configclass
class PegInsertionSide352Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_352.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_352.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-352_v2.pkl"


@configclass
class PegInsertionSide608Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_608.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_608.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-608_v2.pkl"


@configclass
class PegInsertionSide633Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_633.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_633.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-633_v2.pkl"


@configclass
class PegInsertionSide701Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_701.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_701.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-701_v2.pkl"


@configclass
class PegInsertionSide923Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_923.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_923.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-923_v2.pkl"


@configclass
class PegInsertionSide986Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_986.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_986.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-986_v2.pkl"


@configclass
class PegInsertionSide215Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_215.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_215.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-215_v2.pkl"


@configclass
class PegInsertionSide952Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_952.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_952.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-952_v2.pkl"


@configclass
class PegInsertionSide733Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_733.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_733.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-733_v2.pkl"


@configclass
class PegInsertionSide629Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_629.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_629.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-629_v2.pkl"


@configclass
class PegInsertionSide722Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_722.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_722.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-722_v2.pkl"


@configclass
class PegInsertionSide598Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_598.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_598.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-598_v2.pkl"


@configclass
class PegInsertionSide709Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_709.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_709.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-709_v2.pkl"


@configclass
class PegInsertionSide307Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_307.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_307.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-307_v2.pkl"


@configclass
class PegInsertionSide660Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_660.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_660.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-660_v2.pkl"


@configclass
class PegInsertionSide104Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_104.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_104.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-104_v2.pkl"


@configclass
class PegInsertionSide427Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_427.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_427.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-427_v2.pkl"


@configclass
class PegInsertionSide16Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_16.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_16.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-16_v2.pkl"


@configclass
class PegInsertionSide797Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_797.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_797.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-797_v2.pkl"


@configclass
class PegInsertionSide965Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_965.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_965.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-965_v2.pkl"


@configclass
class PegInsertionSide545Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_545.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_545.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-545_v2.pkl"


@configclass
class PegInsertionSide949Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_949.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_949.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-949_v2.pkl"


@configclass
class PegInsertionSide922Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_922.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_922.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-922_v2.pkl"


@configclass
class PegInsertionSide549Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_549.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_549.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-549_v2.pkl"


@configclass
class PegInsertionSide464Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_464.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_464.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-464_v2.pkl"


@configclass
class PegInsertionSide627Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_627.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_627.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-627_v2.pkl"


@configclass
class PegInsertionSide315Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_315.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_315.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-315_v2.pkl"


@configclass
class PegInsertionSide880Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_880.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_880.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-880_v2.pkl"


@configclass
class PegInsertionSide542Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_542.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_542.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-542_v2.pkl"


@configclass
class PegInsertionSide678Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_678.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_678.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-678_v2.pkl"


@configclass
class PegInsertionSide14Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_14.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_14.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-14_v2.pkl"


@configclass
class PegInsertionSide233Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_233.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_233.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-233_v2.pkl"


@configclass
class PegInsertionSide341Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_341.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_341.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-341_v2.pkl"


@configclass
class PegInsertionSide555Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_555.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_555.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-555_v2.pkl"


@configclass
class PegInsertionSide415Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_415.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_415.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-415_v2.pkl"


@configclass
class PegInsertionSide279Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_279.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_279.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-279_v2.pkl"


@configclass
class PegInsertionSide101Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_101.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_101.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-101_v2.pkl"


@configclass
class PegInsertionSide602Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_602.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_602.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-602_v2.pkl"


@configclass
class PegInsertionSide724Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_724.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_724.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-724_v2.pkl"


@configclass
class PegInsertionSide79Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_79.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_79.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-79_v2.pkl"


@configclass
class PegInsertionSide522Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_522.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_522.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-522_v2.pkl"


@configclass
class PegInsertionSide808Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_808.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_808.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-808_v2.pkl"


@configclass
class PegInsertionSide537Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_537.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_537.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-537_v2.pkl"


@configclass
class PegInsertionSide275Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_275.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_275.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-275_v2.pkl"


@configclass
class PegInsertionSide358Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_358.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_358.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-358_v2.pkl"


@configclass
class PegInsertionSide685Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_685.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_685.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-685_v2.pkl"


@configclass
class PegInsertionSide617Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_617.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_617.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-617_v2.pkl"


@configclass
class PegInsertionSide526Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_526.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_526.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-526_v2.pkl"


@configclass
class PegInsertionSide248Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_248.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_248.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-248_v2.pkl"


@configclass
class PegInsertionSide377Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_377.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_377.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-377_v2.pkl"


@configclass
class PegInsertionSide527Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_527.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_527.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-527_v2.pkl"


@configclass
class PegInsertionSide843Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_843.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_843.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-843_v2.pkl"


@configclass
class PegInsertionSide659Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_659.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_659.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-659_v2.pkl"


@configclass
class PegInsertionSide134Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_134.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_134.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-134_v2.pkl"


@configclass
class PegInsertionSide21Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_21.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_21.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-21_v2.pkl"


@configclass
class PegInsertionSide606Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_606.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_606.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-606_v2.pkl"


@configclass
class PegInsertionSide391Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_391.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_391.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-391_v2.pkl"


@configclass
class PegInsertionSide849Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_849.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_849.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-849_v2.pkl"


@configclass
class PegInsertionSide19Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_19.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_19.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-19_v2.pkl"


@configclass
class PegInsertionSide979Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_979.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_979.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-979_v2.pkl"


@configclass
class PegInsertionSide737Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_737.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_737.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-737_v2.pkl"


@configclass
class PegInsertionSide312Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_312.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_312.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-312_v2.pkl"


@configclass
class PegInsertionSide621Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_621.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_621.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-621_v2.pkl"


@configclass
class PegInsertionSide863Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_863.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_863.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-863_v2.pkl"


@configclass
class PegInsertionSide245Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_245.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_245.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-245_v2.pkl"


@configclass
class PegInsertionSide241Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_241.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_241.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-241_v2.pkl"


@configclass
class PegInsertionSide80Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_80.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_80.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-80_v2.pkl"


@configclass
class PegInsertionSide612Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_612.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_612.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-612_v2.pkl"


@configclass
class PegInsertionSide87Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_87.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_87.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-87_v2.pkl"


@configclass
class PegInsertionSide376Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_376.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_376.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-376_v2.pkl"


@configclass
class PegInsertionSide993Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_993.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_993.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-993_v2.pkl"


@configclass
class PegInsertionSide444Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_444.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_444.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-444_v2.pkl"


@configclass
class PegInsertionSide192Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_192.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_192.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-192_v2.pkl"


@configclass
class PegInsertionSide650Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_650.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_650.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-650_v2.pkl"


@configclass
class PegInsertionSide792Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_792.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_792.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-792_v2.pkl"


@configclass
class PegInsertionSide772Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_772.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_772.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-772_v2.pkl"


@configclass
class PegInsertionSide382Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_382.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_382.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-382_v2.pkl"


@configclass
class PegInsertionSide115Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_115.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_115.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-115_v2.pkl"


@configclass
class PegInsertionSide748Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_748.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_748.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-748_v2.pkl"


@configclass
class PegInsertionSide202Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_202.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_202.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-202_v2.pkl"


@configclass
class PegInsertionSide776Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_776.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_776.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-776_v2.pkl"


@configclass
class PegInsertionSide958Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_958.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_958.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-958_v2.pkl"


@configclass
class PegInsertionSide655Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_655.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_655.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-655_v2.pkl"


@configclass
class PegInsertionSide761Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_761.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_761.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-761_v2.pkl"


@configclass
class PegInsertionSide727Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_727.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_727.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-727_v2.pkl"


@configclass
class PegInsertionSide536Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_536.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_536.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-536_v2.pkl"


@configclass
class PegInsertionSide121Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_121.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_121.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-121_v2.pkl"


@configclass
class PegInsertionSide623Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_623.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_623.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-623_v2.pkl"


@configclass
class PegInsertionSide396Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_396.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_396.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-396_v2.pkl"


@configclass
class PegInsertionSide867Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_867.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_867.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-867_v2.pkl"


@configclass
class PegInsertionSide303Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_303.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_303.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-303_v2.pkl"


@configclass
class PegInsertionSide851Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_851.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_851.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-851_v2.pkl"


@configclass
class PegInsertionSide890Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_890.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_890.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-890_v2.pkl"


@configclass
class PegInsertionSide499Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_499.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_499.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-499_v2.pkl"


@configclass
class PegInsertionSide250Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_250.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_250.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-250_v2.pkl"


@configclass
class PegInsertionSide821Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_821.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_821.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-821_v2.pkl"


@configclass
class PegInsertionSide798Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_798.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_798.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-798_v2.pkl"


@configclass
class PegInsertionSide37Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_37.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_37.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-37_v2.pkl"


@configclass
class PegInsertionSide336Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_336.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_336.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-336_v2.pkl"


@configclass
class PegInsertionSide948Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_948.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_948.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-948_v2.pkl"


@configclass
class PegInsertionSide995Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_995.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_995.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-995_v2.pkl"


@configclass
class PegInsertionSide831Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_831.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_831.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-831_v2.pkl"


@configclass
class PegInsertionSide587Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_587.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_587.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-587_v2.pkl"


@configclass
class PegInsertionSide117Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_117.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_117.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-117_v2.pkl"


@configclass
class PegInsertionSide789Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_789.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_789.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-789_v2.pkl"


@configclass
class PegInsertionSide348Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_348.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_348.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-348_v2.pkl"


@configclass
class PegInsertionSide1Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_1.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_1.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-1_v2.pkl"


@configclass
class PegInsertionSide109Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_109.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_109.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-109_v2.pkl"


@configclass
class PegInsertionSide569Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_569.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_569.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-569_v2.pkl"


@configclass
class PegInsertionSide493Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_493.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_493.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-493_v2.pkl"


@configclass
class PegInsertionSide119Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_119.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_119.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-119_v2.pkl"


@configclass
class PegInsertionSide22Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_22.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_22.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-22_v2.pkl"


@configclass
class PegInsertionSide615Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_615.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_615.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-615_v2.pkl"


@configclass
class PegInsertionSide367Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_367.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_367.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-367_v2.pkl"


@configclass
class PegInsertionSide466Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_466.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_466.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-466_v2.pkl"


@configclass
class PegInsertionSide200Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_200.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_200.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-200_v2.pkl"


@configclass
class PegInsertionSide257Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_257.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_257.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-257_v2.pkl"


@configclass
class PegInsertionSide483Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_483.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_483.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-483_v2.pkl"


@configclass
class PegInsertionSide731Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_731.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_731.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-731_v2.pkl"


@configclass
class PegInsertionSide734Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_734.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_734.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-734_v2.pkl"


@configclass
class PegInsertionSide433Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_433.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_433.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-433_v2.pkl"


@configclass
class PegInsertionSide147Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_147.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_147.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-147_v2.pkl"


@configclass
class PegInsertionSide350Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_350.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_350.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-350_v2.pkl"


@configclass
class PegInsertionSide679Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_679.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_679.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-679_v2.pkl"


@configclass
class PegInsertionSide736Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_736.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_736.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-736_v2.pkl"


@configclass
class PegInsertionSide508Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_508.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_508.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-508_v2.pkl"


@configclass
class PegInsertionSide578Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_578.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_578.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-578_v2.pkl"


@configclass
class PegInsertionSide212Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_212.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_212.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-212_v2.pkl"


@configclass
class PegInsertionSide581Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_581.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_581.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-581_v2.pkl"


@configclass
class PegInsertionSide260Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_260.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_260.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-260_v2.pkl"


@configclass
class PegInsertionSide366Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_366.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_366.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-366_v2.pkl"


@configclass
class PegInsertionSide805Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_805.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_805.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-805_v2.pkl"


@configclass
class PegInsertionSide25Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_25.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_25.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-25_v2.pkl"


@configclass
class PegInsertionSide519Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_519.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_519.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-519_v2.pkl"


@configclass
class PegInsertionSide700Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_700.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_700.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-700_v2.pkl"


@configclass
class PegInsertionSide418Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_418.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_418.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-418_v2.pkl"


@configclass
class PegInsertionSide989Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_989.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_989.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-989_v2.pkl"


@configclass
class PegInsertionSide905Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_905.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_905.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-905_v2.pkl"


@configclass
class PegInsertionSide6Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_6.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_6.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-6_v2.pkl"


@configclass
class PegInsertionSide929Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_929.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_929.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-929_v2.pkl"


@configclass
class PegInsertionSide34Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_34.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_34.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-34_v2.pkl"


@configclass
class PegInsertionSide408Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_408.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_408.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-408_v2.pkl"


@configclass
class PegInsertionSide468Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_468.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_468.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-468_v2.pkl"


@configclass
class PegInsertionSide977Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_977.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_977.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-977_v2.pkl"


@configclass
class PegInsertionSide584Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_584.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_584.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-584_v2.pkl"


@configclass
class PegInsertionSide55Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_55.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_55.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-55_v2.pkl"


@configclass
class PegInsertionSide50Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_50.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_50.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-50_v2.pkl"


@configclass
class PegInsertionSide509Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_509.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_509.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-509_v2.pkl"


@configclass
class PegInsertionSide971Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_971.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_971.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-971_v2.pkl"


@configclass
class PegInsertionSide936Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_936.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_936.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-936_v2.pkl"


@configclass
class PegInsertionSide254Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_254.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_254.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-254_v2.pkl"


@configclass
class PegInsertionSide70Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_70.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_70.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-70_v2.pkl"


@configclass
class PegInsertionSide141Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_141.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_141.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-141_v2.pkl"


@configclass
class PegInsertionSide51Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_51.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_51.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-51_v2.pkl"


@configclass
class PegInsertionSide596Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_596.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_596.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-596_v2.pkl"


@configclass
class PegInsertionSide661Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_661.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_661.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-661_v2.pkl"


@configclass
class PegInsertionSide869Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_869.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_869.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-869_v2.pkl"


@configclass
class PegInsertionSide465Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_465.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_465.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-465_v2.pkl"


@configclass
class PegInsertionSide128Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_128.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_128.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-128_v2.pkl"


@configclass
class PegInsertionSide439Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_439.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_439.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-439_v2.pkl"


@configclass
class PegInsertionSide605Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_605.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_605.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-605_v2.pkl"


@configclass
class PegInsertionSide940Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_940.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_940.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-940_v2.pkl"


@configclass
class PegInsertionSide319Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_319.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_319.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-319_v2.pkl"


@configclass
class PegInsertionSide4Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_4.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_4.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-4_v2.pkl"


@configclass
class PegInsertionSide682Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_682.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_682.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-682_v2.pkl"


@configclass
class PegInsertionSide8Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_8.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_8.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-8_v2.pkl"


@configclass
class PegInsertionSide534Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_534.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_534.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-534_v2.pkl"


@configclass
class PegInsertionSide339Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_339.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_339.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-339_v2.pkl"


@configclass
class PegInsertionSide365Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_365.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_365.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-365_v2.pkl"


@configclass
class PegInsertionSide530Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_530.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_530.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-530_v2.pkl"


@configclass
class PegInsertionSide845Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_845.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_845.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-845_v2.pkl"


@configclass
class PegInsertionSide592Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_592.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_592.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-592_v2.pkl"


@configclass
class PegInsertionSide600Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_600.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_600.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-600_v2.pkl"


@configclass
class PegInsertionSide181Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_181.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_181.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-181_v2.pkl"


@configclass
class PegInsertionSide321Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_321.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_321.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-321_v2.pkl"


@configclass
class PegInsertionSide501Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_501.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_501.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-501_v2.pkl"


@configclass
class PegInsertionSide689Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_689.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_689.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-689_v2.pkl"


@configclass
class PegInsertionSide345Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_345.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_345.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-345_v2.pkl"


@configclass
class PegInsertionSide238Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_238.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_238.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-238_v2.pkl"


@configclass
class PegInsertionSide389Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_389.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_389.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-389_v2.pkl"


@configclass
class PegInsertionSide645Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_645.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_645.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-645_v2.pkl"


@configclass
class PegInsertionSide24Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_24.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_24.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-24_v2.pkl"


@configclass
class PegInsertionSide771Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_771.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_771.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-771_v2.pkl"


@configclass
class PegInsertionSide27Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_27.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_27.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-27_v2.pkl"


@configclass
class PegInsertionSide868Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_868.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_868.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-868_v2.pkl"


@configclass
class PegInsertionSide61Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_61.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_61.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-61_v2.pkl"


@configclass
class PegInsertionSide271Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_271.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_271.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-271_v2.pkl"


@configclass
class PegInsertionSide264Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_264.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_264.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-264_v2.pkl"


@configclass
class PegInsertionSide586Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_586.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_586.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-586_v2.pkl"


@configclass
class PegInsertionSide903Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_903.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_903.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-903_v2.pkl"


@configclass
class PegInsertionSide188Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_188.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_188.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-188_v2.pkl"


@configclass
class PegInsertionSide515Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_515.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_515.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-515_v2.pkl"


@configclass
class PegInsertionSide324Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_324.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_324.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-324_v2.pkl"


@configclass
class PegInsertionSide872Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_872.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_872.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-872_v2.pkl"


@configclass
class PegInsertionSide424Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_424.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_424.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-424_v2.pkl"


@configclass
class PegInsertionSide340Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_340.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_340.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-340_v2.pkl"


@configclass
class PegInsertionSide143Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_143.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_143.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-143_v2.pkl"


@configclass
class PegInsertionSide718Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_718.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_718.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-718_v2.pkl"


@configclass
class PegInsertionSide565Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_565.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_565.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-565_v2.pkl"


@configclass
class PegInsertionSide420Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_420.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_420.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-420_v2.pkl"


@configclass
class PegInsertionSide5Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_5.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_5.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-5_v2.pkl"


@configclass
class PegInsertionSide314Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_314.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_314.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-314_v2.pkl"


@configclass
class PegInsertionSide72Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_72.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_72.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-72_v2.pkl"


@configclass
class PegInsertionSide998Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_998.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_998.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-998_v2.pkl"


@configclass
class PegInsertionSide956Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_956.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_956.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-956_v2.pkl"


@configclass
class PegInsertionSide767Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_767.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_767.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-767_v2.pkl"


@configclass
class PegInsertionSide762Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_762.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_762.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-762_v2.pkl"


@configclass
class PegInsertionSide999Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_999.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_999.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-999_v2.pkl"


@configclass
class PegInsertionSide446Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_446.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_446.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-446_v2.pkl"


@configclass
class PegInsertionSide39Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_39.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_39.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-39_v2.pkl"


@configclass
class PegInsertionSide597Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_597.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_597.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-597_v2.pkl"


@configclass
class PegInsertionSide707Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_707.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_707.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-707_v2.pkl"


@configclass
class PegInsertionSide741Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_741.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_741.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-741_v2.pkl"


@configclass
class PegInsertionSide300Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_300.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_300.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-300_v2.pkl"


@configclass
class PegInsertionSide635Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_635.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_635.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-635_v2.pkl"


@configclass
class PegInsertionSide182Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_182.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_182.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-182_v2.pkl"


@configclass
class PegInsertionSide429Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_429.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_429.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-429_v2.pkl"


@configclass
class PegInsertionSide696Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_696.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_696.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-696_v2.pkl"


@configclass
class PegInsertionSide786Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_786.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_786.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-786_v2.pkl"


@configclass
class PegInsertionSide840Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_840.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_840.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-840_v2.pkl"


@configclass
class PegInsertionSide403Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_403.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_403.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-403_v2.pkl"


@configclass
class PegInsertionSide524Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_524.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_524.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-524_v2.pkl"


@configclass
class PegInsertionSide310Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_310.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_310.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-310_v2.pkl"


@configclass
class PegInsertionSide214Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_214.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_214.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-214_v2.pkl"


@configclass
class PegInsertionSide757Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_757.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_757.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-757_v2.pkl"


@configclass
class PegInsertionSide35Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_35.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_35.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-35_v2.pkl"


@configclass
class PegInsertionSide750Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_750.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_750.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-750_v2.pkl"


@configclass
class PegInsertionSide931Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_931.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_931.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-931_v2.pkl"


@configclass
class PegInsertionSide886Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_886.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_886.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-886_v2.pkl"


@configclass
class PegInsertionSide174Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_174.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_174.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-174_v2.pkl"


@configclass
class PegInsertionSide75Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_75.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_75.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-75_v2.pkl"


@configclass
class PegInsertionSide262Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_262.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_262.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-262_v2.pkl"


@configclass
class PegInsertionSide571Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_571.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_571.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-571_v2.pkl"


@configclass
class PegInsertionSide410Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_410.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_410.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-410_v2.pkl"


@configclass
class PegInsertionSide533Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_533.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_533.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-533_v2.pkl"


@configclass
class PegInsertionSide81Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_81.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_81.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-81_v2.pkl"


@configclass
class PegInsertionSide810Cfg(_PegInsertionSideBaseCfg):
    objects = [
        RigidObjCfg(
            name="box",
            usd_path="roboverse_data/assets/maniskill/peg/base_810.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="roboverse_data/assets/maniskill/peg/stick_810.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-810_v2.pkl"
