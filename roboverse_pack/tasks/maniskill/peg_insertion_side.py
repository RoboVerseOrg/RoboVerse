"""The base class and derived classes for the peg-insertion task from ManiSkill."""

from metasim.constants import PhysicStateType
from metasim.example.example_pack.tasks.checkers import DetectedChecker, RelativeBboxDetector
from metasim.scenario.objects import RigidObjCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.registry import register_task

from .maniskill_base import ManiskillBaseTask


@register_task("maniskill.peg_insertion_side_base", "peg_insertion_side_base")
class PegInsertionSideBaseTask(ManiskillBaseTask):
    """The peg-insertion task from ManiSkill.

    The robot is tasked to pick up a peg on the ground and insert it into the hold of a fixed block.
    Note that in this implementation, the hole on the base is slightly enlarged to reduce the difficulty due to the dynamics gap.
    This class should be derived to specify the exact configuration (asset paths and demo path) of the task.
    """

    # task horizon
    max_episode_steps = 250
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


@register_task("maniskill.peg_insertion_side_363", "peg_insertion_side_363")
class PegInsertionSide363Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )

    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-363_v2.pkl"


@register_task("maniskill.peg_insertion_side_976", "peg_insertion_side_976")
class PegInsertionSide976Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-976_v2.pkl"


@register_task("maniskill.peg_insertion_side_458", "peg_insertion_side_458")
class PegInsertionSide458Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-458_v2.pkl"


@register_task("maniskill.peg_insertion_side_268", "peg_insertion_side_268")
class PegInsertionSide268Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )

    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-268_v2.pkl"


@register_task("maniskill.peg_insertion_side_419", "peg_insertion_side_419")
class PegInsertionSide419Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )

    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-419_v2.pkl"


@register_task("maniskill.peg_insertion_side_744", "peg_insertion_side_744")
class PegInsertionSide744Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )

    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-744_v2.pkl"


@register_task("maniskill.peg_insertion_side_461", "peg_insertion_side_461")
class PegInsertionSide461Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )

    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-461_v2.pkl"


@register_task("maniskill.peg_insertion_side_885", "peg_insertion_side_885")
class PegInsertionSide885Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )

    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-885_v2.pkl"


@register_task("maniskill.peg_insertion_side_249", "peg_insertion_side_249")
class PegInsertionSide249Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )

    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-249_v2.pkl"


@register_task("maniskill.peg_insertion_side_957", "peg_insertion_side_957")
class PegInsertionSide957Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )

    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-957_v2.pkl"


@register_task("maniskill.peg_insertion_side_18", "peg_insertion_side_18")
class PegInsertionSide18Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )

    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-18_v2.pkl"


@register_task("maniskill.peg_insertion_side_372", "peg_insertion_side_372")
class PegInsertionSide372Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )

    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-372_v2.pkl"


@register_task("maniskill.peg_insertion_side_473", "peg_insertion_side_473")
class PegInsertionSide473Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )

    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-473_v2.pkl"


@register_task("maniskill.peg_insertion_side_495", "peg_insertion_side_495")
class PegInsertionSide495Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )

    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-495_v2.pkl"


@register_task("maniskill.peg_insertion_side_557", "peg_insertion_side_557")
class PegInsertionSide557Cfg(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )

    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-557_v2.pkl"


@register_task("maniskill.peg_insertion_side_601", "peg_insertion_side_601")
class PegInsertionSide601Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )

    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-601_v2.pkl"


@register_task("maniskill.peg_insertion_side_170", "peg_insertion_side_170")
class PegInsertionSide170Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )

    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-170_v2.pkl"


@register_task("maniskill.peg_insertion_side_705", "peg_insertion_side_705")
class PegInsertionSide705Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )

    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-705_v2.pkl"


@register_task("maniskill.peg_insertion_side_683", "peg_insertion_side_683")
class PegInsertionSide683Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )

    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-683_v2.pkl"


@register_task("maniskill.peg_insertion_side_590", "peg_insertion_side_590")
class PegInsertionSide590Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )

    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-590_v2.pkl"


@register_task("maniskill.peg_insertion_side_263", "peg_insertion_side_263")
class PegInsertionSide263Cfg(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )

    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-263_v2.pkl"


@register_task("maniskill.peg_insertion_side_544", "peg_insertion_side_544")
class PegInsertionSide544Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )

    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-544_v2.pkl"


@register_task("maniskill.peg_insertion_side_476", "peg_insertion_side_476")
class PegInsertionSide476Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )

    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-476_v2.pkl"


@register_task("maniskill.peg_insertion_side_40", "peg_insertion_side_40")
class PegInsertionSide40Cfg(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )

    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-40_v2.pkl"


@register_task("maniskill.peg_insertion_side_227", "peg_insertion_side_227")
class PegInsertionSide227Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )

    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-227_v2.pkl"


@register_task("maniskill.peg_insertion_side_77", "peg_insertion_side_77")
class PegInsertionSide77Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )

    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-77_v2.pkl"


@register_task("maniskill.peg_insertion_side_471", "peg_insertion_side_471")
class PegInsertionSide471Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )

    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-471_v2.pkl"


@register_task("maniskill.peg_insertion_side_915", "peg_insertion_side_915")
class PegInsertionSide915Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )

    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-915_v2.pkl"


@register_task("maniskill.peg_insertion_side_122", "peg_insertion_side_122")
class PegInsertionSide122Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )

    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-122_v2.pkl"


@register_task("maniskill.peg_insertion_side_42", "peg_insertion_side_42")
class PegInsertionSide42Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )

    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-42_v2.pkl"


@register_task("maniskill.peg_insertion_side_216", "peg_insertion_side_216")
class PegInsertionSide216Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )

    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-216_v2.pkl"


@register_task("maniskill.peg_insertion_side_830", "peg_insertion_side_830")
class PegInsertionSide830Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )

    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-830_v2.pkl"


@register_task("maniskill.peg_insertion_side_609", "peg_insertion_side_609")
class PegInsertionSide609Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )

    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-609_v2.pkl"


@register_task("maniskill.peg_insertion_side_291", "peg_insertion_side_291")
class PegInsertionSide291Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )

    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-291_v2.pkl"


@register_task("maniskill.peg_insertion_side_277", "peg_insertion_side_277")
class PegInsertionSide277Cfg(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )

    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-277_v2.pkl"


@register_task("maniskill.peg_insertion_side_980", "peg_insertion_side_980")
class PegInsertionSide980Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )

    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-980_v2.pkl"


@register_task("maniskill.peg_insertion_side_504", "peg_insertion_side_504")
class PegInsertionSide504Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )

    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-504_v2.pkl"


@register_task("maniskill.peg_insertion_side_710", "peg_insertion_side_710")
class PegInsertionSide710Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )

    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-710_v2.pkl"


@register_task("maniskill.peg_insertion_side_490", "peg_insertion_side_490")
class PegInsertionSide490Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )

    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-490_v2.pkl"


@register_task("maniskill.peg_insertion_side_577", "peg_insertion_side_577")
class PegInsertionSide577Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )

    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-577_v2.pkl"


@register_task("maniskill.peg_insertion_side_378", "peg_insertion_side_378")
class PegInsertionSide378Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )

    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-378_v2.pkl"


@register_task("maniskill.peg_insertion_side_149", "peg_insertion_side_149")
class PegInsertionSide149Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )

    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-149_v2.pkl"


@register_task("maniskill.peg_insertion_side_187", "peg_insertion_side_187")
class PegInsertionSide187Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )

    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-187_v2.pkl"


@register_task("maniskill.peg_insertion_side_220", "peg_insertion_side_220")
class PegInsertionSide220Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )

    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-220_v2.pkl"


@register_task("maniskill.peg_insertion_side_304", "peg_insertion_side_304")
class PegInsertionSide304Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )

    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-304_v2.pkl"


@register_task("maniskill.peg_insertion_side_194", "peg_insertion_side_194")
class PegInsertionSide194Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )

    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-194_v2.pkl"


@register_task("maniskill.peg_insertion_side_997", "peg_insertion_side_997")
class PegInsertionSide997Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-997_v2.pkl"


@register_task("maniskill.peg_insertion_side_441", "peg_insertion_side_441")
class PegInsertionSide441Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-441_v2.pkl"


@register_task("maniskill.peg_insertion_side_563", "peg_insertion_side_563")
class PegInsertionSide563Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-563_v2.pkl"


@register_task("maniskill.peg_insertion_side_564", "peg_insertion_side_564")
class PegInsertionSide564Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-564_v2.pkl"


@register_task("maniskill.peg_insertion_side_450", "peg_insertion_side_450")
class PegInsertionSide450Cfg(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-450_v2.pkl"


@register_task("maniskill.peg_insertion_side_370", "peg_insertion_side_370")
class PegInsertionSide370Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-370_v2.pkl"


@register_task("maniskill.peg_insertion_side_243", "peg_insertion_side_243")
class PegInsertionSide243Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-243_v2.pkl"


@register_task("maniskill.peg_insertion_side_426", "peg_insertion_side_426")
class PegInsertionSide426Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-426_v2.pkl"


@register_task("maniskill.peg_insertion_side_58", "peg_insertion_side_58")
class PegInsertionSide58Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-58_v2.pkl"


@register_task("maniskill.peg_insertion_side_311", "peg_insertion_side_311")
class PegInsertionSide311Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-311_v2.pkl"


@register_task("maniskill.peg_insertion_side_92", "peg_insertion_side_92")
class PegInsertionSide92Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-92_v2.pkl"


@register_task("maniskill.peg_insertion_side_673", "peg_insertion_side_673")
class PegInsertionSide673Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-673_v2.pkl"


@register_task("maniskill.peg_insertion_side_494", "peg_insertion_side_494")
class PegInsertionSide494Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-494_v2.pkl"


@register_task("maniskill.peg_insertion_side_664", "peg_insertion_side_664")
class PegInsertionSide664Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-664_v2.pkl"


@register_task("maniskill.peg_insertion_side_825", "peg_insertion_side_825")
class PegInsertionSide825Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-825_v2.pkl"


@register_task("maniskill.peg_insertion_side_106", "peg_insertion_side_106")
class PegInsertionSide106Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-106_v2.pkl"


@register_task("maniskill.peg_insertion_side_199", "peg_insertion_side_199")
class PegInsertionSide199Cfg(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-199_v2.pkl"


@register_task("maniskill.peg_insertion_side_31", "peg_insertion_side_31")
class PegInsertionSide31Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-31_v2.pkl"


@register_task("maniskill.peg_insertion_side_574", "peg_insertion_side_574")
class PegInsertionSide574Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-574_v2.pkl"


@register_task("maniskill.peg_insertion_side_491", "peg_insertion_side_491")
class PegInsertionSide491Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-491_v2.pkl"


@register_task("maniskill.peg_insertion_side_914", "peg_insertion_side_914")
class PegInsertionSide914Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-914_v2.pkl"


@register_task("maniskill.peg_insertion_side_480", "peg_insertion_side_480")
class PegInsertionSide480Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-480_v2.pkl"


@register_task("maniskill.peg_insertion_side_283", "peg_insertion_side_283")
class PegInsertionSide283Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-283_v2.pkl"


@register_task("maniskill.peg_insertion_side_588", "peg_insertion_side_588")
class PegInsertionSide588Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-588_v2.pkl"


@register_task("maniskill.peg_insertion_side_375", "peg_insertion_side_375")
class PegInsertionSide375Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-375_v2.pkl"


@register_task("maniskill.peg_insertion_side_778", "peg_insertion_side_778")
class PegInsertionSide778Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-778_v2.pkl"


@register_task("maniskill.peg_insertion_side_361", "peg_insertion_side_361")
class PegInsertionSide361Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-361_v2.pkl"


@register_task("maniskill.peg_insertion_side_502", "peg_insertion_side_502")
class PegInsertionSide502Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-502_v2.pkl"


@register_task("maniskill.peg_insertion_side_196", "peg_insertion_side_196")
class PegInsertionSide196Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-196_v2.pkl"


@register_task("maniskill.peg_insertion_side_652", "peg_insertion_side_652")
class PegInsertionSide652Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-652_v2.pkl"


@register_task("maniskill.peg_insertion_side_169", "peg_insertion_side_169")
class PegInsertionSide169Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-169_v2.pkl"


@register_task("maniskill.peg_insertion_side_120", "peg_insertion_side_120")
class PegInsertionSide120Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-120_v2.pkl"


@register_task("maniskill.peg_insertion_side_302", "peg_insertion_side_302")
class PegInsertionSide302Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-302_v2.pkl"


@register_task("maniskill.peg_insertion_side_966", "peg_insertion_side_966")
class PegInsertionSide966Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-966_v2.pkl"


@register_task("maniskill.peg_insertion_side_562", "peg_insertion_side_562")
class PegInsertionSide562Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-562_v2.pkl"


@register_task("maniskill.peg_insertion_side_136", "peg_insertion_side_136")
class PegInsertionSide136Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-136_v2.pkl"


@register_task("maniskill.peg_insertion_side_126", "peg_insertion_side_126")
class PegInsertionSide126Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-126_v2.pkl"


@register_task("maniskill.peg_insertion_side_603", "peg_insertion_side_603")
class PegInsertionSide603Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-603_v2.pkl"


@register_task("maniskill.peg_insertion_side_153", "peg_insertion_side_153")
class PegInsertionSide153Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-153_v2.pkl"


@register_task("maniskill.peg_insertion_side_405", "peg_insertion_side_405")
class PegInsertionSide405Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-405_v2.pkl"


@register_task("maniskill.peg_insertion_side_486", "peg_insertion_side_486")
class PegInsertionSide486Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-486_v2.pkl"


@register_task("maniskill.peg_insertion_side_167", "peg_insertion_side_167")
class PegInsertionSide167Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-167_v2.pkl"


@register_task("maniskill.peg_insertion_side_177", "peg_insertion_side_177")
class PegInsertionSide177Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-177_v2.pkl"


@register_task("maniskill.peg_insertion_side_907", "peg_insertion_side_907")
class PegInsertionSide907Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-907_v2.pkl"


@register_task("maniskill.peg_insertion_side_454", "peg_insertion_side_454")
class PegInsertionSide454Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-454_v2.pkl"


@register_task("maniskill.peg_insertion_side_390", "peg_insertion_side_390")
class PegInsertionSide390Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-390_v2.pkl"


@register_task("maniskill.peg_insertion_side_67", "peg_insertion_side_67")
class PegInsertionSide67Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-67_v2.pkl"


@register_task("maniskill.peg_insertion_side_422", "peg_insertion_side_422")
class PegInsertionSide422Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-422_v2.pkl"


@register_task("maniskill.peg_insertion_side_904", "peg_insertion_side_904")
class PegInsertionSide904Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-904_v2.pkl"


@register_task("maniskill.peg_insertion_side_139", "peg_insertion_side_139")
class PegInsertionSide139Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-139_v2.pkl"


@register_task("maniskill.peg_insertion_side_894", "peg_insertion_side_894")
class PegInsertionSide894Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-894_v2.pkl"


@register_task("maniskill.peg_insertion_side_856", "peg_insertion_side_856")
class PegInsertionSide856Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-856_v2.pkl"


@register_task("maniskill.peg_insertion_side_558", "peg_insertion_side_558")
class PegInsertionSide558Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-558_v2.pkl"


@register_task("maniskill.peg_insertion_side_517", "peg_insertion_side_517")
class PegInsertionSide517Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-517_v2.pkl"


@register_task("maniskill.peg_insertion_side_532", "peg_insertion_side_532")
class PegInsertionSide532Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-532_v2.pkl"


@register_task("maniskill.peg_insertion_side_668", "peg_insertion_side_668")
class PegInsertionSide668Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-668_v2.pkl"


@register_task("maniskill.peg_insertion_side_847", "peg_insertion_side_847")
class PegInsertionSide847Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-847_v2.pkl"


@register_task("maniskill.peg_insertion_side_937", "peg_insertion_side_937")
class PegInsertionSide937Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-937_v2.pkl"


@register_task("maniskill.peg_insertion_side_217", "peg_insertion_side_217")
class PegInsertionSide217Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-217_v2.pkl"


@register_task("maniskill.peg_insertion_side_926", "peg_insertion_side_926")
class PegInsertionSide926Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-926_v2.pkl"


@register_task("maniskill.peg_insertion_side_414", "peg_insertion_side_414")
class PegInsertionSide414Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-414_v2.pkl"


@register_task("maniskill.peg_insertion_side_852", "peg_insertion_side_852")
class PegInsertionSide852Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-852_v2.pkl"


@register_task("maniskill.peg_insertion_side_210", "peg_insertion_side_210")
class PegInsertionSide210Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-210_v2.pkl"


@register_task("maniskill.peg_insertion_side_981", "peg_insertion_side_981")
class PegInsertionSide981Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-981_v2.pkl"


@register_task("maniskill.peg_insertion_side_135", "peg_insertion_side_135")
class PegInsertionSide135Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-135_v2.pkl"


@register_task("maniskill.peg_insertion_side_351", "peg_insertion_side_351")
class PegInsertionSide351Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-351_v2.pkl"


@register_task("maniskill.peg_insertion_side_462", "peg_insertion_side_462")
class PegInsertionSide462Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-462_v2.pkl"


@register_task("maniskill.peg_insertion_side_699", "peg_insertion_side_699")
class PegInsertionSide699Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-699_v2.pkl"


@register_task("maniskill.peg_insertion_side_152", "peg_insertion_side_152")
class PegInsertionSide152Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-152_v2.pkl"


@register_task("maniskill.peg_insertion_side_665", "peg_insertion_side_665")
class PegInsertionSide665Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-665_v2.pkl"


@register_task("maniskill.peg_insertion_side_855", "peg_insertion_side_855")
class PegInsertionSide855Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-855_v2.pkl"


@register_task("maniskill.peg_insertion_side_500", "peg_insertion_side_500")
class PegInsertionSide500Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-500_v2.pkl"


@register_task("maniskill.peg_insertion_side_692", "peg_insertion_side_692")
class PegInsertionSide692Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-692_v2.pkl"


@register_task("maniskill.peg_insertion_side_246", "peg_insertion_side_246")
class PegInsertionSide246Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-246_v2.pkl"


@register_task("maniskill.peg_insertion_side_162", "peg_insertion_side_162")
class PegInsertionSide162Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-162_v2.pkl"


@register_task("maniskill.peg_insertion_side_7", "peg_insertion_side_7")
class PegInsertionSide7Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-7_v2.pkl"


@register_task("maniskill.peg_insertion_side_159", "peg_insertion_side_159")
class PegInsertionSide159Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-159_v2.pkl"


@register_task("maniskill.peg_insertion_side_171", "peg_insertion_side_171")
class PegInsertionSide171Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-171_v2.pkl"


@register_task("maniskill.peg_insertion_side_848", "peg_insertion_side_848")
class PegInsertionSide848Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-848_v2.pkl"


@register_task("maniskill.peg_insertion_side_138", "peg_insertion_side_138")
class PegInsertionSide138Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-138_v2.pkl"


@register_task("maniskill.peg_insertion_side_523", "peg_insertion_side_523")
class PegInsertionSide523Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-523_v2.pkl"


@register_task("maniskill.peg_insertion_side_96", "peg_insertion_side_96")
class PegInsertionSide96Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-96_v2.pkl"


@register_task("maniskill.peg_insertion_side_784", "peg_insertion_side_784")
class PegInsertionSide784Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-784_v2.pkl"


@register_task("maniskill.peg_insertion_side_677", "peg_insertion_side_677")
class PegInsertionSide677Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-677_v2.pkl"


@register_task("maniskill.peg_insertion_side_951", "peg_insertion_side_951")
class PegInsertionSide951Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-951_v2.pkl"


@register_task("maniskill.peg_insertion_side_413", "peg_insertion_side_413")
class PegInsertionSide413Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-413_v2.pkl"


@register_task("maniskill.peg_insertion_side_691", "peg_insertion_side_691")
class PegInsertionSide691Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-691_v2.pkl"


@register_task("maniskill.peg_insertion_side_916", "peg_insertion_side_916")
class PegInsertionSide916Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-916_v2.pkl"


@register_task("maniskill.peg_insertion_side_266", "peg_insertion_side_266")
class PegInsertionSide266Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-266_v2.pkl"


@register_task("maniskill.peg_insertion_side_925", "peg_insertion_side_925")
class PegInsertionSide925Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-925_v2.pkl"


@register_task("maniskill.peg_insertion_side_29", "peg_insertion_side_29")
class PegInsertionSide29Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-29_v2.pkl"


@register_task("maniskill.peg_insertion_side_73", "peg_insertion_side_73")
class PegInsertionSide73Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-73_v2.pkl"


@register_task("maniskill.peg_insertion_side_44", "peg_insertion_side_44")
class PegInsertionSide44Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-44_v2.pkl"


@register_task("maniskill.peg_insertion_side_913", "peg_insertion_side_913")
class PegInsertionSide913Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-913_v2.pkl"


@register_task("maniskill.peg_insertion_side_575", "peg_insertion_side_575")
class PegInsertionSide575Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-575_v2.pkl"


@register_task("maniskill.peg_insertion_side_342", "peg_insertion_side_342")
class PegInsertionSide342Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-342_v2.pkl"


@register_task("maniskill.peg_insertion_side_658", "peg_insertion_side_658")
class PegInsertionSide658Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-658_v2.pkl"


@register_task("maniskill.peg_insertion_side_611", "peg_insertion_side_611")
class PegInsertionSide611Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-611_v2.pkl"


@register_task("maniskill.peg_insertion_side_437", "peg_insertion_side_437")
class PegInsertionSide437Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-437_v2.pkl"


@register_task("maniskill.peg_insertion_side_191", "peg_insertion_side_191")
class PegInsertionSide191Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-191_v2.pkl"


@register_task("maniskill.peg_insertion_side_506", "peg_insertion_side_506")
class PegInsertionSide506Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-506_v2.pkl"


@register_task("maniskill.peg_insertion_side_213", "peg_insertion_side_213")
class PegInsertionSide213Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-213_v2.pkl"


@register_task("maniskill.peg_insertion_side_824", "peg_insertion_side_824")
class PegInsertionSide824Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-824_v2.pkl"


@register_task("maniskill.peg_insertion_side_85", "peg_insertion_side_85")
class PegInsertionSide85Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-85_v2.pkl"


@register_task("maniskill.peg_insertion_side_547", "peg_insertion_side_547")
class PegInsertionSide547Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-547_v2.pkl"


@register_task("maniskill.peg_insertion_side_654", "peg_insertion_side_654")
class PegInsertionSide654Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-654_v2.pkl"


@register_task("maniskill.peg_insertion_side_218", "peg_insertion_side_218")
class PegInsertionSide218Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-218_v2.pkl"


@register_task("maniskill.peg_insertion_side_902", "peg_insertion_side_902")
class PegInsertionSide902Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-902_v2.pkl"


@register_task("maniskill.peg_insertion_side_337", "peg_insertion_side_337")
class PegInsertionSide337Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-337_v2.pkl"


@register_task("maniskill.peg_insertion_side_674", "peg_insertion_side_674")
class PegInsertionSide674Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-674_v2.pkl"


@register_task("maniskill.peg_insertion_side_546", "peg_insertion_side_546")
class PegInsertionSide546Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-546_v2.pkl"


@register_task("maniskill.peg_insertion_side_146", "peg_insertion_side_146")
class PegInsertionSide146Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-146_v2.pkl"


@register_task("maniskill.peg_insertion_side_145", "peg_insertion_side_145")
class PegInsertionSide145Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-145_v2.pkl"


@register_task("maniskill.peg_insertion_side_893", "peg_insertion_side_893")
class PegInsertionSide893Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-893_v2.pkl"


@register_task("maniskill.peg_insertion_side_616", "peg_insertion_side_616")
class PegInsertionSide616Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-616_v2.pkl"


@register_task("maniskill.peg_insertion_side_891", "peg_insertion_side_891")
class PegInsertionSide891Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-891_v2.pkl"


@register_task("maniskill.peg_insertion_side_795", "peg_insertion_side_795")
class PegInsertionSide795Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-795_v2.pkl"


@register_task("maniskill.peg_insertion_side_68", "peg_insertion_side_68")
class PegInsertionSide68Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-68_v2.pkl"


@register_task("maniskill.peg_insertion_side_207", "peg_insertion_side_207")
class PegInsertionSide207Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-207_v2.pkl"


@register_task("maniskill.peg_insertion_side_610", "peg_insertion_side_610")
class PegInsertionSide610Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-610_v2.pkl"


@register_task("maniskill.peg_insertion_side_972", "peg_insertion_side_972")
class PegInsertionSide972Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-972_v2.pkl"


@register_task("maniskill.peg_insertion_side_870", "peg_insertion_side_870")
class PegInsertionSide870Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-870_v2.pkl"


@register_task("maniskill.peg_insertion_side_301", "peg_insertion_side_301")
class PegInsertionSide301Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-301_v2.pkl"


@register_task("maniskill.peg_insertion_side_226", "peg_insertion_side_226")
class PegInsertionSide226Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-226_v2.pkl"


@register_task("maniskill.peg_insertion_side_785", "peg_insertion_side_785")
class PegInsertionSide785Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-785_v2.pkl"


@register_task("maniskill.peg_insertion_side_513", "peg_insertion_side_513")
class PegInsertionSide513Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-513_v2.pkl"


@register_task("maniskill.peg_insertion_side_154", "peg_insertion_side_154")
class PegInsertionSide154Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-154_v2.pkl"


@register_task("maniskill.peg_insertion_side_804", "peg_insertion_side_804")
class PegInsertionSide804Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-804_v2.pkl"


@register_task("maniskill.peg_insertion_side_764", "peg_insertion_side_764")
class PegInsertionSide764Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-764_v2.pkl"


@register_task("maniskill.peg_insertion_side_938", "peg_insertion_side_938")
class PegInsertionSide938Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-938_v2.pkl"


@register_task("maniskill.peg_insertion_side_822", "peg_insertion_side_822")
class PegInsertionSide822Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-822_v2.pkl"


@register_task("maniskill.peg_insertion_side_223", "peg_insertion_side_223")
class PegInsertionSide223Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-223_v2.pkl"


@register_task("maniskill.peg_insertion_side_978", "peg_insertion_side_978")
class PegInsertionSide978Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-978_v2.pkl"


@register_task("maniskill.peg_insertion_side_359", "peg_insertion_side_359")
class PegInsertionSide359Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-359_v2.pkl"


@register_task("maniskill.peg_insertion_side_551", "peg_insertion_side_551")
class PegInsertionSide551Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-551_v2.pkl"


@register_task("maniskill.peg_insertion_side_46", "peg_insertion_side_46")
class PegInsertionSide46Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-46_v2.pkl"


@register_task("maniskill.peg_insertion_side_983", "peg_insertion_side_983")
class PegInsertionSide983Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-983_v2.pkl"


@register_task("maniskill.peg_insertion_side_66", "peg_insertion_side_66")
class PegInsertionSide66Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-66_v2.pkl"


@register_task("maniskill.peg_insertion_side_469", "peg_insertion_side_469")
class PegInsertionSide469Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-469_v2.pkl"


@register_task("maniskill.peg_insertion_side_436", "peg_insertion_side_436")
class PegInsertionSide436Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-436_v2.pkl"


@register_task("maniskill.peg_insertion_side_933", "peg_insertion_side_933")
class PegInsertionSide933Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-933_v2.pkl"


@register_task("maniskill.peg_insertion_side_130", "peg_insertion_side_130")
class PegInsertionSide130Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-130_v2.pkl"


@register_task("maniskill.peg_insertion_side_765", "peg_insertion_side_765")
class PegInsertionSide765Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-765_v2.pkl"


@register_task("maniskill.peg_insertion_side_329", "peg_insertion_side_329")
class PegInsertionSide329Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-329_v2.pkl"


@register_task("maniskill.peg_insertion_side_686", "peg_insertion_side_686")
class PegInsertionSide686Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-686_v2.pkl"


@register_task("maniskill.peg_insertion_side_179", "peg_insertion_side_179")
class PegInsertionSide179Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-179_v2.pkl"


@register_task("maniskill.peg_insertion_side_357", "peg_insertion_side_357")
class PegInsertionSide357Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-357_v2.pkl"


@register_task("maniskill.peg_insertion_side_742", "peg_insertion_side_742")
class PegInsertionSide742Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-742_v2.pkl"


@register_task("maniskill.peg_insertion_side_322", "peg_insertion_side_322")
class PegInsertionSide322Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-322_v2.pkl"


@register_task("maniskill.peg_insertion_side_531", "peg_insertion_side_531")
class PegInsertionSide531Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-531_v2.pkl"


@register_task("maniskill.peg_insertion_side_688", "peg_insertion_side_688")
class PegInsertionSide688Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-688_v2.pkl"


@register_task("maniskill.peg_insertion_side_725", "peg_insertion_side_725")
class PegInsertionSide725Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-725_v2.pkl"


@register_task("maniskill.peg_insertion_side_713", "peg_insertion_side_713")
class PegInsertionSide713Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-713_v2.pkl"


@register_task("maniskill.peg_insertion_side_369", "peg_insertion_side_369")
class PegInsertionSide369Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-369_v2.pkl"


@register_task("maniskill.peg_insertion_side_90", "peg_insertion_side_90")
class PegInsertionSide90Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-90_v2.pkl"


@register_task("maniskill.peg_insertion_side_381", "peg_insertion_side_381")
class PegInsertionSide381Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-381_v2.pkl"


@register_task("maniskill.peg_insertion_side_211", "peg_insertion_side_211")
class PegInsertionSide211Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-211_v2.pkl"


@register_task("maniskill.peg_insertion_side_594", "peg_insertion_side_594")
class PegInsertionSide594Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-594_v2.pkl"


@register_task("maniskill.peg_insertion_side_384", "peg_insertion_side_384")
class PegInsertionSide384Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-384_v2.pkl"


@register_task("maniskill.peg_insertion_side_195", "peg_insertion_side_195")
class PegInsertionSide195Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-195_v2.pkl"


@register_task("maniskill.peg_insertion_side_964", "peg_insertion_side_964")
class PegInsertionSide964Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-964_v2.pkl"


@register_task("maniskill.peg_insertion_side_289", "peg_insertion_side_289")
class PegInsertionSide289Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-289_v2.pkl"


@register_task("maniskill.peg_insertion_side_873", "peg_insertion_side_873")
class PegInsertionSide873Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-873_v2.pkl"


@register_task("maniskill.peg_insertion_side_892", "peg_insertion_side_892")
class PegInsertionSide892Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-892_v2.pkl"


@register_task("maniskill.peg_insertion_side_445", "peg_insertion_side_445")
class PegInsertionSide445Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-445_v2.pkl"


@register_task("maniskill.peg_insertion_side_189", "peg_insertion_side_189")
class PegInsertionSide189Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-189_v2.pkl"


@register_task("maniskill.peg_insertion_side_219", "peg_insertion_side_219")
class PegInsertionSide219Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-219_v2.pkl"


@register_task("maniskill.peg_insertion_side_368", "peg_insertion_side_368")
class PegInsertionSide368Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-368_v2.pkl"


@register_task("maniskill.peg_insertion_side_457", "peg_insertion_side_457")
class PegInsertionSide457Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-457_v2.pkl"


@register_task("maniskill.peg_insertion_side_57", "peg_insertion_side_57")
class PegInsertionSide57Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-57_v2.pkl"


@register_task("maniskill.peg_insertion_side_939", "peg_insertion_side_939")
class PegInsertionSide939Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-939_v2.pkl"


@register_task("maniskill.peg_insertion_side_102", "peg_insertion_side_102")
class PegInsertionSide102Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-102_v2.pkl"


@register_task("maniskill.peg_insertion_side_49", "peg_insertion_side_49")
class PegInsertionSide49Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-49_v2.pkl"


@register_task("maniskill.peg_insertion_side_982", "peg_insertion_side_982")
class PegInsertionSide982Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-982_v2.pkl"


@register_task("maniskill.peg_insertion_side_716", "peg_insertion_side_716")
class PegInsertionSide716Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-716_v2.pkl"


@register_task("maniskill.peg_insertion_side_791", "peg_insertion_side_791")
class PegInsertionSide791Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-791_v2.pkl"


@register_task("maniskill.peg_insertion_side_832", "peg_insertion_side_832")
class PegInsertionSide832Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-832_v2.pkl"


@register_task("maniskill.peg_insertion_side_895", "peg_insertion_side_895")
class PegInsertionSide895Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-895_v2.pkl"


@register_task("maniskill.peg_insertion_side_897", "peg_insertion_side_897")
class PegInsertionSide897Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-897_v2.pkl"


@register_task("maniskill.peg_insertion_side_95", "peg_insertion_side_95")
class PegInsertionSide95Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-95_v2.pkl"


@register_task("maniskill.peg_insertion_side_607", "peg_insertion_side_607")
class PegInsertionSide607Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-607_v2.pkl"


@register_task("maniskill.peg_insertion_side_912", "peg_insertion_side_912")
class PegInsertionSide912Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-912_v2.pkl"


@register_task("maniskill.peg_insertion_side_183", "peg_insertion_side_183")
class PegInsertionSide183Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-183_v2.pkl"


@register_task("maniskill.peg_insertion_side_560", "peg_insertion_side_560")
class PegInsertionSide560Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-560_v2.pkl"


@register_task("maniskill.peg_insertion_side_116", "peg_insertion_side_116")
class PegInsertionSide116Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-116_v2.pkl"


@register_task("maniskill.peg_insertion_side_796", "peg_insertion_side_796")
class PegInsertionSide796Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-796_v2.pkl"


@register_task("maniskill.peg_insertion_side_201", "peg_insertion_side_201")
class PegInsertionSide201Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-201_v2.pkl"


@register_task("maniskill.peg_insertion_side_539", "peg_insertion_side_539")
class PegInsertionSide539Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-539_v2.pkl"


@register_task("maniskill.peg_insertion_side_151", "peg_insertion_side_151")
class PegInsertionSide151Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-151_v2.pkl"


@register_task("maniskill.peg_insertion_side_477", "peg_insertion_side_477")
class PegInsertionSide477Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-477_v2.pkl"


@register_task("maniskill.peg_insertion_side_928", "peg_insertion_side_928")
class PegInsertionSide928Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-928_v2.pkl"


@register_task("maniskill.peg_insertion_side_879", "peg_insertion_side_879")
class PegInsertionSide879Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-879_v2.pkl"


@register_task("maniskill.peg_insertion_side_520", "peg_insertion_side_520")
class PegInsertionSide520Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-520_v2.pkl"


@register_task("maniskill.peg_insertion_side_354", "peg_insertion_side_354")
class PegInsertionSide354Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-354_v2.pkl"


@register_task("maniskill.peg_insertion_side_335", "peg_insertion_side_335")
class PegInsertionSide335Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-335_v2.pkl"


@register_task("maniskill.peg_insertion_side_595", "peg_insertion_side_595")
class PegInsertionSide595Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-595_v2.pkl"


@register_task("maniskill.peg_insertion_side_806", "peg_insertion_side_806")
class PegInsertionSide806Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-806_v2.pkl"


@register_task("maniskill.peg_insertion_side_140", "peg_insertion_side_140")
class PegInsertionSide140Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-140_v2.pkl"


@register_task("maniskill.peg_insertion_side_43", "peg_insertion_side_43")
class PegInsertionSide43Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-43_v2.pkl"


@register_task("maniskill.peg_insertion_side_729", "peg_insertion_side_729")
class PegInsertionSide729Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-729_v2.pkl"


@register_task("maniskill.peg_insertion_side_671", "peg_insertion_side_671")
class PegInsertionSide671Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-671_v2.pkl"


@register_task("maniskill.peg_insertion_side_814", "peg_insertion_side_814")
class PegInsertionSide814Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-814_v2.pkl"


@register_task("maniskill.peg_insertion_side_503", "peg_insertion_side_503")
class PegInsertionSide503Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-503_v2.pkl"


@register_task("maniskill.peg_insertion_side_452", "peg_insertion_side_452")
class PegInsertionSide452Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-452_v2.pkl"


@register_task("maniskill.peg_insertion_side_844", "peg_insertion_side_844")
class PegInsertionSide844Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-844_v2.pkl"


@register_task("maniskill.peg_insertion_side_161", "peg_insertion_side_161")
class PegInsertionSide161Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-161_v2.pkl"


@register_task("maniskill.peg_insertion_side_394", "peg_insertion_side_394")
class PegInsertionSide394Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-394_v2.pkl"


@register_task("maniskill.peg_insertion_side_343", "peg_insertion_side_343")
class PegInsertionSide343Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-343_v2.pkl"


@register_task("maniskill.peg_insertion_side_878", "peg_insertion_side_878")
class PegInsertionSide878Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-878_v2.pkl"


@register_task("maniskill.peg_insertion_side_882", "peg_insertion_side_882")
class PegInsertionSide882Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-882_v2.pkl"


@register_task("maniskill.peg_insertion_side_815", "peg_insertion_side_815")
class PegInsertionSide815Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-815_v2.pkl"


@register_task("maniskill.peg_insertion_side_945", "peg_insertion_side_945")
class PegInsertionSide945Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-945_v2.pkl"


@register_task("maniskill.peg_insertion_side_703", "peg_insertion_side_703")
class PegInsertionSide703Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-703_v2.pkl"


@register_task("maniskill.peg_insertion_side_818", "peg_insertion_side_818")
class PegInsertionSide818Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-818_v2.pkl"


@register_task("maniskill.peg_insertion_side_451", "peg_insertion_side_451")
class PegInsertionSide451Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-451_v2.pkl"


@register_task("maniskill.peg_insertion_side_969", "peg_insertion_side_969")
class PegInsertionSide969Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-969_v2.pkl"


@register_task("maniskill.peg_insertion_side_47", "peg_insertion_side_47")
class PegInsertionSide47Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-47_v2.pkl"


@register_task("maniskill.peg_insertion_side_320", "peg_insertion_side_320")
class PegInsertionSide320Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-320_v2.pkl"


@register_task("maniskill.peg_insertion_side_467", "peg_insertion_side_467")
class PegInsertionSide467Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-467_v2.pkl"


@register_task("maniskill.peg_insertion_side_632", "peg_insertion_side_632")
class PegInsertionSide632Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-632_v2.pkl"


@register_task("maniskill.peg_insertion_side_954", "peg_insertion_side_954")
class PegInsertionSide954Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-954_v2.pkl"


@register_task("maniskill.peg_insertion_side_947", "peg_insertion_side_947")
class PegInsertionSide947Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-947_v2.pkl"


@register_task("maniskill.peg_insertion_side_485", "peg_insertion_side_485")
class PegInsertionSide485Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-485_v2.pkl"


@register_task("maniskill.peg_insertion_side_579", "peg_insertion_side_579")
class PegInsertionSide579Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-579_v2.pkl"


@register_task("maniskill.peg_insertion_side_474", "peg_insertion_side_474")
class PegInsertionSide474Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-474_v2.pkl"


@register_task("maniskill.peg_insertion_side_775", "peg_insertion_side_775")
class PegInsertionSide775Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-775_v2.pkl"


@register_task("maniskill.peg_insertion_side_294", "peg_insertion_side_294")
class PegInsertionSide294Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-294_v2.pkl"


@register_task("maniskill.peg_insertion_side_963", "peg_insertion_side_963")
class PegInsertionSide963Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-963_v2.pkl"


@register_task("maniskill.peg_insertion_side_858", "peg_insertion_side_858")
class PegInsertionSide858Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-858_v2.pkl"


@register_task("maniskill.peg_insertion_side_760", "peg_insertion_side_760")
class PegInsertionSide760Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-760_v2.pkl"


@register_task("maniskill.peg_insertion_side_942", "peg_insertion_side_942")
class PegInsertionSide942Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-942_v2.pkl"


@register_task("maniskill.peg_insertion_side_316", "peg_insertion_side_316")
class PegInsertionSide316Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-316_v2.pkl"


@register_task("maniskill.peg_insertion_side_331", "peg_insertion_side_331")
class PegInsertionSide331Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-331_v2.pkl"


@register_task("maniskill.peg_insertion_side_753", "peg_insertion_side_753")
class PegInsertionSide753Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-753_v2.pkl"


@register_task("maniskill.peg_insertion_side_783", "peg_insertion_side_783")
class PegInsertionSide783Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-783_v2.pkl"


@register_task("maniskill.peg_insertion_side_528", "peg_insertion_side_528")
class PegInsertionSide528Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-528_v2.pkl"


@register_task("maniskill.peg_insertion_side_566", "peg_insertion_side_566")
class PegInsertionSide566Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-566_v2.pkl"


@register_task("maniskill.peg_insertion_side_84", "peg_insertion_side_84")
class PegInsertionSide84Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-84_v2.pkl"


@register_task("maniskill.peg_insertion_side_255", "peg_insertion_side_255")
class PegInsertionSide255Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-255_v2.pkl"


@register_task("maniskill.peg_insertion_side_344", "peg_insertion_side_344")
class PegInsertionSide344Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-344_v2.pkl"


@register_task("maniskill.peg_insertion_side_507", "peg_insertion_side_507")
class PegInsertionSide507Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-507_v2.pkl"


@register_task("maniskill.peg_insertion_side_412", "peg_insertion_side_412")
class PegInsertionSide412Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-412_v2.pkl"


@register_task("maniskill.peg_insertion_side_559", "peg_insertion_side_559")
class PegInsertionSide559Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-559_v2.pkl"


@register_task("maniskill.peg_insertion_side_247", "peg_insertion_side_247")
class PegInsertionSide247Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-247_v2.pkl"


@register_task("maniskill.peg_insertion_side_3", "peg_insertion_side_3")
class PegInsertionSide3Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-3_v2.pkl"


@register_task("maniskill.peg_insertion_side_935", "peg_insertion_side_935")
class PegInsertionSide935Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-935_v2.pkl"


@register_task("maniskill.peg_insertion_side_941", "peg_insertion_side_941")
class PegInsertionSide941Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-941_v2.pkl"


@register_task("maniskill.peg_insertion_side_896", "peg_insertion_side_896")
class PegInsertionSide896Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-896_v2.pkl"


@register_task("maniskill.peg_insertion_side_543", "peg_insertion_side_543")
class PegInsertionSide543Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-543_v2.pkl"


@register_task("maniskill.peg_insertion_side_0", "peg_insertion_side_0")
class PegInsertionSide0Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-0_v2.pkl"


@register_task("maniskill.peg_insertion_side_222", "peg_insertion_side_222")
class PegInsertionSide222Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-222_v2.pkl"


@register_task("maniskill.peg_insertion_side_137", "peg_insertion_side_137")
class PegInsertionSide137Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-137_v2.pkl"


@register_task("maniskill.peg_insertion_side_842", "peg_insertion_side_842")
class PegInsertionSide842Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-842_v2.pkl"


@register_task("maniskill.peg_insertion_side_379", "peg_insertion_side_379")
class PegInsertionSide379Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-379_v2.pkl"


@register_task("maniskill.peg_insertion_side_62", "peg_insertion_side_62")
class PegInsertionSide62Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-62_v2.pkl"


@register_task("maniskill.peg_insertion_side_955", "peg_insertion_side_955")
class PegInsertionSide955Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-955_v2.pkl"


@register_task("maniskill.peg_insertion_side_440", "peg_insertion_side_440")
class PegInsertionSide440Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-440_v2.pkl"


@register_task("maniskill.peg_insertion_side_240", "peg_insertion_side_240")
class PegInsertionSide240Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-240_v2.pkl"


@register_task("maniskill.peg_insertion_side_883", "peg_insertion_side_883")
class PegInsertionSide883Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-883_v2.pkl"


@register_task("maniskill.peg_insertion_side_585", "peg_insertion_side_585")
class PegInsertionSide585Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-585_v2.pkl"


@register_task("maniskill.peg_insertion_side_860", "peg_insertion_side_860")
class PegInsertionSide860Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-860_v2.pkl"


@register_task("maniskill.peg_insertion_side_684", "peg_insertion_side_684")
class PegInsertionSide684Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-684_v2.pkl"


@register_task("maniskill.peg_insertion_side_510", "peg_insertion_side_510")
class PegInsertionSide510Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-510_v2.pkl"


@register_task("maniskill.peg_insertion_side_694", "peg_insertion_side_694")
class PegInsertionSide694Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-694_v2.pkl"


@register_task("maniskill.peg_insertion_side_113", "peg_insertion_side_113")
class PegInsertionSide113Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-113_v2.pkl"


@register_task("maniskill.peg_insertion_side_323", "peg_insertion_side_323")
class PegInsertionSide323Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-323_v2.pkl"


@register_task("maniskill.peg_insertion_side_862", "peg_insertion_side_862")
class PegInsertionSide862Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-862_v2.pkl"


@register_task("maniskill.peg_insertion_side_239", "peg_insertion_side_239")
class PegInsertionSide239Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-239_v2.pkl"


@register_task("maniskill.peg_insertion_side_111", "peg_insertion_side_111")
class PegInsertionSide111Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-111_v2.pkl"


@register_task("maniskill.peg_insertion_side_33", "peg_insertion_side_33")
class PegInsertionSide33Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-33_v2.pkl"


@register_task("maniskill.peg_insertion_side_448", "peg_insertion_side_448")
class PegInsertionSide448Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-448_v2.pkl"


@register_task("maniskill.peg_insertion_side_994", "peg_insertion_side_994")
class PegInsertionSide994Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-994_v2.pkl"


@register_task("maniskill.peg_insertion_side_333", "peg_insertion_side_333")
class PegInsertionSide333Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-333_v2.pkl"


@register_task("maniskill.peg_insertion_side_127", "peg_insertion_side_127")
class PegInsertionSide127Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-127_v2.pkl"


@register_task("maniskill.peg_insertion_side_54", "peg_insertion_side_54")
class PegInsertionSide54Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-54_v2.pkl"


@register_task("maniskill.peg_insertion_side_576", "peg_insertion_side_576")
class PegInsertionSide576Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-576_v2.pkl"


@register_task("maniskill.peg_insertion_side_583", "peg_insertion_side_583")
class PegInsertionSide583Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-583_v2.pkl"


@register_task("maniskill.peg_insertion_side_900", "peg_insertion_side_900")
class PegInsertionSide900Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-900_v2.pkl"


@register_task("maniskill.peg_insertion_side_91", "peg_insertion_side_91")
class PegInsertionSide91Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-91_v2.pkl"


@register_task("maniskill.peg_insertion_side_992", "peg_insertion_side_992")
class PegInsertionSide992Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-992_v2.pkl"


@register_task("maniskill.peg_insertion_side_374", "peg_insertion_side_374")
class PegInsertionSide374Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-374_v2.pkl"


@register_task("maniskill.peg_insertion_side_846", "peg_insertion_side_846")
class PegInsertionSide846Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-846_v2.pkl"


@register_task("maniskill.peg_insertion_side_990", "peg_insertion_side_990")
class PegInsertionSide990Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-990_v2.pkl"


@register_task("maniskill.peg_insertion_side_516", "peg_insertion_side_516")
class PegInsertionSide516Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-516_v2.pkl"


@register_task("maniskill.peg_insertion_side_875", "peg_insertion_side_875")
class PegInsertionSide875Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-875_v2.pkl"


@register_task("maniskill.peg_insertion_side_676", "peg_insertion_side_676")
class PegInsertionSide676Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-676_v2.pkl"


@register_task("maniskill.peg_insertion_side_423", "peg_insertion_side_423")
class PegInsertionSide423Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-423_v2.pkl"


@register_task("maniskill.peg_insertion_side_297", "peg_insertion_side_297")
class PegInsertionSide297Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-297_v2.pkl"


@register_task("maniskill.peg_insertion_side_269", "peg_insertion_side_269")
class PegInsertionSide269Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-269_v2.pkl"


@register_task("maniskill.peg_insertion_side_158", "peg_insertion_side_158")
class PegInsertionSide158Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-158_v2.pkl"


@register_task("maniskill.peg_insertion_side_110", "peg_insertion_side_110")
class PegInsertionSide110Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-110_v2.pkl"


@register_task("maniskill.peg_insertion_side_286", "peg_insertion_side_286")
class PegInsertionSide286Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-286_v2.pkl"


@register_task("maniskill.peg_insertion_side_636", "peg_insertion_side_636")
class PegInsertionSide636Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-636_v2.pkl"


@register_task("maniskill.peg_insertion_side_176", "peg_insertion_side_176")
class PegInsertionSide176Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-176_v2.pkl"


@register_task("maniskill.peg_insertion_side_144", "peg_insertion_side_144")
class PegInsertionSide144Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-144_v2.pkl"


@register_task("maniskill.peg_insertion_side_142", "peg_insertion_side_142")
class PegInsertionSide142Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-142_v2.pkl"


@register_task("maniskill.peg_insertion_side_640", "peg_insertion_side_640")
class PegInsertionSide640Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-640_v2.pkl"


@register_task("maniskill.peg_insertion_side_773", "peg_insertion_side_773")
class PegInsertionSide773Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-773_v2.pkl"


@register_task("maniskill.peg_insertion_side_732", "peg_insertion_side_732")
class PegInsertionSide732Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-732_v2.pkl"


@register_task("maniskill.peg_insertion_side_919", "peg_insertion_side_919")
class PegInsertionSide919Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-919_v2.pkl"


@register_task("maniskill.peg_insertion_side_619", "peg_insertion_side_619")
class PegInsertionSide619Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-619_v2.pkl"


@register_task("maniskill.peg_insertion_side_349", "peg_insertion_side_349")
class PegInsertionSide349Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-349_v2.pkl"


@register_task("maniskill.peg_insertion_side_373", "peg_insertion_side_373")
class PegInsertionSide373Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-373_v2.pkl"


@register_task("maniskill.peg_insertion_side_133", "peg_insertion_side_133")
class PegInsertionSide133Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-133_v2.pkl"


@register_task("maniskill.peg_insertion_side_622", "peg_insertion_side_622")
class PegInsertionSide622Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-622_v2.pkl"


@register_task("maniskill.peg_insertion_side_69", "peg_insertion_side_69")
class PegInsertionSide69Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-69_v2.pkl"


@register_task("maniskill.peg_insertion_side_173", "peg_insertion_side_173")
class PegInsertionSide173Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-173_v2.pkl"


@register_task("maniskill.peg_insertion_side_861", "peg_insertion_side_861")
class PegInsertionSide861Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-861_v2.pkl"


@register_task("maniskill.peg_insertion_side_572", "peg_insertion_side_572")
class PegInsertionSide572Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-572_v2.pkl"


@register_task("maniskill.peg_insertion_side_697", "peg_insertion_side_697")
class PegInsertionSide697Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-697_v2.pkl"


@register_task("maniskill.peg_insertion_side_180", "peg_insertion_side_180")
class PegInsertionSide180Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-180_v2.pkl"


@register_task("maniskill.peg_insertion_side_290", "peg_insertion_side_290")
class PegInsertionSide290Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-290_v2.pkl"


@register_task("maniskill.peg_insertion_side_910", "peg_insertion_side_910")
class PegInsertionSide910Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-910_v2.pkl"


@register_task("maniskill.peg_insertion_side_604", "peg_insertion_side_604")
class PegInsertionSide604Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-604_v2.pkl"


@register_task("maniskill.peg_insertion_side_488", "peg_insertion_side_488")
class PegInsertionSide488Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-488_v2.pkl"


@register_task("maniskill.peg_insertion_side_237", "peg_insertion_side_237")
class PegInsertionSide237Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-237_v2.pkl"


@register_task("maniskill.peg_insertion_side_94", "peg_insertion_side_94")
class PegInsertionSide94Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-94_v2.pkl"


@register_task("maniskill.peg_insertion_side_525", "peg_insertion_side_525")
class PegInsertionSide525Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-525_v2.pkl"


@register_task("maniskill.peg_insertion_side_927", "peg_insertion_side_927")
class PegInsertionSide927Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-927_v2.pkl"


@register_task("maniskill.peg_insertion_side_353", "peg_insertion_side_353")
class PegInsertionSide353Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-353_v2.pkl"


@register_task("maniskill.peg_insertion_side_752", "peg_insertion_side_752")
class PegInsertionSide752Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-752_v2.pkl"


@register_task("maniskill.peg_insertion_side_550", "peg_insertion_side_550")
class PegInsertionSide550Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-550_v2.pkl"


@register_task("maniskill.peg_insertion_side_535", "peg_insertion_side_535")
class PegInsertionSide535Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-535_v2.pkl"


@register_task("maniskill.peg_insertion_side_769", "peg_insertion_side_769")
class PegInsertionSide769Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-769_v2.pkl"


@register_task("maniskill.peg_insertion_side_184", "peg_insertion_side_184")
class PegInsertionSide184Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-184_v2.pkl"


@register_task("maniskill.peg_insertion_side_292", "peg_insertion_side_292")
class PegInsertionSide292Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-292_v2.pkl"


@register_task("maniskill.peg_insertion_side_687", "peg_insertion_side_687")
class PegInsertionSide687Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-687_v2.pkl"


@register_task("maniskill.peg_insertion_side_618", "peg_insertion_side_618")
class PegInsertionSide618Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-618_v2.pkl"


@register_task("maniskill.peg_insertion_side_132", "peg_insertion_side_132")
class PegInsertionSide132Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-132_v2.pkl"


@register_task("maniskill.peg_insertion_side_511", "peg_insertion_side_511")
class PegInsertionSide511Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-511_v2.pkl"


@register_task("maniskill.peg_insertion_side_800", "peg_insertion_side_800")
class PegInsertionSide800Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-800_v2.pkl"


@register_task("maniskill.peg_insertion_side_975", "peg_insertion_side_975")
class PegInsertionSide975Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-975_v2.pkl"


@register_task("maniskill.peg_insertion_side_663", "peg_insertion_side_663")
class PegInsertionSide663Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-663_v2.pkl"


@register_task("maniskill.peg_insertion_side_456", "peg_insertion_side_456")
class PegInsertionSide456Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-456_v2.pkl"


@register_task("maniskill.peg_insertion_side_392", "peg_insertion_side_392")
class PegInsertionSide392Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-392_v2.pkl"


@register_task("maniskill.peg_insertion_side_871", "peg_insertion_side_871")
class PegInsertionSide871Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-871_v2.pkl"


@register_task("maniskill.peg_insertion_side_833", "peg_insertion_side_833")
class PegInsertionSide833Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-833_v2.pkl"


@register_task("maniskill.peg_insertion_side_163", "peg_insertion_side_163")
class PegInsertionSide163Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-163_v2.pkl"


@register_task("maniskill.peg_insertion_side_306", "peg_insertion_side_306")
class PegInsertionSide306Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-306_v2.pkl"


@register_task("maniskill.peg_insertion_side_740", "peg_insertion_side_740")
class PegInsertionSide740Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-740_v2.pkl"


@register_task("maniskill.peg_insertion_side_816", "peg_insertion_side_816")
class PegInsertionSide816Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-816_v2.pkl"


@register_task("maniskill.peg_insertion_side_430", "peg_insertion_side_430")
class PegInsertionSide430Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-430_v2.pkl"


@register_task("maniskill.peg_insertion_side_968", "peg_insertion_side_968")
class PegInsertionSide968Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-968_v2.pkl"


@register_task("maniskill.peg_insertion_side_38", "peg_insertion_side_38")
class PegInsertionSide38Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-38_v2.pkl"


@register_task("maniskill.peg_insertion_side_656", "peg_insertion_side_656")
class PegInsertionSide656Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-656_v2.pkl"


@register_task("maniskill.peg_insertion_side_751", "peg_insertion_side_751")
class PegInsertionSide751Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-751_v2.pkl"


@register_task("maniskill.peg_insertion_side_630", "peg_insertion_side_630")
class PegInsertionSide630Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-630_v2.pkl"


@register_task("maniskill.peg_insertion_side_401", "peg_insertion_side_401")
class PegInsertionSide401Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-401_v2.pkl"


@register_task("maniskill.peg_insertion_side_726", "peg_insertion_side_726")
class PegInsertionSide726Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-726_v2.pkl"


@register_task("maniskill.peg_insertion_side_155", "peg_insertion_side_155")
class PegInsertionSide155Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-155_v2.pkl"


@register_task("maniskill.peg_insertion_side_360", "peg_insertion_side_360")
class PegInsertionSide360Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-360_v2.pkl"


@register_task("maniskill.peg_insertion_side_807", "peg_insertion_side_807")
class PegInsertionSide807Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-807_v2.pkl"


@register_task("maniskill.peg_insertion_side_11", "peg_insertion_side_11")
class PegInsertionSide11Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-11_v2.pkl"


@register_task("maniskill.peg_insertion_side_738", "peg_insertion_side_738")
class PegInsertionSide738Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-738_v2.pkl"


@register_task("maniskill.peg_insertion_side_906", "peg_insertion_side_906")
class PegInsertionSide906Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-906_v2.pkl"


@register_task("maniskill.peg_insertion_side_690", "peg_insertion_side_690")
class PegInsertionSide690Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-690_v2.pkl"


@register_task("maniskill.peg_insertion_side_888", "peg_insertion_side_888")
class PegInsertionSide888Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-888_v2.pkl"


@register_task("maniskill.peg_insertion_side_330", "peg_insertion_side_330")
class PegInsertionSide330Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-330_v2.pkl"


@register_task("maniskill.peg_insertion_side_708", "peg_insertion_side_708")
class PegInsertionSide708Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-708_v2.pkl"


@register_task("maniskill.peg_insertion_side_459", "peg_insertion_side_459")
class PegInsertionSide459Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-459_v2.pkl"


@register_task("maniskill.peg_insertion_side_287", "peg_insertion_side_287")
class PegInsertionSide287Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-287_v2.pkl"


@register_task("maniskill.peg_insertion_side_984", "peg_insertion_side_984")
class PegInsertionSide984Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-984_v2.pkl"


@register_task("maniskill.peg_insertion_side_917", "peg_insertion_side_917")
class PegInsertionSide917Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-917_v2.pkl"


@register_task("maniskill.peg_insertion_side_717", "peg_insertion_side_717")
class PegInsertionSide717Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-717_v2.pkl"


@register_task("maniskill.peg_insertion_side_720", "peg_insertion_side_720")
class PegInsertionSide720Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-720_v2.pkl"


@register_task("maniskill.peg_insertion_side_481", "peg_insertion_side_481")
class PegInsertionSide481Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-481_v2.pkl"


@register_task("maniskill.peg_insertion_side_114", "peg_insertion_side_114")
class PegInsertionSide114Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-114_v2.pkl"


@register_task("maniskill.peg_insertion_side_295", "peg_insertion_side_295")
class PegInsertionSide295Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-295_v2.pkl"


@register_task("maniskill.peg_insertion_side_884", "peg_insertion_side_884")
class PegInsertionSide884Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-884_v2.pkl"


@register_task("maniskill.peg_insertion_side_435", "peg_insertion_side_435")
class PegInsertionSide435Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-435_v2.pkl"


@register_task("maniskill.peg_insertion_side_837", "peg_insertion_side_837")
class PegInsertionSide837Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-837_v2.pkl"


@register_task("maniskill.peg_insertion_side_613", "peg_insertion_side_613")
class PegInsertionSide613Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-613_v2.pkl"


@register_task("maniskill.peg_insertion_side_86", "peg_insertion_side_86")
class PegInsertionSide86Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-86_v2.pkl"


@register_task("maniskill.peg_insertion_side_10", "peg_insertion_side_10")
class PegInsertionSide10Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-10_v2.pkl"


@register_task("maniskill.peg_insertion_side_489", "peg_insertion_side_489")
class PegInsertionSide489Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-489_v2.pkl"


@register_task("maniskill.peg_insertion_side_63", "peg_insertion_side_63")
class PegInsertionSide63Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-63_v2.pkl"


@register_task("maniskill.peg_insertion_side_131", "peg_insertion_side_131")
class PegInsertionSide131Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-131_v2.pkl"


@register_task("maniskill.peg_insertion_side_208", "peg_insertion_side_208")
class PegInsertionSide208Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-208_v2.pkl"


@register_task("maniskill.peg_insertion_side_803", "peg_insertion_side_803")
class PegInsertionSide803Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-803_v2.pkl"


@register_task("maniskill.peg_insertion_side_647", "peg_insertion_side_647")
class PegInsertionSide647Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-647_v2.pkl"


@register_task("maniskill.peg_insertion_side_398", "peg_insertion_side_398")
class PegInsertionSide398Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-398_v2.pkl"


@register_task("maniskill.peg_insertion_side_164", "peg_insertion_side_164")
class PegInsertionSide164Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-164_v2.pkl"


@register_task("maniskill.peg_insertion_side_105", "peg_insertion_side_105")
class PegInsertionSide105Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-105_v2.pkl"


@register_task("maniskill.peg_insertion_side_962", "peg_insertion_side_962")
class PegInsertionSide962Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-962_v2.pkl"


@register_task("maniskill.peg_insertion_side_252", "peg_insertion_side_252")
class PegInsertionSide252Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-252_v2.pkl"


@register_task("maniskill.peg_insertion_side_801", "peg_insertion_side_801")
class PegInsertionSide801Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-801_v2.pkl"


@register_task("maniskill.peg_insertion_side_881", "peg_insertion_side_881")
class PegInsertionSide881Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-881_v2.pkl"


@register_task("maniskill.peg_insertion_side_388", "peg_insertion_side_388")
class PegInsertionSide388Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-388_v2.pkl"


@register_task("maniskill.peg_insertion_side_646", "peg_insertion_side_646")
class PegInsertionSide646Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-646_v2.pkl"


@register_task("maniskill.peg_insertion_side_735", "peg_insertion_side_735")
class PegInsertionSide735Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-735_v2.pkl"


@register_task("maniskill.peg_insertion_side_898", "peg_insertion_side_898")
class PegInsertionSide898Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-898_v2.pkl"


@register_task("maniskill.peg_insertion_side_479", "peg_insertion_side_479")
class PegInsertionSide479Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-479_v2.pkl"


@register_task("maniskill.peg_insertion_side_293", "peg_insertion_side_293")
class PegInsertionSide293Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-293_v2.pkl"


@register_task("maniskill.peg_insertion_side_946", "peg_insertion_side_946")
class PegInsertionSide946Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-946_v2.pkl"


@register_task("maniskill.peg_insertion_side_756", "peg_insertion_side_756")
class PegInsertionSide756Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-756_v2.pkl"


@register_task("maniskill.peg_insertion_side_32", "peg_insertion_side_32")
class PegInsertionSide32Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-32_v2.pkl"


@register_task("maniskill.peg_insertion_side_28", "peg_insertion_side_28")
class PegInsertionSide28Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-28_v2.pkl"


@register_task("maniskill.peg_insertion_side_298", "peg_insertion_side_298")
class PegInsertionSide298Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-298_v2.pkl"


@register_task("maniskill.peg_insertion_side_591", "peg_insertion_side_591")
class PegInsertionSide591Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-591_v2.pkl"


@register_task("maniskill.peg_insertion_side_386", "peg_insertion_side_386")
class PegInsertionSide386Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-386_v2.pkl"


@register_task("maniskill.peg_insertion_side_780", "peg_insertion_side_780")
class PegInsertionSide780Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-780_v2.pkl"


@register_task("maniskill.peg_insertion_side_165", "peg_insertion_side_165")
class PegInsertionSide165Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-165_v2.pkl"


@register_task("maniskill.peg_insertion_side_273", "peg_insertion_side_273")
class PegInsertionSide273Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-273_v2.pkl"


@register_task("maniskill.peg_insertion_side_835", "peg_insertion_side_835")
class PegInsertionSide835Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-835_v2.pkl"


@register_task("maniskill.peg_insertion_side_463", "peg_insertion_side_463")
class PegInsertionSide463Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-463_v2.pkl"


@register_task("maniskill.peg_insertion_side_261", "peg_insertion_side_261")
class PegInsertionSide261Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-261_v2.pkl"


@register_task("maniskill.peg_insertion_side_442", "peg_insertion_side_442")
class PegInsertionSide442Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-442_v2.pkl"


@register_task("maniskill.peg_insertion_side_332", "peg_insertion_side_332")
class PegInsertionSide332Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-332_v2.pkl"


@register_task("maniskill.peg_insertion_side_13", "peg_insertion_side_13")
class PegInsertionSide13Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-13_v2.pkl"


@register_task("maniskill.peg_insertion_side_256", "peg_insertion_side_256")
class PegInsertionSide256Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-256_v2.pkl"


@register_task("maniskill.peg_insertion_side_899", "peg_insertion_side_899")
class PegInsertionSide899Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-899_v2.pkl"


@register_task("maniskill.peg_insertion_side_959", "peg_insertion_side_959")
class PegInsertionSide959Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-959_v2.pkl"


@register_task("maniskill.peg_insertion_side_107", "peg_insertion_side_107")
class PegInsertionSide107Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-107_v2.pkl"


@register_task("maniskill.peg_insertion_side_706", "peg_insertion_side_706")
class PegInsertionSide706Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-706_v2.pkl"


@register_task("maniskill.peg_insertion_side_758", "peg_insertion_side_758")
class PegInsertionSide758Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-758_v2.pkl"


@register_task("maniskill.peg_insertion_side_777", "peg_insertion_side_777")
class PegInsertionSide777Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-777_v2.pkl"


@register_task("maniskill.peg_insertion_side_23", "peg_insertion_side_23")
class PegInsertionSide23Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-23_v2.pkl"


@register_task("maniskill.peg_insertion_side_288", "peg_insertion_side_288")
class PegInsertionSide288Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-288_v2.pkl"


@register_task("maniskill.peg_insertion_side_829", "peg_insertion_side_829")
class PegInsertionSide829Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-829_v2.pkl"


@register_task("maniskill.peg_insertion_side_823", "peg_insertion_side_823")
class PegInsertionSide823Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-823_v2.pkl"


@register_task("maniskill.peg_insertion_side_267", "peg_insertion_side_267")
class PegInsertionSide267Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-267_v2.pkl"


@register_task("maniskill.peg_insertion_side_395", "peg_insertion_side_395")
class PegInsertionSide395Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-395_v2.pkl"


@register_task("maniskill.peg_insertion_side_411", "peg_insertion_side_411")
class PegInsertionSide411Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-411_v2.pkl"


@register_task("maniskill.peg_insertion_side_839", "peg_insertion_side_839")
class PegInsertionSide839Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-839_v2.pkl"


@register_task("maniskill.peg_insertion_side_518", "peg_insertion_side_518")
class PegInsertionSide518Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-518_v2.pkl"


@register_task("maniskill.peg_insertion_side_828", "peg_insertion_side_828")
class PegInsertionSide828Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-828_v2.pkl"


@register_task("maniskill.peg_insertion_side_197", "peg_insertion_side_197")
class PegInsertionSide197Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-197_v2.pkl"


@register_task("maniskill.peg_insertion_side_364", "peg_insertion_side_364")
class PegInsertionSide364Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-364_v2.pkl"


@register_task("maniskill.peg_insertion_side_406", "peg_insertion_side_406")
class PegInsertionSide406Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-406_v2.pkl"


@register_task("maniskill.peg_insertion_side_642", "peg_insertion_side_642")
class PegInsertionSide642Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-642_v2.pkl"


@register_task("maniskill.peg_insertion_side_83", "peg_insertion_side_83")
class PegInsertionSide83Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-83_v2.pkl"


@register_task("maniskill.peg_insertion_side_921", "peg_insertion_side_921")
class PegInsertionSide921Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-921_v2.pkl"


@register_task("maniskill.peg_insertion_side_877", "peg_insertion_side_877")
class PegInsertionSide877Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-877_v2.pkl"


@register_task("maniskill.peg_insertion_side_540", "peg_insertion_side_540")
class PegInsertionSide540Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-540_v2.pkl"


@register_task("maniskill.peg_insertion_side_698", "peg_insertion_side_698")
class PegInsertionSide698Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-698_v2.pkl"


@register_task("maniskill.peg_insertion_side_470", "peg_insertion_side_470")
class PegInsertionSide470Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-470_v2.pkl"


@register_task("maniskill.peg_insertion_side_356", "peg_insertion_side_356")
class PegInsertionSide356Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-356_v2.pkl"


@register_task("maniskill.peg_insertion_side_190", "peg_insertion_side_190")
class PegInsertionSide190Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-190_v2.pkl"


@register_task("maniskill.peg_insertion_side_160", "peg_insertion_side_160")
class PegInsertionSide160Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-160_v2.pkl"


@register_task("maniskill.peg_insertion_side_953", "peg_insertion_side_953")
class PegInsertionSide953Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-953_v2.pkl"


@register_task("maniskill.peg_insertion_side_657", "peg_insertion_side_657")
class PegInsertionSide657Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-657_v2.pkl"


@register_task("maniskill.peg_insertion_side_193", "peg_insertion_side_193")
class PegInsertionSide193Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-193_v2.pkl"


@register_task("maniskill.peg_insertion_side_987", "peg_insertion_side_987")
class PegInsertionSide987Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-987_v2.pkl"


@register_task("maniskill.peg_insertion_side_282", "peg_insertion_side_282")
class PegInsertionSide282Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-282_v2.pkl"


@register_task("maniskill.peg_insertion_side_112", "peg_insertion_side_112")
class PegInsertionSide112Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-112_v2.pkl"


@register_task("maniskill.peg_insertion_side_911", "peg_insertion_side_911")
class PegInsertionSide911Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-911_v2.pkl"


@register_task("maniskill.peg_insertion_side_759", "peg_insertion_side_759")
class PegInsertionSide759Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-759_v2.pkl"


@register_task("maniskill.peg_insertion_side_203", "peg_insertion_side_203")
class PegInsertionSide203Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-203_v2.pkl"


@register_task("maniskill.peg_insertion_side_76", "peg_insertion_side_76")
class PegInsertionSide76Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-76_v2.pkl"


@register_task("maniskill.peg_insertion_side_631", "peg_insertion_side_631")
class PegInsertionSide631Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-631_v2.pkl"


@register_task("maniskill.peg_insertion_side_944", "peg_insertion_side_944")
class PegInsertionSide944Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-944_v2.pkl"


@register_task("maniskill.peg_insertion_side_41", "peg_insertion_side_41")
class PegInsertionSide41Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-41_v2.pkl"


@register_task("maniskill.peg_insertion_side_232", "peg_insertion_side_232")
class PegInsertionSide232Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-232_v2.pkl"


@register_task("maniskill.peg_insertion_side_794", "peg_insertion_side_794")
class PegInsertionSide794Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-794_v2.pkl"


@register_task("maniskill.peg_insertion_side_859", "peg_insertion_side_859")
class PegInsertionSide859Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-859_v2.pkl"


@register_task("maniskill.peg_insertion_side_60", "peg_insertion_side_60")
class PegInsertionSide60Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-60_v2.pkl"


@register_task("maniskill.peg_insertion_side_78", "peg_insertion_side_78")
class PegInsertionSide78Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-78_v2.pkl"


@register_task("maniskill.peg_insertion_side_548", "peg_insertion_side_548")
class PegInsertionSide548Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-548_v2.pkl"


@register_task("maniskill.peg_insertion_side_150", "peg_insertion_side_150")
class PegInsertionSide150Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-150_v2.pkl"


@register_task("maniskill.peg_insertion_side_88", "peg_insertion_side_88")
class PegInsertionSide88Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-88_v2.pkl"


@register_task("maniskill.peg_insertion_side_221", "peg_insertion_side_221")
class PegInsertionSide221Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-221_v2.pkl"


@register_task("maniskill.peg_insertion_side_407", "peg_insertion_side_407")
class PegInsertionSide407Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-407_v2.pkl"


@register_task("maniskill.peg_insertion_side_988", "peg_insertion_side_988")
class PegInsertionSide988Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-988_v2.pkl"


@register_task("maniskill.peg_insertion_side_64", "peg_insertion_side_64")
class PegInsertionSide64Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-64_v2.pkl"


@register_task("maniskill.peg_insertion_side_355", "peg_insertion_side_355")
class PegInsertionSide355Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-355_v2.pkl"


@register_task("maniskill.peg_insertion_side_443", "peg_insertion_side_443")
class PegInsertionSide443Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-443_v2.pkl"


@register_task("maniskill.peg_insertion_side_56", "peg_insertion_side_56")
class PegInsertionSide56Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-56_v2.pkl"


@register_task("maniskill.peg_insertion_side_967", "peg_insertion_side_967")
class PegInsertionSide967Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-967_v2.pkl"


@register_task("maniskill.peg_insertion_side_258", "peg_insertion_side_258")
class PegInsertionSide258Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-258_v2.pkl"


@register_task("maniskill.peg_insertion_side_634", "peg_insertion_side_634")
class PegInsertionSide634Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-634_v2.pkl"


@register_task("maniskill.peg_insertion_side_866", "peg_insertion_side_866")
class PegInsertionSide866Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-866_v2.pkl"


@register_task("maniskill.peg_insertion_side_228", "peg_insertion_side_228")
class PegInsertionSide228Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-228_v2.pkl"


@register_task("maniskill.peg_insertion_side_397", "peg_insertion_side_397")
class PegInsertionSide397Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-397_v2.pkl"


@register_task("maniskill.peg_insertion_side_552", "peg_insertion_side_552")
class PegInsertionSide552Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-552_v2.pkl"


@register_task("maniskill.peg_insertion_side_168", "peg_insertion_side_168")
class PegInsertionSide168Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-168_v2.pkl"


@register_task("maniskill.peg_insertion_side_497", "peg_insertion_side_497")
class PegInsertionSide497Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-497_v2.pkl"


@register_task("maniskill.peg_insertion_side_362", "peg_insertion_side_362")
class PegInsertionSide362Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-362_v2.pkl"


@register_task("maniskill.peg_insertion_side_695", "peg_insertion_side_695")
class PegInsertionSide695Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-695_v2.pkl"


@register_task("maniskill.peg_insertion_side_857", "peg_insertion_side_857")
class PegInsertionSide857Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-857_v2.pkl"


@register_task("maniskill.peg_insertion_side_728", "peg_insertion_side_728")
class PegInsertionSide728Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-728_v2.pkl"


@register_task("maniskill.peg_insertion_side_59", "peg_insertion_side_59")
class PegInsertionSide59Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-59_v2.pkl"


@register_task("maniskill.peg_insertion_side_93", "peg_insertion_side_93")
class PegInsertionSide93Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-93_v2.pkl"


@register_task("maniskill.peg_insertion_side_129", "peg_insertion_side_129")
class PegInsertionSide129Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-129_v2.pkl"


@register_task("maniskill.peg_insertion_side_157", "peg_insertion_side_157")
class PegInsertionSide157Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-157_v2.pkl"


@register_task("maniskill.peg_insertion_side_338", "peg_insertion_side_338")
class PegInsertionSide338Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-338_v2.pkl"


@register_task("maniskill.peg_insertion_side_648", "peg_insertion_side_648")
class PegInsertionSide648Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-648_v2.pkl"


@register_task("maniskill.peg_insertion_side_693", "peg_insertion_side_693")
class PegInsertionSide693Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-693_v2.pkl"


@register_task("maniskill.peg_insertion_side_313", "peg_insertion_side_313")
class PegInsertionSide313Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-313_v2.pkl"


@register_task("maniskill.peg_insertion_side_743", "peg_insertion_side_743")
class PegInsertionSide743Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-743_v2.pkl"


@register_task("maniskill.peg_insertion_side_100", "peg_insertion_side_100")
class PegInsertionSide100Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-100_v2.pkl"


@register_task("maniskill.peg_insertion_side_45", "peg_insertion_side_45")
class PegInsertionSide45Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-45_v2.pkl"


@register_task("maniskill.peg_insertion_side_529", "peg_insertion_side_529")
class PegInsertionSide529Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-529_v2.pkl"


@register_task("maniskill.peg_insertion_side_812", "peg_insertion_side_812")
class PegInsertionSide812Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-812_v2.pkl"


@register_task("maniskill.peg_insertion_side_234", "peg_insertion_side_234")
class PegInsertionSide234Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-234_v2.pkl"


@register_task("maniskill.peg_insertion_side_421", "peg_insertion_side_421")
class PegInsertionSide421Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-421_v2.pkl"


@register_task("maniskill.peg_insertion_side_838", "peg_insertion_side_838")
class PegInsertionSide838Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-838_v2.pkl"


@register_task("maniskill.peg_insertion_side_30", "peg_insertion_side_30")
class PegInsertionSide30Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-30_v2.pkl"


@register_task("maniskill.peg_insertion_side_864", "peg_insertion_side_864")
class PegInsertionSide864Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-864_v2.pkl"


@register_task("maniskill.peg_insertion_side_484", "peg_insertion_side_484")
class PegInsertionSide484Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-484_v2.pkl"


@register_task("maniskill.peg_insertion_side_15", "peg_insertion_side_15")
class PegInsertionSide15Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-15_v2.pkl"


@register_task("maniskill.peg_insertion_side_521", "peg_insertion_side_521")
class PegInsertionSide521Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-521_v2.pkl"


@register_task("maniskill.peg_insertion_side_475", "peg_insertion_side_475")
class PegInsertionSide475Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-475_v2.pkl"


@register_task("maniskill.peg_insertion_side_918", "peg_insertion_side_918")
class PegInsertionSide918Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-918_v2.pkl"


@register_task("maniskill.peg_insertion_side_667", "peg_insertion_side_667")
class PegInsertionSide667Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-667_v2.pkl"


@register_task("maniskill.peg_insertion_side_580", "peg_insertion_side_580")
class PegInsertionSide580Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-580_v2.pkl"


@register_task("maniskill.peg_insertion_side_409", "peg_insertion_side_409")
class PegInsertionSide409Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-409_v2.pkl"


@register_task("maniskill.peg_insertion_side_453", "peg_insertion_side_453")
class PegInsertionSide453Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-453_v2.pkl"


@register_task("maniskill.peg_insertion_side_148", "peg_insertion_side_148")
class PegInsertionSide148Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-148_v2.pkl"


@register_task("maniskill.peg_insertion_side_556", "peg_insertion_side_556")
class PegInsertionSide556Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-556_v2.pkl"


@register_task("maniskill.peg_insertion_side_553", "peg_insertion_side_553")
class PegInsertionSide553Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-553_v2.pkl"


@register_task("maniskill.peg_insertion_side_626", "peg_insertion_side_626")
class PegInsertionSide626Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-626_v2.pkl"


@register_task("maniskill.peg_insertion_side_865", "peg_insertion_side_865")
class PegInsertionSide865Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-865_v2.pkl"


@register_task("maniskill.peg_insertion_side_961", "peg_insertion_side_961")
class PegInsertionSide961Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-961_v2.pkl"


@register_task("maniskill.peg_insertion_side_281", "peg_insertion_side_281")
class PegInsertionSide281Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-281_v2.pkl"


@register_task("maniskill.peg_insertion_side_874", "peg_insertion_side_874")
class PegInsertionSide874Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-874_v2.pkl"


@register_task("maniskill.peg_insertion_side_385", "peg_insertion_side_385")
class PegInsertionSide385Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-385_v2.pkl"


@register_task("maniskill.peg_insertion_side_496", "peg_insertion_side_496")
class PegInsertionSide496Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-496_v2.pkl"


@register_task("maniskill.peg_insertion_side_235", "peg_insertion_side_235")
class PegInsertionSide235Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-235_v2.pkl"


@register_task("maniskill.peg_insertion_side_308", "peg_insertion_side_308")
class PegInsertionSide308Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-308_v2.pkl"


@register_task("maniskill.peg_insertion_side_836", "peg_insertion_side_836")
class PegInsertionSide836Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-836_v2.pkl"


@register_task("maniskill.peg_insertion_side_582", "peg_insertion_side_582")
class PegInsertionSide582Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-582_v2.pkl"


@register_task("maniskill.peg_insertion_side_98", "peg_insertion_side_98")
class PegInsertionSide98Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-98_v2.pkl"


@register_task("maniskill.peg_insertion_side_383", "peg_insertion_side_383")
class PegInsertionSide383Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-383_v2.pkl"


@register_task("maniskill.peg_insertion_side_438", "peg_insertion_side_438")
class PegInsertionSide438Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-438_v2.pkl"


@register_task("maniskill.peg_insertion_side_714", "peg_insertion_side_714")
class PegInsertionSide714Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-714_v2.pkl"


@register_task("maniskill.peg_insertion_side_12", "peg_insertion_side_12")
class PegInsertionSide12Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-12_v2.pkl"


@register_task("maniskill.peg_insertion_side_99", "peg_insertion_side_99")
class PegInsertionSide99Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-99_v2.pkl"


@register_task("maniskill.peg_insertion_side_17", "peg_insertion_side_17")
class PegInsertionSide17Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-17_v2.pkl"


@register_task("maniskill.peg_insertion_side_124", "peg_insertion_side_124")
class PegInsertionSide124Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-124_v2.pkl"


@register_task("maniskill.peg_insertion_side_813", "peg_insertion_side_813")
class PegInsertionSide813Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-813_v2.pkl"


@register_task("maniskill.peg_insertion_side_317", "peg_insertion_side_317")
class PegInsertionSide317Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-317_v2.pkl"


@register_task("maniskill.peg_insertion_side_943", "peg_insertion_side_943")
class PegInsertionSide943Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-943_v2.pkl"


@register_task("maniskill.peg_insertion_side_231", "peg_insertion_side_231")
class PegInsertionSide231Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-231_v2.pkl"


@register_task("maniskill.peg_insertion_side_662", "peg_insertion_side_662")
class PegInsertionSide662Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-662_v2.pkl"


@register_task("maniskill.peg_insertion_side_89", "peg_insertion_side_89")
class PegInsertionSide89Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-89_v2.pkl"


@register_task("maniskill.peg_insertion_side_820", "peg_insertion_side_820")
class PegInsertionSide820Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-820_v2.pkl"


@register_task("maniskill.peg_insertion_side_934", "peg_insertion_side_934")
class PegInsertionSide934Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-934_v2.pkl"


@register_task("maniskill.peg_insertion_side_460", "peg_insertion_side_460")
class PegInsertionSide460Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-460_v2.pkl"


@register_task("maniskill.peg_insertion_side_296", "peg_insertion_side_296")
class PegInsertionSide296Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-296_v2.pkl"


@register_task("maniskill.peg_insertion_side_432", "peg_insertion_side_432")
class PegInsertionSide432Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-432_v2.pkl"


@register_task("maniskill.peg_insertion_side_447", "peg_insertion_side_447")
class PegInsertionSide447Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-447_v2.pkl"


@register_task("maniskill.peg_insertion_side_280", "peg_insertion_side_280")
class PegInsertionSide280Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-280_v2.pkl"


@register_task("maniskill.peg_insertion_side_417", "peg_insertion_side_417")
class PegInsertionSide417Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-417_v2.pkl"


@register_task("maniskill.peg_insertion_side_514", "peg_insertion_side_514")
class PegInsertionSide514Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-514_v2.pkl"


@register_task("maniskill.peg_insertion_side_175", "peg_insertion_side_175")
class PegInsertionSide175Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-175_v2.pkl"


@register_task("maniskill.peg_insertion_side_253", "peg_insertion_side_253")
class PegInsertionSide253Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-253_v2.pkl"


@register_task("maniskill.peg_insertion_side_242", "peg_insertion_side_242")
class PegInsertionSide242Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-242_v2.pkl"


@register_task("maniskill.peg_insertion_side_229", "peg_insertion_side_229")
class PegInsertionSide229Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-229_v2.pkl"


@register_task("maniskill.peg_insertion_side_985", "peg_insertion_side_985")
class PegInsertionSide985Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-985_v2.pkl"


@register_task("maniskill.peg_insertion_side_782", "peg_insertion_side_782")
class PegInsertionSide782Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-782_v2.pkl"


@register_task("maniskill.peg_insertion_side_790", "peg_insertion_side_790")
class PegInsertionSide790Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-790_v2.pkl"


@register_task("maniskill.peg_insertion_side_763", "peg_insertion_side_763")
class PegInsertionSide763Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-763_v2.pkl"


@register_task("maniskill.peg_insertion_side_334", "peg_insertion_side_334")
class PegInsertionSide334Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-334_v2.pkl"


@register_task("maniskill.peg_insertion_side_624", "peg_insertion_side_624")
class PegInsertionSide624Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-624_v2.pkl"


@register_task("maniskill.peg_insertion_side_766", "peg_insertion_side_766")
class PegInsertionSide766Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-766_v2.pkl"


@register_task("maniskill.peg_insertion_side_74", "peg_insertion_side_74")
class PegInsertionSide74Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-74_v2.pkl"


@register_task("maniskill.peg_insertion_side_787", "peg_insertion_side_787")
class PegInsertionSide787Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-787_v2.pkl"


@register_task("maniskill.peg_insertion_side_487", "peg_insertion_side_487")
class PegInsertionSide487Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-487_v2.pkl"


@register_task("maniskill.peg_insertion_side_749", "peg_insertion_side_749")
class PegInsertionSide749Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-749_v2.pkl"


@register_task("maniskill.peg_insertion_side_793", "peg_insertion_side_793")
class PegInsertionSide793Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-793_v2.pkl"


@register_task("maniskill.peg_insertion_side_404", "peg_insertion_side_404")
class PegInsertionSide404Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-404_v2.pkl"


@register_task("maniskill.peg_insertion_side_225", "peg_insertion_side_225")
class PegInsertionSide225Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-225_v2.pkl"


@register_task("maniskill.peg_insertion_side_434", "peg_insertion_side_434")
class PegInsertionSide434Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-434_v2.pkl"


@register_task("maniskill.peg_insertion_side_909", "peg_insertion_side_909")
class PegInsertionSide909Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-909_v2.pkl"


@register_task("maniskill.peg_insertion_side_715", "peg_insertion_side_715")
class PegInsertionSide715Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-715_v2.pkl"


@register_task("maniskill.peg_insertion_side_230", "peg_insertion_side_230")
class PegInsertionSide230Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-230_v2.pkl"


@register_task("maniskill.peg_insertion_side_809", "peg_insertion_side_809")
class PegInsertionSide809Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-809_v2.pkl"


@register_task("maniskill.peg_insertion_side_325", "peg_insertion_side_325")
class PegInsertionSide325Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-325_v2.pkl"


@register_task("maniskill.peg_insertion_side_36", "peg_insertion_side_36")
class PegInsertionSide36Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-36_v2.pkl"


@register_task("maniskill.peg_insertion_side_589", "peg_insertion_side_589")
class PegInsertionSide589Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-589_v2.pkl"


@register_task("maniskill.peg_insertion_side_204", "peg_insertion_side_204")
class PegInsertionSide204Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-204_v2.pkl"


@register_task("maniskill.peg_insertion_side_680", "peg_insertion_side_680")
class PegInsertionSide680Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-680_v2.pkl"


@register_task("maniskill.peg_insertion_side_746", "peg_insertion_side_746")
class PegInsertionSide746Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-746_v2.pkl"


@register_task("maniskill.peg_insertion_side_2", "peg_insertion_side_2")
class PegInsertionSide2Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-2_v2.pkl"


@register_task("maniskill.peg_insertion_side_259", "peg_insertion_side_259")
class PegInsertionSide259Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-259_v2.pkl"


@register_task("maniskill.peg_insertion_side_641", "peg_insertion_side_641")
class PegInsertionSide641Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-641_v2.pkl"


@register_task("maniskill.peg_insertion_side_285", "peg_insertion_side_285")
class PegInsertionSide285Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-285_v2.pkl"


@register_task("maniskill.peg_insertion_side_649", "peg_insertion_side_649")
class PegInsertionSide649Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-649_v2.pkl"


@register_task("maniskill.peg_insertion_side_251", "peg_insertion_side_251")
class PegInsertionSide251Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-251_v2.pkl"


@register_task("maniskill.peg_insertion_side_371", "peg_insertion_side_371")
class PegInsertionSide371Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-371_v2.pkl"


@register_task("maniskill.peg_insertion_side_675", "peg_insertion_side_675")
class PegInsertionSide675Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-675_v2.pkl"


@register_task("maniskill.peg_insertion_side_299", "peg_insertion_side_299")
class PegInsertionSide299Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-299_v2.pkl"


@register_task("maniskill.peg_insertion_side_755", "peg_insertion_side_755")
class PegInsertionSide755Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-755_v2.pkl"


@register_task("maniskill.peg_insertion_side_730", "peg_insertion_side_730")
class PegInsertionSide730Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-730_v2.pkl"


@register_task("maniskill.peg_insertion_side_819", "peg_insertion_side_819")
class PegInsertionSide819Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-819_v2.pkl"


@register_task("maniskill.peg_insertion_side_639", "peg_insertion_side_639")
class PegInsertionSide639Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-639_v2.pkl"


@register_task("maniskill.peg_insertion_side_387", "peg_insertion_side_387")
class PegInsertionSide387Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-387_v2.pkl"


@register_task("maniskill.peg_insertion_side_166", "peg_insertion_side_166")
class PegInsertionSide166Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-166_v2.pkl"


@register_task("maniskill.peg_insertion_side_747", "peg_insertion_side_747")
class PegInsertionSide747Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-747_v2.pkl"


@register_task("maniskill.peg_insertion_side_887", "peg_insertion_side_887")
class PegInsertionSide887Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-887_v2.pkl"


@register_task("maniskill.peg_insertion_side_416", "peg_insertion_side_416")
class PegInsertionSide416Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-416_v2.pkl"


@register_task("maniskill.peg_insertion_side_670", "peg_insertion_side_670")
class PegInsertionSide670Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-670_v2.pkl"


@register_task("maniskill.peg_insertion_side_841", "peg_insertion_side_841")
class PegInsertionSide841Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-841_v2.pkl"


@register_task("maniskill.peg_insertion_side_328", "peg_insertion_side_328")
class PegInsertionSide328Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-328_v2.pkl"


@register_task("maniskill.peg_insertion_side_644", "peg_insertion_side_644")
class PegInsertionSide644Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-644_v2.pkl"


@register_task("maniskill.peg_insertion_side_272", "peg_insertion_side_272")
class PegInsertionSide272Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-272_v2.pkl"


@register_task("maniskill.peg_insertion_side_719", "peg_insertion_side_719")
class PegInsertionSide719Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-719_v2.pkl"


@register_task("maniskill.peg_insertion_side_305", "peg_insertion_side_305")
class PegInsertionSide305Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-305_v2.pkl"


@register_task("maniskill.peg_insertion_side_950", "peg_insertion_side_950")
class PegInsertionSide950Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-950_v2.pkl"


@register_task("maniskill.peg_insertion_side_876", "peg_insertion_side_876")
class PegInsertionSide876Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-876_v2.pkl"


@register_task("maniskill.peg_insertion_side_593", "peg_insertion_side_593")
class PegInsertionSide593Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-593_v2.pkl"


@register_task("maniskill.peg_insertion_side_788", "peg_insertion_side_788")
class PegInsertionSide788Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-788_v2.pkl"


@register_task("maniskill.peg_insertion_side_431", "peg_insertion_side_431")
class PegInsertionSide431Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-431_v2.pkl"


@register_task("maniskill.peg_insertion_side_9", "peg_insertion_side_9")
class PegInsertionSide9Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-9_v2.pkl"


@register_task("maniskill.peg_insertion_side_628", "peg_insertion_side_628")
class PegInsertionSide628Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-628_v2.pkl"


@register_task("maniskill.peg_insertion_side_930", "peg_insertion_side_930")
class PegInsertionSide930Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-930_v2.pkl"


@register_task("maniskill.peg_insertion_side_472", "peg_insertion_side_472")
class PegInsertionSide472Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-472_v2.pkl"


@register_task("maniskill.peg_insertion_side_52", "peg_insertion_side_52")
class PegInsertionSide52Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-52_v2.pkl"


@register_task("maniskill.peg_insertion_side_125", "peg_insertion_side_125")
class PegInsertionSide125Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-125_v2.pkl"


@register_task("maniskill.peg_insertion_side_973", "peg_insertion_side_973")
class PegInsertionSide973Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-973_v2.pkl"


@register_task("maniskill.peg_insertion_side_637", "peg_insertion_side_637")
class PegInsertionSide637Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-637_v2.pkl"


@register_task("maniskill.peg_insertion_side_346", "peg_insertion_side_346")
class PegInsertionSide346Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-346_v2.pkl"


@register_task("maniskill.peg_insertion_side_932", "peg_insertion_side_932")
class PegInsertionSide932Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-932_v2.pkl"


@register_task("maniskill.peg_insertion_side_71", "peg_insertion_side_71")
class PegInsertionSide71Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-71_v2.pkl"


@register_task("maniskill.peg_insertion_side_653", "peg_insertion_side_653")
class PegInsertionSide653Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-653_v2.pkl"


@register_task("maniskill.peg_insertion_side_65", "peg_insertion_side_65")
class PegInsertionSide65Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-65_v2.pkl"


@register_task("maniskill.peg_insertion_side_779", "peg_insertion_side_779")
class PegInsertionSide779Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-779_v2.pkl"


@register_task("maniskill.peg_insertion_side_185", "peg_insertion_side_185")
class PegInsertionSide185Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-185_v2.pkl"


@register_task("maniskill.peg_insertion_side_172", "peg_insertion_side_172")
class PegInsertionSide172Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-172_v2.pkl"


@register_task("maniskill.peg_insertion_side_505", "peg_insertion_side_505")
class PegInsertionSide505Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-505_v2.pkl"


@register_task("maniskill.peg_insertion_side_82", "peg_insertion_side_82")
class PegInsertionSide82Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-82_v2.pkl"


@register_task("maniskill.peg_insertion_side_723", "peg_insertion_side_723")
class PegInsertionSide723Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-723_v2.pkl"


@register_task("maniskill.peg_insertion_side_811", "peg_insertion_side_811")
class PegInsertionSide811Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-811_v2.pkl"


@register_task("maniskill.peg_insertion_side_704", "peg_insertion_side_704")
class PegInsertionSide704Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-704_v2.pkl"


@register_task("maniskill.peg_insertion_side_568", "peg_insertion_side_568")
class PegInsertionSide568Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-568_v2.pkl"


@register_task("maniskill.peg_insertion_side_327", "peg_insertion_side_327")
class PegInsertionSide327Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-327_v2.pkl"


@register_task("maniskill.peg_insertion_side_97", "peg_insertion_side_97")
class PegInsertionSide97Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-97_v2.pkl"


@register_task("maniskill.peg_insertion_side_156", "peg_insertion_side_156")
class PegInsertionSide156Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-156_v2.pkl"


@register_task("maniskill.peg_insertion_side_770", "peg_insertion_side_770")
class PegInsertionSide770Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-770_v2.pkl"


@register_task("maniskill.peg_insertion_side_284", "peg_insertion_side_284")
class PegInsertionSide284Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-284_v2.pkl"


@register_task("maniskill.peg_insertion_side_672", "peg_insertion_side_672")
class PegInsertionSide672Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-672_v2.pkl"


@register_task("maniskill.peg_insertion_side_478", "peg_insertion_side_478")
class PegInsertionSide478Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-478_v2.pkl"


@register_task("maniskill.peg_insertion_side_178", "peg_insertion_side_178")
class PegInsertionSide178Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-178_v2.pkl"


@register_task("maniskill.peg_insertion_side_400", "peg_insertion_side_400")
class PegInsertionSide400Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-400_v2.pkl"


@register_task("maniskill.peg_insertion_side_428", "peg_insertion_side_428")
class PegInsertionSide428Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-428_v2.pkl"


@register_task("maniskill.peg_insertion_side_991", "peg_insertion_side_991")
class PegInsertionSide991Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-991_v2.pkl"


@register_task("maniskill.peg_insertion_side_561", "peg_insertion_side_561")
class PegInsertionSide561Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-561_v2.pkl"


@register_task("maniskill.peg_insertion_side_745", "peg_insertion_side_745")
class PegInsertionSide745Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-745_v2.pkl"


@register_task("maniskill.peg_insertion_side_198", "peg_insertion_side_198")
class PegInsertionSide198Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-198_v2.pkl"


@register_task("maniskill.peg_insertion_side_826", "peg_insertion_side_826")
class PegInsertionSide826Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-826_v2.pkl"


@register_task("maniskill.peg_insertion_side_625", "peg_insertion_side_625")
class PegInsertionSide625Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-625_v2.pkl"


@register_task("maniskill.peg_insertion_side_614", "peg_insertion_side_614")
class PegInsertionSide614Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-614_v2.pkl"


@register_task("maniskill.peg_insertion_side_712", "peg_insertion_side_712")
class PegInsertionSide712Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-712_v2.pkl"


@register_task("maniskill.peg_insertion_side_123", "peg_insertion_side_123")
class PegInsertionSide123Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-123_v2.pkl"


@register_task("maniskill.peg_insertion_side_236", "peg_insertion_side_236")
class PegInsertionSide236Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-236_v2.pkl"


@register_task("maniskill.peg_insertion_side_850", "peg_insertion_side_850")
class PegInsertionSide850Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-850_v2.pkl"


@register_task("maniskill.peg_insertion_side_276", "peg_insertion_side_276")
class PegInsertionSide276Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-276_v2.pkl"


@register_task("maniskill.peg_insertion_side_702", "peg_insertion_side_702")
class PegInsertionSide702Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-702_v2.pkl"


@register_task("maniskill.peg_insertion_side_402", "peg_insertion_side_402")
class PegInsertionSide402Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-402_v2.pkl"


@register_task("maniskill.peg_insertion_side_274", "peg_insertion_side_274")
class PegInsertionSide274Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-274_v2.pkl"


@register_task("maniskill.peg_insertion_side_53", "peg_insertion_side_53")
class PegInsertionSide53Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-53_v2.pkl"


@register_task("maniskill.peg_insertion_side_541", "peg_insertion_side_541")
class PegInsertionSide541Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-541_v2.pkl"


@register_task("maniskill.peg_insertion_side_799", "peg_insertion_side_799")
class PegInsertionSide799Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-799_v2.pkl"


@register_task("maniskill.peg_insertion_side_26", "peg_insertion_side_26")
class PegInsertionSide26Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-26_v2.pkl"


@register_task("maniskill.peg_insertion_side_206", "peg_insertion_side_206")
class PegInsertionSide206Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-206_v2.pkl"


@register_task("maniskill.peg_insertion_side_774", "peg_insertion_side_774")
class PegInsertionSide774Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-774_v2.pkl"


@register_task("maniskill.peg_insertion_side_399", "peg_insertion_side_399")
class PegInsertionSide399Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-399_v2.pkl"


@register_task("maniskill.peg_insertion_side_854", "peg_insertion_side_854")
class PegInsertionSide854Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-854_v2.pkl"


@register_task("maniskill.peg_insertion_side_834", "peg_insertion_side_834")
class PegInsertionSide834Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-834_v2.pkl"


@register_task("maniskill.peg_insertion_side_643", "peg_insertion_side_643")
class PegInsertionSide643Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-643_v2.pkl"


@register_task("maniskill.peg_insertion_side_265", "peg_insertion_side_265")
class PegInsertionSide265Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-265_v2.pkl"


@register_task("maniskill.peg_insertion_side_20", "peg_insertion_side_20")
class PegInsertionSide20Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-20_v2.pkl"


@register_task("maniskill.peg_insertion_side_103", "peg_insertion_side_103")
class PegInsertionSide103Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-103_v2.pkl"


@register_task("maniskill.peg_insertion_side_270", "peg_insertion_side_270")
class PegInsertionSide270Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-270_v2.pkl"


@register_task("maniskill.peg_insertion_side_224", "peg_insertion_side_224")
class PegInsertionSide224Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-224_v2.pkl"


@register_task("maniskill.peg_insertion_side_567", "peg_insertion_side_567")
class PegInsertionSide567Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-567_v2.pkl"


@register_task("maniskill.peg_insertion_side_455", "peg_insertion_side_455")
class PegInsertionSide455Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-455_v2.pkl"


@register_task("maniskill.peg_insertion_side_739", "peg_insertion_side_739")
class PegInsertionSide739Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-739_v2.pkl"


@register_task("maniskill.peg_insertion_side_498", "peg_insertion_side_498")
class PegInsertionSide498Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-498_v2.pkl"


@register_task("maniskill.peg_insertion_side_108", "peg_insertion_side_108")
class PegInsertionSide108Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-108_v2.pkl"


@register_task("maniskill.peg_insertion_side_711", "peg_insertion_side_711")
class PegInsertionSide711Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-711_v2.pkl"


@register_task("maniskill.peg_insertion_side_599", "peg_insertion_side_599")
class PegInsertionSide599Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-599_v2.pkl"


@register_task("maniskill.peg_insertion_side_669", "peg_insertion_side_669")
class PegInsertionSide669Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-669_v2.pkl"


@register_task("maniskill.peg_insertion_side_278", "peg_insertion_side_278")
class PegInsertionSide278Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-278_v2.pkl"


@register_task("maniskill.peg_insertion_side_908", "peg_insertion_side_908")
class PegInsertionSide908Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-908_v2.pkl"


@register_task("maniskill.peg_insertion_side_970", "peg_insertion_side_970")
class PegInsertionSide970Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-970_v2.pkl"


@register_task("maniskill.peg_insertion_side_482", "peg_insertion_side_482")
class PegInsertionSide482Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-482_v2.pkl"


@register_task("maniskill.peg_insertion_side_996", "peg_insertion_side_996")
class PegInsertionSide996Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-996_v2.pkl"


@register_task("maniskill.peg_insertion_side_901", "peg_insertion_side_901")
class PegInsertionSide901Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-901_v2.pkl"


@register_task("maniskill.peg_insertion_side_554", "peg_insertion_side_554")
class PegInsertionSide554Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-554_v2.pkl"


@register_task("maniskill.peg_insertion_side_118", "peg_insertion_side_118")
class PegInsertionSide118Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-118_v2.pkl"


@register_task("maniskill.peg_insertion_side_570", "peg_insertion_side_570")
class PegInsertionSide570Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-570_v2.pkl"


@register_task("maniskill.peg_insertion_side_817", "peg_insertion_side_817")
class PegInsertionSide817Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-817_v2.pkl"


@register_task("maniskill.peg_insertion_side_244", "peg_insertion_side_244")
class PegInsertionSide244Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-244_v2.pkl"


@register_task("maniskill.peg_insertion_side_638", "peg_insertion_side_638")
class PegInsertionSide638Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-638_v2.pkl"


@register_task("maniskill.peg_insertion_side_620", "peg_insertion_side_620")
class PegInsertionSide620Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-620_v2.pkl"


@register_task("maniskill.peg_insertion_side_449", "peg_insertion_side_449")
class PegInsertionSide449Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-449_v2.pkl"


@register_task("maniskill.peg_insertion_side_425", "peg_insertion_side_425")
class PegInsertionSide425Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-425_v2.pkl"


@register_task("maniskill.peg_insertion_side_924", "peg_insertion_side_924")
class PegInsertionSide924Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-924_v2.pkl"


@register_task("maniskill.peg_insertion_side_781", "peg_insertion_side_781")
class PegInsertionSide781Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-781_v2.pkl"


@register_task("maniskill.peg_insertion_side_205", "peg_insertion_side_205")
class PegInsertionSide205Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-205_v2.pkl"


@register_task("maniskill.peg_insertion_side_318", "peg_insertion_side_318")
class PegInsertionSide318Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-318_v2.pkl"


@register_task("maniskill.peg_insertion_side_326", "peg_insertion_side_326")
class PegInsertionSide326Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-326_v2.pkl"


@register_task("maniskill.peg_insertion_side_309", "peg_insertion_side_309")
class PegInsertionSide309Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-309_v2.pkl"


@register_task("maniskill.peg_insertion_side_721", "peg_insertion_side_721")
class PegInsertionSide721Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-721_v2.pkl"


@register_task("maniskill.peg_insertion_side_209", "peg_insertion_side_209")
class PegInsertionSide209Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-209_v2.pkl"


@register_task("maniskill.peg_insertion_side_960", "peg_insertion_side_960")
class PegInsertionSide960Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-960_v2.pkl"


@register_task("maniskill.peg_insertion_side_889", "peg_insertion_side_889")
class PegInsertionSide889Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-889_v2.pkl"


@register_task("maniskill.peg_insertion_side_347", "peg_insertion_side_347")
class PegInsertionSide347Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-347_v2.pkl"


@register_task("maniskill.peg_insertion_side_853", "peg_insertion_side_853")
class PegInsertionSide853Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-853_v2.pkl"


@register_task("maniskill.peg_insertion_side_827", "peg_insertion_side_827")
class PegInsertionSide827Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-827_v2.pkl"


@register_task("maniskill.peg_insertion_side_393", "peg_insertion_side_393")
class PegInsertionSide393Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-393_v2.pkl"


@register_task("maniskill.peg_insertion_side_651", "peg_insertion_side_651")
class PegInsertionSide651Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-651_v2.pkl"


@register_task("maniskill.peg_insertion_side_512", "peg_insertion_side_512")
class PegInsertionSide512Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-512_v2.pkl"


@register_task("maniskill.peg_insertion_side_186", "peg_insertion_side_186")
class PegInsertionSide186Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-186_v2.pkl"


@register_task("maniskill.peg_insertion_side_538", "peg_insertion_side_538")
class PegInsertionSide538Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-538_v2.pkl"


@register_task("maniskill.peg_insertion_side_48", "peg_insertion_side_48")
class PegInsertionSide48Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-48_v2.pkl"


@register_task("maniskill.peg_insertion_side_380", "peg_insertion_side_380")
class PegInsertionSide380Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-380_v2.pkl"


@register_task("maniskill.peg_insertion_side_573", "peg_insertion_side_573")
class PegInsertionSide573Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-573_v2.pkl"


@register_task("maniskill.peg_insertion_side_754", "peg_insertion_side_754")
class PegInsertionSide754Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-754_v2.pkl"


@register_task("maniskill.peg_insertion_side_681", "peg_insertion_side_681")
class PegInsertionSide681Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-681_v2.pkl"


@register_task("maniskill.peg_insertion_side_802", "peg_insertion_side_802")
class PegInsertionSide802Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-802_v2.pkl"


@register_task("maniskill.peg_insertion_side_920", "peg_insertion_side_920")
class PegInsertionSide920Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-920_v2.pkl"


@register_task("maniskill.peg_insertion_side_768", "peg_insertion_side_768")
class PegInsertionSide768Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-768_v2.pkl"


@register_task("maniskill.peg_insertion_side_666", "peg_insertion_side_666")
class PegInsertionSide666Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-666_v2.pkl"


@register_task("maniskill.peg_insertion_side_974", "peg_insertion_side_974")
class PegInsertionSide974Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-974_v2.pkl"


@register_task("maniskill.peg_insertion_side_352", "peg_insertion_side_352")
class PegInsertionSide352Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-352_v2.pkl"


@register_task("maniskill.peg_insertion_side_608", "peg_insertion_side_608")
class PegInsertionSide608Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-608_v2.pkl"


@register_task("maniskill.peg_insertion_side_633", "peg_insertion_side_633")
class PegInsertionSide633Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-633_v2.pkl"


@register_task("maniskill.peg_insertion_side_701", "peg_insertion_side_701")
class PegInsertionSide701Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-701_v2.pkl"


@register_task("maniskill.peg_insertion_side_923", "peg_insertion_side_923")
class PegInsertionSide923Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-923_v2.pkl"


@register_task("maniskill.peg_insertion_side_986", "peg_insertion_side_986")
class PegInsertionSide986Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-986_v2.pkl"


@register_task("maniskill.peg_insertion_side_215", "peg_insertion_side_215")
class PegInsertionSide215Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-215_v2.pkl"


@register_task("maniskill.peg_insertion_side_952", "peg_insertion_side_952")
class PegInsertionSide952Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-952_v2.pkl"


@register_task("maniskill.peg_insertion_side_733", "peg_insertion_side_733")
class PegInsertionSide733Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-733_v2.pkl"


@register_task("maniskill.peg_insertion_side_629", "peg_insertion_side_629")
class PegInsertionSide629Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-629_v2.pkl"


@register_task("maniskill.peg_insertion_side_722", "peg_insertion_side_722")
class PegInsertionSide722Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-722_v2.pkl"


@register_task("maniskill.peg_insertion_side_598", "peg_insertion_side_598")
class PegInsertionSide598Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-598_v2.pkl"


@register_task("maniskill.peg_insertion_side_709", "peg_insertion_side_709")
class PegInsertionSide709Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-709_v2.pkl"


@register_task("maniskill.peg_insertion_side_307", "peg_insertion_side_307")
class PegInsertionSide307Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-307_v2.pkl"


@register_task("maniskill.peg_insertion_side_660", "peg_insertion_side_660")
class PegInsertionSide660Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-660_v2.pkl"


@register_task("maniskill.peg_insertion_side_104", "peg_insertion_side_104")
class PegInsertionSide104Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-104_v2.pkl"


@register_task("maniskill.peg_insertion_side_427", "peg_insertion_side_427")
class PegInsertionSide427Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-427_v2.pkl"


@register_task("maniskill.peg_insertion_side_16", "peg_insertion_side_16")
class PegInsertionSide16Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-16_v2.pkl"


@register_task("maniskill.peg_insertion_side_797", "peg_insertion_side_797")
class PegInsertionSide797Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-797_v2.pkl"


@register_task("maniskill.peg_insertion_side_965", "peg_insertion_side_965")
class PegInsertionSide965Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-965_v2.pkl"


@register_task("maniskill.peg_insertion_side_545", "peg_insertion_side_545")
class PegInsertionSide545Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-545_v2.pkl"


@register_task("maniskill.peg_insertion_side_949", "peg_insertion_side_949")
class PegInsertionSide949Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-949_v2.pkl"


@register_task("maniskill.peg_insertion_side_922", "peg_insertion_side_922")
class PegInsertionSide922Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-922_v2.pkl"


@register_task("maniskill.peg_insertion_side_549", "peg_insertion_side_549")
class PegInsertionSide549Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-549_v2.pkl"


@register_task("maniskill.peg_insertion_side_464", "peg_insertion_side_464")
class PegInsertionSide464Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-464_v2.pkl"


@register_task("maniskill.peg_insertion_side_627", "peg_insertion_side_627")
class PegInsertionSide627Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-627_v2.pkl"


@register_task("maniskill.peg_insertion_side_315", "peg_insertion_side_315")
class PegInsertionSide315Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-315_v2.pkl"


@register_task("maniskill.peg_insertion_side_880", "peg_insertion_side_880")
class PegInsertionSide880Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-880_v2.pkl"


@register_task("maniskill.peg_insertion_side_542", "peg_insertion_side_542")
class PegInsertionSide542Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-542_v2.pkl"


@register_task("maniskill.peg_insertion_side_678", "peg_insertion_side_678")
class PegInsertionSide678Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-678_v2.pkl"


@register_task("maniskill.peg_insertion_side_14", "peg_insertion_side_14")
class PegInsertionSide14Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-14_v2.pkl"


@register_task("maniskill.peg_insertion_side_233", "peg_insertion_side_233")
class PegInsertionSide233Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-233_v2.pkl"


@register_task("maniskill.peg_insertion_side_341", "peg_insertion_side_341")
class PegInsertionSide341Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-341_v2.pkl"


@register_task("maniskill.peg_insertion_side_555", "peg_insertion_side_555")
class PegInsertionSide555Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-555_v2.pkl"


@register_task("maniskill.peg_insertion_side_415", "peg_insertion_side_415")
class PegInsertionSide415Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-415_v2.pkl"


@register_task("maniskill.peg_insertion_side_279", "peg_insertion_side_279")
class PegInsertionSide279Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-279_v2.pkl"


@register_task("maniskill.peg_insertion_side_101", "peg_insertion_side_101")
class PegInsertionSide101Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-101_v2.pkl"


@register_task("maniskill.peg_insertion_side_602", "peg_insertion_side_602")
class PegInsertionSide602Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-602_v2.pkl"


@register_task("maniskill.peg_insertion_side_724", "peg_insertion_side_724")
class PegInsertionSide724Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-724_v2.pkl"


@register_task("maniskill.peg_insertion_side_79", "peg_insertion_side_79")
class PegInsertionSide79Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-79_v2.pkl"


@register_task("maniskill.peg_insertion_side_522", "peg_insertion_side_522")
class PegInsertionSide522Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-522_v2.pkl"


@register_task("maniskill.peg_insertion_side_808", "peg_insertion_side_808")
class PegInsertionSide808Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-808_v2.pkl"


@register_task("maniskill.peg_insertion_side_537", "peg_insertion_side_537")
class PegInsertionSide537Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-537_v2.pkl"


@register_task("maniskill.peg_insertion_side_275", "peg_insertion_side_275")
class PegInsertionSide275Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-275_v2.pkl"


@register_task("maniskill.peg_insertion_side_358", "peg_insertion_side_358")
class PegInsertionSide358Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-358_v2.pkl"


@register_task("maniskill.peg_insertion_side_685", "peg_insertion_side_685")
class PegInsertionSide685Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-685_v2.pkl"


@register_task("maniskill.peg_insertion_side_617", "peg_insertion_side_617")
class PegInsertionSide617Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-617_v2.pkl"


@register_task("maniskill.peg_insertion_side_526", "peg_insertion_side_526")
class PegInsertionSide526Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-526_v2.pkl"


@register_task("maniskill.peg_insertion_side_248", "peg_insertion_side_248")
class PegInsertionSide248Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-248_v2.pkl"


@register_task("maniskill.peg_insertion_side_377", "peg_insertion_side_377")
class PegInsertionSide377Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-377_v2.pkl"


@register_task("maniskill.peg_insertion_side_527", "peg_insertion_side_527")
class PegInsertionSide527Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-527_v2.pkl"


@register_task("maniskill.peg_insertion_side_843", "peg_insertion_side_843")
class PegInsertionSide843Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-843_v2.pkl"


@register_task("maniskill.peg_insertion_side_659", "peg_insertion_side_659")
class PegInsertionSide659Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-659_v2.pkl"


@register_task("maniskill.peg_insertion_side_134", "peg_insertion_side_134")
class PegInsertionSide134Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-134_v2.pkl"


@register_task("maniskill.peg_insertion_side_21", "peg_insertion_side_21")
class PegInsertionSide21Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-21_v2.pkl"


@register_task("maniskill.peg_insertion_side_606", "peg_insertion_side_606")
class PegInsertionSide606Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-606_v2.pkl"


@register_task("maniskill.peg_insertion_side_391", "peg_insertion_side_391")
class PegInsertionSide391Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-391_v2.pkl"


@register_task("maniskill.peg_insertion_side_849", "peg_insertion_side_849")
class PegInsertionSide849Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-849_v2.pkl"


@register_task("maniskill.peg_insertion_side_19", "peg_insertion_side_19")
class PegInsertionSide19Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-19_v2.pkl"


@register_task("maniskill.peg_insertion_side_979", "peg_insertion_side_979")
class PegInsertionSide979Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-979_v2.pkl"


@register_task("maniskill.peg_insertion_side_737", "peg_insertion_side_737")
class PegInsertionSide737Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-737_v2.pkl"


@register_task("maniskill.peg_insertion_side_312", "peg_insertion_side_312")
class PegInsertionSide312Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-312_v2.pkl"


@register_task("maniskill.peg_insertion_side_621", "peg_insertion_side_621")
class PegInsertionSide621Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-621_v2.pkl"


@register_task("maniskill.peg_insertion_side_863", "peg_insertion_side_863")
class PegInsertionSide863Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-863_v2.pkl"


@register_task("maniskill.peg_insertion_side_245", "peg_insertion_side_245")
class PegInsertionSide245Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-245_v2.pkl"


@register_task("maniskill.peg_insertion_side_241", "peg_insertion_side_241")
class PegInsertionSide241Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-241_v2.pkl"


@register_task("maniskill.peg_insertion_side_80", "peg_insertion_side_80")
class PegInsertionSide80Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-80_v2.pkl"


@register_task("maniskill.peg_insertion_side_612", "peg_insertion_side_612")
class PegInsertionSide612Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-612_v2.pkl"


@register_task("maniskill.peg_insertion_side_87", "peg_insertion_side_87")
class PegInsertionSide87Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-87_v2.pkl"


@register_task("maniskill.peg_insertion_side_376", "peg_insertion_side_376")
class PegInsertionSide376Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-376_v2.pkl"


@register_task("maniskill.peg_insertion_side_993", "peg_insertion_side_993")
class PegInsertionSide993Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-993_v2.pkl"


@register_task("maniskill.peg_insertion_side_444", "peg_insertion_side_444")
class PegInsertionSide444Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-444_v2.pkl"


@register_task("maniskill.peg_insertion_side_192", "peg_insertion_side_192")
class PegInsertionSide192Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-192_v2.pkl"


@register_task("maniskill.peg_insertion_side_650", "peg_insertion_side_650")
class PegInsertionSide650Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-650_v2.pkl"


@register_task("maniskill.peg_insertion_side_792", "peg_insertion_side_792")
class PegInsertionSide792Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-792_v2.pkl"


@register_task("maniskill.peg_insertion_side_772", "peg_insertion_side_772")
class PegInsertionSide772Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-772_v2.pkl"


@register_task("maniskill.peg_insertion_side_382", "peg_insertion_side_382")
class PegInsertionSide382Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-382_v2.pkl"


@register_task("maniskill.peg_insertion_side_115", "peg_insertion_side_115")
class PegInsertionSide115Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-115_v2.pkl"


@register_task("maniskill.peg_insertion_side_748", "peg_insertion_side_748")
class PegInsertionSide748Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-748_v2.pkl"


@register_task("maniskill.peg_insertion_side_202", "peg_insertion_side_202")
class PegInsertionSide202Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-202_v2.pkl"


@register_task("maniskill.peg_insertion_side_776", "peg_insertion_side_776")
class PegInsertionSide776Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-776_v2.pkl"


@register_task("maniskill.peg_insertion_side_958", "peg_insertion_side_958")
class PegInsertionSide958Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-958_v2.pkl"


@register_task("maniskill.peg_insertion_side_655", "peg_insertion_side_655")
class PegInsertionSide655Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-655_v2.pkl"


@register_task("maniskill.peg_insertion_side_761", "peg_insertion_side_761")
class PegInsertionSide761Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-761_v2.pkl"


@register_task("maniskill.peg_insertion_side_727", "peg_insertion_side_727")
class PegInsertionSide727Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-727_v2.pkl"


@register_task("maniskill.peg_insertion_side_536", "peg_insertion_side_536")
class PegInsertionSide536Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-536_v2.pkl"


@register_task("maniskill.peg_insertion_side_121", "peg_insertion_side_121")
class PegInsertionSide121Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-121_v2.pkl"


@register_task("maniskill.peg_insertion_side_623", "peg_insertion_side_623")
class PegInsertionSide623Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-623_v2.pkl"


@register_task("maniskill.peg_insertion_side_396", "peg_insertion_side_396")
class PegInsertionSide396Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-396_v2.pkl"


@register_task("maniskill.peg_insertion_side_867", "peg_insertion_side_867")
class PegInsertionSide867Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-867_v2.pkl"


@register_task("maniskill.peg_insertion_side_303", "peg_insertion_side_303")
class PegInsertionSide303Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-303_v2.pkl"


@register_task("maniskill.peg_insertion_side_851", "peg_insertion_side_851")
class PegInsertionSide851Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-851_v2.pkl"


@register_task("maniskill.peg_insertion_side_890", "peg_insertion_side_890")
class PegInsertionSide890Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-890_v2.pkl"


@register_task("maniskill.peg_insertion_side_499", "peg_insertion_side_499")
class PegInsertionSide499Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-499_v2.pkl"


@register_task("maniskill.peg_insertion_side_250", "peg_insertion_side_250")
class PegInsertionSide250Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-250_v2.pkl"


@register_task("maniskill.peg_insertion_side_821", "peg_insertion_side_821")
class PegInsertionSide821Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-821_v2.pkl"


@register_task("maniskill.peg_insertion_side_798", "peg_insertion_side_798")
class PegInsertionSide798Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-798_v2.pkl"


@register_task("maniskill.peg_insertion_side_37", "peg_insertion_side_37")
class PegInsertionSide37Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-37_v2.pkl"


@register_task("maniskill.peg_insertion_side_336", "peg_insertion_side_336")
class PegInsertionSide336Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-336_v2.pkl"


@register_task("maniskill.peg_insertion_side_948", "peg_insertion_side_948")
class PegInsertionSide948Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-948_v2.pkl"


@register_task("maniskill.peg_insertion_side_995", "peg_insertion_side_995")
class PegInsertionSide995Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-995_v2.pkl"


@register_task("maniskill.peg_insertion_side_831", "peg_insertion_side_831")
class PegInsertionSide831Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-831_v2.pkl"


@register_task("maniskill.peg_insertion_side_587", "peg_insertion_side_587")
class PegInsertionSide587Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-587_v2.pkl"


@register_task("maniskill.peg_insertion_side_117", "peg_insertion_side_117")
class PegInsertionSide117Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-117_v2.pkl"


@register_task("maniskill.peg_insertion_side_789", "peg_insertion_side_789")
class PegInsertionSide789Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-789_v2.pkl"


@register_task("maniskill.peg_insertion_side_348", "peg_insertion_side_348")
class PegInsertionSide348Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-348_v2.pkl"


@register_task("maniskill.peg_insertion_side_1", "peg_insertion_side_1")
class PegInsertionSide1Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-1_v2.pkl"


@register_task("maniskill.peg_insertion_side_109", "peg_insertion_side_109")
class PegInsertionSide109Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-109_v2.pkl"


@register_task("maniskill.peg_insertion_side_569", "peg_insertion_side_569")
class PegInsertionSide569Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-569_v2.pkl"


@register_task("maniskill.peg_insertion_side_493", "peg_insertion_side_493")
class PegInsertionSide493Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-493_v2.pkl"


@register_task("maniskill.peg_insertion_side_119", "peg_insertion_side_119")
class PegInsertionSide119Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-119_v2.pkl"


@register_task("maniskill.peg_insertion_side_22", "peg_insertion_side_22")
class PegInsertionSide22Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-22_v2.pkl"


@register_task("maniskill.peg_insertion_side_615", "peg_insertion_side_615")
class PegInsertionSide615Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-615_v2.pkl"


@register_task("maniskill.peg_insertion_side_367", "peg_insertion_side_367")
class PegInsertionSide367Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-367_v2.pkl"


@register_task("maniskill.peg_insertion_side_466", "peg_insertion_side_466")
class PegInsertionSide466Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-466_v2.pkl"


@register_task("maniskill.peg_insertion_side_200", "peg_insertion_side_200")
class PegInsertionSide200Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-200_v2.pkl"


@register_task("maniskill.peg_insertion_side_257", "peg_insertion_side_257")
class PegInsertionSide257Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-257_v2.pkl"


@register_task("maniskill.peg_insertion_side_483", "peg_insertion_side_483")
class PegInsertionSide483Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-483_v2.pkl"


@register_task("maniskill.peg_insertion_side_731", "peg_insertion_side_731")
class PegInsertionSide731Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-731_v2.pkl"


@register_task("maniskill.peg_insertion_side_734", "peg_insertion_side_734")
class PegInsertionSide734Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-734_v2.pkl"


@register_task("maniskill.peg_insertion_side_433", "peg_insertion_side_433")
class PegInsertionSide433Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-433_v2.pkl"


@register_task("maniskill.peg_insertion_side_147", "peg_insertion_side_147")
class PegInsertionSide147Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-147_v2.pkl"


@register_task("maniskill.peg_insertion_side_350", "peg_insertion_side_350")
class PegInsertionSide350Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-350_v2.pkl"


@register_task("maniskill.peg_insertion_side_679", "peg_insertion_side_679")
class PegInsertionSide679Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-679_v2.pkl"


@register_task("maniskill.peg_insertion_side_736", "peg_insertion_side_736")
class PegInsertionSide736Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-736_v2.pkl"


@register_task("maniskill.peg_insertion_side_508", "peg_insertion_side_508")
class PegInsertionSide508Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-508_v2.pkl"


@register_task("maniskill.peg_insertion_side_578", "peg_insertion_side_578")
class PegInsertionSide578Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-578_v2.pkl"


@register_task("maniskill.peg_insertion_side_212", "peg_insertion_side_212")
class PegInsertionSide212Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-212_v2.pkl"


@register_task("maniskill.peg_insertion_side_581", "peg_insertion_side_581")
class PegInsertionSide581Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-581_v2.pkl"


@register_task("maniskill.peg_insertion_side_260", "peg_insertion_side_260")
class PegInsertionSide260Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-260_v2.pkl"


@register_task("maniskill.peg_insertion_side_366", "peg_insertion_side_366")
class PegInsertionSide366Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-366_v2.pkl"


@register_task("maniskill.peg_insertion_side_805", "peg_insertion_side_805")
class PegInsertionSide805Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-805_v2.pkl"


@register_task("maniskill.peg_insertion_side_25", "peg_insertion_side_25")
class PegInsertionSide25Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-25_v2.pkl"


@register_task("maniskill.peg_insertion_side_519", "peg_insertion_side_519")
class PegInsertionSide519Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-519_v2.pkl"


@register_task("maniskill.peg_insertion_side_700", "peg_insertion_side_700")
class PegInsertionSide700Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-700_v2.pkl"


@register_task("maniskill.peg_insertion_side_418", "peg_insertion_side_418")
class PegInsertionSide418Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-418_v2.pkl"


@register_task("maniskill.peg_insertion_side_989", "peg_insertion_side_989")
class PegInsertionSide989Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-989_v2.pkl"


@register_task("maniskill.peg_insertion_side_905", "peg_insertion_side_905")
class PegInsertionSide905Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-905_v2.pkl"


@register_task("maniskill.peg_insertion_side_6", "peg_insertion_side_6")
class PegInsertionSide6Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-6_v2.pkl"


@register_task("maniskill.peg_insertion_side_929", "peg_insertion_side_929")
class PegInsertionSide929Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-929_v2.pkl"


@register_task("maniskill.peg_insertion_side_34", "peg_insertion_side_34")
class PegInsertionSide34Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-34_v2.pkl"


@register_task("maniskill.peg_insertion_side_408", "peg_insertion_side_408")
class PegInsertionSide408Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-408_v2.pkl"


@register_task("maniskill.peg_insertion_side_468", "peg_insertion_side_468")
class PegInsertionSide468Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-468_v2.pkl"


@register_task("maniskill.peg_insertion_side_977", "peg_insertion_side_977")
class PegInsertionSide977Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-977_v2.pkl"


@register_task("maniskill.peg_insertion_side_584", "peg_insertion_side_584")
class PegInsertionSide584Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-584_v2.pkl"


@register_task("maniskill.peg_insertion_side_55", "peg_insertion_side_55")
class PegInsertionSide55Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-55_v2.pkl"


@register_task("maniskill.peg_insertion_side_50", "peg_insertion_side_50")
class PegInsertionSide50Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-50_v2.pkl"


@register_task("maniskill.peg_insertion_side_509", "peg_insertion_side_509")
class PegInsertionSide509Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-509_v2.pkl"


@register_task("maniskill.peg_insertion_side_971", "peg_insertion_side_971")
class PegInsertionSide971Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-971_v2.pkl"


@register_task("maniskill.peg_insertion_side_936", "peg_insertion_side_936")
class PegInsertionSide936Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-936_v2.pkl"


@register_task("maniskill.peg_insertion_side_254", "peg_insertion_side_254")
class PegInsertionSide254Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-254_v2.pkl"


@register_task("maniskill.peg_insertion_side_70", "peg_insertion_side_70")
class PegInsertionSide70Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-70_v2.pkl"


@register_task("maniskill.peg_insertion_side_141", "peg_insertion_side_141")
class PegInsertionSide141Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-141_v2.pkl"


@register_task("maniskill.peg_insertion_side_51", "peg_insertion_side_51")
class PegInsertionSide51Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-51_v2.pkl"


@register_task("maniskill.peg_insertion_side_596", "peg_insertion_side_596")
class PegInsertionSide596Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-596_v2.pkl"


@register_task("maniskill.peg_insertion_side_661", "peg_insertion_side_661")
class PegInsertionSide661Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-661_v2.pkl"


@register_task("maniskill.peg_insertion_side_869", "peg_insertion_side_869")
class PegInsertionSide869Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-869_v2.pkl"


@register_task("maniskill.peg_insertion_side_465", "peg_insertion_side_465")
class PegInsertionSide465Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-465_v2.pkl"


@register_task("maniskill.peg_insertion_side_128", "peg_insertion_side_128")
class PegInsertionSide128Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-128_v2.pkl"


@register_task("maniskill.peg_insertion_side_439", "peg_insertion_side_439")
class PegInsertionSide439Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-439_v2.pkl"


@register_task("maniskill.peg_insertion_side_605", "peg_insertion_side_605")
class PegInsertionSide605Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-605_v2.pkl"


@register_task("maniskill.peg_insertion_side_940", "peg_insertion_side_940")
class PegInsertionSide940Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-940_v2.pkl"


@register_task("maniskill.peg_insertion_side_319", "peg_insertion_side_319")
class PegInsertionSide319Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-319_v2.pkl"


@register_task("maniskill.peg_insertion_side_4", "peg_insertion_side_4")
class PegInsertionSide4Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-4_v2.pkl"


@register_task("maniskill.peg_insertion_side_682", "peg_insertion_side_682")
class PegInsertionSide682Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-682_v2.pkl"


@register_task("maniskill.peg_insertion_side_8", "peg_insertion_side_8")
class PegInsertionSide8Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-8_v2.pkl"


@register_task("maniskill.peg_insertion_side_534", "peg_insertion_side_534")
class PegInsertionSide534Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-534_v2.pkl"


@register_task("maniskill.peg_insertion_side_339", "peg_insertion_side_339")
class PegInsertionSide339Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-339_v2.pkl"


@register_task("maniskill.peg_insertion_side_365", "peg_insertion_side_365")
class PegInsertionSide365Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-365_v2.pkl"


@register_task("maniskill.peg_insertion_side_530", "peg_insertion_side_530")
class PegInsertionSide530Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-530_v2.pkl"


@register_task("maniskill.peg_insertion_side_845", "peg_insertion_side_845")
class PegInsertionSide845Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-845_v2.pkl"


@register_task("maniskill.peg_insertion_side_592", "peg_insertion_side_592")
class PegInsertionSide592Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-592_v2.pkl"


@register_task("maniskill.peg_insertion_side_600", "peg_insertion_side_600")
class PegInsertionSide600Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-600_v2.pkl"


@register_task("maniskill.peg_insertion_side_181", "peg_insertion_side_181")
class PegInsertionSide181Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-181_v2.pkl"


@register_task("maniskill.peg_insertion_side_321", "peg_insertion_side_321")
class PegInsertionSide321Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-321_v2.pkl"


@register_task("maniskill.peg_insertion_side_501", "peg_insertion_side_501")
class PegInsertionSide501Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-501_v2.pkl"


@register_task("maniskill.peg_insertion_side_689", "peg_insertion_side_689")
class PegInsertionSide689Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-689_v2.pkl"


@register_task("maniskill.peg_insertion_side_345", "peg_insertion_side_345")
class PegInsertionSide345Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-345_v2.pkl"


@register_task("maniskill.peg_insertion_side_238", "peg_insertion_side_238")
class PegInsertionSide238Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-238_v2.pkl"


@register_task("maniskill.peg_insertion_side_389", "peg_insertion_side_389")
class PegInsertionSide389Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-389_v2.pkl"


@register_task("maniskill.peg_insertion_side_645", "peg_insertion_side_645")
class PegInsertionSide645Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-645_v2.pkl"


@register_task("maniskill.peg_insertion_side_24", "peg_insertion_side_24")
class PegInsertionSide24Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-24_v2.pkl"


@register_task("maniskill.peg_insertion_side_771", "peg_insertion_side_771")
class PegInsertionSide771Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-771_v2.pkl"


@register_task("maniskill.peg_insertion_side_27", "peg_insertion_side_27")
class PegInsertionSide27Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-27_v2.pkl"


@register_task("maniskill.peg_insertion_side_868", "peg_insertion_side_868")
class PegInsertionSide868Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-868_v2.pkl"


@register_task("maniskill.peg_insertion_side_61", "peg_insertion_side_61")
class PegInsertionSide61Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-61_v2.pkl"


@register_task("maniskill.peg_insertion_side_271", "peg_insertion_side_271")
class PegInsertionSide271Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-271_v2.pkl"


@register_task("maniskill.peg_insertion_side_264", "peg_insertion_side_264")
class PegInsertionSide264Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-264_v2.pkl"


@register_task("maniskill.peg_insertion_side_586", "peg_insertion_side_586")
class PegInsertionSide586Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-586_v2.pkl"


@register_task("maniskill.peg_insertion_side_903", "peg_insertion_side_903")
class PegInsertionSide903Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-903_v2.pkl"


@register_task("maniskill.peg_insertion_side_188", "peg_insertion_side_188")
class PegInsertionSide188Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-188_v2.pkl"


@register_task("maniskill.peg_insertion_side_515", "peg_insertion_side_515")
class PegInsertionSide515Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-515_v2.pkl"


@register_task("maniskill.peg_insertion_side_324", "peg_insertion_side_324")
class PegInsertionSide324Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-324_v2.pkl"


@register_task("maniskill.peg_insertion_side_872", "peg_insertion_side_872")
class PegInsertionSide872Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-872_v2.pkl"


@register_task("maniskill.peg_insertion_side_424", "peg_insertion_side_424")
class PegInsertionSide424Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-424_v2.pkl"


@register_task("maniskill.peg_insertion_side_340", "peg_insertion_side_340")
class PegInsertionSide340Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-340_v2.pkl"


@register_task("maniskill.peg_insertion_side_143", "peg_insertion_side_143")
class PegInsertionSide143Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-143_v2.pkl"


@register_task("maniskill.peg_insertion_side_718", "peg_insertion_side_718")
class PegInsertionSide718Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-718_v2.pkl"


@register_task("maniskill.peg_insertion_side_565", "peg_insertion_side_565")
class PegInsertionSide565Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-565_v2.pkl"


@register_task("maniskill.peg_insertion_side_420", "peg_insertion_side_420")
class PegInsertionSide420Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-420_v2.pkl"


@register_task("maniskill.peg_insertion_side_5", "peg_insertion_side_5")
class PegInsertionSide5Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-5_v2.pkl"


@register_task("maniskill.peg_insertion_side_314", "peg_insertion_side_314")
class PegInsertionSide314Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-314_v2.pkl"


@register_task("maniskill.peg_insertion_side_72", "peg_insertion_side_72")
class PegInsertionSide72Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-72_v2.pkl"


@register_task("maniskill.peg_insertion_side_998", "peg_insertion_side_998")
class PegInsertionSide998Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-998_v2.pkl"


@register_task("maniskill.peg_insertion_side_956", "peg_insertion_side_956")
class PegInsertionSide956Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-956_v2.pkl"


@register_task("maniskill.peg_insertion_side_767", "peg_insertion_side_767")
class PegInsertionSide767Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-767_v2.pkl"


@register_task("maniskill.peg_insertion_side_762", "peg_insertion_side_762")
class PegInsertionSide762Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-762_v2.pkl"


@register_task("maniskill.peg_insertion_side_999", "peg_insertion_side_999")
class PegInsertionSide999Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-999_v2.pkl"


@register_task("maniskill.peg_insertion_side_446", "peg_insertion_side_446")
class PegInsertionSide446Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-446_v2.pkl"


@register_task("maniskill.peg_insertion_side_39", "peg_insertion_side_39")
class PegInsertionSide39Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-39_v2.pkl"


@register_task("maniskill.peg_insertion_side_597", "peg_insertion_side_597")
class PegInsertionSide597Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-597_v2.pkl"


@register_task("maniskill.peg_insertion_side_707", "peg_insertion_side_707")
class PegInsertionSide707Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-707_v2.pkl"


@register_task("maniskill.peg_insertion_side_741", "peg_insertion_side_741")
class PegInsertionSide741Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-741_v2.pkl"


@register_task("maniskill.peg_insertion_side_300", "peg_insertion_side_300")
class PegInsertionSide300Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-300_v2.pkl"


@register_task("maniskill.peg_insertion_side_635", "peg_insertion_side_635")
class PegInsertionSide635Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-635_v2.pkl"


@register_task("maniskill.peg_insertion_side_182", "peg_insertion_side_182")
class PegInsertionSide182Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-182_v2.pkl"


@register_task("maniskill.peg_insertion_side_429", "peg_insertion_side_429")
class PegInsertionSide429Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-429_v2.pkl"


@register_task("maniskill.peg_insertion_side_696", "peg_insertion_side_696")
class PegInsertionSide696Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-696_v2.pkl"


@register_task("maniskill.peg_insertion_side_786", "peg_insertion_side_786")
class PegInsertionSide786Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-786_v2.pkl"


@register_task("maniskill.peg_insertion_side_840", "peg_insertion_side_840")
class PegInsertionSide840Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-840_v2.pkl"


@register_task("maniskill.peg_insertion_side_403", "peg_insertion_side_403")
class PegInsertionSide403Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-403_v2.pkl"


@register_task("maniskill.peg_insertion_side_524", "peg_insertion_side_524")
class PegInsertionSide524Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-524_v2.pkl"


@register_task("maniskill.peg_insertion_side_310", "peg_insertion_side_310")
class PegInsertionSide310Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-310_v2.pkl"


@register_task("maniskill.peg_insertion_side_214", "peg_insertion_side_214")
class PegInsertionSide214Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-214_v2.pkl"


@register_task("maniskill.peg_insertion_side_757", "peg_insertion_side_757")
class PegInsertionSide757Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-757_v2.pkl"


@register_task("maniskill.peg_insertion_side_35", "peg_insertion_side_35")
class PegInsertionSide35Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-35_v2.pkl"


@register_task("maniskill.peg_insertion_side_750", "peg_insertion_side_750")
class PegInsertionSide750Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-750_v2.pkl"


@register_task("maniskill.peg_insertion_side_931", "peg_insertion_side_931")
class PegInsertionSide931Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-931_v2.pkl"


@register_task("maniskill.peg_insertion_side_886", "peg_insertion_side_886")
class PegInsertionSide886Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-886_v2.pkl"


@register_task("maniskill.peg_insertion_side_174", "peg_insertion_side_174")
class PegInsertionSide174Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-174_v2.pkl"


@register_task("maniskill.peg_insertion_side_75", "peg_insertion_side_75")
class PegInsertionSide75Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-75_v2.pkl"


@register_task("maniskill.peg_insertion_side_262", "peg_insertion_side_262")
class PegInsertionSide262Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-262_v2.pkl"


@register_task("maniskill.peg_insertion_side_571", "peg_insertion_side_571")
class PegInsertionSide571Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-571_v2.pkl"


@register_task("maniskill.peg_insertion_side_410", "peg_insertion_side_410")
class PegInsertionSide410Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-410_v2.pkl"


@register_task("maniskill.peg_insertion_side_533", "peg_insertion_side_533")
class PegInsertionSide533Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )
    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-533_v2.pkl"


@register_task("maniskill.peg_insertion_side_81", "peg_insertion_side_81")
class PegInsertionSide81Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg
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


@register_task("maniskill.peg_insertion_side_810", "peg_insertion_side_810")
class PegInsertionSide810Task(PegInsertionSideBaseTask):
    scenario = ScenarioCfg(
        objects=[
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
    )

    traj_filepath = "roboverse_data/trajs/maniskill/peg_insertion_side/trajectory-franka-810_v2.pkl"
