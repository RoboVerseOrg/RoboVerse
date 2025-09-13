"""The base class and derived classes for the pick up single YCB object task from ManiSkill."""

from maniskill_base import ManiskillBaseTask

from metasim.constants import PhysicStateType
from metasim.example.example_pack.tasks.checkers.checkers import PositionShiftChecker
from metasim.scenario.objects import RigidObjCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.registry import register_task


@register_task("maniskill.pick_single_ycb_base", "pick_single_ycb_base")
class _PickSingleYcbBaseTask(ManiskillBaseTask):
    """The pick up single YCB object task from ManiSkill.

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


@register_task("maniskill.pick_single_ycb_lego_duplo", "pick_single_ycb_lego_luplo")
class PickSingleYcbLegoDuploTask(_PickSingleYcbBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/ycb/073-g_lego_duplo/object_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/ycb/073-g_lego_duplo/model_scaled.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )

    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_ycb/trajectory-franka-073-g_lego_duplo_v2.pkl"


@register_task("maniskill.pick_single_ycb_wood_block", "pick_single_ycb_wood_block")
class PickSingleYcbWoodBlockTask(_PickSingleYcbBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/ycb/036_wood_block/object_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/ycb/036_wood_block/model_scaled.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )

    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_ycb/trajectory-franka-036_wood_block_v2.pkl"


@register_task("maniskill.pick_single_ycb_flat_screwdriver", "pick_single_ycb_flat_screwdriver")
class PickSingleYcbFlatScrewdriverTask(_PickSingleYcbBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/ycb/044_flat_screwdriver/object_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/ycb/044_flat_screwdriver/model_scaled.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_ycb/trajectory-franka-044_flat_screwdriver_v2.pkl"


@register_task("maniskill.pick_single_ycb_extra_large_clamp", "pick_single_ycb_extra_large_clamp")
class PickSingleYcbExtraLargeClampTask(_PickSingleYcbBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/ycb/052_extra_large_clamp/object_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/ycb/052_extra_large_clamp/model_scaled.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )

    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_ycb/trajectory-franka-052_extra_large_clamp_v2.pkl"


@register_task("maniskill.pick_single_ycb_fork", "pick_single_ycb_fork")
class PickSingleYcbForkTask(_PickSingleYcbBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/ycb/030_fork/object_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/ycb/030_fork/model_scaled.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )

    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_ycb/trajectory-franka-030_fork_v2.pkl"


@register_task("maniskill.pick_single_ycb_cups", "pick_single_ycb_cups")
class PickSingleYcbCupsTask(_PickSingleYcbBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/ycb/065-j_cups/object_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/ycb/065-j_cups/model_scaled.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )

    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_ycb/trajectory-franka-065-j_cups_v2.pkl"


@register_task("maniskill.pick_single_ycb_power_drill", "pick_single_ycb_power_drill")
class PickSingleYcbPowerDrillTask(_PickSingleYcbBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/ycb/035_power_drill/object_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/ycb/035_power_drill/model_scaled.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )

    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_ycb/trajectory-franka-035_power_drill_v2.pkl"


@register_task("maniskill.pick_single_ycb_banana", "pick_single_ycb_banana")
class PickSingleYcbBananaTask(_PickSingleYcbBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/ycb/011_banana/object_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/ycb/011_banana/model_scaled.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )

    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_ycb/trajectory-franka-011_banana_v2.pkl"


@register_task("maniskill.pick_single_ycb_master_chef_can", "pick_single_ycb_master_chef_can")
class PickSingleYcbMasterChefCanTask(_PickSingleYcbBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/ycb/002_master_chef_can/object_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/ycb/002_master_chef_can/model_scaled.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )

    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_ycb/trajectory-franka-002_master_chef_can_v2.pkl"


@register_task("maniskill.pick_single_ycb_phillips_screwdriver", "pick_single_ycb_phillips_screwdriver")
class PickSingleYcbPhillipsScrewdriverTask(_PickSingleYcbBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/ycb/043_phillips_screwdriver/object_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/ycb/043_phillips_screwdriver/model_scaled.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )

    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_ycb/trajectory-franka-043_phillips_screwdriver_v2.pkl"


@register_task("maniskill.pick_single_ycb_hammer", "pick_single_ycb_hammer")
class PickSingleYcbHammerTask(_PickSingleYcbBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/ycb/048_hammer/object_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/ycb/048_hammer/model_scaled.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )

    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_ycb/trajectory-franka-048_hammer_v2.pkl"


@register_task("maniskill.pick_single_ycb_padlock", "pick_single_ycb_padlock")
class PickSingleYcbPadlockTask(_PickSingleYcbBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/ycb/038_padlock/object_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/ycb/038_padlock/model_scaled.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )

    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_ycb/trajectory-franka-038_padlock_v2.pkl"


@register_task("maniskill.pick_single_ycb_orange", "pick_single_ycb_orange")
class PickSingleYcbOrangeTask(_PickSingleYcbBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/ycb/017_orange/object_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/ycb/017_orange/model_scaled.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )

    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_ycb/trajectory-franka-017_orange_v2.pkl"


@register_task("maniskill.pick_single_ycb_rubiks_cube", "pick_single_ycb_rubiks_cube")
class PickSingleYcbRubiksCubeTask(_PickSingleYcbBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/ycb/077_rubiks_cube/object_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/ycb/077_rubiks_cube/model_scaled.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )

    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_ycb/trajectory-franka-077_rubiks_cube_v2.pkl"


@register_task("maniskill.pick_single_ycb_spatula", "pick_single_ycb_spatula")
class PickSingleYcbSpatulaTask(_PickSingleYcbBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/ycb/033_spatula/object_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/ycb/033_spatula/model_scaled.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )

    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_ycb/trajectory-franka-033_spatula_v2.pkl"


@register_task("maniskill.pick_single_ycb_toy_airplane", "pick_single_ycb_toy_airplane")
class PickSingleYcbToyAirplaneTask(_PickSingleYcbBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/ycb/072-e_toy_airplane/object_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/ycb/072-e_toy_airplane/model_scaled.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_ycb/trajectory-franka-072-e_toy_airplane_v2.pkl"


@register_task("maniskill.pick_single_ycb_strawberry", "pick_single_ycb_strawberry")
class PickSingleYcbStrawberryTask(_PickSingleYcbBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/ycb/012_strawberry/object_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/ycb/012_strawberry/model_scaled.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_ycb/trajectory-franka-012_strawberry_v2.pkl"


@register_task("maniskill.pick_single_ycb_lemon", "pick_single_ycb_lemon")
class PickSingleYcbLemonTask(_PickSingleYcbBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/ycb/014_lemon/object_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/ycb/014_lemon/model_scaled.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_ycb/trajectory-franka-014_lemon_v2.pkl"


@register_task("maniskill.pick_single_ycb_nine_hole_peg_test", "pick_single_ycb_nine_hole_peg_test")
class PickSingleYcbNineHolePegTestTask(_PickSingleYcbBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/ycb/071_nine_hole_peg_test/object_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/ycb/071_nine_hole_peg_test/model_scaled.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_ycb/trajectory-franka-071_nine_hole_peg_test_v2.pkl"


@register_task("maniskill.pick_single_ycb_dice", "pick_single_ycb_dice")
class PickSingleYcbDiceTask(_PickSingleYcbBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/ycb/062_dice/object_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/ycb/062_dice/model_scaled.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_ycb/trajectory-franka-062_dice_v2.pkl"


@register_task("maniskill.pick_single_ycb_racquetball", "pick_single_ycb_racquetball")
class PickSingleYcbRacquetballTask(_PickSingleYcbBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/ycb/057_racquetball/object_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/ycb/057_racquetball/model_scaled.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_ycb/trajectory-franka-057_racquetball_v2.pkl"


@register_task("maniskill.pick_single_ycb_bowl", "pick_single_ycb_bowl")
class PickSingleYcbBowlTask(_PickSingleYcbBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/ycb/024_bowl/object_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/ycb/024_bowl/model_scaled.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_ycb/trajectory-franka-024_bowl_v2.pkl"


@register_task("maniskill.pick_single_ycb_tomato_soup_can", "pick_single_ycb_tomato_soup_can")
class PickSingleYcbTomatoSoupCanTask(_PickSingleYcbBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/ycb/005_tomato_soup_can/object_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/ycb/005_tomato_soup_can/model_scaled.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_ycb/trajectory-franka-005_tomato_soup_can_v2.pkl"


@register_task("maniskill.pick_single_ycb_cracker_box", "pick_single_ycb_cracker_box")
class PickSingleYcbCrackerBoxTask(_PickSingleYcbBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/ycb/003_cracker_box/object_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/ycb/003_cracker_box/model_scaled.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_ycb/trajectory-franka-003_cracker_box_v2.pkl"


@register_task("maniskill.pick_single_ycb_scissors", "pick_single_ycb_scissors")
class PickSingleYcbScissorsTask(_PickSingleYcbBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/ycb/037_scissors/object_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/ycb/037_scissors/model_scaled.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_ycb/trajectory-franka-037_scissors_v2.pkl"


@register_task("maniskill.pick_single_ycb_plum", "pick_single_ycb_plum")
class PickSingleYcbPlumTask(_PickSingleYcbBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/ycb/018_plum/object_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/ycb/018_plum/model_scaled.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_ycb/trajectory-franka-018_plum_v2.pkl"


@register_task("maniskill.pick_single_ycb_bleach_cleanser", "pick_single_ycb_bleach_cleanser")
class PickSingleYcbBleachCleanserTask(_PickSingleYcbBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/ycb/021_bleach_cleanser/object_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/ycb/021_bleach_cleanser/model_scaled.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_ycb/trajectory-franka-021_bleach_cleanser_v2.pkl"


@register_task("maniskill.pick_single_ycb_medium_clamp", "pick_single_ycb_medium_clamp")
class PickSingleYcbMediumClampTask(_PickSingleYcbBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/ycb/050_medium_clamp/object_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/ycb/050_medium_clamp/model_scaled.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_ycb/trajectory-franka-050_medium_clamp_v2.pkl"


@register_task("maniskill.pick_single_ycb_sponge", "pick_single_ycb_sponge")
class PickSingleYcbSpongeTask(_PickSingleYcbBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/ycb/026_sponge/object_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/ycb/026_sponge/model_scaled.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_ycb/trajectory-franka-026_sponge_v2.pkl"


@register_task("maniskill.pick_single_ycb_pitcher_base", "pick_single_ycb_pitcher_base")
class PickSingleYcbPitcherBaseTask(_PickSingleYcbBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/ycb/019_pitcher_base/object_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/ycb/019_pitcher_base/model_scaled.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_ycb/trajectory-franka-019_pitcher_base_v2.pkl"


@register_task("maniskill.pick_single_ycb_tennis_ball", "pick_single_ycb_tennis_ball")
class PickSingleYcbTennisBallTask(_PickSingleYcbBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/ycb/056_tennis_ball/object_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/ycb/056_tennis_ball/model_scaled.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_ycb/trajectory-franka-056_tennis_ball_v2.pkl"


@register_task("maniskill.pick_single_ycb_colored_wood_blocks", "pick_single_ycb_colored_wood_blocks")
class PickSingleYcbColoredWoodBlocksTask(_PickSingleYcbBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/ycb/070-b_colored_wood_blocks/object_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/ycb/070-b_colored_wood_blocks/model_scaled.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_ycb/trajectory-franka-070-b_colored_wood_blocks_v2.pkl"


@register_task("maniskill.pick_single_ycb_mug", "pick_single_ycb_mug")
class PickSingleYcbMugTask(_PickSingleYcbBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/ycb/025_mug/object_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/ycb/025_mug/model_scaled.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_ycb/trajectory-franka-025_mug_v2.pkl"


@register_task("maniskill.pick_single_ycb_baseball", "pick_single_ycb_baseball")
class PickSingleYcbBaseballTask(_PickSingleYcbBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/ycb/055_baseball/object_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/ycb/055_baseball/model_scaled.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_ycb/trajectory-franka-055_baseball_v2.pkl"


@register_task("maniskill.pick_single_ycb_gelatin_box", "pick_single_ycb_gelatin_box")
class PickSingleYcbGelatinBoxTask(_PickSingleYcbBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/ycb/009_gelatin_box/object_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/ycb/009_gelatin_box/model_scaled.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_ycb/trajectory-franka-009_gelatin_box_v2.pkl"


@register_task("maniskill.pick_single_ycb_tuna_fish_can", "pick_single_ycb_tuna_fish_can")
class PickSingleYcbTunaFishCanTask(_PickSingleYcbBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/ycb/007_tuna_fish_can/object_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/ycb/007_tuna_fish_can/model_scaled.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_ycb/trajectory-franka-007_tuna_fish_can_v2.pkl"


@register_task("maniskill.pick_single_ycb_large_clamp", "pick_single_ycb_large_clamp")
class PickSingleYcbLargeClampTask(_PickSingleYcbBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/ycb/051_large_clamp/object_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/ycb/051_large_clamp/model_scaled.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_ycb/trajectory-franka-051_large_clamp_v2.pkl"


@register_task("maniskill.pick_single_ycb_peach", "pick_single_ycb_peach")
class PickSingleYcbPeachTask(_PickSingleYcbBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/ycb/015_peach/object_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/ycb/015_peach/model_scaled.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_ycb/trajectory-franka-015_peach_v2.pkl"


@register_task("maniskill.pick_single_ycb_knife", "pick_single_ycb_knife")
class PickSingleYcbKnifeTask(_PickSingleYcbBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/ycb/032_knife/object_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/ycb/032_knife/model_scaled.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_ycb/trajectory-franka-032_knife_v2.pkl"


@register_task("maniskill.pick_single_ycb_apple", "pick_single_ycb_apple")
class PickSingleYcbAppleTask(_PickSingleYcbBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/ycb/013_apple/object_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/ycb/013_apple/model_scaled.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_ycb/trajectory-franka-013_apple_v2.pkl"


@register_task("maniskill.pick_single_ycb_mustard_bottle", "pick_single_ycb_mustard_bottle")
class PickSingleYcbMustardBottleTask(_PickSingleYcbBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/ycb/006_mustard_bottle/object_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/ycb/006_mustard_bottle/model_scaled.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_ycb/trajectory-franka-006_mustard_bottle_v2.pkl"


@register_task("maniskill.pick_single_ycb_pear", "pick_single_ycb_pear")
class PickSingleYcbPearTask(_PickSingleYcbBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/ycb/016_pear/object_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/ycb/016_pear/model_scaled.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_ycb/trajectory-franka-016_pear_v2.pkl"


@register_task("maniskill.pick_single_ycb_large_marker", "pick_single_ycb_large_marker")
class PickSingleYcbLargeMarkerTask(_PickSingleYcbBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/ycb/040_large_marker/object_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/ycb/040_large_marker/model_scaled.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_ycb/trajectory-franka-040_large_marker_v2.pkl"


@register_task("maniskill.pick_single_ycb_adjustable_wrench", "pick_single_ycb_adjustable_wrench")
class PickSingleYcbAdjustableWrenchTask(_PickSingleYcbBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/ycb/042_adjustable_wrench/object_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/ycb/042_adjustable_wrench/model_scaled.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_ycb/trajectory-franka-042_adjustable_wrench_v2.pkl"


@register_task("maniskill.pick_single_ycb_softball", "pick_single_ycb_softball")
class PickSingleYcbSoftballTask(_PickSingleYcbBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/ycb/054_softball/object_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/ycb/054_softball/model_scaled.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_ycb/trajectory-franka-054_softball_v2.pkl"


@register_task("maniskill.pick_single_ycb_foam_brick", "pick_single_ycb_foam_brick")
class PickSingleYcbFoamBrickTask(_PickSingleYcbBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/ycb/061_foam_brick/object_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/ycb/061_foam_brick/model_scaled.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_ycb/trajectory-franka-061_foam_brick_v2.pkl"


@register_task("maniskill.pick_single_ycb_sugar_box", "pick_single_ycb_sugar_box")
class PickSingleYcbSugarBoxTask(_PickSingleYcbBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/ycb/004_sugar_box/object_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/ycb/004_sugar_box/model_scaled.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_ycb/trajectory-franka-004_sugar_box_v2.pkl"


@register_task("maniskill.pick_single_ycb_marbles", "pick_single_ycb_marbles")
class PickSingleYcbMarblesTask(_PickSingleYcbBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/ycb/063-b_marbles/object_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/ycb/063-b_marbles/model_scaled.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_ycb/trajectory-franka-063-b_marbles_v2.pkl"


@register_task("maniskill.pick_single_ycb_potted_meat_can", "pick_single_ycb_potted_meat_can")
class PickSingleYcbPottedMeatCanTask(_PickSingleYcbBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/ycb/010_potted_meat_can/object_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/ycb/010_potted_meat_can/model_scaled.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_ycb/trajectory-franka-010_potted_meat_can_v2.pkl"


@register_task("maniskill.pick_single_ycb_golf_ball", "pick_single_ycb_golf_ball")
class PickSingleYcbGolfBallTask(_PickSingleYcbBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/ycb/058_golf_ball/object_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/ycb/058_golf_ball/model_scaled.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_ycb/trajectory-franka-058_golf_ball_v2.pkl"


@register_task("maniskill.pick_single_ycb_mini_soccer_ball", "pick_single_ycb_mini_soccer_ball")
class PickSingleYcbMiniSoccerBallTask(_PickSingleYcbBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/ycb/053_mini_soccer_ball/object_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/ycb/053_mini_soccer_ball/model_scaled.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_ycb/trajectory-franka-053_mini_soccer_ball_v2.pkl"


@register_task("maniskill.pick_single_ycb_pudding_box", "pick_single_ycb_pudding_box")
class PickSingleYcbPuddingBoxTask(_PickSingleYcbBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/ycb/008_pudding_box/object_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/ycb/008_pudding_box/model_scaled.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_ycb/trajectory-franka-008_pudding_box_v2.pkl"


@register_task("maniskill.pick_single_ycb_spoon", "pick_single_ycb_spoon")
class PickSingleYcbSpoonTask(_PickSingleYcbBaseTask):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="obj",
                usd_path="roboverse_data/assets/maniskill/ycb/031_spoon/object_proc.usd",
                urdf_path="roboverse_data/assets/maniskill/ycb/031_spoon/model_scaled.urdf",
                physics=PhysicStateType.RIGIDBODY,
            )
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_ycb/trajectory-franka-031_spoon_v2.pkl"
