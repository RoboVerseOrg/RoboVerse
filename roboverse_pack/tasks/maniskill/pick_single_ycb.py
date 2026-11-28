"""The base class and derived classes for the pick up single YCB object task from ManiSkill."""

import torch

from metasim.constants import PhysicStateType
from metasim.scenario.objects import PrimitiveCubeCfg, PrimitiveSphereCfg, RigidObjCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.checkers.checkers import PositionShiftChecker
from metasim.task.registry import register_task

from ._dense_rl import DenseRLResetMixin
from .maniskill_base import ManiskillBaseTask


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


# ---------------------------------------------------------------------------
# Dense-reward variant — 1:1 port of ManiSkill3 PickSingleYCB-v1
# ``compute_dense_reward`` (mani_skill/envs/tasks/tabletop/pick_single_ycb.py).
# Distinct from PickCube: adds an ``is_obj_placed·is_grasped`` bonus, max 6.
# Uses a primitive object substitute (YCB meshes are usd-only / asset-heavy;
# the reward uses only the object's *position*, so the formula stays 1:1).
# ---------------------------------------------------------------------------

_YCB_GOAL_THRESH = 0.025
_YCB_TCP_OFFSET = (0.0, 0.0, 0.10312)


@register_task("maniskill.pick_single_ycb_dense", "pick_single_ycb_dense")
class PickSingleYcbDenseTask(DenseRLResetMixin, _PickSingleYcbBaseTask):
    """PickSingleYCB with ManiSkill3's dense reward ported 1:1 (max 6).

    Reward::

        reaching = 1 - tanh(5·||tcp - obj||)
        reward   = reaching + is_grasped
        place    = 1 - tanh(5·||goal - obj||)
        reward  += place·is_grasped
        reward  += is_obj_placed·is_grasped            # PickSingleYCB-specific
        static   = 1 - tanh(5·||arm_qvel||)
        reward  += static·is_obj_placed·is_grasped
        reward[success] = 6

    is_obj_placed = ||obj-goal|| <= goal_thresh (0.025).
    success = placed & robot_static & grasped. Primitive object + goal sphere,
    randomized resets, no demo data. ``is_grasped`` approximated.
    """

    cube_z = 0.02
    goal_z_range = (0.10, 0.30)
    scenario = ScenarioCfg(
        objects=[
            PrimitiveCubeCfg(
                name="obj",
                size=(0.04, 0.04, 0.04),
                mass=0.064,
                physics=PhysicStateType.RIGIDBODY,
                color=(1.0, 0.5, 0.0),
            ),
            PrimitiveSphereCfg(name="goal_site", radius=0.025, physics=PhysicStateType.XFORM, color=(0.0, 1.0, 0.0)),
        ],
        robots=["franka"],
    )
    # inherits base PositionShiftChecker(obj_name="obj") for termination — "obj"
    # exists in this scenario, so reset()/_terminated work; dense reward is separate.

    def _reward(self, env_states):
        from metasim.utils.math import quat_rotate

        rs = env_states.robots["franka"]
        objs = env_states.objects
        device = rs.joint_pos.device
        n = rs.joint_pos.shape[0]

        obj = objs["obj"].root_state[:, :3]
        goal = objs["goal_site"].root_state[:, :3]

        names = rs.body_names
        hidx = next((i for i, nm in enumerate(names) if nm.endswith("panda_hand")), None)
        if hidx is None:
            return torch.zeros(n, device=device)
        hand_pos = rs.body_state[:, hidx, :3]
        hand_quat = rs.body_state[:, hidx, 3:7]
        offset = torch.tensor(_YCB_TCP_OFFSET, device=device, dtype=hand_pos.dtype).expand(n, 3)
        tcp = hand_pos + quat_rotate(hand_quat, offset)

        tcp_to_obj = torch.linalg.norm(obj - tcp, dim=-1)
        reaching = 1.0 - torch.tanh(5.0 * tcp_to_obj)
        gripper_w = rs.joint_pos[:, 0:2].sum(dim=-1)
        is_grasped = ((tcp_to_obj < 0.03) & (gripper_w < 0.07)).float()
        reward = reaching + is_grasped

        obj_to_goal = torch.linalg.norm(goal - obj, dim=-1)
        place = 1.0 - torch.tanh(5.0 * obj_to_goal)
        reward = reward + place * is_grasped

        is_obj_placed = (obj_to_goal <= _YCB_GOAL_THRESH).float()
        reward = reward + is_obj_placed * is_grasped

        arm_qvel = rs.joint_vel[:, 2:9]
        static = 1.0 - torch.tanh(5.0 * torch.linalg.norm(arm_qvel, dim=-1))
        reward = reward + static * is_obj_placed * is_grasped

        is_static = (arm_qvel.abs().amax(dim=-1) < 0.2).float()
        success = (is_obj_placed * is_static * is_grasped).bool()
        reward = torch.where(success, torch.full_like(reward, 6.0), reward)
        return reward


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
