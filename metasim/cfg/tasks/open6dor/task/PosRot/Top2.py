from metasim.cfg.objects import PrimitiveCubeCfg, RigidObjCfg
from metasim.cfg.tasks.base_task_cfg import BaseTaskCfg
from metasim.constants import BenchmarkType, PhysicStateType, TaskType
from metasim.utils import configclass


@configclass
class OpensdorPosRotTopRemoteControlForthRemoteControl1371Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="mixer",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/990d0cb9499540fda49b1ff36be9ba26/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="tissue box",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/6f8040c86eb84e7bba2cd928c0755029/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="hard drive",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/e32b159011544b8aa7cacd812636353e/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="remote control",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/d28b249d997b46c1b42771d8ce869902/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_remote_control_on_top_of_the_mixer_on_the_table.__remote_control_forth/20240824-163619_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopHandleRightMug1372Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="highlighter",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/25481a65cad54e2a956394ff2b2765cd/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="bottle",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/006_mustard_bottle_google_16k/006_mustard_bottle_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="watch",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/243b381dcdc34316a7e78a533572d273/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="mug",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/025_mug_google_16k/025_mug_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_mug_on_top_of_the_highlighter_on_the_table.__handle_right/20240824-230554_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopBallpointRightPen1373Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="shoe",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/aa51af2f1270482193471455b504efc6/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="mug",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/7bc2b14a48d1413e9375c32a53e3ee6f/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="pen",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/fbeb7c776372466daffa619106e0b2e0/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_pen_on_top_of_the_shoe_on_the_table.__ballpoint_right/20240824-172944_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopHandleRightHammer1374Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="shoe",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/d5a5f0a954f94bcea3168329d1605fe9/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="shoe",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/ece4453042144b4a82fb86a4eab1ba7f/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="lighter",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/b640aa4990b64db9ab3d868c6f49820e/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="hammer",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/6d155e8051294915813e0c156e3cd6de/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_hammer_on_top_of_the_shoe_on_the_table.__handle_right/20240824-200504_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopHandleRightHammer1375Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="shoe",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/ae7142127dd84ebbbe7762368ace452c/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="remote control",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/d72eebbf82be48f0a53e7e8b712e6a66/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="hammer",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/d499e1bc8a514e4bb5ca995ea6ba23b6/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_hammer_on_top_of_the_shoe_on_the_table.__handle_right/20240824-214604_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopBulbRightHandleLeftFlashlight1376Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="book",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/de79a920c48745ff95c0df7a7c300091/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="flashlight",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/123a79642c2646d8b315576828fea84a/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_flashlight_on_top_of_the_book_on_the_table.__bulb_right_handle_left/20240824-173905_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopLyingFlatBinder1377Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="screwdriver",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/044_flat_screwdriver_google_16k/044_flat_screwdriver_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="toy",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/9adc77036b434348ae049776c50df624/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="binder",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/78e016eeb0bf45d5bea47547128383a8/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_binder_on_top_of_the_screwdriver_on_the_table.__lying_flat/20240824-224017_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopUpsideDownTextualBox1378Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="cup",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/065-g_cups_google_16k/065-g_cups_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="box",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/009_gelatin_box_google_16k/009_gelatin_box_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_box_on_top_of_the_cup_on_the_table.__upside_down_textual/20240824-164753_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopHandleRightMug1379Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="mouse",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/969b0d9d99a8468898753fdcd219f883/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="mug",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/025_mug_google_16k/025_mug_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_mug_on_top_of_the_mouse_on_the_table.__handle_right/20240824-225707_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopHandleRightSpatula1380Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="toilet paper roll",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/6d5284b842434413a17133f7bf259669/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="hammer",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/35a76a67ea1c45edabbd5013de70d68d/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="scissors",
            urdf_path=(
                "data_isaaclab/assets/open6dor/ycb_16k_backup/037_scissors_google_16k/037_scissors_google_16k.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="spatula",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/033_spatula_google_16k/033_spatula_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_spatula_on_top_of_the_toilet_paper_roll_on_the_table.__handle_right/20240824-184239_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopUprightPowerDrill1381Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="apple",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/fbda0b25f41f40958ea984f460e4770b/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="knife",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/032_knife_google_16k/032_knife_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="mobile phone",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/45c813a9fac4458ead1f90280826c0a4/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="power drill",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/035_power_drill_google_16k/035_power_drill_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_power_drill_on_top_of_the_apple_on_the_table.__upright/20240824-214811_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopTipRightHighlighter1382Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="box",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/009_gelatin_box_google_16k/009_gelatin_box_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="microphone",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/5172dbe9281a45f48cee8c15bdfa1831/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="multimeter",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/79ac684e66b34554ba869bd2fc3c2653/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="highlighter",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/fd058916fb8843c09b4a2e05e3ee894e/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_highlighter_on_top_of_the_box_on_the_table.__tip_right/20240824-231217_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopLyingFlatEraser1383Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="screwdriver",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/044_flat_screwdriver_google_16k/044_flat_screwdriver_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="eraser",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/f0916a69ba514ecc973b44528b9dcc43/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_eraser_on_top_of_the_screwdriver_on_the_table.__lying_flat/20240824-181230_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopHandleLeftMug1384Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="headphone",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/ea91786392284f03809c37976da090bf/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="mug",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/de23895b6e9b4cfd870e30d14c2150dd/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_mug_on_top_of_the_headphone_on_the_table.__handle_left/20240824-184905_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopUprightHotGlueGun1385Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="cup",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/065-h_cups_google_16k/065-h_cups_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="remote control",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/d72eebbf82be48f0a53e7e8b712e6a66/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="bottle",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/8c0f8088a5d546c886d1b07e3f4eafae/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="hot glue gun",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/5ceec13e87774982afe825e7e74a0ce1/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_hot_glue_gun_on_top_of_the_cup_on_the_table.__upright/20240824-224334_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopUprightPowerDrill1386Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="bottle",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/a7785ac7ce7f4ba79df661db33637ba9/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="power drill",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/035_power_drill_google_16k/035_power_drill_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_power_drill_on_top_of_the_bottle_on_the_table.__upright/20240824-215113_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopLyingFlatBook1387Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="spoon",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/031_spoon_google_16k/031_spoon_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="remote control",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/d28b249d997b46c1b42771d8ce869902/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="hammer",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/d499e1bc8a514e4bb5ca995ea6ba23b6/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="book",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/2a371c84c5454f7facba7bb2f5312ad6/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_book_on_top_of_the_spoon_on_the_table.__lying_flat/20240824-164227_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopSidewaysBottle1388Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="apple",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/013_apple_google_16k/013_apple_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="bottle",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/022_windex_bottle_google_16k/022_windex_bottle_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="bottle",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/8c0f8088a5d546c886d1b07e3f4eafae/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_bottle_on_top_of_the_apple_on_the_table.__sideways/20240824-184233_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopRemoteControlForthRemoteControl1389Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="power drill",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/035_power_drill_google_16k/035_power_drill_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="remote control",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/89dcd45ac1f04680b76b709357a3dba3/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="remote control",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/c7ec86e7feb746489b30b4a01c2510af/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_remote_control_on_top_of_the_power_drill_on_the_table.__remote_control_forth/20240824-204341_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopSidewaysBottle1390Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="binder clips",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/5e55c38b24dc4c239448efa1c6b7f49f/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="tape measure",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/11307d06bb254318b95021d68c6fa12f/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="bottle",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/021_bleach_cleanser_google_16k/021_bleach_cleanser_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_bottle_on_top_of_the_binder_clips_on_the_table.__sideways/20240824-225730_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopHandleRightHammer1391Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="tissue box",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/8eda72f64e6e4443af4fdbeed07cad29/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="shoe",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/ae7142127dd84ebbbe7762368ace452c/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="wineglass",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/dc707f81a94e48078069c6a463f59fcf/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="hammer",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/4f24b866dab248b684149ec6bb40101f/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_hammer_on_top_of_the_tissue_box_on_the_table.__handle_right/20240824-163041_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopSidewaysCan1392Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="knife",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/f93d05bc2db144f7a50374794ef5cf8d/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="box",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/009_gelatin_box_google_16k/009_gelatin_box_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="can",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/005_tomato_soup_can_google_16k/005_tomato_soup_can_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_can_on_top_of_the_knife_on_the_table.__sideways/20240824-183302_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopUprightMouse1393Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="remote control",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/f1bbc61a42b94ee9a2976ca744f8962e/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="tape measure",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/11307d06bb254318b95021d68c6fa12f/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="mouse",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/969b0d9d99a8468898753fdcd219f883/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_mouse_on_top_of_the_remote_control_on_the_table.__upright/20240824-170057_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopUprightMouse1394Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="remote control",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/89dcd45ac1f04680b76b709357a3dba3/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="hammer",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/4f24b866dab248b684149ec6bb40101f/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="shoe",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/d6951661c1e445d8a1d00b7d38d86030/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="mouse",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/f496168739384cdb86ef6d65e2068a3f/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_mouse_on_top_of_the_remote_control_on_the_table.__upright/20240824-194431_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopClaspRightWallet1395Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="tape measure",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/d1d1c4b27f5c45a29910830e268f7ee2/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="mouse",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/e51e63e374ee4c25ae3fb4a9c74bd335/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="wallet",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/f47fdcf9615d4e94a71e6731242a4c94/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_wallet_on_top_of_the_tape_measure_on_the_table.__clasp_right/20240824-181526_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopUpsideDownCup1396Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="camera",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/f6f8675406b7437faaf96a1f82062028/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="can",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/002_master_chef_can_google_16k/002_master_chef_can_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="cup",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/065-h_cups_google_16k/065-h_cups_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_cup_on_top_of_the_camera_on_the_table.__upside_down/20240824-223700_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopUpsideDownCup1397Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="camera",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/6a63fc4af48b453c91ca2b335a4d464d/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="cup",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/065-h_cups_google_16k/065-h_cups_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_cup_on_top_of_the_camera_on_the_table.__upside_down/20240824-171431_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopUpsideDownCup1398Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="apple",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/f53d75bd123b40bca14d12d54286f432/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="cup",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/065-b_cups_google_16k/065-b_cups_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_cup_on_top_of_the_apple_on_the_table.__upside_down/20240824-214609_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopUpsideDownPlate1399Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="mouse",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/969b0d9d99a8468898753fdcd219f883/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="plate",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/029_plate_google_16k/029_plate_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_plate_on_top_of_the_mouse_on_the_table.__upside_down/20240824-163100_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopUpsideDownPlate1400Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="toilet paper roll",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/6d5284b842434413a17133f7bf259669/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="headphone",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/ea91786392284f03809c37976da090bf/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="plate",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/6d5a4e1174dd4f5fa0a9a9076e476b91/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_plate_on_top_of_the_toilet_paper_roll_on_the_table.__upside_down/20240824-200405_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopUprightStapler1401Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="wrench",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/042_adjustable_wrench_google_16k/042_adjustable_wrench_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="stapler",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/d2c345be5c084b4e8c92b48f7be69315/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_stapler_on_top_of_the_wrench_on_the_table.__upright/20240824-194015_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopUprightKeyboard1402Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="screwdriver",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/043_phillips_screwdriver_google_16k/043_phillips_screwdriver_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="wallet",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/ccfcc4ae6efa44a2ba34c4c479be7daf/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="mug",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/c609327dd3a74fb597584e1b4a14a615/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="keyboard",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/0161b6232216457aa8f4f3393960ec85/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_keyboard_on_top_of_the_screwdriver_on_the_table.__upright/20240824-193746_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopRemoteControlForthRemoteControl1403Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="lighter",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/b640aa4990b64db9ab3d868c6f49820e/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="remote control",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/89dcd45ac1f04680b76b709357a3dba3/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_remote_control_on_top_of_the_lighter_on_the_table.__remote_control_forth/20240824-190804_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopUpsideDownWineglass1404Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="hammer",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/9bf1b31f641543c9a921042dac3b527f/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="bottle",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/6add0a1fa81942acbb24504b130661c1/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="wineglass",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/dc707f81a94e48078069c6a463f59fcf/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_wineglass_on_top_of_the_hammer_on_the_table.__upside_down/20240824-192925_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopUpsideDownCalculator1405Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="scissors",
            urdf_path=(
                "data_isaaclab/assets/open6dor/ycb_16k_backup/037_scissors_google_16k/037_scissors_google_16k.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="cup",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/065-d_cups_google_16k/065-d_cups_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="calculator",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/054be0c3f09143f38c4d8038eb2588c6/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_calculator_on_top_of_the_scissors_on_the_table.__upside_down/20240824-212400_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopRemoteControlForthRemoteControl1406Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="shoe",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/d6951661c1e445d8a1d00b7d38d86030/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="remote control",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/d72eebbf82be48f0a53e7e8b712e6a66/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_remote_control_on_top_of_the_shoe_on_the_table.__remote_control_forth/20240824-182210_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopUprightKeyboard1407Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="plate",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/8d50c19e138b403e8b2ede9b47d8be3c/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="keyboard",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/44bb12d306864e2cb4256a61d4168942/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_keyboard_on_top_of_the_plate_on_the_table.__upright/20240824-191921_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopHandleLeftMug1408Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="USB",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/878049e8c3174fa68a56530e5aef7a5a/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="mug",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/2c88306f3d724a839054f3c2913fb1d5/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_mug_on_top_of_the_USB_on_the_table.__handle_left/20240824-224948_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopHandleLeftMug1409Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="pitcher base",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/019_pitcher_base_google_16k/019_pitcher_base_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="mug",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/ca4f9a92cc2f4ee98fe9332db41bf7f7/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_mug_on_top_of_the_pitcher_base_on_the_table.__handle_left/20240824-170611_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopLyingFlatEraser1410Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="calculator",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/69511a7fad2f42ee8c4b0579bbc8fec6/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="eraser",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/f0916a69ba514ecc973b44528b9dcc43/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_eraser_on_top_of_the_calculator_on_the_table.__lying_flat/20240824-164422_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopEarpieceFarHeadphone1411Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="cup",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/065-j_cups_google_16k/065-j_cups_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="book",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/b2d30398ba3741da9f9aef2319ee2b8b/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="watch",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/243b381dcdc34316a7e78a533572d273/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="headphone",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/ea91786392284f03809c37976da090bf/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_headphone_on_top_of_the_cup_on_the_table.__earpiece_far/20240824-192140_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopEarpieceFarHeadphone1412Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="cup",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/065-h_cups_google_16k/065-h_cups_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="headphone",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/ea91786392284f03809c37976da090bf/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_headphone_on_top_of_the_cup_on_the_table.__earpiece_far/20240824-232337_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopScoopRightSpoon1413Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="stapler",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/d2c345be5c084b4e8c92b48f7be69315/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="apple",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/fbda0b25f41f40958ea984f460e4770b/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="hammer",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/9bf1b31f641543c9a921042dac3b527f/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="spoon",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/031_spoon_google_16k/031_spoon_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_spoon_on_top_of_the_stapler_on_the_table.__scoop_right/20240824-160733_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopUpsideDownBowl1414Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="lighter",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/d9675ab05c39447baf27e19ea07d484e/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="microphone",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/5172dbe9281a45f48cee8c15bdfa1831/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="paperweight",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/21c10181c122474a8f72a4ad0331f185/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="bowl",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/bdd36c94f3f74f22b02b8a069c8d97b7/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_bowl_on_top_of_the_lighter_on_the_table.__upside_down/20240824-184519_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopSpoutRightPot1415Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="stapler",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/d2c345be5c084b4e8c92b48f7be69315/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="pot",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/cba3e4df7c354d22b7913d45a28a8159/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_pot_on_top_of_the_stapler_on_the_table.__spout_right/20240824-164954_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopLeftToy1416Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="multimeter",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/79ac684e66b34554ba869bd2fc3c2653/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="skillet",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/027_skillet_google_16k/027_skillet_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="toy",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/072-a_toy_airplane_google_16k/072-a_toy_airplane_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_toy_on_top_of_the_multimeter_on_the_table.__left/20240824-223345_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopUprightMouse1417Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="spoon",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/e1745c4aa7c046a2b5193d2d23be0192/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="bottle",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/794b730ae452424bb3a9ce3c6caaff7a/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="mouse",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/f017c73a85484b6da3a66ec1cda70c71/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_mouse_on_top_of_the_spoon_on_the_table.__upright/20240824-155559_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopRemoteControlForthRemoteControl1418Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="clock",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/fdf932b04e6c4e0fbd6e274563b94536/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="remote control",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/c7ec86e7feb746489b30b4a01c2510af/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_remote_control_on_top_of_the_clock_on_the_table.__remote_control_forth/20240824-200905_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopEarpieceFarHeadphone1419Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="wrench",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/042_adjustable_wrench_google_16k/042_adjustable_wrench_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="toy",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/072-a_toy_airplane_google_16k/072-a_toy_airplane_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="headphone",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/ea91786392284f03809c37976da090bf/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_headphone_on_top_of_the_wrench_on_the_table.__earpiece_far/20240824-181039_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopRemoteControlForthRemoteControl1420Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="microphone",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/5172dbe9281a45f48cee8c15bdfa1831/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="remote control",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/89dcd45ac1f04680b76b709357a3dba3/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_remote_control_on_top_of_the_microphone_on_the_table.__remote_control_forth/20240824-203354_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopBallpointRightPen1421Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="spoon",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/e1745c4aa7c046a2b5193d2d23be0192/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="calculator",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/f0b1f7c17d70489888f5ae922169c0ce/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="pen",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/fbeb7c776372466daffa619106e0b2e0/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_pen_on_top_of_the_spoon_on_the_table.__ballpoint_right/20240824-191842_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopRemoteControlForthRemoteControl1422Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="SD card",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/02f7f045679d402da9b9f280030821d4/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="mug",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/a564a11174be455bbce4698f7da92fdf/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="remote control",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/89dcd45ac1f04680b76b709357a3dba3/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_remote_control_on_top_of_the_SD_card_on_the_table.__remote_control_forth/20240824-233320_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopBladeRightKnife1423Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="can",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/007_tuna_fish_can_google_16k/007_tuna_fish_can_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="clipboard",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/b31c69728a414e639eef2fccd1c3dd75/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="knife",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/85db1d392a89459dbde5cdd951c4f6fb/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_knife_on_top_of_the_can_on_the_table.__blade_right/20240824-184845_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopUprightOrganizer1424Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="knife",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/032_knife_google_16k/032_knife_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="remote control",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/c7ec86e7feb746489b30b4a01c2510af/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="microphone",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/fac08019fef5474189f965bb495771eb/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="organizer",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/4f52ebe6c8cf432986e47d8de83386ce/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_organizer_on_top_of_the_knife_on_the_table.__upright/20240824-173110_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopLyingFlatBook1425Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="tape measure",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/7138f5e13b2e4e17975c23ba5584164b/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="wrench",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/19c1913ba85042938f0a87d94397c946/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="book",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/c61227cac7224b86b43c53ac2a2b6ec7/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_book_on_top_of_the_tape_measure_on_the_table.__lying_flat/20240824-194114_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopHandleRightHammer1426Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="can",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/002_master_chef_can_google_16k/002_master_chef_can_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="glue gun",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/a1da12b81f034db894496c1ffc4be1f5/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="hammer",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/8ed38a92668a425eb16da938622d9ace/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_hammer_on_top_of_the_can_on_the_table.__handle_right/20240824-203052_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopHandleRightHammer1427Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="scissors",
            urdf_path=(
                "data_isaaclab/assets/open6dor/ycb_16k_backup/037_scissors_google_16k/037_scissors_google_16k.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="cup",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/065-c_cups_google_16k/065-c_cups_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="box",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/9660e0c0326b4f7386014e27717231ae/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="hammer",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/35a76a67ea1c45edabbd5013de70d68d/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_hammer_on_top_of_the_scissors_on_the_table.__handle_right/20240824-231647_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopPlugRightUsb1428Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="calculator",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/b4bb8bd7791648fdb36c458fd9a877bf/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="binder",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/a0026e7d1b244b9a9223daf4223c9372/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="clipboard",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/b31c69728a414e639eef2fccd1c3dd75/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="USB",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/0a51815f3c0941ae8312fc6917173ed6/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_USB_on_top_of_the_calculator_on_the_table.__plug_right/20240824-221701_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopRollSidewaysToiletPaperRoll1429Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="wrench",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/8a6cb4f7b0004f53830e270dc6e1ff1d/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="toilet paper roll",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/6d5284b842434413a17133f7bf259669/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_toilet_paper_roll_on_top_of_the_wrench_on_the_table.__roll_sideways/20240824-224425_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopSidewaysShoe1430Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="USB",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/0a51815f3c0941ae8312fc6917173ed6/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="shoe",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/d6951661c1e445d8a1d00b7d38d86030/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_shoe_on_top_of_the_USB_on_the_table.__sideways/20240824-180612_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopUprightApple1431Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="highlighter",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/fd058916fb8843c09b4a2e05e3ee894e/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="apple",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/fbda0b25f41f40958ea984f460e4770b/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_apple_on_top_of_the_highlighter_on_the_table.__upright/20240824-221411_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopUpsideDownCup1432Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="spatula",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/4b27628be0804c0285d0b13dda025a0d/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="tape measure",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/7138f5e13b2e4e17975c23ba5584164b/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="toy",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/9adc77036b434348ae049776c50df624/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="cup",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/065-h_cups_google_16k/065-h_cups_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_cup_on_top_of_the_spatula_on_the_table.__upside_down/20240824-212537_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopUpsideDownCup1433Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="spatula",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/4b27628be0804c0285d0b13dda025a0d/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="hammer",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/d499e1bc8a514e4bb5ca995ea6ba23b6/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="cup",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/065-g_cups_google_16k/065-g_cups_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_cup_on_top_of_the_spatula_on_the_table.__upside_down/20240824-164949_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopHandleLeftMug1434Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="scissors",
            urdf_path=(
                "data_isaaclab/assets/open6dor/ycb_16k_backup/037_scissors_google_16k/037_scissors_google_16k.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="apple",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/013_apple_google_16k/013_apple_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="mug",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/a564a11174be455bbce4698f7da92fdf/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_mug_on_top_of_the_scissors_on_the_table.__handle_left/20240824-184031_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopUpsideDownCalculator1435Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="hammer",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/4f24b866dab248b684149ec6bb40101f/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="tape measure",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/11307d06bb254318b95021d68c6fa12f/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="calculator",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/054be0c3f09143f38c4d8038eb2588c6/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_calculator_on_top_of_the_hammer_on_the_table.__upside_down/20240824-203458_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopScoopRightSpoon1436Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="hat",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/dc02f9c1fd01432fadd6e7d15851dc37/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="hammer",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/048_hammer_google_16k/048_hammer_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="spoon",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/2613c5f24bb3407da2664c410c34d523/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_spoon_on_top_of_the_hat_on_the_table.__scoop_right/20240824-202213_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopUprightTissueBox1437Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="keyboard",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/0161b6232216457aa8f4f3393960ec85/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="tissue box",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/213d8df9c37a4d8cba794965676f7a75/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_tissue_box_on_top_of_the_keyboard_on_the_table.__upright/20240824-194820_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopRemoteControlForthRemoteControl1438Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="hard drive",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/f8de4975c0f54c1a97203c6a674f6a39/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="remote control",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/d72eebbf82be48f0a53e7e8b712e6a66/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_remote_control_on_top_of_the_hard_drive_on_the_table.__remote_control_forth/20240824-173658_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopHandleRightWrench1439Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="binder",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/a0026e7d1b244b9a9223daf4223c9372/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="pot",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/af249e607bea40cfa2f275e5e23b8283/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="wineglass",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/afca0b1933b84b1dbebd1244f25e72fc/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="wrench",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/19c1913ba85042938f0a87d94397c946/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_wrench_on_top_of_the_binder_on_the_table.__handle_right/20240824-201622_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopUprightApple1440Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="mouse",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/e51e63e374ee4c25ae3fb4a9c74bd335/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="mug",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/922cd7d18c6748d49fe651ded8a04cf4/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="apple",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/f53d75bd123b40bca14d12d54286f432/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_apple_on_top_of_the_mouse_on_the_table.__upright/20240824-164936_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopUprightOrganizer1441Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="box",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/003_cracker_box_google_16k/003_cracker_box_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="mug",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/3ea2fd4e065147048064f4c97a89fe6f/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="USB",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/b1098389cc3b460bb1f1ce558d4b0764/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="organizer",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/4f52ebe6c8cf432986e47d8de83386ce/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_organizer_on_top_of_the_box_on_the_table.__upright/20240824-231032_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopLowerRimGlasses1442Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="SD card",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/02f7f045679d402da9b9f280030821d4/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="mug",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/de23895b6e9b4cfd870e30d14c2150dd/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="marker",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/040_large_marker_google_16k/040_large_marker_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="glasses",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/e4f348e98ceb45e3abc77da5b738f1b2/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_glasses_on_top_of_the_SD_card_on_the_table.__lower_rim/20240824-221620_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopLyingFlatBinder1443Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="wrench",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/5bf781c9fd8d4121b735503b13ba2eaf/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="fork",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/e98e7fb33743442383812b68608f7006/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="binder",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/a0026e7d1b244b9a9223daf4223c9372/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_binder_on_top_of_the_wrench_on_the_table.__lying_flat/20240824-231704_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopUpsideDownBowl1444Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="fork",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/e98e7fb33743442383812b68608f7006/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="toilet paper roll",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/ea0cf48616e34708b73c82eb7c7366ca/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="wineglass",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/dc707f81a94e48078069c6a463f59fcf/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="bowl",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/024_bowl_google_16k/024_bowl_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_bowl_on_top_of_the_fork_on_the_table.__upside_down/20240824-161144_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopLyingFlatEraser1445Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="pen",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/f126e15c0f904fccbd76f9348baea12c/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="scissors",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/2b1120c4b517409184413ed709c57a66/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="calculator",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/f0b1f7c17d70489888f5ae922169c0ce/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="eraser",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/f0916a69ba514ecc973b44528b9dcc43/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_eraser_on_top_of_the_pen_on_the_table.__lying_flat/20240824-162200_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopCapRightBottomLeftBottle1446Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="orange",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/a4c9a503f63e440e8d6d924d4e5c36b1/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="pitcher base",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/019_pitcher_base_google_16k/019_pitcher_base_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="bottle",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/006_mustard_bottle_google_16k/006_mustard_bottle_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_bottle_on_top_of_the_orange_on_the_table.__cap_right_bottom_left/20240824-224542_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopUprightKeyboard1447Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="shoe",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/ece4453042144b4a82fb86a4eab1ba7f/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="keyboard",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/0161b6232216457aa8f4f3393960ec85/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_keyboard_on_top_of_the_shoe_on_the_table.__upright/20240824-223613_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopHandleRightScrewdriver1448Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="shoe",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/d6951661c1e445d8a1d00b7d38d86030/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="hard drive",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/e32b159011544b8aa7cacd812636353e/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="screwdriver",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/044_flat_screwdriver_google_16k/044_flat_screwdriver_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="screwdriver",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/043_phillips_screwdriver_google_16k/043_phillips_screwdriver_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_screwdriver_on_top_of_the_shoe_on_the_table.__handle_right/20240824-231509_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopLighterForthLighter1449Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="cap",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/87e82aa304ce4571b69bdd5182549c72/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="lighter",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/b640aa4990b64db9ab3d868c6f49820e/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_lighter_on_top_of_the_cap_on_the_table.__lighter_forth/20240824-190127_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopUpsideDownBowl1450Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="toy",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/9adc77036b434348ae049776c50df624/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="bowl",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/024_bowl_google_16k/024_bowl_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_bowl_on_top_of_the_toy_on_the_table.__upside_down/20240824-231710_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopUpsideDownTextualBox1451Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="wrench",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/042_adjustable_wrench_google_16k/042_adjustable_wrench_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="box",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/008_pudding_box_google_16k/008_pudding_box_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_box_on_top_of_the_wrench_on_the_table.__upside_down_textual/20240824-180254_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopUprightMouse1452Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="book",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/b2d30398ba3741da9f9aef2319ee2b8b/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="toilet paper roll",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/ea0cf48616e34708b73c82eb7c7366ca/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="hammer",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/6d155e8051294915813e0c156e3cd6de/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="mouse",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/e51e63e374ee4c25ae3fb4a9c74bd335/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_mouse_on_top_of_the_book_on_the_table.__upright/20240824-180103_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopUpsideDownCalculator1453Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="shoe",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/ae7142127dd84ebbbe7762368ace452c/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="calculator",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/f0b1f7c17d70489888f5ae922169c0ce/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_calculator_on_top_of_the_shoe_on_the_table.__upside_down/20240824-182336_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopUpsideDownCalculator1454Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="wallet",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/dbb07d13a33546f09ac8ca98b1ddef20/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="calculator",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/b4bb8bd7791648fdb36c458fd9a877bf/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_calculator_on_top_of_the_wallet_on_the_table.__upside_down/20240824-201117_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopRightToy1455Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="book",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/2a371c84c5454f7facba7bb2f5312ad6/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="power drill",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/035_power_drill_google_16k/035_power_drill_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="spoon",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/2613c5f24bb3407da2664c410c34d523/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="toy",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/072-b_toy_airplane_google_16k/072-b_toy_airplane_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_toy_on_top_of_the_book_on_the_table.__right/20240824-181538_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopSidewaysTextualBox1456Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="keyboard",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/0161b6232216457aa8f4f3393960ec85/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="remote control",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/89dcd45ac1f04680b76b709357a3dba3/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="box",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/5e1a1527a0bd49ed8e473079392c31c8/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_box_on_top_of_the_keyboard_on_the_table.__sideways_textual/20240824-190031_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopUprightApple1457Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="remote control",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/89dcd45ac1f04680b76b709357a3dba3/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="apple",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/fbda0b25f41f40958ea984f460e4770b/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_apple_on_top_of_the_remote_control_on_the_table.__upright/20240824-193943_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopUprightApple1458Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="remote control",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/d28b249d997b46c1b42771d8ce869902/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="apple",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/f53d75bd123b40bca14d12d54286f432/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_apple_on_top_of_the_remote_control_on_the_table.__upright/20240824-182944_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopHandleRightScrewdriver1459Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="wrench",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/042_adjustable_wrench_google_16k/042_adjustable_wrench_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="screwdriver",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/044_flat_screwdriver_google_16k/044_flat_screwdriver_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_screwdriver_on_top_of_the_wrench_on_the_table.__handle_right/20240824-172624_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopHatSidewaysHat1460Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="mug",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/3ea2fd4e065147048064f4c97a89fe6f/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="can",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/010_potted_meat_can_google_16k/010_potted_meat_can_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="hat",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/dc02f9c1fd01432fadd6e7d15851dc37/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_hat_on_top_of_the_mug_on_the_table.__hat_sideways/20240824-231839_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopTapeMeasureUprightTapeMeasure1461Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="cup",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/065-g_cups_google_16k/065-g_cups_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="mug",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/9321c45cb9cf459f9f803507d3a11fb3/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="mug",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/922cd7d18c6748d49fe651ded8a04cf4/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="tape measure",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/d1d1c4b27f5c45a29910830e268f7ee2/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_tape_measure_on_top_of_the_cup_on_the_table.__tape_measure_upright/20240824-194335_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopTapeMeasureUprightTapeMeasure1462Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="cup",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/065-b_cups_google_16k/065-b_cups_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="hammer",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/4f24b866dab248b684149ec6bb40101f/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="tape measure",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/1574b13827734d54b6d9d86be8be71d1/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_tape_measure_on_top_of_the_cup_on_the_table.__tape_measure_upright/20240824-224930_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopUpsideDownClipboard1463Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="book",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/c61227cac7224b86b43c53ac2a2b6ec7/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="clipboard",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/a37158f20ccf436483029e8295629738/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_clipboard_on_top_of_the_book_on_the_table.__upside_down/20240824-224754_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopBladeRightKnife1464Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="knife",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/0df75457524743ec81468c916fddb930/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="toy",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/9adc77036b434348ae049776c50df624/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="spatula",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/033_spatula_google_16k/033_spatula_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="knife",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/032_knife_google_16k/032_knife_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_knife_on_top_of_the_knife_on_the_table.__blade_right/20240824-190015_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopSidewaysBottle1465Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="knife",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/f93d05bc2db144f7a50374794ef5cf8d/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="binder",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/a0026e7d1b244b9a9223daf4223c9372/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="toy",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/072-b_toy_airplane_google_16k/072-b_toy_airplane_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="bottle",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/8c0f8088a5d546c886d1b07e3f4eafae/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_bottle_on_top_of_the_knife_on_the_table.__sideways/20240824-175151_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopUprightHotGlueGun1466Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="highlighter",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/fd058916fb8843c09b4a2e05e3ee894e/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="hot glue gun",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/5ceec13e87774982afe825e7e74a0ce1/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_hot_glue_gun_on_top_of_the_highlighter_on_the_table.__upright/20240824-213118_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopUpsideDownBowl1467Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="USB",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/b1098389cc3b460bb1f1ce558d4b0764/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="shoe",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/aa51af2f1270482193471455b504efc6/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="headphone",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/f8891134871e48ce8cb5053c4287272b/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="bowl",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/bdd36c94f3f74f22b02b8a069c8d97b7/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_bowl_on_top_of_the_USB_on_the_table.__upside_down/20240824-225247_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopCapLeftBottomRightBottle1468Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="cup",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/065-i_cups_google_16k/065-i_cups_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="bottle",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/d17b8f44d1544364b8ad9fe2554ddfdf/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_bottle_on_top_of_the_cup_on_the_table.__cap_left_bottom_right/20240824-212036_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopCapRightBottomLeftBottle1469Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="cup",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/065-i_cups_google_16k/065-i_cups_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="calculator",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/d9ffbc4bbe1044bb902e1dddac52b0de/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="bottle",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/021_bleach_cleanser_google_16k/021_bleach_cleanser_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_bottle_on_top_of_the_cup_on_the_table.__cap_right_bottom_left/20240824-201611_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopUprightKeyboard1470Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="calipers",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/95c967114b624291bc26bcd705aaa334/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="mouse",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/f017c73a85484b6da3a66ec1cda70c71/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="keyboard",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/f5080a3df6d847b693c5cca415886c61/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_keyboard_on_top_of_the_calipers_on_the_table.__upright/20240824-203637_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopTapeMeasureUprightTapeMeasure1471Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="eraser",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/f0916a69ba514ecc973b44528b9dcc43/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="plate",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/8d50c19e138b403e8b2ede9b47d8be3c/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="box",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/9660e0c0326b4f7386014e27717231ae/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="tape measure",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/1574b13827734d54b6d9d86be8be71d1/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_tape_measure_on_top_of_the_eraser_on_the_table.__tape_measure_upright/20240824-193240_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopUpsideDownCup1472Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="scissors",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/2b1120c4b517409184413ed709c57a66/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="pen",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/fbeb7c776372466daffa619106e0b2e0/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="cup",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/065-d_cups_google_16k/065-d_cups_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_cup_on_top_of_the_scissors_on_the_table.__upside_down/20240824-225942_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopHatSidewaysHat1473Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="shoe",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/d6951661c1e445d8a1d00b7d38d86030/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="cup",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/065-f_cups_google_16k/065-f_cups_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="hat",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/dc02f9c1fd01432fadd6e7d15851dc37/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_hat_on_top_of_the_shoe_on_the_table.__hat_sideways/20240824-224231_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopUpsideDownCup1474Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="SD card",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/02f7f045679d402da9b9f280030821d4/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="box",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/9660e0c0326b4f7386014e27717231ae/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="cup",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/065-f_cups_google_16k/065-f_cups_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_cup_on_top_of_the_SD_card_on_the_table.__upside_down/20240824-194133_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopUprightTextualSpeaker1475Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="spoon",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/218768789fdd42bd95eb933046698499/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="speaker",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/4445f4442a884999960e7e6660459095/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_speaker_on_top_of_the_spoon_on_the_table.__upright_textual/20240824-222608_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopUprightTextualSpeaker1476Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="book",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/c61227cac7224b86b43c53ac2a2b6ec7/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="binder",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/78e016eeb0bf45d5bea47547128383a8/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="speaker",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/4445f4442a884999960e7e6660459095/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_speaker_on_top_of_the_book_on_the_table.__upright_textual/20240824-231447_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopUpsideDownTextualBox1477Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="flashlight",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/123a79642c2646d8b315576828fea84a/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="mug",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/3ea2fd4e065147048064f4c97a89fe6f/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="headphone",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/ea91786392284f03809c37976da090bf/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="box",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/f81ca566b008446eb704cad4844603b6/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_box_on_top_of_the_flashlight_on_the_table.__upside_down_textual/20240824-214655_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopUprightTissueBox1478Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="camera",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/f6f8675406b7437faaf96a1f82062028/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="tissue box",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/8eda72f64e6e4443af4fdbeed07cad29/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_tissue_box_on_top_of_the_camera_on_the_table.__upright/20240824-155714_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopUpsideDownCup1479Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="headphone",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/ea91786392284f03809c37976da090bf/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="cup",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/065-d_cups_google_16k/065-d_cups_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_cup_on_top_of_the_headphone_on_the_table.__upside_down/20240824-184629_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopSidewaysTextualBox1480Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="tissue box",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/8eda72f64e6e4443af4fdbeed07cad29/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="box",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/5e1a1527a0bd49ed8e473079392c31c8/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_box_on_top_of_the_tissue_box_on_the_table.__sideways_textual/20240824-170830_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopCapLeftBottomRightBottle1481Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="toy",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/072-a_toy_airplane_google_16k/072-a_toy_airplane_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="bottle",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/d17b8f44d1544364b8ad9fe2554ddfdf/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_bottle_on_top_of_the_toy_on_the_table.__cap_left_bottom_right/20240824-192038_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopLeftToy1482Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="bottle",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/022_windex_bottle_google_16k/022_windex_bottle_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="toy",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/072-a_toy_airplane_google_16k/072-a_toy_airplane_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_toy_on_top_of_the_bottle_on_the_table.__left/20240824-161629_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopLowerRimGlasses1483Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="watch",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/243b381dcdc34316a7e78a533572d273/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="glasses",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/e4f348e98ceb45e3abc77da5b738f1b2/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_glasses_on_top_of_the_watch_on_the_table.__lower_rim/20240824-210426_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopSpoutRightPot1484Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="mug",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/7bc2b14a48d1413e9375c32a53e3ee6f/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="pot",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/af249e607bea40cfa2f275e5e23b8283/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_pot_on_top_of_the_mug_on_the_table.__spout_right/20240824-203228_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopHandleRightWrench1485Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="pen",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/f126e15c0f904fccbd76f9348baea12c/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="box",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/009_gelatin_box_google_16k/009_gelatin_box_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="shoe",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/d6951661c1e445d8a1d00b7d38d86030/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="wrench",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/8a6cb4f7b0004f53830e270dc6e1ff1d/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_wrench_on_top_of_the_pen_on_the_table.__handle_right/20240824-170546_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopBulbRightHandleLeftFlashlight1486Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="bottle",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/8c0f8088a5d546c886d1b07e3f4eafae/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="flashlight",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/123a79642c2646d8b315576828fea84a/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_flashlight_on_top_of_the_bottle_on_the_table.__bulb_right_handle_left/20240824-210506_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopBulbRightHandleLeftFlashlight1487Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="bottle",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/022_windex_bottle_google_16k/022_windex_bottle_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="flashlight",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/123a79642c2646d8b315576828fea84a/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_flashlight_on_top_of_the_bottle_on_the_table.__bulb_right_handle_left/20240824-205320_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopLighterForthLighter1488Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="knife",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/85db1d392a89459dbde5cdd951c4f6fb/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="cup",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/065-f_cups_google_16k/065-f_cups_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="box",
            urdf_path=(
                "data_isaaclab/assets/open6dor/ycb_16k_backup/004_sugar_box_google_16k/004_sugar_box_google_16k.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="lighter",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/b640aa4990b64db9ab3d868c6f49820e/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_lighter_on_top_of_the_knife_on_the_table.__lighter_forth/20240824-181929_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopHandleRightMug1489Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="watch",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/243b381dcdc34316a7e78a533572d273/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="cup",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/065-f_cups_google_16k/065-f_cups_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="mug",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/a564a11174be455bbce4698f7da92fdf/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_mug_on_top_of_the_watch_on_the_table.__handle_right/20240824-195515_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopHatSidewaysHat1490Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="cup",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/065-i_cups_google_16k/065-i_cups_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="hammer",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/4f24b866dab248b684149ec6bb40101f/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="hat",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/dc02f9c1fd01432fadd6e7d15851dc37/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_hat_on_top_of_the_cup_on_the_table.__hat_sideways/20240824-231637_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopHandleRightWrench1491Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="wallet",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/dbb07d13a33546f09ac8ca98b1ddef20/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="watch",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/243b381dcdc34316a7e78a533572d273/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="wrench",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/19c1913ba85042938f0a87d94397c946/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_wrench_on_top_of_the_wallet_on_the_table.__handle_right/20240824-161403_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopHandleRightSpatula1492Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="spoon",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/2613c5f24bb3407da2664c410c34d523/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="spatula",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/4b27628be0804c0285d0b13dda025a0d/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_spatula_on_top_of_the_spoon_on_the_table.__handle_right/20240824-181729_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopUprightOrange1493Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="book",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/b2d30398ba3741da9f9aef2319ee2b8b/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="calculator",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/b4bb8bd7791648fdb36c458fd9a877bf/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="orange",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/a4c9a503f63e440e8d6d924d4e5c36b1/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_orange_on_top_of_the_book_on_the_table.__upright/20240824-223944_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopUprightOrganizer1494Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="pear",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/f3dd3064db4f4e8880344425970cecad/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="organizer",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/4f52ebe6c8cf432986e47d8de83386ce/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_organizer_on_top_of_the_pear_on_the_table.__upright/20240824-220800_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopCapForthCap1495Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="cup",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/065-c_cups_google_16k/065-c_cups_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="cap",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/87e82aa304ce4571b69bdd5182549c72/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_cap_on_top_of_the_cup_on_the_table.__cap_forth/20240824-173233_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopRightToy1496Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="mug",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/ca4f9a92cc2f4ee98fe9332db41bf7f7/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="mug",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/2c88306f3d724a839054f3c2913fb1d5/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="toy",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/073-g_lego_duplo_google_16k/073-g_lego_duplo_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_toy_on_top_of_the_mug_on_the_table.__right/20240824-173335_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopRightToy1497Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="mug",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/ca4f9a92cc2f4ee98fe9332db41bf7f7/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="toy",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/9adc77036b434348ae049776c50df624/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_toy_on_top_of_the_mug_on_the_table.__right/20240824-204652_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopSidewaysShoe1498Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="watch",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/d6e38c73f8f2402aac890e8ebb115302/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="keyboard",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/f5080a3df6d847b693c5cca415886c61/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="knife",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/032_knife_google_16k/032_knife_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="shoe",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/d6951661c1e445d8a1d00b7d38d86030/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_shoe_on_top_of_the_watch_on_the_table.__sideways/20240824-192429_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopLeftToy1499Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="tissue box",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/213d8df9c37a4d8cba794965676f7a75/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="apple",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/fbda0b25f41f40958ea984f460e4770b/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="toy",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/072-a_toy_airplane_google_16k/072-a_toy_airplane_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_toy_on_top_of_the_tissue_box_on_the_table.__left/20240824-195947_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopHandleRightMug1500Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="mug",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/b8478cea51454555b90de0fe6ba7ba83/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="credit card",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/5de830b2cccf4fe7a2e6b400abf26ca7/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="mug",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/ca4f9a92cc2f4ee98fe9332db41bf7f7/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_mug_on_top_of_the_mug_on_the_table.__handle_right/20240824-193219_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopUprightMicrophone1501Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="headphone",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/ea91786392284f03809c37976da090bf/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="USB",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/0a51815f3c0941ae8312fc6917173ed6/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="spatula",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/4b27628be0804c0285d0b13dda025a0d/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="microphone",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/f9d07bb386a04de8b0e1a9a28a936985/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_microphone_on_top_of_the_headphone_on_the_table.__upright/20240824-194511_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopEarpieceFarHeadphone1502Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="USB",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/b1098389cc3b460bb1f1ce558d4b0764/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="headphone",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/f8891134871e48ce8cb5053c4287272b/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_headphone_on_top_of_the_USB_on_the_table.__earpiece_far/20240824-210030_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopHandleRightPitcherBase1503Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="mobile phone",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/45c813a9fac4458ead1f90280826c0a4/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="pitcher base",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/019_pitcher_base_google_16k/019_pitcher_base_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_pitcher_base_on_top_of_the_mobile_phone_on_the_table.__handle_right/20240824-193052_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopSidewaysTextualBox1504Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="bowl",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/bdd36c94f3f74f22b02b8a069c8d97b7/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="skillet",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/027_skillet_google_16k/027_skillet_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="box",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/f81ca566b008446eb704cad4844603b6/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_box_on_top_of_the_bowl_on_the_table.__sideways_textual/20240824-232252_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopUprightKeyboard1505Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="mouse",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/f017c73a85484b6da3a66ec1cda70c71/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="keyboard",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/f5080a3df6d847b693c5cca415886c61/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_keyboard_on_top_of_the_mouse_on_the_table.__upright/20240824-180458_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopHandleRightHammer1506Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="bowl",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/bf83213efac34b3cb2ad6ac5ddaf05d9/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="lighter",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/b640aa4990b64db9ab3d868c6f49820e/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="tape measure",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/11307d06bb254318b95021d68c6fa12f/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="hammer",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/8ed38a92668a425eb16da938622d9ace/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_hammer_on_top_of_the_bowl_on_the_table.__handle_right/20240824-233241_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopUprightHotGlueGun1507Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="glasses",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/e4f348e98ceb45e3abc77da5b738f1b2/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="plate",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/8d50c19e138b403e8b2ede9b47d8be3c/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="can",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/007_tuna_fish_can_google_16k/007_tuna_fish_can_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="hot glue gun",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/5ceec13e87774982afe825e7e74a0ce1/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_hot_glue_gun_on_top_of_the_glasses_on_the_table.__upright/20240824-223938_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopUpsideDownWineglass1508Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="scissors",
            urdf_path=(
                "data_isaaclab/assets/open6dor/ycb_16k_backup/037_scissors_google_16k/037_scissors_google_16k.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="wineglass",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/dc707f81a94e48078069c6a463f59fcf/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_wineglass_on_top_of_the_scissors_on_the_table.__upside_down/20240824-212144_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopRollSidewaysToiletPaperRoll1509Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="pen",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/f126e15c0f904fccbd76f9348baea12c/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="mug",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/c609327dd3a74fb597584e1b4a14a615/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="USB",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/878049e8c3174fa68a56530e5aef7a5a/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="toilet paper roll",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/ea0cf48616e34708b73c82eb7c7366ca/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_toilet_paper_roll_on_top_of_the_pen_on_the_table.__roll_sideways/20240824-165818_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopScoopRightSpoon1510Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="mug",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/a564a11174be455bbce4698f7da92fdf/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="keyboard",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/f5080a3df6d847b693c5cca415886c61/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="spoon",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/e1745c4aa7c046a2b5193d2d23be0192/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_spoon_on_top_of_the_mug_on_the_table.__scoop_right/20240824-173957_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopUprightApple1511Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="toilet paper roll",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/7ca7ebcfad964498b49af73be442acf9/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="apple",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/f53d75bd123b40bca14d12d54286f432/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_apple_on_top_of_the_toilet_paper_roll_on_the_table.__upright/20240824-185428_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopBladeRightKnife1512Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="mug",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/e5e87ddbf3f6470384bef58431351e2a/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="plate",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/6d5a4e1174dd4f5fa0a9a9076e476b91/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="knife",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/f93d05bc2db144f7a50374794ef5cf8d/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_knife_on_top_of_the_mug_on_the_table.__blade_right/20240824-204614_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopLyingFlatHardDrive1513Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="flashlight",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/123a79642c2646d8b315576828fea84a/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="hard drive",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/e32b159011544b8aa7cacd812636353e/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_hard_drive_on_top_of_the_flashlight_on_the_table.__lying_flat/20240824-170854_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopUpsideDownCup1514Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="mixer",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/990d0cb9499540fda49b1ff36be9ba26/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="pot",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/cba3e4df7c354d22b7913d45a28a8159/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="cup",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/065-d_cups_google_16k/065-d_cups_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_cup_on_top_of_the_mixer_on_the_table.__upside_down/20240824-214954_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopClaspRightWallet1515Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="microphone",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/f9d07bb386a04de8b0e1a9a28a936985/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="calculator",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/b4bb8bd7791648fdb36c458fd9a877bf/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="lighter",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/d9675ab05c39447baf27e19ea07d484e/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="wallet",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/f47fdcf9615d4e94a71e6731242a4c94/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_wallet_on_top_of_the_microphone_on_the_table.__clasp_right/20240824-212632_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopUpsideDownCup1516Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="flashlight",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/123a79642c2646d8b315576828fea84a/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="watch",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/d6e38c73f8f2402aac890e8ebb115302/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="box",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/5e1a1527a0bd49ed8e473079392c31c8/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="cup",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/065-e_cups_google_16k/065-e_cups_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_cup_on_top_of_the_flashlight_on_the_table.__upside_down/20240824-164048_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopSidewaysTextualBox1517Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="mug",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/922cd7d18c6748d49fe651ded8a04cf4/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="box",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/5e1a1527a0bd49ed8e473079392c31c8/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_box_on_top_of_the_mug_on_the_table.__sideways_textual/20240824-191422_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopSidewaysTextualBox1518Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="mug",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/c25119c5ac6e4654be3b75d78e34a912/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="hammer",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/4f24b866dab248b684149ec6bb40101f/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="box",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/003_cracker_box_google_16k/003_cracker_box_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_box_on_top_of_the_mug_on_the_table.__sideways_textual/20240824-205307_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopSidewaysTextualBox1519Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="eraser",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/f0916a69ba514ecc973b44528b9dcc43/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="cup",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/065-d_cups_google_16k/065-d_cups_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="box",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/5e1a1527a0bd49ed8e473079392c31c8/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_box_on_top_of_the_eraser_on_the_table.__sideways_textual/20240824-165211_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopHandleRightPitcherBase1520Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="hammer",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/6d155e8051294915813e0c156e3cd6de/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="knife",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/032_knife_google_16k/032_knife_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="pitcher base",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/019_pitcher_base_google_16k/019_pitcher_base_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_pitcher_base_on_top_of_the_hammer_on_the_table.__handle_right/20240824-181544_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopTapeMeasureUprightTapeMeasure1521Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="bowl",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/bdd36c94f3f74f22b02b8a069c8d97b7/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="tape measure",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/11307d06bb254318b95021d68c6fa12f/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_tape_measure_on_top_of_the_bowl_on_the_table.__tape_measure_upright/20240824-185922_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopUprightMicrophone1522Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="can",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/002_master_chef_can_google_16k/002_master_chef_can_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="cup",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/065-d_cups_google_16k/065-d_cups_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="box",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/008_pudding_box_google_16k/008_pudding_box_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="microphone",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/fac08019fef5474189f965bb495771eb/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_microphone_on_top_of_the_can_on_the_table.__upright/20240824-174937_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopHandleRightWrench1523Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="hammer",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/8ed38a92668a425eb16da938622d9ace/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="toy",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/073-g_lego_duplo_google_16k/073-g_lego_duplo_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="shoe",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/aa51af2f1270482193471455b504efc6/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="wrench",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/8a6cb4f7b0004f53830e270dc6e1ff1d/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_wrench_on_top_of_the_hammer_on_the_table.__handle_right/20240824-200558_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopSidewaysTextualBox1524Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="remote control",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/d72eebbf82be48f0a53e7e8b712e6a66/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="camera",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/6a63fc4af48b453c91ca2b335a4d464d/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="cup",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/065-a_cups_google_16k/065-a_cups_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="box",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/9660e0c0326b4f7386014e27717231ae/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_box_on_top_of_the_remote_control_on_the_table.__sideways_textual/20240824-183438_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopBallpointRightPen1525Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="box",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/f81ca566b008446eb704cad4844603b6/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="calculator",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/f0b1f7c17d70489888f5ae922169c0ce/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="pen",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/fbeb7c776372466daffa619106e0b2e0/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_pen_on_top_of_the_box_on_the_table.__ballpoint_right/20240824-171528_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopTapeMeasureUprightTapeMeasure1526Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="mixer",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/990d0cb9499540fda49b1ff36be9ba26/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="binder",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/78e016eeb0bf45d5bea47547128383a8/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="cup",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/065-h_cups_google_16k/065-h_cups_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="tape measure",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/7138f5e13b2e4e17975c23ba5584164b/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_tape_measure_on_top_of_the_mixer_on_the_table.__tape_measure_upright/20240824-164825_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopUpsideDownTextualBox1527Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="book",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/de79a920c48745ff95c0df7a7c300091/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="hammer",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/35a76a67ea1c45edabbd5013de70d68d/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="box",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/9660e0c0326b4f7386014e27717231ae/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_box_on_top_of_the_book_on_the_table.__upside_down_textual/20240824-212337_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopHandleRightMug1528Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="knife",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/0df75457524743ec81468c916fddb930/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="book",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/2a371c84c5454f7facba7bb2f5312ad6/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="keyboard",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/f5080a3df6d847b693c5cca415886c61/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="mug",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/7bc2b14a48d1413e9375c32a53e3ee6f/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_mug_on_top_of_the_knife_on_the_table.__handle_right/20240824-183133_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopUprightOrganizer1529Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="pot",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/af249e607bea40cfa2f275e5e23b8283/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="organizer",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/4f52ebe6c8cf432986e47d8de83386ce/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_organizer_on_top_of_the_pot_on_the_table.__upright/20240824-212115_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopUprightMicrophone1530Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="bottle",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/8c0f8088a5d546c886d1b07e3f4eafae/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="microphone",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/f9d07bb386a04de8b0e1a9a28a936985/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_microphone_on_top_of_the_bottle_on_the_table.__upright/20240824-181746_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopRightToy1531Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="tape measure",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/11307d06bb254318b95021d68c6fa12f/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="toy",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/073-g_lego_duplo_google_16k/073-g_lego_duplo_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_toy_on_top_of_the_tape_measure_on_the_table.__right/20240824-221337_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopUprightTissueBox1532Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="toy",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/9adc77036b434348ae049776c50df624/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="envelope box",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/40b28d1cbfbd4e9f9acf653b748324ee/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="tissue box",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/6e9e9135bb5747f58d388e1e92477f02/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_tissue_box_on_top_of_the_toy_on_the_table.__upright/20240824-233815_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopLowerRimGlasses1533Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="binder",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/78e016eeb0bf45d5bea47547128383a8/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="glasses",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/e4f348e98ceb45e3abc77da5b738f1b2/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_glasses_on_top_of_the_binder_on_the_table.__lower_rim/20240824-193104_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopHandleLeftMug1534Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="spoon",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/218768789fdd42bd95eb933046698499/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="remote control",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/f1bbc61a42b94ee9a2976ca744f8962e/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="cup",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/065-h_cups_google_16k/065-h_cups_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="mug",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/9321c45cb9cf459f9f803507d3a11fb3/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_mug_on_top_of_the_spoon_on_the_table.__handle_left/20240824-215718_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopHandleRightHammer1535Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="pitcher base",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/019_pitcher_base_google_16k/019_pitcher_base_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="hammer",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/35a76a67ea1c45edabbd5013de70d68d/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="hammer",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/4f24b866dab248b684149ec6bb40101f/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_hammer_on_top_of_the_pitcher_base_on_the_table.__handle_right/20240824-162353_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopSidewaysBottle1536Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="highlighter",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/25481a65cad54e2a956394ff2b2765cd/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="watch",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/d6e38c73f8f2402aac890e8ebb115302/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="bottle",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/d17b8f44d1544364b8ad9fe2554ddfdf/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_bottle_on_top_of_the_highlighter_on_the_table.__sideways/20240824-210958_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopBladeRightKnife1537Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="wallet",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/ccfcc4ae6efa44a2ba34c4c479be7daf/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="knife",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/85db1d392a89459dbde5cdd951c4f6fb/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_knife_on_top_of_the_wallet_on_the_table.__blade_right/20240824-155707_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopHandleLeftMug1538Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="watch",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/fdb8227a88b2448d8c64fa82ffee3f58/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="cap",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/87e82aa304ce4571b69bdd5182549c72/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="mug",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/ca4f9a92cc2f4ee98fe9332db41bf7f7/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_mug_on_top_of_the_watch_on_the_table.__handle_left/20240824-195129_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopSidewaysCan1539Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="plate",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/029_plate_google_16k/029_plate_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="hammer",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/35a76a67ea1c45edabbd5013de70d68d/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="can",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/010_potted_meat_can_google_16k/010_potted_meat_can_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_can_on_top_of_the_plate_on_the_table.__sideways/20240824-175737_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopHandleRightMug1540Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="remote control",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/d28b249d997b46c1b42771d8ce869902/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="calipers",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/95c967114b624291bc26bcd705aaa334/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="hard drive",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/e32b159011544b8aa7cacd812636353e/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="mug",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/ca4f9a92cc2f4ee98fe9332db41bf7f7/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_mug_on_top_of_the_remote_control_on_the_table.__handle_right/20240824-230721_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopTapeMeasureUprightTapeMeasure1541Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="toy",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/072-a_toy_airplane_google_16k/072-a_toy_airplane_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="toy",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/072-b_toy_airplane_google_16k/072-b_toy_airplane_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="tape measure",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/1574b13827734d54b6d9d86be8be71d1/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_tape_measure_on_top_of_the_toy_on_the_table.__tape_measure_upright/20240824-195423_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopUprightLensForthCamera1542Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="box",
            urdf_path=(
                "data_isaaclab/assets/open6dor/ycb_16k_backup/004_sugar_box_google_16k/004_sugar_box_google_16k.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="camera",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/6a63fc4af48b453c91ca2b335a4d464d/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_camera_on_top_of_the_box_on_the_table.__upright_lens_forth/20240824-220400_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopHandleRightMug1543Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="box",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/008_pudding_box_google_16k/008_pudding_box_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="toy",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/073-g_lego_duplo_google_16k/073-g_lego_duplo_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="mug",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/922cd7d18c6748d49fe651ded8a04cf4/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_mug_on_top_of_the_box_on_the_table.__handle_right/20240824-223933_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopHandleRightMug1544Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="box",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/009_gelatin_box_google_16k/009_gelatin_box_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="mug",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/ca4f9a92cc2f4ee98fe9332db41bf7f7/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_mug_on_top_of_the_box_on_the_table.__handle_right/20240824-175448_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopClaspRightWallet1545Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="fork",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/e98e7fb33743442383812b68608f7006/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="highlighter",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/fd058916fb8843c09b4a2e05e3ee894e/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="mug",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/de23895b6e9b4cfd870e30d14c2150dd/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="wallet",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/ccfcc4ae6efa44a2ba34c4c479be7daf/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_wallet_on_top_of_the_fork_on_the_table.__clasp_right/20240824-181653_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopRollSidewaysToiletPaperRoll1546Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="hammer",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/048_hammer_google_16k/048_hammer_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="wallet",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/dbb07d13a33546f09ac8ca98b1ddef20/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="toilet paper roll",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/ea0cf48616e34708b73c82eb7c7366ca/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_toilet_paper_roll_on_top_of_the_hammer_on_the_table.__roll_sideways/20240824-221558_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopUpsideDownCalculator1547Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="mouse",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/f017c73a85484b6da3a66ec1cda70c71/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="wrench",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/5bf781c9fd8d4121b735503b13ba2eaf/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="shoe",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/ece4453042144b4a82fb86a4eab1ba7f/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="calculator",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/d9ffbc4bbe1044bb902e1dddac52b0de/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_calculator_on_top_of_the_mouse_on_the_table.__upside_down/20240824-212023_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopProngRightFork1548Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="skillet",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/027_skillet_google_16k/027_skillet_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="toy",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/073-g_lego_duplo_google_16k/073-g_lego_duplo_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="fork",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/e98e7fb33743442383812b68608f7006/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_fork_on_top_of_the_skillet_on_the_table.__prong_right/20240824-182527_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopUpsideDownBowl1549Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="tissue box",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/213d8df9c37a4d8cba794965676f7a75/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="box",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/003_cracker_box_google_16k/003_cracker_box_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="shoe",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/d6951661c1e445d8a1d00b7d38d86030/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="bowl",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/024_bowl_google_16k/024_bowl_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_bowl_on_top_of_the_tissue_box_on_the_table.__upside_down/20240824-210611_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopUprightTissueBox1550Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="remote control",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/d28b249d997b46c1b42771d8ce869902/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="clipboard",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/a37158f20ccf436483029e8295629738/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="cup",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/065-e_cups_google_16k/065-e_cups_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="tissue box",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/dc4c91abf45342b4bb8822f50fa162b2/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_tissue_box_on_top_of_the_remote_control_on_the_table.__upright/20240824-224357_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopSidewaysTextualBox1551Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="marker",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/040_large_marker_google_16k/040_large_marker_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="multimeter",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/79ac684e66b34554ba869bd2fc3c2653/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="box",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/008_pudding_box_google_16k/008_pudding_box_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_box_on_top_of_the_marker_on_the_table.__sideways_textual/20240824-160958_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopScoopRightLadle1552Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="cup",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/065-g_cups_google_16k/065-g_cups_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="tape measure",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/1574b13827734d54b6d9d86be8be71d1/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="ladle",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/ce27e3369587476c85a436905f1e5c00/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_ladle_on_top_of_the_cup_on_the_table.__scoop_right/20240824-221957_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopCapRightBottomLeftBottle1553Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="box",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/008_pudding_box_google_16k/008_pudding_box_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="cup",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/065-g_cups_google_16k/065-g_cups_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="bottle",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/8c0f8088a5d546c886d1b07e3f4eafae/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_bottle_on_top_of_the_box_on_the_table.__cap_right_bottom_left/20240824-212943_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopUpsideDownCup1554Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="clipboard",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/a37158f20ccf436483029e8295629738/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="cup",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/065-a_cups_google_16k/065-a_cups_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_cup_on_top_of_the_clipboard_on_the_table.__upside_down/20240824-183223_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopPhoneForthMobilePhone1555Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="shoe",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/d6951661c1e445d8a1d00b7d38d86030/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="spoon",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/031_spoon_google_16k/031_spoon_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="apple",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/013_apple_google_16k/013_apple_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="mobile phone",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/45c813a9fac4458ead1f90280826c0a4/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_mobile_phone_on_top_of_the_shoe_on_the_table.__phone_forth/20240824-232615_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopUpsideDownCalculator1556Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="mug",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/c609327dd3a74fb597584e1b4a14a615/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="calculator",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/054be0c3f09143f38c4d8038eb2588c6/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_calculator_on_top_of_the_mug_on_the_table.__upside_down/20240824-174732_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopSidewaysShoe1557Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="mouse",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/e51e63e374ee4c25ae3fb4a9c74bd335/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="skillet",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/027_skillet_google_16k/027_skillet_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="shoe",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/aa51af2f1270482193471455b504efc6/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_shoe_on_top_of_the_mouse_on_the_table.__sideways/20240824-194932_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopHandleRightWrench1558Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="shoe",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/aa51af2f1270482193471455b504efc6/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="plate",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/029_plate_google_16k/029_plate_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="keyboard",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/f5080a3df6d847b693c5cca415886c61/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="wrench",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/8a6cb4f7b0004f53830e270dc6e1ff1d/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_wrench_on_top_of_the_shoe_on_the_table.__handle_right/20240824-163525_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopRemoteControlForthRemoteControl1559Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="wrench",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/8a6cb4f7b0004f53830e270dc6e1ff1d/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="clipboard",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/b31c69728a414e639eef2fccd1c3dd75/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="mug",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/c25119c5ac6e4654be3b75d78e34a912/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="remote control",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/89dcd45ac1f04680b76b709357a3dba3/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_remote_control_on_top_of_the_wrench_on_the_table.__remote_control_forth/20240824-170819_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopRemoteControlForthRemoteControl1560Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="calipers",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/95c967114b624291bc26bcd705aaa334/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="envelope box",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/40b28d1cbfbd4e9f9acf653b748324ee/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="eraser",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/f0916a69ba514ecc973b44528b9dcc43/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="remote control",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/d28b249d997b46c1b42771d8ce869902/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_remote_control_on_top_of_the_calipers_on_the_table.__remote_control_forth/20240824-225051_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopBallpointRightPen1561Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="mouse",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/f496168739384cdb86ef6d65e2068a3f/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="pen",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/f126e15c0f904fccbd76f9348baea12c/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_pen_on_top_of_the_mouse_on_the_table.__ballpoint_right/20240824-195435_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotTopHandleRightMug1562Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        RigidObjCfg(
            name="hammer",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/6d155e8051294915813e0c156e3cd6de/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="shoe",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/ae7142127dd84ebbbe7762368ace452c/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="mouse",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/f017c73a85484b6da3a66ec1cda70c71/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="mug",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/b8478cea51454555b90de0fe6ba7ba83/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/top/Place_the_mug_on_top_of_the_hammer_on_the_table.__handle_right/20240824-204142_no_interaction/trajectory-unified_wo_traj_v2.pkl"
