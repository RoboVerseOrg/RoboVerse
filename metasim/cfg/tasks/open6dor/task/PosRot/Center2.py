from metasim.cfg.objects import PrimitiveCubeCfg, RigidObjCfg
from metasim.cfg.tasks.base_task_cfg import BaseTaskCfg
from metasim.constants import BenchmarkType, PhysicStateType, TaskType
from metasim.utils import configclass


@configclass
class OpensdorPosRotCenterCardForthTextualCreditCard1680Cfg(BaseTaskCfg):
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
            name="mug",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/c25119c5ac6e4654be3b75d78e34a912/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="toilet paper roll",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/7ca7ebcfad964498b49af73be442acf9/material_2.urdf"
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
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/center/Place_the_credit_card_at_the_center_of_all_the_objects_on_the_table.__card_forth_textual/20240824-182938_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotCenterUprightApple1681Cfg(BaseTaskCfg):
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
            name="can",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/010_potted_meat_can_google_16k/010_potted_meat_can_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="power drill",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/035_power_drill_google_16k/035_power_drill_google_16k.urdf",
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
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/center/Place_the_apple_at_the_center_of_all_the_objects_on_the_table.__upright/20240824-161257_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotCenterUprightApple1682Cfg(BaseTaskCfg):
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
            name="watch",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/243b381dcdc34316a7e78a533572d273/material_2.urdf"
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
            name="apple",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/013_apple_google_16k/013_apple_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/center/Place_the_apple_at_the_center_of_all_the_objects_on_the_table.__upright/20240824-215046_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotCenterUprightApple1683Cfg(BaseTaskCfg):
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
            name="ladle",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/ce27e3369587476c85a436905f1e5c00/material_2.urdf"
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
        RigidObjCfg(
            name="apple",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/f53d75bd123b40bca14d12d54286f432/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/center/Place_the_apple_at_the_center_of_all_the_objects_on_the_table.__upright/20240824-165332_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotCenterUprightApple1684Cfg(BaseTaskCfg):
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
            name="can",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/007_tuna_fish_can_google_16k/007_tuna_fish_can_google_16k.urdf",
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
            name="apple",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/fbda0b25f41f40958ea984f460e4770b/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/center/Place_the_apple_at_the_center_of_all_the_objects_on_the_table.__upright/20240824-223329_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotCenterUprightApple1685Cfg(BaseTaskCfg):
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
            name="mug",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/a564a11174be455bbce4698f7da92fdf/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="box",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/009_gelatin_box_google_16k/009_gelatin_box_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="apple",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/013_apple_google_16k/013_apple_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/center/Place_the_apple_at_the_center_of_all_the_objects_on_the_table.__upright/20240824-232753_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotCenterBladeRightKnife1686Cfg(BaseTaskCfg):
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
                "data_isaaclab/assets/open6dor/objaverse_rescale/de23895b6e9b4cfd870e30d14c2150dd/material_2.urdf"
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
        RigidObjCfg(
            name="binder clips",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/5e55c38b24dc4c239448efa1c6b7f49f/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="knife",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/0df75457524743ec81468c916fddb930/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/center/Place_the_knife_at_the_center_of_all_the_objects_on_the_table.__blade_right/20240824-220956_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotCenterBladeRightKnife1687Cfg(BaseTaskCfg):
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
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/065-d_cups_google_16k/065-d_cups_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="bottle",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/022_windex_bottle_google_16k/022_windex_bottle_google_16k.urdf",
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
            name="knife",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/032_knife_google_16k/032_knife_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/center/Place_the_knife_at_the_center_of_all_the_objects_on_the_table.__blade_right/20240824-231719_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotCenterBladeRightKnife1688Cfg(BaseTaskCfg):
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
                "data_isaaclab/assets/open6dor/objaverse_rescale/44bb12d306864e2cb4256a61d4168942/material_2.urdf"
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
            name="hammer",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/d499e1bc8a514e4bb5ca995ea6ba23b6/material_2.urdf"
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
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/center/Place_the_knife_at_the_center_of_all_the_objects_on_the_table.__blade_right/20240824-163644_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotCenterBladeRightKnife1689Cfg(BaseTaskCfg):
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
            name="clock",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/fdf932b04e6c4e0fbd6e274563b94536/material_2.urdf"
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
        RigidObjCfg(
            name="knife",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/0df75457524743ec81468c916fddb930/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/center/Place_the_knife_at_the_center_of_all_the_objects_on_the_table.__blade_right/20240824-214436_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotCenterBladeRightKnife1690Cfg(BaseTaskCfg):
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
                "data_isaaclab/assets/open6dor/objaverse_rescale/6e9e9135bb5747f58d388e1e92477f02/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="box",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/003_cracker_box_google_16k/003_cracker_box_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="knife",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/0df75457524743ec81468c916fddb930/material_2.urdf"
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
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/center/Place_the_knife_at_the_center_of_all_the_objects_on_the_table.__blade_right/20240824-173502_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotCenterBladeRightKnife1691Cfg(BaseTaskCfg):
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
            name="clock",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/fdf932b04e6c4e0fbd6e274563b94536/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="box",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/003_cracker_box_google_16k/003_cracker_box_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="knife",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/0df75457524743ec81468c916fddb930/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/center/Place_the_knife_at_the_center_of_all_the_objects_on_the_table.__blade_right/20240824-225855_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotCenterBladeRightKnife1692Cfg(BaseTaskCfg):
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
            name="can",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/010_potted_meat_can_google_16k/010_potted_meat_can_google_16k.urdf",
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
            name="knife",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/0df75457524743ec81468c916fddb930/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/center/Place_the_knife_at_the_center_of_all_the_objects_on_the_table.__blade_right/20240824-182419_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotCenterBladeRightKnife1693Cfg(BaseTaskCfg):
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
                "data_isaaclab/assets/open6dor/objaverse_rescale/6d5a4e1174dd4f5fa0a9a9076e476b91/material_2.urdf"
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
            name="tissue box",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/8eda72f64e6e4443af4fdbeed07cad29/material_2.urdf"
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
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/center/Place_the_knife_at_the_center_of_all_the_objects_on_the_table.__blade_right/20240824-214842_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotCenterBladeRightKnife1694Cfg(BaseTaskCfg):
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
                "data_isaaclab/assets/open6dor/objaverse_rescale/c7ec86e7feb746489b30b4a01c2510af/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="spoon",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/031_spoon_google_16k/031_spoon_google_16k.urdf",
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
            name="knife",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/032_knife_google_16k/032_knife_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/center/Place_the_knife_at_the_center_of_all_the_objects_on_the_table.__blade_right/20240824-213221_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotCenterBladeRightKnife1695Cfg(BaseTaskCfg):
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
                "data_isaaclab/assets/open6dor/objaverse_rescale/f0b1f7c17d70489888f5ae922169c0ce/material_2.urdf"
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
            name="tape measure",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/11307d06bb254318b95021d68c6fa12f/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="knife",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/0df75457524743ec81468c916fddb930/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/center/Place_the_knife_at_the_center_of_all_the_objects_on_the_table.__blade_right/20240824-230230_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotCenterBladeRightKnife1696Cfg(BaseTaskCfg):
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
            name="mug",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/9321c45cb9cf459f9f803507d3a11fb3/material_2.urdf"
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
        RigidObjCfg(
            name="knife",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/f93d05bc2db144f7a50374794ef5cf8d/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/center/Place_the_knife_at_the_center_of_all_the_objects_on_the_table.__blade_right/20240824-155946_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotCenterUprightPaperweight1697Cfg(BaseTaskCfg):
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
            name="hammer",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/35a76a67ea1c45edabbd5013de70d68d/material_2.urdf"
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
            name="paperweight",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/21c10181c122474a8f72a4ad0331f185/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/center/Place_the_paperweight_at_the_center_of_all_the_objects_on_the_table.__upright/20240824-200324_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotCenterUprightPaperweight1698Cfg(BaseTaskCfg):
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
            name="headphone",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/ea91786392284f03809c37976da090bf/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="wrench",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/042_adjustable_wrench_google_16k/042_adjustable_wrench_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="paperweight",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/21c10181c122474a8f72a4ad0331f185/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/center/Place_the_paperweight_at_the_center_of_all_the_objects_on_the_table.__upright/20240824-210447_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotCenterUprightPaperweight1699Cfg(BaseTaskCfg):
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
            name="hammer",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/d499e1bc8a514e4bb5ca995ea6ba23b6/material_2.urdf"
            ),
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
            name="paperweight",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/21c10181c122474a8f72a4ad0331f185/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/center/Place_the_paperweight_at_the_center_of_all_the_objects_on_the_table.__upright/20240824-204157_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotCenterUprightPaperweight1700Cfg(BaseTaskCfg):
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
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/010_potted_meat_can_google_16k/010_potted_meat_can_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="can",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/005_tomato_soup_can_google_16k/005_tomato_soup_can_google_16k.urdf",
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
            name="paperweight",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/21c10181c122474a8f72a4ad0331f185/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/center/Place_the_paperweight_at_the_center_of_all_the_objects_on_the_table.__upright/20240824-175750_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotCenterUprightMicrophone1701Cfg(BaseTaskCfg):
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
                "data_isaaclab/assets/open6dor/objaverse_rescale/d499e1bc8a514e4bb5ca995ea6ba23b6/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="mug",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/77d41aba53e4455f9f84fa04b175dff4/material_2.urdf"
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
            name="microphone",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/fac08019fef5474189f965bb495771eb/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/center/Place_the_microphone_at_the_center_of_all_the_objects_on_the_table.__upright/20240824-170508_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotCenterUprightMicrophone1702Cfg(BaseTaskCfg):
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
            name="hammer",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/6d155e8051294915813e0c156e3cd6de/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="power drill",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/035_power_drill_google_16k/035_power_drill_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="microphone",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/5172dbe9281a45f48cee8c15bdfa1831/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/center/Place_the_microphone_at_the_center_of_all_the_objects_on_the_table.__upright/20240824-182506_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotCenterUprightMicrophone1703Cfg(BaseTaskCfg):
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
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/072-b_toy_airplane_google_16k/072-b_toy_airplane_google_16k.urdf",
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
            name="can",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/010_potted_meat_can_google_16k/010_potted_meat_can_google_16k.urdf",
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
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/center/Place_the_microphone_at_the_center_of_all_the_objects_on_the_table.__upright/20240824-163015_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotCenterUprightMicrophone1704Cfg(BaseTaskCfg):
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
            name="mouse",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/e51e63e374ee4c25ae3fb4a9c74bd335/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="SD card",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/02f7f045679d402da9b9f280030821d4/material_2.urdf"
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
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/center/Place_the_microphone_at_the_center_of_all_the_objects_on_the_table.__upright/20240824-173426_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotCenterUprightMicrophone1705Cfg(BaseTaskCfg):
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
            name="hammer",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/9bf1b31f641543c9a921042dac3b527f/material_2.urdf"
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
            name="microphone",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/5172dbe9281a45f48cee8c15bdfa1831/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/center/Place_the_microphone_at_the_center_of_all_the_objects_on_the_table.__upright/20240824-173714_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotCenterUprightPear1706Cfg(BaseTaskCfg):
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
            name="bottle",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/794b730ae452424bb3a9ce3c6caaff7a/material_2.urdf"
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
            name="pear",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/f9ac5185507b4d0dbc744941e9055b96/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/center/Place_the_pear_at_the_center_of_all_the_objects_on_the_table.__upright/20240824-192625_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotCenterUprightPear1707Cfg(BaseTaskCfg):
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
                "data_isaaclab/assets/open6dor/objaverse_rescale/35a76a67ea1c45edabbd5013de70d68d/material_2.urdf"
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
        RigidObjCfg(
            name="book",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/b2d30398ba3741da9f9aef2319ee2b8b/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="pear",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/f3dd3064db4f4e8880344425970cecad/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/center/Place_the_pear_at_the_center_of_all_the_objects_on_the_table.__upright/20240824-164900_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotCenterUprightPear1708Cfg(BaseTaskCfg):
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
                "data_isaaclab/assets/open6dor/objaverse_rescale/d499e1bc8a514e4bb5ca995ea6ba23b6/material_2.urdf"
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
        RigidObjCfg(
            name="tissue box",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/dc4c91abf45342b4bb8822f50fa162b2/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="pear",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/f9ac5185507b4d0dbc744941e9055b96/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/center/Place_the_pear_at_the_center_of_all_the_objects_on_the_table.__upright/20240824-233007_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotCenterWatchUprightWatch1709Cfg(BaseTaskCfg):
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
                "data_isaaclab/assets/open6dor/objaverse_rescale/f47fdcf9615d4e94a71e6731242a4c94/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="SD card",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/02f7f045679d402da9b9f280030821d4/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="spoon",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/031_spoon_google_16k/031_spoon_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="watch",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/243b381dcdc34316a7e78a533572d273/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/center/Place_the_watch_at_the_center_of_all_the_objects_on_the_table.__watch_upright/20240824-170120_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotCenterWatchUprightWatch1710Cfg(BaseTaskCfg):
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
            name="bowl",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/bdd36c94f3f74f22b02b8a069c8d97b7/material_2.urdf"
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
            name="watch",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/243b381dcdc34316a7e78a533572d273/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/center/Place_the_watch_at_the_center_of_all_the_objects_on_the_table.__watch_upright/20240824-203534_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotCenterWatchUprightWatch1711Cfg(BaseTaskCfg):
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
            name="highlighter",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/25481a65cad54e2a956394ff2b2765cd/material_2.urdf"
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
            name="watch",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/fdb8227a88b2448d8c64fa82ffee3f58/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/center/Place_the_watch_at_the_center_of_all_the_objects_on_the_table.__watch_upright/20240824-223216_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotCenterWatchUprightWatch1712Cfg(BaseTaskCfg):
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
            name="cup",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/065-g_cups_google_16k/065-g_cups_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="keyboard",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/44bb12d306864e2cb4256a61d4168942/material_2.urdf"
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
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/center/Place_the_watch_at_the_center_of_all_the_objects_on_the_table.__watch_upright/20240824-162112_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotCenterWatchUprightWatch1713Cfg(BaseTaskCfg):
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
            name="SD card",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/02f7f045679d402da9b9f280030821d4/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="bottle",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/021_bleach_cleanser_google_16k/021_bleach_cleanser_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="watch",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/d6e38c73f8f2402aac890e8ebb115302/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/center/Place_the_watch_at_the_center_of_all_the_objects_on_the_table.__watch_upright/20240824-223635_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotCenterWatchUprightWatch1714Cfg(BaseTaskCfg):
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
                "data_isaaclab/assets/open6dor/objaverse_rescale/054be0c3f09143f38c4d8038eb2588c6/material_2.urdf"
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
            name="paperweight",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/21c10181c122474a8f72a4ad0331f185/material_2.urdf"
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
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/center/Place_the_watch_at_the_center_of_all_the_objects_on_the_table.__watch_upright/20240824-184025_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotCenterWatchUprightWatch1715Cfg(BaseTaskCfg):
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
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/010_potted_meat_can_google_16k/010_potted_meat_can_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="wrench",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/8a6cb4f7b0004f53830e270dc6e1ff1d/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="cup",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/065-j_cups_google_16k/065-j_cups_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="watch",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/243b381dcdc34316a7e78a533572d273/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/center/Place_the_watch_at_the_center_of_all_the_objects_on_the_table.__watch_upright/20240824-210836_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotCenterWatchUprightWatch1716Cfg(BaseTaskCfg):
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
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/072-b_toy_airplane_google_16k/072-b_toy_airplane_google_16k.urdf",
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
            name="hammer",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/048_hammer_google_16k/048_hammer_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="watch",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/fdb8227a88b2448d8c64fa82ffee3f58/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/center/Place_the_watch_at_the_center_of_all_the_objects_on_the_table.__watch_upright/20240824-210119_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotCenterHandleRightHammer1717Cfg(BaseTaskCfg):
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
            name="mouse",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/f496168739384cdb86ef6d65e2068a3f/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="cup",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/065-e_cups_google_16k/065-e_cups_google_16k.urdf",
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
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/center/Place_the_hammer_at_the_center_of_all_the_objects_on_the_table.__handle_right/20240824-220517_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotCenterHandleRightHammer1718Cfg(BaseTaskCfg):
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
                "data_isaaclab/assets/open6dor/objaverse_rescale/d499e1bc8a514e4bb5ca995ea6ba23b6/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="box",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/009_gelatin_box_google_16k/009_gelatin_box_google_16k.urdf",
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
            name="hammer",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/35a76a67ea1c45edabbd5013de70d68d/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/center/Place_the_hammer_at_the_center_of_all_the_objects_on_the_table.__handle_right/20240824-164152_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotCenterHandleRightHammer1719Cfg(BaseTaskCfg):
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
                "data_isaaclab/assets/open6dor/objaverse_rescale/f5080a3df6d847b693c5cca415886c61/material_2.urdf"
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
        RigidObjCfg(
            name="cup",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/065-j_cups_google_16k/065-j_cups_google_16k.urdf",
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
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/center/Place_the_hammer_at_the_center_of_all_the_objects_on_the_table.__handle_right/20240824-195700_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotCenterHandleRightHammer1720Cfg(BaseTaskCfg):
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
            name="binder clips",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/5e55c38b24dc4c239448efa1c6b7f49f/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="toy",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/072-a_toy_airplane_google_16k/072-a_toy_airplane_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="hammer",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/048_hammer_google_16k/048_hammer_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/center/Place_the_hammer_at_the_center_of_all_the_objects_on_the_table.__handle_right/20240824-221930_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotCenterHandleRightHammer1721Cfg(BaseTaskCfg):
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
                "data_isaaclab/assets/open6dor/objaverse_rescale/fbeb7c776372466daffa619106e0b2e0/material_2.urdf"
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
            name="calipers",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/95c967114b624291bc26bcd705aaa334/material_2.urdf"
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
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/center/Place_the_hammer_at_the_center_of_all_the_objects_on_the_table.__handle_right/20240824-181612_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotCenterHandleRightHammer1722Cfg(BaseTaskCfg):
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
            name="binder",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/a0026e7d1b244b9a9223daf4223c9372/material_2.urdf"
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
            name="hammer",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/048_hammer_google_16k/048_hammer_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/center/Place_the_hammer_at_the_center_of_all_the_objects_on_the_table.__handle_right/20240824-215602_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotCenterHandleRightHammer1723Cfg(BaseTaskCfg):
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
            name="wallet",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/ccfcc4ae6efa44a2ba34c4c479be7daf/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="shoe",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/d5a5f0a954f94bcea3168329d1605fe9/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="hammer",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/048_hammer_google_16k/048_hammer_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/center/Place_the_hammer_at_the_center_of_all_the_objects_on_the_table.__handle_right/20240824-162021_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotCenterHandleRightHammer1724Cfg(BaseTaskCfg):
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
            name="binder",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/78e016eeb0bf45d5bea47547128383a8/material_2.urdf"
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
        RigidObjCfg(
            name="hammer",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/9bf1b31f641543c9a921042dac3b527f/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/center/Place_the_hammer_at_the_center_of_all_the_objects_on_the_table.__handle_right/20240824-200755_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotCenterHandleRightHammer1725Cfg(BaseTaskCfg):
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
            name="camera",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/f6f8675406b7437faaf96a1f82062028/material_2.urdf"
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
        RigidObjCfg(
            name="hammer",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/8ed38a92668a425eb16da938622d9ace/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/center/Place_the_hammer_at_the_center_of_all_the_objects_on_the_table.__handle_right/20240824-225156_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotCenterHandleRightHammer1726Cfg(BaseTaskCfg):
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
                "data_isaaclab/assets/open6dor/objaverse_rescale/7ca7ebcfad964498b49af73be442acf9/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="cup",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/065-f_cups_google_16k/065-f_cups_google_16k.urdf",
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
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/center/Place_the_hammer_at_the_center_of_all_the_objects_on_the_table.__handle_right/20240824-221516_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotCenterHandleRightHammer1727Cfg(BaseTaskCfg):
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
            name="mug",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/025_mug_google_16k/025_mug_google_16k.urdf",
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
            name="hammer",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/4f24b866dab248b684149ec6bb40101f/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/center/Place_the_hammer_at_the_center_of_all_the_objects_on_the_table.__handle_right/20240824-184419_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotCenterHandleRightHammer1728Cfg(BaseTaskCfg):
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
                "data_isaaclab/assets/open6dor/objaverse_rescale/b4bb8bd7791648fdb36c458fd9a877bf/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="mug",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/db9345f568e8499a9eac2577302b5f51/material_2.urdf"
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
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/center/Place_the_hammer_at_the_center_of_all_the_objects_on_the_table.__handle_right/20240824-211055_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotCenterHandleRightHammer1729Cfg(BaseTaskCfg):
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
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/072-b_toy_airplane_google_16k/072-b_toy_airplane_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="cup",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/065-d_cups_google_16k/065-d_cups_google_16k.urdf",
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
            name="hammer",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/d499e1bc8a514e4bb5ca995ea6ba23b6/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/center/Place_the_hammer_at_the_center_of_all_the_objects_on_the_table.__handle_right/20240824-212320_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotCenterHandleRightHammer1730Cfg(BaseTaskCfg):
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
                "data_isaaclab/assets/open6dor/objaverse_rescale/f5080a3df6d847b693c5cca415886c61/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="calculator",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/69511a7fad2f42ee8c4b0579bbc8fec6/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
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
                "data_isaaclab/assets/open6dor/objaverse_rescale/4f24b866dab248b684149ec6bb40101f/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/center/Place_the_hammer_at_the_center_of_all_the_objects_on_the_table.__handle_right/20240824-185520_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotCenterSpoutRightPot1731Cfg(BaseTaskCfg):
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
                "data_isaaclab/assets/open6dor/objaverse_rescale/de23895b6e9b4cfd870e30d14c2150dd/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="bowl",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/024_bowl_google_16k/024_bowl_google_16k.urdf",
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
            name="pot",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/af249e607bea40cfa2f275e5e23b8283/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/center/Place_the_pot_at_the_center_of_all_the_objects_on_the_table.__spout_right/20240824-164557_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotCenterSpoutRightPot1732Cfg(BaseTaskCfg):
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
            name="wineglass",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/afca0b1933b84b1dbebd1244f25e72fc/material_2.urdf"
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
            name="headphone",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/f8891134871e48ce8cb5053c4287272b/material_2.urdf"
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
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/center/Place_the_pot_at_the_center_of_all_the_objects_on_the_table.__spout_right/20240824-172619_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotCenterSpoutRightPot1733Cfg(BaseTaskCfg):
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
            name="spatula",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/4b27628be0804c0285d0b13dda025a0d/material_2.urdf"
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
            name="pot",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/cba3e4df7c354d22b7913d45a28a8159/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/center/Place_the_pot_at_the_center_of_all_the_objects_on_the_table.__spout_right/20240824-201144_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotCenterUpsideDownCalculator1734Cfg(BaseTaskCfg):
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
            name="envelope box",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/40b28d1cbfbd4e9f9acf653b748324ee/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="cup",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/065-c_cups_google_16k/065-c_cups_google_16k.urdf",
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
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/center/Place_the_calculator_at_the_center_of_all_the_objects_on_the_table.__upside_down/20240824-172401_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotCenterUpsideDownCalculator1735Cfg(BaseTaskCfg):
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
            name="highlighter",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/fd058916fb8843c09b4a2e05e3ee894e/material_2.urdf"
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
        RigidObjCfg(
            name="calculator",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/b4bb8bd7791648fdb36c458fd9a877bf/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/center/Place_the_calculator_at_the_center_of_all_the_objects_on_the_table.__upside_down/20240824-182312_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotCenterUpsideDownCalculator1736Cfg(BaseTaskCfg):
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
            name="hammer",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/6d155e8051294915813e0c156e3cd6de/material_2.urdf"
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
        RigidObjCfg(
            name="calculator",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/d9ffbc4bbe1044bb902e1dddac52b0de/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/center/Place_the_calculator_at_the_center_of_all_the_objects_on_the_table.__upside_down/20240824-180326_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotCenterUpsideDownCalculator1737Cfg(BaseTaskCfg):
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
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/021_bleach_cleanser_google_16k/021_bleach_cleanser_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="box",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/f81ca566b008446eb704cad4844603b6/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="spoon",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/218768789fdd42bd95eb933046698499/material_2.urdf"
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
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/center/Place_the_calculator_at_the_center_of_all_the_objects_on_the_table.__upside_down/20240824-225133_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotCenterUpsideDownCalculator1738Cfg(BaseTaskCfg):
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
            name="camera",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/6a63fc4af48b453c91ca2b335a4d464d/material_2.urdf"
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
            name="calculator",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/054be0c3f09143f38c4d8038eb2588c6/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/center/Place_the_calculator_at_the_center_of_all_the_objects_on_the_table.__upside_down/20240824-223409_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotCenterUpsideDownCalculator1739Cfg(BaseTaskCfg):
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
            name="wallet",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/ccfcc4ae6efa44a2ba34c4c479be7daf/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="pear",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/f9ac5185507b4d0dbc744941e9055b96/material_2.urdf"
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
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/center/Place_the_calculator_at_the_center_of_all_the_objects_on_the_table.__upside_down/20240824-212130_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotCenterUpsideDownCalculator1740Cfg(BaseTaskCfg):
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
            name="knife",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/f93d05bc2db144f7a50374794ef5cf8d/material_2.urdf"
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
            name="calculator",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/d9ffbc4bbe1044bb902e1dddac52b0de/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/center/Place_the_calculator_at_the_center_of_all_the_objects_on_the_table.__upside_down/20240824-202707_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotCenterHandleRightWrench1741Cfg(BaseTaskCfg):
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
                "data_isaaclab/assets/open6dor/objaverse_rescale/6add0a1fa81942acbb24504b130661c1/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
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
            name="wrench",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/8a6cb4f7b0004f53830e270dc6e1ff1d/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/center/Place_the_wrench_at_the_center_of_all_the_objects_on_the_table.__handle_right/20240824-194737_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotCenterHandleRightWrench1742Cfg(BaseTaskCfg):
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
            name="fork",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/030_fork_google_16k/030_fork_google_16k.urdf",
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
            name="wrench",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/5bf781c9fd8d4121b735503b13ba2eaf/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/center/Place_the_wrench_at_the_center_of_all_the_objects_on_the_table.__handle_right/20240824-194601_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotCenterHandleRightWrench1743Cfg(BaseTaskCfg):
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
            name="hot glue gun",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/5ceec13e87774982afe825e7e74a0ce1/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="knife",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/0df75457524743ec81468c916fddb930/material_2.urdf"
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
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/center/Place_the_wrench_at_the_center_of_all_the_objects_on_the_table.__handle_right/20240824-160642_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotCenterHandleRightWrench1744Cfg(BaseTaskCfg):
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
            name="spatula",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/033_spatula_google_16k/033_spatula_google_16k.urdf",
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
            name="wrench",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/8a6cb4f7b0004f53830e270dc6e1ff1d/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/center/Place_the_wrench_at_the_center_of_all_the_objects_on_the_table.__handle_right/20240824-212443_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotCenterHandleRightWrench1745Cfg(BaseTaskCfg):
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
                "data_isaaclab/assets/open6dor/objaverse_rescale/19c1913ba85042938f0a87d94397c946/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="spatula",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/81b129ecf5724f63b5b044702a3140c2/material_2.urdf"
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
            name="wrench",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/5bf781c9fd8d4121b735503b13ba2eaf/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/center/Place_the_wrench_at_the_center_of_all_the_objects_on_the_table.__handle_right/20240824-224809_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotCenterHandleRightWrench1746Cfg(BaseTaskCfg):
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
                "data_isaaclab/assets/open6dor/objaverse_rescale/2c88306f3d724a839054f3c2913fb1d5/material_2.urdf"
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
            name="shoe",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/d5a5f0a954f94bcea3168329d1605fe9/material_2.urdf"
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
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/center/Place_the_wrench_at_the_center_of_all_the_objects_on_the_table.__handle_right/20240824-201910_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotCenterHandleRightWrench1747Cfg(BaseTaskCfg):
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
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/030_fork_google_16k/030_fork_google_16k.urdf",
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
            name="box",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/9660e0c0326b4f7386014e27717231ae/material_2.urdf"
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
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/center/Place_the_wrench_at_the_center_of_all_the_objects_on_the_table.__handle_right/20240824-175109_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotCenterHandleRightWrench1748Cfg(BaseTaskCfg):
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
            name="wallet",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/dbb07d13a33546f09ac8ca98b1ddef20/material_2.urdf"
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
            name="wrench",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/19c1913ba85042938f0a87d94397c946/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/center/Place_the_wrench_at_the_center_of_all_the_objects_on_the_table.__handle_right/20240824-210431_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotCenterHandleRightWrench1749Cfg(BaseTaskCfg):
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
            name="wallet",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/dbb07d13a33546f09ac8ca98b1ddef20/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="bottle",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/a7785ac7ce7f4ba79df661db33637ba9/material_2.urdf"
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
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/center/Place_the_wrench_at_the_center_of_all_the_objects_on_the_table.__handle_right/20240824-223457_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotCenterHandleRightWrench1750Cfg(BaseTaskCfg):
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
            name="mug",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/3ea2fd4e065147048064f4c97a89fe6f/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="pitcher base",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/019_pitcher_base_google_16k/019_pitcher_base_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="wrench",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/5bf781c9fd8d4121b735503b13ba2eaf/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/center/Place_the_wrench_at_the_center_of_all_the_objects_on_the_table.__handle_right/20240824-185259_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotCenterCapRightBottomLeftBottle1751Cfg(BaseTaskCfg):
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
                "data_isaaclab/assets/open6dor/objaverse_rescale/c7ec86e7feb746489b30b4a01c2510af/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="can",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/002_master_chef_can_google_16k/002_master_chef_can_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="hammer",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/048_hammer_google_16k/048_hammer_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="bottle",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/006_mustard_bottle_google_16k/006_mustard_bottle_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/center/Place_the_bottle_at_the_center_of_all_the_objects_on_the_table.__cap_right_bottom_left/20240824-190900_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotCenterCapRightBottomLeftBottle1752Cfg(BaseTaskCfg):
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
        RigidObjCfg(
            name="bottle",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/6add0a1fa81942acbb24504b130661c1/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/center/Place_the_bottle_at_the_center_of_all_the_objects_on_the_table.__cap_right_bottom_left/20240824-195024_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotCenterCapRightBottomLeftBottle1753Cfg(BaseTaskCfg):
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
            name="calculator",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/d9ffbc4bbe1044bb902e1dddac52b0de/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="mixer",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/990d0cb9499540fda49b1ff36be9ba26/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="bottle",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/006_mustard_bottle_google_16k/006_mustard_bottle_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/center/Place_the_bottle_at_the_center_of_all_the_objects_on_the_table.__cap_right_bottom_left/20240824-214221_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotCenterCapRightBottomLeftBottle1754Cfg(BaseTaskCfg):
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
            name="screwdriver",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/043_phillips_screwdriver_google_16k/043_phillips_screwdriver_google_16k.urdf",
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
            name="bottle",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/021_bleach_cleanser_google_16k/021_bleach_cleanser_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/center/Place_the_bottle_at_the_center_of_all_the_objects_on_the_table.__cap_right_bottom_left/20240824-203957_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotCenterCapRightBottomLeftBottle1755Cfg(BaseTaskCfg):
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
            name="can",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/005_tomato_soup_can_google_16k/005_tomato_soup_can_google_16k.urdf",
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
            name="bottle",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/022_windex_bottle_google_16k/022_windex_bottle_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/center/Place_the_bottle_at_the_center_of_all_the_objects_on_the_table.__cap_right_bottom_left/20240824-232355_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotCenterCapRightBottomLeftBottle1756Cfg(BaseTaskCfg):
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
            name="hammer",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/048_hammer_google_16k/048_hammer_google_16k.urdf",
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
            name="bottle",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/021_bleach_cleanser_google_16k/021_bleach_cleanser_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/center/Place_the_bottle_at_the_center_of_all_the_objects_on_the_table.__cap_right_bottom_left/20240824-201525_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotCenterCapRightBottomLeftBottle1757Cfg(BaseTaskCfg):
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
            name="book",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/de79a920c48745ff95c0df7a7c300091/material_2.urdf"
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
        RigidObjCfg(
            name="bottle",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/006_mustard_bottle_google_16k/006_mustard_bottle_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/center/Place_the_bottle_at_the_center_of_all_the_objects_on_the_table.__cap_right_bottom_left/20240824-215256_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotCenterEarpieceFarHeadphone1758Cfg(BaseTaskCfg):
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
            name="tissue box",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/2bd98e8ea09f49bd85453a927011d31d/material_2.urdf"
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
            name="headphone",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/f8891134871e48ce8cb5053c4287272b/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/center/Place_the_headphone_at_the_center_of_all_the_objects_on_the_table.__earpiece_far/20240824-175336_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotCenterEarpieceFarHeadphone1759Cfg(BaseTaskCfg):
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
            name="spoon",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/218768789fdd42bd95eb933046698499/material_2.urdf"
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
            name="headphone",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/f8891134871e48ce8cb5053c4287272b/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/center/Place_the_headphone_at_the_center_of_all_the_objects_on_the_table.__earpiece_far/20240824-174517_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotCenterEarpieceFarHeadphone1760Cfg(BaseTaskCfg):
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
            name="keyboard",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/44bb12d306864e2cb4256a61d4168942/material_2.urdf"
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
            name="headphone",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/f8891134871e48ce8cb5053c4287272b/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/center/Place_the_headphone_at_the_center_of_all_the_objects_on_the_table.__earpiece_far/20240824-230416_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotCenterEarpieceFarHeadphone1761Cfg(BaseTaskCfg):
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
            name="tissue box",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/6e9e9135bb5747f58d388e1e92477f02/material_2.urdf"
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
            name="headphone",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/f8891134871e48ce8cb5053c4287272b/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/center/Place_the_headphone_at_the_center_of_all_the_objects_on_the_table.__earpiece_far/20240824-192258_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotCenterEarpieceFarHeadphone1762Cfg(BaseTaskCfg):
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
            name="knife",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/f93d05bc2db144f7a50374794ef5cf8d/material_2.urdf"
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
            name="headphone",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/ea91786392284f03809c37976da090bf/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/center/Place_the_headphone_at_the_center_of_all_the_objects_on_the_table.__earpiece_far/20240824-232607_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotCenterEarpieceFarHeadphone1763Cfg(BaseTaskCfg):
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
            name="wallet",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/ccfcc4ae6efa44a2ba34c4c479be7daf/material_2.urdf"
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
            name="headphone",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/ea91786392284f03809c37976da090bf/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/center/Place_the_headphone_at_the_center_of_all_the_objects_on_the_table.__earpiece_far/20240824-214555_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotCenterPlugRightUsb1764Cfg(BaseTaskCfg):
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
            name="mug",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/db9345f568e8499a9eac2577302b5f51/material_2.urdf"
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
        RigidObjCfg(
            name="USB",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/0a51815f3c0941ae8312fc6917173ed6/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/center/Place_the_USB_at_the_center_of_all_the_objects_on_the_table.__plug_right/20240824-170931_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotCenterPlugRightUsb1765Cfg(BaseTaskCfg):
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
            name="bottle",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/a7785ac7ce7f4ba79df661db33637ba9/material_2.urdf"
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
        RigidObjCfg(
            name="USB",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/878049e8c3174fa68a56530e5aef7a5a/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/center/Place_the_USB_at_the_center_of_all_the_objects_on_the_table.__plug_right/20240824-190207_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotCenterPlugRightUsb1766Cfg(BaseTaskCfg):
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
            name="USB",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/878049e8c3174fa68a56530e5aef7a5a/material_2.urdf"
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
            name="USB",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/b1098389cc3b460bb1f1ce558d4b0764/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/center/Place_the_USB_at_the_center_of_all_the_objects_on_the_table.__plug_right/20240824-211118_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotCenterPlugRightUsb1767Cfg(BaseTaskCfg):
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
            name="pen",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/f126e15c0f904fccbd76f9348baea12c/material_2.urdf"
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
            name="USB",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/878049e8c3174fa68a56530e5aef7a5a/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/center/Place_the_USB_at_the_center_of_all_the_objects_on_the_table.__plug_right/20240824-215734_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotCenterSidewaysBottle1768Cfg(BaseTaskCfg):
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
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/072-b_toy_airplane_google_16k/072-b_toy_airplane_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="tissue box",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/2bd98e8ea09f49bd85453a927011d31d/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="bowl",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/024_bowl_google_16k/024_bowl_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="bottle",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/006_mustard_bottle_google_16k/006_mustard_bottle_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/center/Place_the_bottle_at_the_center_of_all_the_objects_on_the_table.__sideways/20240824-224549_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotCenterSidewaysBottle1769Cfg(BaseTaskCfg):
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
                "data_isaaclab/assets/open6dor/objaverse_rescale/9321c45cb9cf459f9f803507d3a11fb3/material_2.urdf"
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
            name="apple",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/fbda0b25f41f40958ea984f460e4770b/material_2.urdf"
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
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/center/Place_the_bottle_at_the_center_of_all_the_objects_on_the_table.__sideways/20240824-212245_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotCenterSidewaysBottle1770Cfg(BaseTaskCfg):
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
            name="remote control",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/c7ec86e7feb746489b30b4a01c2510af/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="bottle",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/022_windex_bottle_google_16k/022_windex_bottle_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="bottle",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/006_mustard_bottle_google_16k/006_mustard_bottle_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/center/Place_the_bottle_at_the_center_of_all_the_objects_on_the_table.__sideways/20240824-214446_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotCenterSidewaysBottle1771Cfg(BaseTaskCfg):
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
            name="can",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/010_potted_meat_can_google_16k/010_potted_meat_can_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="spatula",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/033_spatula_google_16k/033_spatula_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="bottle",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/a7785ac7ce7f4ba79df661db33637ba9/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/center/Place_the_bottle_at_the_center_of_all_the_objects_on_the_table.__sideways/20240824-211547_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotCenterSidewaysBottle1772Cfg(BaseTaskCfg):
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
            name="speaker",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/4445f4442a884999960e7e6660459095/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="cup",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/065-g_cups_google_16k/065-g_cups_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="keyboard",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/44bb12d306864e2cb4256a61d4168942/material_2.urdf"
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
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/center/Place_the_bottle_at_the_center_of_all_the_objects_on_the_table.__sideways/20240824-231045_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotCenterUpsideDownClipboard1773Cfg(BaseTaskCfg):
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
                "data_isaaclab/assets/open6dor/objaverse_rescale/e5e87ddbf3f6470384bef58431351e2a/material_2.urdf"
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
        RigidObjCfg(
            name="clipboard",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/b31c69728a414e639eef2fccd1c3dd75/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/center/Place_the_clipboard_at_the_center_of_all_the_objects_on_the_table.__upside_down/20240824-222838_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotCenterUpsideDownClipboard1774Cfg(BaseTaskCfg):
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
            name="lighter",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/b640aa4990b64db9ab3d868c6f49820e/material_2.urdf"
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
            name="clipboard",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/a37158f20ccf436483029e8295629738/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/center/Place_the_clipboard_at_the_center_of_all_the_objects_on_the_table.__upside_down/20240824-155702_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotCenterUpsideDownClipboard1775Cfg(BaseTaskCfg):
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
                "data_isaaclab/assets/open6dor/objaverse_rescale/e32b159011544b8aa7cacd812636353e/material_2.urdf"
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
        RigidObjCfg(
            name="mixer",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/990d0cb9499540fda49b1ff36be9ba26/material_2.urdf"
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
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/center/Place_the_clipboard_at_the_center_of_all_the_objects_on_the_table.__upside_down/20240824-171240_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotCenterUpsideDownClipboard1776Cfg(BaseTaskCfg):
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
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/010_potted_meat_can_google_16k/010_potted_meat_can_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="highlighter",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/25481a65cad54e2a956394ff2b2765cd/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="tissue box",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/dc4c91abf45342b4bb8822f50fa162b2/material_2.urdf"
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
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/center/Place_the_clipboard_at_the_center_of_all_the_objects_on_the_table.__upside_down/20240824-222106_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotCenterMultimeterForthMultimeter1777Cfg(BaseTaskCfg):
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
            name="USB",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/878049e8c3174fa68a56530e5aef7a5a/material_2.urdf"
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
            name="multimeter",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/79ac684e66b34554ba869bd2fc3c2653/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/center/Place_the_multimeter_at_the_center_of_all_the_objects_on_the_table.__multimeter_forth/20240824-233205_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotCenterMultimeterForthMultimeter1778Cfg(BaseTaskCfg):
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
            name="mug",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/7bc2b14a48d1413e9375c32a53e3ee6f/material_2.urdf"
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
            name="multimeter",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/79ac684e66b34554ba869bd2fc3c2653/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/center/Place_the_multimeter_at_the_center_of_all_the_objects_on_the_table.__multimeter_forth/20240824-191716_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotCenterHandleLeftPitcherBase1779Cfg(BaseTaskCfg):
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
            name="fork",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/e98e7fb33743442383812b68608f7006/material_2.urdf"
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
            name="pitcher base",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/019_pitcher_base_google_16k/019_pitcher_base_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/center/Place_the_pitcher_base_at_the_center_of_all_the_objects_on_the_table.__handle_left/20240824-232229_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotCenterUpsideDownPlate1780Cfg(BaseTaskCfg):
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
            name="toy",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/9adc77036b434348ae049776c50df624/material_2.urdf"
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
        RigidObjCfg(
            name="plate",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/029_plate_google_16k/029_plate_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/center/Place_the_plate_at_the_center_of_all_the_objects_on_the_table.__upside_down/20240824-224731_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotCenterUpsideDownPlate1781Cfg(BaseTaskCfg):
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
            name="mouse",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/969b0d9d99a8468898753fdcd219f883/material_2.urdf"
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
            name="plate",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/029_plate_google_16k/029_plate_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/center/Place_the_plate_at_the_center_of_all_the_objects_on_the_table.__upside_down/20240824-190156_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotCenterUpsideDownPlate1782Cfg(BaseTaskCfg):
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
            name="bottle",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/021_bleach_cleanser_google_16k/021_bleach_cleanser_google_16k.urdf",
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
            name="plate",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/8d50c19e138b403e8b2ede9b47d8be3c/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/center/Place_the_plate_at_the_center_of_all_the_objects_on_the_table.__upside_down/20240824-165125_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotCenterUpsideDownPlate1783Cfg(BaseTaskCfg):
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
            name="apple",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/013_apple_google_16k/013_apple_google_16k.urdf",
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
            name="plate",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/029_plate_google_16k/029_plate_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/center/Place_the_plate_at_the_center_of_all_the_objects_on_the_table.__upside_down/20240824-222339_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotCenterUpsideDownPlate1784Cfg(BaseTaskCfg):
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
            name="wallet",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/ccfcc4ae6efa44a2ba34c4c479be7daf/material_2.urdf"
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
            name="plate",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/8d50c19e138b403e8b2ede9b47d8be3c/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/center/Place_the_plate_at_the_center_of_all_the_objects_on_the_table.__upside_down/20240824-212420_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotCenterUpsideDownPlate1785Cfg(BaseTaskCfg):
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
            name="box",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/5e1a1527a0bd49ed8e473079392c31c8/material_2.urdf"
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
        RigidObjCfg(
            name="plate",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/6d5a4e1174dd4f5fa0a9a9076e476b91/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/center/Place_the_plate_at_the_center_of_all_the_objects_on_the_table.__upside_down/20240824-165717_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotCenterUpsideDownPlate1786Cfg(BaseTaskCfg):
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
                "data_isaaclab/assets/open6dor/objaverse_rescale/2bd98e8ea09f49bd85453a927011d31d/material_2.urdf"
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
            name="glasses",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/e4f348e98ceb45e3abc77da5b738f1b2/material_2.urdf"
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
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/center/Place_the_plate_at_the_center_of_all_the_objects_on_the_table.__upside_down/20240824-225508_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotCenterUpsideDownPlate1787Cfg(BaseTaskCfg):
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
                "data_isaaclab/assets/open6dor/objaverse_rescale/2bd98e8ea09f49bd85453a927011d31d/material_2.urdf"
            ),
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
            name="wallet",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/f47fdcf9615d4e94a71e6731242a4c94/material_2.urdf"
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
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/center/Place_the_plate_at_the_center_of_all_the_objects_on_the_table.__upside_down/20240824-214906_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotCenterHandleRightJawLeftCalipers1788Cfg(BaseTaskCfg):
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
            name="hat",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/dc02f9c1fd01432fadd6e7d15851dc37/material_2.urdf"
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
            name="calipers",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/95c967114b624291bc26bcd705aaa334/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/center/Place_the_calipers_at_the_center_of_all_the_objects_on_the_table.__handle_right_jaw_left/20240824-170322_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotCenterUpsideDownCup1789Cfg(BaseTaskCfg):
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
            name="bottle",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/a7785ac7ce7f4ba79df661db33637ba9/material_2.urdf"
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
            name="cup",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/065-c_cups_google_16k/065-c_cups_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/center/Place_the_cup_at_the_center_of_all_the_objects_on_the_table.__upside_down/20240824-213711_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotCenterUpsideDownCup1790Cfg(BaseTaskCfg):
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
                "data_isaaclab/assets/open6dor/objaverse_rescale/6e9e9135bb5747f58d388e1e92477f02/material_2.urdf"
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
            name="pitcher base",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/019_pitcher_base_google_16k/019_pitcher_base_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="cup",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/065-i_cups_google_16k/065-i_cups_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/center/Place_the_cup_at_the_center_of_all_the_objects_on_the_table.__upside_down/20240824-211123_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotCenterUpsideDownCup1791Cfg(BaseTaskCfg):
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
            name="watch",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/d6e38c73f8f2402aac890e8ebb115302/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="mug",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/db9345f568e8499a9eac2577302b5f51/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="cup",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/065-j_cups_google_16k/065-j_cups_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/center/Place_the_cup_at_the_center_of_all_the_objects_on_the_table.__upside_down/20240824-221354_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotCenterUpsideDownCup1792Cfg(BaseTaskCfg):
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
            name="wrench",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/8a6cb4f7b0004f53830e270dc6e1ff1d/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="fork",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/030_fork_google_16k/030_fork_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="cup",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/065-h_cups_google_16k/065-h_cups_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/center/Place_the_cup_at_the_center_of_all_the_objects_on_the_table.__upside_down/20240824-210805_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotCenterUpsideDownCup1793Cfg(BaseTaskCfg):
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
            name="SD card",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/02f7f045679d402da9b9f280030821d4/material_2.urdf"
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
            name="cup",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/065-e_cups_google_16k/065-e_cups_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/center/Place_the_cup_at_the_center_of_all_the_objects_on_the_table.__upside_down/20240824-162521_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotCenterUpsideDownCup1794Cfg(BaseTaskCfg):
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
            name="keyboard",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/0161b6232216457aa8f4f3393960ec85/material_2.urdf"
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
        RigidObjCfg(
            name="cup",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/065-h_cups_google_16k/065-h_cups_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/center/Place_the_cup_at_the_center_of_all_the_objects_on_the_table.__upside_down/20240824-171939_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotCenterUpsideDownCup1795Cfg(BaseTaskCfg):
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
            name="hot glue gun",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/5ceec13e87774982afe825e7e74a0ce1/material_2.urdf"
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
        RigidObjCfg(
            name="book",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/de79a920c48745ff95c0df7a7c300091/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="cup",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/065-f_cups_google_16k/065-f_cups_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/center/Place_the_cup_at_the_center_of_all_the_objects_on_the_table.__upside_down/20240824-174841_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotCenterUpsideDownCup1796Cfg(BaseTaskCfg):
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
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/010_potted_meat_can_google_16k/010_potted_meat_can_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="mug",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/77d41aba53e4455f9f84fa04b175dff4/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="watch",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/fdb8227a88b2448d8c64fa82ffee3f58/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="cup",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/065-j_cups_google_16k/065-j_cups_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/center/Place_the_cup_at_the_center_of_all_the_objects_on_the_table.__upside_down/20240824-182632_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorPosRotCenterUpsideDownCup1797Cfg(BaseTaskCfg):
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
            name="envelope box",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/40b28d1cbfbd4e9f9acf653b748324ee/material_2.urdf"
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
        RigidObjCfg(
            name="apple",
            urdf_path=(
                "data_isaaclab/assets/open6dor/objaverse_rescale/f53d75bd123b40bca14d12d54286f432/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="cup",
            urdf_path="data_isaaclab/assets/open6dor/ycb_16k_backup/065-e_cups_google_16k/065-e_cups_google_16k.urdf",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_6dof/center/Place_the_cup_at_the_center_of_all_the_objects_on_the_table.__upside_down/20240824-204555_no_interaction/trajectory-unified_wo_traj_v2.pkl"
