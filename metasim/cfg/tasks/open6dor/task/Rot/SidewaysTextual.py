from metasim.cfg.objects import PrimitiveCubeCfg, RigidObjCfg
from metasim.cfg.tasks.base_task_cfg import BaseTaskCfg
from metasim.constants import BenchmarkType, PhysicStateType, TaskType
from metasim.utils import configclass


@configclass
class OpensdorRotSidewaysTextualBox45Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
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
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_rot_only/rot_ins/003_sideways_textual/20240826-212342_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorRotSidewaysTextualBox46Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
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
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_rot_only/rot_ins/003_sideways_textual/20240826-220936_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorRotSidewaysTextualBox47Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
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
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_rot_only/rot_ins/003_sideways_textual/20240826-215229_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorRotSidewaysTextualBox48Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
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
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_rot_only/rot_ins/003_sideways_textual/20240826-212102_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorRotSidewaysTextualBox49Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
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
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_rot_only/rot_ins/003_sideways_textual/20240826-220627_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorRotSidewaysTextualBox50Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
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
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_rot_only/rot_ins/003_sideways_textual/20240826-211736_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorRotSidewaysTextualBox276Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
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
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_rot_only/rot_ins/004_sideways_textual/20240826-211544_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorRotSidewaysTextualBox277Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
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
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_rot_only/rot_ins/004_sideways_textual/20240826-215434_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorRotSidewaysTextualBox278Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
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
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_rot_only/rot_ins/004_sideways_textual/20240826-212458_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorRotSidewaysTextualBox279Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
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
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_rot_only/rot_ins/004_sideways_textual/20240826-222442_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorRotSidewaysTextualBox280Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
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
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_rot_only/rot_ins/004_sideways_textual/20240826-220553_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorRotSidewaysTextualBox281Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
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
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_rot_only/rot_ins/004_sideways_textual/20240826-221921_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorRotSidewaysTextualBox888Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
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
                "data_isaaclab/assets/open6dor/objaverse_rescale/9660e0c0326b4f7386014e27717231ae/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_rot_only/rot_ins/9660e0c0326b4f7386014e27717231ae_sideways_textual/20240826-221524_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorRotSidewaysTextualBox889Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
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
                "data_isaaclab/assets/open6dor/objaverse_rescale/9660e0c0326b4f7386014e27717231ae/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_rot_only/rot_ins/9660e0c0326b4f7386014e27717231ae_sideways_textual/20240826-213209_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorRotSidewaysTextualBox890Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
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
                "data_isaaclab/assets/open6dor/objaverse_rescale/9660e0c0326b4f7386014e27717231ae/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_rot_only/rot_ins/9660e0c0326b4f7386014e27717231ae_sideways_textual/20240826-221653_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorRotSidewaysTextualBox891Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
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
                "data_isaaclab/assets/open6dor/objaverse_rescale/9660e0c0326b4f7386014e27717231ae/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_rot_only/rot_ins/9660e0c0326b4f7386014e27717231ae_sideways_textual/20240826-213713_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorRotSidewaysTextualBox892Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
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
                "data_isaaclab/assets/open6dor/objaverse_rescale/9660e0c0326b4f7386014e27717231ae/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_rot_only/rot_ins/9660e0c0326b4f7386014e27717231ae_sideways_textual/20240826-221204_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorRotSidewaysTextualBox893Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
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
                "data_isaaclab/assets/open6dor/objaverse_rescale/9660e0c0326b4f7386014e27717231ae/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_rot_only/rot_ins/9660e0c0326b4f7386014e27717231ae_sideways_textual/20240826-220259_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorRotSidewaysTextualBox983Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
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
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_rot_only/rot_ins/009_sideways_textual/20240826-213709_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorRotSidewaysTextualBox984Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
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
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_rot_only/rot_ins/009_sideways_textual/20240826-215450_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorRotSidewaysTextualBox985Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
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
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_rot_only/rot_ins/009_sideways_textual/20240826-221744_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorRotSidewaysTextualBox986Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
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
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_rot_only/rot_ins/009_sideways_textual/20240826-220855_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorRotSidewaysTextualBox987Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
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
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_rot_only/rot_ins/009_sideways_textual/20240826-213336_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorRotSidewaysTextualBox988Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
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
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_rot_only/rot_ins/009_sideways_textual/20240826-211913_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorRotSidewaysTextualBox1080Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
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
                "data_isaaclab/assets/open6dor/objaverse_rescale/5e1a1527a0bd49ed8e473079392c31c8/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_rot_only/rot_ins/5e1a1527a0bd49ed8e473079392c31c8_sideways_textual/20240826-215725_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorRotSidewaysTextualBox1081Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
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
                "data_isaaclab/assets/open6dor/objaverse_rescale/5e1a1527a0bd49ed8e473079392c31c8/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_rot_only/rot_ins/5e1a1527a0bd49ed8e473079392c31c8_sideways_textual/20240826-221029_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorRotSidewaysTextualBox1082Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
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
                "data_isaaclab/assets/open6dor/objaverse_rescale/5e1a1527a0bd49ed8e473079392c31c8/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_rot_only/rot_ins/5e1a1527a0bd49ed8e473079392c31c8_sideways_textual/20240826-212918_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorRotSidewaysTextualBox1083Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
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
                "data_isaaclab/assets/open6dor/objaverse_rescale/5e1a1527a0bd49ed8e473079392c31c8/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_rot_only/rot_ins/5e1a1527a0bd49ed8e473079392c31c8_sideways_textual/20240826-222138_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorRotSidewaysTextualBox1084Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
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
                "data_isaaclab/assets/open6dor/objaverse_rescale/5e1a1527a0bd49ed8e473079392c31c8/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_rot_only/rot_ins/5e1a1527a0bd49ed8e473079392c31c8_sideways_textual/20240826-211805_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorRotSidewaysTextualBox1085Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
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
                "data_isaaclab/assets/open6dor/objaverse_rescale/5e1a1527a0bd49ed8e473079392c31c8/material_2.urdf"
            ),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_rot_only/rot_ins/5e1a1527a0bd49ed8e473079392c31c8_sideways_textual/20240826-213807_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorRotSidewaysTextualBox1265Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
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
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_rot_only/rot_ins/008_sideways_textual/20240826-214113_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorRotSidewaysTextualBox1266Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
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
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_rot_only/rot_ins/008_sideways_textual/20240826-220719_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorRotSidewaysTextualBox1267Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
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
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_rot_only/rot_ins/008_sideways_textual/20240826-213004_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorRotSidewaysTextualBox1268Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
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
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_rot_only/rot_ins/008_sideways_textual/20240826-223609_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorRotSidewaysTextualBox1269Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
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
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_rot_only/rot_ins/008_sideways_textual/20240826-222425_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorRotSidewaysTextualBox1329Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
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
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_rot_only/rot_ins/f81ca566b008446eb704cad4844603b6_sideways_textual/20240826-223341_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorRotSidewaysTextualBox1330Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
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
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_rot_only/rot_ins/f81ca566b008446eb704cad4844603b6_sideways_textual/20240826-221824_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorRotSidewaysTextualBox1331Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
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
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_rot_only/rot_ins/f81ca566b008446eb704cad4844603b6_sideways_textual/20240826-220919_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorRotSidewaysTextualBox1332Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
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
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_rot_only/rot_ins/f81ca566b008446eb704cad4844603b6_sideways_textual/20240826-212640_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorRotSidewaysTextualBox1333Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
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
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_rot_only/rot_ins/f81ca566b008446eb704cad4844603b6_sideways_textual/20240826-214930_no_interaction/trajectory-unified_wo_traj_v2.pkl"


@configclass
class OpensdorRotSidewaysTextualBox1334Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
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
    ]
    traj_filepath = "data_isaaclab/source_data/open6dor/task_refine_rot_only/rot_ins/f81ca566b008446eb704cad4844603b6_sideways_textual/20240826-211337_no_interaction/trajectory-unified_wo_traj_v2.pkl"
