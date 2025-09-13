from metasim.utils import configclass

from .scene.scene_A import SCENE_A
from .scene.scene_B import SCENE_B
from .scene.scene_C import SCENE_C
from .scene.scene_D import SCENE_D
from .task.close_drawer_cfg import CloseDrawerCfg
from .task.lift_blue_block_drawer_cfg import LiftBlueBlockDrawerCfg
from .task.lift_blue_block_slider_cfg import LiftBlueBlockSliderCfg
from .task.lift_blue_block_table_cfg import LiftBlueBlockTableCfg
from .task.lift_pink_block_drawer_cfg import LiftPinkBlockDrawerCfg
from .task.lift_pink_block_slider_cfg import LiftPinkBlockSliderCfg
from .task.lift_pink_block_table_cfg import LiftPinkBlockTableCfg
from .task.lift_red_block_drawer_cfg import LiftRedBlockDrawerCfg
from .task.lift_red_block_slider_cfg import LiftRedBlockSliderCfg
from .task.lift_red_block_table_cfg import LiftRedBlockTableCfg
from .task.move_slider_left_cfg import MoveSliderLeftCfg
from .task.move_slider_right_cfg import MoveSliderRightCfg
from .task.open_drawer_cfg import OpenDrawerCfg
from .task.place_in_drawer_cfg import PlaceInDrawerCfg
from .task.place_in_slider_cfg import PlaceInSliderCfg
from .task.push_blue_block_left_cfg import PushBlueBlockLeftCfg
from .task.push_blue_block_right_cfg import PushBlueBlockRightCfg
from .task.push_into_drawer_cfg import PushIntoDrawerCfg
from .task.push_pink_block_left_cfg import PushPinkBlockLeftCfg
from .task.push_pink_block_right_cfg import PushPinkBlockRightCfg
from .task.push_red_block_left_cfg import PushRedBlockLeftCfg
from .task.push_red_block_right_cfg import PushRedBlockRightCfg
from .task.rotate_blue_block_left_cfg import RotateBlueBlockLeftCfg
from .task.rotate_blue_block_right_cfg import RotateBlueBlockRightCfg
from .task.rotate_pink_block_left_cfg import RotatePinkBlockLeftCfg
from .task.rotate_pink_block_right_cfg import RotatePinkBlockRightCfg
from .task.rotate_red_block_left_cfg import RotateRedBlockLeftCfg
from .task.rotate_red_block_right_cfg import RotateRedBlockRightCfg
from .task.stack_block_cfg import StackBlockCfg
from .task.unstack_block_cfg import UnstackBlockCfg


@configclass
class LiftRedBlockTableACfg(LiftRedBlockTableCfg):
    objects = SCENE_A.objects
    traj_filepath = "roboverse_data/trajs/calvin/lift_red_block_table_a/v2"


@configclass
class LiftRedBlockTableBCfg(LiftRedBlockTableCfg):
    objects = SCENE_B.objects
    traj_filepath = "roboverse_data/trajs/calvin/lift_red_block_table_a/v2"  # TODO: use scene_a for testing


@configclass
class LiftRedBlockTableCCfg(LiftRedBlockTableCfg):
    objects = SCENE_C.objects
    traj_filepath = "roboverse_data/trajs/calvin/lift_red_block_table_a/v2"  # TODO: use scene_a for testing


@configclass
class LiftRedBlockTableDCfg(LiftRedBlockTableCfg):
    objects = SCENE_D.objects
    traj_filepath = "roboverse_data/trajs/calvin/lift_red_block_table_a/v2"  # TODO: use scene_a for testing


@configclass
class LiftRedBlockSliderACfg(LiftRedBlockSliderCfg):
    objects = SCENE_A.objects
    traj_filepath = "roboverse_data/trajs/calvin/lift_red_block_slider_a/v2"


@configclass
class LiftRedBlockDrawerACfg(LiftRedBlockDrawerCfg):
    objects = SCENE_A.objects
    traj_filepath = "roboverse_data/trajs/calvin/lift_red_block_drawer_a/v2"


@configclass
class LiftBlueBlockTableACfg(LiftBlueBlockTableCfg):
    objects = SCENE_A.objects
    traj_filepath = "roboverse_data/trajs/calvin/lift_blue_block_table_a/v2"


@configclass
class LiftBlueBlockSliderACfg(LiftBlueBlockSliderCfg):
    objects = SCENE_A.objects
    traj_filepath = "roboverse_data/trajs/calvin/lift_blue_block_slider_a/v2"


@configclass
class LiftBlueBlockDrawerACfg(LiftBlueBlockDrawerCfg):
    objects = SCENE_A.objects
    traj_filepath = "roboverse_data/trajs/calvin/lift_blue_block_drawer_a/v2"


@configclass
class LiftPinkBlockTableACfg(LiftPinkBlockTableCfg):
    objects = SCENE_A.objects
    traj_filepath = "roboverse_data/trajs/calvin/lift_pink_block_table_a/v2"


@configclass
class LiftPinkBlockSliderACfg(LiftPinkBlockSliderCfg):
    objects = SCENE_A.objects
    traj_filepath = "roboverse_data/trajs/calvin/lift_pink_block_slider_a/v2"


@configclass
class LiftPinkBlockDrawerACfg(LiftPinkBlockDrawerCfg):
    objects = SCENE_A.objects
    traj_filepath = "roboverse_data/trajs/calvin/lift_pink_block_drawer_a/v2"


@configclass
class PlaceInSliderACfg(PlaceInSliderCfg):
    objects = SCENE_A.objects
    traj_filepath = "roboverse_data/trajs/calvin/place_in_slider_a/v2"


@configclass
class PlaceInDrawerACfg(PlaceInDrawerCfg):
    objects = SCENE_A.objects
    traj_filepath = "roboverse_data/trajs/calvin/place_in_drawer_a/v2"


@configclass
class RotateRedBlockRightACfg(RotateRedBlockRightCfg):
    objects = SCENE_A.objects
    traj_filepath = "roboverse_data/trajs/calvin/rotate_red_block_right_a/v2"


@configclass
class RotateRedBlockLeftACfg(RotateRedBlockLeftCfg):
    objects = SCENE_A.objects
    traj_filepath = "roboverse_data/trajs/calvin/rotate_red_block_left_a/v2"


@configclass
class RotateBlueBlockRightACfg(RotateBlueBlockRightCfg):
    objects = SCENE_A.objects
    traj_filepath = "roboverse_data/trajs/calvin/rotate_blue_block_right_a/v2"


@configclass
class RotateBlueBlockLeftACfg(RotateBlueBlockLeftCfg):
    objects = SCENE_A.objects
    traj_filepath = "roboverse_data/trajs/calvin/rotate_blue_block_left_a/v2"


@configclass
class RotatePinkBlockRightACfg(RotatePinkBlockRightCfg):
    objects = SCENE_A.objects
    traj_filepath = "roboverse_data/trajs/calvin/rotate_pink_block_right_a/v2"


@configclass
class RotatePinkBlockLeftACfg(RotatePinkBlockLeftCfg):
    objects = SCENE_A.objects
    traj_filepath = "roboverse_data/trajs/calvin/rotate_pink_block_left_a/v2"


@configclass
class PushRedBlockRightACfg(PushRedBlockRightCfg):
    objects = SCENE_A.objects
    traj_filepath = "roboverse_data/trajs/calvin/push_red_block_right_a/v2"


@configclass
class PushRedBlockLeftACfg(PushRedBlockLeftCfg):
    objects = SCENE_A.objects
    traj_filepath = "roboverse_data/trajs/calvin/push_red_block_left_a/v2"


@configclass
class PushBlueBlockRightACfg(PushBlueBlockRightCfg):
    objects = SCENE_A.objects
    traj_filepath = "roboverse_data/trajs/calvin/push_blue_block_right_a/v2"


@configclass
class PushBlueBlockLeftACfg(PushBlueBlockLeftCfg):
    objects = SCENE_A.objects
    traj_filepath = "roboverse_data/trajs/calvin/push_blue_block_left_a/v2"


@configclass
class PushPinkBlockRightACfg(PushPinkBlockRightCfg):
    objects = SCENE_A.objects
    traj_filepath = "roboverse_data/trajs/calvin/push_pink_block_right_a/v2"


@configclass
class PushPinkBlockLeftACfg(PushPinkBlockLeftCfg):
    objects = SCENE_A.objects
    traj_filepath = "roboverse_data/trajs/calvin/push_pink_block_left_a/v2"


@configclass
class MoveSliderLeftACfg(MoveSliderLeftCfg):
    objects = SCENE_A.objects
    traj_filepath = "roboverse_data/trajs/calvin/move_slider_left_a/v2"


@configclass
class MoveSliderRightACfg(MoveSliderRightCfg):
    objects = SCENE_A.objects
    traj_filepath = "roboverse_data/trajs/calvin/move_slider_right_a/v2"


@configclass
class OpenDrawerACfg(OpenDrawerCfg):
    objects = SCENE_A.objects
    traj_filepath = "roboverse_data/trajs/calvin/open_drawer_a/v2"


@configclass
class CloseDrawerACfg(CloseDrawerCfg):
    objects = SCENE_A.objects
    traj_filepath = "roboverse_data/trajs/calvin/close_drawer_a/v2"


@configclass
class StackBlockACfg(StackBlockCfg):
    objects = SCENE_A.objects
    traj_filepath = "roboverse_data/trajs/calvin/stack_block_a/v2"


@configclass
class UnstackBlockACfg(UnstackBlockCfg):
    objects = SCENE_A.objects
    traj_filepath = "roboverse_data/trajs/calvin/unstack_block_a/v2"


@configclass
class PushIntoDrawerACfg(PushIntoDrawerCfg):
    objects = SCENE_A.objects
    traj_filepath = "roboverse_data/trajs/calvin/push_into_drawer_a/v2"
