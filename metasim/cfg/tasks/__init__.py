# ruff: noqa: F401

"""Sub-module containing the task configuration."""

import time

from loguru import logger as log

from .base_task_cfg import BaseTaskCfg


def __get_quick_ref():
    tic = time.time()

    from .calvin.calvin import MoveSliderLeftACfg

    # from .debug.reach_origin_cfg import ReachOriginCfg
    from .dmcontrol.walker_walk_cfg import WalkerWalkCfg
    from .fetch import FetchCloseBoxCfg
    from .gapartnet import GapartnetOpenDrawerCfg
    from .humanoidbench import StandCfg
    from .isaacgym_envs.allegrohand_reorientation_cfg import AllegroHandReorientationCfg
    from .isaacgym_envs.ant_isaacgym_cfg import AntIsaacGymCfg
    from .libero.libero_objects.libero_pick_alphabet_soup import LiberoPickAlphabetSoupCfg
    from .libero.libero_objects.libero_pick_bbq_sauce import LiberoPickBbqSauceCfg
    from .libero.libero_objects.libero_pick_butter import LiberoPickButterCfg
    from .libero.libero_objects.libero_pick_chocolate_pudding import LiberoPickChocolatePuddingCfg
    from .libero.libero_objects.libero_pick_cream_cheese import LiberoPickCreamCheeseCfg
    from .libero.libero_objects.libero_pick_ketchup import LiberoPickKetchupCfg
    from .libero.libero_objects.libero_pick_milk import LiberoPickMilkCfg
    from .libero.libero_objects.libero_pick_orange_juice import LiberoPickOrangeJuiceCfg
    from .libero.libero_objects.libero_pick_salad_dressing import LiberoPickSaladDressingCfg
    from .libero.libero_objects.libero_pick_tomato_sauce import LiberoPickTomatoSauceCfg
    from .maniskill.pick_cube_cfg import PickCubeCfg
    from .maniskill.pick_single_ycb import PickSingleYcbCrackerBoxCfg
    from .maniskill.stack_cube_cfg import StackCubeCfg
    from .rlafford.rl_afford_open_door_cfg import RlAffordOpenDoorCfg
    from .rlbench.basketball_in_hoop_cfg import BasketballInHoopCfg
    from .rlbench.close_box_cfg import CloseBoxCfg
    from .robosuite import SquareD0Cfg, SquareD1Cfg, SquareD2Cfg, StackD0Cfg
    from .simpler_env.simpler_env_grasp_opened_coke_can_cfg import SimplerEnvGraspOpenedCokeCanCfg
    from .simpler_env.simpler_env_move_near import SimplerEnvMoveNearCfg

    # from .skillblender import G1BaseTaskCfg, H1BaseTaskCfg
    from .uh1 import MabaoguoCfg

    toc = time.time()

    log.trace(f"Time taken to load quick ref: {toc - tic:.2f} seconds")

    return locals()


__quick_ref = __get_quick_ref()


def __getattr__(name):
    if name in __quick_ref:
        return __quick_ref[name]

    if name.startswith("GraspNet") and name.endswith("Cfg"):
        from .graspnet import __getattr__ as graspnet_getattr

        return graspnet_getattr(name)

    elif name.startswith("GAPartManip") and name.endswith("Cfg"):
        from .gapartmanip import __getattr__ as gapartmanip_getattr

        return gapartmanip_getattr(name)

    else:
        raise AttributeError(f"Module {__name__} has no attribute {name}")
        raise AttributeError(f"Module {__name__} has no attribute {name}")
