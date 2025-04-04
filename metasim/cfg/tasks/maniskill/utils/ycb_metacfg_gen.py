import os

ycb_object_list = [
    "002_master_chef_can",
    "003_cracker_box",
    "004_sugar_box",
    "005_tomato_soup_can",
    "006_mustard_bottle",
    "007_tuna_fish_can",
    "008_pudding_box",
    "009_gelatin_box",
    "010_potted_meat_can",
    "011_banana",
    "012_strawberry",
    "013_apple",
    "014_lemon",
    "015_peach",
    "016_pear",
    "017_orange",
    "018_plum",
    "019_pitcher_base",
    "021_bleach_cleanser",
    "024_bowl",
    "025_mug",
    "026_sponge",
    "030_fork",
    "031_spoon",
    "032_knife",
    "033_spatula",
    "035_power_drill",
    "036_wood_block",
    "037_scissors",
    "038_padlock",
    "040_large_marker",
    "042_adjustable_wrench",
    "043_phillips_screwdriver",
    "044_flat_screwdriver",
    "048_hammer",
    "050_medium_clamp",
    "051_large_clamp",
    "052_extra_large_clamp",
    "053_mini_soccer_ball",
    "054_softball",
    "055_baseball",
    "056_tennis_ball",
    "057_racquetball",
    "058_golf_ball",
    "061_foam_brick",
    "062_dice",
    "063-a_marbles",
    "063-b_marbles",
    "065-a_cups",
    "065-b_cups",
    "065-c_cups",
    "065-d_cups",
    "065-e_cups",
    "065-f_cups",
    "065-g_cups",
    "065-h_cups",
    "065-i_cups",
    "065-j_cups",
    "070-a_colored_wood_blocks",
    "070-b_colored_wood_blocks",
    "071_nine_hole_peg_test",
    "072-a_toy_airplane",
    "072-b_toy_airplane",
    "072-c_toy_airplane",
    "072-d_toy_airplane",
    "072-e_toy_airplane",
    "073-a_lego_duplo",
    "073-b_lego_duplo",
    "073-c_lego_duplo",
    "073-d_lego_duplo",
    "073-e_lego_duplo",
    "073-f_lego_duplo",
    "073-g_lego_duplo",
    "077_rubiks_cube",
]

base_cfg = """from metasim.cfg.checkers import PositionShiftChecker
from metasim.cfg.objects import RigidObjCfg
from metasim.constants import PhysicStateType
from metasim.utils import configclass

from ..maniskill_task_cfg import ManiskillTaskCfg


@configclass
class PickSingleYcb{Obj}Cfg(ManiskillTaskCfg):
    episode_length = 250
    objects = [
        RigidObjCfg(
            name="obj",
            usd_path="roboverse_data/assets/maniskill/ycb/{ycb_obj}/object_proc.usd",
            urdf_path="roboverse_data/assets/maniskill/ycb/{ycb_obj}/model_scaled.urdf",
            physics=PhysicStateType.RIGIDBODY,
        )
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_single_ycb/trajectory-franka-{ycb_obj}_v2.pkl"

    checker = PositionShiftChecker(
        obj_name="obj",
        distance=0.075,
        axis="z",
    )
"""


init_script_content = ""
task_names = []
os.makedirs("metasim/cfg/tasks/maniskill/pick_single_ycb", exist_ok=True)
for ycb_obj in ycb_object_list:
    obj = "_".join(ycb_obj.split("_")[1:])
    Obj = obj.title().replace("_", "")
    with open(f"metasim/cfg/tasks/maniskill/pick_single_ycb/pick_single_ycb_{obj}_cfg.py", "w") as f:
        f.write(base_cfg.format(ycb_obj=ycb_obj, obj=obj, Obj=Obj))
    init_script_content += f"from .pick_single_ycb_{obj}_cfg import PickSingleYcb{Obj}Cfg\n"
    task_names.append(f"PickSingleYcb{Obj}")

with open("metasim/cfg/tasks/maniskill/pick_single_ycb/__init__.py", "w") as f:
    f.write(init_script_content)

with open("metasim/cfg/tasks/maniskill/run_pick_single_ycb.sh", "w") as f:
    f.write(f"""#!/bin/bash

for task in {" ".join(task_names)}; do
    python metasim/scripts/replay_demo.py --sim=isaaclab --task=$task --num_envs=16
done

""")
