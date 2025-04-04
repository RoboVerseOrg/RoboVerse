import os

base_cfg = """from metasim.cfg.checkers.checkers import DetectedChecker
from metasim.cfg.checkers.detectors import RelativeBboxDetector
from metasim.cfg.objects import RigidObjCfg
from metasim.constants import PhysicStateType
from metasim.utils import configclass

from ..maniskill_task_cfg import ManiskillTaskCfg


@configclass
class PegInsertionSide{i_case}Cfg(ManiskillTaskCfg):
    episode_length = 250
    objects = [
        # The hole on the base is slightly enlarged to reduce the difficulty due to the dynamics gap
        RigidObjCfg(
            name="box",
            usd_path="data_isaaclab/assets/maniskill2/peg_insertion/usd/base_{i_case}.usd",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="stick",
            usd_path="data_isaaclab/assets/maniskill2/peg_insertion/usd/stick_{i_case}.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    traj_filepath = "data/source_data/maniskill2/rigid_body/PegInsertionSide-v0/trajs/trajectory-franka-{i_case}_v2.pkl"

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
"""

# For the checker: The hole on the base are at different positions for different asset.
# The core logic is to check the stick position along the x-axis, relative to the base.

# The stick is slightly shrinked for the peg insertion task.

n_case = 1000

init_script_content = ""
os.makedirs("metasim/cfg/tasks/maniskill/peg_insertion_side", exist_ok=True)
for i_case in range(n_case):
    with open(f"metasim/cfg/tasks/maniskill/peg_insertion_side/peg_insertion_side_{i_case}_cfg.py", "w") as f:
        f.write(base_cfg.format(i_case=i_case))
    init_script_content += f"from .peg_insertion_side_{i_case}_cfg import PegInsertionSide{i_case}Cfg\n"

with open("metasim/cfg/tasks/maniskill/peg_insertion_side/__init__.py", "w") as f:
    f.write(init_script_content)
