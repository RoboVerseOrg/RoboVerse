"""Backward-compat smoke for all 12 mjlab v2 tasks.

Run after any RoboVerse / MetaSim change to confirm:
  - All 12 v2 task names register
  - All 12 load + reset + step on mujoco (scene-MJCF path)
  - 8/12 load + reset + step on Newton GPU (RobotCfg path)

Exits 0 if mujoco 12/12 + newton ≥ 8/12, else 1.
"""

from __future__ import annotations

import sys

import torch

import roboverse_pack  # noqa: F401  (triggers task registry)
from metasim.scenario.scenario import ScenarioCfg
from metasim.scenario.simulator_params import SimParamCfg
from metasim.task.registry import get_task_class

TASKS = [
    ("mjlab.cartpole_balance_v2", "mjlab_cartpole", 1),
    ("mjlab.cartpole_swingup_v2", "mjlab_cartpole", 1),
    ("mjlab.velocity_flat_go1_v2", "mjlab_go1", 12),
    ("mjlab.velocity_rough_go1_v2", "mjlab_go1", 12),
    ("mjlab.velocity_flat_g1_v2", "mjlab_g1", 29),
    ("mjlab.velocity_rough_g1_v2", "mjlab_g1", 29),
    ("mjlab.tracking_flat_g1_v2", "mjlab_g1", 29),
    ("mjlab.tracking_flat_g1_no_state_est_v2", "mjlab_g1", 29),
    ("mjlab.lift_cube_yam_v2", "mjlab_yam", 6),
    ("mjlab.lift_cube_yam_depth_v2", "mjlab_yam", 6),
    ("mjlab.lift_cube_yam_rgb_v2", "mjlab_yam", 6),
    ("mjlab.multi_cube_seg_yam_v2", "mjlab_yam", 6),
]

# Newton dual-path now covers all 12 tasks. The yam lift_cube family injects
# its cube + static table as MetaSim scene objects (PrimitiveCubeCfg) on the
# Newton path — the equivalent of mjlab's MJCF cube/table patch.
NEWTON_DEFERRED: set[str] = set()


def smoke_mujoco() -> tuple[int, int]:
    print("=== Mujoco scene-MJCF backward-compat ===")
    pass_ct = 0
    for task_name, _robot, n_actions in TASKS:
        try:
            TCls = get_task_class(task_name)
            env = TCls()
            env.reset()
            action = torch.zeros((1, n_actions))
            ret = env.step(action)
            print(f"  PASS {task_name}  r={ret[1].item():.3f}")
            pass_ct += 1
        except Exception as e:
            print(f"  FAIL {task_name}: {type(e).__name__}: {str(e)[:120]}")
    return pass_ct, len(TASKS)


def smoke_newton() -> tuple[int, int]:
    print("=== Newton GPU dual-path smoke ===")
    pass_ct = 0
    attempted = 0
    for task_name, robot_name, n_actions in TASKS:
        if task_name in NEWTON_DEFERRED:
            print(f"  SKIP {task_name}  (yam cube/table injection deferred)")
            continue
        attempted += 1
        try:
            scn = ScenarioCfg(
                robots=[robot_name],
                objects=[],
                cameras=[],
                sim_params=SimParamCfg(dt=0.005),
                decimation=4,
                simulator="newton",
                num_envs=4,
                headless=True,
                add_default_ground=True,
            )
            TCls = get_task_class(task_name)
            env = TCls(scenario=scn, device="cuda:0")
            env.reset()
            action = torch.zeros((4, n_actions), device="cuda:0")
            for _ in range(3):
                ret = env.step(action)
            print(f"  PASS {task_name}  r={ret[1].mean().item():.3f}")
            pass_ct += 1
        except Exception as e:
            print(f"  FAIL {task_name}: {type(e).__name__}: {str(e)[:120]}")
    return pass_ct, attempted


if __name__ == "__main__":
    mj_ok, mj_total = smoke_mujoco()
    nt_ok, nt_total = smoke_newton()
    print()
    print(f"Mujoco:  {mj_ok}/{mj_total}")
    print(f"Newton:  {nt_ok}/{nt_total}  (yam {len(NEWTON_DEFERRED)} deferred)")
    if mj_ok < mj_total:
        print("FAIL: mujoco regression — backward compat broken")
        sys.exit(1)
    if nt_ok < nt_total:
        print("FAIL: newton dual-path regression")
        sys.exit(1)
    print("OK: 12/12 mujoco + 8/8 newton dual-path")
    sys.exit(0)
