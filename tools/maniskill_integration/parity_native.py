"""Measure native-ManiSkill ↔ SAPIEN3-recipe 1:1 parity for a task.

For one ManiSkill task we:

1. Reset the native env on ``physx_cpu`` (which is bit-for-bit deterministic),
   capture the full scene (``recipe.capture_native``), then step a fixed,
   reproducible action sequence and record robot ``qpos`` and every actor's pose
   at every control step.
2. Rebuild the captured scene from scratch with the reproduction recipe
   (``recipe.build_replica``) and replay the *identical* actions through the
   vendored ManiSkill controller.
3. Report the per-step max delta on robot ``qpos`` and on each object's pose.

``cube``-style object state comes out bit-for-bit (Δ = 0); robot ``qpos`` lands
at float32 solver-roundoff (~1e-4) over dozens of aggressive random steps.

Usage::

    python -m tools.maniskill_integration.parity_native --task PickCube-v1 --steps 30
    python -m tools.maniskill_integration.parity_native --all --json out.json
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from . import recipe as R

# Tasks whose scene is built entirely from primitive / convex rigid bodies on
# the standard table — the ones the harness can rebuild generically today.
PRIMITIVE_TASKS = [
    "PickCube-v1",
    "PushCube-v1",
    "PullCube-v1",
    "StackCube-v1",
    "PokeCube-v1",
    "LiftPegUpright-v1",
    "RollBall-v1",
    "PlaceSphere-v1",
    "PullCubeTool-v1",
    "PegInsertionSide-v1",
    "StackPyramid-v1",
    "PlugCharger-v1",
    "PushT-v1",  # panda_stick (7-dim arm-only) — action-level 1:1
    "DrawTriangle-v1",  # panda_stick (7-dim arm-only)
]

# Known limitations of the generic harness (honest coverage notes):
#  - Multi-agent tasks (TwoRobot*) expose ``agent`` as a MultiAgent, not a single
#    robot — needs per-agent capture/replay.
#  - Mesh-asset objects (PickSingleYCB foam brick) come out near-bitwise (~1e-5)
#    because the convex collision hull is rebuilt from captured vertices rather
#    than the exact source mesh.

# Default control mode per task (all panda tabletop tasks share the same one).
DEFAULT_CONTROL_MODE = "pd_joint_delta_pos"


def _make_controller(action_dim: int, n_active_joints: int):
    """Build the pd_joint_delta_pos controller matching the robot's action layout.

    A mimic gripper adds one extra active DOF beyond the action width (the panda's two fingers
    against one gripper action), so ``n_active == action_dim + 1`` ⇒ arm + gripper; ``==`` ⇒ arm-only
    (panda_stick / pusher robots).
    """
    from roboverse_pack.tasks.maniskill._native.control import PDJointDeltaPos

    gripper = n_active_joints == action_dim + 1
    arm_dof = action_dim - 1 if gripper else action_dim
    return PDJointDeltaPos(arm_dof=arm_dof, gripper=gripper)


def run_task(task_id: str, steps: int, seed: int, control_mode: str = DEFAULT_CONTROL_MODE) -> dict:
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401 — registers envs
    import torch

    rng = np.random.RandomState(seed + 1)

    # --- 1. native reference -------------------------------------------------
    env = gym.make(task_id, num_envs=1, obs_mode="state", control_mode=control_mode, sim_backend="physx_cpu")
    u = env.unwrapped
    env.reset(seed=seed)
    cap = R.capture_native(env)
    adim = env.action_space.shape[-1]
    actions = rng.uniform(-1, 1, size=(steps, adim)).astype(np.float32)

    ref_qpos = []
    ref_actors = {a.name: [] for a in cap.actors}
    for a in actions:
        env.step(torch.tensor(a).unsqueeze(0))
        ref_qpos.append(u.agent.robot.get_qpos().cpu().numpy().ravel().copy())
        for name in ref_actors:
            ref_actors[name].append(u.scene.actors[name].pose.raw_pose.cpu().numpy().ravel().copy())
    env.close()
    ref_qpos = np.asarray(ref_qpos)
    ref_actors = {k: np.asarray(v) for k, v in ref_actors.items()}

    # --- 2. recipe replica ---------------------------------------------------
    scene, robot, actor_objs = R.build_replica(cap)
    decim = cap.sim.decimation
    rep_qpos = []
    rep_actors = {name: [] for name in actor_objs}
    ajoints = robot.get_active_joints()
    controller = _make_controller(adim, len(ajoints))
    for a in actions:
        q = robot.get_qpos().astype(np.float32).copy()
        target = controller.compute_targets(q, a)
        for k, j in enumerate(ajoints):
            j.set_drive_target(float(target[k]))
        for _ in range(decim):
            scene.step()
        rep_qpos.append(robot.get_qpos().copy())
        for name, obj in actor_objs.items():
            p = obj.get_pose()
            rep_actors[name].append(np.concatenate([p.p, p.q]))
    rep_qpos = np.asarray(rep_qpos)
    rep_actors = {k: np.asarray(v) for k, v in rep_actors.items()}

    # --- 3. compare ----------------------------------------------------------
    dq = np.abs(ref_qpos - rep_qpos)
    result = {
        "task": task_id,
        "steps": steps,
        "seed": seed,
        "control_mode": control_mode,
        "action_dim": adim,
        "n_actors": len(cap.actors),
        "qpos_delta_max": float(dq.max()) if dq.size else None,
        "qpos_delta_step0": float(dq[0].max()) if dq.size else None,
        "qpos_delta_final": float(dq[-1].max()) if dq.size else None,
        "objects": {},
    }
    for name in rep_actors:
        if name not in ref_actors:
            continue
        d = np.abs(ref_actors[name] - rep_actors[name])
        result["objects"][name] = {
            "delta_max": float(d.max()),
            "delta_final": float(d[-1].max()),
            "bitwise": bool(d.max() == 0.0),
        }
    return result


def _fmt(r: dict) -> str:
    parts = []
    for n, o in r["objects"].items():
        tag = "Δ=0" if o["bitwise"] else "{:.2e}".format(o["delta_max"])
        parts.append(f"{n}:{tag}")
    objs = "  ".join(parts)
    return f"{r['task']:20s} qpos_max={r['qpos_delta_max']:.2e} (step0={r['qpos_delta_step0']:.2e})  objects[{objs}]"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", default="PickCube-v1")
    ap.add_argument("--all", action="store_true", help="run every primitive task")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--control-mode", default=DEFAULT_CONTROL_MODE)
    ap.add_argument("--json", default=None, help="write results to this JSON file")
    args = ap.parse_args(argv)

    tasks = PRIMITIVE_TASKS if args.all else [args.task]
    results = []
    for t in tasks:
        try:
            r = run_task(t, args.steps, args.seed, args.control_mode)
            results.append(r)
            print(_fmt(r))
        except Exception as e:
            print(f"{t:20s} ERROR: {type(e).__name__}: {e}")
            results.append({"task": t, "error": f"{type(e).__name__}: {e}"})

    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
