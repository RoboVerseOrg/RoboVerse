"""Replay official ManiSkill demonstration trajectories in the shipped native tasks.

Loads a ManiSkill ``pd_joint_delta_pos`` demo ``.h5`` (e.g.
``~/.maniskill/demos/PickCube-v1/rl/trajectory.none.pd_joint_delta_pos.physx_cuda.h5``), sets each
episode's initial state (object poses + robot qpos/qvel + goal) on the shipped
``maniskill.<name>_native`` task, replays the recorded 8-dim actions, and checks whether the task's
ported success fires — i.e. demo-level 1:1 reproduction through the standard handler path.

Run::

    SAPIEN_HEADLESS=1 python -m tools.maniskill_integration.replay_demo --task pick_cube --episodes 25
"""

from __future__ import annotations

import argparse
import copy

import numpy as np

# articulation env-state layout: root_pose(7) + root_vel(6) + qpos(n) + qvel(n).
_ROOT = 13

# task_key -> (demo h5 path, object name in the demo whose pose seeds the goal).
DEMO_PATHS = {
    "pick_cube": "~/.maniskill/demos/PickCube-v1/rl/trajectory.none.pd_joint_delta_pos.physx_cuda.h5",
}
GOAL_ACTOR = {"pick_cube": "goal_site"}


def replay(task_key: str, episodes: int, h5_path: str | None = None) -> dict:
    import os

    import h5py
    import sapien
    import torch

    import roboverse_pack.tasks.maniskill  # noqa: F401 — registers tasks
    from metasim.task.registry import get_task_class

    path = os.path.expanduser(h5_path or DEMO_PATHS[task_key])
    f = h5py.File(path, "r")
    traj_keys = sorted((k for k in f.keys() if k.startswith("traj_")), key=lambda s: int(s.split("_")[1]))
    traj_keys = traj_keys[:episodes]

    cls = get_task_class(f"maniskill.{task_key}_native")
    sc = copy.deepcopy(cls.scenario)
    sc.simulator = "sapien3"
    sc.num_envs = 1
    sc.headless = True
    sc.cameras = []
    task = cls(sc)
    # object names present both in the demo and the shipped scenario (skip table/goal markers).
    scene_objs = {o.name for o in sc.objects}
    goal_actor = GOAL_ACTOR.get(task_key)

    demo_succ = our_succ = 0
    for tk in traj_keys:
        g = f[tk]
        es = g["env_states"]
        actions = np.asarray(g["actions"])  # (T, 8)
        demo_success = bool(np.asarray(g["success"])[-1])
        demo_succ += int(demo_success)

        task.reset()
        # set robot to the demo's initial state
        art = np.asarray(es["articulations"]["panda"])[0]
        nq = (len(art) - _ROOT) // 2
        task.handler.object_ids["panda"].set_qpos(art[_ROOT : _ROOT + nq].astype(np.float32))
        task.handler.object_ids["panda"].set_qvel(art[_ROOT + nq : _ROOT + 2 * nq].astype(np.float32))
        for name in es["actors"]:
            # Only seed the manipuland objects — the kinematic table stays at its recipe default
            # (the demo stores a table pose in its own frame; re-setting it would shift the scene).
            if name in scene_objs and "table" not in name:
                p = np.asarray(es["actors"][name])[0]
                task.handler.object_ids[name].set_pose(sapien.Pose(p[:3], p[3:7]))
        if goal_actor is not None:
            task.goal_pos = np.asarray(es["actors"][goal_actor])[0][:3].astype(np.float64)

        terminated = False
        for a in actions:
            out = task.step(torch.tensor(a, dtype=torch.float32).unsqueeze(0))
            terminated = terminated or bool(out[2].reshape(-1)[0])
        our_succ += int(terminated)
    task.close()
    f.close()
    return {"task": task_key, "episodes": len(traj_keys), "demo_success": demo_succ, "replay_success": our_succ}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", default="pick_cube")
    ap.add_argument("--episodes", type=int, default=25)
    ap.add_argument("--h5", default=None)
    args = ap.parse_args(argv)
    r = replay(args.task, args.episodes, args.h5)
    print(
        f"{r['task']}: replayed {r['episodes']} demos — demo success {r['demo_success']}/{r['episodes']}, "
        f"shipped-task replay success {r['replay_success']}/{r['episodes']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
