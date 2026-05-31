"""YAM lift_cube proprio actor-obs parity: native mjlab vs RoboVerse v2 port.

Injects K identical matched states (8 joints + cube free-joint pose + goal
target + last_action) into BOTH the native mjlab YAM env and the RoboVerse
``mjlab.lift_cube_yam_v2`` env, recomputes the full ``actor`` observation on
each side, and reports max|Δ| overall and per native obs term.

Reuses the build / inject / layout helpers from ``scripts/parity_obs_all.py``
(``build_mjlab`` / ``build_rv`` / ``mjlab_inject`` / ``rv_inject`` /
``_yam_builder`` / ``native_term_layout`` / ``native_compute_actor`` /
``rv_compute_actor``).

The IMAGE obs group of the camera variants (_depth/_rgb/_seg) is OUT OF SCOPE
for bitwise parity (needs matched rendering); only the proprio actor group of
``lift_cube_yam_v2`` is compared here.

Run:
    MJLAB_REPO=/workspace/mjlab_upstream PYTHONPATH=$PWD MUJOCO_GL=egl \
        CUDA_VISIBLE_DEVICES=0 python scripts/parity_obs_yam.py
"""

from __future__ import annotations

import numpy as np
import torch

from scripts.parity_obs_all import (
    DEVICE,
    build_mjlab,
    build_rv,
    native_compute_actor,
    native_term_layout,
    rv_compute_actor,
    _yam_builder,
)

K = 32
SEED = 0
TOL = 1e-5

# native YAM joint order (entity.joint_names) = MJCF declaration order.
_JOINTS = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "left_finger", "right_finger")


def _sample(rng):
    """Sample a matched YAM state: 8 joints, cube pose, goal, 7-dim last action."""
    jpos = rng.uniform(-0.5, 0.5, size=8)
    # Keep finger joints inside their (tiny) ranges so the equality stays sane.
    jpos[6] = rng.uniform(-0.002, 0.037)
    jpos[7] = -jpos[6]
    jvel = rng.uniform(-1.0, 1.0, size=8)
    cube_pos = np.array([rng.uniform(0.2, 0.4), rng.uniform(-0.2, 0.2), rng.uniform(0.03, 0.3)])
    yaw = rng.uniform(-3.14, 3.14)
    cube_quat = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])  # wxyz
    goal = np.array([rng.uniform(0.2, 0.4), rng.uniform(-0.2, 0.2), rng.uniform(0.15, 0.35)])
    act = rng.uniform(-1.0, 1.0, size=7)
    return jpos, jvel, cube_pos, cube_quat, goal, act


def _mjlab_inject(env, *, jpos, jvel, cube_pos, cube_quat, goal, act):
    robot = env.scene["robot"]
    cube = env.scene["cube"]
    robot.write_joint_state_to_sim(
        torch.tensor([jpos], dtype=torch.float, device=DEVICE),
        torch.tensor([jvel], dtype=torch.float, device=DEVICE),
    )
    # cube root state = pos(3) quat_wxyz(4) linvel(3) angvel(3).
    root = np.concatenate([cube_pos, cube_quat, np.zeros(6)])
    cube.write_root_state_to_sim(torch.tensor([root], dtype=torch.float, device=DEVICE))
    env.sim.forward()
    env.action_manager.process_action(torch.tensor([act], dtype=torch.float, device=DEVICE))
    term = env.command_manager.get_term("lift_height")
    term.target_pos[:] = torch.tensor(goal, dtype=torch.float, device=DEVICE)


def _rv_inject(env, *, jpos, jvel, cube_pos, cube_quat, goal, act):
    ph = env.handler.physics
    mp = ph.model.ptr if hasattr(ph.model, "ptr") else ph.model
    import mujoco

    # qpos offsets per joint (handles 8 arm/finger joints + cube freejoint).
    jadr = [int(mp.jnt_qposadr[mujoco.mj_name2id(mp, mujoco.mjtObj.mjOBJ_JOINT, n)]) for n in _JOINTS]
    dadr = [int(mp.jnt_dofadr[mujoco.mj_name2id(mp, mujoco.mjtObj.mjOBJ_JOINT, n)]) for n in _JOINTS]
    cube_bid = mujoco.mj_name2id(mp, mujoco.mjtObj.mjOBJ_BODY, "cube")
    cube_jadr = int(mp.jnt_qposadr[mp.body_jntadr[cube_bid]])
    cube_dadr = int(mp.jnt_dofadr[mp.body_jntadr[cube_bid]])
    with ph.reset_context():
        for a, v in zip(jadr, jpos):
            ph.data.qpos[a] = v
        for a, v in zip(dadr, jvel):
            ph.data.qvel[a] = v
        ph.data.qpos[cube_jadr : cube_jadr + 3] = cube_pos
        ph.data.qpos[cube_jadr + 3 : cube_jadr + 7] = cube_quat
        ph.data.qvel[cube_dadr : cube_dadr + 6] = 0.0
    env._action = torch.tensor([act], dtype=torch.float, device=env.device)
    mgr = env.command_managers["lift_height"]
    mgr._target_pos[:] = torch.tensor(goal, dtype=torch.float, device=env.device)


def main():
    rng = np.random.default_rng(SEED)
    mj = build_mjlab(_yam_builder)
    rv = build_rv("mjlab.lift_cube_yam_v2")

    mj_layout = native_term_layout(mj)
    print("native actor layout:", mj_layout)
    overall = 0.0
    per_term = {name: 0.0 for name, _ in mj_layout}

    for _ in range(K):
        jpos, jvel, cube_pos, cube_quat, goal, act = _sample(rng)
        _mjlab_inject(mj, jpos=jpos, jvel=jvel, cube_pos=cube_pos, cube_quat=cube_quat, goal=goal, act=act)
        _rv_inject(rv, jpos=jpos, jvel=jvel, cube_pos=cube_pos, cube_quat=cube_quat, goal=goal, act=act)
        mo = native_compute_actor(mj)
        ro = rv_compute_actor(rv)
        assert mo.shape == ro.shape, f"shape mismatch native={mo.shape} rv={ro.shape}"
        off = 0
        for name, d in mj_layout:
            md = float(np.abs(mo[off : off + d] - ro[off : off + d]).max())
            per_term[name] = max(per_term[name], md)
            overall = max(overall, md)
            off += d

    print(f"\nnative_dim={sum(d for _, d in mj_layout)} rv_dim={ro.shape[-1]}  K={K}")
    print(f"{'term':16s} {'dim':>4s} {'max|Δ|':>12s}  verdict")
    print("-" * 46)
    for name, d in mj_layout:
        v = "PASS" if per_term[name] <= TOL else "FAIL"
        print(f"{name:16s} {d:>4d} {per_term[name]:>12.3e}  {v}")
    print("-" * 46)
    print(f"max|Δproprio| over {K} states = {overall:.3e}  -> {'PASS' if overall <= TOL else 'FAIL'}")


if __name__ == "__main__":
    main()
