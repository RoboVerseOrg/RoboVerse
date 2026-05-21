"""Cross-framework obs + reward parity harness: RoboVerse vs mjlab native.

Reward *formula* parity was already shown byte-identical in isolation. This
harness goes further: it drives the *full manager stack* on both sides from
the *same* joint state + raw action, and compares

  - the assembled actor observation vector (5D), and
  - the total (raw, weight=1) reward

so any divergence in obs ordering, default-offset handling, or reward
aggregation surfaces. State is injected directly (no integrator stepping),
so this isolates the MDP layer from backend physics.

Run:
    MUJOCO_GL=egl python scripts/parity_obs_reward_cartpole.py
"""

from __future__ import annotations

import numpy as np
import torch

K = 64  # matched random states to compare
SEED = 0
DEVICE = "cuda:0"


def build_mjlab():
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.cartpole.cartpole_env_cfg import cartpole_balance_env_cfg

    cfg = cartpole_balance_env_cfg(play=True)  # play => obs corruption off
    cfg.scene.num_envs = 1
    env = ManagerBasedRlEnv(cfg, device=DEVICE)
    env.reset()
    return env


def build_roboverse():
    import roboverse_pack  # noqa: F401
    from metasim.task.registry import get_task_class

    env = get_task_class("mjlab.cartpole_balance_v2")()  # mujoco scene-MJCF, 1 env
    env.reset()
    return env


def mjlab_set_and_eval(env, q, qd, action):
    ent = env.scene["cartpole"]
    pos = torch.tensor([q], dtype=torch.float, device=DEVICE)
    vel = torch.tensor([qd], dtype=torch.float, device=DEVICE)
    ent.write_joint_state_to_sim(pos, vel)
    # raw action enters reward via small_control; set the action buffer.
    a = torch.tensor([[action]], dtype=torch.float, device=DEVICE)
    env.action_manager.process_action(a)
    # compute() caches _obs_buffer (populated at reset); invalidate so the
    # injected joint state is reflected. Cartpole has no history/delay terms.
    env.observation_manager._obs_buffer = None
    obs = env.observation_manager.compute()["actor"][0].detach().cpu().numpy()
    rew = float(env.reward_manager.compute(env.step_dt)[0].item()) / env.step_dt
    return obs, rew


def rv_set_and_eval(env, q, qd, action):
    physics = env.handler.physics
    with physics.reset_context():
        physics.data.qpos[0], physics.data.qpos[1] = q
        physics.data.qvel[0], physics.data.qvel[1] = qd
    env._action = torch.tensor([[action]], dtype=torch.float, device=env.device)
    states = env.handler.get_states(mode="tensor")
    obs = env._observation_group("actor", states)[0].detach().cpu().numpy()
    raw = float(env._reward(states)[0].item()) / (env.step_dt)
    return obs, raw


def main():
    rng = np.random.default_rng(SEED)
    print("building mjlab native cartpole ...")
    mj = build_mjlab()
    print("building RoboVerse cartpole_balance_v2 ...")
    rv = build_roboverse()

    obs_dmax = 0.0
    rew_dmax = 0.0
    rows = []
    for i in range(K):
        slider = float(rng.uniform(-1.5, 1.5))
        hinge = float(rng.uniform(-np.pi, np.pi))
        svel = float(rng.uniform(-3, 3))
        hvel = float(rng.uniform(-6, 6))
        act = float(rng.uniform(-1.5, 1.5))
        q = (slider, hinge)
        qd = (svel, hvel)
        mo, mr = mjlab_set_and_eval(mj, q, qd, act)
        ro, rr = rv_set_and_eval(rv, q, qd, act)
        od = float(np.max(np.abs(mo - ro)))
        rd = abs(mr - rr)
        obs_dmax = max(obs_dmax, od)
        rew_dmax = max(rew_dmax, rd)
        if i < 5:
            rows.append((i, mo, ro, mr, rr, od, rd))

    print("\nfirst 5 matched states (mjlab vs RoboVerse):")
    for i, mo, ro, mr, rr, od, rd in rows:
        print(f"  [{i}] obs|Δ|={od:.2e} rew mjlab={mr:.6f} rv={rr:.6f} |Δ|={rd:.2e}")
        print(f"        mjlab obs={np.array2string(mo, precision=5)}")
        print(f"        rv    obs={np.array2string(ro, precision=5)}")

    print(f"\n=== over {K} matched states ===")
    print(f"max |Δ obs|    = {obs_dmax:.3e}")
    print(f"max |Δ reward| = {rew_dmax:.3e}")
    tol = 1e-4
    ok = obs_dmax < tol and rew_dmax < tol
    print(f"VERDICT: {'PASS (1:1 within 1e-4)' if ok else 'MISMATCH'}")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
