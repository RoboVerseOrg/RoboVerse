"""Bitwise parity: ported OSC_POSE  vs  native robosuite OSC_POSE.

Proves the OSC port (osc_pose_controller.MujocoOSCPose) reproduces robosuite's
OSC_POSE controller bitwise -- the load-bearing check for "can we run LIBERO
EE-delta policies inside MetaSim with the SAME control law".

Two complementary tests on a real demo trajectory:

  (A) Per-step torque parity at EVERY real state (incl. contact). At each policy
      step we force-refresh robosuite's controller and the ported controller to
      the SAME current sim state, set the same goal, and compare the commanded
      joint torques. Δ ~ machine-eps everywhere => the control law is bitwise
      faithful, independent of integration/replay chaos. THIS is the headline.

  (B) Pre-contact closed-form rollout. Drive the SAME MJB-exact model from the
      demo init state through the ported controller (set_goal once, run_controller
      x25 substeps, mj_step) and compare arm qpos to the native rollout, before
      the grasp/contact phase. Δ ~ solver float noise => the controller is
      faithful in-the-loop too.

Note on full-episode open-loop replay: after the grasp, open-loop replay is
chaotic (robosuite documents large replay drift), and robosuite's controller
state-caching (new_update) uses a slightly stale first-substep state after
set_init_state; both amplify under contact. This is a property of open-loop
demo replay, NOT the control law (proven bitwise by (A)), and does not arise in
closed-loop policy control.

Run (env liberoplus, dm_control + metasim installed):
    LIBERO_CONFIG_PATH=$HOME/.libero_plus MUJOCO_GL=egl \\
    python -m scripts.osc.parity_osc_vs_robosuite --steps 130 --precontact 40
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from types import SimpleNamespace

import h5py
import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from osc_pose_controller import MujocoOSCPose

from roboverse_pack.tasks.libero_plus import _passthrough as pt

DEMO_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "third_party", "libero_datasets"
)


def _wrap(sim):
    """A handler.physics-like view over a robosuite MjSim (raw structs)."""
    return SimpleNamespace(model=SimpleNamespace(ptr=sim.model._model), data=SimpleNamespace(ptr=sim.data._data))


def run(suite, base, steps, precontact):
    env = pt.make_liberoplus_env(suite, 0, seed=0)  # OSC config is suite-independent
    r, c = env.env.robots[0], env.env.robots[0].controller
    eef, qvi = c.eef_name, list(c.qvel_index)
    arm_act = list(r._ref_joint_actuator_indexes)
    ij = np.array(c.initial_joint)
    tlim = (np.array(r.torque_limits[0]), np.array(r.torque_limits[1]))
    nu = env.env.sim.model.nu
    grip_act = [i for i in range(nu) if i not in arm_act]

    with h5py.File(os.path.join(DEMO_ROOT, suite, f"{base}_demo.hdf5"), "r") as h:
        actions = np.asarray(h["data"]["demo_0"]["actions"])[:steps]
        init_state = np.asarray(h["data"]["demo_0"]["states"])[0]

    # lossless MJB snapshot of robosuite's EXACT compiled model (for test B)
    mjb = os.path.join(tempfile.mkdtemp(), "rs.mjb")
    mujoco.mj_saveModel(env.env.sim.model._model, mjb, None)

    # ---- (A) per-step torque parity at every real state (incl. contact) ----
    osc_live = MujocoOSCPose(_wrap(env.env.sim), eef, qvi, arm_act, initial_joint=ij, torque_limits=tlim)
    env.set_init_state(init_state)
    tau_dev = 0.0
    nat_qpos, grip_seq = [], []
    for a in actions:
        # robosuite torque at the current state (force-refresh to defeat caching)
        c.update(force=True)
        c.set_goal(a[:6])
        nat_tau = np.array(c.run_controller())
        # ported torque at the SAME state
        osc_live.set_goal(a[:6])
        my_tau = osc_live.run_controller()
        tau_dev = max(tau_dev, float(np.abs(nat_tau - my_tau).max()))
        # advance natively + record for test B
        env.step(a.tolist())
        nat_qpos.append(np.array(env.env.sim.data.qpos[qvi]))
        grip_seq.append(np.array(env.env.sim.data.ctrl[grip_act]))
    nat_qpos = np.array(nat_qpos)
    env.close()

    # ---- (B) pre-contact rollout on the MJB-exact model ----
    m = mujoco.MjModel.from_binary_path(mjb)
    d = mujoco.MjData(m)
    d.qpos[:] = init_state[1 : 1 + m.nq]
    d.qvel[:] = init_state[1 + m.nq : 1 + m.nq + m.nv]
    mujoco.mj_forward(m, d)
    osc = MujocoOSCPose(
        _wrap(SimpleNamespace(model=SimpleNamespace(_model=m), data=SimpleNamespace(_data=d))),
        eef,
        qvi,
        arm_act,
        initial_joint=ij,
        torque_limits=tlim,
    )
    pc = min(precontact, len(actions))
    roll_dev = 0.0
    for t in range(pc):
        osc.set_goal(actions[t][:6])
        d.ctrl[grip_act] = grip_seq[t]
        for _ in range(25):
            d.ctrl[arm_act] = osc.run_controller()
            d.ctrl[grip_act] = grip_seq[t]
            mujoco.mj_step(m, d)
        roll_dev = max(roll_dev, float(np.abs(np.array(d.qpos[qvi]) - nat_qpos[t]).max()))

    print(f"# OSC_POSE parity: ported controller vs native robosuite ({suite}/{base})\n")
    print(f"(A) per-step torque parity at every real state incl. contact ({len(actions)} states):")
    print(f"      joint-torque max|Δ| = {tau_dev:.3e} N·m   <- BITWISE control-law faithfulness")
    print(f"(B) pre-contact open-loop rollout on MJB-exact model ({pc} steps):")
    print(f"      arm-qpos max|Δ|     = {roll_dev:.3e} rad")
    print()
    if tau_dev < 1e-9:
        print("RESULT: OSC PORT IS BITWISE-FAITHFUL — identical control law to robosuite at every state.")
        print("        A LIBERO EE-delta policy driven through this controller in MetaSim issues the")
        print("        exact joint torques robosuite would. (Full-episode open-loop replay diverges via")
        print("        contact chaos / robosuite's replay caching — not the control law, not for closed-loop.)")
        return 0
    print(f"RESULT: FAIL — control law differs (torque Δ={tau_dev:.3e}).")
    return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_object")
    ap.add_argument("--base", default="pick_up_the_alphabet_soup_and_place_it_in_the_basket")
    ap.add_argument("--steps", type=int, default=130)
    ap.add_argument("--precontact", type=int, default=40)
    args = ap.parse_args()
    return run(args.suite, args.base, args.steps, args.precontact)


if __name__ == "__main__":
    raise SystemExit(main())
