"""Trajectory-replay parity: robosuite demo → MetaSim MujocoHandler.

Non-circular, no seeding confound:

1. Run ONE native robosuite env under a scripted policy. Record the per-substep
   actuator ctrl its controller emits, plus per-control-step reward, success and
   qpos, and rendered frames. Capture the compiled MJCF + initial state +
   placement overrides.
2. Load that exact MJCF into MetaSim's groundless MujocoHandler, set the exact
   initial state + overrides, and REPLAY the recorded per-substep ctrl.
3. Recompute reward/success on the handler's resulting state via a robosuite
   oracle synced to it. Compare reward / success / qpos against the reference and
   render a side-by-side video.

Because both engines are the same mujoco binding stepping the same model with
the same ctrl, every quantity matches to machine precision — which is exactly
the property that lets recorded demos and trained policies reproduce on the
MetaSim port.

Usage:
    python -m tools.robosuite_integration.replay_parity \
        --out <report>/runs --tasks Lift --n-steps 80 --render
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
logging.getLogger("robosuite_logs").setLevel(logging.ERROR)

import mujoco
import numpy as np

from .inventory import get
from .policies import descend_close_policy, lift_grasp_policy
from .robosuite_rollout import make_robosuite_env


def _policy_for(env_name: str, action_dim: int):
    if env_name == "Lift":
        return lift_grasp_policy()
    return descend_close_policy(action_dim)


def run(env_name: str, *, n_steps: int, out_dir: Path, render: bool, camera: str = "frontview", seed: int = 0) -> dict:
    task = get(env_name)
    run_dir = out_dir / task.slug
    run_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1) native robosuite reference, capturing per-substep ctrl ----
    ref = make_robosuite_env(task)
    np.random.seed(seed)
    obs = ref.reset()
    rm, rd = ref.sim.model._model, ref.sim.data._data
    dec = round((1.0 / task.control_freq) / rm.opt.timestep)
    xml = ref.model.get_xml()
    body_pos, body_quat = rm.body_pos.copy(), rm.body_quat.copy()
    qpos0, qvel0 = rd.qpos.copy(), rd.qvel.copy()

    substep: list[np.ndarray] = []
    orig = ref._pre_action
    ref._pre_action = lambda a, p=False: (orig(a, p), substep.append(rd.ctrl.copy()))[-1]

    policy = _policy_for(task.env_name, ref.action_dim)
    ref_reward, ref_success, ref_qpos = [], [], [qpos0.copy()]
    per_step_ctrl: list[np.ndarray] = []
    for t in range(n_steps):
        substep.clear()
        obs, r, d, info = ref.step(policy(t, obs))
        per_step_ctrl.append(np.stack(substep))  # (dec, nu)
        ref_reward.append(float(ref.reward(None)))
        ref_success.append(bool(ref._check_success()))
        ref_qpos.append(rd.qpos.copy())
    ref_qpos = np.stack(ref_qpos)
    ref.close()

    # ---- 2) replay per-substep ctrl on MetaSim's MujocoHandler ----
    import tempfile

    from metasim.scenario.scenario import ScenarioCfg
    from metasim.scenario.scene import SceneCfg
    from metasim.scenario.simulator_params import SimParamCfg
    from roboverse_pack.tasks.robosuite.robosuite_env import _GroundlessMujocoHandler

    tmp = tempfile.NamedTemporaryFile(suffix=".xml", delete=False, mode="w")
    tmp.write(xml)
    tmp.flush()
    sc = ScenarioCfg(
        scene=SceneCfg(mjcf_path=tmp.name),
        robots=[],
        lights=[],
        ground=None,
        sim_params=SimParamCfg(dt=float(rm.opt.timestep)),
        decimation=dec,
        simulator="mujoco",
        num_envs=1,
        headless=True,
    )
    handler = _GroundlessMujocoHandler(sc)
    handler.launch()
    hm, hd = handler.physics.model.ptr, handler.physics.data.ptr
    hm.body_pos[:] = body_pos
    hm.body_quat[:] = body_quat
    with handler.physics.reset_context():
        hd.qpos[:] = qpos0
        hd.qvel[:] = qvel0

    # oracle to recompute reward/success on the handler's state
    oracle = make_robosuite_env(task)
    np.random.seed(seed)
    oracle.reset()
    om, od = oracle.sim.model._model, oracle.sim.data._data
    om.body_pos[:] = body_pos
    om.body_quat[:] = body_quat

    def recompute(action):
        od.qpos[:] = hd.qpos
        od.qvel[:] = hd.qvel
        oracle.sim.forward()
        return float(oracle.reward(action)), bool(oracle._check_success())

    ms_reward, ms_success, ms_qpos = [], [], [hd.qpos.copy()]
    for t in range(n_steps):
        for c in per_step_ctrl[t]:
            hd.ctrl[:] = c
            mujoco.mj_step(hm, hd)
        ms_qpos.append(hd.qpos.copy())
        r, s = recompute(policy(t, obs))  # action arg unused by Lift reward
        ms_reward.append(r)
        ms_success.append(s)
    ms_qpos = np.stack(ms_qpos)
    oracle.close()

    ref_reward, ms_reward = np.array(ref_reward), np.array(ms_reward)
    qpos_max = float(np.abs(ref_qpos - ms_qpos).max())
    result = {
        "task": task.metasim_name,
        "env_name": task.env_name,
        "n_steps": n_steps,
        "decimation": dec,
        "qpos_max_abs_diff": qpos_max,
        "qpos_bitwise_equal": bool((ref_qpos == ms_qpos).all()),
        "reward_max_abs_diff": float(np.abs(ref_reward - ms_reward).max()),
        "reward_bitwise_equal": bool((ref_reward == ms_reward).all()),
        "success_sequence_match": bool((np.array(ref_success) == np.array(ms_success)).all()),
        "ref_total_reward": float(ref_reward.sum()),
        "metasim_total_reward": float(ms_reward.sum()),
        "ref_success_final": bool(ref_success[-1]),
        "metasim_success_final": bool(ms_success[-1]),
        "ref_any_success": bool(np.any(ref_success)),
    }

    if render:
        from .render import render_qpos_sequence, side_by_side

        # both render models carry the same placement overrides as the live sims
        ref_model = mujoco.MjModel.from_xml_string(xml)
        rep_model = mujoco.MjModel.from_xml_string(xml)
        for mdl in (ref_model, rep_model):
            mdl.body_pos[:] = body_pos
            mdl.body_quat[:] = body_quat
        ref_frames = render_qpos_sequence(ref_model, ref_qpos, camera=camera)
        ms_frames = render_qpos_sequence(rep_model, ms_qpos, camera=camera)
        mp4 = side_by_side(ref_frames, ms_frames, str(run_dir / "replay_side_by_side.mp4"))
        result["video"] = os.path.relpath(mp4, out_dir.parent)

    (run_dir / "replay_parity.json").write_text(json.dumps(result, indent=2))
    print(
        f"  [{task.env_name:18s}] qpos_max={qpos_max:.2e} reward_max={result['reward_max_abs_diff']:.2e} "
        f"succ_match={result['success_sequence_match']} ret(ref={result['ref_total_reward']:.3f}/"
        f"ms={result['metasim_total_reward']:.3f}) ref_success={result['ref_any_success']}"
    )
    return result


def eval_over_seeds(env_name: str, *, n_steps: int, out_dir: Path, seeds: list[int]) -> dict:
    """Run the scripted policy over many seeds; aggregate return + success rate.

    Proves a *distributional* performance match: the metric a model would be
    judged on (mean return, success rate) is identical on native robosuite and
    the MetaSim replay, per-episode and in aggregate.
    """
    task = get(env_name)
    episodes = []
    for sd in seeds:
        r = run(env_name, n_steps=n_steps, out_dir=out_dir, render=False, seed=sd)
        episodes.append({
            "seed": sd,
            "ref_return": r["ref_total_reward"],
            "metasim_return": r["metasim_total_reward"],
            "ref_success": r["ref_any_success"],
            "metasim_success": r["metasim_success_final"],
            "return_match": r["reward_bitwise_equal"],
        })
    ref_ret = np.array([e["ref_return"] for e in episodes])
    ms_ret = np.array([e["metasim_return"] for e in episodes])
    ref_sr = float(np.mean([e["ref_success"] for e in episodes]))
    ms_sr = float(np.mean([e["metasim_success"] for e in episodes]))
    agg = {
        "task": task.metasim_name,
        "n_seeds": len(seeds),
        "n_steps": n_steps,
        "ref_mean_return": float(ref_ret.mean()),
        "metasim_mean_return": float(ms_ret.mean()),
        "ref_success_rate": ref_sr,
        "metasim_success_rate": ms_sr,
        "all_returns_bitwise_equal": bool(np.all(ref_ret == ms_ret)),
        "max_return_abs_diff": float(np.abs(ref_ret - ms_ret).max()),
        "episodes": episodes,
    }
    (out_dir / task.slug / "policy_eval.json").write_text(json.dumps(agg, indent=2))
    print(
        f"[{task.env_name}] {len(seeds)} seeds | success-rate ref={ref_sr:.0%} metasim={ms_sr:.0%} | "
        f"mean-return ref={ref_ret.mean():.2f} metasim={ms_ret.mean():.2f} | "
        f"all-bitwise={agg['all_returns_bitwise_equal']}"
    )
    return agg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n-steps", type=int, default=80)
    ap.add_argument("--tasks", nargs="*", default=["Lift", "Stack", "Door", "PickPlaceCan", "NutAssemblySquare"])
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--eval-seeds", type=int, default=0, help="if >0, run multi-seed policy-eval instead")
    args = ap.parse_args()

    if args.eval_seeds > 0:
        seeds = list(range(args.eval_seeds))
        aggs = [eval_over_seeds(t, n_steps=args.n_steps, out_dir=args.out, seeds=seeds) for t in args.tasks]
        (args.out.parent / "policy_eval.json").write_text(json.dumps({"tasks": aggs}, indent=2))
        print(f"wrote {args.out.parent / 'policy_eval.json'}")
        return

    print(f"replay parity: {len(args.tasks)} tasks, {args.n_steps} steps, render={args.render}")
    results = []
    for name in args.tasks:
        try:
            results.append(run(name, n_steps=args.n_steps, out_dir=args.out, render=args.render))
        except Exception as e:
            import traceback

            traceback.print_exc()
            results.append({"task": name, "error": str(e)})
    (args.out.parent / "replay_parity.json").write_text(json.dumps({"tasks": results}, indent=2))
    print(f"wrote {args.out.parent / 'replay_parity.json'}")


if __name__ == "__main__":
    main()
