"""Closed-loop policy parity: native SimplerEnv vs RoboVerse passthrough, bitwise.

The strongest 1:1 statement for the integration. The other parity harness
(``parity_simpler_env.py``) drives both envs with a *fixed* seeded action sequence —
which proves the env forwards reset/step identically, but the action stream is the
same by construction. This harness instead runs a **real closed-loop policy** (Octo):
each action depends on the *previous observation*, so if the passthrough perturbed
the observation in any way, the policy would diverge and the trajectories would split.

For each task we run the identical policy (same ``init_rng``) on:
  * native   ``simpler_env.make(task)``
  * wrapped  ``gymnasium.make("SimplerEnv/<task>")``  (the RoboVerse passthrough)
and compare, per step, the full signature:
  * observation image content (sum + sum-of-squares of the policy's input camera)
  * the **action vector the policy emitted** (world_vector ‖ rot_axangle ‖ gripper)
  * reward, terminated, truncated, ``info["success"]``, language instruction.

A correct passthrough yields ``diffs == 0`` — the policy walks the *same* trajectory
on both, step for step, to the same success/failure. Octo is deterministic given
``init_rng`` (``self.rng = jax.random.PRNGKey(init_rng)`` split per step), so the only
source of divergence would be the wrapper itself.

Each env runs in its own subprocess (SAPIEN keeps process-global renderer state).
Octo runs on CPU jax (``JAX_PLATFORMS=cpu``); SAPIEN still renders on the GPU.

Usage::

    JAX_PLATFORMS=cpu VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
      python scripts/parity_policy_simpler_env.py \
        --tasks google_robot_pick_coke_can --seed 8 --steps 25 --json /tmp/policy_parity.json
"""

from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys

# ---------------------------------------------------------------------------
# Worker: run ONE side (native|wrapped) of ONE task in a fresh subprocess,
# driving the real Octo policy closed-loop; emit a base64 JSON trajectory.
# ---------------------------------------------------------------------------

_WORKER = r"""
import base64, json, sys, traceback
import numpy as np

SIDE, TASK, SEED, STEPS, INIT_RNG = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])


def _policy_setup(task):
    return "widowx_bridge" if task.startswith("widowx") else "google_robot"


def main():
    try:
        from simpler_env.policies.octo.octo_model import OctoInference
        from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict

        if SIDE == "native":
            import simpler_env
            env = simpler_env.make(TASK)
        else:
            import gymnasium as gym
            from roboverse_pack.tasks.simpler_env import register_simpler_env_passthrough
            register_simpler_env_passthrough()
            env = gym.make(f"SimplerEnv/{TASK}")

        model = OctoInference(model_type="octo-base", policy_setup=_policy_setup(TASK),
                              action_scale=1.0, init_rng=INIT_RNG)

        obs, info = env.reset(seed=SEED)
        instruction = env.get_language_instruction()
        is_final = env.is_final_subtask()
        image = get_image_from_maniskill2_obs_dict(env, obs)
        model.reset(instruction)

        def osig(o):
            im = np.asarray(get_image_from_maniskill2_obs_dict(env, o)).astype(np.int64)
            return [int(im.sum()), int((im * im).sum())]

        frames = [{"obs": osig(obs), "act": None, "rew": 0.0, "term": False,
                   "trunc": False, "success": bool(info.get("success", False)), "instr": instruction}]
        pred_term = truncated = False
        steps = 0
        while not (pred_term or truncated) and steps < STEPS:
            _, action = model.step(image, instruction)
            pred_term = bool(action["terminate_episode"][0] > 0)
            if pred_term and not is_final:
                pred_term = False
                env.advance_to_next_subtask()
            act_vec = np.concatenate([action["world_vector"], action["rot_axangle"], action["gripper"]])
            obs, reward, done, truncated, info = env.step(act_vec)
            instruction = env.get_language_instruction()
            is_final = env.is_final_subtask()
            image = get_image_from_maniskill2_obs_dict(env, obs)
            steps += 1
            frames.append({
                "obs": osig(obs),
                "act": [round(float(x), 8) for x in np.asarray(act_vec).ravel()],
                "rew": float(reward), "term": bool(done), "trunc": bool(truncated),
                "success": bool(info.get("success", False)), "instr": instruction,
            })
            if done:
                break
        env.close()
        print("TRAJ_B64:" + base64.b64encode(json.dumps({"task": TASK, "side": SIDE, "frames": frames}).encode()).decode())
    except Exception:
        print("ERR_B64:" + base64.b64encode(traceback.format_exc().encode()).decode())


if __name__ == "__main__":
    main()
"""


def _run_side(side, task, seed, steps, init_rng):
    proc = subprocess.run(
        [sys.executable, "-c", _WORKER, side, task, str(seed), str(steps), str(init_rng)],
        check=False,
        capture_output=True,
        text=True,
    )
    for line in proc.stdout.splitlines():
        if line.startswith("TRAJ_B64:"):
            return json.loads(base64.b64decode(line[len("TRAJ_B64:") :]).decode()), None
        if line.startswith("ERR_B64:"):
            return None, base64.b64decode(line[len("ERR_B64:") :]).decode()
    return None, (proc.stderr[-1500:] or "no trajectory emitted")


def _diff_frames(nat, wrp):
    """Return list of per-step diffs between two policy trajectories."""
    diffs = []
    fn, fw = nat["frames"], wrp["frames"]
    if len(fn) != len(fw):
        diffs.append(f"length {len(fn)} vs {len(fw)}")
    for i, (a, b) in enumerate(zip(fn, fw)):
        for k in ("obs", "rew", "term", "trunc", "success", "instr"):
            if a[k] != b[k]:
                diffs.append(f"step{i}.{k}: {a[k]} != {b[k]}")
        # action vector: exact equality (policy is deterministic given identical obs)
        if a["act"] != b["act"]:
            diffs.append(f"step{i}.act")
    return diffs


def check_task(task, *, seed, steps, init_rng):
    nat, en = _run_side("native", task, seed, steps, init_rng)
    if en:
        return {"task": task, "policy_parity_1to1": False, "error": "native: " + en.strip().splitlines()[-1][:150]}
    wrp, ew = _run_side("wrapped", task, seed, steps, init_rng)
    if ew:
        return {"task": task, "policy_parity_1to1": False, "error": "wrapped: " + ew.strip().splitlines()[-1][:150]}
    diffs = _diff_frames(nat, wrp)
    n = min(len(nat["frames"]), len(wrp["frames"]))
    return {
        "task": task,
        "seed": seed,
        "init_rng": init_rng,
        "native_steps": len(nat["frames"]) - 1,
        "wrapped_steps": len(wrp["frames"]) - 1,
        "native_success": nat["frames"][-1]["success"],
        "wrapped_success": wrp["frames"][-1]["success"],
        "compared_steps": n,
        "n_diffs": len(diffs),
        "diffs": diffs[:20],
        "policy_parity_1to1": len(diffs) == 0,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tasks", nargs="*", default=["google_robot_pick_coke_can"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=25)
    p.add_argument("--init-rng", type=int, default=0)
    p.add_argument("--json", default=None)
    args = p.parse_args()

    results = []
    for t in args.tasks:
        r = check_task(t, seed=args.seed, steps=args.steps, init_rng=args.init_rng)
        results.append(r)
        tag = "OK 1:1" if r.get("policy_parity_1to1") else "DIFF/ERR"
        extra = f" steps(nat={r.get('native_steps')},wrp={r.get('wrapped_steps')}) succ(nat={r.get('native_success')},wrp={r.get('wrapped_success')}) diffs={r.get('n_diffs')}"
        if "error" in r:
            extra = f" err={r['error']}"
        print(f"[{tag:8}] {t:42s}{extra}")

    n_ok = sum(1 for r in results if r.get("policy_parity_1to1"))
    print(f"\n{n_ok}/{len(results)} tasks closed-loop policy-parity 1:1")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(
                {
                    "results": results,
                    "n_ok": n_ok,
                    "n_total": len(results),
                    "seed": args.seed,
                    "steps": args.steps,
                    "init_rng": args.init_rng,
                },
                f,
                indent=2,
            )
        print(f"wrote {args.json}")

    # No tasks checked -> nothing proven -> not a PASS. Exit status carries the verdict.
    ok = bool(results) and n_ok == len(results)
    print("RESULT: PASS — closed-loop policy parity is 1:1." if ok else "RESULT: FAIL — policy parity is not 1:1.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
