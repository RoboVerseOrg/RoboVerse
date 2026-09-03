"""Tests for the SimplerEnv passthrough registration + 1:1 parity.

The registration tests run anywhere (they must not raise when SimplerEnv is
absent). The reset/step and bitwise-parity tests are skipped unless the
``simpler_env`` package is importable (dedicated conda env).
"""

from __future__ import annotations

import importlib.util

import pytest

simpler_available = importlib.util.find_spec("simpler_env") is not None
octo_available = importlib.util.find_spec("octo") is not None


@pytest.mark.general
def test_register_simpler_env_passthrough_no_raise():
    """Registration must not raise even when simpler_env is absent."""
    from roboverse_pack.tasks.simpler_env import register_simpler_env_passthrough

    ids = register_simpler_env_passthrough()
    assert isinstance(ids, list)


@pytest.mark.general
def test_simpler_env_prefix_registered():
    """The SimplerEnv prefix is tracked by the passthrough index."""
    from roboverse_pack.tasks._passthrough_index import PASSTHROUGH_PREFIXES

    assert "SimplerEnv/" in PASSTHROUGH_PREFIXES


@pytest.mark.skipif(not simpler_available, reason="simpler_env not installed")
def test_simpler_env_ids_match_upstream():
    """Every registered id must mirror an upstream ``simpler_env.ENVIRONMENTS`` task."""
    import simpler_env

    from roboverse_pack.tasks.simpler_env import register_simpler_env_passthrough

    ids = register_simpler_env_passthrough(prefix="SimplerEnvPassthrough/")
    assert len(ids) == len(simpler_env.ENVIRONMENTS)
    assert {i.removeprefix("SimplerEnvPassthrough/") for i in ids} == set(simpler_env.ENVIRONMENTS)


@pytest.mark.skipif(not simpler_available, reason="simpler_env not installed")
def test_make_reset_step():
    """Smoke-test a single env construction + reset + step when available."""
    import gymnasium as gym

    from roboverse_pack.tasks.simpler_env import register_simpler_env_passthrough

    register_simpler_env_passthrough(prefix="SimplerEnvPassthrough/")
    env = gym.make("SimplerEnvPassthrough/google_robot_pick_coke_can")
    obs, info = env.reset(seed=0)
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    env.close()


@pytest.mark.skipif(not simpler_available, reason="simpler_env not installed")
def test_passthrough_matches_native_bitwise():
    """The passthrough must reproduce native ``simpler_env.make`` bitwise.

    Same task, same seed, same deterministic action sequence -> identical obs,
    reward, termination flags and ``info["success"]``.

    SimplerEnv (ManiSkill2-real2sim + SAPIEN) keeps process-global renderer/engine
    state, so two SAPIEN scenes alive at once perturb each other's success
    evaluation. We therefore roll the native env out **fully and close it before**
    creating the wrapped env — one scene alive at a time. (The cross-task parity
    sweep in ``scripts/parity_simpler_env.py`` isolates each task in its own
    subprocess for the same reason.)
    """
    import gc

    import gymnasium as gym
    import numpy as np
    import simpler_env

    from roboverse_pack.tasks.simpler_env import register_simpler_env_passthrough

    register_simpler_env_passthrough(prefix="SimplerEnvPassthrough/")
    task_name = "google_robot_pick_coke_can"
    seed, steps = 0, 5

    def leaves(o, p=""):
        d = {}
        if isinstance(o, dict):
            for k, v in o.items():
                d.update(leaves(v, f"{p}{k}/"))
        else:
            d[p] = np.asarray(o)
        return d

    def rollout(env, acts):
        obs, info = env.reset(seed=seed)
        out = [(leaves(obs), 0.0, False, False, bool(info.get("success", False)))]
        for a in acts:
            o, rw, tm, tr, inf = env.step(a)
            out.append((leaves(o), float(rw), bool(tm), bool(tr), bool(inf.get("success", False))))
        return out

    # native alone -> close -> wrapped alone (one SAPIEN scene at a time)
    native = simpler_env.make(task_name)
    rng = np.random.RandomState(seed)
    acts = [rng.uniform(native.action_space.low, native.action_space.high) for _ in range(steps)]
    fn = rollout(native, acts)
    native.close()
    del native
    gc.collect()

    wrapped = gym.make(f"SimplerEnvPassthrough/{task_name}")
    fw = rollout(wrapped, acts)
    wrapped.close()

    for (on, rn, tn, un, sn), (ow, rw, tw, uw, sw) in zip(fn, fw, strict=False):
        assert set(on) == set(ow)
        for k in on:
            assert np.array_equal(on[k], ow[k]), f"obs mismatch at {k}"
        assert rn == rw
        assert tn == tw and un == uw
        assert sn == sw


@pytest.mark.skipif(not simpler_available, reason="simpler_env not installed")
def test_trajectory_reproducible_and_deterministic():
    """A trajectory must regenerate bit-for-bit from ``(task, seed, actions)``.

    Records a seeded rollout, then (a) replays the *recorded action sequence* and
    (b) re-records the same ``(task, seed)`` from scratch — both must match the
    original exactly (image content hashes, reward, flags, success). This is the
    data/trajectory reproducibility contract for the passthrough.
    """
    import gymnasium as gym
    import numpy as np

    from roboverse_pack.tasks.simpler_env import register_simpler_env_passthrough

    register_simpler_env_passthrough(prefix="SimplerEnvPassthrough/")
    task_name, seed, steps = "google_robot_pick_coke_can", 0, 5
    cam = "overhead_camera"

    def rollout(actions=None):
        env = gym.make(f"SimplerEnvPassthrough/{task_name}")
        if actions is None:
            rng = np.random.RandomState(seed)
            sp = env.action_space
            actions = np.stack([rng.uniform(sp.low, sp.high).astype(sp.dtype) for _ in range(steps)])

        def sig(o, r, t, tr, inf):
            im = np.asarray(o["image"][cam]["rgb"]).astype(np.int64)
            return (int(im.sum()), int((im * im).sum()), float(r), bool(t), bool(tr), bool(inf.get("success", False)))

        obs, info = env.reset(seed=seed)
        frames = [sig(obs, 0.0, False, False, info)]
        for a in actions:
            o, r, t, tr, inf = env.step(a)
            frames.append(sig(o, r, t, tr, inf))
        env.close()
        return actions, frames

    acts, base = rollout()
    _, replayed = rollout(actions=acts)
    assert replayed == base, "replay of recorded actions diverged"
    acts2, redo = rollout()
    assert np.array_equal(acts, acts2), "seed did not regenerate the same action sequence"
    assert redo == base, "re-recording the same (task, seed) diverged"


@pytest.mark.skipif(not (simpler_available and octo_available), reason="simpler_env / octo not installed")
def test_closed_loop_policy_parity_native_vs_passthrough():
    """A real closed-loop Octo policy must walk an identical trajectory on native vs passthrough.

    Stronger than the fixed-action parity test: each action depends on the previous
    observation, so any perturbation the wrapper introduced would make the policy
    diverge. Same ``init_rng`` + same seed -> identical per-step obs content, emitted
    action vector, reward, flags and success on ``simpler_env.make`` and
    ``gym.make("SimplerEnv/<task>")``. One SAPIEN scene per process, so the native
    env is fully rolled out and closed before the wrapped one is created.
    """
    import gc

    import gymnasium as gym
    import numpy as np
    import simpler_env
    from simpler_env.policies.octo.octo_model import OctoInference
    from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict

    from roboverse_pack.tasks.simpler_env import register_simpler_env_passthrough

    register_simpler_env_passthrough(prefix="SimplerEnvPassthrough/")
    task_name, seed, steps = "google_robot_pick_coke_can", 0, 4

    def rollout(env):
        model = OctoInference(model_type="octo-small", policy_setup="google_robot", action_scale=1.0, init_rng=0)
        obs, info = env.reset(seed=seed)
        instr = env.get_language_instruction()
        image = get_image_from_maniskill2_obs_dict(env, obs)
        model.reset(instr)

        def osig(o):
            im = np.asarray(get_image_from_maniskill2_obs_dict(env, o)).astype(np.int64)
            return (int(im.sum()), int((im * im).sum()))

        frames = [(osig(obs), None, 0.0, False, bool(info.get("success", False)))]
        for _ in range(steps):
            _, action = model.step(image, instr)
            act = np.concatenate([action["world_vector"], action["rot_axangle"], action["gripper"]])
            obs, r, done, trunc, info = env.step(act)
            instr = env.get_language_instruction()
            image = get_image_from_maniskill2_obs_dict(env, obs)
            frames.append((
                osig(obs),
                tuple(np.round(act, 8)),
                float(r),
                bool(done) or bool(trunc),
                bool(info.get("success", False)),
            ))
            if done or trunc:
                break
        env.close()
        return frames

    native = simpler_env.make(task_name)
    fn = rollout(native)
    del native
    gc.collect()

    wrapped = gym.make(f"SimplerEnvPassthrough/{task_name}")
    fw = rollout(wrapped)

    assert len(fn) == len(fw), f"trajectory length differs: {len(fn)} vs {len(fw)}"
    for i, (a, b) in enumerate(zip(fn, fw, strict=False)):
        assert a == b, f"closed-loop divergence at step {i}: {a} != {b}"
