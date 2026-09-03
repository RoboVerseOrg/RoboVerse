"""Prove the native SimplerEnv registration works with the cloned library DELETED.

Installs a meta-path blocker so ``mani_skill2_real2sim`` / ``simpler_env`` are unimportable
(simulating ``rm -rf`` of the clone), then:
  --mode registry : import roboverse_pack, assert all 25 ``SimplerEnv/<task>`` ids are registered
                    and zero upstream modules are in sys.modules.
  --mode run      : for one task, gym.make + reset + 2 steps (own subprocess; SAPIEN is global),
                    asserting zero upstream import.
  (default driver runs the registry check in-process, then each of the 25 tasks in a subprocess.)

Run:  JAX_PLATFORMS=cpu python scripts/verify_native_registration.py
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys


class _BlockUpstream:
    def find_spec(self, name, path=None, target=None):
        if name == "simpler_env" or name == "mani_skill2_real2sim" or name.startswith("mani_skill2_real2sim."):
            raise ModuleNotFoundError(f"blocked (simulating deleted clone): {name}")
        return None


def _block():
    sys.meta_path.insert(0, _BlockUpstream())


def _assert_clean():
    bad = [m for m in sys.modules if m == "simpler_env" or m.startswith("mani_skill2_real2sim")]
    assert not bad, f"upstream leaked: {bad}"


def run_registry():
    _block()
    import gymnasium as gym

    import roboverse_pack.tasks.simpler_env  # noqa: F401  triggers registration
    from roboverse_pack.tasks.simpler_env._metasim.registry import ENVIRONMENTS

    ids = {f"SimplerEnv/{t}" for t in ENVIRONMENTS}
    have = {s for s in gym.envs.registry if s.startswith("SimplerEnv/")}
    missing = ids - have
    _assert_clean()
    assert not missing, f"missing native ids: {missing}"
    print(f"REGISTRY_OK {len(ids)} native ids registered with clone blocked; zero upstream imports")


def run_one(task):
    _block()
    import gymnasium as gym
    import numpy as np

    import roboverse_pack.tasks.simpler_env  # noqa: F401

    env = gym.make(f"SimplerEnv/{task}")
    obs, info = env.reset(seed=0)
    instr = env.get_language_instruction()
    for _ in range(2):
        a = np.zeros(7, dtype=np.float32)
        obs, r, term, trunc, info = env.step(a)
    img = obs["image"]
    cam = next(iter(img))
    _assert_clean()
    print(f"RUN_OK {task}: instr={instr!r} img={cam}{img[cam]['rgb'].shape} success={info.get('success')}")
    env.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["registry", "run", "all"], default="all")
    ap.add_argument("--task", default=None)
    a = ap.parse_args()
    if a.mode == "registry":
        return run_registry()
    if a.mode == "run":
        return run_one(a.task)
    # driver
    env = dict(os.environ, JAX_PLATFORMS="cpu")
    r = subprocess.run([sys.executable, __file__, "--mode", "registry"], env=env, check=False)
    if r.returncode != 0:
        sys.exit(1)
    from roboverse_pack.tasks.simpler_env._metasim.registry import ENVIRONMENTS  # safe (will block again in children)

    ok = 0
    for t in ENVIRONMENTS:
        rc = subprocess.run([sys.executable, __file__, "--mode", "run", "--task", t], env=env, check=False).returncode
        ok += rc == 0
        if rc != 0:
            print(f"FAIL {t}")
    print(
        f"=== native registration: registry OK + {ok}/{len(ENVIRONMENTS)} tasks make/reset/step with clone deleted ==="
    )
    # A task that failed to make/reset/step is a verification failure; say so in the
    # exit status, or a caller cannot tell this run from a clean one.
    sys.exit(0 if ENVIRONMENTS and ok == len(ENVIRONMENTS) else 1)


if __name__ == "__main__":
    main()
