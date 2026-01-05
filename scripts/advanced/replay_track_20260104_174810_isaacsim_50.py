from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import imageio as iio
import numpy as np
import torch
from loguru import logger as log
from torchvision.utils import make_grid

# Ensure we import RoboVerse's own `metasim` (avoid picking up other repos' metasim on PYTHONPATH).
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from metasim.scenario.cameras import PinholeCameraCfg
from metasim.task.registry import get_task_class
from metasim.utils.demo_util import get_traj
from metasim.utils.state import TensorState

# ---------------- HARD-CODED CONFIG ----------------
TASK_NAME = "pick_place.track_il"
SIM = "isaacsim"
RENDERER = "isaacsim"
NUM_ENVS = 1
HEADLESS = True

# NOTE: user said `val_trajs/...` but that directory doesn't exist in this repo;
# the file exists under `eval_trajs/`.
TRAJ_FILEPATH = "eval_trajs/track_franka_eval_settle_20260104_174810_v2.pkl"

DEMO_START_IDX = 0
DEMO_COUNT = 50

# Output: each demo gets its own mp4 to avoid OOM.
VIDEO_ROOT = "test_output/track_replay_20260104_174810.mp4"
FPS = 30
DEBUG_CHECK_TARGET_MATCH = True
DEBUG_CHECK_STEPS = 5
DEBUG_COMPARE_STATES = True
DEBUG_COMPARE_DEMO_IDX = 0
DEBUG_COMPARE_EVERY_DEMO = True


class ObsSaver:
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.images: list[np.ndarray] = []

    def add(self, state: TensorState) -> None:
        rgb_data = next(iter(state.cameras.values())).rgb  # (N,H,W,C)
        image = make_grid(rgb_data.permute(0, 3, 1, 2) / 255, nrow=int(rgb_data.shape[0] ** 0.5))  # (C,H,W)
        image = image.cpu().numpy().transpose(1, 2, 0)
        image = (image * 255).astype(np.uint8)
        self.images.append(image)

    def save(self) -> None:
        if not self.images:
            return
        os.makedirs(os.path.dirname(self.video_path), exist_ok=True)
        log.info(f"Saving video of {len(self.images)} frames to {self.video_path}")
        iio.mimsave(self.video_path, self.images, fps=FPS)


def main() -> None:
    assert NUM_ENVS == 1, "This hardcoded replay script assumes NUM_ENVS=1 for IsaacSim stability."
    assert os.path.exists(TRAJ_FILEPATH), f"Trajectory file not found: {TRAJ_FILEPATH}"

    task_cls = get_task_class(TASK_NAME)
    camera = PinholeCameraCfg(pos=(1.5, -1.5, 1.5), look_at=(0.0, 0.0, 0.0))

    scenario = task_cls.scenario.update(
        robots=["franka"],
        cameras=[camera],
        simulator=SIM,
        renderer=RENDERER,
        num_envs=NUM_ENVS,
        headless=bool(HEADLESS),
    )
    # IMPORTANT:
    # For replay-vs-eval consistency, do NOT override decimation / actuator gains here.

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tic = time.time()
    env = task_cls(scenario, device=device)
    log.info(f"Env launch time: {time.time() - tic:.2f}s")

    # Load trajectory (v2 -> v3 conversion happens inside get_traj)
    init_states, all_actions, all_states = get_traj(TRAJ_FILEPATH, scenario.robots[0], env.handler)
    assert all_actions is not None and len(all_actions) > 0, "No actions found in trajectory."

    demo_end = min(DEMO_START_IDX + DEMO_COUNT, len(all_actions))
    for demo_i in range(DEMO_START_IDX, demo_end):
        log.info(f"[replay] demo={demo_i}/{len(all_actions) - 1}")

        # Per-demo mp4
        root, ext = os.path.splitext(VIDEO_ROOT)
        if not ext:
            ext = ".mp4"
        video_path = f"{root}_demo{demo_i:04d}{ext}"
        saver = ObsSaver(video_path=video_path)

        # Reset to that demo init state
        env.reset(states=[init_states[demo_i]], env_ids=[0])
        obs = env.handler.get_states()
        saver.add(obs)

        # Optionally compare replayed states to saved per-step states.
        # If DEBUG_COMPARE_EVERY_DEMO=True, we compare for all demos (lightweight: only a few steps printed).
        compare_this = (
            bool(DEBUG_COMPARE_STATES)
            and (all_states is not None)
            and (bool(DEBUG_COMPARE_EVERY_DEMO) or (demo_i == int(DEBUG_COMPARE_DEMO_IDX)))
        )
        saved_states = all_states[demo_i] if (compare_this and all_states is not None) else None
        if compare_this:
            log.info(
                f"[debug] state-compare enabled for demo={demo_i}: saved_states_len={len(saved_states) if saved_states is not None else None}"
            )
            # Compare reset state vs init_state (object pose)
            try:
                st0 = env.handler.get_states(mode="tensor")
                bbq0 = st0.objects["bbq_sauce"].root_state[0, 0:3].detach().cpu().numpy()
                bbq0_ref = init_states[demo_i]["objects"]["bbq_sauce"]["pos"].detach().cpu().numpy()
                err0 = float(np.linalg.norm(bbq0 - bbq0_ref))
                log.info(f"[debug] demo={demo_i} reset bbq_pos_err_vs_init={err0:.4f}m")
            except Exception as e:
                log.warning(f"[debug] reset state compare failed: {e}")

        # Replay all steps
        steps = all_actions[demo_i]
        for step_idx, step_action in enumerate(steps):
            desired = None
            if DEBUG_CHECK_TARGET_MATCH and step_idx < int(DEBUG_CHECK_STEPS):
                desired = step_action[scenario.robots[0].name]["dof_pos_target"]
            _obs, _reward, _terminated, time_out, _info = env.step([step_action])
            if desired is not None:
                st = env.handler.get_states(mode="tensor")
                got = st.robots[scenario.robots[0].name].joint_pos_target[0].detach().cpu().numpy()
                jn = env.handler.get_joint_names(scenario.robots[0].name, sort=True)
                ref = np.array([float(desired[n]) for n in jn], dtype=np.float32)
                err = float(np.max(np.abs(got - ref)))
                log.info(f"[debug] demo={demo_i} step={step_idx} max|target_err|={err:.6f}")

            if saved_states is not None and step_idx < len(saved_states):
                # Compare a few key quantities vs saved state.
                try:
                    st = env.handler.get_states(mode="tensor")
                    # bbq pos/rot
                    bbq_pos = st.objects["bbq_sauce"].root_state[0, 0:3].detach().cpu().numpy()
                    bbq_pos_ref = saved_states[step_idx]["objects"]["bbq_sauce"]["pos"].detach().cpu().numpy()
                    bbq_pos_err = float(np.linalg.norm(bbq_pos - bbq_pos_ref))

                    bbq_quat = st.objects["bbq_sauce"].root_state[0, 3:7].detach().cpu().numpy()
                    bbq_quat_ref = saved_states[step_idx]["objects"]["bbq_sauce"]["rot"].detach().cpu().numpy()
                    bbq_quat_err = float(np.linalg.norm(bbq_quat - bbq_quat_ref))

                    # basket pos/rot
                    basket_pos = st.objects["basket"].root_state[0, 0:3].detach().cpu().numpy()
                    basket_pos_ref = saved_states[step_idx]["objects"]["basket"]["pos"].detach().cpu().numpy()
                    basket_pos_err = float(np.linalg.norm(basket_pos - basket_pos_ref))

                    basket_quat = st.objects["basket"].root_state[0, 3:7].detach().cpu().numpy()
                    basket_quat_ref = saved_states[step_idx]["objects"]["basket"]["rot"].detach().cpu().numpy()
                    basket_quat_err = float(np.linalg.norm(basket_quat - basket_quat_ref))

                    # robot joint_pos (measured) vs saved state's dof_pos (measured)
                    joint_names = env.handler.get_joint_names(scenario.robots[0].name, sort=True)
                    q = st.robots[scenario.robots[0].name].joint_pos[0].detach().cpu().numpy()
                    q_ref_dict = saved_states[step_idx]["robots"][scenario.robots[0].name]["dof_pos"]
                    q_ref = np.array([float(q_ref_dict[n]) for n in joint_names], dtype=np.float32)
                    q_err = float(np.linalg.norm(q - q_ref))

                    if step_idx in (0, 79, 80, 109, 110, 111, 150, 200):
                        log.info(
                            f"[debug] demo={demo_i} step={step_idx} "
                            f"bbq_pos_err={bbq_pos_err:.4f}m bbq_quat_err={bbq_quat_err:.4e} "
                            f"basket_pos_err={basket_pos_err:.4f}m basket_quat_err={basket_quat_err:.4e} "
                            f"franka_q_err={q_err:.4f}"
                        )
                except Exception as e:
                    log.warning(f"[debug] state compare failed at step={step_idx}: {e}")
            saver.add(_obs)
            if bool(time_out[0].item()):
                break

        saver.save()

    env.close()
    if SIM == "isaacsim":
        try:
            env.handler.simulation_app.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
