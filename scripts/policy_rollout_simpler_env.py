"""Run a REAL pretrained policy (Octo) on SimplerEnv tasks through the RoboVerse passthrough.

This is the *useful* video: a generalist manipulation policy (Octo) actually attempting —
and (when it works) completing — the SimplerEnv real-to-sim tasks, driven through
``gymnasium.make("SimplerEnv/<task>")`` (the RoboVerse passthrough), not random actions.

It reuses SimplerEnv's own policy class (``OctoInference``) and the exact eval contract from
``simpler_env/evaluation/maniskill2_evaluator.py``::

    obs, _ = env.reset(...)
    instruction = env.get_language_instruction()
    model.reset(instruction)
    while not done:
        raw, action = model.step(image, instruction)
        obs, reward, done, trunc, info = env.step(
            concat(action["world_vector"], action["rot_axangle"], action["gripper"]))

Each rendered frame is annotated with the task name, the policy, the language instruction,
the step counter, and a live status; the final frames carry a held SUCCESS / FAILURE banner —
so the video is self-explanatory about *which* task is run and *whether* it was solved.

Octo's checkpoint auto-downloads from HuggingFace (``hf://rail-berkeley/octo-small``). The
policy runs on CPU jax (set ``JAX_PLATFORMS=cpu``) to avoid the RTX 5090 / sm_120 vs.
octo-1.0-pinned-jaxlib mismatch; SAPIEN still renders the env on the GPU.

SAPIEN keeps process-global renderer state, so run ONE task per process invocation.

Usage::

    JAX_PLATFORMS=cpu VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
        python scripts/policy_rollout_simpler_env.py \
        --task google_robot_pick_coke_can --out /tmp/octo_videos/google_robot_pick_coke_can.mp4
"""

from __future__ import annotations

import argparse
import json
import os
import textwrap

import numpy as np


def _policy_setup(task_name: str) -> str:
    return "widowx_bridge" if task_name.startswith("widowx") else "google_robot"


def _annotate(
    frame: np.ndarray,
    *,
    task: str,
    policy: str,
    instruction: str,
    step: int,
    status: str,
    status_color: tuple[int, int, int],
    scale: int = 2,
) -> np.ndarray:
    """Overlay task / instruction / step / status onto an RGB uint8 frame (returns a new array).

    Upscales small SimplerEnv frames so the text is legible, draws a translucent top banner
    with the task + step + status, and wraps the language instruction along the bottom.
    """
    import cv2

    img = np.ascontiguousarray(frame.astype(np.uint8))
    if scale != 1:
        img = cv2.resize(img, (img.shape[1] * scale, img.shape[0] * scale), interpolation=cv2.INTER_NEAREST)
    h, w = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    def _band(y0, y1, alpha=0.55):
        sub = img[y0:y1].astype(np.float32)
        img[y0:y1] = (sub * (1 - alpha)).astype(np.uint8)

    # top banner: task + policy + step + status
    _band(0, 58)
    cv2.putText(img, f"{task}", (8, 22), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, f"policy: {policy}   step: {step}", (8, 44), font, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(img, status, (w - 230, 30), font, 0.7, status_color, 2, cv2.LINE_AA)

    # bottom: language instruction (wrapped)
    lines = textwrap.wrap(f'instruction: "{instruction}"', width=max(20, w // 11)) or [""]
    bh = 20 * len(lines) + 12
    _band(h - bh, h)
    for i, ln in enumerate(lines):
        cv2.putText(img, ln, (8, h - bh + 20 + i * 20), font, 0.5, (235, 235, 235), 1, cv2.LINE_AA)
    return img


def run(task_name: str, out_mp4: str, *, model_type: str, max_steps: int, seed: int):
    import gymnasium as gym
    import imageio.v2 as imageio
    from simpler_env.policies.octo.octo_model import OctoInference
    from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict

    from roboverse_pack.tasks.simpler_env import register_simpler_env_passthrough

    register_simpler_env_passthrough()

    policy = f"octo:{model_type}"
    env = gym.make(f"SimplerEnv/{task_name}")
    model = OctoInference(model_type=model_type, policy_setup=_policy_setup(task_name), action_scale=1.0)

    obs, _ = env.reset(seed=seed)
    instruction = env.get_language_instruction()
    is_final = env.is_final_subtask()
    image = get_image_from_maniskill2_obs_dict(env, obs)
    model.reset(instruction)

    frames = [
        _annotate(
            image,
            task=task_name,
            policy=policy,
            instruction=instruction,
            step=0,
            status="RUNNING",
            status_color=(255, 210, 0),
        )
    ]
    pred_term = truncated = done = False
    success = False
    steps = 0
    while not (pred_term or truncated) and steps < max_steps:
        _, action = model.step(image, instruction)
        pred_term = bool(action["terminate_episode"][0] > 0)
        if pred_term and not is_final:
            pred_term = False
            env.advance_to_next_subtask()
        obs, reward, done, truncated, info = env.step(
            np.concatenate([action["world_vector"], action["rot_axangle"], action["gripper"]])
        )
        instruction = env.get_language_instruction()
        is_final = env.is_final_subtask()
        image = get_image_from_maniskill2_obs_dict(env, obs)
        steps += 1
        success = bool(info.get("success", False))
        frames.append(
            _annotate(
                image,
                task=task_name,
                policy=policy,
                instruction=instruction,
                step=steps,
                status="SUCCESS" if success else "RUNNING",
                status_color=(0, 220, 0) if success else (255, 210, 0),
            )
        )
        if done or success:
            break

    env.close()

    # hold a final verdict banner for ~1.5 s so the outcome is unmistakable
    verdict = "SUCCESS" if success else "FAILURE"
    color = (0, 220, 0) if success else (235, 60, 60)
    final = _annotate(
        image, task=task_name, policy=policy, instruction=instruction, step=steps, status=verdict, status_color=color
    )
    frames += [final] * 15

    os.makedirs(os.path.dirname(os.path.abspath(out_mp4)), exist_ok=True)
    imageio.mimsave(out_mp4, frames, fps=10)

    result = {
        "task": task_name,
        "policy": policy,
        "seed": seed,
        "steps": steps,
        "success": success,
        "final_instruction": instruction,
        "video": out_mp4,
    }
    print(json.dumps(result))
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True)
    p.add_argument("--out", required=True, help="output .mp4 path")
    p.add_argument("--model-type", default="octo-small", choices=["octo-small", "octo-base"])
    p.add_argument("--max-steps", type=int, default=120)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json", default=None, help="append the result dict to this JSON-lines file")
    args = p.parse_args()

    res = run(args.task, args.out, model_type=args.model_type, max_steps=args.max_steps, seed=args.seed)
    if args.json:
        with open(args.json, "a") as f:
            f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
