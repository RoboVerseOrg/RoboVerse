"""Roll out ManiSkill tasks and render artifacts.

For each task in `inventory.ALL_TASKS`, run a scripted (zero-action +
optional sinusoid) episode with a fixed seed and capture state, reward,
success flags, and rgb frames. Save:
- `runs/<task>/maniskill.npz` — full numeric trace
- `runs/<task>/maniskill.mp4` — rendered episode
- `runs/<task>/summary.json` — sub-set of the rollout summary

The MetaSim/sapien parity column is intentionally deferred — see report
for the gap analysis (`docs/source/dataset_benchmark/integrations/
maniskill.md` once landed).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from . import inventory as inv
from .maniskill_rollout import rollout_maniskill, save_rollout


def _maybe_write_mp4(out_path: Path, frames: list, fps: float = 20.0) -> str | None:
    try:
        import imageio
    except Exception as e:
        print(f"[maniskill_sweep] imageio missing: {e}")
        return None
    if not frames:
        return None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.stack(frames)
    if arr.ndim == 4 and arr.shape[-1] in (1, 3, 4):
        pass
    elif arr.ndim == 5 and arr.shape[1] == 1:  # (T, 1, H, W, C) — batched render
        arr = arr[:, 0]
    try:
        imageio.mimwrite(out_path, arr, fps=fps, codec="libx264", quality=7)
    except Exception:
        gif = out_path.with_suffix(".gif")
        imageio.mimwrite(gif, arr, fps=fps)
        out_path = gif
    return str(out_path)


def run_all(reports_dir: Path, n_steps: int = 50, seed: int = 0) -> dict:
    summary: dict = {"runs": []}
    for task in inv.ALL_TASKS:
        run_dir = reports_dir / "runs" / task.metasim_name
        run_dir.mkdir(parents=True, exist_ok=True)
        try:
            r = rollout_maniskill(
                task.gym_id,
                n_steps=n_steps,
                seed=seed,
                control_mode=task.control_mode,
                obs_mode=task.obs_mode,
                render=True,
                render_size=(320, 240),
            )
        except Exception as e:
            print(f"[maniskill_sweep] {task.gym_id}: rollout failed — {e}")
            summary["runs"].append({"task": task.metasim_name, "error": repr(e)})
            continue

        save_rollout(run_dir / "maniskill.npz", r)
        mp4_path = _maybe_write_mp4(run_dir / "maniskill.mp4", r.frames, fps=r.fps)

        s = r.summary()
        s["mp4"] = mp4_path
        s["task"] = task.metasim_name
        summary["runs"].append(s)
        (run_dir / "summary.json").write_text(json.dumps(s, indent=2))
        print(
            f"[maniskill_sweep] {task.gym_id:18s}  T={s['T']:>3}  "
            f"Σr={s['total_reward']:7.3f}  success_any={s['any_success']}  mp4={mp4_path}"
        )

    summary_path = reports_dir / "maniskill_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[maniskill_sweep] wrote {summary_path}")
    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="/home/ghr/projects/RoboVerse/reports/maniskill_integration")
    p.add_argument("--n-steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    run_all(Path(args.out), n_steps=args.n_steps, seed=args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
