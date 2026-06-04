"""Build report assets from the all-130 native sweep: per-suite montages + GIFs.

Reads ``outputs/native_sweep/{results.json,thumbs,videos}`` and writes, under
``--out``:
  * ``montage_<suite>.png`` — a contact sheet of every task's mid-trajectory
    side-by-side (libero | MetaSim-native), downscaled + PNG-optimized.
  * ``hl_<suite>__<task>.gif`` — a few right-side-up side-by-side highlight GIFs
    (palette-optimized, shrunk to stay < ``--max_kb``) for embedding in the hub.
  * ``summary.json`` — aggregate 1:1 stats for the report tables.
"""

from __future__ import annotations

import argparse
import collections
import json
import os

import imageio.v2 as imageio
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _grid(images, cols, pad=3, bg=255):
    h, w = images[0].shape[:2]
    rows = (len(images) + cols - 1) // cols
    canvas = np.full((rows * h + (rows + 1) * pad, cols * w + (cols + 1) * pad, 3), bg, np.uint8)
    for i, im in enumerate(images):
        r, c = divmod(i, cols)
        y, x = pad + r * (h + pad), pad + c * (w + pad)
        canvas[y : y + h, x : x + w] = im
    return canvas


def _gif(video_path, gif_path, max_kb, scale=1.0):
    frames = imageio.mimread(video_path, memtest=False)
    for s, fps, colors in [(scale, 12, 128), (scale * 0.8, 10, 96), (scale * 0.6, 8, 64), (scale * 0.5, 6, 48)]:
        out = []
        for f in frames:
            im = Image.fromarray(f).convert("RGB")
            if s != 1.0:
                im = im.resize((int(im.width * s), int(im.height * s)), Image.BILINEAR)
            out.append(im.convert("P", palette=Image.ADAPTIVE, colors=colors))
        out[0].save(gif_path, save_all=True, append_images=out[1:], duration=int(1000 / fps), loop=0, optimize=True)
        if os.path.getsize(gif_path) <= max_kb * 1024:
            return os.path.getsize(gif_path) // 1024, len(frames)
    return os.path.getsize(gif_path) // 1024, len(frames)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default="outputs/native_sweep")
    ap.add_argument("--out", default="outputs/native_sweep/report_assets")
    ap.add_argument("--thumb_w", type=int, default=232, help="downscaled montage cell width")
    ap.add_argument("--max_kb", type=int, default=480)
    args = ap.parse_args()

    sweep = os.path.join(ROOT, args.sweep)
    out = os.path.join(ROOT, args.out)
    os.makedirs(out, exist_ok=True)
    results = [r for r in json.load(open(os.path.join(sweep, "results.json"))) if "error" not in r]
    by_suite = collections.defaultdict(list)
    for r in results:
        by_suite[r["suite"]].append(r)

    # highlight GIFs: 2 per suite (prefer an articulated/multi-pred task + a pick-place)
    highlights = []
    for suite in ["libero_object", "libero_goal", "libero_spatial", "libero_10", "libero_90"]:
        rs = sorted(by_suite.get(suite, []), key=lambda r: (-len(r["predicates"]), r["task"]))
        for r in rs[:2]:
            tag = f"{r['suite']}__{r['task']}"
            vp = os.path.join(sweep, "videos", tag + ".mp4")
            if os.path.isfile(vp):
                gp = os.path.join(out, f"hl_{tag}.gif")
                kb, nf = _gif(vp, gp, args.max_kb)
                highlights.append({
                    "suite": suite,
                    "task": r["task"],
                    "gif": f"hl_{tag}.gif",
                    "kb": kb,
                    "preds": r["predicates"],
                })
                print(f"[gif] {tag[:60]:60s} {kb}KB")

    # per-suite montage contact sheets
    montages = {}
    for suite, rs in by_suite.items():
        rs = sorted(rs, key=lambda r: r["task"])
        cells = []
        for r in rs:
            tp = os.path.join(sweep, "thumbs", f"{r['suite']}__{r['task']}.png")
            if not os.path.isfile(tp):
                continue
            im = Image.open(tp).convert("RGB")
            w = args.thumb_w
            im = im.resize((w, int(im.height * w / im.width)), Image.BILINEAR)
            cells.append(np.asarray(im))
        if not cells:
            continue
        cols = {"libero_90": 6}.get(suite, 5)
        grid = _grid(cells, cols)
        mp = os.path.join(out, f"montage_{suite}.png")
        Image.fromarray(grid).save(mp, optimize=True)
        montages[suite] = {"png": f"montage_{suite}.png", "n": len(cells), "shape": list(grid.shape[:2])}
        print(
            f"[montage] {suite:14s} {len(cells)} tasks -> {grid.shape[1]}x{grid.shape[0]}  {os.path.getsize(mp) // 1024}KB"
        )

    # aggregate summary
    def cnt(pred):
        return sum(1 for r in results if pred(r))

    summary = {
        "n_tasks": len(results),
        "checker_1to1_zero_mismatch": cnt(lambda r: r["state_replay_checker_mismatches"] == 0),
        "native_eq_gt_checker": cnt(lambda r: r["native_vs_gt_checker_mismatches"] == 0),
        "render_bitwise_mae0": cnt(lambda r: (r["render_mae_mean"] or 0) < 0.5),
        "render_mae_mean_overall": round(
            float(np.mean([r["render_mae_mean"] for r in results if r["render_mae_mean"] is not None])), 4
        ),
        "render_mae_max_overall": round(
            float(np.max([r["render_mae_max"] for r in results if r["render_mae_max"] is not None])), 3
        ),
        "gt_state_replay_success": cnt(lambda r: r["gt_state_replay_success"]),
        "native_state_replay_success": cnt(lambda r: r["native_state_replay_success"]),
        "native_osc_success": cnt(lambda r: r["native_osc_first_success_step"] is not None),
        "by_suite": {s: len(v) for s, v in by_suite.items()},
        "predicate_coverage": dict(collections.Counter(p for r in results for p in r["predicates"])),
        "highlights": highlights,
        "montages": montages,
    }
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=2)
    print("\n=== summary ===")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("highlights", "montages")}, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
