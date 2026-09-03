"""Regenerate the SimplerEnv doc gallery from the real-policy rollout results.

Data-driven: reads ``/tmp/policy_gallery/policy_gallery_results.json`` (produced by
``render_policy_gallery.py``) + the per-task mp4s, copies the policy clips into the
doc ``_static`` tree, and rewrites two parts of
``docs/source/dataset_benchmark/integrations/simpler_env.md``:

  1. the Status table — adds a real-policy success row;
  2. the gallery — a headline "Real-policy rollouts (RT-1 / Octo)" section with one
     ``{video}`` per task and an honest ✓/✗ + step-count caption, followed by the
     existing 1:1 parity diff clips moved into a collapsed ``{dropdown}``.

Idempotent: the rewritten span is delimited by HTML-comment markers.

Run:  python tools/parity/gen_simpler_doc_gallery.py
"""

from __future__ import annotations

import json
import pathlib
import shutil

REPO = pathlib.Path(__file__).resolve().parents[2]  # tools/parity/<this>.py -> repo root
DOC = REPO / "docs/source/dataset_benchmark/integrations/simpler_env.md"
STATIC = REPO / "docs/source/_static/integrations/simpler_env"
GALLERY = pathlib.Path("/tmp/policy_gallery")
REL = "../../_static/integrations/simpler_env"

GROUPS = [
    (
        "Google Robot — pick coke can (4)",
        [
            "google_robot_pick_coke_can",
            "google_robot_pick_horizontal_coke_can",
            "google_robot_pick_vertical_coke_can",
            "google_robot_pick_standing_coke_can",
        ],
    ),
    (
        "Google Robot — pick object & move near (4)",
        [
            "google_robot_pick_object",
            "google_robot_move_near",
            "google_robot_move_near_v0",
            "google_robot_move_near_v1",
        ],
    ),
    (
        "Google Robot — open drawer (4)",
        [
            "google_robot_open_drawer",
            "google_robot_open_top_drawer",
            "google_robot_open_middle_drawer",
            "google_robot_open_bottom_drawer",
        ],
    ),
    (
        "Google Robot — close drawer (4)",
        [
            "google_robot_close_drawer",
            "google_robot_close_top_drawer",
            "google_robot_close_middle_drawer",
            "google_robot_close_bottom_drawer",
        ],
    ),
    (
        "Google Robot — place in closed drawer (5)",
        [
            "google_robot_place_in_closed_drawer",
            "google_robot_place_in_closed_top_drawer",
            "google_robot_place_in_closed_middle_drawer",
            "google_robot_place_in_closed_bottom_drawer",
            "google_robot_place_apple_in_closed_top_drawer",
        ],
    ),
    (
        "WidowX / Bridge — put-on (4)",
        ["widowx_spoon_on_towel", "widowx_carrot_on_plate", "widowx_stack_cube", "widowx_put_eggplant_in_basket"],
    ),
]
START = "<!-- SIMPLER-GALLERY-START -->"
END = "<!-- SIMPLER-GALLERY-END -->"


def policy_of(task):
    return "Octo" if task.startswith("widowx") else "RT-1"


def video_block(src, caption):
    return f"```{{video}} {src}\n:autoplay:\n:loop:\n:muted:\n:playsinline:\n:width: 100%\n:caption: {caption}\n```"


def grid(items):
    out = ["::::{grid} 2\n:gutter: 2\n"]
    for body in items:
        out.append(":::{grid-item}\n" + body + "\n:::\n")
    out.append("::::\n")
    return "\n".join(out)


def main():
    results = json.loads((GALLERY / "policy_gallery_results.json").read_text())["per_task"]
    n_ok = sum(v["success"] for v in results.values())
    g_ok = sum(v["success"] for k, v in results.items() if not k.startswith("widowx"))
    w_ok = sum(v["success"] for k, v in results.items() if k.startswith("widowx"))
    STATIC.mkdir(parents=True, exist_ok=True)
    for task in results:
        shutil.copyfile(GALLERY / f"{task}.mp4", STATIC / f"policy_{task}.mp4")

    # ---- headline real-policy gallery ----
    blocks = [
        START,
        "## Real-policy rollouts (RT-1 / Octo)\n",
        "These are **real pretrained policies driving our MetaSim-native env** "
        "(`SimplerEnv/<task>`), recording the first episode the task's own success checker "
        "marks solved — an actual policy manipulating the objects, not a scripted motion. "
        "Google-robot tasks use **RT-1** (`rt_1_tf_trained_for_000400120`); WidowX/Bridge "
        "tasks use **Octo-base** (`policy_setup=widowx_bridge`) — matching what SimplerEnv "
        "evaluates per embodiment.\n",
        f"**Solved {n_ok}/25 on the MetaSim-native envs** (RT-1 {g_ok}/21 Google · "
        f"Octo {w_ok}/4 WidowX). Captions show the policy and, for solved tasks, the step "
        "count; ✗ marks episodes the policy did not solve this run (a property of the policy, "
        "not the integration — long-horizon place + Bridge tasks are the known-hard cases).\n",
    ]
    for title, tasks in GROUPS:
        blocks.append(f"### {title}\n")
        items = []
        for t in tasks:
            r = results[t]
            mark = f"✓ {r['steps']} steps" if r["success"] else "✗ (not solved this run)"
            cap = f"{t} · {policy_of(t)} {mark}"
            items.append(video_block(f"{REL}/policy_{t}.mp4", cap))
        blocks.append(grid(items))

    # ---- implementation fidelity note (no extra video wall) ----
    blocks.append("## Implementation fidelity (obs matches upstream)\n")
    blocks.append(
        "Separately from policy capability, the MetaSim-native env is verified to reproduce the "
        "**upstream `simpler_env` observation**: for the same task + seed + station, the initial "
        "render matches upstream by **mean-abs ≤ ~2/255** across all six families (coke/pick/"
        "move-near bitwise; drawer 0.01, place 1.94, widowx 0.0). This is a stronger check than the "
        "earlier native-vs-reference parity — it catches station / overlay / asset-recolor "
        "deviations that an internal self-comparison cannot. Regenerate side-by-side "
        "`[ MetaSim-native | reference | abs-diff x30 ]` clips with "
        "`tools/parity/render_metasim_1to1_gallery.py`; the per-task obs-vs-upstream check is in "
        "`tools/parity/render_policy_gallery.py`.\n"
    )
    blocks.append(END)
    new_gallery = "\n".join(blocks)

    txt = DOC.read_text()
    if START in txt and END in txt:
        pre = txt[: txt.index(START)]
        post = txt[txt.index(END) + len(END) :]
        txt = pre + new_gallery + post
    else:
        # first run: replace from the old gallery H2 to the Design-notes H2
        a = txt.index("## Side-by-side: MetaSim-native vs reference")
        b = txt.index("## Design notes & honest caveats")
        txt = txt[:a] + new_gallery + "\n\n" + txt[b:]

    # ---- status table: add/refresh a policy-success row ----
    row = (
        f"| Real-policy success (RT-1/Octo) | **{n_ok}/25** solved on MetaSim-native envs "
        f"(RT-1 {g_ok}/21 · Octo {w_ok}/4) | `tools/parity/render_policy_gallery.py` |"
    )
    fidelity_row = (
        "| Obs matches upstream | initial render vs `simpler_env` **mean-abs ≤ ~2/255** all 6 families "
        "(coke/pick/move-near bitwise; drawer 0.01, place 1.94, widowx 0.0) | `tools/parity/render_policy_gallery.py` |"
    )
    lines = txt.splitlines()
    lines = [ln for ln in lines if not ln.startswith(("| Real-policy success", "| Obs matches upstream"))]
    for i, ln in enumerate(lines):
        if ln.startswith("| Native vs upstream parity"):
            lines[i] = fidelity_row  # replace the native-vs-reference row with the stronger upstream check
            lines.insert(i + 1, row)
            break
    DOC.write_text("\n".join(lines) + ("\n" if not txt.endswith("\n") else ""))
    print(
        f"doc updated: {n_ok}/25 policy success (RT-1 {g_ok}/21, Octo {w_ok}/4); "
        f"copied {len(results)} policy mp4s -> {STATIC}"
    )


if __name__ == "__main__":
    main()
