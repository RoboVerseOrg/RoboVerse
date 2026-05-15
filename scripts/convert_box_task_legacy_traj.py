#!/usr/bin/env python3
"""One-shot converter for the legacy box_task trajectory pickle.

The bundled pkl from the original recording was keyed by ``openarm_wuji``
and used legacy hand-joint names ``{side}_hand_finger{i}_joint{j}``. The
current ``OpenarmBimanualWujiCfg`` advertises ``openarm_bimanual_wuji``
and ``{side}_finger{i}_joint{j}``, so the pkl needs both the robot key
and every per-frame dof dictionary remapped before ``get_traj`` can read
it.

Run once at integration time. The output is what the task references via
``traj_filepath``; the legacy input lives outside the repo (it's an
external recording artifact).

Usage::

    python scripts/convert_box_task_legacy_traj.py \\
        --src ~/projects/RoboVerse/box_task_replay_render_bundle_clean/\\
assets/traj/task3_meshycup_openarm_wuji_20260513_232823_0_v2.pkl \\
        --dst roboverse_data/trajs/box_task/task3_openarm_bimanual_wuji_v2.pkl
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

LEGACY_ROBOT_NAME = "openarm_wuji"
CURRENT_ROBOT_NAME = "openarm_bimanual_wuji"

DOF_FIELDS = ("dof_pos", "dof_vel", "dof_pos_target", "dof_vel_target", "dof_torque", "dof_effort_target")


def _finger_remap() -> dict[str, str]:
    """``{side}_hand_finger{i}_joint{j}`` → ``{side}_finger{i}_joint{j}``."""
    return {
        f"{side}_hand_finger{i}_joint{j}": f"{side}_finger{i}_joint{j}"
        for side in ("left", "right")
        for i in range(1, 6)
        for j in range(1, 5)
    }


def _remap_dof_dict(dofs: dict, mapping: dict[str, str]) -> dict:
    return {mapping.get(k, k): v for k, v in dofs.items()}


def _remap_robot_state(robot_state: dict, mapping: dict[str, str]) -> dict:
    out = dict(robot_state)
    for field in DOF_FIELDS:
        v = out.get(field)
        if isinstance(v, dict):
            out[field] = _remap_dof_dict(v, mapping)
    return out


def _remap_v2_state(state: dict, mapping: dict[str, str]) -> dict:
    """v2 state is flat: ``{object_name: object_state, robot_name: robot_state}``."""
    out = {}
    for k, v in state.items():
        if k == LEGACY_ROBOT_NAME and isinstance(v, dict):
            out[CURRENT_ROBOT_NAME] = _remap_robot_state(v, mapping)
        else:
            out[k] = v
    return out


def _remap_action(action: dict, mapping: dict[str, str]) -> dict:
    """Per-step action dict carries dof_*_target fields keyed by joint name."""
    return _remap_robot_state(action, mapping)


def convert(src: Path, dst: Path) -> None:
    mapping = _finger_remap()
    with src.open("rb") as f:
        data = pickle.load(f)

    if LEGACY_ROBOT_NAME not in data:
        raise KeyError(
            f"Source pkl does not contain top-level key {LEGACY_ROBOT_NAME!r}; got keys: {list(data.keys())}"
        )
    episodes = data[LEGACY_ROBOT_NAME]

    new_episodes = []
    for ep in episodes:
        new_ep = dict(ep)
        if "init_state" in ep and isinstance(ep["init_state"], dict):
            new_ep["init_state"] = _remap_v2_state(ep["init_state"], mapping)
        if "states" in ep and isinstance(ep["states"], list):
            new_ep["states"] = [_remap_v2_state(s, mapping) for s in ep["states"]]
        if "actions" in ep and isinstance(ep["actions"], list):
            new_ep["actions"] = [_remap_action(a, mapping) for a in ep["actions"]]
        new_episodes.append(new_ep)

    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("wb") as f:
        pickle.dump({CURRENT_ROBOT_NAME: new_episodes}, f, protocol=pickle.HIGHEST_PROTOCOL)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, required=True, help="Legacy v2 pkl from the bundled recording.")
    parser.add_argument("--dst", type=Path, required=True, help="Output path for the converted pkl.")
    args = parser.parse_args()

    convert(args.src, args.dst)
    print(f"converted {args.src} -> {args.dst}")


if __name__ == "__main__":
    main()
