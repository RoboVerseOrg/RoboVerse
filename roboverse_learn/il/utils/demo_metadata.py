"""What a demo's ``metadata.json`` must carry before a converter reads a single frame.

Used by the zarr (diffusion policy) converter, so a demo that cannot feed the requested observation /
action space is refused at the start with the reason, instead of a ``KeyError`` or ``TypeError`` deep
inside the frame loop. The two LeRobot converters run inside the policy's own environment, where this
package is not importable, and carry a local copy of the null-target check. Import-safe: no converter
dependencies.
"""

from __future__ import annotations


def check_demo_metadata(
    metadata: dict,
    *,
    observation_space: str = "joint_pos",
    action_space: str = "joint_pos",
    demo_dir: str,
    state_key: str = "joint_qpos",
    action_key: str = "joint_qpos_target",
) -> None:
    """Refuse a demo whose ``metadata.json`` cannot feed the requested spaces, before any frame is read.

    The ``ee`` spaces read ``robot_ee_state`` / ``robot_ee_state_target`` as ``[pos, quat, ...]`` rows;
    the demo writer (``metasim.utils.save_util.save_demo``) records ``ee_state`` as ``[pos, rpy, grip]``
    and no EE *target* at all, so those spaces used to fail with a ``KeyError`` deep inside the frame
    loop, or run on mislabelled numbers if a key happened to exist. ``joint_pos`` actions need the joint
    targets, which a recording backend that reports none leaves as ``null`` (a ``TypeError`` on slicing).
    """
    if state_key not in metadata:
        raise ValueError(f"{demo_dir}: metadata.json has no {state_key!r}; not a RoboVerse demo")
    if "ee" in (observation_space, action_space):
        needed = ["robot_root_state", "robot_ee_state"] + (["robot_ee_state_target"] if action_space == "ee" else [])
        missing = [k for k in needed if k not in metadata]
        if missing:
            raise ValueError(
                f"{demo_dir}: the 'ee' observation/action space needs {needed} in metadata.json ([pos, quat, ...] "
                f"rows), missing {missing}. The current demo writer records 'ee_state' as [pos, rpy, gripper] and no EE "
                "target, so re-collect with a writer that records them or use --observation_space/--action_space joint_pos."
            )
    if action_space in ("joint_pos", "ee"):
        targets = metadata.get(action_key)
        if targets is None or any(t is None for t in targets):
            raise ValueError(
                f"{demo_dir}: {action_key!r} is missing or null (the recording backend reported no joint "
                "targets), so no action can be built; re-collect on a backend that reports joint_pos_target"
            )
        if len(targets) != len(metadata[state_key]):
            raise ValueError(
                f"{demo_dir}: {len(targets)} joint targets for {len(metadata[state_key])} states; the demo is misaligned"
            )
