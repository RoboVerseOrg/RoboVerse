"""mjlab → MetaSim/MuJoCo integration tools.

Lives in RoboVerse (downstream) since it depends on `roboverse_pack.tasks.mjlab`.
The harness loads mjlab MJCFs through (a) raw `mujoco` (reference) and (b) the
MetaSim `MujocoHandler`, then diffs qpos/qvel/ctrl trajectories and renders
video pairs for visual inspection.

See `tools/mjlab_integration/README.md` for usage.
"""
