"""MetaSim-native ManiSkill reproduction layer (clone-deletable, no runtime mani_skill import).

This package holds the reusable logic that lets the shipped ``maniskill.*`` tasks reproduce native
ManiSkill (``physx_cpu``) dynamics 1:1 through the standard MetaSim ``BaseTaskEnv`` + sapien3 handler
path, using the opt-in PhysX recipe knobs added to ``SimParamCfg`` / the sapien3 handler.

- :mod:`control` — the vendored ManiSkill ``pd_joint_*`` controllers.
- :mod:`recipe`  — the panda robot cfg, table-workspace box, and the ManiSkill-faithful SimParamCfg.
- :mod:`base`    — :class:`ManiSkillNativeTask`, the shared task base.
"""
