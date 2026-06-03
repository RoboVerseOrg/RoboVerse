"""Native SimplerEnv port (zero external mani_skill2_real2sim / simpler_env dependency).

Goal: reproduce SimplerEnv's tasks + trajectories on MetaSim's own SAPIEN backend so the
upstream packages can be deleted. The controller stack under ``control/`` is vendored
verbatim from the ManiSkill2-real2sim fork (pure SAPIEN, no env/gym-framework coupling)
and verified to behave bit-identically to upstream on the same articulation.
"""

from __future__ import annotations
