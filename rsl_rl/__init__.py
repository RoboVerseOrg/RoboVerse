"""Compatibility shim for `rsl_rl` across multiple Python versions.

This repo uses RSL-RL for training/evaluation. Some environments (notably IsaacGym)
still use older Python versions, so we expose `rsl_rl` as a pkgutil namespace package
and patch a small import-time incompatibility in `rsl_rl.algorithms`.
"""

from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)
