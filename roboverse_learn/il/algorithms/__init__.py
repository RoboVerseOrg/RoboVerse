from __future__ import annotations

from .base_runner import PolicyRunner


def get_runner(algo: str) -> type[PolicyRunner]:
    if algo == "diffusion_policy":
        from .dp_runner import DPRunner

        return DPRunner
    elif algo.lower() == "openvla":
        from .vla_runner import OpenVLARunner

        return OpenVLARunner
    elif algo.lower() == "octo":
        from .vla_runner import OctoVLARunner

        return OctoVLARunner

    elif algo.lower() == "act":
        from .act_runner import ACTRunner

        return ACTRunner
    else:
        raise ValueError(f"Unknown algorithm: {algo}")
