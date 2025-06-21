"""Dex utility package for robot hand retargeting and manipulation."""

from importlib.util import find_spec

if find_spec("torch") is None:
    raise ImportError(
        "Torch is not installed. You can install it using the following command:\n"
        "pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
        "Note that cpu-only torch already works for dex-retargeting."
    )
