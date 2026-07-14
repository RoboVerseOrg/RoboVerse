# RoboVerse-original: two-line re-export of RoboVerse's own TFDS builder. Upstream
# rlds_dataset_builder ships an empty example_dataset/__init__.py, so no upstream code is here.

from .roboverse import BridgeOrigDataset

__all__ = ["BridgeOrigDataset"]
