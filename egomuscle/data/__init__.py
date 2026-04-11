"""Data utilities for EgoMuscle.

Import heavy dataset code lazily so lightweight preparation scripts can run
without requiring the full training stack at import time.
"""

__all__ = ["EgoMuscleDataset", "EgoMuscleSample", "discover_records", "load_video_frames"]


def __getattr__(name):
    if name in __all__:
        from .dataset import EgoMuscleDataset, EgoMuscleSample, discover_records, load_video_frames

        namespace = {
            "EgoMuscleDataset": EgoMuscleDataset,
            "EgoMuscleSample": EgoMuscleSample,
            "discover_records": discover_records,
            "load_video_frames": load_video_frames,
        }
        return namespace[name]
    raise AttributeError(name)
