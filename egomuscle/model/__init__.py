"""Model components for EgoMuscle."""

from .egomuscle import EgoMuscleModel
from .fusion import CrossAttentionFusion, LateFusionBlock
from .muscle_encoder import MuscleEncoder
from .video_encoder import FrozenVideoEncoder, VideoAdapter

__all__ = [
    "CrossAttentionFusion",
    "EgoMuscleModel",
    "FrozenVideoEncoder",
    "LateFusionBlock",
    "MuscleEncoder",
    "VideoAdapter",
]
