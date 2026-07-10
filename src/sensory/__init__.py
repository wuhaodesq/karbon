"""Public API for :mod:`src.sensory`.

Phase 4+ sensory modality encoders.
"""

from .audio_encoder import AudioEncoder, AudioEncoderConfig

__all__ = [
    "AudioEncoder",
    "AudioEncoderConfig",
]
