"""Public API for :mod:`src.training`.

Phase 0+: Dreamer-style imagination training for 10x sample efficiency.
"""

from .imagination_trainer import ImaginationTrainer, ImaginationConfig

__all__ = [
    "ImaginationConfig",
    "ImaginationTrainer",
]
