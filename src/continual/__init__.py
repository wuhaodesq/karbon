"""Public API for :mod:`src.continual`."""

from .consolidation import (
    ConsolidationConfig,
    ConsolidationCounters,
    ConsolidationTask,
    SleepConsolidationLoop,
)
from .online_ewc import OnlineEWC, OnlineEWCConfig

__all__ = [
    "ConsolidationConfig",
    "ConsolidationCounters",
    "ConsolidationTask",
    "OnlineEWC",
    "OnlineEWCConfig",
    "SleepConsolidationLoop",
]
