"""Public API for :mod:`src.platform`."""

from .device import (
    DeviceInfo,
    get_device,
    get_device_info,
    is_cpu,
    is_cuda,
    reset_cache,
)
from .memory_probe import MemorySnapshot, empty_cache, snapshot
from .paths import (
    ckpt_dir,
    configs_dir,
    data_dir,
    docs_dir,
    ensure_writable,
    logs_dir,
    project_root,
    stage_ckpt_path,
    stage_log_dir,
)

__all__ = [
    "DeviceInfo",
    "MemorySnapshot",
    "ckpt_dir",
    "configs_dir",
    "data_dir",
    "docs_dir",
    "empty_cache",
    "ensure_writable",
    "get_device",
    "get_device_info",
    "is_cpu",
    "is_cuda",
    "logs_dir",
    "project_root",
    "reset_cache",
    "snapshot",
    "stage_ckpt_path",
    "stage_log_dir",
]
