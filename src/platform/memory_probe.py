"""Platform abstraction: memory probe.

Cross-platform memory reporting. On CUDA hosts it reports GPU VRAM;
on CPU-only hosts it reports process RSS + system RAM.

跨平台内存探测。有 CUDA 时报 VRAM，纯 CPU 时报进程 RSS 与系统 RAM。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import psutil
import torch

from .device import get_device_info

# pynvml is imported lazily and gracefully degraded
_pynvml: Any = None


def _try_pynvml() -> Any:
    global _pynvml
    if _pynvml is not None:
        return _pynvml
    try:
        import pynvml as _mod  # type: ignore

        _mod.nvmlInit()
        _pynvml = _mod
    except Exception:
        _pynvml = False  # sentinel: probed and failed
    return _pynvml


@dataclass(frozen=True)
class MemorySnapshot:
    """A single snapshot of memory state.

    All values are in bytes. When the primary device is CPU, ``used_bytes`` and
    ``total_bytes`` refer to system RAM; ``process_rss_bytes`` is the current
    Python process footprint.

    单次内存快照。CPU 模式下 used/total 是系统 RAM，process_rss 是当前进程占用。
    """

    kind: str  # "cuda" | "cpu"
    used_bytes: int
    total_bytes: int
    process_rss_bytes: int

    @property
    def used_gb(self) -> float:
        return self.used_bytes / (1024**3)

    @property
    def total_gb(self) -> float:
        return self.total_bytes / (1024**3)

    @property
    def process_rss_gb(self) -> float:
        return self.process_rss_bytes / (1024**3)

    @property
    def used_fraction(self) -> float:
        return (self.used_bytes / self.total_bytes) if self.total_bytes > 0 else 0.0


def snapshot() -> MemorySnapshot:
    """Take a snapshot of the primary compute device's memory.

    On CUDA: reports allocated + reserved bytes and total VRAM.
    On CPU:  reports system RAM used + total; process RSS separately.

    CUDA 上报 allocated+reserved 与总 VRAM；CPU 上报系统 RAM 与进程 RSS。
    """
    info = get_device_info()
    rss = psutil.Process().memory_info().rss

    if info.kind == "cuda":
        # Prefer NVML for a true "used" reading (includes other processes).
        # Fall back to torch's own accounting.
        pynvml = _try_pynvml()
        total = info.total_memory_bytes
        used = 0
        if pynvml:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(info.device.index or 0)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                used = int(mem.used)
                total = int(mem.total)
            except Exception:
                used = 0
        if used == 0:
            # Torch-side "reserved" is the closest to what our process has committed.
            used = int(torch.cuda.memory_reserved(info.device))
        return MemorySnapshot(
            kind="cuda",
            used_bytes=used,
            total_bytes=total,
            process_rss_bytes=int(rss),
        )

    # CPU / XPU / MPS: report system RAM
    vm = psutil.virtual_memory()
    return MemorySnapshot(
        kind="cpu",
        used_bytes=int(vm.total - vm.available),
        total_bytes=int(vm.total),
        process_rss_bytes=int(rss),
    )


def empty_cache() -> None:
    """Ask the allocator to release cached blocks.

    On CUDA this calls ``torch.cuda.empty_cache()``. On CPU it's a no-op.

    请求分配器释放缓存块。CUDA 上调 empty_cache；CPU 上什么都不做。
    """
    info = get_device_info()
    if info.kind == "cuda":
        torch.cuda.empty_cache()
