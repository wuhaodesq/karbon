"""Platform abstraction: device detection.

Single source of truth for choosing compute device. The rest of the codebase must
call :func:`get_device` and use the returned :class:`torch.device` — never call
``.cuda()`` directly, never hardcode ``"cuda"`` strings.

设备探测的唯一入口。业务代码必须通过 :func:`get_device` 获取设备，
禁止直接写 ``.cuda()`` 或 ``"cuda"`` 硬编码。
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache

import torch

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DeviceInfo:
    """Descriptor for the selected device.

    对所选设备的描述。
    """

    device: torch.device
    kind: str  # "cuda" | "xpu" | "mps" | "cpu"
    name: str
    total_memory_bytes: int  # 0 if unknown (e.g., CPU)


def _probe_cuda() -> DeviceInfo | None:
    if not torch.cuda.is_available():
        return None
    idx = 0
    props = torch.cuda.get_device_properties(idx)
    return DeviceInfo(
        device=torch.device(f"cuda:{idx}"),
        kind="cuda",
        name=props.name,
        total_memory_bytes=int(props.total_memory),
    )


def _probe_xpu() -> DeviceInfo | None:
    # Intel XPU (via intel-extension-for-pytorch). Optional.
    if not hasattr(torch, "xpu"):
        return None
    try:
        if not torch.xpu.is_available():  # type: ignore[attr-defined]
            return None
    except Exception:  # pragma: no cover
        return None
    return DeviceInfo(
        device=torch.device("xpu:0"),
        kind="xpu",
        name=torch.xpu.get_device_name(0),  # type: ignore[attr-defined]
        total_memory_bytes=0,
    )


def _probe_mps() -> DeviceInfo | None:
    mps = getattr(torch.backends, "mps", None)
    if mps is None or not mps.is_available():
        return None
    return DeviceInfo(
        device=torch.device("mps"),
        kind="mps",
        name="Apple MPS",
        total_memory_bytes=0,
    )


def _probe_cpu() -> DeviceInfo:
    return DeviceInfo(
        device=torch.device("cpu"),
        kind="cpu",
        name="CPU",
        total_memory_bytes=0,
    )


@lru_cache(maxsize=1)
def get_device_info(preferred: str | None = None) -> DeviceInfo:
    """Detect device once and cache. Order: CUDA → XPU → MPS → CPU.

    ``preferred`` may force one of ``"cuda" | "xpu" | "mps" | "cpu"``.
    Environment variable ``DEVAGI_DEVICE`` also honored.
    """
    forced = preferred or os.environ.get("DEVAGI_DEVICE") or ""
    forced = forced.strip().lower()

    order = ["cuda", "xpu", "mps", "cpu"]
    if forced:
        if forced not in order:
            raise ValueError(f"Unknown DEVAGI_DEVICE={forced!r}, expected one of {order}")
        order = [forced]

    for kind in order:
        info = {
            "cuda": _probe_cuda,
            "xpu": _probe_xpu,
            "mps": _probe_mps,
            "cpu": _probe_cpu,
        }[kind]()
        if info is not None:
            logger.info("Selected device: %s (%s)", info.kind, info.name)
            return info

    # Should never reach: cpu probe never returns None.
    return _probe_cpu()  # pragma: no cover


def get_device(preferred: str | None = None) -> torch.device:
    """Return the ``torch.device`` for the selected accelerator.

    Business code should use this everywhere in place of raw device strings.
    """
    return get_device_info(preferred).device


def is_cuda() -> bool:
    return get_device_info().kind == "cuda"


def is_cpu() -> bool:
    return get_device_info().kind == "cpu"


def reset_cache() -> None:
    """Testing hook: clear the LRU cache so the next call re-probes.

    仅用于测试：清 LRU cache，让下次调用重新探测。
    """
    get_device_info.cache_clear()
