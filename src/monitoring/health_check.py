"""Health check: verify every bounded component is within its declared capacity.

Registers bounded components at construction; a periodic sweep asserts each
component reports ``len(comp) <= comp.capacity``. Any breach raises
:class:`BoundedComponentError` (unless in report-only mode).

组件健康检查：注册每个 bounded 组件，周期性 sweep 校验容量约束。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)


class BoundedComponentError(RuntimeError):
    """Raised when a bounded component exceeds its declared capacity."""


@runtime_checkable
class BoundedComponent(Protocol):
    """Any object exposing a fixed capacity and current size.

    协议：任何声明容量上限并可回报当前长度的组件。
    """

    @property
    def capacity(self) -> int: ...

    def __len__(self) -> int: ...


@dataclass
class HealthReport:
    name: str
    size: int
    capacity: int
    ok: bool


class HealthChecker:
    """Registry + sweeper for bounded components.

    Every stage's bootstrap should register its bounded components here.
    """

    def __init__(self, *, strict: bool = True) -> None:
        self._components: dict[str, BoundedComponent] = {}
        self._strict = strict

    def register(self, name: str, component: BoundedComponent) -> None:
        if not isinstance(component, BoundedComponent):
            raise TypeError(
                f"Component {name!r} does not conform to BoundedComponent protocol "
                f"(needs `capacity` property and `__len__`)"
            )
        if name in self._components:
            raise ValueError(f"Component {name!r} already registered")
        self._components[name] = component
        logger.debug("Registered bounded component: %s (cap=%d)", name, component.capacity)

    def unregister(self, name: str) -> None:
        self._components.pop(name, None)

    def sweep(self) -> list[HealthReport]:
        """Verify each registered component. Return one report per component.

        In strict mode, first breach raises :class:`BoundedComponentError`.
        """
        reports: list[HealthReport] = []
        for name, comp in self._components.items():
            size = len(comp)
            cap = comp.capacity
            ok = size <= cap
            report = HealthReport(name=name, size=size, capacity=cap, ok=ok)
            reports.append(report)
            if not ok:
                msg = (
                    f"Bounded component {name!r} exceeded capacity: "
                    f"size={size} cap={cap}"
                )
                logger.error(msg)
                if self._strict:
                    raise BoundedComponentError(msg)
        return reports

    def summary(self) -> dict:
        return {
            name: {"size": len(comp), "capacity": comp.capacity}
            for name, comp in self._components.items()
        }

    @property
    def registered(self) -> list[str]:
        return list(self._components.keys())
