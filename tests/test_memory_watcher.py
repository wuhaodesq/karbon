"""Memory watcher / health checker unit tests."""

from __future__ import annotations

import time

import pytest

from src.monitoring import HealthChecker, HealthReport, MemoryWatcher, WatcherConfig
from src.monitoring.health_check import BoundedComponent, BoundedComponentError


class FakeBounded:
    """Minimal BoundedComponent for tests."""

    def __init__(self, cap: int) -> None:
        self._cap = cap
        self._size = 0

    @property
    def capacity(self) -> int:
        return self._cap

    def __len__(self) -> int:
        return self._size

    def push(self) -> None:
        self._size += 1


def test_bounded_protocol_conformance():
    b = FakeBounded(10)
    assert isinstance(b, BoundedComponent)
    assert b.capacity == 10
    assert len(b) == 0


def test_health_checker_passes_when_ok():
    hc = HealthChecker(strict=True)
    b = FakeBounded(5)
    hc.register("b", b)
    for _ in range(3):
        b.push()
    reports = hc.sweep()
    assert len(reports) == 1
    assert reports[0].ok
    assert reports[0].size == 3


def test_health_checker_raises_when_exceeded():
    hc = HealthChecker(strict=True)
    b = FakeBounded(2)
    hc.register("b", b)
    b.push()
    b.push()
    b.push()  # oops, over
    with pytest.raises(BoundedComponentError):
        hc.sweep()


def test_health_checker_report_only_mode():
    hc = HealthChecker(strict=False)
    b = FakeBounded(2)
    hc.register("b", b)
    b.push()
    b.push()
    b.push()
    reports = hc.sweep()
    assert not reports[0].ok
    assert reports[0].size == 3


def test_memory_watcher_ticks_and_stores():
    w = MemoryWatcher(WatcherConfig(sample_interval_s=0.01, rolling_window_s=1.0))
    for i in range(5):
        w.tick(step=i)
        time.sleep(0.02)
    summary = w.snapshot_summary()
    assert summary["num_samples"] >= 3


def test_memory_watcher_rate_limits():
    w = MemoryWatcher(WatcherConfig(sample_interval_s=5.0))
    w.tick(step=0)
    # Second tick immediately should be rate-limited (return None)
    r = w.tick(step=1)
    assert r is None


def test_memory_watcher_warmup_suppresses_alarm():
    """During warmup period, no alarm should fire even if slope crosses threshold.

    We fake this by installing a very short warmup and calling _check_slope
    directly with a synthetic sample inside the warmup window.
    """
    from src.monitoring.memory_watcher import _Sample
    import time as _time
    w = MemoryWatcher(WatcherConfig(
        warmup_seconds=60.0,
        slope_alarm_gb_per_hour=0.001,   # ridiculously low so any slope trips
        rolling_window_s=1.0,
    ))
    # Inject two samples spanning >10% of window to make slope compute
    now = _time.time()
    with w._lock:
        w._history.append(_Sample(ts=now - 0.5, used_bytes=0, total_bytes=1, process_rss_bytes=0, kind="cuda"))
        w._history.append(_Sample(ts=now, used_bytes=10**9, total_bytes=1, process_rss_bytes=0, kind="cuda"))
    # Latest sample within warmup → no alarm
    w._check_slope(w._history[-1])
    assert not w._alarm_fired


def test_memory_watcher_alarm_fires_after_warmup():
    """After warmup, a fast slope should trip the alarm."""
    from src.monitoring.memory_watcher import _Sample
    import time as _time
    w = MemoryWatcher(WatcherConfig(
        warmup_seconds=0.0,     # no warmup
        slope_alarm_gb_per_hour=0.001,
        rolling_window_s=1.0,
    ))
    now = _time.time()
    with w._lock:
        w._history.append(_Sample(ts=now - 0.5, used_bytes=0, total_bytes=1, process_rss_bytes=0, kind="cuda"))
        w._history.append(_Sample(ts=now, used_bytes=10**9, total_bytes=1, process_rss_bytes=0, kind="cuda"))
    w._check_slope(w._history[-1])
    assert w._alarm_fired
