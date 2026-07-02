"""Memory watcher: periodic sampling + rolling-window slope alarm.

Implements Axiom 5 (Fragmentation Governance) and part of Axiom 1 (Zero
Unbounded Structures on GPU): every training run must observe VRAM/RAM
trajectory and abort if a leak-like slope is detected.

内存监视器：周期采样 + 滚动窗口斜率告警。
"""

from __future__ import annotations

import csv
import logging
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Deque

from src.platform import empty_cache, snapshot
from src.platform.memory_probe import MemorySnapshot

logger = logging.getLogger(__name__)


@dataclass
class WatcherConfig:
    """Configuration for :class:`MemoryWatcher`.

    - ``sample_interval_s``: how often to snapshot memory.
    - ``rolling_window_s``: how much history to keep for slope computation.
    - ``slope_alarm_gb_per_hour``: alarm threshold (used_gb slope over the window).
    - ``empty_cache_every_steps``: proactively call ``empty_cache`` this often.
    - ``csv_path``: optional CSV file to persist snapshots.
    - ``max_history_points``: cap the in-memory ring buffer (Axiom 1).
    """

    sample_interval_s: float = 5.0
    rolling_window_s: float = 3600.0  # 1 hour default
    slope_alarm_gb_per_hour: float = 0.2
    empty_cache_every_steps: int = 10_000
    csv_path: Path | None = None
    max_history_points: int = 4096
    on_alarm: Callable[[float, MemorySnapshot], None] | None = None


@dataclass
class _Sample:
    ts: float
    used_bytes: int
    total_bytes: int
    process_rss_bytes: int
    kind: str


class MemoryWatcher:
    """Bounded rolling memory observer.

    Not a training-loop callback per se — this class exposes ``tick`` (call it
    inside the loop) and an optional background thread mode ``start()`` /
    ``stop()`` for long-running perpetual training.

    有界的滚动式内存观测器。既能循环内 tick 也可后台线程运行。
    """

    def __init__(self, config: WatcherConfig | None = None) -> None:
        self.config = config or WatcherConfig()
        # Ring buffer of samples; capacity enforced (Axiom 1).
        self._history: Deque[_Sample] = deque(maxlen=self.config.max_history_points)
        self._last_sample_ts: float = 0.0
        self._csv_file: Path | None = self.config.csv_path
        self._csv_header_written: bool = False
        self._alarm_fired: bool = False
        self._thread: threading.Thread | None = None
        self._stop_flag = threading.Event()
        self._step_counter: int = 0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ helpers

    def _write_csv(self, s: _Sample) -> None:
        if self._csv_file is None:
            return
        self._csv_file.parent.mkdir(parents=True, exist_ok=True)
        write_header = not self._csv_header_written and not self._csv_file.exists()
        with self._csv_file.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["ts", "kind", "used_bytes", "total_bytes", "process_rss_bytes"])
            w.writerow([s.ts, s.kind, s.used_bytes, s.total_bytes, s.process_rss_bytes])
        self._csv_header_written = True

    def _prune_history(self, now: float) -> None:
        # Keep only points inside the rolling window
        cutoff = now - self.config.rolling_window_s
        while self._history and self._history[0].ts < cutoff:
            self._history.popleft()

    # ------------------------------------------------------------------ core API

    def sample_now(self) -> _Sample:
        snap = snapshot()
        s = _Sample(
            ts=time.time(),
            used_bytes=snap.used_bytes,
            total_bytes=snap.total_bytes,
            process_rss_bytes=snap.process_rss_bytes,
            kind=snap.kind,
        )
        with self._lock:
            self._history.append(s)
            self._prune_history(s.ts)
        self._write_csv(s)
        return s

    def tick(self, step: int | None = None) -> _Sample | None:
        """Should be called from the training loop each step (fast; rate-limited).

        - Rate-limited to ``sample_interval_s``.
        - Every ``empty_cache_every_steps`` steps, proactively releases cache.
        - Checks slope; fires ``on_alarm`` at most once until ``reset_alarm()``.
        """
        if step is not None:
            self._step_counter = step

        now = time.time()
        if now - self._last_sample_ts < self.config.sample_interval_s:
            self._maybe_empty_cache()
            return None

        s = self.sample_now()
        self._last_sample_ts = now
        self._maybe_empty_cache()
        self._check_slope(s)
        return s

    def _maybe_empty_cache(self) -> None:
        if (
            self._step_counter > 0
            and self._step_counter % self.config.empty_cache_every_steps == 0
        ):
            empty_cache()

    def _check_slope(self, latest: _Sample) -> None:
        slope = self.slope_gb_per_hour()
        if slope is None:
            return
        if slope > self.config.slope_alarm_gb_per_hour and not self._alarm_fired:
            self._alarm_fired = True
            msg = (
                f"[MemoryWatcher] ALARM: {latest.kind} memory slope "
                f"{slope:+.3f} GB/h exceeds threshold "
                f"{self.config.slope_alarm_gb_per_hour} GB/h "
                f"(used={latest.used_bytes / 1024**3:.2f} GB, "
                f"rss={latest.process_rss_bytes / 1024**3:.2f} GB)"
            )
            logger.warning(msg)
            if self.config.on_alarm:
                snap = MemorySnapshot(
                    kind=latest.kind,
                    used_bytes=latest.used_bytes,
                    total_bytes=latest.total_bytes,
                    process_rss_bytes=latest.process_rss_bytes,
                )
                try:
                    self.config.on_alarm(slope, snap)
                except Exception:  # pragma: no cover
                    logger.exception("on_alarm callback raised")

    # ----------------------------------------------------------------- analytics

    def slope_gb_per_hour(self) -> float | None:
        """Compute the ``used_bytes`` slope over the rolling window.

        Uses (last - first) / duration. Returns None if <2 points or
        span <10% of the configured rolling window.
        """
        with self._lock:
            if len(self._history) < 2:
                return None
            first = self._history[0]
            last = self._history[-1]
        dt = last.ts - first.ts
        if dt < self.config.rolling_window_s * 0.1:
            return None
        d_bytes = last.used_bytes - first.used_bytes
        gb_per_second = (d_bytes / (1024**3)) / dt
        return gb_per_second * 3600.0

    def snapshot_summary(self) -> dict:
        with self._lock:
            n = len(self._history)
            last = self._history[-1] if n else None
        summary = {
            "num_samples": n,
            "alarm_fired": self._alarm_fired,
            "slope_gb_per_hour": self.slope_gb_per_hour(),
        }
        if last is not None:
            summary.update(asdict(last))
        return summary

    def reset_alarm(self) -> None:
        self._alarm_fired = False

    # ------------------------------------------------------------------ threaded

    def start(self) -> None:
        """Start a background sampling thread (optional).

        Prefer the ``tick()`` in-loop usage. This mode is for perpetual daemons
        where the main thread might not tick predictably.
        """
        if self._thread and self._thread.is_alive():
            return
        self._stop_flag.clear()
        self._thread = threading.Thread(
            target=self._run_loop, name="MemoryWatcher", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop_flag.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def _run_loop(self) -> None:
        while not self._stop_flag.is_set():
            try:
                s = self.sample_now()
                self._check_slope(s)
            except Exception:  # pragma: no cover
                logger.exception("MemoryWatcher background loop error")
            self._stop_flag.wait(self.config.sample_interval_s)
