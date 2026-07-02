"""Longevity test harness.

Runs a bounded workload for a target duration (default 24h) and produces a
report on VRAM/RAM stability. Passing this harness is Stage 0's exit criterion.

存活性测试框架。对目标时长运行并出报告；这是 Stage 0 的 exit 标准。
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from src.monitoring.memory_watcher import MemoryWatcher, WatcherConfig
from src.platform import project_root, stage_log_dir

logger = logging.getLogger(__name__)


@dataclass
class LongevityConfig:
    duration_seconds: float
    stage: int
    run_id: str
    sample_interval_s: float = 5.0
    max_used_slope_gb_per_hour: float = 0.2
    tick_interval_s: float = 0.5


@dataclass
class LongevityReport:
    passed: bool
    duration_seconds: float
    final_slope_gb_per_hour: float | None
    alarm_fired: bool
    summary: dict


def run_longevity(
    cfg: LongevityConfig,
    workload_step: Callable[[int], None],
) -> LongevityReport:
    """Run a longevity test.

    ``workload_step(step_idx)`` will be called in a tight loop; it should
    approximate a real training iteration in terms of allocator pressure.

    循环调用 ``workload_step(step_idx)``，产出报告。
    """
    log_dir = stage_log_dir(cfg.stage, cfg.run_id)
    csv_path = log_dir / "memory.csv"
    watcher = MemoryWatcher(
        WatcherConfig(
            sample_interval_s=cfg.sample_interval_s,
            slope_alarm_gb_per_hour=cfg.max_used_slope_gb_per_hour,
            csv_path=csv_path,
            max_history_points=int(cfg.duration_seconds / cfg.sample_interval_s) + 128,
        )
    )

    t0 = time.time()
    step = 0
    deadline = t0 + cfg.duration_seconds
    logger.info(
        "Starting longevity test: duration=%.0fs slope_threshold=%.3f GB/h  logs=%s",
        cfg.duration_seconds,
        cfg.max_used_slope_gb_per_hour,
        log_dir,
    )

    while time.time() < deadline:
        workload_step(step)
        watcher.tick(step=step)
        step += 1
        # Fine sleep so we don't 100% burn CPU on a smoke workload
        time.sleep(cfg.tick_interval_s)

    slope = watcher.slope_gb_per_hour()
    summary = watcher.snapshot_summary()
    passed = (slope is None) or (slope <= cfg.max_used_slope_gb_per_hour)

    report = LongevityReport(
        passed=passed,
        duration_seconds=time.time() - t0,
        final_slope_gb_per_hour=slope,
        alarm_fired=summary["alarm_fired"],
        summary=summary,
    )

    report_path = log_dir / "longevity_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "passed": report.passed,
                "duration_seconds": report.duration_seconds,
                "final_slope_gb_per_hour": report.final_slope_gb_per_hour,
                "alarm_fired": report.alarm_fired,
                "config": {
                    "stage": cfg.stage,
                    "run_id": cfg.run_id,
                    "duration_seconds": cfg.duration_seconds,
                    "sample_interval_s": cfg.sample_interval_s,
                    "max_used_slope_gb_per_hour": cfg.max_used_slope_gb_per_hour,
                },
                "summary": report.summary,
            },
            f,
            indent=2,
            default=str,
        )
    logger.info("Longevity report written: %s (passed=%s)", report_path, report.passed)
    return report


def _noop_workload(step: int) -> None:
    """A no-op workload used only for framework self-test."""
    return None


def _main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="devagi longevity harness (framework self-test)")
    ap.add_argument("--stage", type=int, default=0)
    ap.add_argument("--duration", type=float, default=60.0, help="seconds; default 60s for smoke")
    ap.add_argument("--run-id", type=str, default=time.strftime("%Y%m%d_%H%M%S"))
    ap.add_argument("--slope-threshold", type=float, default=0.2)
    args = ap.parse_args()

    cfg = LongevityConfig(
        duration_seconds=args.duration,
        stage=args.stage,
        run_id=args.run_id,
        max_used_slope_gb_per_hour=args.slope_threshold,
    )
    report = run_longevity(cfg, _noop_workload)
    print(f"passed={report.passed}  slope={report.final_slope_gb_per_hour}")
    return 0 if report.passed else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
