"""Structured logging + config loader utilities."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from src.platform import configs_dir, project_root, stage_log_dir


def setup_logging(level: str = "INFO", log_file: Path | None = None) -> None:
    """Basic structured logging to stdout (+optional file).

    结构化日志：stdout（+可选文件）。
    """
    fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(level=level.upper(), format=fmt, handlers=handlers, force=True)


def _deep_merge(a: dict, b: dict) -> dict:
    """Return a new dict where ``b`` overrides ``a`` recursively."""
    out = dict(a)
    for k, v in b.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(stage_config_name: str, preset: str) -> dict[str, Any]:
    """Load ``configs/_presets/{preset}.yaml`` then overlay stage config.

    先加载 preset，再叠加 stage 配置。
    """
    preset_path = configs_dir() / "_presets" / f"{preset}.yaml"
    if not preset_path.exists():
        raise FileNotFoundError(f"Preset not found: {preset_path}")

    stage_path = configs_dir() / stage_config_name
    if not stage_path.exists():
        raise FileNotFoundError(f"Stage config not found: {stage_path}")

    with preset_path.open("r", encoding="utf-8") as f:
        preset_cfg = yaml.safe_load(f) or {}
    with stage_path.open("r", encoding="utf-8") as f:
        stage_cfg = yaml.safe_load(f) or {}

    merged = _deep_merge(preset_cfg, stage_cfg)
    # Meta
    merged.setdefault("_meta", {})
    merged["_meta"]["preset_path"] = str(preset_path)
    merged["_meta"]["stage_path"] = str(stage_path)
    return merged


def make_run_id(short_sha: str | None = None) -> str:
    """Generate a run identifier: ``YYYYMMDD_HHMMSS_{shortsha}``."""
    ts = time.strftime("%Y%m%d_%H%M%S")
    if short_sha is None:
        short_sha = git_short_sha()
    return f"{ts}_{short_sha}"


def git_short_sha() -> str:
    """Return the current git short SHA, or 'nogit' if unavailable."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=project_root(),
            stderr=subprocess.DEVNULL,
        )
        return out.decode("ascii").strip()
    except Exception:
        return "nogit"


def env_flag(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}


def open_stage_log_dir(stage: int, run_id: str) -> Path:
    d = stage_log_dir(stage, run_id)
    return d
