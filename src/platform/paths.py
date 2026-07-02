"""Platform abstraction: paths.

Single source of truth for filesystem paths. All paths are ``pathlib.Path``.
Environment variables override defaults so the same code runs identically on
Windows, Linux cloud, and Linux home rig.

路径抽象层。所有路径通过 ``pathlib.Path``，
环境变量可覆盖，保证 Windows / 云 Linux / 家用 Linux 代码一致。
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def project_root() -> Path:
    """Return the project root directory.

    Currently pinned as the parent of the ``src`` package (which is where this
    file lives, three levels up). Override with ``DEVAGI_PROJECT_ROOT`` if the
    package is installed rather than imported by path.
    """
    override = os.environ.get("DEVAGI_PROJECT_ROOT")
    if override:
        return Path(override).resolve()
    # this file: D:\karbon\src\platform\paths.py  →  root is 3 levels up
    return Path(__file__).resolve().parents[2]


def _env_path(env: str, default_subdir: str) -> Path:
    override = os.environ.get(env)
    if override:
        return Path(override).expanduser().resolve()
    return (project_root() / default_subdir).resolve()


def data_dir() -> Path:
    """Return the data directory (replay cold tier, datasets)."""
    return _env_path("DEVAGI_DATA_DIR", "data")


def logs_dir() -> Path:
    """Return the logs directory."""
    return _env_path("DEVAGI_LOGS_DIR", "logs")


def ckpt_dir() -> Path:
    """Return the checkpoints directory."""
    return _env_path("DEVAGI_CKPT_DIR", "checkpoints")


def configs_dir() -> Path:
    return project_root() / "configs"


def docs_dir() -> Path:
    return project_root() / "docs"


def ensure_writable(path: Path) -> Path:
    """Create the directory (and parents) if missing. Return the same path.

    确保目录存在并可写。
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def stage_log_dir(stage: int, run_id: str) -> Path:
    """Compose the log directory for a given stage and run.

    Layout: ``logs/stage{N}/{run_id}/``
    """
    return ensure_writable(logs_dir() / f"stage{stage}" / run_id)


def stage_ckpt_path(stage: int, step: int) -> Path:
    """Compose the checkpoint file path for a given stage and step.

    Layout: ``checkpoints/ckpt_stage{N}_{step:09d}.pt``
    """
    ensure_writable(ckpt_dir())
    return ckpt_dir() / f"ckpt_stage{stage}_{step:09d}.pt"
