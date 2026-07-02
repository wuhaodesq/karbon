"""Preflight checklist for cloud training.

Runs on the cloud machine right after ``setup_env.sh``. Verifies that the
environment is ready to start Stage 0+ training. Exits with a non-zero code
if any critical check fails.

Checks performed:
  1. Python version (3.10 / 3.11 / 3.12)
  2. torch importable + CUDA available
  3. GPU detection + VRAM sane
  4. Triton importable (Stage 2b hard-requirement)
  5. Disk free space (project data + ckpts + replay)
  6. Environment variables set (or defaulted)
  7. Project modules importable (src.*)
  8. Bounded-axiom static check clean
  9. Preset files present
 10. Attempt a 20-step smoke of src.train

Usage:
    python -m scripts.preflight
    python -m scripts.preflight --preset cloud_5090

Exit code:
    0 — everything green, ready to train
    1 — at least one check failed
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logger = logging.getLogger("preflight")


CHECK_MARK = "[OK]"
FAIL_MARK = "[FAIL]"
WARN_MARK = "[WARN]"


class PreflightError(Exception):
    """A fatal preflight failure."""


def _ok(msg: str) -> None:
    print(f"  {CHECK_MARK} {msg}")


def _warn(msg: str) -> None:
    print(f"  {WARN_MARK} {msg}")


def _fail(msg: str) -> None:
    print(f"  {FAIL_MARK} {msg}", file=sys.stderr)


# =====================================================================
# Checks
# =====================================================================


def check_python() -> None:
    print("[1/10] Python version")
    v = sys.version_info
    if v.major != 3 or v.minor not in (10, 11, 12):
        raise PreflightError(f"Python 3.10/3.11/3.12 required, got {v.major}.{v.minor}")
    _ok(f"Python {v.major}.{v.minor}.{v.micro}")


def check_torch() -> tuple[str, bool, int]:
    print("[2/10] PyTorch + CUDA")
    try:
        import torch
    except ImportError as exc:
        raise PreflightError(f"torch not importable: {exc}") from exc
    _ok(f"torch: {torch.__version__}")
    cuda_ok = torch.cuda.is_available()
    if not cuda_ok:
        _warn("CUDA not available — cloud training will fall back to CPU (slow)")
        return torch.__version__, False, 0
    n = torch.cuda.device_count()
    _ok(f"CUDA available; {n} device(s)")
    return torch.__version__, True, n


def check_gpus() -> None:
    print("[3/10] GPU inventory")
    import torch
    if not torch.cuda.is_available():
        _warn("skipped (no CUDA)")
        return
    total_gb = 0.0
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        gb = props.total_memory / 1024**3
        total_gb += gb
        cc = f"sm_{props.major}{props.minor}"
        _ok(f"cuda:{i}  {props.name}  {gb:.1f} GB  {cc}")
    _ok(f"Total VRAM: {total_gb:.1f} GB")
    if total_gb < 8:
        raise PreflightError(f"VRAM {total_gb:.1f} GB < 8 GB minimum for cloud training")


def check_triton() -> None:
    print("[4/10] Triton (Stage 2b requirement)")
    try:
        import triton
        _ok(f"triton: {triton.__version__}")
    except ImportError:
        _warn("triton not installed — Stage 2b will fall back to PyTorch backend")


def check_disk() -> None:
    print("[5/10] Disk space")
    total, used, free = shutil.disk_usage(str(_ROOT))
    free_gb = free / 1024**3
    total_gb = total / 1024**3
    _ok(f"Project filesystem: {free_gb:.1f} GB free / {total_gb:.1f} GB total")
    if free_gb < 30:
        raise PreflightError(f"Only {free_gb:.1f} GB free — need at least 30 GB (recommend 100+)")
    if free_gb < 100:
        _warn(f"{free_gb:.1f} GB free — OK for Stage 0-2, but expand to 170 GB before Stage 3")


def check_env_vars() -> None:
    print("[6/10] Environment variables")
    import os
    for name, default in [
        ("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"),
        ("DEVAGI_PRESET", "cloud_5090"),
    ]:
        v = os.environ.get(name)
        if v is None:
            _warn(f"{name} unset (default: {default!r})")
        else:
            _ok(f"{name}={v}")


def check_project_imports() -> None:
    print("[7/10] Project modules importable")
    modules = [
        "src.platform",
        "src.monitoring",
        "src.memory",
        "src.models",
        "src.intrinsic",
        "src.curriculum",
        "src.continual",
        "src.envs",
        "src.utils",
    ]
    for m in modules:
        try:
            __import__(m)
        except ImportError as exc:
            raise PreflightError(f"cannot import {m}: {exc}") from exc
    _ok(f"{len(modules)} project subpackages OK")


def check_bounded() -> None:
    print("[8/10] Bounded-axiom static check")
    proc = subprocess.run(
        [sys.executable, str(_ROOT / "scripts" / "ci" / "check_bounded.py")],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        _fail(proc.stdout + proc.stderr)
        raise PreflightError("check_bounded reported findings")
    _ok(proc.stdout.strip())


def check_presets(preset: str) -> None:
    print("[9/10] Preset files")
    from src.utils import load_config
    try:
        cfg = load_config("stage0_baseline.yaml", preset)
    except Exception as exc:
        raise PreflightError(f"cannot load preset {preset!r}: {exc}") from exc
    _ok(f"preset={preset} model.hidden_size={cfg['model']['hidden_size']} "
        f"batch={cfg['train']['batch_size']} seq={cfg['train']['seq_len']} "
        f"num_envs={cfg['env']['num_envs']}")


def check_smoke(preset: str) -> None:
    print("[10/10] Smoke: 20-step training tick")
    cmd = [
        sys.executable, "-m", "src.train",
        "--stage", "0", "--preset", preset, "--smoke-only",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(_ROOT))
    if proc.returncode != 0:
        _fail("smoke run failed:")
        sys.stderr.write(proc.stdout[-2000:] if proc.stdout else "")
        sys.stderr.write(proc.stderr[-2000:] if proc.stderr else "")
        raise PreflightError("smoke run returned non-zero")
    # A rough sanity check: look for the "Training finished" log line
    if "Training finished" not in (proc.stdout or ""):
        _warn("smoke completed but 'Training finished' not found in output")
    else:
        _ok("smoke completed — training loop is healthy")


# =====================================================================
# Driver
# =====================================================================


def main() -> int:
    logging.basicConfig(level=logging.WARNING)
    ap = argparse.ArgumentParser(description="devagi preflight checklist")
    ap.add_argument("--preset", type=str, default="cloud_5090",
                    help="Preset used for the smoke step (default: cloud_5090)")
    ap.add_argument("--skip-smoke", action="store_true",
                    help="Skip the final 20-step smoke (faster; verifies env only)")
    args = ap.parse_args()

    print("== devagi preflight ==\n")
    try:
        check_python()
        check_torch()
        check_gpus()
        check_triton()
        check_disk()
        check_env_vars()
        check_project_imports()
        check_bounded()
        check_presets(args.preset)
        if not args.skip_smoke:
            check_smoke(args.preset)
        else:
            print("[10/10] Smoke: SKIPPED (--skip-smoke)")
    except PreflightError as exc:
        print()
        print("========================================")
        print("PREFLIGHT FAILED:", exc)
        print("========================================")
        return 1

    print()
    print("========================================")
    print("PREFLIGHT PASSED — ready for training")
    print("========================================")
    print()
    print("Recommended next commands:")
    print(f"  bash scripts/cloud/run_stage.sh 0 {args.preset}")
    print("  bash scripts/cloud/longevity_24h.sh 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
