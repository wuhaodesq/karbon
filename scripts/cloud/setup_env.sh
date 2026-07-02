#!/usr/bin/env bash
# devagi · cloud Linux environment setup (Phase 2)
# 云端 Linux 环境安装脚本
#
# Usage:
#   bash scripts/cloud/setup_env.sh [--force] [--python <python-exec>] [--skip-torch]
#
# Assumes:
#   - Linux (Ubuntu 22.04 recommended)
#   - CUDA runtime installed (verify with nvidia-smi)
#   - Python 3.10 / 3.11 / 3.12 available
#
# --skip-torch:
#   If the platform image already ships a working PyTorch (e.g., the
#   "PyTorch 2.8.0 / CUDA 12.8" preset image), pass --skip-torch to avoid
#   re-installing torch on top. Non-torch deps are still installed.

set -euo pipefail

FORCE=0
SKIP_TORCH=0
PYTHON_EXE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --force) FORCE=1; shift ;;
        --skip-torch) SKIP_TORCH=1; shift ;;
        --python) PYTHON_EXE="$2"; shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done

# Auto-pick python if not overridden
if [[ -z "${PYTHON_EXE}" ]]; then
    for cand in python3.12 python3.11 python3.10 python3; do
        if command -v "${cand}" >/dev/null 2>&1; then
            PYTHON_EXE="${cand}"
            break
        fi
    done
fi
if [[ -z "${PYTHON_EXE}" ]]; then
    echo "ERROR: no python3.{10,11,12} found in PATH" >&2
    exit 1
fi

# Project root = parent of parent of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VENV_PATH="${PROJECT_ROOT}/.venv"

echo "==> devagi cloud environment setup"
echo "Project root: ${PROJECT_ROOT}"
echo "Venv path:    ${VENV_PATH}"

# --- Verify Python ---
if ! command -v "${PYTHON_EXE}" >/dev/null 2>&1; then
    echo "ERROR: ${PYTHON_EXE} not found in PATH" >&2
    exit 1
fi
pyver="$("${PYTHON_EXE}" --version)"
echo "Detected Python: ${pyver}"
case "${pyver}" in
    "Python 3.10."*|"Python 3.11."*|"Python 3.12."*)
        ;;
    *)
        echo "WARN: expected Python 3.10 / 3.11 / 3.12, got: ${pyver}"
        if [[ ${FORCE} -eq 0 ]]; then
            echo "Aborting. Re-run with --force to override." >&2
            exit 1
        fi
        ;;
esac

# --- Verify NVIDIA / CUDA ---
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "WARN: nvidia-smi not found — this is meant for CUDA hosts."
    if [[ ${FORCE} -eq 0 ]]; then
        echo "Aborting. Use --force to install anyway (torch will be broken)." >&2
        exit 1
    fi
else
    nvidia-smi | head -20
fi

# --- Create venv ---
if [[ -d "${VENV_PATH}" ]]; then
    if [[ ${FORCE} -eq 1 ]]; then
        echo "Removing existing venv (--force)"
        rm -rf "${VENV_PATH}"
    else
        echo "Venv already exists. Use --force to recreate. Skipping creation."
    fi
fi
if [[ ! -d "${VENV_PATH}" ]]; then
    # --skip-torch requires the venv to see the system's pre-installed torch,
    # so we opt into --system-site-packages in that mode.
    if [[ ${SKIP_TORCH} -eq 1 ]]; then
        echo "==> Creating venv (with --system-site-packages for pre-installed torch)..."
        "${PYTHON_EXE}" -m venv --system-site-packages "${VENV_PATH}"
    else
        echo "==> Creating venv..."
        "${PYTHON_EXE}" -m venv "${VENV_PATH}"
    fi
fi

# shellcheck disable=SC1090,SC1091
source "${VENV_PATH}/bin/activate"

echo "==> Upgrading pip / wheel / setuptools..."
python -m pip install --upgrade pip wheel setuptools

# --- Torch install strategy ---
# If the platform image already provides a working torch (e.g., PyTorch 2.8.0
# on CUDA 12.8), --skip-torch avoids re-installing it. Otherwise pick the
# right CUDA wheel by inspecting the GPU and driver.

if [[ ${SKIP_TORCH} -eq 1 ]]; then
    echo "==> --skip-torch: using pre-installed torch from system site-packages"
    if ! python -c "import torch; print(f'  torch: {torch.__version__} (cuda: {torch.cuda.is_available()})')" 2>/dev/null; then
        echo "ERROR: torch not visible in the venv." >&2
        echo "       The venv was created without --system-site-packages OR the image lacks torch." >&2
        echo "       Fix: remove .venv and re-run with --force." >&2
        exit 1
    fi
else
    # Auto-select wheel: prefer cu128 for RTX 50-series or CUDA 12.8+ hosts.
    CUDA_REQ_FILE="${PROJECT_ROOT}/requirements/cuda121.txt"
    if command -v nvidia-smi >/dev/null 2>&1; then
        gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
        driver_cuda=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
        echo "Detected GPU:    ${gpu_name}"
        echo "Detected driver: ${driver_cuda}"

        case "${gpu_name}" in
            *5090*|*5080*|*RTX\ 50*)
                echo "==> Blackwell GPU — using CUDA 12.8 wheels (torch >= 2.8)"
                CUDA_REQ_FILE="${PROJECT_ROOT}/requirements/cuda128.txt"
                ;;
            *)
                # Ampere/Hopper — cu121 is fine but if driver is 12.4+, prefer cu124.
                # (Keep it simple: default to cu121.)
                echo "==> Non-Blackwell GPU — using CUDA 12.1 wheels (torch 2.5.1)"
                ;;
        esac
    fi

    echo "==> Installing base + $(basename "${CUDA_REQ_FILE}") + dev requirements..."
    pip install -r "${PROJECT_ROOT}/requirements/base.txt" \
                -r "${CUDA_REQ_FILE}" \
                -r "${PROJECT_ROOT}/requirements/dev.txt"
fi

# When --skip-torch: still install the non-torch deps
if [[ ${SKIP_TORCH} -eq 1 ]]; then
    pip install -r "${PROJECT_ROOT}/requirements/base.txt" \
                -r "${PROJECT_ROOT}/requirements/dev.txt"
fi

echo "==> Verifying install..."
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda device:", torch.cuda.get_device_name(0))
try:
    import triton
    print("triton:", triton.__version__)
except ImportError:
    print("triton: NOT installed (Stage 2b will fall back to PyTorch backend)")
import minigrid, gymnasium
print("minigrid + gymnasium ok")
PY

echo
echo "==> Setup complete."
echo
echo "Next steps:"
echo "  1. Activate venv: source .venv/bin/activate"
echo "  2. Run smoke:     bash scripts/cloud/run_stage.sh 0 cloud_5090 --smoke-only"
echo "                    (or cloud_24g for 24 GB cards, home_64g for 48+ GB)"
echo "  3. Run tests:     pytest -x tests/"
