#!/usr/bin/env bash
# devagi · cloud Linux environment setup (Phase 2)
# 云端 Linux 环境安装脚本
#
# Usage:
#   bash scripts/cloud/setup_env.sh [--force] [--python python3.10]
#
# Assumes:
#   - Linux (Ubuntu 22.04 recommended)
#   - CUDA 12.1 driver installed (verify with nvidia-smi)
#   - Python 3.10 available
#   - git already clones this repo

set -euo pipefail

FORCE=0
PYTHON_EXE="python3.10"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --force) FORCE=1; shift ;;
        --python) PYTHON_EXE="$2"; shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done

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
if [[ "${pyver}" != Python\ 3.10.* ]]; then
    echo "WARN: expected Python 3.10.x, got: ${pyver}"
    if [[ ${FORCE} -eq 0 ]]; then
        echo "Aborting. Re-run with --force to override." >&2
        exit 1
    fi
fi

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
    echo "==> Creating venv..."
    "${PYTHON_EXE}" -m venv "${VENV_PATH}"
fi

# shellcheck disable=SC1090,SC1091
source "${VENV_PATH}/bin/activate"

echo "==> Upgrading pip / wheel / setuptools..."
python -m pip install --upgrade pip wheel setuptools

echo "==> Installing base + cuda121 + dev requirements..."
pip install -r "${PROJECT_ROOT}/requirements/base.txt" \
            -r "${PROJECT_ROOT}/requirements/cuda121.txt" \
            -r "${PROJECT_ROOT}/requirements/dev.txt"

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
echo "  2. Run smoke:     bash scripts/cloud/run_stage.sh 0 cloud_24g --smoke-only"
echo "  3. Run tests:     pytest -x tests/"
