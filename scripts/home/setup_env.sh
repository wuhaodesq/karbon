#!/usr/bin/env bash
# devagi · home 64G rig environment setup (Phase 3)
# 家用 64GB 显存 Linux 训练机环境安装
#
# Usage:
#   bash scripts/home/setup_env.sh [--force] [--python <python-exec>] [--skip-torch]
#
# Assumes:
#   - Linux (Ubuntu 22.04 or 24.04 LTS recommended)
#   - NVIDIA driver + CUDA installed (verify with nvidia-smi)
#   - Total VRAM ≥ 48 GB (target 64 GB); single or dual-GPU
#   - Python 3.10 / 3.11 / 3.12 available
#
# --skip-torch: reuse pre-installed torch (e.g., platform images that ship
# PyTorch 2.8.0 / CUDA 12.8) instead of pinning our own wheel.

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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VENV_PATH="${PROJECT_ROOT}/.venv"

echo "==> devagi · Phase 3 home rig environment setup"
echo "Project root: ${PROJECT_ROOT}"
echo "Venv path:    ${VENV_PATH}"

# --- Python ---
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

# --- NVIDIA / CUDA + VRAM sanity ---
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi not found — this is the Phase-3 home-rig script and needs NVIDIA." >&2
    exit 1
fi
nvidia-smi | head -20

# Report total VRAM available across all GPUs
total_mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | awk '{s+=$1} END {print s}')
echo "Total VRAM detected: ${total_mib} MiB"
if [[ "${total_mib}" -lt 40000 && ${FORCE} -eq 0 ]]; then
    echo "WARN: Phase-3 home rig expects ≥ 40 GB total VRAM (target 64 GB)."
    echo "You appear to have < 40 GB. Prefer scripts/cloud/setup_env.sh for smaller GPUs."
    echo "Re-run with --force to install anyway."
    exit 1
fi

# --- venv ---
if [[ -d "${VENV_PATH}" ]]; then
    if [[ ${FORCE} -eq 1 ]]; then
        echo "Removing existing venv (--force)"
        rm -rf "${VENV_PATH}"
    else
        echo "Venv already exists. Use --force to recreate. Skipping."
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

if [[ ${SKIP_TORCH} -eq 1 ]]; then
    echo "==> --skip-torch: using pre-installed torch"
    python -c "import torch; print(f'  torch: {torch.__version__}')"
    pip install -r "${PROJECT_ROOT}/requirements/base.txt" \
                -r "${PROJECT_ROOT}/requirements/dev.txt"
else
    # Choose CUDA wheel variant based on detected GPU
    CUDA_REQ_FILE="${PROJECT_ROOT}/requirements/cuda121.txt"
    gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    echo "Detected GPU (first): ${gpu_name}"
    case "${gpu_name}" in
        *5090*|*5080*|*RTX\ 50*)
            echo "==> Blackwell GPU — using CUDA 12.8 wheels (torch >= 2.8)"
            CUDA_REQ_FILE="${PROJECT_ROOT}/requirements/cuda128.txt"
            ;;
        *)
            echo "==> Non-Blackwell GPU — using CUDA 12.1 wheels (torch 2.5.1)"
            ;;
    esac

    echo "==> Installing base + $(basename "${CUDA_REQ_FILE}") + dev requirements..."
    pip install -r "${PROJECT_ROOT}/requirements/base.txt" \
                -r "${CUDA_REQ_FILE}" \
                -r "${PROJECT_ROOT}/requirements/dev.txt"
fi

echo "==> Verifying install..."
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    total = 0
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        gb = props.total_memory / 1024**3
        total += gb
        print(f"  cuda:{i}: {props.name}  ({gb:.1f} GB)")
    print(f"Total VRAM: {total:.1f} GB")
try:
    import triton
    print("triton:", triton.__version__)
except ImportError:
    print("triton: NOT installed")
import minigrid, gymnasium
print("minigrid + gymnasium ok")
PY

# tmux is the recommended long-run supervisor for Phase 3
if command -v tmux >/dev/null 2>&1; then
    echo "tmux: $(tmux -V)"
else
    echo "WARN: tmux not installed. Install with: sudo apt-get install -y tmux"
fi

echo
echo "==> Phase 3 setup complete."
echo
echo "Next steps for perpetual training:"
echo "  1. source .venv/bin/activate"
echo "  2. pytest -x tests/                                # sanity check"
echo "  3. bash scripts/home/run_perpetual.sh <stage> home_64g [--resume <ckpt>]"
echo "  4. In a separate shell: bash scripts/home/health_daemon.sh"
