#!/usr/bin/env bash
# devagi · Phase 3 perpetual training launcher
# 家用 64GB 永续训练启动脚本
#
# Usage:
#   bash scripts/home/run_perpetual.sh <stage> [preset] [extra flags to src.train]
#
# Example:
#   bash scripts/home/run_perpetual.sh 5 home_64g
#   bash scripts/home/run_perpetual.sh 6 home_64g --resume checkpoints/ckpt_stage5_XXX.pt
#
# Design:
#   - Wraps training under tmux for SSH-disconnect survivability.
#   - Sets PYTORCH_CUDA_ALLOC_CONF for fragmentation mitigation (Axiom 5).
#   - Writes stdout/stderr to logs/perpetual/<run_id>.log.
#   - Emits a health-daemon reminder.

set -euo pipefail

if [[ $# -lt 1 ]]; then
    cat <<EOF >&2
usage: bash scripts/home/run_perpetual.sh <stage> [preset] [extra flags]

  stage:   0 | 1 | 2 | 3 | 4 | 5 | 6
  preset:  home_64g (default) | cloud_24g | local_smoke

Example:
  bash scripts/home/run_perpetual.sh 6 home_64g
EOF
    exit 2
fi

STAGE="$1"; shift
PRESET="${1:-home_64g}"
if [[ $# -gt 0 && "$1" == "$PRESET" ]]; then shift; fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VENV_PATH="${PROJECT_ROOT}/.venv"
LOG_DIR="${PROJECT_ROOT}/logs/perpetual"
mkdir -p "${LOG_DIR}"

RUN_ID="$(date -u +%Y%m%d_%H%M%S)_stage${STAGE}"
LOG_FILE="${LOG_DIR}/${RUN_ID}.log"

if [[ ! -d "${VENV_PATH}" ]]; then
    echo "ERROR: venv missing. Run scripts/home/setup_env.sh first." >&2
    exit 1
fi

# Fragmentation mitigation (Axiom 5)
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

cd "${PROJECT_ROOT}"

CMD=(
    "${VENV_PATH}/bin/python" -m src.train
    --stage "${STAGE}"
    --preset "${PRESET}"
    "$@"
)

echo "==> Phase 3 perpetual training"
echo "    stage=${STAGE} preset=${PRESET}"
echo "    run_id=${RUN_ID}"
echo "    log=${LOG_FILE}"
echo "    cmd: ${CMD[*]}"

if command -v tmux >/dev/null 2>&1; then
    SESSION="devagi_perpetual_${RUN_ID}"
    tmux new-session -d -s "${SESSION}" \
        "${CMD[*]} 2>&1 | tee ${LOG_FILE}"
    echo "==> tmux session started: ${SESSION}"
    echo "    attach:  tmux attach -t ${SESSION}"
    echo "    detach:  Ctrl-B D"
elif command -v nohup >/dev/null 2>&1; then
    nohup "${CMD[@]}" >"${LOG_FILE}" 2>&1 &
    echo "==> nohup pid=$!"
else
    "${CMD[@]}" 2>&1 | tee "${LOG_FILE}"
fi

echo
echo "==> Reminder: launch the health daemon in another shell:"
echo "    bash scripts/home/health_daemon.sh ${LOG_DIR}/${RUN_ID}"
