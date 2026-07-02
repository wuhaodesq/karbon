#!/usr/bin/env bash
# devagi · 24-hour longevity test (Stage 0 exit criterion)
# 24 小时永续测试（Stage 0 exit 标准）
#
# Usage:
#   bash scripts/cloud/longevity_24h.sh [stage] [preset] [duration_seconds]
#
# Defaults: stage=0, preset=cloud_24g, duration=86400 (24 h)
#
# Design:
#   - runs `python -m src.monitoring.longevity_test` under tmux for
#     survivability against SSH disconnects.
#   - if tmux is unavailable, falls back to nohup.

set -euo pipefail

STAGE="${1:-0}"
PRESET="${2:-cloud_24g}"
DURATION="${3:-86400}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VENV_PATH="${PROJECT_ROOT}/.venv"

if [[ ! -d "${VENV_PATH}" ]]; then
    echo "ERROR: venv missing. Run scripts/cloud/setup_env.sh first." >&2
    exit 1
fi

# Slope threshold: 0.2 GB/h (Axiom 5)
SLOPE_THRESHOLD="0.2"
RUN_ID="$(date -u +%Y%m%d_%H%M%S)_longevity"

CMD=(
    "${VENV_PATH}/bin/python" -m src.monitoring.longevity_test
    --stage "${STAGE}"
    --duration "${DURATION}"
    --run-id "${RUN_ID}"
    --slope-threshold "${SLOPE_THRESHOLD}"
)

echo "==> Launching longevity test"
echo "    stage=${STAGE} preset=${PRESET} duration=${DURATION}s (~$((DURATION / 3600))h)"
echo "    slope threshold: ${SLOPE_THRESHOLD} GB/h"
echo "    run_id: ${RUN_ID}"

cd "${PROJECT_ROOT}"

if command -v tmux >/dev/null 2>&1; then
    SESSION="devagi_longevity_${RUN_ID}"
    tmux new-session -d -s "${SESSION}" "${CMD[*]}"
    echo "==> Started tmux session: ${SESSION}"
    echo "    attach with: tmux attach -t ${SESSION}"
    echo "    detach:      Ctrl-B then D"
elif command -v nohup >/dev/null 2>&1; then
    LOG_FILE="${PROJECT_ROOT}/logs/longevity_${RUN_ID}.log"
    mkdir -p "$(dirname "${LOG_FILE}")"
    nohup "${CMD[@]}" >"${LOG_FILE}" 2>&1 &
    echo "==> Started nohup pid=$!  log=${LOG_FILE}"
else
    echo "==> Running in foreground (no tmux, no nohup)"
    "${CMD[@]}"
fi
