#!/usr/bin/env bash
# devagi · Phase 3 perpetual supervisor (open-gap A#10)
#
# Wraps training under a self-healing loop: if the training process dies
# (crash / OOM / driver hang), this supervisor restarts it automatically so
# the 30-consecutive-day-no-manual-restart bar (Stage 6 exit criterion) can be
# met without a human on call.
#
# Usage:
#   bash scripts/home/perpetual_supervise.sh <stage> [preset] [extra flags]
#
# Behavior:
#   - Launches src.train via the same command as run_perpetual.sh.
#   - Polls every HEALTH_CHECK_S seconds; if no python -m src.train process is
#     found, restarts it (up to MAX_RESTARTS, default 9999 = effectively
#     unbounded but counted).
#   - Logs every (re)start to logs/perpetual/supervise_<run_id>.log.
#   - Refuses to restart if the last run exited *cleanly* with a success marker
#     (logs/perpetual/<run_id>.done), so a deliberate stop is honored.
#
# Pair with scripts/home/health_daemon.sh for out-of-process VRAM/disk watch.

set -uo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: bash scripts/home/perpetual_supervise.sh <stage> [preset] [extra flags]" >&2
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
SUP_LOG="${LOG_DIR}/supervise_${RUN_ID}.log"
TRAIN_LOG="${LOG_DIR}/${RUN_ID}.log"

HEALTH_CHECK_S="${HEALTH_CHECK_S:-30}"
MAX_RESTARTS="${MAX_RESTARTS:-9999}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

cd "${PROJECT_ROOT}"

log() { echo "$(date -u +%FT%TZ) $*" | tee -a "${SUP_LOG}"; }

train_running() {
    # True if a python -m src.train process exists AND has not written a .done marker.
    pgrep -f "python -m src.train" >/dev/null 2>&1
}

start_train() {
    local attempt="$1"
    log "START training (attempt ${attempt}) stage=${STAGE} preset=${PRESET}"
    rm -f "${TRAIN_LOG}.done"
    "${VENV_PATH}/bin/python" -m src.train \
        --stage "${STAGE}" --preset "${PRESET}" "$@" \
        >"${TRAIN_LOG}" 2>&1 &
    TRAIN_PID=$!
    log "  pid=${TRAIN_PID}"
}

attempt=0
start_train "${attempt}" "$@"

while true; do
    sleep "${HEALTH_CHECK_S}"
    if train_running; then
        continue
    fi
    # Process gone. Check for a clean-stop marker.
    if [[ -f "${TRAIN_LOG}.done" ]]; then
        log "CLEAN STOP marker found (${TRAIN_LOG}.done). Supervisor exits."
        exit 0
    fi
    attempt=$((attempt + 1))
    if [[ ${attempt} -gt ${MAX_RESTARTS} ]]; then
        log "RESTART LIMIT (${MAX_RESTARTS}) reached. Giving up."
        exit 1
    fi
    log "TRAIN PROCESS MISSING — restarting (attempt ${attempt})"
    start_train "${attempt}" "$@"
done
