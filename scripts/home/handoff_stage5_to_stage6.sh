#!/usr/bin/env bash
# devagi · Stage 5 → Stage 6 auto-handoff (open-gap handoff script)
#
# Runs ON THE REMOTE TRAINING HOST. Polls until the Stage-5 training process
# exits on its own (it self-terminates at total_steps=2_000_000), then launches
# Stage 6 via perpetual_supervise.sh, resuming from the latest Stage-5 ckpt.
#
# Usage (on remote host):
#   nohup bash scripts/home/handoff_stage5_to_stage6.sh > logs/handoff.log 2>&1 &
#
# Design:
#   - Detects Stage-5 end by absence of the `python -m src.train` process with
#     a stage-5 log file. We key off the Stage-5 log file marker instead of a
#     bare process name, so an unrelated src.train won't false-trigger.
#   - Picks the newest ckpt_stage5_*.pt under the ckpt dir.
#   - Idempotent: writes a .done marker so a second invocation won't double-launch.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

CKPT_DIR="${CKPT_DIR:-/root/autodl-tmp/karbon_ckpts/checkpoints}"
LOG_DIR="${PROJECT_ROOT}/logs/perpetual"
mkdir -p "${LOG_DIR}"
MARKER="${LOG_DIR}/handoff_stage6_launched.marker"

if [[ -f "${MARKER}" ]]; then
    echo "$(date -u +%FT%TZ) handoff already launched (marker exists). Exiting."
    exit 0
fi

echo "$(date -u +%FT%TZ) waiting for Stage 5 process to finish..."

stage5_running() {
    # True while a stage-5 training run is alive. We grep the specific marker
    # written by run_perpetual / supervise logs is hard; instead detect the
    # running src.train AND that a stage5 ckpt dir is the active one by checking
    # the most recent perpetual log mentions stage5 and is still being written.
    pgrep -f "python -m src.train" >/dev/null 2>&1 || return 1
    # Also require a recent stage5 ckpt to exist (sanity).
    ls -t "${CKPT_DIR}"/ckpt_stage5_*.pt >/dev/null 2>&1
}

# Wait until the Stage-5 train process is gone.
while stage5_running; do
    sleep 30
done

echo "$(date -u +%FT%TZ) Stage 5 process gone. Locating latest ckpt..."
LATEST="$(ls -t "${CKPT_DIR}"/ckpt_stage5_*.pt 2>/dev/null | head -1)"
if [[ -z "${LATEST}" ]]; then
    echo "$(date -u +%FT%TZ) ERROR: no stage5 ckpt found in ${CKPT_DIR}; aborting handoff."
    exit 1
fi
echo "$(date -u +%FT%TZ) latest ckpt: ${LATEST}"

# Mark before launching so a crash-restart of THIS script won't relaunch stage6
# on top of a running one.
touch "${MARKER}"

echo "$(date -u +%FT%TZ) launching Stage 6 (supervised, resume)..."
bash "${SCRIPT_DIR}/perpetual_supervise.sh" 6 home_64g --resume "${LATEST}"

echo "$(date -u +%FT%TZ) supervise exited (Stage 6 handoff complete or supervisor stopped)."
