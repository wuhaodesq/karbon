#!/usr/bin/env bash
# devagi · nightly training runner
# 夜间训练启动器：自动从最新 ckpt 续训 + 开启 autosync 守护 + 定时结束
#
# What this does:
#   1. Finds the newest checkpoint for the given stage.
#   2. Launches training in a tmux session with --resume.
#   3. Launches the autosync daemon in another tmux session.
#   4. Optional: --duration N sleeps N seconds then gracefully kills the
#      training session, leaving one final ckpt + autosync push before exit.
#
# Usage:
#   bash scripts/cloud/nightly_run.sh <stage> [preset] [--duration <seconds>]
#
# Examples:
#   # Run indefinitely (Ctrl-C to stop)
#   bash scripts/cloud/nightly_run.sh 1 cloud_5090
#
#   # Run for 6 hours then auto-stop (perfect for nightly rentals)
#   bash scripts/cloud/nightly_run.sh 1 cloud_5090 --duration 21600
#
#   # Run until 08:00 UTC tomorrow (use `date` to compute duration)
#   bash scripts/cloud/nightly_run.sh 1 cloud_5090 --until "tomorrow 08:00"

set -uo pipefail

if [[ $# -lt 1 ]]; then
    cat <<EOF >&2
usage: bash scripts/cloud/nightly_run.sh <stage> [preset] [--duration <sec>] [--until <time>]

  stage:    0-6
  preset:   cloud_24g | cloud_5090 (default) | cloud_12g | home_64g
  --duration <sec>:   auto-stop after N seconds (e.g. 21600 = 6h)
  --until <time>:     auto-stop at this time (accepts `date` syntax)

Examples:
  bash scripts/cloud/nightly_run.sh 1 cloud_5090 --duration 21600
  bash scripts/cloud/nightly_run.sh 2 cloud_5090 --until "tomorrow 08:00"
EOF
    exit 2
fi

STAGE="$1"; shift
PRESET="cloud_5090"
DURATION=""
UNTIL=""

if [[ $# -gt 0 && "$1" != --* ]]; then
    PRESET="$1"; shift
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --duration) DURATION="$2"; shift 2 ;;
        --until)    UNTIL="$2";    shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done

# Resolve --until to a duration if provided
if [[ -n "${UNTIL}" ]]; then
    target_ts=$(date -d "${UNTIL}" +%s 2>/dev/null || true)
    if [[ -z "${target_ts}" ]]; then
        echo "ERROR: cannot parse --until '${UNTIL}'" >&2
        exit 2
    fi
    now_ts=$(date +%s)
    DURATION=$((target_ts - now_ts))
    if [[ ${DURATION} -le 0 ]]; then
        echo "ERROR: --until is in the past (${UNTIL})" >&2
        exit 2
    fi
    echo "==> Auto-stop scheduled: ${UNTIL} (in ${DURATION}s)"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

if [[ ! -d .venv ]]; then
    echo "ERROR: .venv missing. Run scripts/cloud/setup_env.sh first." >&2
    exit 1
fi

# --- Detect latest checkpoint for this stage ---
LATEST_CKPT=""
if compgen -G "checkpoints/ckpt_stage${STAGE}_*.pt" > /dev/null 2>&1; then
    LATEST_CKPT=$(ls -t "checkpoints/ckpt_stage${STAGE}"_*.pt 2>/dev/null | head -1 || true)
fi

echo "==> devagi nightly runner"
echo "    stage:      ${STAGE}"
echo "    preset:     ${PRESET}"
echo "    latest ckpt:${LATEST_CKPT:-<none, cold-start>}"
echo "    duration:   ${DURATION:-<indefinite>}"

# --- Fragmentation mitigation ---
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# --- Build train command ---
TRAIN_CMD="${PROJECT_ROOT}/.venv/bin/python -m src.train --stage ${STAGE} --preset ${PRESET}"
if [[ -n "${LATEST_CKPT}" ]]; then
    TRAIN_CMD="${TRAIN_CMD} --resume ${LATEST_CKPT}"
fi

RUN_ID="$(date -u +%Y%m%d_%H%M%S)_stage${STAGE}"
LOG_DIR="${PROJECT_ROOT}/logs/nightly"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/${RUN_ID}.log"

TRAIN_SESSION="devagi_train_${RUN_ID}"
AUTOSYNC_SESSION="devagi_autosync_${RUN_ID}"

# --- Launch training in tmux ---
if ! command -v tmux >/dev/null 2>&1; then
    echo "ERROR: tmux not installed. Install with: sudo apt install -y tmux" >&2
    exit 1
fi

echo "==> Launching training in tmux: ${TRAIN_SESSION}"
tmux new-session -d -s "${TRAIN_SESSION}" \
    "cd ${PROJECT_ROOT} && ${TRAIN_CMD} 2>&1 | tee ${LOG_FILE}"

# --- Launch autosync in parallel ---
echo "==> Launching autosync daemon in tmux: ${AUTOSYNC_SESSION}"
tmux new-session -d -s "${AUTOSYNC_SESSION}" \
    "cd ${PROJECT_ROOT} && bash scripts/cloud/autosync_daemon.sh --stage ${STAGE} --interval 1800"

echo
echo "==> Both sessions launched:"
echo "    tmux attach -t ${TRAIN_SESSION}      # watch training"
echo "    tmux attach -t ${AUTOSYNC_SESSION}   # watch sync"
echo "    log file: ${LOG_FILE}"

# --- Optional auto-stop after duration ---
if [[ -n "${DURATION}" ]]; then
    echo
    echo "==> Will auto-stop training in ${DURATION}s"
    echo "    (You can safely close this SSH session; the tmux sessions"
    echo "    continue in the background and the auto-stop still fires)"
    (
        sleep "${DURATION}"
        echo "==> Duration elapsed, stopping training gracefully"
        # SIGINT gives src.train a chance to flush a final ckpt
        tmux send-keys -t "${TRAIN_SESSION}" C-c 2>/dev/null || true
        sleep 30
        # Now run one last autosync cycle
        cd "${PROJECT_ROOT}"
        bash scripts/cloud/sync_to_git.sh "${STAGE}" 2>&1 | tee -a "${LOG_FILE}" || true
        # Then stop autosync
        tmux kill-session -t "${AUTOSYNC_SESSION}" 2>/dev/null || true
        tmux kill-session -t "${TRAIN_SESSION}" 2>/dev/null || true
        echo "==> Nightly run complete. Safe to shut down instance."
    ) &
    DISOWN_PID=$!
    echo "==> auto-stop watcher pid: ${DISOWN_PID}"
    echo "    to cancel: kill ${DISOWN_PID}"
fi
