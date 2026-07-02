#!/usr/bin/env bash
# devagi · Autosync daemon
# 后台定时同步守护：训练时自动把产物推到 GitHub / rsync 镜像 / TOS。
#
# What this does, every INTERVAL seconds (default 3600 = 1h):
#   1. Small artefacts → Git remote (docs/, configs/, CHANGELOG.md)
#   2. Checkpoints → rsync mirror (if DEVAGI_REMOTE_TARGET set)
#   3. Latest ckpt → HF-format export (auto-refreshed for TOS upload)
#
# What is NOT touched:
#   - logs/**/*.csv (may be large; kept local until end of Stage)
#   - data/replay/* (huge; local only)
#   - Git tags (created manually at Stage exit via sync_to_git.sh)
#
# Usage:
#   bash scripts/cloud/autosync_daemon.sh [--interval 3600] [--stage 0]
#
# Recommended: launch inside its own tmux session parallel to the trainer.
#   tmux new -d -s devagi_autosync "bash scripts/cloud/autosync_daemon.sh --stage 0"
#
# Stop:
#   tmux kill-session -t devagi_autosync
#
# All actions are best-effort: any failure is logged, sleep, retry next cycle.
# Never crashes; never blocks training.

set -uo pipefail

# ---------------------------------------------------------- args

INTERVAL_S=3600         # default: every hour
STAGE=""
GIT_BRANCH="main"
DO_GIT=1
DO_RSYNC=1
DO_EXPORT=0             # off by default; opt in with --export
EXPORT_ARCH="hybrid_backbone"
EXPORT_DTYPE="float16"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --interval) INTERVAL_S="$2"; shift 2 ;;
        --stage) STAGE="$2"; shift 2 ;;
        --branch) GIT_BRANCH="$2"; shift 2 ;;
        --no-git) DO_GIT=0; shift ;;
        --no-rsync) DO_RSYNC=0; shift ;;
        --export) DO_EXPORT=1; shift ;;
        --arch) EXPORT_ARCH="$2"; shift 2 ;;
        --dtype) EXPORT_DTYPE="$2"; shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done

# ---------------------------------------------------------- paths

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LOG_DIR="${PROJECT_ROOT}/logs/autosync"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/autosync_$(date -u +%Y%m%d).log"

log() {
    local msg="$1"
    local ts
    ts="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    echo "[${ts}] ${msg}" | tee -a "${LOG_FILE}"
}

log "==> devagi autosync daemon starting"
log "    project_root:  ${PROJECT_ROOT}"
log "    interval:      ${INTERVAL_S}s"
log "    stage:         ${STAGE:-<any>}"
log "    git branch:    ${GIT_BRANCH}"
log "    do_git:        ${DO_GIT}"
log "    do_rsync:      ${DO_RSYNC}"
log "    do_export:     ${DO_EXPORT}"
log "    log_file:      ${LOG_FILE}"

cd "${PROJECT_ROOT}"

# ---------------------------------------------------------- termination

trap 'log "==> received SIGTERM/SIGINT, exiting"; exit 0' INT TERM

# ---------------------------------------------------------- helpers

sync_git() {
    if [[ ${DO_GIT} -eq 0 ]]; then
        return 0
    fi
    if [[ ! -d .git ]]; then
        log "  git: skip (not a git repo)"
        return 0
    fi
    if ! git remote get-url origin >/dev/null 2>&1; then
        log "  git: skip (no origin remote)"
        return 0
    fi

    # Only stage small text artefacts. Never adds ckpt / logs / data.
    git add -A docs/ configs/ CHANGELOG.md 2>/dev/null || true

    if git diff --cached --quiet; then
        log "  git: no new artefacts to commit"
        return 0
    fi

    local subject="autosync: stage ${STAGE:-N/A} @ $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    if ! git commit -m "${subject}" >/dev/null 2>&1; then
        log "  git: commit failed"
        return 1
    fi

    if git push origin "${GIT_BRANCH}" >/dev/null 2>&1; then
        log "  git: pushed to origin/${GIT_BRANCH}"
    else
        log "  git: push failed (will retry next cycle)"
        return 1
    fi
}

sync_rsync() {
    if [[ ${DO_RSYNC} -eq 0 ]]; then
        return 0
    fi
    if [[ -z "${DEVAGI_REMOTE_TARGET:-}" ]]; then
        log "  rsync: skip (DEVAGI_REMOTE_TARGET not set)"
        return 0
    fi
    if ! command -v rsync >/dev/null 2>&1; then
        log "  rsync: skip (rsync not installed)"
        return 0
    fi

    for dir in checkpoints docs/figures; do
        if [[ ! -d "${dir}" ]]; then
            continue
        fi
        # Bandwidth-friendly options + resume on partial
        if rsync -avz --partial --append-verify --timeout=120 \
                "${dir}/" "${DEVAGI_REMOTE_TARGET}/${dir}/" \
                >>"${LOG_FILE}" 2>&1; then
            log "  rsync: ${dir}/ → ${DEVAGI_REMOTE_TARGET}/${dir}/"
        else
            log "  rsync: ${dir}/ failed"
        fi
    done
}

sync_export() {
    if [[ ${DO_EXPORT} -eq 0 ]]; then
        return 0
    fi

    # Pick the newest checkpoint (if any)
    local latest
    latest=$(ls -t checkpoints/ckpt_*.pt 2>/dev/null | head -1 || true)
    if [[ -z "${latest}" ]]; then
        log "  export: no checkpoints found"
        return 0
    fi

    # Only re-export if this ckpt is newer than any existing export dir
    local out_dir="exports/latest"
    local marker="${out_dir}/.exported_from"
    if [[ -f "${marker}" ]] && [[ "$(cat "${marker}")" == "${latest}" ]]; then
        log "  export: latest already up-to-date (${latest})"
        return 0
    fi

    rm -rf "${out_dir}"
    if python -m scripts.export_hf \
            --ckpt "${latest}" \
            --output-dir "${out_dir}" \
            --model-name "devagi-autosync-$(basename "${latest}" .pt)" \
            --arch "${EXPORT_ARCH}" \
            --dtype "${EXPORT_DTYPE}" \
            >>"${LOG_FILE}" 2>&1; then
        echo "${latest}" > "${marker}"
        log "  export: ${latest} → ${out_dir}"
    else
        log "  export: failed for ${latest}"
    fi
}

# ---------------------------------------------------------- main loop

cycle=0
while true; do
    cycle=$((cycle + 1))
    log "-- cycle ${cycle} --"

    sync_git    || true
    sync_rsync  || true
    sync_export || true

    log "  sleeping ${INTERVAL_S}s..."
    sleep "${INTERVAL_S}"
done
