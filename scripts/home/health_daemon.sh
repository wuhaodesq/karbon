#!/usr/bin/env bash
# devagi · Phase 3 health daemon
# 定期采样 GPU / RAM / 磁盘 / 训练进程状态，写入 CSV，超过阈值告警。
#
# Usage:
#   bash scripts/home/health_daemon.sh [output_dir]
#
# Defaults:
#   output_dir = logs/health/<UTC-timestamp>/
#   sample interval = 30s
#   VRAM slope threshold = 0.1 GB/hour (over 1h window)
#
# Design:
#   - Never terminates on its own (perpetual). Kill with Ctrl-C.
#   - Writes CSV rows to health.csv.
#   - Emits ALERT lines when slope exceeds threshold.
#
# Not a replacement for src/monitoring/memory_watcher.py — that's the
# in-process watcher inside the training loop. This daemon watches from *outside*
# to catch things like driver hangs, disk-full errors, or runaway other tenants.

set -euo pipefail

OUT_DIR="${1:-logs/health/$(date -u +%Y%m%d_%H%M%S)}"
mkdir -p "${OUT_DIR}"
CSV="${OUT_DIR}/health.csv"

SAMPLE_INTERVAL_S="${SAMPLE_INTERVAL_S:-30}"
WINDOW_HOURS="${WINDOW_HOURS:-1}"
SLOPE_THRESHOLD_GB_PER_H="${SLOPE_THRESHOLD_GB_PER_H:-0.1}"

echo "==> devagi health daemon"
echo "    output: ${CSV}"
echo "    interval: ${SAMPLE_INTERVAL_S}s"
echo "    slope threshold: ${SLOPE_THRESHOLD_GB_PER_H} GB/h over ${WINDOW_HOURS} h window"
echo "    Press Ctrl-C to stop."

# CSV header
if [[ ! -s "${CSV}" ]]; then
    echo "ts_unix,gpu_used_mib,gpu_total_mib,cpu_ram_used_mib,cpu_ram_total_mib,disk_free_gb" > "${CSV}"
fi

# Rolling-window state (recorded VRAM values with timestamps)
declare -a ts_hist=()
declare -a used_hist=()

trap 'echo; echo "== stopped =="; exit 0' INT TERM

while true; do
    now=$(date +%s)

    # GPU sample (sum across GPUs)
    if command -v nvidia-smi >/dev/null 2>&1; then
        gpu_line=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits \
            | awk '{u+=$1; gsub(/,/,""); t+=$2} END {print u","t}')
        gpu_used="${gpu_line%%,*}"
        gpu_total="${gpu_line##*,}"
    else
        gpu_used=0
        gpu_total=0
    fi

    # CPU RAM sample
    if command -v free >/dev/null 2>&1; then
        # free -m: fields "used", "total"
        ram_line=$(free -m | awk '/^Mem:/ {print $3","$2}')
        ram_used="${ram_line%%,*}"
        ram_total="${ram_line##*,}"
    else
        ram_used=0
        ram_total=0
    fi

    # Disk free (in GB) on project's filesystem
    disk_free_gb=$(df -BG . | awk 'NR==2 {gsub(/G/,"",$4); print $4}')

    echo "${now},${gpu_used},${gpu_total},${ram_used},${ram_total},${disk_free_gb}" >> "${CSV}"

    # Update rolling window
    ts_hist+=("${now}")
    used_hist+=("${gpu_used}")
    cutoff=$((now - WINDOW_HOURS * 3600))
    # Trim old entries
    while [[ ${#ts_hist[@]} -gt 0 ]] && [[ ${ts_hist[0]} -lt ${cutoff} ]]; do
        ts_hist=("${ts_hist[@]:1}")
        used_hist=("${used_hist[@]:1}")
    done

    # Compute slope in GB/h if we have >= 2 samples spanning >= 10% of window
    n=${#ts_hist[@]}
    if [[ ${n} -ge 2 ]]; then
        first_ts=${ts_hist[0]}
        last_ts=${ts_hist[$((n-1))]}
        dt=$((last_ts - first_ts))
        min_span=$((WINDOW_HOURS * 360))  # 10% of window in seconds
        if [[ ${dt} -ge ${min_span} ]]; then
            first_u=${used_hist[0]}
            last_u=${used_hist[$((n-1))]}
            # slope_gb_per_h = (last_u - first_u) MiB / (dt sec) * 3600 / 1024
            slope=$(awk -v du=$((last_u - first_u)) -v dt=${dt} 'BEGIN {printf "%.4f", (du/1024.0) * 3600.0 / dt}')
            # Compare abs(slope) > threshold
            over=$(awk -v s="${slope}" -v th="${SLOPE_THRESHOLD_GB_PER_H}" 'BEGIN {s=(s<0?-s:s); print (s>th)?1:0}')
            if [[ "${over}" == "1" ]]; then
                echo "$(date -u +%FT%TZ) ALERT: VRAM slope ${slope} GB/h exceeds ${SLOPE_THRESHOLD_GB_PER_H} GB/h"
            fi
        fi
    fi

    sleep "${SAMPLE_INTERVAL_S}"
done
