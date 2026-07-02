#!/usr/bin/env bash
# devagi · rsync training artifacts back to local mirror or object storage
# 训练完把 ckpt/logs/docs 拉回本地或对象存储
#
# Configure the destination via env vars:
#   DEVAGI_REMOTE_TARGET   e.g., user@laptop:/mnt/karbon-mirror
# or via first CLI arg:
#   bash scripts/cloud/pull_logs.sh user@laptop:/mnt/karbon-mirror
#
# Uses `rsync -avz --partial --append-verify --progress` for robustness.

set -euo pipefail

TARGET="${1:-${DEVAGI_REMOTE_TARGET:-}}"

if [[ -z "${TARGET}" ]]; then
    cat <<EOF >&2
ERROR: No target specified.

Set DEVAGI_REMOTE_TARGET env var, or pass as arg:
  bash scripts/cloud/pull_logs.sh user@host:/path/to/karbon-mirror
EOF
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

echo "==> Sync target: ${TARGET}"
cd "${PROJECT_ROOT}"

for dir in checkpoints logs docs/figures; do
    if [[ -d "${dir}" ]]; then
        echo "--> Syncing ${dir}/ ..."
        rsync -avz --partial --append-verify --progress \
            "${dir}/" "${TARGET}/${dir}/"
    else
        echo "--> Skipping ${dir}/ (does not exist)"
    fi
done

echo "==> Done."
