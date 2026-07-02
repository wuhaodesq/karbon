#!/usr/bin/env bash
# devagi · sync training artefacts to the Git remote
# 将训练产物（小文件）同步到 Git 远程
#
# What syncs to Git (small text/config artefacts):
#   - docs/stage*_report.md               (bilingual reports)
#   - docs/figures/*.png                  (loss/memory curves)
#   - configs/stage*.yaml                 (frozen config snapshots)
#   - CHANGELOG.md                        (updated per stage)
#   - Git tags (v0.X.0-stageN)
#
# What does NOT go to Git (large binaries):
#   - checkpoints/*.pt                    (upload to TOS / rsync back to mirror)
#   - logs/**/*.csv                       (may be large; kept local)
#   - data/replay/*                       (very large; local only)
#   - exports/**                          (already gitignored; upload to TOS)
#
# Usage:
#   bash scripts/cloud/sync_to_git.sh <stage> [git-tag] [remote-branch]
#
# Example:
#   bash scripts/cloud/sync_to_git.sh 0 v0.0.0-stage0-cloud
#   bash scripts/cloud/sync_to_git.sh 1 v0.1.0-stage1 main
#
# Prerequisite:
#   Git credentials must already work on this machine (SSH key or PAT).
#   Set them once with:
#     git config --global user.name "your-name"
#     git config --global user.email "you@example.com"
#     # then use `gh auth login` or write a PAT to ~/.git-credentials

set -euo pipefail

if [[ $# -lt 1 ]]; then
    cat <<EOF >&2
usage: bash scripts/cloud/sync_to_git.sh <stage> [git-tag] [remote-branch]

  stage:         0 | 1 | 2 | 3 | 4 | 5 | 6
  git-tag:       optional tag to push (e.g., v0.1.0-stage1)
  remote-branch: default 'main'

Example:
  bash scripts/cloud/sync_to_git.sh 0 v0.0.0-stage0-cloud
EOF
    exit 2
fi

STAGE="$1"
TAG="${2:-}"
BRANCH="${3:-main}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${PROJECT_ROOT}"

echo "==> devagi · Sync Stage ${STAGE} results to Git"
echo "    project root: ${PROJECT_ROOT}"
echo "    branch:       ${BRANCH}"
echo "    tag:          ${TAG:-<none>}"

# --- Confirm we're inside a Git repo ---
if [[ ! -d .git ]]; then
    echo "ERROR: not inside a Git repo. Clone with git first." >&2
    exit 1
fi

# --- Verify remote is set ---
if ! git remote get-url origin >/dev/null 2>&1; then
    echo "ERROR: no 'origin' remote. Set it first:" >&2
    echo "       git remote add origin https://github.com/<user>/<repo>.git" >&2
    exit 1
fi
echo "    remote:       $(git remote get-url origin)"

# --- Pull latest ---
echo "==> Pulling latest ${BRANCH}..."
git pull --rebase origin "${BRANCH}"

# --- Verify small artefacts exist ---
STAGE_REPORT="docs/stage${STAGE}_report.md"
if [[ ! -f "${STAGE_REPORT}" ]]; then
    echo "WARN: ${STAGE_REPORT} not found — nothing to sync from docs/"
fi

# --- Stage the intended small artefacts ---
git add -A docs/ configs/ CHANGELOG.md 2>/dev/null || true

# Confirm staged changes exist
if git diff --cached --quiet; then
    echo "==> No stage-${STAGE} artefacts to commit."
else
    echo "==> Committing changes:"
    git diff --cached --stat
    git commit -m "Stage ${STAGE}: sync training results (reports, figures, config snapshot)"
fi

# --- Push commit ---
echo "==> Pushing to origin/${BRANCH}..."
git push origin "${BRANCH}"

# --- Optional: push tag ---
if [[ -n "${TAG}" ]]; then
    # Create tag if not present
    if git rev-parse "${TAG}" >/dev/null 2>&1; then
        echo "==> Tag ${TAG} already exists locally, pushing..."
    else
        echo "==> Creating tag ${TAG}"
        git tag -a "${TAG}" -m "Stage ${STAGE} exit"
    fi
    git push origin "${TAG}"
fi

echo
echo "==> Sync complete."
echo
echo "Reminder: large artefacts (*.pt, replay data) go to TOS/mirror, not Git."
echo "         Run: bash scripts/cloud/pull_logs.sh   (rsync back to local mirror)"
echo "         Or:  python -m scripts.export_hf ...   (HF-format upload to TOS)"
